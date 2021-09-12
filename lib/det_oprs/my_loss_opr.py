import torch
import torch.nn.functional as F
from det_oprs.bbox_opr import bbox_transform_inv_opr, bbox_transform_opr
from det_oprs.bbox_opr import box_overlap_opr, align_box_giou_opr
from config import config

EPS = 1e-12

def freeanchor_loss(anchors, cls_prob, bbox_preds, gt_boxes, im_info):
    gt_labels, gt_bboxes = [], []
    cls_prob = cls_prob.reshape(config.train_batch_per_gpu, -1, config.num_classes-1)
    bbox_preds = bbox_preds.reshape(config.train_batch_per_gpu, -1, 4)
    gt_boxes = [gt_boxes[bid, :int(im_info[bid, 5]), :] for bid in range(config.train_batch_per_gpu)]
    for gt_box in gt_boxes:
        obj_mask = torch.where(gt_box[:, -1] == 1)[0] 
        gt_labels.append(torch.zeros_like(gt_box[obj_mask, -1]).long())
        gt_bboxes.append(gt_box[obj_mask, :4])
    box_prob = []
    num_pos = 0
    positive_losses = []
    for _, (gt_labels_, gt_bboxes_, cls_prob_,bbox_preds_) in \
                        enumerate(zip(gt_labels, gt_bboxes, cls_prob, bbox_preds)):
        with torch.no_grad():
            if len(gt_bboxes_) == 0:
                image_box_prob = torch.zeros(anchors.size(0), config.num_classes-1).type_as(bbox_preds_)
            else:
                # box_localization: a_{j}^{loc}, shape: [j, 4]
                pred_boxes = bbox_transform_inv_opr(anchors, bbox_preds_)

                # object_box_iou: IoU_{ij}^{loc}, shape: [i, j]
                object_box_iou = box_overlap_opr(gt_bboxes_, pred_boxes)

                # object_box_prob: P{a_{j} -> b_{i}}, shape: [i, j]
                t1 = config.bbox_thr
                t2 = object_box_iou.max(dim=1, keepdim=True).values.clamp(min=t1 + 1e-12)
                object_box_prob = ((object_box_iou - t1) / (t2 - t1)).clamp(min=0, max=1)

                # object_cls_box_prob: P{a_{j} -> b_{i}}, shape: [i, c, j]
                num_obj = gt_labels_.size(0)
                indices = torch.stack([torch.arange(num_obj).type_as(gt_labels_), gt_labels_],dim=0)
                object_cls_box_prob = torch.sparse.FloatTensor(indices, object_box_prob)
                                        
                # image_box_iou: P{a_{j} \in A_{+}}, shape: [c, j]
                box_cls_prob = torch.sparse.sum(object_cls_box_prob, dim=0).to_dense()
                indices = torch.nonzero(box_cls_prob, as_tuple=False).t_()
                if indices.numel() == 0:
                    image_box_prob = torch.zeros(anchors.size(0), config.num_classes-1).type_as(object_box_prob)
                else:
                    nonzero_box_prob = torch.where((gt_labels_.unsqueeze(dim=-1) == indices[0]),
                                        object_box_prob[:, indices[1]],
                                        torch.tensor([0]).type_as(object_box_prob)).max(dim=0).values
                    # upmap to shape [j, c]
                    image_box_prob = torch.sparse.FloatTensor(
                                        indices.flip([0]),
                                        nonzero_box_prob,
                                        size=(anchors.size(0), config.num_classes-1)).to_dense()
            box_prob.append(image_box_prob)
        # construct bags for objects
        match_quality_matrix = box_overlap_opr(gt_bboxes_, anchors)
        _, matched = torch.topk(match_quality_matrix, config.pre_anchor_topk, dim=1, sorted=False)
        del match_quality_matrix

        # matched_cls_prob: P_{ij}^{cls}
        matched_cls_prob = torch.gather(
                            cls_prob_[matched], 2,
                            gt_labels_.view(-1, 1, 1).repeat(1, config.pre_anchor_topk, 1)).squeeze(2)

        # matched_box_prob: P_{ij}^{loc}
        pred_boxes = bbox_transform_inv_opr(anchors, bbox_preds_)
        object_box_iou = box_overlap_opr(gt_bboxes_, pred_boxes)
        matched_box_prob = torch.gather(object_box_iou, 1, matched).clamp(min=1e-6)
        num_pos += len(gt_bboxes_)
        positive_losses.append(positive_bag_loss(matched_cls_prob, matched_box_prob))
    positive_loss = torch.cat(positive_losses).sum() / max(1, num_pos)

    # box_prob: P{a_{j} \in A_{+}}
    box_prob = torch.stack(box_prob, dim=0)

    # negative_loss:
    # \sum_{j}{ FL((1 - P{a_{j} \in A_{+}}) * (1 - P_{j}^{bg})) } / n||B||
    negative_loss = negative_bag_loss(cls_prob, box_prob).sum() / max(1, num_pos * config.pre_anchor_topk)

    # avoid the absence of gradients in regression subnet
    # when no ground-truth in a batch
    if num_pos == 0:
        positive_loss = bbox_preds.sum() * 0

    losses = {
        'positive_bag_loss': positive_loss,
        'negative_bag_loss': negative_loss
    }
    return losses

def positive_bag_loss(matched_cls_prob, matched_box_prob):
    # bag_prob = Mean-max(matched_prob)
    matched_prob = matched_cls_prob * matched_box_prob
    weight = 1 / torch.clamp(1 - matched_prob, 1e-12, None)
    weight /= weight.sum(dim=1).unsqueeze(dim=-1)
    bag_prob = (weight * matched_prob).sum(dim=1)
    # positive_bag_loss = -self.alpha * log(bag_prob)
    return config.loss_box_alpha * F.binary_cross_entropy(bag_prob, torch.ones_like(bag_prob), reduction='none')

def negative_bag_loss(cls_prob, box_prob):
    prob = cls_prob * (1 - box_prob)
    # There are some cases when neg_prob = 0.
    # This will cause the neg_prob.log() to be inf without clamp.
    prob = prob.clamp(min=EPS, max=1 - EPS)
    negative_bag_loss = prob**config.loss_box_gamma * F.binary_cross_entropy(
        prob, torch.zeros_like(prob), reduction='none')
    return (1 - config.loss_box_alpha) * negative_bag_loss