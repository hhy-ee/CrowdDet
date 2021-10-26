import torch
import torch.nn.functional as F
from det_oprs.bbox_opr import bbox_transform_inv_opr
from det_oprs.bbox_opr import box_overlap_opr, align_box_giou_opr
from config import config

def softmax_loss(score, label, ignore_label=-1):
    with torch.no_grad():
        max_score = score.max(axis=1, keepdims=True)[0]
    score = score - max_score
    log_prob = score - torch.log(torch.exp(score).sum(axis=1, keepdims=True))
    mask = label != ignore_label
    vlabel = label * mask
    onehot = torch.zeros(vlabel.shape[0], config.num_classes, device=score.device)
    onehot.scatter_(1, vlabel.reshape(-1, 1), 1)
    loss = -(log_prob * onehot).sum(axis=1)
    loss = loss * mask
    return loss

def refined_softmax_loss(score, var_score, weight, label, ignore_label=-1):
    with torch.no_grad():
        max_score = score.max(axis=1, keepdims=True)[0]
        max_var_score = var_score.max(axis=1, keepdims=True)[0]
    score -= max_score
    var_score -= max_var_score

    log_prob = score - torch.log(torch.exp(score).sum(axis=1, keepdims=True))
    var_log_prob = var_score - torch.log(torch.exp(var_score).sum(axis=1, keepdims=True))
    weight = torch.sigmoid(weight)[:,1:]

    refined_prob = weight * log_prob.exp()[:,1:] + (1-weight) * var_log_prob.exp()[:,1:]
    refined_log_prob = torch.log(torch.cat([1-refined_prob, refined_prob], dim=1))
    # one-hot label
    mask = label != ignore_label
    vlabel = label * mask
    onehot = torch.zeros(vlabel.shape[0], config.num_classes, device=score.device)
    onehot.scatter_(1, vlabel.reshape(-1, 1), 1)
    # origin loss
    loss = -(log_prob * onehot).sum(axis=1)
    loss = loss * mask
    # refined loss
    refined_loss = -(refined_log_prob * onehot).sum(axis=1)
    refined_loss = refined_loss * mask
    return loss, refined_loss 

def refined_var_softmax_loss(score, var_score, weight, label, ignore_label=-1):
    with torch.no_grad():
        max_score = score.max(axis=1, keepdims=True)[0]
        max_var_score = var_score.max(axis=1, keepdims=True)[0]
    score -= max_score
    var_score -= max_var_score

    log_prob = score - torch.log(torch.exp(score).sum(axis=1, keepdims=True))
    var_log_prob = var_score - torch.log(torch.exp(var_score).sum(axis=1, keepdims=True))
    weight = torch.sigmoid(weight)[:,1:]

    refined_prob = weight * log_prob.exp()[:,1:] + (1-weight) * var_log_prob.exp()[:,1:]
    refined_log_prob = torch.log(torch.cat([1-refined_prob, refined_prob], dim=1))
    # one-hot label
    mask = label != ignore_label
    vlabel = label * mask
    onehot = torch.zeros(vlabel.shape[0], config.num_classes, device=score.device)
    onehot.scatter_(1, vlabel.reshape(-1, 1), 1)
    # origin loss
    loss = -(log_prob * onehot).sum(axis=1)
    loss = loss * mask
    # var loss
    var_loss = -(var_log_prob * onehot).sum(axis=1)
    var_loss = var_loss * mask
    # refined loss
    refined_loss = -(refined_log_prob * onehot).sum(axis=1)
    refined_loss = refined_loss * mask
    return loss, var_loss, refined_loss 

def smooth_l1_loss(pred, target, beta: float):
    if beta < 1e-5:
        loss = torch.abs(input - target)
    else:
        abs_x = torch.abs(pred- target)
        in_mask = abs_x < beta
        loss = torch.where(in_mask, 0.5 * abs_x ** 2 / beta, abs_x - 0.5 * beta)
    return loss.sum(axis=1)

def focal_loss(inputs, targets, alpha=-1, gamma=2, eps=1e-8):
    class_range = torch.arange(1, inputs.shape[1] + 1, device=inputs.device)
    pos_pred = (1 - inputs) ** gamma * torch.log(inputs + eps)
    neg_pred = inputs ** gamma * torch.log(1 - inputs + eps)

    pos_loss = (targets == class_range) * pos_pred * alpha
    neg_loss = (targets != class_range) * neg_pred * (1 - alpha)
    loss = -(pos_loss + neg_loss)
    return loss.sum(axis=1)

def vpd_focal_loss(inputs, targets, lstd, alpha=-1, gamma=2, eps=1e-8):
    std = lstd.exp().mean(1).unsqueeze(1)
    class_range = torch.arange(1, inputs.shape[1] + 1, device=inputs.device)
    pos_std = std[torch.where(targets==class_range)[0]]

    pos_norm_std = std/pos_std.max()
    pos_pred = pos_norm_std ** gamma * torch.log(inputs + eps)
    neg_pred = inputs ** gamma * torch.log(1 - inputs + eps)

    pos_loss = (targets == class_range) * pos_pred * alpha
    neg_loss = (targets != class_range) * neg_pred * (1 - alpha)
    loss = -(pos_loss + neg_loss)
    return loss.sum(axis=1)

def vpd1_focal_loss(inputs, targets, lstd, alpha=-1, gamma=2, eps=1e-8):
    std = lstd.exp().mean(1).unsqueeze(1)
    class_range = torch.arange(1, inputs.shape[1] + 1, device=inputs.device)
    
    pos_std = std[torch.where(targets==class_range)[0]]
    pos_norm_std = std/pos_std.max()
    neg_std = std[torch.where(targets!=class_range)[0]]
    neg_norm_std = (neg_std.min() + eps)/(std + eps)
    pos_pred = pos_norm_std ** gamma * torch.log(inputs + eps)
    neg_pred = neg_norm_std ** gamma * torch.log(1 - inputs + eps)

    pos_loss = (targets == class_range) * pos_pred * alpha
    neg_loss = (targets != class_range) * neg_pred * (1 - alpha)
    loss = -(pos_loss + neg_loss)
    return loss.sum(axis=1)

def pull_loss(labels, regs, anchors, weight):
    iou = 0.0
    pair_num = 0
    regs = regs.reshape(config.train_batch_per_gpu, -1, 4)
    for bid in range(config.train_batch_per_gpu):
        pull_label = labels[bid]
        for pull_boxes_idx in pull_label:
            pred_boxes = bbox_transform_inv_opr(anchors[pull_boxes_idx], regs[bid][pull_boxes_idx])
            pull_iou = box_overlap_opr(pred_boxes, pred_boxes)
            pull_iou_mask = torch.tril(torch.ones_like(pull_iou)).eq(0)
            iou += pull_iou.mul(pull_iou_mask).sum() / pull_iou_mask.sum()
        pair_num +=  len(pull_label)
    if  pair_num != 0:
        pull_loss = -(iou / pair_num).log()
    else:
        pull_loss = torch.tensor(0).type_as(regs)
    return weight * pull_loss

def push_loss(labels, regs, anchors, weight):
    iou = 0.0
    pair_num = 0
    regs = regs.reshape(config.train_batch_per_gpu, -1, 4)
    for bid in range(config.train_batch_per_gpu):
        push_label = labels[bid]
        for push_boxes_idx in push_label:
            pos_boxes = bbox_transform_inv_opr(anchors[push_boxes_idx[0]], regs[bid][push_boxes_idx[0]])
            neg_boxes = bbox_transform_inv_opr(anchors[push_boxes_idx[1]], regs[bid][push_boxes_idx[1]])
            iou += box_overlap_opr(pos_boxes, neg_boxes).mean()
        pair_num +=  len(push_label)
    if pair_num != 0:
        push_loss = torch.relu(-torch.log((1 - iou / pair_num)/(1 - config.test_nms)))
    else:
        push_loss = torch.tensor(0).type_as(regs)
    return weight * push_loss

def kldiv_loss(pred_mean, pred_lstd, kl_weight):
    loss = (1 + pred_lstd.mul(2) - pred_mean.pow(2) - pred_lstd.mul(2).exp()).mul(-0.5)
    return kl_weight * loss.mean()

def kldiv_nvpd_loss(pred_mean, pred_lstd, kl_weight):
    loss = (1 + pred_lstd.mul(2) - pred_lstd.mul(2).exp()).mul(-0.5)
    return kl_weight * loss.mean()
    
def rcnn_kldiv_loss(pred_mean, pred_lstd, kl_weight):
    loss = (1 + pred_lstd.mul(2) - pred_mean.pow(2) - pred_lstd.mul(2).exp()).mul(-0.5)
    return kl_weight * loss.mean(axis=1)

def rcnn_mvpd_kldiv_loss(mean, lstd, mean_targets, std_targets, kl_weight):
    loss = (1 + 2 * std_targets.log() - 2 * lstd - (std_targets.pow(2) + 
                    (mean - mean_targets).pow(2)).div(lstd.mul(2).exp())).mul(-0.5)
    return kl_weight * loss.sum(axis=1)

def iouvar_loss(anchors, bbox_target, reg_samples, iouvar_weight):
    target_bbox = bbox_transform_inv_opr(anchors, bbox_target)
    samples_iou = align_box_giou_opr(reg_samples.reshape(-1, 4), target_bbox.repeat(config.sample_num,1))
    var_loss = torch.var(samples_iou.reshape(config.sample_num , -1), dim=0)
    return iouvar_weight * var_loss 

def gmkl_loss(pred_mean, pred_lstd, kl_weight):
    loss = (1 + pred_lstd.mul(2) - pred_mean.pow(2) - pred_lstd.mul(2).exp()).mul(-0.5)
    return kl_weight * loss.sum(axis=1) / config.n_components

def gmdl_loss(pred_mean, dl_weight):
    pred_mean = pred_mean.reshape(-1, config.n_components, 4)
    if config.n_components == 2:
        dist = torch.pow((pred_mean[:, 0, 0:2] - pred_mean[:, 1, 0:2]), 2).sum(dim=1)
    elif config.n_components == 3:
        dist1 = torch.pow((pred_mean[:, 0, 0:2] - pred_mean[:, 1, 0:2]), 2).sum(dim=1)
        dist2 = torch.pow((pred_mean[:, 0, 0:2] - pred_mean[:, 2, 0:2]), 2).sum(dim=1)
        dist3 = torch.pow((pred_mean[:, 1, 0:2] - pred_mean[:, 2, 0:2]), 2).sum(dim=1)
        dist = (dist1 + dist2 + dist3) / 3
    loss = F.relu(config.maxdist - dist)
    return dl_weight * loss

def log_normal(x, mean, lstd):
    var = lstd.mul(2).exp()
    # log(2 * pi) = 1.8379
    log_prob = -0.5 * torch.sum(1.8379 + torch.log(var) + torch.pow(x - mean, 2) / var, dim=1)
    return log_prob

def gmm_kld_loss(pred_box, pred_mean, pred_lstd, pred_prior_mean, pred_prior_lstd, kll_weight):
    kl_loss = log_normal(pred_box, pred_mean, pred_lstd) - log_normal(pred_box, pred_prior_mean, pred_prior_lstd)
    return kll_weight * kl_loss

def gmm_nent_loss(pred_bbox_qy_logit, enl_weight):
    nent_loss = (F.log_softmax(pred_bbox_qy_logit, dim=1) * F.softmax(pred_bbox_qy_logit, dim=1)).sum(axis=1)
    return enl_weight * nent_loss

def emd_loss_softmax(p_b0, p_s0, p_b1, p_s1, targets, labels):
    # reshape
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().flatten()
    # cons masks
    valid_masks = labels >= 0
    fg_masks = labels > 0
    # multiple class
    pred_delta = pred_delta.reshape(-1, config.num_classes, 4)
    fg_gt_classes = labels[fg_masks]
    pred_delta = pred_delta[fg_masks, fg_gt_classes, :]
    # loss for regression
    localization_loss = smooth_l1_loss(
        pred_delta,
        targets[fg_masks],
        config.rcnn_smooth_l1_beta)
    # loss for classification
    objectness_loss = softmax_loss(pred_score, labels)
    loss = objectness_loss * valid_masks
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)

def refine_loss_softmax(pred_delta, pred_score, targets, labels):
    # reshape
    targets = targets.reshape(-1, 4)
    labels = labels.long().flatten()
    # cons masks
    valid_masks = labels >= 0
    fg_masks = labels > 0
    # multiple class
    pred_delta = pred_delta.reshape(-1, config.num_classes, 4)
    fg_gt_classes = labels[fg_masks]
    pred_delta = pred_delta[fg_masks, fg_gt_classes, :]
    # loss for regression
    localization_loss = smooth_l1_loss(
        pred_delta,
        targets[fg_masks],
        config.rcnn_smooth_l1_beta)
    # loss for classification
    objectness_loss = softmax_loss(pred_score, labels)
    loss = objectness_loss * valid_masks
    loss[fg_masks] = loss[fg_masks] + localization_loss
    return loss.reshape(-1, 1)

def emd_loss_focal(p_b0, p_s0, p_b1, p_s1, targets, labels):
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().reshape(-1, 1)
    valid_mask = (labels >= 0).flatten()
    objectness_loss = focal_loss(pred_score, labels,
            config.focal_loss_alpha, config.focal_loss_gamma)
    fg_masks = (labels > 0).flatten()
    localization_loss = smooth_l1_loss(
            pred_delta[fg_masks],
            targets[fg_masks],
            config.smooth_l1_beta)
    loss = objectness_loss * valid_mask
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)
