import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from config import config
from backbone.resnet50 import ResNet50
from backbone.fpn import FPN
from det_oprs.anchors_generator import AnchorGenerator
from det_oprs.atss_anchor_target import atss_anchor_target, centerness_target
from det_oprs.bbox_opr import bbox_transform_inv_opr
from det_oprs.loss_opr import focal_loss, giou_loss, js_gmm_loss
from det_oprs.utils import get_padded_tensor

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = ResNet50(config.backbone_freeze_at, False)
        self.FPN = FPN(self.resnet50, 3, 7)
        self.R_Head = RetinaNet_Head()
        self.R_Anchor = RetinaNet_Anchor()
        self.R_Criteria = RetinaNet_Criteria()

    def forward(self, image, im_info, epoch=None, gt_boxes=None):
        # pre-processing the data
        image = (image - torch.tensor(config.image_mean[None, :, None, None]).type_as(image)) / (
                torch.tensor(config.image_std[None, :, None, None]).type_as(image))
        image = get_padded_tensor(image, 64)
        # do inference
        # stride: 128,64,32,16,8, p7->p3
        fpn_fms = self.FPN(image)
        anchors_list = self.R_Anchor(fpn_fms)
        pred_cls_list, pred_reg_list, pred_ctn_list = self.R_Head(fpn_fms)
        # release the useless data
        if self.training:
            loss_dict = self.R_Criteria(
                    pred_cls_list, pred_reg_list, pred_ctn_list, anchors_list, gt_boxes, im_info)
            return loss_dict
        else:
            #pred_bbox = union_inference(
            #        anchors_list, pred_cls_list, pred_reg_list, im_info)
            pred_bbox = per_layer_inference(
                    anchors_list, pred_cls_list, pred_reg_list, pred_ctn_list, im_info)
            return pred_bbox.cpu().detach()

class RetinaNet_Anchor():
    def __init__(self):
        self.anchors_generator = AnchorGenerator(
            config.anchor_base_size,
            config.anchor_aspect_ratios,
            config.anchor_base_scale)

    def __call__(self, fpn_fms):
        # get anchors
        all_anchors_list = []
        base_stride = 8
        off_stride = 2**(len(fpn_fms)-1) # 16
        for fm in fpn_fms:
            layer_anchors = self.anchors_generator(fm, base_stride, off_stride)
            off_stride = off_stride // 2
            all_anchors_list.append(layer_anchors)
        return all_anchors_list

class RetinaNet_Criteria(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_normalizer = 100 # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9

    def __call__(self, pred_cls_list, pred_reg_list, pred_ctn_list, anchors_list, gt_boxes, im_info):
        num_levels = [int(cls.shape[1]) for cls in pred_cls_list]
        all_anchors = torch.cat(anchors_list, axis=0)
        all_pred_cls = torch.cat(pred_cls_list, axis=1).reshape(-1, config.num_classes-1)
        all_pred_ctn = torch.cat(pred_ctn_list, axis=1).reshape(-1)
        all_pred_dist = torch.cat(pred_reg_list, axis=1).reshape(-1, 4, 2 * config.project.shape[1])
        # get ground truth
        labels, bbox_target = atss_anchor_target(all_anchors, gt_boxes, num_levels, im_info)
        fg_mask = (labels > 0).flatten()
        valid_mask = (labels >= 0).flatten()
        anchor_target = all_anchors.repeat(config.train_batch_per_gpu, 1)[fg_mask]
        ctn_target = centerness_target(anchor_target, bbox_target[fg_mask])
        # gumbel max
        n_component = config.project.shape[1]
        pos_pred_prob = all_pred_dist[..., :n_component][fg_mask].reshape(-1, n_component)
        gumbel_sample = -torch.log(-torch.log(torch.rand_like(pos_pred_prob) + 1e-10) + 1e-10)
        gumbel_weight = F.softmax((gumbel_sample + pos_pred_prob) / config.gumbel_temperature, dim=1)
        pos_weight = F.softmax(pos_pred_prob, dim=1)
        # variational inference
        project_mean = torch.tensor(config.project).type_as(all_pred_dist).repeat(gumbel_weight.shape[0], 1)
        pos_pred_lstd = all_pred_dist[..., n_component:][fg_mask].reshape(-1, n_component)
        pos_pred_delta = project_mean + pos_pred_lstd.exp() * torch.randn_like(project_mean)
        pos_pred_delta = gumbel_weight.mul(pos_pred_delta).sum(dim=1).reshape(-1, 4)
        # regression loss
        loss_ctn = F.binary_cross_entropy_with_logits(
                all_pred_ctn[fg_mask], ctn_target)
        loss_reg = giou_loss( 
                pos_pred_delta,
                bbox_target[fg_mask],
                anchor_target)
        loss_cls = focal_loss(
                all_pred_cls[valid_mask],
                labels[valid_mask],
                config.focal_loss_alpha,
                config.focal_loss_gamma)
        loss_jsd = js_gmm_loss(
                pos_weight,
                project_mean,
                pos_pred_lstd, 
                bbox_target[fg_mask],
                config.kl_weight)
        num_pos_anchors = fg_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
            ) * max(num_pos_anchors, 1)
        loss_ctn = loss_ctn.sum() / self.loss_normalizer
        loss_reg = loss_reg.sum() / self.loss_normalizer
        loss_cls = loss_cls.sum() / self.loss_normalizer
        loss_jsd = loss_jsd.sum() / self.loss_normalizer
        loss_dict = {}
        loss_dict['atss_focal_loss'] = loss_cls
        loss_dict['atss_smooth_l1'] = loss_reg
        loss_dict['atss_centerness'] = loss_ctn
        loss_dict['atss_jsd_loss'] = loss_jsd
        return loss_dict

class RetinaNet_Head(nn.Module):
    def __init__(self):
        super().__init__()
        num_convs = 4
        in_channels = 256
        reg_channels = 64
        if config.stat_mode == 'std' or config.stat_mode == 'prb' or config.stat_mode == 'pmf':
            ref_channels = 16
        elif config.stat_mode == 'std&prb' or config.stat_mode == 'std&pmf':
            ref_channels = 32
        elif config.stat_mode == 'std&prb&pmf':
            ref_channels = 48
        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(nn.ReLU(inplace=True))
            bbox_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_subnet.append(nn.ReLU(inplace=True))
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)

        # refinement
        conf_vector = [nn.Conv2d(ref_channels, reg_channels, 1)]
        conf_vector += [nn.ReLU(inplace=True)]
        conf_vector += [nn.Conv2d(reg_channels, 1, 1), nn.Sigmoid()]
        self.reg_conf = nn.Sequential(*conf_vector)

        # predictor
        self.cls_score = nn.Conv2d(
            in_channels, config.num_cell_anchors * (config.num_classes-1),
            kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, config.num_cell_anchors * 4 * 2 * config.project.shape[1],
            kernel_size=3, stride=1, padding=1)
        self.centerness_pred = nn.Conv2d(
            in_channels, config.num_cell_anchors * 1,
            kernel_size=3, stride=1, padding=1)
        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, 
                        self.bbox_pred, self.centerness_pred, self.reg_conf]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
        prior_prob = 0.01
        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        pred_cls = []
        pred_reg = []
        pred_ctn = []
        for feature in features:
            cls_score = self.cls_score(self.cls_subnet(feature))
            bbox_pred = self.bbox_pred(self.bbox_subnet(feature))
            # refinement
            N, _, H, W = bbox_pred.size()
            n_component = config.project.shape[1]
            acc = config.project[0,-1] / (config.project.shape[1] - 1) 
            logit = bbox_pred.permute(0,2,3,1).reshape(-1, n_component*2, 1)[:, :n_component]
            prob = torch.softmax(logit, dim=1)
            lstd = bbox_pred.permute(0,2,3,1).reshape(-1, n_component*2, 1)[:, n_component:]
            mean = torch.tensor(config.project).type_as(logit). \
                repeat(logit.shape[0], 1).reshape(-1, n_component, 1)
            q0 = torch.distributions.normal.Normal(mean, lstd.exp())
            z0 = torch.tensor(config.project).type_as(logit).reshape(1, 1, n_component).\
                repeat(logit.shape[0], logit.shape[1], 1)
            pmf = (q0.cdf(z0 + acc) - q0.cdf(z0 - acc)).mul(prob).sum(dim=1, keepdim=False)
            # pmf refine
            toppmf, idx = pmf.topk(config.reg_topk, dim=1)
            toppmf = toppmf.reshape(N, H, W, -1).permute(0, 3, 1, 2)
            # lstd refine
            toplstd = torch.gather(lstd.reshape(-1, n_component), dim=1, index=idx)
            toplstd = toplstd.reshape(N, H, W, -1).permute(0, 3, 1, 2)
            # prob refine
            topprob = torch.gather(prob.reshape(-1, n_component), dim=1, index=idx)
            topprob = topprob.reshape(N, H, W, -1).permute(0, 3, 1, 2)
            if config.stat_mode == 'std':
                stat = toplstd
            elif config.stat_mode == 'prb':
                stat = topprob
            elif config.stat_mode == 'pmf':
                stat = toppmf
            elif config.stat_mode == 'std&prb':
                stat = torch.cat([toplstd, topprob], dim=1)
            elif config.stat_mode == 'std&pmf':
                stat = torch.cat([toplstd, toppmf], dim=1)
            elif config.stat_mode == 'std&prb&pmf':
                stat = torch.cat([toplstd, topprob, toppmf], dim=1)
            quality_score = self.reg_conf(stat)
            cls_score = cls_score.sigmoid() * quality_score
            pred_cls.append(cls_score)
            pred_reg.append(bbox_pred)
            pred_ctn.append(self.centerness_pred(self.bbox_subnet(feature)))
        # reshape the predictions
        assert pred_cls[0].dim() == 4
        pred_cls_list = [
            _.permute(0, 2, 3, 1).reshape(pred_cls[0].shape[0], -1, config.num_classes-1)
            for _ in pred_cls]
        pred_reg_list = [
            _.permute(0, 2, 3, 1).reshape(pred_reg[0].shape[0], -1, 4 * 2 * config.project.shape[1])
            for _ in pred_reg]
        pred_ctn_list = [
            _.permute(0, 2, 3, 1).reshape(pred_reg[0].shape[0], -1, 1)
            for _ in pred_ctn]
        return pred_cls_list, pred_reg_list, pred_ctn_list

def per_layer_inference(anchors_list, pred_cls_list, pred_reg_list, pred_ctn_list, im_info):
    keep_anchors = []
    keep_scr = []
    keep_reg = []
    class_num = pred_cls_list[0].shape[-1]
    for l_id in range(len(anchors_list)):
        anchors = anchors_list[l_id].reshape(-1, 4)
        pred_cls = pred_cls_list[l_id][0].reshape(-1, class_num)
        pred_reg = pred_reg_list[l_id][0].reshape(-1, 2 * config.project.shape[1])
        weight = F.softmax(pred_reg[:, :config.project.shape[1]], dim=1)
        project = torch.tensor(config.project).type_as(pred_reg).repeat(weight.shape[0], 1)
        pred_reg = weight.mul(project).sum(dim=1).reshape(-1, 4)
        pred_ctn = pred_ctn_list[l_id][0].reshape(-1, 1)
        pred_scr = pred_cls * torch.sigmoid(pred_ctn)
        if len(anchors) > config.test_layer_topk:
            ruler = pred_scr.max(axis=1)[0]
            _, inds = ruler.topk(config.test_layer_topk, dim=0)
            inds = inds.flatten()
            keep_anchors.append(anchors[inds])
            keep_scr.append(pred_scr[inds])
            keep_reg.append(pred_reg[inds])
        else:
            keep_anchors.append(anchors)
            keep_scr.append(pred_scr)
            keep_reg.append(pred_reg)
    keep_anchors = torch.cat(keep_anchors, axis = 0)
    keep_scr = torch.cat(keep_scr, axis = 0)
    keep_reg = torch.cat(keep_reg, axis = 0)
    # multiclass
    tag = torch.arange(class_num).type_as(keep_scr)+1
    tag = tag.repeat(keep_scr.shape[0], 1).reshape(-1,1)
    pred_scores = keep_scr.reshape(-1, 1)
    pred_bbox = restore_bbox(keep_anchors, keep_reg, False)
    pred_bbox = pred_bbox.repeat(1, class_num).reshape(-1, 4)
    pred_bbox = torch.cat([pred_bbox, pred_scores, tag], axis=1)
    return pred_bbox

def union_inference(anchors_list, pred_cls_list, pred_reg_list, im_info):
    anchors = torch.cat(anchors_list, axis = 0)
    pred_cls = torch.cat(pred_cls_list, axis = 1)[0]
    pred_cls = torch.sigmoid(pred_cls)
    pred_reg = torch.cat(pred_reg_list, axis = 1)[0]
    class_num = pred_cls_list[0].shape[-1]
    # multiclass
    tag = torch.arange(class_num).type_as(keep_cls)+1
    tag = tag.repeat(keep_cls.shape[0], 1).reshape(-1,1)
    pred_scores = keep_cls.reshape(-1, 1)
    pred_bbox = restore_bbox(keep_anchors, keep_reg, False)
    pred_bbox = pred_bbox.repeat(1, class_num).reshape(-1, 4)
    pred_bbox = torch.cat([pred_bbox, pred_scores, tag], axis=1)
    return pred_bbox

def restore_bbox(rois, deltas, unnormalize=True):
    if unnormalize:
        std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(deltas)
        mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(deltas)
        deltas = deltas * std_opr
        deltas = deltas + mean_opr
    pred_bbox = bbox_transform_inv_opr(rois, deltas)
    return pred_bbox

