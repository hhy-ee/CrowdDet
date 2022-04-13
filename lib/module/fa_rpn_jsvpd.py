import torch
import math
from torch import nn
import torch.nn.functional as F
import numpy as np

from config import config
from det_oprs.anchors_generator import AnchorGenerator
from det_oprs.find_top_rpn_proposals import find_top_rpn_vpd_proposals_fa
from det_oprs.fa_anchor_target import fa_anchor_target
from det_oprs.fpn_anchor_target import fpn_anchor_target, fpn_rpn_reshape
from det_oprs.loss_opr import softmax_loss, smooth_l1_loss, js_gaussian_loss
from det_oprs.my_loss_opr import freeanchor_vpd_loss_iou

class RPN(nn.Module):
    def __init__(self, rpn_channel = 256):
        super().__init__()
        self.anchors_generator = AnchorGenerator(
            config.anchor_base_size,
            config.anchor_aspect_ratios,
            config.anchor_base_scale)
        self.rpn_conv = nn.Conv2d(256, rpn_channel, kernel_size=3, stride=1, padding=1)
        self.rpn_cls_score = nn.Conv2d(rpn_channel, config.num_cell_anchors * 1, kernel_size=1, stride=1)
        self.rpn_bbox_offsets = nn.Conv2d(rpn_channel, config.num_cell_anchors * 8, kernel_size=1, stride=1)

        for l in [self.rpn_conv, self.rpn_cls_score, self.rpn_bbox_offsets]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        prior_prob = 0.01
        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.rpn_cls_score.bias, bias_value)

    def forward(self, features, im_info, boxes=None):
        # prediction
        pred_cls_score_list = []
        pred_bbox_dists_list = []
        pred_bbox_offsets_list = []
        pred_bbox_vpd_offsets_list = []
        for x in features:
            t = F.relu(self.rpn_conv(x))
            pred_cls_score_list.append(self.rpn_cls_score(t))
            pred_bbox_dists_list.append(self.rpn_bbox_offsets(t))
        # get anchors
        all_anchors_list = []
        # stride: 64,32,16,8,4 p6->p2
        base_stride = 4
        off_stride = 2**(len(features)-1) # 16
        for fm in features:
            layer_anchors = self.anchors_generator(fm, base_stride, off_stride)
            off_stride = off_stride // 2
            all_anchors_list.append(layer_anchors)

         # variational inference
        for dist in pred_bbox_dists_list:
            N, C, W, H = dist.shape
            pred_mean = dist.reshape(N, config.num_cell_anchors, 8, W, H)[:, :, :4]
            pred_lstd = dist.reshape(N, config.num_cell_anchors, 8, W, H)[:, :, 4:]
            pred_offset = pred_mean + pred_lstd.exp() * torch.randn_like(pred_mean)
            pred_bbox_offsets_list.append(pred_mean.reshape(N, -1, W, H))
            if config.sampling:
                pred_bbox_vpd_offsets_list.append(pred_offset.reshape(N, -1, W, H))
            else:
                pred_bbox_vpd_offsets_list.append(pred_mean.reshape(N, -1, W, H))
    
        # sample from the predictions
        vpd_rois, rois = find_top_rpn_vpd_proposals_fa(
                self.training, pred_bbox_offsets_list, pred_bbox_vpd_offsets_list, 
                pred_cls_score_list, all_anchors_list, im_info)
        if self.training:
            rpn_vpd_rois = vpd_rois.type_as(features[0])
            rpn_rois = rois.type_as(features[0])
            all_anchors = torch.cat(all_anchors_list, axis=0)
            rpn_labels, rpn_bbox_targets = fa_anchor_target(
                all_anchors, boxes, im_info, top_k=config.pre_anchor_topk)
            # rpn loss
            valid_masks = rpn_labels >= 0
            pred_cls_list = [
                _.permute(0, 2, 3, 1).reshape(pred_cls_score_list[0].shape[0], -1, 1)
                for _ in pred_cls_score_list]
            pred_mean_list = [
                _.permute(0, 2, 3, 1).reshape(pred_bbox_offsets_list[0].shape[0], -1, 4)
                for _ in pred_bbox_offsets_list]
            pred_sample_list = [
                _.permute(0, 2, 3, 1).reshape(pred_bbox_vpd_offsets_list[0].shape[0], -1, 4)
                for _ in pred_bbox_vpd_offsets_list]
            pred_dist_list = [
                _.permute(0, 2, 3, 1).reshape(pred_bbox_dists_list[0].shape[0], -1, 8)
                for _ in pred_bbox_dists_list]
            all_pred_cls = torch.cat(pred_cls_list, axis=1).reshape(-1, 1)
            all_pred_cls = torch.sigmoid(all_pred_cls)
            all_pred_mean = torch.cat(pred_mean_list, axis=1).reshape(-1, 4)
            all_pred_sample = torch.cat(pred_sample_list, axis=1).reshape(-1, 4)
            all_pred_dist = torch.cat(pred_dist_list, axis=1).reshape(-1, 8)
            loss_dict = freeanchor_vpd_loss_iou(
                all_anchors, all_pred_cls, all_pred_mean, 
                all_pred_sample, boxes, im_info)
            pos_masks = (rpn_labels > 0).flatten()
            loss_jsd = js_gaussian_loss(
                    all_pred_dist[pos_masks],
                    rpn_bbox_targets[pos_masks],
                    config.kl_weight)
            normalizer = 1 / valid_masks.sum().item()
            loss_rpn_jsd = loss_jsd.sum() * normalizer
            loss_dict['loss_rpn_jsd'] = loss_rpn_jsd
            return (rpn_vpd_rois, rpn_rois), loss_dict
        else:
            rpn_rois = rois.type_as(features[0])
            return rpn_rois