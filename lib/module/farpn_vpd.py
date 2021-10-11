import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from config import config
from det_oprs.anchors_generator import AnchorGenerator
from det_oprs.find_top_rpn_proposals import find_top_rpn_proposals, find_test_top_rpn_proposals
from det_oprs.fpn_anchor_target import fpn_anchor_target, fpn_rpn_reshape
from det_oprs.loss_opr import softmax_loss, smooth_l1_loss, kldiv_loss
from det_oprs.my_loss_opr import freeanchor_loss

class RPN(nn.Module):
    def __init__(self, rpn_channel = 256):
        super().__init__()
        self.anchors_generator = AnchorGenerator(
            config.anchor_base_size,
            config.anchor_aspect_ratios,
            config.anchor_base_scale)
        self.rpn_conv = nn.Conv2d(256, rpn_channel, kernel_size=3, stride=1, padding=1)
        self.rpn_cls_score = nn.Conv2d(rpn_channel, config.num_cell_anchors * 2, kernel_size=1, stride=1)
        self.rpn_bbox_offsets = nn.Conv2d(rpn_channel, config.num_cell_anchors * 6, kernel_size=1, stride=1)

        for l in [self.rpn_conv, self.rpn_cls_score, self.rpn_bbox_offsets]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, features, im_info, boxes=None, epoch=None):
        # prediction
        pred_cls_score_list = []
        pred_bbox_offsets_list = []
        for x in features:
            t = F.relu(self.rpn_conv(x))
            pred_cls_score_list.append(self.rpn_cls_score(t))
            pred_bbox_offsets_list.append(self.rpn_bbox_offsets(t))
        # get anchors
        all_anchors_list = []
        # stride: 64,32,16,8,4 p6->p2
        base_stride = 4
        off_stride = 2**(len(features)-1) # 16
        for fm in features:
            layer_anchors = self.anchors_generator(fm, base_stride, off_stride)
            off_stride = off_stride // 2
            all_anchors_list.append(layer_anchors)

        list_size = len(pred_bbox_offsets_list)
        if self.training:
            pred_bbox_list = []
            pred_mean_list = []
            pred_lstd_list = []
            for l in range(list_size):
                mean_perlvl = pred_bbox_offsets_list[l][:, :4*config.num_cell_anchors]
                meanxy_perlvl = mean_perlvl[:, :2*config.num_cell_anchors]
                meanwh_perlvl = mean_perlvl[:, 2*config.num_cell_anchors:]
                lstd_perlvl = pred_bbox_offsets_list[l][:, 4*config.num_cell_anchors:]
                scale = torch.tensor(config.prior_std).type_as(lstd_perlvl)
                std_perlvl = lstd_perlvl.exp().permute(0,2,3,1).mul(scale).permute(0,3,1,2)
                bbox_perlvl = torch.cat([meanxy_perlvl, meanwh_perlvl + std_perlvl * \
                                        torch.randn_like(meanwh_perlvl)], dim=1)
                pred_bbox_list.append(bbox_perlvl)
                pred_mean_list.append(meanwh_perlvl.permute(0,2,3,1).reshape(config.train_batch_per_gpu,-1,2))
                pred_lstd_list.append(lstd_perlvl.permute(0,2,3,1).reshape(config.train_batch_per_gpu,-1,2))
        else:
            pred_bbox_list = [pred_bbox_offsets_list[l][:, :4*config.num_cell_anchors] for l in range(list_size)]
            pred_lstd_list = [pred_bbox_offsets_list[l][:, 4*config.num_cell_anchors:] for l in range(list_size)]
        # sample from the predictions
        rpn_rois = find_top_rpn_proposals(
                self.training, pred_bbox_list, pred_cls_score_list,
                all_anchors_list, im_info)
        rpn_rois = rpn_rois.type_as(features[0])
        if self.training:
            pred_cls_list = [
                _.permute(0, 2, 3, 1).reshape(pred_cls_score_list[0].shape[0], -1, config.num_classes)
                for _ in pred_cls_score_list]
            pred_reg_list = [
                _.permute(0, 2, 3, 1).reshape(pred_bbox_list[0].shape[0], -1, 4)
                for _ in pred_bbox_list]
            all_anchors = torch.cat(all_anchors_list, axis=0)
            all_pred_cls = torch.cat(pred_cls_list, axis=1).reshape(-1, config.num_classes)
            all_pred_cls = torch.softmax(all_pred_cls, dim=-1)[:, 1]
            all_pred_reg = torch.cat(pred_reg_list, axis=1).reshape(-1, 4)
            loss_dict = freeanchor_loss(all_anchors, all_pred_cls, all_pred_reg, boxes, im_info)
            all_pred_meanwh = torch.cat(pred_mean_list, axis=1).reshape(-1, 2)
            all_pred_lstd = torch.cat(pred_lstd_list, axis=1).reshape(-1, 2)
            kl_weight = config.kl_weight * np.power(config.kl_decay, epoch/config.max_epoch)
            loss_kld = kldiv_loss(
                all_pred_meanwh,
                all_pred_lstd,
                kl_weight)
            loss_dict['freeanchor_kldiv_loss'] = loss_kld
            return rpn_rois, loss_dict
        else:
            if config.save_data or config.plot_type != 'normal_plot':
                rpn_rois = find_test_top_rpn_proposals(
                    self.training, pred_bbox_list, pred_cls_score_list, pred_lstd_list, 
                    all_anchors_list, im_info)
            return rpn_rois

