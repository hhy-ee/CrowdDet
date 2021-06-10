import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from config import config
from det_oprs.anchors_generator import AnchorGenerator
from det_oprs.find_top_rpn_proposals import find_top_rpn_proposals
from det_oprs.fpn_anchor_target import fpn_anchor_target, fpn_rpn_gmvpd_reshape
from det_oprs.loss_opr import softmax_loss, smooth_l1_loss, gmkl_loss, gmdl_loss

class RPN(nn.Module):
    def __init__(self, rpn_channel = 256):
        super().__init__()
        self.anchors_generator = AnchorGenerator(
            config.anchor_base_size,
            config.anchor_aspect_ratios,
            config.anchor_base_scale)
        self.rpn_conv = nn.Conv2d(256, rpn_channel, kernel_size=3, stride=1, padding=1)
        self.rpn_cls_score = nn.Conv2d(rpn_channel, config.num_cell_anchors * 2, kernel_size=1, stride=1)
        self.rpn_bbox_offsets = nn.Conv2d(rpn_channel, config.num_cell_anchors * config.n_components* 9, kernel_size=1, stride=1)

        for l in [self.rpn_conv, self.rpn_cls_score, self.rpn_bbox_offsets]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, features, im_info, boxes=None):
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
        
        # variational inference
        list_size = len(pred_bbox_offsets_list)
        if self.training:
            pred_bbox_list = []
            pred_mean_list = []
            pred_lstd_list = []
            for l in range(list_size):
                mean_perlvl = pred_bbox_offsets_list[l][:, :config.num_cell_anchors * config.n_components * 4]
                lstd_perlvl = pred_bbox_offsets_list[l][:, config.num_cell_anchors * config.n_components * 4 : config.num_cell_anchors * config.n_components * 8]
                weight_perlvl = pred_bbox_offsets_list[l][:, config.num_cell_anchors * config.n_components * 8:]
                gaussian_sample = self.gaussian_reparameterize(mean_perlvl, lstd_perlvl)
                bbox_perlvl = self.gumbel_max(gaussian_sample, weight_perlvl)
                pred_bbox_list.append(bbox_perlvl)
                pred_mean_list.append(mean_perlvl)
                pred_lstd_list.append(lstd_perlvl)
        else:
            pred_bbox_list = [pred_bbox_offsets_list[l][:, :4*config.num_cell_anchors] for l in range(list_size)]

        # sample from the predictions
        rpn_rois = find_top_rpn_proposals(
                self.training, pred_bbox_list, pred_cls_score_list,
                all_anchors_list, im_info)
        rpn_rois = rpn_rois.type_as(features[0])
        if self.training:
            rpn_labels, rpn_bbox_targets = fpn_anchor_target(
                    boxes, im_info, all_anchors_list)
            #rpn_labels = rpn_labels.astype(np.int32)
            pred_cls_score, pred_bbox_offsets, pred_mean, pred_lstd = fpn_rpn_gmvpd_reshape(
                pred_cls_score_list, pred_bbox_list, pred_mean_list, pred_lstd_list)
            # rpn loss
            valid_masks = rpn_labels >= 0
            objectness_loss = softmax_loss(
                pred_cls_score[valid_masks],
                rpn_labels[valid_masks])

            pos_masks = rpn_labels > 0
            localization_loss = smooth_l1_loss(
                pred_bbox_offsets[pos_masks],
                rpn_bbox_targets[pos_masks],
                config.rpn_smooth_l1_beta)
            
            kldivergence_loss = gmkl_loss(
                pred_mean[valid_masks],
                pred_lstd[valid_masks],
                config.rpn_kld_beta)

            distance_loss = gmdl_loss(
                pred_mean[valid_masks],
                config.rpn_dil_beta
            )

            normalizer = 1 / valid_masks.sum().item()
            loss_rpn_cls = objectness_loss.sum() * normalizer
            loss_rpn_loc = localization_loss.sum() * normalizer
            loss_rpn_kld = kldivergence_loss.sum() * normalizer
            loss_rpn_dil = distance_loss.sum() * normalizer
            loss_dict = {}
            loss_dict['loss_rpn_cls'] = loss_rpn_cls
            loss_dict['loss_rpn_loc'] = loss_rpn_loc
            loss_dict['loss_rpn_kld'] = loss_rpn_kld
            loss_dict['loss_rpn_dil'] = loss_rpn_dil
            return rpn_rois, loss_dict
        else:
            return rpn_rois
        
    def gaussian_reparameterize(self, mean, sigma):
        samples = mean + sigma.exp() * torch.randn_like(mean)
        return samples
    
    def gumbel_max(self, sample, weight):
        gumbel_sample = -torch.log(-torch.log(torch.rand_like(weight) + 1e-10) + 1e-10)
        component_weight = F.softmax((gumbel_sample + weight) / config.gumbel_temperature, dim=1)
        component_weight = component_weight.permute(0, 2, 3, 1).unsqueeze(-2)
        sample = sample.permute(0, 2, 3, 1).reshape(sample.shape[0], sample.shape[2], 
                                                    sample.shape[3], config.n_components, 4)
        gumbel_max_sample = torch.matmul(component_weight, sample).squeeze(-2)
        return gumbel_max_sample.permute(0, 3, 1, 2)

