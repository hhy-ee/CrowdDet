import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from config import config
from det_oprs.anchors_generator import AnchorGenerator
from det_oprs.find_top_rpn_proposals import find_top_rpn_proposals, find_top_rpn_gmm_box
from det_oprs.fpn_anchor_target import fpn_anchor_target
from det_oprs.fpn_anchor_target import fpn_rpn_vpd_reshape, fpn_rpn_mgmvpd_mv_reshape, fpn_rpn_mgmvpd_qy_reshape

from det_oprs.loss_opr import softmax_loss, smooth_l1_loss, gmm_kld_loss, gmm_nent_loss

class RPN(nn.Module):
    def __init__(self, rpn_channel = 256):
        super().__init__()
        self.anchors_generator = AnchorGenerator(
            config.anchor_base_size,
            config.anchor_aspect_ratios,
            config.anchor_base_scale)
        self.rpn_bbox_prob = rpn_qy_graph()
        self.rpn_conv = nn.Conv2d(int(256 + config.n_components), rpn_channel, kernel_size=3, stride=1, padding=1)
        self.rpn_cls_score = nn.Conv2d(rpn_channel, config.num_cell_anchors * 2, kernel_size=1, stride=1)
        self.rpn_bbox_mean = nn.Conv2d(rpn_channel, config.num_cell_anchors * 4, kernel_size=1, stride=1)
        self.rpn_bbox_lstd = nn.Conv2d(rpn_channel, config.num_cell_anchors * 4, kernel_size=1, stride=1)
        self.rpn_mean_prior = nn.Conv2d(config.n_components, 4, kernel_size=1, stride=1)
        self.rpn_lstd_prior = nn.Conv2d(config.n_components, 4, kernel_size=1, stride=1)

        for l in [self.rpn_conv, self.rpn_cls_score, self.rpn_bbox_mean, 
                    self.rpn_bbox_lstd, self.rpn_mean_prior, self.rpn_lstd_prior]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)


    def forward(self, features, im_info, boxes=None):
        # prediction
        pred_cls_score_list = []
        pred_bbox_mean_list = []
        pred_bbox_lstd_list = []
        pred_bbox_qy_logit_list = []
        pred_prior_mean_list = []
        pred_prior_lstd_list = []

        for x in features:
            pred_bbox_qy_logit = self.rpn_bbox_prob(x)
            pred_bbox_qy_logit_list.append(pred_bbox_qy_logit)

        for n in range(config.n_components):
            for l in range(len(features)):
                x = features[l]
                y = torch.zeros_like(pred_bbox_qy_logit_list[l])
                y[:, n, :, :] = 1
                pred_prior_mean = self.rpn_mean_prior(y)
                pred_prior_lstd = self.rpn_lstd_prior(y)
                t = F.relu(self.rpn_conv(torch.cat((x, y), dim=1)))
                pred_cls_score_list.append(self.rpn_cls_score(t))
                pred_bbox_mean_list.append(self.rpn_bbox_mean(t))
                pred_bbox_lstd_list.append(self.rpn_bbox_lstd(t))
                pred_prior_mean_list.append(pred_prior_mean)
                pred_prior_lstd_list.append(pred_prior_lstd)

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
        list_size = len(pred_bbox_mean_list)
        if self.training:
            pred_bbox_list = []
            pred_mean_list = []
            pred_lstd_list = []
            for l in range(list_size):
                mean_perlvl = pred_bbox_mean_list[l]
                lstd_perlvl = pred_bbox_lstd_list[l]
                bbox_perlvl  = mean_perlvl + lstd_perlvl.exp() * torch.randn_like(mean_perlvl)
                pred_bbox_list.append(bbox_perlvl)
                pred_mean_list.append(mean_perlvl)
                pred_lstd_list.append(bbox_perlvl)
        else:
            pred_bbox_list = [pred_bbox_mean_list[l] for l in range(list_size)]

        # sample from the predictions
        f_size = len(features)
        loss_rpn_cls = 0.0
        loss_rpn_loc = 0.0
        loss_rpn_kld = 0.0
        if self.training:
            rpn_labels, rpn_bbox_targets = fpn_anchor_target(
                    boxes, im_info, all_anchors_list)
            valid_masks = rpn_labels >= 0
            pos_masks = rpn_labels > 0
            normalizer = 1 / valid_masks.sum().item()
            pred_bbox_qy_logit = fpn_rpn_mgmvpd_qy_reshape(pred_bbox_qy_logit_list)
            negentropy_loss = gmm_nent_loss(pred_bbox_qy_logit[valid_masks], config.rpn_enl_beta)
            loss_rpn_enl = negentropy_loss.sum() * normalizer
            cls_weight = F.softmax(pred_bbox_qy_logit[valid_masks], dim=1)
            loc_weight = F.softmax(pred_bbox_qy_logit[pos_masks], dim=1)
            kld_weight = cls_weight
            
            for n in range(config.n_components):
                pred_cls_score, pred_bbox_offsets, pred_mean, pred_lstd = fpn_rpn_vpd_reshape(
                    pred_cls_score_list[n*f_size: (n+1)*f_size], 
                    pred_bbox_list[n*f_size: (n+1)*f_size], 
                    pred_mean_list[n*f_size: (n+1)*f_size], 
                    pred_lstd_list[n*f_size: (n+1)*f_size],
                    )
                pred_prior_mean, pred_prior_lstd = fpn_rpn_mgmvpd_mv_reshape(
                    pred_prior_mean_list[n*f_size: (n+1)*f_size], 
                    pred_prior_lstd_list[n*f_size: (n+1)*f_size]
                    )

                rpn_rois = find_top_rpn_proposals(
                        self.training, 
                        pred_bbox_list[n*f_size: (n+1)*f_size], 
                        pred_cls_score_list[n*f_size: (n+1)*f_size],
                        all_anchors_list, 
                        im_info)
                rpn_rois = rpn_rois.type_as(features[0])
                #rpn_labels = rpn_labels.astype(np.int32)
                
                # rpn loss
                objectness_loss = softmax_loss(
                    pred_cls_score[valid_masks],
                    rpn_labels[valid_masks])

                localization_loss = smooth_l1_loss(
                    pred_bbox_offsets[pos_masks],
                    rpn_bbox_targets[pos_masks],
                    config.rpn_smooth_l1_beta)
            
                kldivergence_loss = gmm_kld_loss(
                    pred_bbox_offsets[valid_masks],
                    pred_mean[valid_masks],
                    pred_lstd[valid_masks],
                    pred_prior_mean[valid_masks],
                    pred_prior_lstd[valid_masks],
                    config.rpn_kld_beta)

                loss_rpn_cls += (objectness_loss * cls_weight[:, n]).sum() * normalizer
                loss_rpn_loc += (localization_loss * loc_weight[:, n]).sum() * normalizer
                loss_rpn_kld += (kldivergence_loss * kld_weight[:, n]).sum() * normalizer
            
            loss_dict = {}
            loss_dict['loss_rpn_cls'] = loss_rpn_cls
            loss_dict['loss_rpn_loc'] = loss_rpn_loc
            loss_dict['loss_rpn_kld'] = loss_rpn_kld
            loss_dict['loss_rpn_enl'] = loss_rpn_enl

            final_pred_bbox_list, final_pred_cls_score_list = find_top_rpn_gmm_box(pred_bbox_list, 
                                                                                   pred_cls_score_list, 
                                                                                   pred_bbox_qy_logit_list)
            rpn_rois = find_top_rpn_proposals(
                        self.training, 
                        final_pred_bbox_list, 
                        final_pred_cls_score_list,
                        all_anchors_list, 
                        im_info)
            return rpn_rois, loss_dict
        else:
            final_pred_bbox_list, final_pred_cls_score_list = find_top_rpn_gmm_box(pred_bbox_list, 
                                                                                   pred_cls_score_list, 
                                                                                   pred_bbox_qy_logit_list)
            return rpn_rois

class rpn_qy_graph(nn.Module):
    def __init__(self, ):
        super().__init__()
        num_convs = 2
        in_channels = 256
        qy_subnet = []
        for _ in range(num_convs):
            qy_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            qy_subnet.append(nn.ReLU(inplace=True))
        self.qy_subnet = nn.Sequential(*qy_subnet)
        self.qy_net = nn.Conv2d(
            in_channels, config.num_cell_anchors * config.n_components,
            kernel_size=3, stride=1, padding=1)
        for modules in [self.qy_subnet, self.qy_net]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        qy_logit = self.qy_net(self.qy_subnet(features))
        return qy_logit