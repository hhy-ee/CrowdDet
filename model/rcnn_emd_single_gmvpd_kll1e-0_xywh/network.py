import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from config import config
from backbone.resnet50 import ResNet50
from backbone.fpn import FPN
from module.rpn import RPN
from layers.pooler import roi_pooler
from det_oprs.bbox_opr import bbox_transform_inv_opr
from det_oprs.fpn_roi_target import fpn_roi_target
from det_oprs.loss_opr import emd_gmvpd_loss_kl
from det_oprs.utils import get_padded_tensor

EPS = 1e-10

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = ResNet50(config.backbone_freeze_at, False)
        self.FPN = FPN(self.resnet50, 2, 6)
        self.RPN = RPN(config.rpn_channel)
        self.RCNN = RCNN()

    def forward(self, image, im_info, epoch=None, gt_boxes=None):
        image = (image - torch.tensor(config.image_mean[None, :, None, None]).type_as(image)) / (
                torch.tensor(config.image_std[None, :, None, None]).type_as(image))
        image = get_padded_tensor(image, 64)
        if self.training:
            return self._forward_train(image, im_info, gt_boxes)
        else:
            return self._forward_test(image, im_info)

    def _forward_train(self, image, im_info, gt_boxes):
        loss_dict = {}
        fpn_fms = self.FPN(image)
        # fpn_fms stride: 64,32,16,8,4, p6->p2
        rpn_rois, loss_dict_rpn = self.RPN(fpn_fms, im_info, gt_boxes)
        rcnn_rois, rcnn_labels, rcnn_bbox_targets = fpn_roi_target(
                rpn_rois, im_info, gt_boxes, top_k=2)
        loss_dict_rcnn = self.RCNN(fpn_fms, rcnn_rois,
                rcnn_labels, rcnn_bbox_targets)
        loss_dict.update(loss_dict_rpn)
        loss_dict.update(loss_dict_rcnn)
        return loss_dict

    def _forward_test(self, image, im_info):
        fpn_fms = self.FPN(image)
        rpn_rois = self.RPN(fpn_fms, im_info)
        pred_bbox = self.RCNN(fpn_fms, rpn_rois)
        return pred_bbox.cpu().detach()

class RCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # roi head
        self.fc1 = nn.Linear(256*7*7 + config.num_components, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc_l1 = nn.Linear(256*7*7, 1024)
        self.fc_l2 = nn.Linear(1024, 1024)
        self.logits = nn.Linear(1024, config.num_components)

        for l in [self.fc1, self.fc2, self.logits]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)
        # box predictor
        self.pred_cls = nn.Linear(1024, config.num_classes)
        self.pred_delta = nn.Linear(1024, (config.num_classes-1) * 8)
        for l in [self.pred_cls]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        for l in [self.pred_delta]:
            nn.init.normal_(l.weight, std=0.001)
            nn.init.constant_(l.bias, 0)

    def forward(self, fpn_fms, rcnn_rois, labels=None, bbox_targets=None):
        # stride: 64,32,16,8,4 -> 4, 8, 16, 32
        fpn_fms = fpn_fms[1:][::-1]
        stride = [4, 8, 16, 32]
        pool_features = roi_pooler(fpn_fms, rcnn_rois, stride, (7, 7), "ROIAlignV2")
        flatten_feature = torch.flatten(pool_features, start_dim=1)
        # gumbel max
        logit_feature = F.relu_(self.fc_l1(flatten_feature))
        logit_feature = F.relu_(self.fc_l2(logit_feature))
        logits = self.logits(logit_feature).view(-1, config.num_components)
        gumbel = -torch.log(-torch.log(torch.rand_like(logits) + EPS) + EPS)
        gumbel_prob = F.softmax((logits + gumbel) / config.gumbel_temperature, dim=1)
        # soft GMM
        gumbel_feature = torch.cat([flatten_feature, gumbel_prob], dim=1)
        gumbel_feature = F.relu_(self.fc1(gumbel_feature))
        gumbel_feature = F.relu_(self.fc2(gumbel_feature))
        pred_gumbel_cls = self.pred_cls(gumbel_feature)
        pred_gumbel_delta = self.pred_delta(gumbel_feature)
        # hard GMM
        prob_features = []
        for n in range(config.num_components):
            hard_prob = torch.zeros_like(gumbel_prob)
            hard_prob[:, n] = 1
            prob_feature = torch.cat([flatten_feature, hard_prob], dim=1)
            prob_feature = F.relu_(self.fc1(prob_feature))
            prob_feature = F.relu_(self.fc2(prob_feature))
            prob_features.append(prob_feature.unsqueeze(0))
        prob_features = torch.cat(prob_features, dim=0)
        pred_cls = self.pred_cls(prob_features)
        pred_delta = self.pred_delta(prob_features)
        pred_cls_0, pred_cls_1 = pred_cls[0,:], pred_cls[1,:]
        pred_delta_0, pred_delta_1 = pred_delta[0,:], pred_delta[1,:]
        # loss computation
        if self.training:
            loss0 = emd_gmvpd_loss_kl(
                        pred_delta_0, pred_cls_0,
                        pred_delta_1, pred_cls_1,
                        bbox_targets, labels)
            loss1 = emd_gmvpd_loss_kl(
                        pred_delta_1, pred_cls_1,
                        pred_delta_0, pred_cls_0,
                        bbox_targets, labels)
            loss = torch.cat([loss0, loss1], axis=1)
            # requires_grad = False
            _, min_indices = loss.min(axis=1)
            loss_emd = loss[torch.arange(loss.shape[0]), min_indices]
            loss_emd = loss_emd.mean()
            loss_dict = {}
            loss_dict['loss_rcnn_emd'] = loss_emd
            return loss_dict
        else:
            class_num = pred_cls_0.shape[-1] - 1
            tag = torch.arange(class_num).type_as(pred_cls_0)+1
            tag = tag.repeat(pred_cls_0.shape[0], 1).reshape(-1,1)
            pred_scores_0 = F.softmax(pred_cls_0, dim=-1)[:, 1:].reshape(-1, 1)
            pred_scores_1 = F.softmax(pred_cls_1, dim=-1)[:, 1:].reshape(-1, 1)
            pred_delta_0 = pred_delta_0.reshape(-1,8)[:, :4]
            pred_delta_1 = pred_delta_1.reshape(-1,8)[:, :4]
            pred_lstd_0 = pred_delta_0.reshape(-1,8)[:, 4:8]
            pred_lstd_1 = pred_delta_1.reshape(-1,8)[:, 4:8]
            pred_logit_0 = logits.reshape(-1,2)[:, 0:1]
            pred_logit_1 = logits.reshape(-1,2)[:, 1:2]
            base_rois = rcnn_rois[:, 1:5].repeat(1, class_num).reshape(-1, 4)
            pred_bbox_0 = restore_bbox(base_rois, pred_delta_0, True)
            pred_bbox_1 = restore_bbox(base_rois, pred_delta_1, True)
            if 'kl' not in config.test_nms_method:
                pred_bbox_0 = torch.cat([pred_bbox_0, pred_scores_0, tag], axis=1)
                pred_bbox_1 = torch.cat([pred_bbox_1, pred_scores_1, tag], axis=1)
            else:
                # mutil-box kld
                # pred_mip_kld_0 = (1 + pred_lstd_0.mul(2) - pred_lstd_1.mul(2) - (pred_lstd_0.mul(2).exp() + \
                # (pred_delta_0 - pred_delta_1).pow(2))/ pred_lstd_1.mul(2).exp()).mul(-0.5).mean(dim=1, keepdim=True)
                # pred_mip_kld_1 = (1 + pred_lstd_1.mul(2) - pred_lstd_0.mul(2) - (pred_lstd_1.mul(2).exp() + \
                # (pred_delta_1 - pred_delta_0).pow(2))/ pred_lstd_0.mul(2).exp()).mul(-0.5).mean(dim=1, keepdim=True)
                
                # box vs N(0,1) kld
                scale = torch.tensor(config.prior_std).type_as(pred_lstd_0)
                pred_scale_lstd_0 = pred_lstd_0.exp().mul(scale).log()
                pred_scale_lstd_1 = pred_lstd_1.exp().mul(scale).log()
                pred_prob = F.softmax(torch.cat([pred_logit_0, pred_logit_1], dim=1), dim=1) 
                pred_prob_0, pred_prob_1 = torch.split(pred_prob, 1, dim=1)
                
                pred_bbox_0 = torch.cat([pred_bbox_0, pred_scores_0, tag, pred_prob_0, pred_scale_lstd_0], axis=1)
                pred_bbox_1 = torch.cat([pred_bbox_1, pred_scores_1, tag, pred_prob_1, pred_scale_lstd_1], axis=1)
            pred_bbox = torch.cat((pred_bbox_0, pred_bbox_1), axis=1)
            return pred_bbox

def restore_bbox(rois, deltas, unnormalize=True):
    if unnormalize:
        std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(deltas)
        mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(deltas)
        deltas = deltas * std_opr
        deltas = deltas + mean_opr
    pred_bbox = bbox_transform_inv_opr(rois, deltas)
    return pred_bbox