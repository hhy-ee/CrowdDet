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
from det_oprs.loss_opr import softmax_loss, smooth_l1_loss
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
                rpn_rois, im_info, gt_boxes, top_k=1)
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
        self.fc1 = nn.Linear(256*7*7, 1024)
        self.fc2 = nn.Linear(1024 + config.num_components, 1024)
        self.fc_l1 = nn.Linear(256*7*7, 1024)
        self.fc_l2 = nn.Linear(1024, 1024)
        self.logits = nn.Linear(1024, config.num_components)
        self.fc_mu = nn.Linear(2, 4)
        self.fc_std = nn.Linear(2, 4)

        for l in [self.fc1, self.fc2, self.logits, self.fc_mu, self.fc_std]:
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
        prior_mean = self.fc_mu(gumbel_prob)
        prior_lstd = self.fc_std(gumbel_prob)
        # soft GMM
        gumbel_feature = F.relu_(self.fc1(flatten_feature))
        gumbel_feature = torch.cat([gumbel_feature, gumbel_prob], dim=1)
        gumbel_feature = F.relu_(self.fc2(gumbel_feature))
        pred_cls = self.pred_cls(gumbel_feature)
        pred_dist = self.pred_delta(gumbel_feature)
        # loss computation
        if self.training:
            # loss for regression
            labels = labels.long().flatten()
            fg_masks = labels > 0
            valid_masks = labels >= 0
            # multi class
            pred_dist = pred_dist[fg_masks, :]
            # Gaussian reparameterization
            pred_mean = pred_dist[:, :4]
            pred_lstd = pred_dist[:, 4:]
            pred_delta =  pred_mean + pred_lstd.exp()*torch.randn_like(pred_mean)
            localization_loss = smooth_l1_loss(
                pred_delta,
                bbox_targets[fg_masks],
                config.rcnn_smooth_l1_beta)
            # loss for classification
            objectness_loss = softmax_loss(pred_cls, labels)
            objectness_loss = objectness_loss * valid_masks
            # loss for kl divergence
            pos_prior_mean = prior_mean[fg_masks]
            pos_prior_lstd = prior_lstd[fg_masks]
            q0 = torch.distributions.normal.Normal(pred_mean, pred_lstd.exp())
            prior = torch.distributions.normal.Normal(pos_prior_mean, pos_prior_lstd.exp())
            kldivergence_loss = q0.log_prob(pred_delta) - prior.log_prob(pred_delta)
            # loss for categorical component
            pos_prob = F.softmax(logits[fg_masks], dim=1)
            pos_logprob = F.log_softmax(logits[fg_masks], dim=1) 
            categorical_loss = (pos_prob * pos_logprob).sum(dim=1) - np.log(0.5)

            normalizer = 1.0 / valid_masks.sum().item()
            loss_rcnn_loc = localization_loss.sum() * normalizer
            loss_rcnn_cls = objectness_loss.sum() * normalizer
            loss_rcnn_kld = kldivergence_loss.sum() * normalizer
            loss_rcnn_cat = categorical_loss.sum() * normalizer
            loss_dict = {}
            loss_dict['loss_rcnn_loc'] = loss_rcnn_loc
            loss_dict['loss_rcnn_cls'] = loss_rcnn_cls
            loss_dict['loss_rcnn_kld'] = loss_rcnn_kld
            loss_dict['loss_rcnn_cat'] = loss_rcnn_cat
            return loss_dict
        else:
            class_num = pred_cls.shape[-1] - 1
            tag = torch.arange(class_num).type_as(pred_cls)+1
            tag = tag.repeat(pred_cls.shape[0], 1).reshape(-1,1)
            pred_scores = F.softmax(pred_cls, dim=-1)[:, 1:].reshape(-1, 1)
            pred_delta = pred_delta[:, 4:].reshape(-1, 4)
            base_rois = rcnn_rois[:, 1:5].repeat(1, class_num).reshape(-1, 4)
            pred_bbox = restore_bbox(base_rois, pred_delta, True)
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
