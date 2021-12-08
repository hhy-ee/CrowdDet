import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from config import config
from backbone.resnet50 import ResNet50
from backbone.fpn import FPN
from det_oprs.anchors_generator import AnchorGenerator
from det_oprs.retina_anchor_target import retina_anchor_target
from det_oprs.bbox_opr import bbox_transform_inv_opr
from det_oprs.loss_opr import focal_loss, smooth_l1_loss, kldiv_loss
from det_oprs.utils import get_padded_tensor

EPS = 1e-10

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
        pred_cls_list, pred_reg_list, pred_prior_mean_list, \
            pred_prior_lstd_list, pred_logits_list= self.R_Head(fpn_fms)
        # release the useless data
        if self.training:
            loss_dict = self.R_Criteria(
                    pred_cls_list, pred_reg_list, pred_prior_mean_list, 
                    pred_prior_lstd_list, pred_logits_list, anchors_list, gt_boxes, im_info)
            return loss_dict
        else:
            #pred_bbox = union_inference(
            #        anchors_list, pred_cls_list, pred_reg_list, im_info)
            pred_bbox = per_layer_inference(
                    anchors_list, pred_cls_list, pred_reg_list, im_info)
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

    def __call__(self, pred_cls_list, pred_reg_list, pred_prior_mean_list, 
            pred_prior_lstd_list, pred_logits_list, anchors_list, gt_boxes, im_info):
        all_anchors = torch.cat(anchors_list, axis=0)
        all_pred_cls = torch.cat(pred_cls_list, axis=1).reshape(-1, config.num_classes-1)
        all_pred_cls = torch.sigmoid(all_pred_cls)
        all_pred_reg = torch.cat(pred_reg_list, axis=1).reshape(-1, 8)
        all_prior_mean = torch.cat(pred_prior_mean_list, axis=1).reshape(-1, 4)
        all_prior_std = torch.cat(pred_prior_lstd_list, axis=1).reshape(-1, 4)
        all_pred_logits = torch.cat(pred_logits_list, axis=1).reshape(-1, 2)
        # variational inference
        all_pred_mean = all_pred_reg[:, :config.num_cell_anchors * 4]
        all_pred_lstd = all_pred_reg[:, config.num_cell_anchors * 4:]
        all_pred_reg = all_pred_mean + all_pred_lstd * torch.randn_like(all_pred_lstd)
        # get ground truth
        labels, bbox_target = retina_anchor_target(all_anchors, gt_boxes, im_info, top_k=1)
        # regression loss
        fg_mask = (labels > 0).flatten()
        valid_mask = (labels >= 0).flatten()
        loss_reg = smooth_l1_loss(
                all_pred_reg[fg_mask],
                bbox_target[fg_mask],
                config.smooth_l1_beta)
        loss_cls = focal_loss(
                all_pred_cls[valid_mask],
                labels[valid_mask],
                config.focal_loss_alpha,
                config.focal_loss_gamma)
        # loss for kl divergence
        pos_prior_mean = all_prior_mean[fg_mask]
        pos_prior_lstd = all_prior_std[fg_mask]
        pos_mean = all_pred_mean[fg_mask]
        pos_lstd = all_pred_lstd[fg_mask]
        q0 = torch.distributions.normal.Normal(pos_mean, pos_lstd.exp())
        prior = torch.distributions.normal.Normal(pos_prior_mean, pos_prior_lstd.exp())
        loss_kld = q0.log_prob(all_pred_reg[fg_mask]) - prior.log_prob(all_pred_reg[fg_mask])
        # loss for categorical component
        pos_prob = F.softmax(all_pred_logits[fg_mask], dim=1)
        pos_logprob = F.log_softmax(all_pred_logits[fg_mask], dim=1) 
        loss_cat = (pos_prob * pos_logprob).sum(dim=1) - np.log(0.5)

        num_pos_anchors = fg_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
            ) * max(num_pos_anchors, 1)
        loss_reg = loss_reg.sum() / self.loss_normalizer
        loss_cls = loss_cls.sum() / self.loss_normalizer
        loss_kld = loss_kld.sum() / self.loss_normalizer
        loss_cat = loss_cat.sum() / self.loss_normalizer
        loss_dict = {}
        loss_dict['retina_focal_loss'] = loss_cls
        loss_dict['retina_smooth_l1'] = loss_reg
        loss_dict['retina_kld_loss'] = loss_kld
        loss_dict['retina_cat_loss'] = loss_cat
        return loss_dict

class RetinaNet_Head(nn.Module):
    def __init__(self):
        super().__init__()
        num_convs = 4
        in_channels = 256
        cls_subnet = []
        bbox_subnet = []
        self.conv_mu = nn.Conv2d(config.num_components, 4, kernel_size=3, stride=1, padding=1)
        self.conv_std = nn.Conv2d(config.num_components, 4, kernel_size=3, stride=1, padding=1)
        cls_subnet.append(
             nn.Conv2d(in_channels + config.num_components, in_channels, kernel_size=3, stride=1, padding=1)
        )
        cls_subnet.append(nn.ReLU(inplace=True))
        bbox_subnet.append(
                nn.Conv2d(in_channels + config.num_components, in_channels, kernel_size=3, stride=1, padding=1)
            )
        bbox_subnet.append(nn.ReLU(inplace=True))
        for _ in range(num_convs-1):
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
        # predictor
        self.cls_score = nn.Conv2d(
            in_channels, config.num_cell_anchors * (config.num_classes-1),
            kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, config.num_cell_anchors * 8,
            kernel_size=3, stride=1, padding=1)
        # gmm prob
        self.gmm_prob = nn.Sequential(*[
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, config.num_cell_anchors * config.num_components,
            kernel_size=3, stride=1, padding=1)
            ])

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, 
                        self.bbox_pred, self.gmm_prob, self.conv_mu, self.conv_std]:
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
        pred_prior_mean = []
        pred_prior_lstd = []
        pred_logits = []
        for feature in features:
            logits = self.gmm_prob(feature)
            gumbel = -torch.log(-torch.log(torch.rand_like(logits) + EPS) + EPS)
            gumbel_prob = F.softmax((logits + gumbel) / config.gumbel_temperature, dim=1)
            gumbel_feature = torch.cat([feature, gumbel_prob], dim=1)
            pred_cls.append(self.cls_score(self.cls_subnet(gumbel_feature)))
            pred_reg.append(self.bbox_pred(self.bbox_subnet(gumbel_feature)))
            pred_prior_mean.append(self.conv_mu(gumbel_prob))
            pred_prior_lstd.append(self.conv_std(gumbel_prob))
            pred_logits.append(logits)

        # reshape the predictions
        assert pred_cls[0].dim() == 4
        pred_cls_list = [
            _.permute(0, 2, 3, 1).reshape(pred_cls[0].shape[0], -1, config.num_classes-1)
            for _ in pred_cls]
        pred_reg_list = [
            _.permute(0, 2, 3, 1).reshape(pred_reg[0].shape[0], -1, 8)
            for _ in pred_reg]
        pred_prior_mean_list = [
            _.permute(0,2,3,1).reshape(pred_reg[0].shape[0], -1, 4) 
            for _ in pred_prior_mean]
        pred_prior_lstd_list = [
            _.permute(0,2,3,1).reshape(pred_reg[0].shape[0], -1, 4) 
            for _ in pred_prior_lstd]
        pred_logits_list = [
            _.permute(0,2,3,1).reshape(pred_reg[0].shape[0], -1, 2) 
            for _ in pred_logits]
        return pred_cls_list, pred_reg_list, pred_prior_mean_list, pred_prior_lstd_list, pred_logits_list

def per_layer_inference(anchors_list, pred_cls_list, pred_reg_list, im_info):
    keep_anchors = []
    keep_cls = []
    keep_reg = []
    class_num = pred_cls_list[0].shape[-1]
    for l_id in range(len(anchors_list)):
        anchors = anchors_list[l_id].reshape(-1, 4)
        pred_cls = pred_cls_list[l_id][0].reshape(-1, class_num)
        pred_reg = pred_reg_list[l_id][0].reshape(-1, 8)[:, :4]
        if len(anchors) > config.test_layer_topk:
            ruler = pred_cls.max(axis=1)[0]
            _, inds = ruler.topk(config.test_layer_topk, dim=0)
            inds = inds.flatten()
            keep_anchors.append(anchors[inds])
            keep_cls.append(torch.sigmoid(pred_cls[inds]))
            keep_reg.append(pred_reg[inds])
        else:
            keep_anchors.append(anchors)
            keep_cls.append(torch.sigmoid(pred_cls))
            keep_reg.append(pred_reg)
    keep_anchors = torch.cat(keep_anchors, axis = 0)
    keep_cls = torch.cat(keep_cls, axis = 0)
    keep_reg = torch.cat(keep_reg, axis = 0)
    # multiclass
    tag = torch.arange(class_num).type_as(keep_cls)+1
    tag = tag.repeat(keep_cls.shape[0], 1).reshape(-1,1)
    pred_scores = keep_cls.reshape(-1, 1)
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

