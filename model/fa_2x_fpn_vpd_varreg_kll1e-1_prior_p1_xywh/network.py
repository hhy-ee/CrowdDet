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
from det_oprs.my_loss_opr import freeanchor_loss
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
        pred_cls_list, pred_reg_list, pred_refined_reg_list = self.R_Head(fpn_fms)
        # release the useless data
        if self.training:
            loss_dict = self.R_Criteria(
                pred_cls_list, pred_reg_list, pred_refined_reg_list, anchors_list, gt_boxes, im_info)
            return loss_dict
        else:
            #pred_bbox = union_inference(
            #        anchors_list, pred_cls_list, pred_reg_list, im_info)
            pred_bbox = per_layer_inference(
                    anchors_list, pred_cls_list, pred_reg_list, pred_refined_reg_list, im_info)
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

    def __call__(self, pred_cls_list, pred_reg_list, pred_refined_reg_list, anchors_list, gt_boxes, im_info):
        all_anchors = torch.cat(anchors_list, axis=0)
        all_pred_cls = torch.cat(pred_cls_list, axis=1).reshape(-1, config.num_classes-1)
        all_pred_cls = torch.sigmoid(all_pred_cls)
        all_pred_reg = torch.cat(pred_reg_list, axis=1).reshape(-1, 8)
        all_refined_reg_list = torch.cat(pred_refined_reg_list, axis=1).reshape(-1, 4)
        # variational inference
        all_pred_mean = all_pred_reg[:, :config.num_cell_anchors * 4]
        all_pred_lstd = all_pred_reg[:, config.num_cell_anchors * 4:]
        scale = torch.tensor(config.prior_std).type_as(all_pred_lstd)
        pred_scale_std = all_pred_lstd.exp().mul(scale)
        all_pred_reg = all_pred_mean + pred_scale_std * torch.randn_like(all_pred_lstd)

        # get ground truth
        loss_dict = freeanchor_loss(all_anchors, all_pred_cls, all_pred_reg, gt_boxes, im_info)
        refined_loss_dict = freeanchor_loss(all_anchors, all_pred_cls, all_refined_reg_list, gt_boxes, im_info)
        loss_kld = kldiv_loss(
                all_pred_mean,
                all_pred_lstd,
                config.kl_weight)
        
        loss_dict['freeanchor_kldiv_loss'] = loss_kld
        loss_dict['refined_positive_bag_loss'] = refined_loss_dict['positive_bag_loss']
        loss_dict['refined_negative_bag_loss'] = refined_loss_dict['negative_bag_loss']
        
        return loss_dict

class RetinaNet_Head(nn.Module):
    def __init__(self):
        super().__init__()
        num_convs = 4
        in_channels = 256
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
        # predictor
        self.cls_score = nn.Conv2d(
            in_channels, config.num_cell_anchors * (config.num_classes-1),
            kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, config.num_cell_anchors * 8,
            kernel_size=3, stride=1, padding=1)
        
        # refined reg predict
        self.refined_reg_pred = nn.Sequential(
            nn.Conv2d(8, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, config.num_cell_anchors * 4, 
                kernel_size=3, stride=1, padding=1),
        )

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, 
                        self.bbox_pred, self.refined_reg_pred]:
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
        for feature in features:
            pred_cls.append(self.cls_score(self.cls_subnet(feature)))
            pred_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        # refined reg prediction
        pred_refined_reg = []
        for (cls, reg) in zip(pred_cls, pred_reg):
            in_reg = reg
            pred_refined_reg.append(self.refined_reg_pred(in_reg))

        # reshape the predictions
        assert pred_cls[0].dim() == 4
        pred_cls_list = [
            _.permute(0, 2, 3, 1).reshape(pred_cls[0].shape[0], -1, config.num_classes-1)
            for _ in pred_cls]
        pred_reg_list = [
            _.permute(0, 2, 3, 1).reshape(pred_reg[0].shape[0], -1, 8)
            for _ in pred_reg]
        pred_refined_reg_list = [
            _.permute(0, 2, 3, 1).reshape(pred_reg[0].shape[0], -1, 4)
            for _ in pred_refined_reg]
        return pred_cls_list, pred_reg_list, pred_refined_reg_list

def per_layer_inference(anchors_list, pred_cls_list, pred_reg_list, pred_refined_reg_list, im_info):
    keep_anchors = []
    keep_cls = []
    keep_reg = []
    keep_lstd = []
    class_num = pred_cls_list[0].shape[-1]
    for l_id in range(len(anchors_list)):
        anchors = anchors_list[l_id].reshape(-1, 4)
        pred_cls = pred_cls_list[l_id][0].reshape(-1, class_num)
        pred_reg = pred_reg_list[l_id][0].reshape(-1, 8)[:, :4]
        pred_lstd = pred_reg_list[l_id][0].reshape(-1, 8)[:, 4:]
        if len(anchors) > config.test_layer_topk:
            ruler = pred_cls.max(axis=1)[0]
            _, inds = ruler.topk(config.test_layer_topk, dim=0)
            inds = inds.flatten()
            keep_anchors.append(anchors[inds])
            keep_cls.append(torch.sigmoid(pred_cls[inds]))
            keep_reg.append(pred_reg[inds])
            keep_lstd.append(pred_lstd[inds])
        else:
            keep_anchors.append(anchors)
            keep_cls.append(torch.sigmoid(pred_cls))
            keep_reg.append(pred_reg)
            keep_lstd.append(pred_lstd)
    keep_anchors = torch.cat(keep_anchors, axis = 0)
    keep_cls = torch.cat(keep_cls, axis = 0)
    keep_reg = torch.cat(keep_reg, axis = 0)
    keep_lstd = torch.cat(keep_lstd, axis = 0)
    # multiclass
    tag = torch.arange(class_num).type_as(keep_cls)+1
    tag = tag.repeat(keep_cls.shape[0], 1).reshape(-1,1)
    pred_scores = keep_cls.reshape(-1, 1)
    if config.add_test_noise:
        keep_reg = keep_reg + 0.05 * torch.randn_like(keep_reg)
    pred_bbox = restore_bbox(keep_anchors, keep_reg, False)
    pred_bbox = pred_bbox.repeat(1, class_num).reshape(-1, 4)
    if config.save_data or config.test_nms_method == 'kl_nms':
        pred_bbox = torch.cat([pred_bbox, pred_scores, tag, keep_lstd], axis=1)
    else:
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