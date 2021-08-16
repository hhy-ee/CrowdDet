import torch

import numpy as np
from config import config
from det_oprs.bbox_opr import box_overlap_opr, bbox_transform_opr

@torch.no_grad()
def retina_anchor_target(anchors, gt_boxes, im_info, top_k=1):
    total_anchor = anchors.shape[0]
    return_labels = []
    return_bbox_targets = []
    # get per image proposals and gt_boxes
    for bid in range(config.train_batch_per_gpu):
        gt_boxes_perimg = gt_boxes[bid, :int(im_info[bid, 5]), :]
        anchors = anchors.type_as(gt_boxes_perimg)
        overlaps = box_overlap_opr(anchors, gt_boxes_perimg[:, :-1])
        # gt max and indices
        max_overlaps, gt_assignment = overlaps.topk(top_k, dim=1, sorted=True)
        max_overlaps= max_overlaps.flatten()
        gt_assignment= gt_assignment.flatten()
        _, gt_assignment_for_gt = torch.max(overlaps, axis=0)
        del overlaps
        # cons labels
        labels = gt_boxes_perimg[gt_assignment, 4]
        labels = labels * (max_overlaps >= config.negative_thresh)
        ignore_mask = (max_overlaps < config.positive_thresh) * (
                max_overlaps >= config.negative_thresh)
        labels[ignore_mask] = -1
        # cons bbox targets
        target_boxes = gt_boxes_perimg[gt_assignment, :4]
        target_anchors = anchors.repeat(1, top_k).reshape(-1, anchors.shape[-1])
        bbox_targets = bbox_transform_opr(target_anchors, target_boxes)
        if config.allow_low_quality:
            labels[gt_assignment_for_gt] = gt_boxes_perimg[:, 4]
            low_quality_bbox_targets = bbox_transform_opr(
                anchors[gt_assignment_for_gt], gt_boxes_perimg[:, :4])
            bbox_targets[gt_assignment_for_gt] = low_quality_bbox_targets
        labels = labels.reshape(-1, 1 * top_k)
        bbox_targets = bbox_targets.reshape(-1, 4 * top_k)
        return_labels.append(labels)
        return_bbox_targets.append(bbox_targets)

    if config.train_batch_per_gpu == 1:
        return labels, bbox_targets
    else:
        return_labels = torch.cat(return_labels, axis=0)
        return_bbox_targets = torch.cat(return_bbox_targets, axis=0)
        return return_labels, return_bbox_targets

@torch.no_grad()
def retina_anchor_target_new(anchors, gt_boxes, im_info, top_k=1):
    total_anchor = anchors.shape[0]
    return_labels = []
    return_bbox_targets = []
    return_pull_loss_labels = []
    return_push_loss_labels = []
    # get per image proposals and gt_boxes
    for bid in range(config.train_batch_per_gpu):
        gt_boxes_perimg = gt_boxes[bid, :int(im_info[bid, 5]), :]
        anchors = anchors.type_as(gt_boxes_perimg)
        overlaps = box_overlap_opr(anchors, gt_boxes_perimg[:, :-1])
        # gt max and indices
        max_overlaps, gt_assignment = overlaps.topk(top_k, dim=1, sorted=True)
        max_overlaps= max_overlaps.flatten()
        gt_assignment= gt_assignment.flatten()
        gt_assignment_for_gt = torch.zeros(overlaps.shape[1]).type_as(gt_assignment)
        max_iou_gt = torch.max(overlaps, axis=0).values.sort(descending=True).indices
        for i in range(overlaps.shape[1]):
            gt_assignment_for_gt[max_iou_gt[i]] = torch.argmax(overlaps[:, max_iou_gt[i]])
            overlaps[gt_assignment_for_gt[max_iou_gt[i]], :] = 0
        del overlaps
        # cons labels
        labels = gt_boxes_perimg[gt_assignment, 4]
        labels = labels * (max_overlaps >= config.negative_thresh)
        ignore_mask = (max_overlaps < config.positive_thresh) * (
                max_overlaps >= config.negative_thresh)
        labels[ignore_mask] = -1
        # cons bbox targets
        target_boxes = gt_boxes_perimg[gt_assignment, :4]
        target_anchors = anchors.repeat(1, top_k).reshape(-1, anchors.shape[-1])
        bbox_targets = bbox_transform_opr(target_anchors, target_boxes)
        if config.allow_low_quality:
            labels[gt_assignment_for_gt] = gt_boxes_perimg[:, 4]
            low_quality_bbox_targets = bbox_transform_opr(
                anchors[gt_assignment_for_gt], gt_boxes_perimg[:, :4])
            bbox_targets[gt_assignment_for_gt] = low_quality_bbox_targets
        labels = labels.reshape(-1, 1 * top_k)
        # pull loss
        gt_assignment[gt_assignment_for_gt] = torch.arange(gt_boxes_perimg.shape[0]).type_as(gt_assignment)
        new_gt_assignment = torch.zeros_like(gt_assignment, dtype=torch.int64)
        new_gt_assignment[torch.where(labels==1)[0]] = gt_assignment[torch.where(labels==1)[0]] + 1
        pull_loss_labels = []
        for gt in torch.where(gt_boxes_perimg[:, 4]==1)[0]:
            if len(torch.where(new_gt_assignment == gt + 1)[0]) > 1:
                pull_loss_labels.append(torch.where(new_gt_assignment == gt+1)[0])
        # push loss
        gt_overlaps = box_overlap_opr(gt_boxes_perimg[:,:4], gt_boxes_perimg[:,:4])
        gt_ignore_mask = gt_boxes_perimg[:, 4].eq(-1).repeat(gt_boxes_perimg.shape[0], 1)
        gt_ignore_mask = (~gt_ignore_mask).mul((~gt_ignore_mask).permute(1,0))
        gt_tril_mask = torch.tril(torch.ones_like(gt_overlaps)).eq(0)
        gt_overlaps = gt_overlaps * gt_ignore_mask * gt_tril_mask
        overlap_gt = torch.where(gt_overlaps > 0.5)
        push_loss_labels = []
        if overlap_gt[0].shape[0] > 0:
            for i in range(overlap_gt[0].shape[0]):
                pos_push_index = torch.where(new_gt_assignment==overlap_gt[0][i] + 1)[0]
                neg_push_index = torch.where(new_gt_assignment==overlap_gt[1][i] + 1)[0]
                push_loss_labels.append([pos_push_index, neg_push_index])

        bbox_targets = bbox_targets.reshape(-1, 4 * top_k)
        return_labels.append(labels)
        return_bbox_targets.append(bbox_targets)
        return_pull_loss_labels.append(pull_loss_labels)
        return_push_loss_labels.append(push_loss_labels)
        
    if config.train_batch_per_gpu == 1:
        return labels, bbox_targets
    else:
        return_labels = torch.cat(return_labels, axis=0)
        return_bbox_targets = torch.cat(return_bbox_targets, axis=0)
        return return_labels, return_bbox_targets, return_pull_loss_labels, return_push_loss_labels

@torch.no_grad()
def retina_avpd_anchor_target(avpd_anchors, anchors, gt_boxes, im_info, top_k=1):
    total_anchor = avpd_anchors.shape[0]
    return_labels = []
    return_bbox_targets = []
    avpd_anchors = avpd_anchors.reshape(config.train_batch_per_gpu, -1, 4)
    anchors = anchors.reshape(config.train_batch_per_gpu, -1, 4)
    # get per image proposals and gt_boxes
    for bid in range(config.train_batch_per_gpu):
        gt_boxes_perimg = gt_boxes[bid, :int(im_info[bid, 5]), :]
        batch_anchors = anchors[bid, :].type_as(gt_boxes_perimg)
        batch_avpd_anchors = avpd_anchors[bid, :].type_as(gt_boxes_perimg)
        overlaps = box_overlap_opr(batch_avpd_anchors, gt_boxes_perimg[:, :-1])
        # gt max and indices
        max_overlaps, gt_assignment = overlaps.topk(top_k, dim=1, sorted=True)
        max_overlaps= max_overlaps.flatten()
        gt_assignment= gt_assignment.flatten()
        _, gt_assignment_for_gt = torch.max(overlaps, axis=0)
        del overlaps
        # cons labels
        labels = gt_boxes_perimg[gt_assignment, 4]
        labels = labels * (max_overlaps >= config.negative_thresh)
        ignore_mask = (max_overlaps < config.positive_thresh) * (
                max_overlaps >= config.negative_thresh)
        labels[ignore_mask] = -1
        # cons bbox targets
        target_boxes = gt_boxes_perimg[gt_assignment, :4]
        target_anchors = batch_anchors.repeat(1, top_k).reshape(-1, anchors.shape[-1])
        bbox_targets = bbox_transform_opr(target_anchors, target_boxes)
        if config.allow_low_quality:
            labels[gt_assignment_for_gt] = gt_boxes_perimg[:, 4]
            low_quality_bbox_targets = bbox_transform_opr(
                batch_anchors[gt_assignment_for_gt], gt_boxes_perimg[:, :4])
            bbox_targets[gt_assignment_for_gt] = low_quality_bbox_targets
        labels = labels.reshape(-1, 1 * top_k)
        bbox_targets = bbox_targets.reshape(-1, 4 * top_k)
        return_labels.append(labels)
        return_bbox_targets.append(bbox_targets)

    if config.train_batch_per_gpu == 1:
        return labels, bbox_targets
    else:
        return_labels = torch.cat(return_labels, axis=0)
        return_bbox_targets = torch.cat(return_bbox_targets, axis=0)
        return return_labels, return_bbox_targets