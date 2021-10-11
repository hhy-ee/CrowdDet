import torch
import torch.nn.functional as F

from config import config
from det_oprs.bbox_opr import bbox_transform_inv_opr, clip_boxes_opr, \
    filter_boxes_opr
from torchvision.ops import nms

@torch.no_grad()
def find_top_rpn_proposals(is_train, rpn_bbox_offsets_list, rpn_cls_prob_list,
        all_anchors_list, im_info):
    prev_nms_top_n = config.train_prev_nms_top_n \
        if is_train else config.test_prev_nms_top_n
    post_nms_top_n = config.train_post_nms_top_n \
        if is_train else config.test_post_nms_top_n
    batch_per_gpu = config.train_batch_per_gpu if is_train else 1
    nms_threshold = config.rpn_nms_threshold
    box_min_size = config.rpn_min_box_size
    bbox_normalize_targets = config.rpn_bbox_normalize_targets
    bbox_normalize_means = config.bbox_normalize_means
    bbox_normalize_stds = config.bbox_normalize_stds
    list_size = len(rpn_bbox_offsets_list)

    return_rois = []
    return_inds = []
    for bid in range(batch_per_gpu):
        batch_proposals_list = []
        batch_probs_list = []
        for l in range(list_size):
            # get proposals and probs
            offsets = rpn_bbox_offsets_list[l][bid] \
                .permute(1, 2, 0).reshape(-1, 4)
            if bbox_normalize_targets:
                std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(bbox_targets)
                mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(bbox_targets)
                pred_offsets = pred_offsets * std_opr
                pred_offsets = pred_offsets + mean_opr
            all_anchors = all_anchors_list[l]
            proposals = bbox_transform_inv_opr(all_anchors, offsets)
            if config.anchor_within_border:
                proposals = clip_boxes_opr(proposals, im_info[bid, :])
            probs = rpn_cls_prob_list[l][bid] \
                    .permute(1,2,0).reshape(-1, 2)
            probs = torch.softmax(probs, dim=-1)[:, 1]
            # gather the proposals and probs
            batch_proposals_list.append(proposals)
            batch_probs_list.append(probs)
        batch_proposals = torch.cat(batch_proposals_list, dim=0)
        batch_probs = torch.cat(batch_probs_list, dim=0)
        # filter the zero boxes.
        batch_keep_mask = filter_boxes_opr(
                batch_proposals, box_min_size * im_info[bid, 2])
        batch_proposals = batch_proposals[batch_keep_mask]
        batch_probs = batch_probs[batch_keep_mask]
        # prev_nms_top_n
        num_proposals = min(prev_nms_top_n, batch_probs.shape[0])
        batch_probs, idx = batch_probs.sort(descending=True)
        batch_probs = batch_probs[:num_proposals]
        topk_idx = idx[:num_proposals].flatten()
        batch_proposals = batch_proposals[topk_idx]
        # For each image, run a total-level NMS, and choose topk results.
        keep = nms(batch_proposals, batch_probs, nms_threshold)
        keep = keep[:post_nms_top_n]
        batch_proposals = batch_proposals[keep]
        #batch_probs = batch_probs[keep]
        # cons the rois
        batch_inds = torch.ones(batch_proposals.shape[0], 1).type_as(batch_proposals) * bid
        batch_rois = torch.cat([batch_inds, batch_proposals], axis=1)
        return_rois.append(batch_rois)

    if batch_per_gpu == 1:
        return batch_rois
    else:
        concated_rois = torch.cat(return_rois, axis=0)
        return concated_rois

@torch.no_grad()
def find_test_top_rpn_proposals(is_train, rpn_bbox_offsets_list, rpn_cls_prob_list,
        rpn_bbox_lstd_list, all_anchors_list, im_info):
    prev_nms_top_n = config.train_prev_nms_top_n \
        if is_train else config.test_prev_nms_top_n
    post_nms_top_n = config.train_post_nms_top_n \
        if is_train else config.test_post_nms_top_n
    batch_per_gpu = config.train_batch_per_gpu if is_train else 1
    nms_threshold = config.rpn_nms_threshold
    box_min_size = config.rpn_min_box_size
    bbox_normalize_targets = config.rpn_bbox_normalize_targets
    bbox_normalize_means = config.bbox_normalize_means
    bbox_normalize_stds = config.bbox_normalize_stds
    list_size = len(rpn_bbox_offsets_list)

    return_rois = []
    return_anchors = []
    return_lstds = []
    return_inds = []
    for bid in range(batch_per_gpu):
        batch_proposals_list = []
        batch_anchors_list = []
        batch_lstds_list = []
        batch_probs_list = []
        for l in range(list_size):
            # get proposals and probs
            offsets = rpn_bbox_offsets_list[l][bid] \
                .permute(1, 2, 0).reshape(-1, 4)
            lstds = rpn_bbox_lstd_list[l][bid] \
                .permute(1, 2, 0).reshape(-1, 2)
            if bbox_normalize_targets:
                std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(bbox_targets)
                mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(bbox_targets)
                pred_offsets = pred_offsets * std_opr
                pred_offsets = pred_offsets + mean_opr
            all_anchors = all_anchors_list[l]
            proposals = bbox_transform_inv_opr(all_anchors, offsets)
            if config.anchor_within_border:
                proposals = clip_boxes_opr(proposals, im_info[bid, :])
            probs = rpn_cls_prob_list[l][bid] \
                    .permute(1,2,0).reshape(-1, 2)
            probs = torch.softmax(probs, dim=-1)[:, 1]
            # gather the proposals and probs
            batch_proposals_list.append(proposals)
            batch_anchors_list.append(all_anchors)
            batch_lstds_list.append(lstds)
            batch_probs_list.append(probs)
        batch_proposals = torch.cat(batch_proposals_list, dim=0)
        batch_anchors = torch.cat(batch_anchors_list, dim=0)
        batch_lstds = torch.cat(batch_lstds_list, dim=0)
        batch_probs = torch.cat(batch_probs_list, dim=0)
        # filter the zero boxes.
        batch_keep_mask = filter_boxes_opr(
                batch_proposals, box_min_size * im_info[bid, 2])
        batch_proposals = batch_proposals[batch_keep_mask]
        batch_anchors = batch_anchors[batch_keep_mask]
        batch_lstds = batch_lstds[batch_keep_mask]
        batch_probs = batch_probs[batch_keep_mask]
        # prev_nms_top_n
        num_proposals = min(prev_nms_top_n, batch_probs.shape[0])
        batch_probs, idx = batch_probs.sort(descending=True)
        batch_probs = batch_probs[:num_proposals]
        topk_idx = idx[:num_proposals].flatten()
        batch_proposals = batch_proposals[topk_idx]
        batch_anchors = batch_anchors[topk_idx]
        batch_lstds = batch_lstds[topk_idx]
        # For each image, run a total-level NMS, and choose topk results.
        keep = nms(batch_proposals, batch_probs, nms_threshold)
        keep = keep[:post_nms_top_n]
        batch_proposals = batch_proposals[keep]
        batch_anchors = batch_anchors[keep]
        batch_lstds = batch_lstds[keep]
        #batch_probs = batch_probs[keep]
        # cons the rois
        batch_inds = torch.ones(batch_proposals.shape[0], 1).type_as(batch_proposals) * bid
        batch_rois = torch.cat([batch_inds, batch_proposals], axis=1)
        return_rois.append(batch_rois)
        return_anchors.append(batch_anchors)
        return_lstds.append(batch_lstds)

    if batch_per_gpu == 1:
        return (batch_rois, batch_lstds, batch_anchors)
    else:
        concated_rois = torch.cat(return_rois, axis=0)
        concated_anchors = torch.cat(return_anchors, axis=0)
        concated_lstds = torch.cat(return_lstds, axis=0)
        return (concated_rois, concated_lstds, concated_anchors)

def find_top_rpn_gmm_box(pred_bbox_list, pred_cls_score_list, logit_list):
    pred_bbox_list = [pred_bbox.unsqueeze(0) for pred_bbox in pred_bbox_list]
    pred_cls_score_list = [pred_cls_score.unsqueeze(0) for pred_cls_score in pred_cls_score_list]
    final_pred_bbox_list = []
    final_pred_cls_score_list = []
    top_index = [F.softmax(qy_logit, dim=1).topk(dim=1, k=config.n_components).indices for qy_logit in logit_list]
    for l in range(len(top_index)):
        pred_bbox_perlvl_list = pred_bbox_list[l:len(pred_bbox_list):len(top_index)]
        pred_bbox_perlvl = torch.cat(pred_bbox_perlvl_list, dim=0)
        pred_cls_score_perlvl_list = pred_cls_score_list[l:len(pred_bbox_list):len(top_index)]
        pred_cls_score_perlvl = torch.cat(pred_cls_score_perlvl_list, dim=0)
        top_index_perlvl = top_index[l].permute(1, 0, 2, 3).unsqueeze(2).repeat(1, 1, 4, 1, 1)
        final_pred_bbox_list.append(torch.gather(pred_bbox_perlvl, 0, top_index_perlvl)[0, :])
        top_index_perlvl = top_index[l].permute(1, 0, 2, 3).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        final_pred_cls_score_list.append(torch.gather(pred_cls_score_perlvl, 0, top_index_perlvl)[0, :])
    return final_pred_bbox_list, final_pred_cls_score_list

@torch.no_grad()
def find_top_rpn_proposals_va(is_train, rpn_bbox_offsets_list, rpn_cls_prob_list,
         rpn_bbox_dist_list, all_anchors_list, im_info):
    prev_nms_top_n = config.train_prev_nms_top_n \
        if is_train else config.test_prev_nms_top_n
    post_nms_top_n = config.train_post_nms_top_n \
        if is_train else config.test_post_nms_top_n
    batch_per_gpu = config.train_batch_per_gpu if is_train else 1
    nms_threshold = config.rpn_nms_threshold
    box_min_size = config.rpn_min_box_size
    bbox_normalize_targets = config.rpn_bbox_normalize_targets
    bbox_normalize_means = config.bbox_normalize_means
    bbox_normalize_stds = config.bbox_normalize_stds
    list_size = len(rpn_bbox_offsets_list)

    return_rois = []
    return_dists = []
    return_inds = []
    for bid in range(batch_per_gpu):
        batch_proposals_list = []
        batch_dists_list = []
        batch_probs_list = []
        for l in range(list_size):
            # get proposals and probs
            offsets = rpn_bbox_offsets_list[l][bid] \
                .permute(1, 2, 0).reshape(-1, 4)
            dists = rpn_bbox_dist_list[l][bid] \
                .permute(1, 2, 0).reshape(-1, 4)
            if bbox_normalize_targets:
                std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(bbox_targets)
                mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(bbox_targets)
                pred_offsets = pred_offsets * std_opr
                pred_offsets = pred_offsets + mean_opr
            all_anchors = all_anchors_list[l]
            proposals = bbox_transform_inv_opr(all_anchors, offsets)
            if config.anchor_within_border:
                proposals = clip_boxes_opr(proposals, im_info[bid, :])
            probs = rpn_cls_prob_list[l][bid] \
                    .permute(1,2,0).reshape(-1, 2)
            probs = torch.softmax(probs, dim=-1)[:, 1]
            # gather the proposals and probs
            batch_proposals_list.append(proposals)
            batch_dists_list.append(dists)
            batch_probs_list.append(probs)
        batch_proposals = torch.cat(batch_proposals_list, dim=0)
        batch_dists = torch.cat(batch_dists_list, dim=0)
        batch_probs = torch.cat(batch_probs_list, dim=0)
        # filter the zero boxes.
        batch_keep_mask = filter_boxes_opr(
                batch_proposals, box_min_size * im_info[bid, 2])
        batch_proposals = batch_proposals[batch_keep_mask]
        batch_dists = batch_dists[batch_keep_mask]
        batch_probs = batch_probs[batch_keep_mask]
        # prev_nms_top_n
        num_proposals = min(prev_nms_top_n, batch_probs.shape[0])
        batch_probs, idx = batch_probs.sort(descending=True)
        batch_probs = batch_probs[:num_proposals]
        topk_idx = idx[:num_proposals].flatten()
        batch_proposals = batch_proposals[topk_idx]
        batch_dists = batch_dists[topk_idx]
        # For each image, run a total-level NMS, and choose topk results.
        keep = nms(batch_proposals, batch_probs, nms_threshold)
        keep = keep[:post_nms_top_n]
        batch_proposals = batch_proposals[keep]
        batch_dists = batch_dists[keep]
        #batch_probs = batch_probs[keep]
        # cons the rois
        batch_inds = torch.ones(batch_proposals.shape[0], 1).type_as(batch_proposals) * bid
        batch_rois = torch.cat([batch_inds, batch_proposals], axis=1)
        batch_dists = torch.cat([batch_inds, batch_dists], axis=1)
        return_rois.append(batch_rois)
        return_dists.append(batch_dists)

    if batch_per_gpu == 1:
        return batch_rois, batch_dists
    else:
        concated_rois = torch.cat(return_rois, axis=0)
        concated_dists = torch.cat(return_dists, axis=0)
        return concated_rois, concated_dists