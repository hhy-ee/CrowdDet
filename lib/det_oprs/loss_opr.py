import torch
import numpy as np
import torch.nn.functional as F
from det_oprs.bbox_opr import bbox_transform_inv_opr
from det_oprs.bbox_opr import box_overlap_opr, align_box_giou_opr
from config import config
from utils import cal_utils

INF = 100000000
EPS = 1e-6

def softmax_loss(score, label, ignore_label=-1):
    with torch.no_grad():
        max_score = score.max(axis=1, keepdims=True)[0]
    score = score - max_score
    log_prob = score - torch.log(torch.exp(score).sum(axis=1, keepdims=True))
    mask = label != ignore_label
    vlabel = label * mask
    onehot = torch.zeros(vlabel.shape[0], config.num_classes, device=score.device)
    onehot.scatter_(1, vlabel.reshape(-1, 1), 1)
    loss = -(log_prob * onehot).sum(axis=1)
    loss = loss * mask
    return loss

def refined_softmax_loss(score, var_score, weight, label, ignore_label=-1):
    with torch.no_grad():
        max_score = score.max(axis=1, keepdims=True)[0]
        max_var_score = var_score.max(axis=1, keepdims=True)[0]
    score -= max_score
    var_score -= max_var_score

    log_prob = score - torch.log(torch.exp(score).sum(axis=1, keepdims=True))
    var_log_prob = var_score - torch.log(torch.exp(var_score).sum(axis=1, keepdims=True))
    weight = torch.sigmoid(weight)[:,1:]

    refined_prob = weight * log_prob.exp()[:,1:] + (1-weight) * var_log_prob.exp()[:,1:]
    refined_log_prob = torch.log(torch.cat([1-refined_prob, refined_prob], dim=1))
    # one-hot label
    mask = label != ignore_label
    vlabel = label * mask
    onehot = torch.zeros(vlabel.shape[0], config.num_classes, device=score.device)
    onehot.scatter_(1, vlabel.reshape(-1, 1), 1)
    # origin loss
    loss = -(log_prob * onehot).sum(axis=1)
    loss = loss * mask
    # refined loss
    refined_loss = -(refined_log_prob * onehot).sum(axis=1)
    refined_loss = refined_loss * mask
    return loss, refined_loss 

def refined_var_softmax_loss(score, var_score, weight, label, ignore_label=-1):
    with torch.no_grad():
        max_score = score.max(axis=1, keepdims=True)[0]
        max_var_score = var_score.max(axis=1, keepdims=True)[0]
    score -= max_score
    var_score -= max_var_score

    log_prob = score - torch.log(torch.exp(score).sum(axis=1, keepdims=True))
    var_log_prob = var_score - torch.log(torch.exp(var_score).sum(axis=1, keepdims=True))
    weight = torch.sigmoid(weight)[:,1:]

    refined_prob = weight * log_prob.exp()[:,1:] + (1-weight) * var_log_prob.exp()[:,1:]
    refined_log_prob = torch.log(torch.cat([1-refined_prob, refined_prob], dim=1))
    # one-hot label
    mask = label != ignore_label
    vlabel = label * mask
    onehot = torch.zeros(vlabel.shape[0], config.num_classes, device=score.device)
    onehot.scatter_(1, vlabel.reshape(-1, 1), 1)
    # origin loss
    loss = -(log_prob * onehot).sum(axis=1)
    loss = loss * mask
    # var loss
    var_loss = -(var_log_prob * onehot).sum(axis=1)
    var_loss = var_loss * mask
    # refined loss
    refined_loss = -(refined_log_prob * onehot).sum(axis=1)
    refined_loss = refined_loss * mask
    return loss, var_loss, refined_loss 

def smooth_l1_loss(pred, target, beta: float):
    if beta < 1e-5:
        loss = torch.abs(input - target)
    else:
        abs_x = torch.abs(pred- target)
        in_mask = abs_x < beta
        loss = torch.where(in_mask, 0.5 * abs_x ** 2 / beta, abs_x - 0.5 * beta)
    return loss.sum(axis=1)

def giou_loss(pred, target, anchor):
    pred_boxes = bbox_transform_inv_opr(anchor, pred)
    target_boxes = bbox_transform_inv_opr(anchor, target)
    gious = align_box_giou_opr(pred_boxes, target_boxes)
    loss = 1 - gious
    return loss

def dfl_xywh_loss(pred, target, loss_weight):
    scale = (config.project.shape[1] - 1) / 2 / config.project[0,-1]
    pred = pred.reshape(-1, pred.shape[-1])
    target = target.reshape(-1) * scale + scale
    target = target.clamp(min=EPS, max=2*scale-EPS)
    dis_left = target.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - target
    weight_right = target - dis_left.float()
    loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left \
        + F.cross_entropy(pred, dis_right, reduction='none') * weight_right
    return loss.reshape(-1, 4).sum(dim=1) * loss_weight

def kl_kdn_loss(pred, target, loss_weight):
    scale = (config.project.shape[1] - 1) / 2 / config.project[0,-1]
    pred = pred.reshape(-1, pred.shape[-1])
    target = (target.reshape(-1) + config.project[0,-1]) * scale
    target = target.clamp(min=EPS, max=2*config.project[0,-1]*scale-EPS)
    dis_left = target.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - target
    weight_right = target - dis_left.float()
    pred = F.softmax(pred, dim=1)
    pred_weight_left = (pred, 1, dis_left.reshape(-1,1)).reshape(-1)
    pred_weight_right = torch.gather(pred, 1, dis_right.reshape(-1,1)).reshape(-1)
    loss = weight_left * torch.log((weight_left + EPS) / (pred_weight_left + EPS)) + \
        weight_right * torch.log((weight_right + EPS) / (pred_weight_right + EPS))
    return loss.reshape(-1, 4).sum(dim=1) * loss_weight

def kl_kdn_loss_complete(pred, target, loss_weight):
    scale = (config.project.shape[1] - 1) / 2 / config.project[0,-1]
    pred = pred.reshape(-1, pred.shape[-1])
    target = (target.reshape(-1) + config.project[0,-1]) * scale
    target = target.clamp(min=EPS, max=2 * config.project[0,-1] * scale-EPS)
    dis_left = target.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - target
    weight_right = target - dis_left.float()
    pred_weight_left = torch.gather(pred, 1, dis_left.reshape(-1,1)).reshape(-1)
    pred_weight_right = torch.gather(pred, 1, dis_right.reshape(-1,1)).reshape(-1)
    loss = weight_left * torch.log((weight_left + EPS) / (pred_weight_left + EPS)) + \
        weight_right * torch.log((weight_right + EPS) / (pred_weight_right + EPS))
    return loss.reshape(-1, 4).sum(dim=1) * loss_weight

def kl_gaussian_loss(dist, target, loss_weight):
    scale = (config.project.shape[1] - 1) / 2 / config.project[0,-1]
    acc = 1 / scale / 2
    target = (target.reshape(-1) + config.project[0,-1]) * scale
    target = target.clamp(min=EPS, max=2 * config.project[0,-1] * scale-EPS)
    idx_left = target.long()
    idx_right = idx_left + 1
    weight_left = idx_right.float() - target
    weight_right = target - idx_left.float()
    # GMM discreting
    mean= dist[:, :4].reshape(-1, 1)
    lstd= dist[:, 4:].reshape(-1, 1)
    Qg = torch.distributions.normal.Normal(mean, lstd.exp())
    project = torch.tensor(config.project).type_as(mean).repeat(mean.shape[0],1)
    dis_left = torch.gather(project, 1, idx_left.reshape(-1, 1))
    dis_right = torch.gather(project, 1, idx_right.reshape(-1, 1))
    pred_weight_left = Qg.cdf(dis_left + acc) - Qg.cdf(dis_left - acc)
    pred_weight_right = Qg.cdf(dis_right + acc) - Qg.cdf(dis_right - acc)
    loss = weight_left * torch.log((weight_left + EPS) / (pred_weight_left.reshape(-1) + EPS)) + \
        weight_right * torch.log((weight_right + EPS) / (pred_weight_right.reshape(-1) + EPS))
    return loss.reshape(-1, 4).mean(dim=1) * loss_weight

def kl_gmm_loss(prob, mean, lstd, target, loss_weight):
    scale = (config.project.shape[1] - 1) / 2 / config.project[0,-1]
    acc = 1 / scale / 2
    target = (target.reshape(-1) + config.project[0,-1]) * scale
    target = target.clamp(min=EPS, max=2 * config.project[0,-1] * scale-EPS)
    idx_left = target.long()
    idx_right = idx_left + 1
    weight_left = idx_right.float() - target
    weight_right = target - idx_left.float()
    # GMM discreting
    Qgmm = torch.distributions.normal.Normal(mean, lstd.exp())
    dis_left = torch.gather(mean, 1, idx_left.reshape(-1, 1))
    dis_right = torch.gather(mean, 1, idx_right.reshape(-1, 1))
    pred_weight_left = Qgmm.cdf(dis_left + acc).mul(prob).sum(1) - \
        Qgmm.cdf(dis_left - acc).mul(prob).sum(1)
    pred_weight_right = Qgmm.cdf(dis_right + acc).mul(prob).sum(1) - \
        Qgmm.cdf(dis_right - acc).mul(prob).sum(1)
    loss = weight_left * torch.log((weight_left + EPS) / (pred_weight_left + EPS)) + \
        weight_right * torch.log((weight_right + EPS) / (pred_weight_right + EPS))
    return loss.reshape(-1, 4).mean(dim=1) * loss_weight

def nflow_dist_loss(pred, target, loss_weight):
    # Discretize target
    target = target.reshape(-1, 1)
    target = target.clamp(min=-config.bound, max=config.bound)
    target = (target + config.bound) / config.acc
    dis_left_index = target.long()
    dis_right_index = dis_left_index + 1
    target_dis_left = dis_left_index * config.acc - config.bound
    target_dis_right = dis_right_index * config.acc - config.bound
    target_weight_left = dis_right_index.float() - target
    target_weight_right = target - dis_left_index.float()
    # Normalizing flow
    pred = pred.reshape(-1, pred.shape[2])
    mean, lstd = pred[..., 0].reshape(-1,1), pred[..., 1].reshape(-1,1)
    flow = pred[..., 2:].reshape(-1, config.nflow_layers, 3)
    nf_u, nf_w, nf_b = torch.split(flow, 1, dim=2)
    q0 = torch.distributions.normal.Normal(mean, lstd.exp())
    zk = torch.cat([target_dis_left, target_dis_right], dim=1)
    z0 = cal_utils.pf_inv_mapping(flow, zk, config.nflow_layers)
    z0 = torch.tensor(z0).type_as(zk)
    pred_log_pdf = q0.log_prob(z0)
    for l in range(config.nflow_layers):
        psi = (1 - torch.tanh(nf_w[:, l] * z0 + nf_b[:, l]).pow(2)) * nf_w[:, l]
        pred_log_pdf -= torch.log(torch.abs(1 + psi * nf_u[:, l]))
    pred_weight_left, pred_weight_right = torch.split(pred_log_pdf.exp() * config.acc, 1, dim=1)
    loss = target_weight_left * torch.log((target_weight_left + EPS) / (pred_weight_left + EPS)) + \
        target_weight_right * torch.log((target_weight_right + EPS) / (pred_weight_right + EPS))
    return loss.reshape(-1, 4).sum(dim=1) * loss_weight

def nflow_dist_loss1(pred, target, loss_weight):
    # Discretize target
    target = target.reshape(-1, 1)
    target = target.clamp(min=-config.target_bound+EPS, max=config.target_bound-EPS)
    target = (target + config.target_bound) / config.target_acc
    dis_left_index = target.long()
    dis_right_index = dis_left_index + 1
    target_dis_left = dis_left_index * config.target_acc - config.target_bound
    target_dis_right = dis_right_index * config.target_acc - config.target_bound
    target_weight_left = dis_right_index.float() - target
    target_weight_right = target - dis_left_index.float()
    # Normalizing flow
    pred = pred.reshape(-1, pred.shape[2])
    mean, lstd = torch.split(pred[:, :2], 1, dim=1)
    flow = pred[..., 2:].reshape(-1, config.nflow_layers, 3)
    nf_u, nf_w, nf_b = torch.split(flow, 1, dim=2)
    nf_u, nf_w = torch.exp(nf_u), torch.exp(nf_w)
    flow = torch.cat([nf_u, nf_w, nf_b], dim=2)
    q0 = torch.distributions.normal.Normal(mean, lstd.exp())
    zk = torch.cat([target_dis_left, target_dis_right], dim=1)
    z0 = cal_utils.pf_inv_mapping(flow, zk, config.nflow_layers)
    z0 = torch.tensor(z0).type_as(zk)
    pred_log_pdf = q0.log_prob(z0)
    for l in range(config.nflow_layers):
        psi = (1 - torch.tanh(nf_w[:, l] * z0 + nf_b[:, l]).pow(2)) * nf_w[:, l]
        pred_log_pdf = pred_log_pdf - torch.log(torch.abs(1 + psi * nf_u[:, l]))
    pred_weight_left, pred_weight_right = torch.split(pred_log_pdf.exp() * config.target_acc, 1, dim=1)
    loss = target_weight_left * torch.log((target_weight_left + EPS) / (pred_weight_left + EPS)) + \
        target_weight_right * torch.log((target_weight_right + EPS) / (pred_weight_right + EPS))
    return loss.reshape(-1, 4).sum(dim=1) * loss_weight

def nflow_dist_loss2(pred, flow1, target, loss_weight):
    # Discretize target
    target = target.reshape(-1, 1)
    target = target.clamp(min=-config.target_bound+EPS, max=config.target_bound-EPS)
    target = (target + config.target_bound) / config.target_acc
    dis_left_index = target.long()
    dis_right_index = dis_left_index + 1
    target_dis_left = dis_left_index * config.target_acc - config.target_bound
    target_dis_right = dis_right_index * config.target_acc - config.target_bound
    target_weight_left = dis_right_index.float() - target
    target_weight_right = target - dis_left_index.float()
    # Normalizing flow
    pred = pred.reshape(-1, pred.shape[2])
    mean, lstd = torch.split(pred[:, :2], 1, dim=1)
    flow = pred[..., 2:].reshape(-1, config.nflow_layers, 3)
    nf_u, nf_w, nf_b = torch.split(flow, 1, dim=2)
    nf_u = (torch.log(1 + torch.exp(nf_u * nf_w)) \
        - 1 - nf_u * nf_w) * (nf_w / nf_w.pow(2)) + nf_u
    flow = torch.cat([nf_u, nf_w, nf_b], dim=2)
    q0 = torch.distributions.normal.Normal(mean, lstd.exp())
    zk = torch.cat([target_dis_left, target_dis_right], dim=1)
    z0 = cal_utils.pf_inv_mapping(flow, zk, config.nflow_layers)
    z0 = torch.tensor(z0).type_as(zk)
    pred_log_pdf = q0.log_prob(z0)
    for l in range(config.nflow_layers):
        psi = (1 - torch.tanh(nf_w[:, l] * z0 + nf_b[:, l]).pow(2)) * nf_w[:, l]
        pred_log_pdf = pred_log_pdf - torch.log(torch.abs(1 + psi * nf_u[:, l]))
    pred_weight_left, pred_weight_right = torch.split(pred_log_pdf.exp() * config.target_acc, 1, dim=1)
    loss = target_weight_left * torch.log((target_weight_left + EPS) / (pred_weight_left + EPS)) + \
        target_weight_right * torch.log((target_weight_right + EPS) / (pred_weight_right + EPS))
    return loss.reshape(-1, 4).sum(dim=1) * loss_weight

def focal_loss(inputs, targets, alpha=-1, gamma=2, eps=1e-8):
    class_range = torch.arange(1, inputs.shape[1] + 1, device=inputs.device)
    pos_pred = (1 - inputs) ** gamma * torch.log(inputs + eps)
    neg_pred = inputs ** gamma * torch.log(1 - inputs + eps)

    pos_loss = (targets == class_range) * pos_pred * alpha
    neg_loss = (targets != class_range) * neg_pred * (1 - alpha)
    loss = -(pos_loss + neg_loss)
    return loss.sum(axis=1)

def vpd_focal_loss(inputs, targets, lstd, alpha=-1, gamma=2, eps=1e-8):
    std = lstd.exp().mean(1).unsqueeze(1)
    class_range = torch.arange(1, inputs.shape[1] + 1, device=inputs.device)
    pos_std = std[torch.where(targets==class_range)[0]]

    pos_norm_std = std/pos_std.max()
    pos_pred = pos_norm_std ** gamma * torch.log(inputs + eps)
    neg_pred = inputs ** gamma * torch.log(1 - inputs + eps)

    pos_loss = (targets == class_range) * pos_pred * alpha
    neg_loss = (targets != class_range) * neg_pred * (1 - alpha)
    loss = -(pos_loss + neg_loss)
    return loss.sum(axis=1)

def vpd1_focal_loss(inputs, targets, lstd, alpha=-1, gamma=2, eps=1e-8):
    std = lstd.exp().mean(1).unsqueeze(1)
    class_range = torch.arange(1, inputs.shape[1] + 1, device=inputs.device)
    
    pos_std = std[torch.where(targets==class_range)[0]]
    pos_norm_std = std/pos_std.max()
    neg_std = std[torch.where(targets!=class_range)[0]]
    neg_norm_std = (neg_std.min() + eps)/(std + eps)
    pos_pred = pos_norm_std ** gamma * torch.log(inputs + eps)
    neg_pred = neg_norm_std ** gamma * torch.log(1 - inputs + eps)

    pos_loss = (targets == class_range) * pos_pred * alpha
    neg_loss = (targets != class_range) * neg_pred * (1 - alpha)
    loss = -(pos_loss + neg_loss)
    return loss.sum(axis=1)

def pull_loss(labels, regs, anchors, weight):
    iou = 0.0
    pair_num = 0
    regs = regs.reshape(config.train_batch_per_gpu, -1, 4)
    for bid in range(config.train_batch_per_gpu):
        pull_label = labels[bid]
        for pull_boxes_idx in pull_label:
            pred_boxes = bbox_transform_inv_opr(anchors[pull_boxes_idx], regs[bid][pull_boxes_idx])
            pull_iou = box_overlap_opr(pred_boxes, pred_boxes)
            pull_iou_mask = torch.tril(torch.ones_like(pull_iou)).eq(0)
            iou += pull_iou.mul(pull_iou_mask).sum() / pull_iou_mask.sum()
        pair_num +=  len(pull_label)
    if  pair_num != 0:
        pull_loss = -(iou / pair_num).log()
    else:
        pull_loss = torch.tensor(0).type_as(regs)
    return weight * pull_loss

def push_loss(labels, regs, anchors, weight):
    iou = 0.0
    pair_num = 0
    regs = regs.reshape(config.train_batch_per_gpu, -1, 4)
    for bid in range(config.train_batch_per_gpu):
        push_label = labels[bid]
        for push_boxes_idx in push_label:
            pos_boxes = bbox_transform_inv_opr(anchors[push_boxes_idx[0]], regs[bid][push_boxes_idx[0]])
            neg_boxes = bbox_transform_inv_opr(anchors[push_boxes_idx[1]], regs[bid][push_boxes_idx[1]])
            iou += box_overlap_opr(pos_boxes, neg_boxes).mean()
        pair_num +=  len(push_label)
    if pair_num != 0:
        push_loss = torch.relu(-torch.log((1 - iou / pair_num)/(1 - config.test_nms)))
    else:
        push_loss = torch.tensor(0).type_as(regs)
    return weight * push_loss

def kldiv_loss(pred_mean, pred_lstd, kl_weight):
    loss = (1 + pred_lstd.mul(2) - pred_mean.pow(2) - pred_lstd.mul(2).exp()).mul(-0.5)
    return kl_weight * loss.mean()

def kldivergence_loss(pred, target, kl_weight):
    mean, lstd = torch.split(pred, 4, dim=1)
    loss = (mean - target).pow(2) / 2 / lstd.mul(2).exp() + lstd.exp()
    return kl_weight * loss.sum(dim=1)

def smooth_kl_loss(pred, target, kl_weight, beta: float):
    mean, lstd = torch.split(pred, 4, dim=1)
    abs_x = torch.abs(mean - target)
    in_mask = abs_x < beta
    mean_loss = torch.where(in_mask, abs_x ** 2, abs_x + beta**2 - beta)
    loss = mean_loss / 2 / lstd.mul(2).exp() + lstd
    return kl_weight * loss.sum(dim=1)

def kldiv_nvpd_loss(pred_mean, pred_lstd, kl_weight):
    loss = (1 + pred_lstd.mul(2) - pred_lstd.mul(2).exp()).mul(-0.5)
    return kl_weight * loss.mean()
    
def rcnn_kldiv_loss(pred_mean, pred_lstd):
    loss = (1 + pred_lstd.mul(2) - pred_mean.pow(2) - pred_lstd.mul(2).exp()).mul(-0.5)
    return loss.mean(axis=1)

def rcnn_mvpd_kldiv_loss(mean, lstd, mean_targets, std_targets, kl_weight):
    loss = (1 + 2 * std_targets.log() - 2 * lstd - (std_targets.pow(2) + 
                    (mean - mean_targets).pow(2)).div(lstd.mul(2).exp())).mul(-0.5)
    return kl_weight * loss.sum(axis=1)

def iouvar_loss(anchors, bbox_target, reg_samples, iouvar_weight):
    target_bbox = bbox_transform_inv_opr(anchors, bbox_target)
    samples_iou = align_box_giou_opr(reg_samples.reshape(-1, 4), target_bbox.repeat(config.sample_num,1))
    var_loss = torch.var(samples_iou.reshape(config.sample_num , -1), dim=0)
    return iouvar_weight * var_loss 

def gmkl_loss(pred_mean, pred_lstd, kl_weight):
    loss = (1 + pred_lstd.mul(2) - pred_mean.pow(2) - pred_lstd.mul(2).exp()).mul(-0.5)
    return kl_weight * loss.sum(axis=1) / config.n_components

def gmdl_loss(pred_mean, dl_weight):
    pred_mean = pred_mean.reshape(-1, config.n_components, 4)
    if config.n_components == 2:
        dist = torch.pow((pred_mean[:, 0, 0:2] - pred_mean[:, 1, 0:2]), 2).sum(dim=1)
    elif config.n_components == 3:
        dist1 = torch.pow((pred_mean[:, 0, 0:2] - pred_mean[:, 1, 0:2]), 2).sum(dim=1)
        dist2 = torch.pow((pred_mean[:, 0, 0:2] - pred_mean[:, 2, 0:2]), 2).sum(dim=1)
        dist3 = torch.pow((pred_mean[:, 1, 0:2] - pred_mean[:, 2, 0:2]), 2).sum(dim=1)
        dist = (dist1 + dist2 + dist3) / 3
    loss = F.relu(config.maxdist - dist)
    return dl_weight * loss

def log_normal(x, mean, lstd):
    var = lstd.mul(2).exp()
    # log(2 * pi) = 1.8379
    log_prob = -0.5 * torch.sum(1.8379 + torch.log(var) + torch.pow(x - mean, 2) / var, dim=1)
    return log_prob

def gmm_kld_loss(pred_box, pred_mean, pred_lstd, pred_prior_mean, pred_prior_lstd, kll_weight):
    kl_loss = log_normal(pred_box, pred_mean, pred_lstd) - log_normal(pred_box, pred_prior_mean, pred_prior_lstd)
    return kll_weight * kl_loss

def gmm_nent_loss(pred_bbox_qy_logit, enl_weight):
    nent_loss = (F.log_softmax(pred_bbox_qy_logit, dim=1) * F.softmax(pred_bbox_qy_logit, dim=1)).sum(axis=1)
    return enl_weight * nent_loss

def emd_loss_softmax(p_b0, p_s0, p_b1, p_s1, targets, labels):
    # reshape
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().flatten()
    # cons masks
    valid_masks = labels >= 0
    fg_masks = labels > 0
    # multiple class
    pred_delta = pred_delta.reshape(-1, config.num_classes, 4)
    fg_gt_classes = labels[fg_masks]
    pred_delta = pred_delta[fg_masks, fg_gt_classes, :]
    # loss for regression
    localization_loss = smooth_l1_loss(
        pred_delta,
        targets[fg_masks],
        config.rcnn_smooth_l1_beta)
    # loss for classification
    objectness_loss = softmax_loss(pred_score, labels)
    loss = objectness_loss * valid_masks
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)

def emd_gmvpd_loss_kl(p_b0, p_s0, p_b1, p_s1, targets, labels):
    # reshape
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().flatten()
    # cons masks
    valid_masks = labels >= 0
    fg_masks = labels > 0
    # multiple class
    pred_delta = pred_delta.reshape(-1, config.num_classes, 4)
    pred_delta = pred_delta[fg_masks, :4]
    # loss for regression
    localization_loss = smooth_l1_loss(
        pred_delta,
        targets[fg_masks],
        config.rcnn_smooth_l1_beta)
    # loss for classification
    objectness_loss = softmax_loss(pred_score, labels)
    loss = objectness_loss * valid_masks
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)

def mip_loss_softmax(p_b0, p_s0, p_b1, p_s1, targets, labels):
    # reshape
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().flatten()
    # cons masks
    valid_masks = labels >= 0
    fg_masks = labels > 0
    # multiple class
    pred_delta = pred_delta.reshape(pred_delta.shape[0], config.num_classes, -1)
    fg_gt_classes = labels[fg_masks]
    pred_delta = pred_delta[fg_masks, fg_gt_classes, :4]
    # loss for regression
    localization_loss = smooth_l1_loss(
        pred_delta,
        targets[fg_masks],
        config.rcnn_smooth_l1_beta)
    # loss for classification
    objectness_loss = softmax_loss(pred_score, labels)
    loss = objectness_loss * valid_masks
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)

def mip_loss_softmax_reg(p_b0, p_s0, p_b1, p_s1, targets, labels):
    # reshape
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().flatten()
    # cons masks
    valid_masks = labels >= 0
    fg_masks = labels > 0
    # multiple class
    pred_delta = pred_delta.reshape(pred_delta.shape[0], config.num_classes, -1)
    fg_gt_classes = labels[fg_masks]
    pred_delta = pred_delta[fg_masks, fg_gt_classes, :4]
    # loss for regression
    localization_loss = smooth_l1_loss(
        pred_delta,
        targets[fg_masks],
        config.rcnn_smooth_l1_beta)
    # loss for classification
    loss = pred_score.new_full((pred_score.shape[0],), 0, dtype=torch.float32)
    loss = loss * valid_masks
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)

def mip_2xreg_loss_softmax(p_b0, p_s0, p_b1, p_s1, targets, labels):
    # reshape
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().flatten()
    # cons masks
    valid_masks = labels >= 0
    fg_masks = labels > 0
    # multiple class
    pred_delta = pred_delta.reshape(pred_delta.shape[0], config.num_classes, -1)
    fg_gt_classes = labels[fg_masks]
    pred_delta = pred_delta[fg_masks, fg_gt_classes, :4]
    # loss for regression
    localization_loss = smooth_l1_loss(
        pred_delta,
        targets[fg_masks],
        config.rcnn_smooth_l1_beta)
    # loss for classification
    objectness_loss = softmax_loss(pred_score, labels)
    loss = objectness_loss * valid_masks
    loss[fg_masks] = loss[fg_masks] + localization_loss * 2
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)

def mip_normal1_loss_softmax(p_b0, p_s0, p_b1, p_s1, targets, labels):
    # reshape
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().flatten()
    # cons masks
    valid_masks = labels >= 0
    fg_masks = labels > 0
    # multiple class
    pred_delta = pred_delta[fg_masks, :4]
    # loss for regression
    localization_loss = smooth_l1_loss(
        pred_delta,
        targets[fg_masks],
        config.rcnn_smooth_l1_beta)
    # loss for classification
    objectness_loss = softmax_loss(pred_score, labels)
    loss = objectness_loss * valid_masks
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)

def mip_normal2_loss_softmax(p_b0, p_s0, p_b1, p_s1, targets, labels):
    # reshape
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().flatten()
    # cons masks
    valid_masks = labels >= 0
    fg_masks = labels > 0
    fg_mip_masks = fg_masks.reshape(-1, 2).sum(dim=1) >= 1
    fg_mip_masks = fg_mip_masks.repeat(2).reshape(2, -1).t().reshape(-1)
    # loss for regression
    localization_loss = smooth_l1_loss(
        pred_delta[fg_mip_masks, :4],
        targets[fg_mip_masks],
        config.rcnn_smooth_l1_beta)
    # loss for classification
    objectness_loss = softmax_loss(pred_score, labels)
    loss = objectness_loss * valid_masks
    loss[fg_mip_masks] = loss[fg_mip_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)
    
def emd_vpd_loss_softmax(p_b0, p_s0, p_b1, p_s1, targets, labels):
    # reshape
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().flatten()
    # cons masks
    valid_masks = labels >= 0
    fg_masks = labels > 0
    # multiple class
    pred_delta = pred_delta.reshape(-1, config.num_classes, 8)
    # variational inference
    pred_mean = pred_delta[:, :, :4]
    pred_lstd = pred_delta[:, :, 4:]
    scale = torch.tensor(config.prior_std).type_as(pred_lstd)
    pred_scale_std = pred_lstd.exp().mul(scale)
    pred_delta = pred_mean + pred_scale_std * torch.randn_like(pred_mean)
    fg_gt_classes = labels[fg_masks]
    pred_delta = pred_delta[fg_masks, fg_gt_classes, :]
    # loss for regression
    localization_loss = smooth_l1_loss(
        pred_delta,
        targets[fg_masks],
        config.rcnn_smooth_l1_beta)
    # loss for classification
    objectness_loss = softmax_loss(pred_score, labels)
    vl_gt_classes = labels[valid_masks]
    pred_mean = pred_mean[valid_masks, vl_gt_classes, :]
    pred_lstd = pred_lstd[valid_masks, vl_gt_classes, :]
    # loss for KL
    kldivergence_loss = rcnn_kldiv_loss(
        pred_mean,
        pred_lstd,
        config.kl_weight)
    loss = objectness_loss * valid_masks + kldivergence_loss * valid_masks
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)

def mip_vpd_loss_softmax(p_b0, p_s0, p_b1, p_s1, targets, labels):
    # reshape
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().flatten()
    # cons masks
    valid_masks = labels >= 0
    fg_masks = labels > 0
    # multiple class
    pred_delta = pred_delta.reshape(-1, config.num_classes, 8)
    # variational inference
    pred_mean = pred_delta[:, :, :4]
    pred_lstd = pred_delta[:, :, 4:]
    scale = torch.tensor(config.prior_std).type_as(pred_lstd)
    pred_scale_std = pred_lstd.exp().mul(scale)
    pred_delta = pred_mean + pred_scale_std * torch.randn_like(pred_mean)
    fg_gt_classes = labels[fg_masks]
    pred_delta = pred_delta[fg_masks, fg_gt_classes, :]
    # loss for regression
    localization_loss = smooth_l1_loss(
        pred_delta,
        targets[fg_masks],
        config.rcnn_smooth_l1_beta)
    # loss for classification
    objectness_loss = softmax_loss(pred_score, labels)
    vl_gt_classes = labels[valid_masks]
    pred_mean = pred_mean[valid_masks, vl_gt_classes, :]
    pred_lstd = pred_lstd[valid_masks, vl_gt_classes, :]
    # loss for KL
    kldivergence_loss = rcnn_kldiv_loss(
        pred_mean,
        pred_lstd,
        config.kl_weight)
    # loss for KL
    loss = objectness_loss * valid_masks + kldivergence_loss * valid_masks
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)

def mip_pos1_gmvpd_loss_softmax(p_b0, p_b1, targets, labels):
    # reshape
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().flatten()
    # cons masks
    valid_masks = labels >= 0
    fg_masks = labels > 0
    fg_mip_idx = torch.where(fg_masks.reshape(-1, 2).sum(dim=1) >= 1)[0]
    fg_mip_masks = fg_masks.reshape(-1, 2).sum(dim=1) >= 1
    fg_mip_masks = fg_mip_masks.repeat(2).reshape(2, -1).t().reshape(-1)
    # gaussian param
    pred_mean = pred_delta[:, :4]
    pred_lstd = pred_delta[:, 4:8]
    pred_prob = pred_delta[:, 8:9]
    # gaussian variational inference
    scale = torch.tensor(config.prior_std).type_as(pred_lstd)
    pred_scale_std = pred_lstd.exp().mul(scale)
    pred_delta = pred_mean + pred_scale_std * torch.randn_like(pred_mean)
    # gumble mixture
    pred_prob = pred_prob.reshape(-1, 2, 1)
    pred_delta = pred_delta.reshape(-1, 2, 4)
    gumbel_sample = -torch.log(-torch.log(torch.rand_like(pred_prob) + 1e-10) + 1e-10)
    weight = F.softmax((gumbel_sample + pred_prob) / config.gumbel_temperature, dim=1)
    pred_delta = torch.sum(pred_delta.mul(weight), dim=1, keepdim=True)
    pred_delta = pred_delta.repeat(1, 2, 1).reshape(-1, 4)

    # loss for regression
    localization_loss = pred_delta.new_full((pred_delta.shape[0],), INF, dtype=torch.float32)
    localization_loss[fg_mip_masks] = smooth_l1_loss(
        pred_delta[fg_mip_masks],
        targets[fg_mip_masks],
        config.rcnn_smooth_l1_beta)
    localization_loss = localization_loss.reshape(-1, 2)[fg_mip_idx, :]
    _, min_idx_loc = localization_loss.min(axis=1)
    localization_loss = localization_loss[torch.arange(localization_loss.shape[0]), min_idx_loc]

    # loss for category
    pred_prob = pred_prob.reshape(-1,2)[fg_mip_idx, :]
    categorical_loss = -entropy_loss(pred_prob) + 0.693147

    # loss for KL
    kldivergence_loss = rcnn_kldiv_loss(pred_mean[fg_mip_masks, :], pred_lstd[fg_mip_masks, :],
                        config.kl_weight).reshape(-1, 2).mean(dim=1)

    # all loss
    loss = {}
    loss['loss_loc'] = localization_loss.sum() / len(p_b0)
    loss['loss_cat'] = categorical_loss.sum() / len(p_b0)
    loss['loss_kld'] = kldivergence_loss.sum() / len(p_b0)
    return loss

def mip_pos2_gmvpd_loss_softmax(p_b0, p_s0, p_b1, p_s1, targets, labels):    
    # reshape
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().flatten()
    # cons masks
    fg_masks = labels > 0
    fg_mip_idx = torch.where(fg_masks.reshape(-1, 2).sum(dim=1) >= 1)[0]
    fg_mip_masks = fg_masks.reshape(-1, 2).sum(dim=1) >= 1
    fg_mip_masks = fg_mip_masks.repeat(2).reshape(2, -1).t().reshape(-1)
    # gaussian param
    pred_mean = pred_delta[:, :4]
    pred_lstd = pred_delta[:, 4:]
    pred_prob = F.softmax(pred_score, dim=1)[:, 1:]
    # gaussian variational inference
    scale = torch.tensor(config.prior_std).type_as(pred_lstd)
    pred_scale_std = pred_lstd.exp().mul(scale)
    pred_delta = pred_mean + pred_scale_std * torch.randn_like(pred_mean)
    # gumble mixture
    pred_prob = pred_prob.reshape(-1, 2, 1)
    pred_delta = pred_delta.reshape(-1, 2, 4)
    gumbel_sample = -torch.log(-torch.log(torch.rand_like(pred_prob) + 1e-10) + 1e-10)
    weight = F.softmax((gumbel_sample + pred_prob) / config.gumbel_temperature, dim=1)
    pred_delta = torch.sum(pred_delta.mul(weight), dim=1, keepdim=True)
    pred_delta = pred_delta.repeat(1, 2, 1).reshape(-1, 4)

    # loss for regression
    localization_loss = pred_delta.new_full((pred_delta.shape[0],), INF, dtype=torch.float32)
    localization_loss[fg_mip_masks] = smooth_l1_loss(
        pred_delta[fg_mip_masks],
        targets[fg_mip_masks],
        config.rcnn_smooth_l1_beta)
    localization_loss = localization_loss.reshape(-1, 2)[fg_mip_idx, :]
    _, min_idx_loc = localization_loss.min(axis=1)
    localization_loss = localization_loss[torch.arange(localization_loss.shape[0]), min_idx_loc]

    # loss for category
    pred_prob = pred_prob.reshape(-1,2)[fg_mip_idx, :]
    categorical_loss = -entropy_loss(pred_prob) + 0.693147

    # loss for KL
    kldivergence_loss = rcnn_kldiv_loss(pred_mean[fg_mip_masks, :], pred_lstd[fg_mip_masks, :],
                        config.kl_weight).reshape(-1, 2).mean(dim=1)

    # all loss
    loss = {}
    loss['loss_loc'] = localization_loss.sum() / len(p_b0)
    loss['loss_cat'] = categorical_loss.sum() / len(p_b0)
    loss['loss_kld'] = kldivergence_loss.sum() / len(p_b0)
    return loss

def mip_pos3_gmvpd_loss_softmax(p_b0, p_b1, targets, labels):
    # reshape
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().flatten()
    # cons masks
    valid_masks = labels >= 0
    fg_masks = labels > 0
    fg_mip_idx = torch.where(fg_masks.reshape(-1, 2).sum(dim=1) >= 1)[0]
    fg_mip_masks = fg_masks.reshape(-1, 2).sum(dim=1) >= 1
    fg_mip_masks = fg_mip_masks.repeat(2).reshape(2, -1).t().reshape(-1)
    # gaussian param
    pred_mean = pred_delta[:, :4]
    pred_lstd = pred_delta[:, 4:8]
    pred_prob = pred_delta[:, 8:9]
    # gaussian variational inference
    mip_targets = targets.reshape(-1, 2, 4) 
    scale = torch.abs(mip_targets[:, 0] - mip_targets[:, 1]) / 4
    scale = scale.repeat(1, 2).reshape(-1, 4)
    pred_scale_std = pred_lstd.exp().mul(scale)
    pred_delta = pred_mean + pred_scale_std * torch.randn_like(pred_mean)
    # gumble mixture
    pred_prob = pred_prob.reshape(-1, 2, 1)
    pred_delta = pred_delta.reshape(-1, 2, 4)
    gumbel_sample = -torch.log(-torch.log(torch.rand_like(pred_prob) + 1e-10) + 1e-10)
    weight = F.softmax((gumbel_sample + pred_prob) / config.gumbel_temperature, dim=1)
    pred_delta = torch.sum(pred_delta.mul(weight), dim=1, keepdim=True)
    pred_delta = pred_delta.repeat(1, 2, 1).reshape(-1, 4)

    # loss for regression
    localization_loss = pred_delta.new_full((pred_delta.shape[0],), INF, dtype=torch.float32)
    localization_loss[fg_mip_masks] = smooth_l1_loss(
        pred_delta[fg_mip_masks],
        targets[fg_mip_masks],
        config.rcnn_smooth_l1_beta)
    localization_loss = localization_loss.reshape(-1, 2)[fg_mip_idx, :]
    _, min_idx_loc = localization_loss.min(axis=1)
    localization_loss = localization_loss[torch.arange(localization_loss.shape[0]), min_idx_loc]

    # loss for category
    pred_prob = pred_prob.reshape(-1,2)[fg_mip_idx, :]
    categorical_loss = -entropy_loss(pred_prob) + 0.693147

    # loss for KL
    kldivergence_loss = rcnn_kldiv_loss(pred_mean[fg_mip_masks, :], pred_lstd[fg_mip_masks, :],
                        config.kl_weight).reshape(-1, 2).mean(dim=1)

    # all loss
    loss = {}
    loss['loss_loc'] = localization_loss.sum() / len(p_b0)
    loss['loss_cat'] = categorical_loss.sum() / len(p_b0)
    loss['loss_kld'] = kldivergence_loss.sum() / len(p_b0)
    return loss

def mip_pos4_gmvpd_loss_softmax(p_b0, p_s0, p_b1, p_s1, targets, labels):    
    # reshape
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().flatten()
    # cons masks
    valid_masks = labels >= 0
    fg_masks = labels > 0
    fg_mip_idx = torch.where(fg_masks.reshape(-1, 2).sum(dim=1) >= 1)[0]
    fg_mip_masks = fg_masks.reshape(-1, 2).sum(dim=1) >= 1
    fg_mip_masks = fg_mip_masks.repeat(2).reshape(2, -1).t().reshape(-1)
    # gaussian param
    pred_mean = pred_delta[:, :4]
    pred_lstd = pred_delta[:, 4:8]
    pred_prob = pred_delta[:, 8:9]
    # gaussian variational inference
    mip_targets = targets.reshape(-1, 2, 4) 
    scale = torch.abs(mip_targets[:, 0] - mip_targets[:, 1]) / 4
    scale = scale.repeat(1, 2).reshape(-1, 4)
    pred_scale_std = pred_lstd.exp().mul(scale)
    pred_delta = pred_mean + pred_scale_std * torch.randn_like(pred_mean)
    # gumble mixture
    pred_prob = pred_prob.reshape(-1, 2, 1)
    pred_delta = pred_delta.reshape(-1, 2, 4)
    gumbel_sample = -torch.log(-torch.log(torch.rand_like(pred_prob) + 1e-10) + 1e-10)
    weight = F.softmax((gumbel_sample + pred_prob) / config.gumbel_temperature, dim=1)
    pred_delta = torch.sum(pred_delta.mul(weight), dim=1, keepdim=True)
    pred_delta = pred_delta.repeat(1, 2, 1).reshape(-1, 4)

    # loss for classification
    objectness_loss = softmax_loss(pred_score, labels) * valid_masks
    objectness_loss = objectness_loss.reshape(-1, 2).sum(axis=1)

    # loss for regression
    localization_loss = pred_delta.new_full((pred_delta.shape[0],), INF, dtype=torch.float32)
    localization_loss[fg_mip_masks] = smooth_l1_loss(
        pred_delta[fg_mip_masks],
        targets[fg_mip_masks],
        config.rcnn_smooth_l1_beta)
    localization_loss = localization_loss.reshape(-1, 2)[fg_mip_idx, :]
    _, min_idx_loc = localization_loss.min(axis=1)
    localization_loss = localization_loss[torch.arange(localization_loss.shape[0]), min_idx_loc]

    # loss for category
    pred_prob = pred_prob.reshape(-1,2)[fg_mip_idx, :]
    categorical_loss = -entropy_loss(pred_prob) + 0.693147

    # loss for KL
    kldivergence_loss = rcnn_kldiv_loss(pred_mean[fg_mip_masks, :], pred_lstd[fg_mip_masks, :],
                        config.kl_weight).reshape(-1, 2).mean(dim=1)

    # all loss
    loss= {}
    loss['loss_cls'] = objectness_loss.sum() / len(p_b0)
    loss['loss_loc'] = localization_loss.sum() / len(p_b0)
    loss['loss_cat'] = categorical_loss.sum() / len(p_b0)
    loss['loss_kld'] = kldivergence_loss.sum() / len(p_b0)
    return loss

def mip_pos5_gmvpd_loss_softmax(p_b0, p_b1, targets, labels):
    # reshape
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().flatten()
    # cons masks
    fg_masks = labels > 0
    fg_mip_idx = torch.where(fg_masks.reshape(-1, 2).sum(dim=1) >= 1)[0]
    fg_mip_masks = fg_masks.reshape(-1, 2).sum(dim=1) >= 1
    fg_mip_masks = fg_mip_masks.repeat(2).reshape(2, -1).t().reshape(-1)
    # gaussian param
    pred_mean = pred_delta[:, :4]
    pred_lstd = pred_delta[:, 4:8]
    pred_prob = pred_delta[:, 8:9]
    # gaussian variational inference
    scale = torch.tensor(config.prior_std).type_as(pred_lstd)
    pred_scale_std = pred_lstd.exp().mul(scale)
    pred_delta = pred_mean + pred_scale_std * torch.randn_like(pred_mean)
    # gumble mixture
    pred_prob = pred_prob.reshape(-1, 2, 1)
    pred_delta = pred_delta.reshape(-1, 2, 4)
    gumbel_sample = -torch.log(-torch.log(torch.rand_like(pred_prob) + 1e-10) + 1e-10)
    weight = F.softmax((gumbel_sample + pred_prob) / config.gumbel_temperature, dim=1)
    pred_delta = torch.sum(pred_delta.mul(weight), dim=1, keepdim=True)
    pred_delta = pred_delta.repeat(1, 2, 1).reshape(-1, 4)

    # loss for regression
    localization_loss = pred_delta.new_full((pred_delta.shape[0],), 0, dtype=torch.float32)
    localization_loss[fg_mip_masks] = smooth_l1_loss(
        pred_delta[fg_mip_masks],
        targets[fg_mip_masks],
        config.rcnn_smooth_l1_beta)
    localization_loss = localization_loss.reshape(-1, 2)[fg_mip_idx, :]
    _, max_idx_loc = localization_loss.max(axis=1)
    localization_loss = localization_loss[torch.arange(localization_loss.shape[0]), max_idx_loc]

    # loss for category
    pred_prob = pred_prob.reshape(-1,2)[fg_mip_idx, :]
    categorical_loss = entropy_loss(pred_prob)

    # loss for KL
    kldivergence_loss = rcnn_kldiv_loss(pred_mean[fg_mip_masks, :], pred_lstd[fg_mip_masks, :],
                        config.kl_weight).reshape(-1, 2).mean(dim=1)

    # all loss
    loss = {}
    loss['loss_loc'] = localization_loss.sum() / len(p_b0)
    loss['loss_cat'] = categorical_loss.sum() / len(p_b0)
    loss['loss_kld'] = kldivergence_loss.sum() / len(p_b0)
    return loss

def entropy_loss(logits):
    log_q = F.log_softmax(logits, dim=-1)
    q = F.softmax(logits, dim=-1)
    return -torch.sum(q * log_q, dim=-1)

def mip_vpd_mkl_loss_softmax(p_b0, p_s0, p_b1, p_s1, targets, labels):
    # reshape
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().flatten()
    # cons masks
    valid_masks = labels >= 0
    fg_masks = labels > 0
    # multiple class
    pred_delta = pred_delta.reshape(-1, config.num_classes, 8)
    # variational inference
    pred_mean = pred_delta[:, :, :4]
    pred_lstd = pred_delta[:, :, 4:]
    scale = torch.tensor(config.prior_std).type_as(pred_lstd)
    pred_scale_std = pred_lstd.exp().mul(scale)
    pred_delta = pred_mean + pred_scale_std * torch.randn_like(pred_mean)
    fg_gt_classes = labels[fg_masks]
    pred_delta = pred_delta[fg_masks, fg_gt_classes, :]
    # loss for regression
    localization_loss = smooth_l1_loss(
        pred_delta,
        targets[fg_masks],
        config.rcnn_smooth_l1_beta)
    # loss for classification
    objectness_loss = softmax_loss(pred_score, labels)
    vl_gt_classes = labels[valid_masks]
    pred_mean = pred_mean[valid_masks, vl_gt_classes, :]
    pred_lstd = pred_lstd[valid_masks, vl_gt_classes, :]
    # loss for KL
    kldivergence_loss = rcnn_kldiv_loss(
        pred_mean,
        pred_lstd,
        config.kl_weight)
    # loss for mkl
    pred_multi_mean = pred_mean.reshape(-1,2,4)
    pred_multi_lstd = pred_mean.reshape(-1,2,4)
    multi_kldivergence_loss = rcnn_mkl_kldiv_loss(
        pred_multi_mean[:,0,:], pred_multi_mean[:,1,:], 
        pred_multi_lstd[:,0,:], pred_multi_lstd[:,1,:])

    loss = objectness_loss * valid_masks + kldivergence_loss * valid_masks + \
            multi_kldivergence_loss * valid_masks
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)

def rcnn_mkl_kldiv_loss(pred_mean1, pred_mean2, pred_lstd1, pred_lstd2):
    loss0 = (1 + pred_lstd1.mul(2) - pred_lstd2.mul(2) - (pred_lstd1.mul(2).exp() + \
                (pred_mean1 - pred_mean2).pow(2))/ pred_lstd2.mul(2).exp()).mul(-0.5)
    loss1 = (1 + pred_lstd2.mul(2) - pred_lstd1.mul(2) - (pred_lstd2.mul(2).exp() + \
                (pred_mean2 - pred_mean1).pow(2))/ pred_lstd1.mul(2).exp()).mul(-0.5)
    loss = torch.cat([loss0, loss1], dim=0)
    return config.kl_weight * torch.relu(config.kl_delta-loss.mean(axis=1))

def refine_loss_softmax(pred_delta, pred_score, targets, labels):
    # reshape
    targets = targets.reshape(-1, 4)
    labels = labels.long().flatten()
    # cons masks
    valid_masks = labels >= 0
    fg_masks = labels > 0
    # multiple class
    pred_delta = pred_delta.reshape(-1, config.num_classes, 4)
    fg_gt_classes = labels[fg_masks]
    pred_delta = pred_delta[fg_masks, fg_gt_classes, :]
    # loss for regression
    localization_loss = smooth_l1_loss(
        pred_delta,
        targets[fg_masks],
        config.rcnn_smooth_l1_beta)
    # loss for classification
    objectness_loss = softmax_loss(pred_score, labels)
    loss = objectness_loss * valid_masks
    loss[fg_masks] = loss[fg_masks] + localization_loss
    return loss.reshape(-1, 1)

def emd_loss_focal(p_b0, p_s0, p_b1, p_s1, targets, labels):
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().reshape(-1, 1)
    valid_mask = (labels >= 0).flatten()
    objectness_loss = focal_loss(pred_score, labels,
            config.focal_loss_alpha, config.focal_loss_gamma)
    fg_masks = (labels > 0).flatten()
    localization_loss = smooth_l1_loss(
            pred_delta[fg_masks],
            targets[fg_masks],
            config.smooth_l1_beta)
    loss = objectness_loss * valid_mask
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)

def mip_loss_focal(p_b0, p_s0, p_b1, p_s1, targets, labels):
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().reshape(-1, 1)
    valid_mask = (labels >= 0).flatten()
    objectness_loss = focal_loss(pred_score, labels,
            config.focal_loss_alpha, config.focal_loss_gamma)
    fg_masks = (labels > 0).flatten()
    localization_loss = smooth_l1_loss(
            pred_delta[fg_masks],
            targets[fg_masks],
            config.smooth_l1_beta)
    loss = objectness_loss * valid_mask
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    return loss.reshape(-1, 1)

def refine_loss_focal(pred_delta, pred_score, targets, labels):
    targets = targets.reshape(-1, 4)
    labels = labels.long().reshape(-1, 1)
    valid_mask = (labels >= 0).flatten()
    objectness_loss = focal_loss(pred_score, labels,
            config.focal_loss_alpha, config.focal_loss_gamma)
    fg_masks = (labels > 0).flatten()
    localization_loss = smooth_l1_loss(
            pred_delta[fg_masks],
            targets[fg_masks],
            config.smooth_l1_beta)
    loss = objectness_loss * valid_mask
    loss[fg_masks] = loss[fg_masks] + localization_loss
    return loss.reshape(-1, 1)

def mip_pos_gmvpd_loss_focal(p_b0, p_b1, targets, labels):
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().reshape(-1, 1)
    fg_masks = (labels > 0).flatten()
    fg_mip_idx = torch.where(fg_masks.reshape(-1, 2).sum(dim=1) >= 1)[0]
    fg_mip_masks = fg_masks.reshape(-1, 2).sum(dim=1) >= 1
    fg_mip_masks = fg_mip_masks.repeat(2).reshape(2, -1).t().reshape(-1)

    # gaussian mixture variational inference
    pred_mean = pred_delta[:, :4]
    pred_lstd = pred_delta[:, 4:8]
    pred_prob = pred_delta[:, 8:9]
    # gaussian variational inference
    scale = torch.tensor(config.prior_std).type_as(pred_lstd)
    pred_scale_std = pred_lstd.exp().mul(scale)
    pred_delta = pred_mean + pred_scale_std * torch.randn_like(pred_mean)
    # gumble mixture
    pred_prob = pred_prob.reshape(-1, 2, 1)
    pred_delta = pred_delta.reshape(-1, 2, 4)
    gumbel_sample = -torch.log(-torch.log(torch.rand_like(pred_prob) + 1e-10) + 1e-10)
    weight = F.softmax((gumbel_sample + pred_prob) / config.gumbel_temperature, dim=1)
    pred_delta = torch.sum(pred_delta.mul(weight), dim=1, keepdim=True)
    pred_delta = pred_delta.repeat(1, 2, 1).reshape(-1, 4)

    # loss for regression
    localization_loss = pred_delta.new_full((pred_delta.shape[0],), INF, dtype=torch.float32)
    localization_loss[fg_masks] = smooth_l1_loss(
        pred_delta[fg_masks],
        targets[fg_masks],
        config.smooth_l1_beta)
    localization_loss = localization_loss.reshape(-1, 2)[fg_mip_idx, :]
    _, min_idx_loc = localization_loss.min(axis=1)
    localization_loss = localization_loss[torch.arange(localization_loss.shape[0]), min_idx_loc]

    # loss for category
    pred_prob = pred_prob.reshape(-1,2)[fg_mip_idx, :]
    categorical_loss = -entropy_loss(pred_prob) + 0.693147

    # loss for KL
    kldivergence_loss = rcnn_kldiv_loss(pred_mean[fg_mip_masks, :], pred_lstd[fg_mip_masks, :],
                        config.kl_weight).reshape(-1, 2).sum(dim=1)

    loss = {}
    loss['loss_loc'] = localization_loss
    loss['loss_cat'] = categorical_loss
    loss['loss_kld'] = kldivergence_loss

    return loss

def mip_pos2_gmvpd_loss_focal(p_b0, p_s0, p_b1, p_s1, targets, labels):
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().reshape(-1, 1)
    fg_masks = (labels > 0).flatten()
    fg_mip_idx = torch.where(fg_masks.reshape(-1, 2).sum(dim=1) >= 1)[0]
    fg_mip_masks = fg_masks.reshape(-1, 2).sum(dim=1) >= 1
    fg_mip_masks = fg_mip_masks.repeat(2).reshape(2, -1).t().reshape(-1)

    # gaussian mixture variational inference
    pred_mean = pred_delta[:, :4]
    pred_lstd = pred_delta[:, 4:]
    pred_prob = pred_score
    # gaussian variational inference
    scale = torch.tensor(config.prior_std).type_as(pred_lstd)
    pred_scale_std = pred_lstd.exp().mul(scale)
    pred_delta = pred_mean + pred_scale_std * torch.randn_like(pred_mean)
    # gumble mixture
    pred_prob = pred_prob.reshape(-1, 2, 1)
    pred_delta = pred_delta.reshape(-1, 2, 4)
    gumbel_sample = -torch.log(-torch.log(torch.rand_like(pred_prob) + 1e-10) + 1e-10)
    weight = F.softmax((gumbel_sample + pred_prob) / config.gumbel_temperature, dim=1)
    pred_delta = torch.sum(pred_delta.mul(weight), dim=1, keepdim=True)
    pred_delta = pred_delta.repeat(1, 2, 1).reshape(-1, 4)

    # loss for regression
    localization_loss = pred_delta.new_full((pred_delta.shape[0],), INF, dtype=torch.float32)
    localization_loss[fg_mip_masks] = smooth_l1_loss(
        pred_delta[fg_mip_masks],
        targets[fg_mip_masks],
        config.smooth_l1_beta)
    localization_loss = localization_loss.reshape(-1, 2)[fg_mip_idx, :]
    _, min_idx_loc = localization_loss.min(axis=1)
    localization_loss = localization_loss[torch.arange(localization_loss.shape[0]), min_idx_loc]

    # loss for category
    pred_prob = pred_prob.reshape(-1,2)[fg_mip_idx, :]
    categorical_loss = -entropy_loss(pred_prob) + 0.693147

    # loss for KL
    kldivergence_loss = rcnn_kldiv_loss(pred_mean[fg_mip_masks, :], pred_lstd[fg_mip_masks, :],
                        config.kl_weight).reshape(-1, 2).sum(dim=1)

    loss = {}
    loss['loss_loc'] = localization_loss
    loss['loss_cat'] = categorical_loss
    loss['loss_kld'] = kldivergence_loss

    return loss

def mip_pos3_gmvpd_loss_focal(p_b0, p_b1, targets, labels):
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().reshape(-1, 1)
    fg_masks = (labels > 0).flatten()
    fg_mip_idx = torch.where(fg_masks.reshape(-1, 2).sum(dim=1) >= 2)[0]
    fg_mip_masks = fg_masks.reshape(-1, 2).sum(dim=1) >= 2
    fg_mip_masks = fg_mip_masks.repeat(2).reshape(2, -1).t().reshape(-1)

    # gaussian mixture variational inference
    pred_mean = pred_delta[:, :4]
    pred_lstd = pred_delta[:, 4:8]
    pred_prob = pred_delta[:, 8:9]
    # gaussian variational inference
    scale = torch.tensor(config.prior_std).type_as(pred_lstd)
    pred_scale_std = pred_lstd.exp().mul(scale)
    pred_delta = pred_mean + pred_scale_std * torch.randn_like(pred_mean)
    # gumble mixture
    pred_prob = pred_prob.reshape(-1, 2, 1)
    pred_delta = pred_delta.reshape(-1, 2, 4)
    gumbel_sample = -torch.log(-torch.log(torch.rand_like(pred_prob) + 1e-10) + 1e-10)
    weight = F.softmax((gumbel_sample + pred_prob) / config.gumbel_temperature, dim=1)
    pred_delta = torch.sum(pred_delta.mul(weight), dim=1, keepdim=True)
    pred_delta = pred_delta.repeat(1, 2, 1).reshape(-1, 4)

    # loss for regression
    localization_loss = pred_delta.new_full((pred_delta.shape[0],), INF, dtype=torch.float32)
    localization_loss[fg_mip_masks] = smooth_l1_loss(
        pred_delta[fg_mip_masks],
        targets[fg_mip_masks],
        config.smooth_l1_beta)
    localization_loss = localization_loss.reshape(-1, 2)[fg_mip_idx, :]
    if localization_loss.shape[0] != 0:
        _, min_idx_loc = localization_loss.min(axis=1)
        localization_loss = localization_loss[torch.arange(localization_loss.shape[0]), min_idx_loc]
        # loss for category
        pred_prob = pred_prob.reshape(-1,2)[fg_mip_idx, :]
        categorical_loss = -entropy_loss(pred_prob) + 0.693147
        # loss for KL
        kldivergence_loss = rcnn_kldiv_loss(
            pred_mean[fg_mip_masks, :], 
            pred_lstd[fg_mip_masks, :]).reshape(-1, 2).sum(dim=1)
    else:
        localization_loss = pred_delta.new_full((1,), 0, dtype=torch.float32)
        categorical_loss = pred_delta.new_full((1,), 0, dtype=torch.float32)
        kldivergence_loss = pred_delta.new_full((1,), 0, dtype=torch.float32)
    loss = {}
    loss['loss_loc'] = localization_loss
    loss['loss_cat'] = categorical_loss * config.kl_weight
    loss['loss_kld'] = kldivergence_loss * config.kl_weight
    return loss