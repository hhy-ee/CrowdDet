import torch
import numpy as np
import torch.nn.functional as F
from det_oprs.bbox_opr import bbox_transform_inv_opr
from det_oprs.bbox_opr import box_overlap_opr, align_box_giou_opr, align_box_overlap_opr
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

def smooth_l1_loss(pred, target, beta: float):
    if beta < 1e-5:
        loss = torch.abs(input - target)
    else:
        abs_x = torch.abs(pred- target)
        in_mask = abs_x < beta
        loss = torch.where(in_mask, 0.5 * abs_x ** 2 / beta, abs_x - 0.5 * beta)
    return loss.sum(axis=-1)

def iou_loss(pred, target, anchor):
    pred_boxes = bbox_transform_inv_opr(anchor, pred)
    target_boxes = bbox_transform_inv_opr(anchor, target)
    ious = align_box_overlap_opr(pred_boxes, target_boxes)
    loss = 1 - ious
    return loss

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

def kdl_xywh_loss(pred, target, loss_weight):
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

def ce_gaussian_loss(dist, target, loss_weight):
    scale = (config.project.shape[1] - 1) / 2 / config.project[0,-1]
    acc = 1 / scale / 2
    target = (target.reshape(-1) + config.project[0,-1]) * scale
    target = target.clamp(min=EPS, max=2 * config.project[0,-1] * scale-EPS)
    idx_left = target.long()
    idx_right = idx_left + 1
    weight_left = idx_right.float() - target
    weight_right = target - idx_left.float()
    # target distribution
    target_dist = weight_left.new_full((weight_left.shape[0], \
        config.project.shape[1]), 0, dtype=torch.float32)
    target_dist[torch.arange(target_dist.shape[0]), idx_left] = weight_left
    target_dist[torch.arange(target_dist.shape[0]), idx_right] = weight_right
    # predict distribution
    mean= dist[:, :4].reshape(-1, 1)
    lstd= dist[:, 4:].reshape(-1, 1)
    Qg = torch.distributions.normal.Normal(mean, lstd.exp())
    project = torch.tensor(config.project).type_as(mean).repeat(mean.shape[0],1)
    pred_dist = Qg.cdf(project + acc) - Qg.cdf(project - acc)
    # CE distance
    loss = F.cross_entropy(pred_dist, idx_left, reduction='none') * weight_left \
        + F.cross_entropy(pred_dist, idx_right, reduction='none') * weight_right
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
    return loss.reshape(-1, 4).sum(dim=1) * loss_weight

def js_gaussian_loss(dist, target, loss_weight):
    scale = (config.project.shape[1] - 1) / 2 / config.project[0,-1]
    acc = 1 / scale / 2
    target = (target.reshape(-1) + config.project[0,-1]) * scale
    target = target.clamp(min=EPS, max=2 * config.project[0,-1] * scale-EPS)
    idx_left = target.long()
    idx_right = idx_left + 1
    weight_left = idx_right.float() - target
    weight_right = target - idx_left.float()
    # target distribution
    target_dist = weight_left.new_full((weight_left.shape[0], \
        config.project.shape[1]), 0, dtype=torch.float32)
    target_dist[torch.arange(target_dist.shape[0]), idx_left] = weight_left
    target_dist[torch.arange(target_dist.shape[0]), idx_right] = weight_right
    # predict distribution
    mean= dist[:, :4].reshape(-1, 1)
    lstd= dist[:, 4:].reshape(-1, 1)
    Qg = torch.distributions.normal.Normal(mean, lstd.exp())
    project = torch.tensor(config.project).type_as(mean).repeat(mean.shape[0],1)
    pred_dist = Qg.cdf(project + acc) - Qg.cdf(project - acc)
    # JS distance
    total_dist = (target_dist + pred_dist) / 2
    loss1 = pred_dist * torch.log((pred_dist + EPS) / (total_dist + EPS))
    loss2 = target_dist * torch.log((target_dist + EPS) / (total_dist + EPS))
    loss = (loss1 + loss2).sum(dim=1) / 2
    return loss.reshape(-1, 4).sum(dim=1) * loss_weight

def asymmetric_js_gaussian_loss(dist, target, loss_weight):
    scale = (config.project.shape[1] - 1) / 2 / config.project[0,-1]
    acc = 1 / scale / 2
    target = (target.reshape(-1) + config.project[0,-1]) * scale
    target = target.clamp(min=EPS, max=2 * config.project[0,-1] * scale-EPS)
    idx_left = target.long()
    idx_right = idx_left + 1
    weight_left = idx_right.float() - target
    weight_right = target - idx_left.float()
    # target distribution
    target_dist = weight_left.new_full((weight_left.shape[0], \
        config.project.shape[1]), 0, dtype=torch.float32)
    target_dist[torch.arange(target_dist.shape[0]), idx_left] = weight_left
    target_dist[torch.arange(target_dist.shape[0]), idx_right] = weight_right
    # predict distribution
    mean= dist[:, :4].reshape(-1, 1)
    lstd= dist[:, 4:].reshape(-1, 1)
    Qg = torch.distributions.normal.Normal(mean, lstd.exp())
    project = torch.tensor(config.project).type_as(mean).repeat(mean.shape[0],1)
    pred_dist = Qg.cdf(project + acc) - Qg.cdf(project - acc)
    # JS distance
    total_dist = pred_dist * config.alpha_skew + target_dist * (1 - config.alpha_skew)
    loss1 = pred_dist * torch.log((pred_dist + EPS) / (total_dist + EPS))
    loss2 = target_dist * torch.log((target_dist + EPS) / (total_dist + EPS))
    loss = (loss1 * config.alpha_skew + loss2 * (1 - config.alpha_skew)).sum(dim=1)
    return loss.reshape(-1, 4).sum(dim=1) * loss_weight

def ws_gaussian_loss(dist, target, loss_weight, p=1):
    scale = (config.project.shape[1] - 1) / 2 / config.project[0,-1]
    acc = 1 / scale / 2
    target = (target.reshape(-1) + config.project[0,-1]) * scale
    target = target.clamp(min=EPS, max=2 * config.project[0,-1] * scale-EPS)
    idx_left = target.long()
    idx_right = idx_left + 1
    weight_left = idx_right.float() - target
    weight_right = target - idx_left.float()
    # target distribution
    target_dist = weight_left.new_full((weight_left.shape[0], \
        config.project.shape[1]), 0, dtype=torch.float32)
    target_dist[torch.arange(target_dist.shape[0]), idx_left] = weight_left
    target_dist[torch.arange(target_dist.shape[0]), idx_right] = weight_right
    # predict distribution
    mean= dist[:, :4].reshape(-1, 1)
    lstd= dist[:, 4:].reshape(-1, 1)
    Qg = torch.distributions.normal.Normal(mean, lstd.exp())
    project = torch.tensor(config.project).type_as(mean).repeat(mean.shape[0],1)
    pred_dist = Qg.cdf(project + acc) - Qg.cdf(project - acc)
    # WS distance
    pred_cdf = torch.cumsum(pred_dist, dim=1)
    target_cdf = torch.cumsum(target_dist, dim=1)
    if p == 1:
        loss =  torch.sum(torch.abs(pred_cdf - target_cdf) * acc * 2, dim=1)
    return loss.reshape(-1, 4).sum(dim=1) * loss_weight

def ws_gaussian_loss_(dist, target, loss_weight, p=1):
    scale = (config.project.shape[1] - 1) / 2 / config.project[0,-1]
    acc = 1 / scale / 2
    target = (target.reshape(-1) + config.project[0,-1]) * scale
    target = target.clamp(min=EPS, max=2 * config.project[0,-1] * scale-EPS)
    idx_left = target.long()
    idx_right = idx_left + 1
    weight_left = idx_right.float() - target
    weight_right = target - idx_left.float()
    # target distribution
    project = torch.tensor(config.project).type_as(weight_left).repeat(weight_left.shape[0],1)
    u1 = project[torch.arange(weight_left.shape[0]), idx_left]
    u2 = project[torch.arange(weight_left.shape[0]), idx_right]
    u_left = torch.cat([(u1 - acc).reshape(-1, 1, 1), (u2 - acc).reshape(-1, 1, 1)], dim=2)
    u_right = torch.cat([(u1 + acc).reshape(-1, 1, 1), (u2 + acc).reshape(-1, 1, 1)], dim=2)
    uniform = torch.distributions.uniform.Uniform(u_left, u_right)
    cat = torch.distributions.categorical.Categorical(torch.cat([weight_left.\
        reshape(-1, 1, 1), weight_right.reshape(-1, 1, 1)], dim=2))
    target_dist = torch.distributions.mixture_same_family.MixtureSameFamily(cat, uniform)
    # predict distribution
    mean= dist[:, :4].reshape(-1, 1)
    lstd= dist[:, 4:].reshape(-1, 1)
    Qg = torch.distributions.normal.Normal(mean, lstd.exp())
    # WS distance
    pred_cdf = Qg.cdf(project)
    target_cdf = target_dist.cdf(project)
    if p == 1:
        loss =  torch.sum(torch.abs(pred_cdf - target_cdf) * acc * 2, dim=1)
    return loss.reshape(-1, 4).sum(dim=1) * loss_weight

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
    return loss.reshape(-1, 4).sum(dim=1) * loss_weight

def js_gmm_loss(prob, mean, lstd, target, loss_weight):
    scale = (config.project.shape[1] - 1) / 2 / config.project[0,-1]
    acc = 1 / scale / 2
    target = (target.reshape(-1) + config.project[0,-1]) * scale
    target = target.clamp(min=EPS, max=2 * config.project[0,-1] * scale-EPS)
    idx_left = target.long()
    idx_right = idx_left + 1
    weight_left = idx_right.float() - target
    weight_right = target - idx_left.float()
    # target distribution
    target_dist = weight_left.new_full((weight_left.shape[0], \
        config.project.shape[1]), 0, dtype=torch.float32)
    target_dist[torch.arange(target_dist.shape[0]), idx_left] = weight_left
    target_dist[torch.arange(target_dist.shape[0]), idx_right] = weight_right
    # predict distribution
    mean = mean.reshape(mean.shape[0], config.component.shape[1], 1)
    lstd = lstd.reshape(mean.shape[0], config.component.shape[1], 1)
    prob = prob.reshape(mean.shape[0], config.component.shape[1], 1)
    Qgmm = torch.distributions.normal.Normal(mean, lstd.exp())
    project = torch.tensor(config.project).type_as(mean)
    project = project.unsqueeze(1).repeat(mean.shape[0], mean.shape[1], 1)
    pred_dist = Qgmm.cdf(project + acc).mul(prob).sum(dim=1) - \
        Qgmm.cdf(project - acc).mul(prob).sum(dim=1)
    # JS distance
    total_dist = (target_dist + pred_dist) / 2
    loss1 = pred_dist * torch.log((pred_dist + EPS) / (total_dist + EPS))
    loss2 = target_dist * torch.log((target_dist + EPS) / (total_dist + EPS))
    loss = (loss1 + loss2).sum(dim=1) / 2
    return loss.reshape(-1, 4).sum(dim=1) * loss_weight

def asymmetric_js_gmm_loss(prob, mean, lstd, target, loss_weight):
    scale = (config.project.shape[1] - 1) / 2 / config.project[0,-1]
    acc = 1 / scale / 2
    target = (target.reshape(-1) + config.project[0,-1]) * scale
    target = target.clamp(min=EPS, max=2 * config.project[0,-1] * scale-EPS)
    idx_left = target.long()
    idx_right = idx_left + 1
    weight_left = idx_right.float() - target
    weight_right = target - idx_left.float()
    # target distribution
    target_dist = weight_left.new_full((weight_left.shape[0], \
        config.project.shape[1]), 0, dtype=torch.float32)
    target_dist[torch.arange(target_dist.shape[0]), idx_left] = weight_left
    target_dist[torch.arange(target_dist.shape[0]), idx_right] = weight_right
    # predict distribution
    mean = mean.reshape(mean.shape[0], config.component.shape[1], 1)
    lstd = lstd.reshape(mean.shape[0], config.component.shape[1], 1)
    prob = prob.reshape(mean.shape[0], config.component.shape[1], 1)
    Qgmm = torch.distributions.normal.Normal(mean, lstd.exp())
    project = torch.tensor(config.project).type_as(mean)
    project = project.unsqueeze(1).repeat(mean.shape[0], mean.shape[1], 1)
    pred_dist = Qgmm.cdf(project + acc).mul(prob).sum(dim=1) - \
        Qgmm.cdf(project - acc).mul(prob).sum(dim=1)
    # JS distance
    total_dist = pred_dist * config.alpha_skew + target_dist * (1 - config.alpha_skew)
    loss1 = pred_dist * torch.log((pred_dist + EPS) / (total_dist + EPS))
    loss2 = target_dist * torch.log((target_dist + EPS) / (total_dist + EPS))
    loss = (loss1 * config.alpha_skew + loss2 * (1 - config.alpha_skew)).sum(dim=1)
    return loss.reshape(-1, 4).sum(dim=1) * loss_weight

def ws_gmm_loss(prob, mean, lstd, target, loss_weight, p=1):
    scale = (config.project.shape[1] - 1) / 2 / config.project[0,-1]
    acc = 1 / scale / 2
    target = (target.reshape(-1) + config.project[0,-1]) * scale
    target = target.clamp(min=EPS, max=2 * config.project[0,-1] * scale-EPS)
    idx_left = target.long()
    idx_right = idx_left + 1
    weight_left = idx_right.float() - target
    weight_right = target - idx_left.float()
    # target distribution
    target_dist = weight_left.new_full((weight_left.shape[0], \
        config.project.shape[1]), 0, dtype=torch.float32)
    target_dist[torch.arange(target_dist.shape[0]), idx_left] = weight_left
    target_dist[torch.arange(target_dist.shape[0]), idx_right] = weight_right
    # predict distribution
    mean = mean.reshape(mean.shape[0], config.component.shape[1], 1)
    lstd = lstd.reshape(mean.shape[0], config.component.shape[1], 1)
    prob = prob.reshape(mean.shape[0], config.component.shape[1], 1)
    Qgmm = torch.distributions.normal.Normal(mean, lstd.exp())
    project = torch.tensor(config.project).type_as(mean)
    project = project.unsqueeze(1).repeat(mean.shape[0], mean.shape[1], 1)
    pred_dist = Qgmm.cdf(project + acc).mul(prob).sum(dim=1) - \
        Qgmm.cdf(project - acc).mul(prob).sum(dim=1)
    # WS distance
    pred_cdf = torch.cumsum(pred_dist, dim=1)
    target_cdf = torch.cumsum(target_dist, dim=1)
    if p == 1:
        loss =  torch.sum(torch.abs(pred_cdf - target_cdf) * acc * 2, dim=1)
    return loss.reshape(-1, 4).sum(dim=1) * loss_weight

def continuous_ws_gmm_loss(prob, mean, lstd, target, loss_weight, p=1):
    scale = (config.project.shape[1] - 1) / 2 / config.project[0,-1]
    acc = 1 / scale / 2
    target = (target.reshape(-1) + config.project[0,-1]) * scale
    target = target.clamp(min=EPS, max=2 * config.project[0,-1] * scale-EPS)
    idx_left = target.long()
    idx_right = idx_left + 1
    weight_left = idx_right.float() - target
    weight_right = target - idx_left.float()
    # target distribution
    project = torch.tensor(config.project).type_as(weight_left).repeat(weight_left.shape[0],1)
    u1 = project[torch.arange(weight_left.shape[0]), idx_left]
    u2 = project[torch.arange(weight_left.shape[0]), idx_right]
    u_left = torch.cat([(u1 - acc).reshape(-1, 1, 1), (u2 - acc).reshape(-1, 1, 1)], dim=2)
    u_right = torch.cat([(u1 + acc).reshape(-1, 1, 1), (u2 + acc).reshape(-1, 1, 1)], dim=2)
    uniform = torch.distributions.uniform.Uniform(u_left, u_right)
    cat = torch.distributions.categorical.Categorical(torch.cat([weight_left.\
        reshape(-1, 1, 1), weight_right.reshape(-1, 1, 1)], dim=2))
    target_dist = torch.distributions.mixture_same_family.MixtureSameFamily(cat, uniform)
    # predict distribution
    mean = mean.reshape(weight_left.shape[0], config.component.shape[1], 1)
    lstd = lstd.reshape(weight_left.shape[0], config.component.shape[1], 1)
    prob = prob.reshape(weight_left.shape[0], config.component.shape[1], 1)
    Qgmm = torch.distributions.normal.Normal(mean, lstd.exp())
    # WS distance
    project = torch.tensor(config.project).type_as(mean)
    mm_project = project.unsqueeze(1).repeat(1, mean.shape[1], 1)
    pred_cdf = Qgmm.cdf(mm_project).mul(prob).sum(dim=1)
    target_cdf = target_dist.cdf(project)
    if p == 1:
        loss =  torch.sum(torch.abs(pred_cdf - target_cdf) * acc * 2, dim=1)
    return loss.reshape(-1, 4).sum(dim=1) * loss_weight

def focal_loss(inputs, targets, alpha=-1, gamma=2, eps=1e-8):
    class_range = torch.arange(1, inputs.shape[1] + 1, device=inputs.device)
    pos_pred = (1 - inputs) ** gamma * torch.log(inputs + eps)
    neg_pred = inputs ** gamma * torch.log(1 - inputs + eps)

    pos_loss = (targets == class_range) * pos_pred * alpha
    neg_loss = (targets != class_range) * neg_pred * (1 - alpha)
    loss = -(pos_loss + neg_loss)
    return loss.sum(axis=1)

def kldivergence_loss(pred_mean, pred_lstd, kl_weight):
    loss = (1 + pred_lstd.mul(2) - pred_mean.pow(2) - pred_lstd.mul(2).exp()).mul(-0.5)
    return kl_weight * loss.mean()

def log_normal(x, mean, lstd):
    var = lstd.mul(2).exp()
    # log(2 * pi) = 1.8379
    log_prob = -0.5 * torch.sum(1.8379 + torch.log(var) + torch.pow(x - mean, 2) / var, dim=1)
    return log_prob

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

def entropy_loss(logits):
    log_q = F.log_softmax(logits, dim=-1)
    q = F.softmax(logits, dim=-1)
    return -torch.sum(q * log_q, dim=-1)

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