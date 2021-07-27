import torch
import torch.nn.functional as F
from config import config

def softmax_loss(score, label, ignore_label=-1):
    with torch.no_grad():
        max_score = score.max(axis=1, keepdims=True)[0]
    score -= max_score
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
    return loss.sum(axis=1)

def focal_loss(inputs, targets, alpha=-1, gamma=2, eps=1e-8):
    class_range = torch.arange(1, inputs.shape[1] + 1, device=inputs.device)
    pos_pred = (1 - inputs) ** gamma * torch.log(inputs + eps)
    neg_pred = inputs ** gamma * torch.log(1 - inputs + eps)

    pos_loss = (targets == class_range) * pos_pred * alpha
    neg_loss = (targets != class_range) * neg_pred * (1 - alpha)
    loss = -(pos_loss + neg_loss)
    return loss.sum(axis=1)

def kldiv_loss(pred_mean, pred_lstd, kl_weight):
    loss = (1 + pred_lstd.mul(2) - pred_mean.pow(2) - pred_lstd.mul(2).exp()).mul(-0.5)
    return kl_weight * loss.sum(axis=1) 

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
