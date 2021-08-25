import math

import torch
from torch.nn import functional as F
from torchvision.ops import roi_align

def point_sample(input, point_coords, **kwargs):
    input = input.unsqueeze(0)
    point_coords = point_coords.unsqueeze(0)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    return output.squeeze(0).transpose(1, 0)

def get_point_coords_wrt_image(boxes_coords, point_coords):
    with torch.no_grad():
        point_coords_wrt_image = point_coords.clone()
        point_coords_wrt_image[:, :, 0] = point_coords_wrt_image[:, :, 0] * (
            boxes_coords[:, None, 2] - boxes_coords[:, None, 0]
        )
        point_coords_wrt_image[:, :, 1] = point_coords_wrt_image[:, :, 1] * (
            boxes_coords[:, None, 3] - boxes_coords[:, None, 1]
        )
        point_coords_wrt_image[:, :, 0] += boxes_coords[:, None, 0]
        point_coords_wrt_image[:, :, 1] += boxes_coords[:, None, 1]
    return point_coords_wrt_image

def assign_boxes_to_levels(rois, min_level, max_level, canonical_box_size=224, canonical_level=4):
    """
        rois (Tensor): A tensor of shape (N, 5).
        min_level (int), max_level (int), canonical_box_size (int), canonical_level (int).
        Return a tensor of length N.
    """
    eps = 1e-6
    box_sizes = torch.sqrt((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size + eps)
    )
    # clamp level to (min, max), in case the box size is too large or too small
    # for the available feature maps
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level


def va_roi_align(features, boxes, dists, side_size, feature_scales, sample_ratio):
    assert sample_ratio <= 1
    num_boxes = boxes.shape[0]
    num_va_sampled = int(side_size[0] * side_size[1] * sample_ratio)
    num_uniform_sampled = int(side_size[0] * side_size[1]) - num_va_sampled

    va_features = []
    for bid, feature in enumerate(features):
        h, w = feature.shape[-2:]
        scale = torch.tensor([w, h], device=feature.device) / feature_scales
        batch_inds = torch.where(boxes[:, 0] == bid)[0]
        batch_dists = dists[batch_inds, 1:]
        batch_mean = batch_dists[:, :2].unsqueeze(1)
        batch_lstd = batch_dists[:, 2:].unsqueeze(1)
        point_va_coords = torch.sigmoid(batch_mean + batch_lstd.exp() * 
                                torch.randn_like(batch_mean.repeat(1, num_va_sampled, 1)))
        point_uniform_coords = torch.rand(len(batch_inds), num_uniform_sampled, 2, device=feature.device)
        point_sampling_coords = torch.cat((point_va_coords, point_uniform_coords), dim=1)
        point_coords_wrt_image = get_point_coords_wrt_image(boxes[batch_inds, 1:], point_sampling_coords)
        point_coords_scaled = point_coords_wrt_image / scale
        va_feature = point_sample(feature, point_coords_scaled, align_corners=False)
        va_features.append(va_feature)
    return torch.cat(va_features, dim=0).reshape(num_boxes, -1, side_size[0], side_size[1])


def va_roi_pooler(fpn_fms, rois, dists, stride, pool_shape, va_sample_ratio):
    assert len(fpn_fms) == len(stride)
    max_level = int(math.log2(stride[-1]))
    min_level = int(math.log2(stride[0]))
    assert (len(stride) == max_level - min_level + 1)
    level_assignments = assign_boxes_to_levels(rois, min_level, max_level, 224, 4)
    dtype, device = fpn_fms[0].dtype, fpn_fms[0].device
    output = torch.zeros((len(rois), fpn_fms[0].shape[1], pool_shape[0], pool_shape[1]),
            dtype=dtype, device=device)
    for level, (fm_level, scale_level) in enumerate(zip(fpn_fms, stride)):
        inds = torch.nonzero(level_assignments == level, as_tuple=False).squeeze(1)
        rois_level = rois[inds]
        dists_level = dists[inds]
        gt_roi_ind = torch.where(dists_level[:, 1:].sum(1) == 0)[0]
        pr_roi_ind = torch.where(dists_level[:, 1:].sum(1) != 0)[0]
        output[gt_roi_ind] = roi_align(fm_level, rois_level[gt_roi_ind], pool_shape, spatial_scale=1.0/scale_level,
                sampling_ratio=-1, aligned=True)
        output[pr_roi_ind] = va_roi_align(fm_level, rois_level[pr_roi_ind], dists_level[pr_roi_ind], pool_shape, 
                1.0/scale_level, va_sample_ratio)
    return output

