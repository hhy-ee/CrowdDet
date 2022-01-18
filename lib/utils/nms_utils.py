import numpy as np
import pdb
import torch

def set_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    def _overlap(det_boxes, basement, others):
        eps = 1e-8
        x1_basement, y1_basement, x2_basement, y2_basement \
                = det_boxes[basement, 0], det_boxes[basement, 1], \
                  det_boxes[basement, 2], det_boxes[basement, 3]
        x1_others, y1_others, x2_others, y2_others \
                = det_boxes[others, 0], det_boxes[others, 1], \
                  det_boxes[others, 2], det_boxes[others, 3]
        areas_basement = (x2_basement - x1_basement) * (y2_basement - y1_basement)
        areas_others = (x2_others - x1_others) * (y2_others - y1_others)
        xx1 = np.maximum(x1_basement, x1_others)
        yy1 = np.maximum(y1_basement, y1_others)
        xx2 = np.minimum(x2_basement, x2_others)
        yy2 = np.minimum(y2_basement, y2_others)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas_basement + areas_others - inter + eps)
        return ovr
    scores = dets[:, 4]
    order = np.argsort(-scores)
    dets = dets[order]

    numbers = dets[:, -1]
    keep = np.ones(len(dets)) == 1
    ruler = np.arange(len(dets))
    while ruler.size>0:
        basement = ruler[0]
        ruler=ruler[1:]
        num = numbers[basement]
        # calculate the body overlap
        overlap = _overlap(dets[:, :4], basement, ruler)
        indices = np.where(overlap > thresh)[0]
        loc = np.where(numbers[ruler][indices] == num)[0]
        # the mask won't change in the step
        mask = keep[ruler[indices][loc]]#.copy()
        keep[ruler[indices]] = False
        keep[ruler[indices][loc][mask]] = True
        ruler[~keep[ruler]] = -1
        ruler = ruler[ruler>0]
    keep = keep[np.argsort(order)]
    return keep

def set_cpu_kl_nms(dets, thresh):
    """Pure Python NMS baseline."""
    def _overlap(det_boxes, basement, others):
        eps = 1e-8
        x1_basement, y1_basement, x2_basement, y2_basement \
                = det_boxes[basement, 0], det_boxes[basement, 1], \
                  det_boxes[basement, 2], det_boxes[basement, 3]
        x1_others, y1_others, x2_others, y2_others \
                = det_boxes[others, 0], det_boxes[others, 1], \
                  det_boxes[others, 2], det_boxes[others, 3]
        areas_basement = (x2_basement - x1_basement) * (y2_basement - y1_basement)
        areas_others = (x2_others - x1_others) * (y2_others - y1_others)
        xx1 = np.maximum(x1_basement, x1_others)
        yy1 = np.maximum(y1_basement, y1_others)
        xx2 = np.minimum(x2_basement, x2_others)
        yy2 = np.minimum(y2_basement, y2_others)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas_basement + areas_others - inter + eps)
        return ovr
    scores = dets[:, 4]
    order = np.argsort(-scores)
    dets = dets[order]

    numbers = dets[:, -1]
    probs = dets[:, 6]
    keep = np.ones(len(dets)) == 1
    ruler = np.arange(len(dets))
    while ruler.size>0:
        basement = ruler[0]
        ruler=ruler[1:]
        num = numbers[basement]
        # calculate the body overlap
        overlap = _overlap(dets[:, :4], basement, ruler)
        indices = np.where(overlap > thresh)[0]
        loc = np.where(numbers[ruler][indices] == num)[0]
        # the mask won't change in the step
        mask = keep[ruler[indices][loc]]#.copy()
        keep[ruler[indices]] = False
        keep[ruler[indices][loc][mask]] = True
        # if loc.shape[0] != 0:
        #     if np.abs(probs[basement] - probs[ruler[indices][loc]]) < 0.01:
        #         keep[ruler[indices][loc][mask]] = False
        ruler[~keep[ruler]] = -1
        ruler = ruler[ruler>0]
    keep = keep[np.argsort(order)]
    return keep

def cpu_nms(dets, base_thr):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)

    keep = []
    eps = 1e-8
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + eps)

        inds = np.where(ovr <= base_thr)[0]
        indices = np.where(ovr > base_thr)[0]
        order = order[inds + 1]
    return np.array(keep)

def cpu_kl_nms(dets, base_thr):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    lstd = dets[:, 6:8].mean(1)
    scores = scores * (1 - lstd.sigmoid())
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)

    keep = []
    real_keep = []
    eps = 1e-8
    while len(order) > 0:
        i = order[0]

        # kl_nms before normalnms
        # xx1 = np.maximum(x1[i], x1[order])
        # yy1 = np.maximum(y1[i], y1[order])
        # xx2 = np.minimum(x2[i], x2[order])
        # yy2 = np.minimum(y2[i], y2[order])

        # w = np.maximum(0.0, xx2 - xx1)
        # h = np.maximum(0.0, yy2 - yy1)
        # inter = w * h
        # ovr = inter / (areas[i] + areas[order] - inter + eps)
        # candidate_inds = np.where(ovr >= 0.9)[0]
        # candidate_lstd = lstd[order[candidate_inds]]
        # i = order[candidate_inds[np.argmin(candidate_lstd)]]

        # normal nms
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + eps)

        inds = np.where(ovr <= base_thr)[0]
        supp_inds = np.where(ovr > base_thr)[0]
        order = order[inds + 1]

        # ambi_inds = np.where((ovr <= base_thr) * (ovr > base_thr - 0.1))[0]
        # ambi_order = order[ambi_inds + 1]
        # supp_order = order[supp_inds + 1]
        # if ambi_order.shape[0] != 0 and supp_order.shape[0] != 0:    
        #     exsupp_idx = np.where(lstd[ambi_order] < lstd[supp_order].mean())[0]
        #     order = np.where(order == ambi_order[exsupp_idx])
            
    return np.array(keep)

def rpn_kl_nms(pred_box, box_lstd, box_scr, base_thr):
    """Pure Python NMS baseline."""
    x1 = pred_box[:, 0]
    y1 = pred_box[:, 1]
    x2 = pred_box[:, 2]
    y2 = pred_box[:, 3]
    scores = box_scr
    lstd = box_lstd.mean(dim=1)

    areas = (x2 - x1) * (y2 - y1)
    order = torch.argsort(-scores)

    keep = []
    real_keep = []
    eps = 1e-8
    while len(order) > 0:
        i = order[0]
        keep.append(i.reshape(1))
        xx1 = torch.max(x1[i], x1[order])
        yy1 = torch.max(y1[i], y1[order])
        xx2 = torch.min(x2[i], x2[order])
        yy2 = torch.min(y2[i], y2[order])

        w = torch.max(torch.zeros_like(xx2), xx2 - xx1)
        h = torch.max(torch.zeros_like(yy2), yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order] - inter + eps)

        inds = torch.where(ovr <= base_thr)[0]
        supp_inds = torch.where(ovr > base_thr)[0]
        real_keep.append(order[supp_inds[lstd[supp_inds].argmax()].reshape(1)])
        order = order[inds]
    return torch.cat(keep)

def new_rpn_kl_nms(pred_box, box_lstd, box_scr, base_thr):
    _, idx = box_scr.sort(0, descending=True)
    boxes_idx = pred_box[idx]
    lstd_idx = box_lstd[idx].mean(dim=1)
    iou = box_overlap_opr(boxes_idx, boxes_idx).triu_(diagonal=1)
    B = iou
    while 1:
        A = B
        maxA, _ = torch.max(A, dim=0)
        E = (maxA <= base_thr).float().unsqueeze(1).expand_as(A)
        B = iou.mul(E)
        if A.equal(B) == True:
            break
    keep_idx = idx[maxA <= base_thr]
    lstd = torch.where(
        iou[keep_idx, :] > base_thr,
        lstd_idx[None, :],
        20 * torch.ones(1, dtype=pred_box.dtype, device=pred_box.device)
    )
    real_keep_idx = torch.min(lstd, dim=1).indices
    del A,B,iou

    # for i in keep_idx:
    #     supp = torch.where(iou[i, :] > base_thr)[0]
    #     supp_lstd = 1
    # keep1 = rpn_kl_nms(pred_box, box_lstd, box_scr, base_thr)
    return real_keep_idx

def box_overlap_opr(box1, box2):
    assert box1.ndim == 2
    assert box2.ndim == 2
    area_box1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area_box2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    width_height = torch.min(box1[:, None, 2:], box2[:, 2:]) - \
                    torch.max(box1[:, None, :2], box2[:, :2])
    width_height.clamp_(min=0)
    inter = width_height.prod(dim=2)
    del width_height
    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area_box1[:, None] + area_box2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou

def nms_for_plot(dets, base_thr):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)

    keep = []
    supp = []
    eps = 1e-8
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + eps)

        inds = np.where(ovr <= base_thr)[0]
        # indices = np.where((ovr <= base_thr) * (ovr > base_thr-0.1))[0]
        indices = np.where( ovr > base_thr)[0]
        supp.append(order[indices + 1])
        order = order[inds + 1]
    return np.array(keep), supp

def _test():
    box1 = np.array([33,45,145,230,0.7])[None,:]
    box2 = np.array([44,54,123,348,0.8])[None,:]
    box3 = np.array([88,12,340,342,0.65])[None,:]
    boxes = np.concatenate([box1,box2,box3],axis = 0)
    nms_thresh = 0.5
    keep = py_cpu_nms(boxes,nms_thresh)
    alive_boxes = boxes[keep]

if __name__=='__main__':
    _test()
