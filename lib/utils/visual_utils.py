import os
import json
import numpy as np
import cv2
from scipy.stats import beta

color = {'green':(0,255,0),
        'blue':(255,165,0),
        'dark red':(0,0,139),
        'red':(0, 0, 255),
        'dark slate blue':(139,61,72),
        'aqua':(255,255,0),
        'brown':(42,42,165),
        'deep pink':(147,20,255),
        'fuchisia':(255,0,255),
        'yello':(0,238,238),
        'orange':(0,165,255),
        'saddle brown':(19,69,139),
        'black':(0,0,0),
        'white':(255,255,255)}

class_names = ['background', 'person']

def draw_boxes(img, boxes, scores=None, tags=None, line_thick=1, line_color='white'):
    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        one_box = boxes[i]
        one_box = np.array([max(one_box[0], 0), max(one_box[1], 0),
                    min(one_box[2], width - 1), min(one_box[3], height - 1)])
        x1,y1,x2,y2 = np.array(one_box[:4]).astype(int)
        cv2.rectangle(img, (x1,y1), (x2,y2), color[line_color], line_thick)
        if scores is not None:
            text = "{} {:.3f}".format(tags[i], scores[i])
            cv2.putText(img, text, (x1, y1 - 7), cv2.FONT_ITALIC, 0.5, color[line_color], line_thick)
    return img


def draw_supp_boxes(img, boxes, supp_boxes, args, line_thick=1, line_color=('red','green')):
    width = img.shape[1]
    height = img.shape[0]
    boxes_scr = boxes[:, 4]
    boxes_loc = boxes[:, :4]
    boxes_lstd = boxes[:, 6:].mean(1)
    supp_boxes_scr = [supp_box[:, 4] for supp_box in supp_boxes]
    supp_boxes_loc = [supp_box[:, :4] for supp_box in supp_boxes]
    supp_boxes_lstd = [supp_box[:, 6:].mean(1) for supp_box in supp_boxes]
    for i in range(len(boxes)):
        plot_img = img
        one_box = boxes_loc[i]
        one_box = np.array([max(one_box[0], 0), max(one_box[1], 0),
                    min(one_box[2], width - 1), min(one_box[3], height - 1)])
        x1,y1,x2,y2 = np.array(one_box[:4]).astype(int)
        cv2.rectangle(plot_img, (x1,y1), (x2,y2), color[line_color[0]], line_thick)
        text = "{:.3f} {:.3f}".format(boxes_scr[i], boxes_lstd[i])
        cv2.putText(plot_img, text, (x1, y1 - 7), cv2.FONT_ITALIC, 0.5, color['black'], line_thick)
        for j in range(3):
            high_scr_idx = np.where(supp_boxes_scr[i] > boxes_scr[i] - (j+1)/10)[0]
            high_std_idx = np.argmax(supp_boxes_lstd[i][high_scr_idx])
            supp_box = supp_boxes_loc[i][high_std_idx]
            supp_box = np.array([max(supp_box[0], 0), max(supp_box[1], 0),
                        min(supp_box[2], width - 1), min(supp_box[3], height - 1)])
            x1,y1,x2,y2 = np.array(supp_box[:4]).astype(int)
            cv2.rectangle(plot_img, (x1,y1), (x2,y2), color[line_color[1]], line_thick)
            text = "{:.3f} {:.3f}".format(supp_boxes_scr[i][high_std_idx], supp_boxes_lstd[i][high_std_idx])
            cv2.putText(plot_img, text, (x1, y1 - 7), cv2.FONT_ITALIC, 0.5, color['black'], line_thick)
        name = args.img_path.split('/')[-1].split('.')[-2]
        fpath = 'outputs/{}_gt{:d}.png'.format(name, i)
        cv2.imwrite(fpath, plot_img)


def draw_my_boxes(img, boxes, plot_box_info, args, line_thick=1, line_color=('red','green')):
    width = img.shape[1]
    height = img.shape[0]
    (supp_boxes, inf_result, gt_boxes, gt_matched) = plot_box_info
    boxes_scr = boxes[:, 4]
    boxes_loc = boxes[:, :4]
    boxes_lstd = boxes[:, 6:8].mean(1)
    boxes_anchor = boxes[:, 8:12]
    supp_boxes_scr = [supp_box[:, 4] for supp_box in supp_boxes]
    supp_boxes_loc = [supp_box[:, :4] for supp_box in supp_boxes]
    supp_boxes_lstd = [supp_box[:, 6:8].mean(1) for supp_box in supp_boxes]

    if_tp = np.where(np.array([re[1] for re in inf_result]) == 0)[0]
    dt_match = np.array([re[2] for re in inf_result])
    dt_iou = np.array([re[3] for re in inf_result])

    for i in range(len(if_tp)):
        plot_img = img
        gt_idx_for_fp = dt_match[if_tp[i]]
        tp_dt_idx = int(gt_matched[gt_idx_for_fp] - 1)
        # plot tp
        one_box = boxes_loc[tp_dt_idx]
        one_anchor = boxes_anchor[tp_dt_idx]
        one_box = np.array([max(one_box[0], 0), max(one_box[1], 0),
                    min(one_box[2], width - 1), min(one_box[3], height - 1)])
        x1,y1,x2,y2 = np.array(one_box[:4]).astype(int)
        a1,b1,a2,b2 = np.array(one_anchor[:4]).astype(int)
        cv2.rectangle(plot_img, (x1,y1), (x2,y2), color[line_color[0]], line_thick)
        cv2.rectangle(plot_img, (a1,b1), (a2,b2), color[line_color[1]], line_thick)
        text = "{:.3f} {:.3f}".format(dt_iou[tp_dt_idx], boxes_lstd[tp_dt_idx])
        cv2.putText(plot_img, text, (x1, y1 - 7), cv2.FONT_ITALIC, 0.5, color['black'], line_thick)

        # plot fp
        one_box = boxes_loc[if_tp[i]]
        one_anchor = boxes_anchor[if_tp[i]]
        one_box = np.array([max(one_box[0], 0), max(one_box[1], 0),
                    min(one_box[2], width - 1), min(one_box[3], height - 1)])
        x1,y1,x2,y2 = np.array(one_box[:4]).astype(int)
        a1,b1,a2,b2 = np.array(one_anchor[:4]).astype(int)
        cv2.rectangle(plot_img, (x1,y1), (x2,y2), color['black'], line_thick)
        cv2.rectangle(plot_img, (a1,b1), (a2,b2), color['blue'], line_thick)
        text = "{:.3f} {:.3f}".format(dt_iou[if_tp[i]], boxes_lstd[if_tp[i]])
        cv2.putText(plot_img, text, (x1, y1 - 7), cv2.FONT_ITALIC, 0.5, color['black'], line_thick)


        # for j in range(3):
        #     high_scr_idx = np.where(supp_boxes_scr[i] > boxes_scr[i] - (j+1)/10)[0]
        #     high_std_idx = np.argmax(supp_boxes_lstd[i][high_scr_idx])
        #     supp_box = supp_boxes_loc[i][high_std_idx]
        #     supp_box = np.array([max(supp_box[0], 0), max(supp_box[1], 0),
        #                 min(supp_box[2], width - 1), min(supp_box[3], height - 1)])
        #     x1,y1,x2,y2 = np.array(supp_box[:4]).astype(int)
        #     cv2.rectangle(plot_img, (x1,y1), (x2,y2), color[line_color[1]], line_thick)
        #     text = "{:.3f} {:.3f}".format(supp_boxes_scr[i][high_std_idx], supp_boxes_lstd[i][high_std_idx])
        #     cv2.putText(plot_img, text, (x1, y1 - 7), cv2.FONT_ITALIC, 0.5, color['black'], line_thick)
        name = args.img_path.split('/')[-1].split('.')[-2]
        fpath = 'outputs/{}_gt{:d}.png'.format(name, i)
        cv2.imwrite(fpath, plot_img)


def draw_dists(img, boxes, dists, va_beta, scores=None):
    width = img.shape[1]
    height = img.shape[0]
    seman_map = np.zeros((int(height), int(width)))
    for i in range(len(boxes)):
        # if scores[i] < 0.7:
            one_box = boxes[i]
            one_box = np.array([max(one_box[0], 0), max(one_box[1], 0),
                        min(one_box[2], width - 1), min(one_box[3], height - 1)])
            x1,y1,x2,y2 = np.array(one_box[:4]).astype(int)
            dx = beta_dist(x2-x1, dists[i, :2], va_beta)
            dy = beta_dist(y2-y1, dists[i, 2:], va_beta)
            mask = np.multiply(dy, np.transpose(dx))
            seman_map[y1:y2, x1:x2] = np.maximum(seman_map[y1:y2, x1:x2], mask)
            norm_img = np.asarray(seman_map*255, dtype=np.uint8)
            heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
            # heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
            img = cv2.addWeighted(img, 0.9, heat_img, 0.1, 0)
    return img

def beta_dist(kernel, beta_para, va_beta):
    beta_a = sigmoid(beta_para[0]) * va_beta[0] + va_beta[1]
    beta_b = sigmoid(beta_para[1]) * va_beta[0] + va_beta[1]
    dist = beta(beta_a, beta_b)
    dist_x = (np.arange(kernel) + 1) / (kernel + 1)
    dist_y = np.reshape(dist.pdf(dist_x), (-1, 1))
    return dist_y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inference_result(gt_path, img_path, pred_boxes, im_info):
    pred_boxes_lstd = np.mean(pred_boxes[:,6:8], axis=1, keepdims=True)
    pred_boxes = np.concatenate([pred_boxes[:,:6], pred_boxes_lstd, pred_boxes[:,8:]], axis=1)
    with open(gt_path, "r") as f:
        lines = f.readlines()
        records = [json.loads(line.strip('\n')) for line in lines]
    for record in records:
        if record["ID"] == img_path.split('/')[-1].split('.')[0]:
            h, w=int(im_info[0, -3]), int(im_info[0, -2])
            gt_bboxes = load_gt_boxes(record, 'gtboxes')
            pred_boxes, gt_bboxes = clip_all_boader(pred_boxes, gt_bboxes, h, w)
            score_list, gt_list, gt_matched = compare_caltech(pred_boxes, gt_bboxes, 0.5)
            score_list.sort(key=lambda x: x[0][4], reverse=True)
    return score_list, gt_list, gt_matched

def compare_caltech(dtboxes, gtboxes, thres):
    """
    :meth: match the detection results with the groundtruth by Caltech matching strategy
    :param thres: iou threshold
    :type thres: float
    :return: a list of tuples (dtbox, imageID), in the descending sort of dtbox.score
    """
    dt_matched = np.zeros(len(dtboxes))
    gt_matched = np.zeros(len(gtboxes))

    dtboxes = np.array(sorted(dtboxes, key=lambda x: x[4], reverse=True))
    gtboxes = np.array(sorted(gtboxes, key=lambda x: x[4], reverse=True))
    if len(dtboxes):
        overlap_iou = box_overlap_opr(dtboxes, gtboxes, True)
        overlap_ioa = box_overlap_opr(dtboxes, gtboxes, False)
    else:
        return list()

    scorelist = list()
    for i, dt in enumerate(dtboxes):
        maxpos = -1
        maxiou = thres
        for j, gt in enumerate(gtboxes):
            if gt_matched[j] >= 1:
                continue
            if gt[-1] > 0:
                overlap = overlap_iou[i][j]
                if overlap > maxiou:
                    maxiou = overlap
                    maxpos = j
            else:
                if maxpos >= 0:
                    break
                else:
                    overlap = overlap_ioa[i][j]
                    if overlap > thres:
                        maxiou = overlap
                        maxpos = j
        if maxpos >= 0:
            if gtboxes[maxpos, -1] > 0:
                gt_matched[maxpos] = int(1 + i)
                dt_matched[i] = maxpos+1
                maxj = overlap_iou[i].argmax()
                scorelist.append((dt, 1, maxj, overlap_iou[i][maxj]))
            else:
                dt_matched[i] = -1
                maxj = overlap_iou[i].argmax()
                scorelist.append((dt, -1, maxj, overlap_iou[i][maxj]))
        else:
            dt_matched[i] = 0
            maxj = overlap_iou[i].argmax()
            scorelist.append((dt, 0, maxj, overlap_iou[i][maxj]))
    return scorelist, gtboxes, gt_matched

def box_overlap_opr(dboxes:np.ndarray, gboxes:np.ndarray, if_iou):
    eps = 1e-6
    assert dboxes.shape[-1] >= 4 and gboxes.shape[-1] >= 4
    N, K = dboxes.shape[0], gboxes.shape[0]
    dtboxes = np.tile(np.expand_dims(dboxes, axis = 1), (1, K, 1))
    gtboxes = np.tile(np.expand_dims(gboxes, axis = 0), (N, 1, 1))

    iw = np.minimum(dtboxes[:,:,2], gtboxes[:,:,2]) - np.maximum(dtboxes[:,:,0], gtboxes[:,:,0])
    ih = np.minimum(dtboxes[:,:,3], gtboxes[:,:,3]) - np.maximum(dtboxes[:,:,1], gtboxes[:,:,1])
    inter = np.maximum(0, iw) * np.maximum(0, ih)

    dtarea = (dtboxes[:,:,2] - dtboxes[:,:,0]) * (dtboxes[:,:,3] - dtboxes[:,:,1])
    if if_iou:
        gtarea = (gtboxes[:,:,2] - gtboxes[:,:,0]) * (gtboxes[:,:,3] - gtboxes[:,:,1]) 
        ious = inter / (dtarea + gtarea - inter + eps)
    else:
        ious = inter / (dtarea + eps)
    return ious

def load_gt_boxes(dict_input, key_name):
    assert key_name in dict_input
    if len(dict_input[key_name]) < 1:
        return np.empty([0, 5])
    head_bbox = []
    body_bbox = []
    for rb in dict_input[key_name]:
        if rb['tag'] in class_names:
            body_tag = class_names.index(rb['tag'])
            head_tag = 1
        else:
            body_tag = -1
            head_tag = -1
        if 'extra' in rb:
            if 'ignore' in rb['extra']:
                if rb['extra']['ignore'] != 0:
                    body_tag = -1
                    head_tag = -1
        if 'head_attr' in rb:
            if 'ignore' in rb['head_attr']:
                if rb['head_attr']['ignore'] != 0:
                    head_tag = -1
        # head_bbox.append(np.hstack((rb['hbox'], head_tag)))
        body_bbox.append(np.hstack((rb['fbox'], body_tag)))
    # head_bbox = np.array(head_bbox)
    # head_bbox[:, 2:4] += head_bbox[:, :2]
    body_bbox = np.array(body_bbox)
    body_bbox[:, 2:4] += body_bbox[:, :2]
    # return body_bbox, head_bbox
    return body_bbox

def clip_all_boader(dtboxes, gtboxes, _height, _width):
    def _clip_boundary(boxes,height,width):
        assert boxes.shape[-1]>=4
        boxes[:,0] = np.minimum(np.maximum(boxes[:,0],0), width - 1)
        boxes[:,1] = np.minimum(np.maximum(boxes[:,1],0), height - 1)
        boxes[:,2] = np.maximum(np.minimum(boxes[:,2],width), 0)
        boxes[:,3] = np.maximum(np.minimum(boxes[:,3],height), 0)
        return boxes

    assert dtboxes.shape[-1]>=4
    assert gtboxes.shape[-1]>=4
    dtboxes = _clip_boundary(dtboxes, _height, _width)
    gtboxes = _clip_boundary(gtboxes, _height, _width)
    return dtboxes, gtboxes