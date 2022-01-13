import os
import json
import numpy as np
import cv2
from scipy.stats import beta
import matplotlib.pyplot as plt

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
plot_color = ['red', 'orange', 'yellow', 'green', 'blue']
score_level = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

def draw_boxes(img, boxes, scores=None, tags=None, line_thick=1, line_color='white', putText=False):
    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        one_box = boxes[i]
        one_box = np.array([max(one_box[0], 0), max(one_box[1], 0),
                    min(one_box[2], width - 1), min(one_box[3], height - 1)])
        x1,y1,x2,y2 = np.array(one_box[:4]).astype(int)
        cv2.rectangle(img, (x1,y1), (x2,y2), color[line_color], line_thick)
        if putText:
            text = "{} {:.3f}".format(tags[i], scores[i])
            cv2.putText(img, text, (x1, y1 - 7), cv2.FONT_ITALIC, 0.5, color[line_color], line_thick)
    return img


def draw_mip_for_set_kl(img, boxes, boxes_info, args, line_thick=1, line_color=('green','blue','red')):
    width = img.shape[1]
    height = img.shape[0]
    (inf_result, gt_boxes, gt_matched) = boxes_info
    boxes_scr = boxes[:, 4]
    boxes_loc = boxes[:, :4]
    boxes_lstd = boxes[:, 6:10].mean(1)
    boxes_num = boxes[:, 10]

    ruler = np.arange(len(boxes_num))
    i=0
    while ruler.size>0:
        basement = ruler[0]
        ruler = ruler[1:]
        loc = np.where(boxes_num[ruler] == boxes_num[basement])[0]
        if loc.shape[0] != 0:
            plot_img = img.copy()
            box_1 = boxes_loc[basement]
            box_2 = boxes_loc[ruler[loc].squeeze()]
            box_1 = np.array([max(box_1[0], 0), max(box_1[1], 0),
                    min(box_1[2], width - 1), min(box_1[3], height - 1)])
            box_2 = np.array([max(box_2[0], 0), max(box_2[1], 0),
                    min(box_2[2], width - 1), min(box_2[3], height - 1)])
            gt_1 = gt_boxes[inf_result[basement][2]]
            gt_2 = gt_boxes[inf_result[ruler[loc].squeeze()][2]]
            gt_1 = np.array([max(gt_1[0], 0), max(gt_1[1], 0),
                    min(gt_1[2], width - 1), min(gt_1[3], height - 1)])
            gt_2 = np.array([max(gt_2[0], 0), max(gt_2[1], 0),
                    min(gt_2[2], width - 1), min(gt_2[3], height - 1)])
            is_fp_1 = inf_result[basement][1]
            is_fp_2 = inf_result[ruler[loc].squeeze()][1]

            x11,y11,x12,y12 = np.array(box_1[:4]).astype(int)
            x21,y21,x22,y22 = np.array(box_2[:4]).astype(int)

            a11,b11,a12,b12 = np.array(gt_1[:4]).astype(int)
            a21,b21,a22,b22 = np.array(gt_2[:4]).astype(int)

            cv2.rectangle(plot_img, (x11,y11), (x12,y12), color[line_color[is_fp_1]], line_thick)
            cv2.rectangle(plot_img, (x21,y21), (x22,y22), color[line_color[is_fp_2]], line_thick)
            cv2.rectangle(plot_img, (a11,b11), (a12,b12), color[line_color[2]], line_thick)
            cv2.rectangle(plot_img, (a21,b21), (a22,b22), color[line_color[2]], line_thick)

            text1 = "{:.3f} {:.3f}".format(boxes_scr[basement], boxes_lstd[basement])
            text2 = "{:.3f} {:.3f}".format(boxes_scr[ruler[loc].squeeze()], boxes_lstd[ruler[loc].squeeze()])
            cv2.putText(plot_img, text1, (x11, y11 - 7), cv2.FONT_ITALIC, 0.5, color['white'], line_thick)
            cv2.putText(plot_img, text2, (x21, y21 - 7), cv2.FONT_ITALIC, 0.5, color['white'], line_thick)
            name = args.img_path.split('/')[-1].split('.')[-2]
            fpath = 'outputs/{}_{:.1f}.png'.format(name, i)
            cv2.imwrite(fpath, plot_img)
            ruler = np.delete(ruler, loc)
            i = i+1


def draw_fn_boxes(img, boxes, plot_box_info, args, line_thick=1, line_color=('red','green')):
    width = img.shape[1]
    height = img.shape[0]
    (supp_boxes, inf_result, gt_boxes, gt_matched) = plot_box_info
    # plot fn & ign
    gt_fn = gt_boxes[np.where(gt_matched == 0)[0]]
    gt_fn_ins = gt_fn[np.where(gt_fn[:,-1]==1)[0]]
    gt_fn_ign = gt_fn[np.where(gt_fn[:,-1]==-1)[0]]
    plot_img = img.copy()
    for i in range(len(gt_fn_ins)):
        one_box = gt_fn_ins[i]
        one_box = np.array([max(one_box[0], 0), max(one_box[1], 0),
                    min(one_box[2], width - 1), min(one_box[3], height - 1)])
        x1,y1,x2,y2 = np.array(one_box[:4]).astype(int)
        cv2.rectangle(plot_img, (x1,y1), (x2,y2), color[line_color[0]], line_thick)
    for i in range(len(gt_fn_ign)):
        one_box = gt_fn_ign[i]
        one_box = np.array([max(one_box[0], 0), max(one_box[1], 0),
                    min(one_box[2], width - 1), min(one_box[3], height - 1)])
        x1,y1,x2,y2 = np.array(one_box[:4]).astype(int)
        cv2.rectangle(plot_img, (x1,y1), (x2,y2), color[line_color[1]], line_thick)
    name = args.img_path.split('/')[-1].split('.')[-2]
    fpath = 'outputs/{}_fn.png'.format(name)
    cv2.imwrite(fpath, plot_img)
    # plot gt
    plot_img = img.copy()
    gt = gt_boxes[np.where(gt_boxes[:,-1]==1)[0]][:,:4]
    for i in range(len(gt)):
        one_box = gt[i]
        one_box = np.array([max(one_box[0], 0), max(one_box[1], 0),
                    min(one_box[2], width - 1), min(one_box[3], height - 1)])
        x1,y1,x2,y2 = np.array(one_box[:4]).astype(int)
        cv2.rectangle(plot_img, (x1,y1), (x2,y2), color[line_color[0]], line_thick)
    name = args.img_path.split('/')[-1].split('.')[-2]
    fpath = 'outputs/{}_gt.png'.format(name)
    cv2.imwrite(fpath, plot_img)


def draw_supp_scatter(img, boxes, plot_box_info, args, line_thick=1):
    width = img.shape[1]
    height = img.shape[0]
    (supp_boxes, inf_result, gt_boxes, gt_matched) = plot_box_info
    boxes_scr = boxes[:, 4]
    boxes_loc = boxes[:, :4]
    boxes_std = boxes[:, 6:8].mean(1)
    supp_boxes_scr = [supp_box[:, 4] for supp_box in supp_boxes]
    supp_boxes_loc = [supp_box[:, :4] for supp_box in supp_boxes]
    supp_boxes_lstd = [supp_box[:, 6:8].mean(1) for supp_box in supp_boxes]

    is_fp = np.where(np.array([re[1] for re in inf_result]) == 0)[0]
    fp_scrs = boxes_scr[is_fp]
    fp_lstds = boxes_std[is_fp]
    fp_loc = boxes_loc[is_fp]
    fp_gts = np.array([re[2] for re in inf_result])[is_fp]
    fp_ious = np.array([re[3] for re in inf_result])[is_fp]
    for i in range(len(score_level)-1):
        scatter_fp = np.where((fp_scrs<=score_level[i]) * (fp_scrs>score_level[i+1]))[0]
        for j in range(len(scatter_fp)):
            ign = gt_boxes[fp_gts[scatter_fp[j]]][-1]
            iou = fp_ious[scatter_fp[j]]
            if ign == 1 and iou > 1e-2:
                dt_idx = int(gt_matched[fp_gts[scatter_fp[j]]]) - 1
                if supp_boxes_scr[dt_idx].shape[0] != 0 and dt_idx >= 0:
                    supp_scr = supp_boxes_scr[dt_idx]
                    supp_lstd = supp_boxes_lstd[dt_idx]
                    keep_box = np.expand_dims(boxes_loc[dt_idx], axis=0)
                    supp_iou = box_overlap_opr(keep_box, supp_boxes_loc[dt_idx], True)
                    plt.scatter(supp_lstd, supp_scr, s=10, marker='o', c=plot_color[i])
                    fp_scr = fp_scrs[scatter_fp[j]]
                    fp_lstd = fp_lstds[scatter_fp[j]]
                    fp_box = np.expand_dims(fp_loc[scatter_fp[j]], axis=0)
                    fp_iou = box_overlap_opr(keep_box, fp_box, True)
                    plt.scatter(fp_lstd, fp_scr, s=15, marker='^', c=plot_color[i])
    name = args.img_path.split('/')[-1].split('.')[-2]
    fpath = 'outputs/{}_var_scr.png'.format(name)
    plt.savefig(fpath)
    plt.show()
        


def draw_fp_boxes(img, boxes, plot_box_info, args, line_thick=1, line_color=('green','blue','red')):
    width = img.shape[1]
    height = img.shape[0]
    (supp_boxes, inf_result, gt_boxes, gt_matched) = plot_box_info
    boxes_scr = boxes[:, 4]
    boxes_loc = boxes[:, :4]
    boxes_lstd = boxes[:, 6:8].mean(1)
    boxes_anchor = boxes[:, 8:12]

    is_fp = np.where(np.array([re[1] for re in inf_result]) == 0)[0]
    fp_scr = boxes_scr[is_fp]
    fp_gt = np.array([re[2] for re in inf_result])[is_fp]
    fp_iou = np.array([re[3] for re in inf_result])[is_fp]
    for i in range(len(score_level)-1):
        plot_img = img.copy()
        plot_fp = np.where((fp_scr<=score_level[i]) * (fp_scr>score_level[i+1]))[0]
        for j in range(len(plot_fp)):
            one_box = boxes_loc[is_fp[plot_fp[j]]]
            one_anchor = boxes_anchor[is_fp[plot_fp[j]]]
            one_gt = gt_boxes[fp_gt[plot_fp[j]]]
            one_box = np.array([max(one_box[0], 0), max(one_box[1], 0),
                    min(one_box[2], width - 1), min(one_box[3], height - 1)])
            one_gt = np.array([max(one_gt[0], 0), max(one_gt[1], 0),
                    min(one_gt[2], width - 1), min(one_gt[3], height - 1), one_gt[4]])
            x1,y1,x2,y2 = np.array(one_box[:4]).astype(int)
            a1,b1,a2,b2 = np.array(one_anchor[:4]).astype(int)
            m1,n1,m2,n2= np.array(one_gt[:4]).astype(int)
            cv2.rectangle(plot_img, (x1,y1), (x2,y2), color[line_color[0]], line_thick)
            cv2.rectangle(plot_img, (a1,b1), (a2,b2), color[line_color[1]], line_thick)
            if one_gt[-1] == 1 and fp_iou[plot_fp[j]] > 1e-2:
                cv2.rectangle(plot_img, (m1,n1), (m2,n2), color[line_color[2]], line_thick)
            text = "{:.3f} {:.3f}".format(fp_iou[plot_fp[j]], boxes_lstd[is_fp[plot_fp[j]]])
            cv2.putText(plot_img, text, (x1, y1 - 7), cv2.FONT_ITALIC, 0.5, color['white'], line_thick)
        name = args.img_path.split('/')[-1].split('.')[-2]
        fpath = 'outputs/{}_fp{:.1f}.png'.format(name, score_level[i+1])
        cv2.imwrite(fpath, plot_img)


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

def inference_result_for_set_kl(gt_path, img_path, pred_boxes, im_info):
    pred_boxes_lstd = np.mean(pred_boxes[:,6:10], axis=1, keepdims=True)
    pred_boxes = np.concatenate([pred_boxes[:,:6], pred_boxes_lstd, pred_boxes[:,10:11]], axis=1)
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

def test_result_for_set_kl(gt_bboxes, pred_boxes, im_info):
    pred_boxes_lstd = np.mean(pred_boxes[:,6:10], axis=1, keepdims=True)
    pred_boxes = np.concatenate([pred_boxes[:,:6], pred_boxes_lstd, pred_boxes[:,10:11]], axis=1)
    h, w=int(im_info[0, -3]), int(im_info[0, -2])
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
                dt_matched[i] = maxpos + 1
                gt_matched[maxpos] = int(1 + i)
                scorelist.append((dt, 1, maxpos, overlap_iou[i][maxpos]))
            else:
                dt_matched[i] = maxpos + 1
                gt_matched[maxpos] = int(1 + i)
                scorelist.append((dt, -1, maxpos, overlap_iou[i][maxpos]))
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