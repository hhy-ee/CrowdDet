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