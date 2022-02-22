import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import argparse

import cv2
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

lib_dir = os.path.join(os.path.abspath(__file__).split('tools')[0], 'lib')
model_dir = os.path.join(os.path.abspath(__file__).split('tools')[0], 'model')
sys.path.insert(0, lib_dir)
sys.path.insert(0, model_dir)

from utils import misc_utils, visual_utils, nms_utils

def inference(args, config, network):
    # model_path
    misc_utils.ensure_dir('outputs')
    saveDir = os.path.join('../model', args.model_dir, config.model_dir)
    model_file = os.path.join(saveDir,
            'dump-{}.pth'.format(args.resume_weights))
    assert os.path.exists(model_file)
    # build network
    net = network()
    net.eval()
    check_point = torch.load(model_file, map_location=torch.device('cpu'))
    net.load_state_dict(check_point['state_dict'])
    # get data
    records = misc_utils.load_json_lines(config.eval_source)
    if args.img_name == 'None':
        start = np.int(args.img_num.split('-')[0])
        end = np.int(args.img_num.split('-')[1])
        pbar = tqdm(total=end-start, ncols=50)
        for i in range(start, end):
            img_path = args.img_path + records[i]['ID'] + '.jpg'
            image, resized_img, im_info = get_data(
                    img_path, config.eval_image_short_size, config.eval_image_max_size) 
            pred_boxes = net(resized_img, im_info).numpy()
            pred_boxes, supp_boxes, pred_boxes_before_nms = post_process(pred_boxes, config, im_info[0, 2])
            inf_result, gt_boxes, gt_matched = visual_utils.my_inference_result(
                    config.eval_source, img_path, pred_boxes, im_info)
            pred_tags = pred_boxes[:, 5].astype(np.int32).flatten()
            pred_tags_name = np.array(config.class_names)[pred_tags]
            visualization(image, resized_img, pred_boxes, supp_boxes, pred_boxes_before_nms, inf_result, 
            gt_boxes, gt_matched, pred_tags_name, im_info, records, img_path, net, args, config, i)
            pbar.update(1)
        pbar.close()
    else:
        img_path = args.img_path + args.img_name + '.jpg'
        image, resized_img, im_info = get_data(
                img_path, config.eval_image_short_size, config.eval_image_max_size) 
        pred_boxes = net(resized_img, im_info).numpy()
        pred_boxes, supp_boxes, pred_boxes_before_nms = post_process(pred_boxes, config, im_info[0, 2])
        inf_result, gt_boxes, gt_matched = visual_utils.my_inference_result(
                config.eval_source, img_path, pred_boxes, im_info)
        pred_tags = pred_boxes[:, 5].astype(np.int32).flatten()
        pred_tags_name = np.array(config.class_names)[pred_tags]
        visualization(image, resized_img, pred_boxes, supp_boxes, pred_boxes_before_nms, inf_result, 
            gt_boxes, gt_matched, pred_tags_name, im_info, records, img_path, net, args, config)

def visualization(image, resized_img, pred_boxes, supp_boxes, pred_boxes_before_nms, inf_result, gt_boxes, 
                    gt_matched, pred_tags_name, im_info, records, img_path, net, args, config, img=0):
    # inplace draw
    if args.vis_mode == 'draw_boxes':
        image = visual_utils.draw_boxes(
                image,
                pred_boxes[:, :4],
                scores=pred_boxes[:, 4],
                tags=pred_tags_name,
                line_thick=2, line_color='red')
        name = img_path.split('/')[-1].split('.')[-2]
        fpath = 'outputs/{}.png'.format(name)
        cv2.imwrite(fpath, image)
    if args.vis_mode == 'draw_boxes_before_nms':
        image = visual_utils.draw_boxes(
                image,
                pred_boxes_before_nms[:, :4],
                scores=pred_boxes[:, 4],
                tags=pred_tags_name,
                line_thick=3, line_color='green')
        name = img_path.split('/')[-1].split('.')[-2]
        fpath = 'outputs/{}.png'.format(name)
        cv2.imwrite(fpath, image)
    if args.vis_mode == 'draw_boxes_heavy':
        gt_boxes = visual_utils.cal_vis_part(gt_boxes, records[img]['gtboxes'])
        image = visual_utils.draw_heavy_boxes(
                image,
                pred_boxes[:, :4],
                (supp_boxes, inf_result, gt_boxes, gt_matched),
                scores=pred_boxes[:, 4],
                tags=pred_tags_name,
                line_thick=1, line_color='red')
        name = img_path.split('/')[-1].split('.')[-2]
        fpath = 'outputs/{}.png'.format(name)
        cv2.imwrite(fpath, image)
    if args.vis_mode == 'scatter_scr_std_before_nms':
        scores = np.array([])
        ious = np.array([])
        lstds = np.array([])
        for i in range(len(inf_result)):
            pred_scores = np.concatenate([pred_boxes[i:i+1, 4], supp_boxes[i][:, 4]])
            keep = pred_scores > config.visulize_threshold
            pred_scores = pred_scores[keep]
            pred_lstds = np.concatenate([pred_boxes[i:i+1, 6:10], supp_boxes[i][:, 6:10]])
            pred_lstds = pred_lstds[keep].mean(axis=1)
            pred_bboxes = np.concatenate([pred_boxes[i:i+1, :4], supp_boxes[i][:, :4]])
            pred_bboxes = pred_bboxes[keep]
            gt_box = gt_boxes[inf_result[i][2], :4].reshape(1,-1)
            pred_iou = visual_utils.box_overlap_opr(pred_bboxes, gt_box, True).reshape(-1)
            scores = np.concatenate([scores, pred_scores])
            ious = np.concatenate([ious, pred_iou])
            lstds = np.concatenate([lstds, pred_lstds])
        f = open("./vis_data.txt",'a')
        data = np.concatenate([scores.reshape(-1,1), lstds.reshape(-1,1), \
            ious.reshape(-1,1)], axis=1)
        np.savetxt(f, data)
        f.close()
    if args.vis_mode == 'scatter_scr_std_after_nms':
        pred_score = np.array([r[0][4] for r in inf_result]).reshape(-1,1)
        pred_lstd = np.array([r[0][6:10] for r in inf_result])
        pred_iou = np.array([r[3] for r in inf_result]).reshape(-1,1)
        f = open("./vis_data.txt",'a')
        data = np.concatenate([pred_score, pred_lstd, pred_iou], axis=1)
        np.savetxt(f, data)
        f.close()
    if args.vis_mode == 'fpn_heatmap':
        pred_scr_list, pred_dist_list = net.inference(resized_img, im_info)
        scrs = np.zeros((image.shape[0], image.shape[1]))
        lstds = np.zeros((image.shape[0], image.shape[1]))
        for j in range(len(pred_scr_list)):
            scr = pred_scr_list[j][0, :, :, 0].numpy()
            scr = cv2.resize(scr, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            scrs += scr
            if j ==3 or j ==4:
                lstd = pred_dist_list[j][0, :, :, 4:].mean(dim=2).numpy()
                lstd = cv2.resize(lstd, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
                lstds += lstd
        lstd_map, scr_map = None, None
        scr_map = cv2.normalize(scrs, scr_map, alpha=0, beta=255, \
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        lstd_map = cv2.normalize(lstds, lstd_map, alpha=0, beta=255, \
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        scr_map = cv2.applyColorMap(scr_map, cv2.COLORMAP_JET)
        lstd_map = cv2.applyColorMap(lstd_map, cv2.COLORMAP_JET)
        lstd_map = cv2.cvtColor(lstd_map, cv2.COLOR_BGR2RGB)
        name = img_path.split('/')[-1].split('.')[-2]
        fpath = 'outputs/{}.png'.format(name)
        cv2.imwrite(fpath, image)
        fpath = 'outputs/{}_scoremap.png'.format(name)
        cv2.imwrite(fpath, scr_map)
        fpath = 'outputs/{}_lstdmap.png'.format(name)
        cv2.imwrite(fpath, lstd_map)
    if args.vis_mode == 'each_fpn_heatmap':
        pred_scr_list, pred_dist_list = net.inference(resized_img, im_info)
        for j in range(len(pred_scr_list)):
            lstds = np.zeros((image.shape[0], image.shape[1]))
            lstd = pred_dist_list[j][0, :, :, 4:].mean(dim=2).numpy()
            lstd = cv2.resize(lstd, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            lstd_map = cv2.normalize(lstd, None, alpha=0, beta=255, \
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            lstd_map = cv2.applyColorMap(lstd_map, cv2.COLORMAP_JET)
            lstd_map = cv2.cvtColor(lstd_map, cv2.COLOR_BGR2RGB)
            name = img_path.split('/')[-1].split('.')[-2]
            fpath = 'outputs/{}.png'.format(name)
            fpath = 'outputs/{}_lstdmap{}.png'.format(name, j)
            cv2.imwrite(fpath, lstd_map)

def post_process(pred_boxes, config, scale):
    if config.test_nms_method == 'set_nms':
        assert pred_boxes.shape[-1] > 6, "Not EMD Network! Using normal_nms instead."
        assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
        top_k = pred_boxes.shape[-1] // 6
        n = pred_boxes.shape[0]
        pred_boxes = pred_boxes.reshape(-1, 6)
        idents = np.tile(np.arange(n)[:,None], (1, top_k)).reshape(-1, 1)
        pred_boxes = np.hstack((pred_boxes, idents))
        keep = pred_boxes[:, 4] > config.pred_cls_threshold
        pred_boxes = pred_boxes[keep]
        keep = nms_utils.set_cpu_nms(pred_boxes, 0.5)
        pred_boxes = pred_boxes[keep]
    elif config.test_nms_method == 'normal_nms':
        keep = pred_boxes[:, 4] > config.pred_cls_threshold
        pred_boxes = pred_boxes[keep]
        keep, supp = nms_utils.nms_for_plot(pred_boxes, config.test_nms)
        pre_boxes_before_nms = pred_boxes
        pred_boxes = pred_boxes[keep]
    elif config.test_nms_method == 'none':
        assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
        pred_boxes = pred_boxes.reshape(-1, 6)
        keep = pred_boxes[:, 4] > config.pred_cls_threshold
        pred_boxes = pred_boxes[keep]
    #if pred_boxes.shape[0] > config.detection_per_image and \
    #    config.test_nms_method != 'none':
    #    order = np.argsort(-pred_boxes[:, 4])
    #    order = order[:config.detection_per_image]
    #    pred_boxes = pred_boxes[order]
    # recovery the scale
    pred_boxes[:, :4] /= scale
    keep = pred_boxes[:, 4] > config.visulize_threshold
    pred_boxes = pred_boxes[keep]

    pre_boxes_before_nms[:, :4] /= scale
    boxes_supp = supp[:keep.sum()]
    supp_boxes = [pre_boxes_before_nms[box_supp] for box_supp in boxes_supp]

    keep = pre_boxes_before_nms[:, 4] > config.visulize_threshold
    pre_boxes_before_nms = pre_boxes_before_nms[keep]

    return pred_boxes, supp_boxes, pre_boxes_before_nms

def get_data(img_path, short_size, max_size):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    resized_img, scale = resize_img(
            image, short_size, max_size)

    original_height, original_width = image.shape[0:2]
    height, width = resized_img.shape[0:2]
    resized_img = resized_img.transpose(2, 0, 1)
    im_info = np.array([height, width, scale, original_height, original_width, 0])
    return image, torch.tensor([resized_img]).float(), torch.tensor([im_info])

def resize_img(image, short_size, max_size):
    height = image.shape[0]
    width = image.shape[1]
    im_size_min = np.min([height, width])
    im_size_max = np.max([height, width])
    scale = (short_size + 0.0) / im_size_min
    if scale * im_size_max > max_size:
        scale = (max_size + 0.0) / im_size_max
    t_height, t_width = int(round(height * scale)), int(
        round(width * scale))
    resized_image = cv2.resize(
            image, (t_width, t_height), interpolation=cv2.INTER_LINEAR)
    return resized_image, scale

def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-md', default=None, required=True, type=str)
    parser.add_argument('--resume_weights', '-r', default=None, required=True, type=str)
    parser.add_argument('--img_path', '-i', default=None, required=True, type=str)
    parser.add_argument('--img_num', '-nu', default=None, required=False, type=str)
    parser.add_argument('--img_name', '-na', default=None, required=False, type=str)
    parser.add_argument('--vis_mode', '-vi', default=None, required=False, type=str)
    # args = parser.parse_args()
    args = parser.parse_args(['--model_dir', 'fa_fpn_jsgauvpd_kll1e-0_scale2_xywh',
                                '--resume_weights', '30',
                                '--img_path', './data/CrowdHuman/Images/',
                                '--img_num', '0-400',
                                '--img_name', '273275,c5d22000e47802ff',
                                '--vis_mode', 'fpn_heatmap'])
    # import libs
    model_root_dir = os.path.join(model_dir, args.model_dir)
    sys.path.insert(0, model_root_dir)
    from config import config
    from network import Network
    inference(args, config, Network)

if __name__ == '__main__':
    run_inference()