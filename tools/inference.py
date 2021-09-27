import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import argparse

import cv2
import torch
import numpy as np

lib_dir = os.path.join(os.path.abspath(__file__).split('tools')[0], 'lib')
model_dir = os.path.join(os.path.abspath(__file__).split('tools')[0], 'model')
sys.path.insert(0, lib_dir)
sys.path.insert(0, model_dir)

from utils import misc_utils, visual_utils, nms_utils

def inference(args, config, network):
    # model_path
    misc_utils.ensure_dir('outputs')
    saveDir = os.path.join(model_dir, args.model_dir, config.model_dir)
    model_file = os.path.join(saveDir,
            'dump-{}.pth'.format(args.resume_weights))
    assert os.path.exists(model_file)
    # build network
    net = network()
    net.eval()
    check_point = torch.load(model_file, map_location=torch.device('cpu'))
    net.load_state_dict(check_point['state_dict'])
    # get data
    image, resized_img, im_info = get_data(
            args.img_path, config.eval_image_short_size, config.eval_image_max_size)
    pred_boxes = net(resized_img, im_info).numpy()
    pred_boxes = post_process(pred_boxes, config, im_info[0, 2])
    if config.plot_type != 'normal_plot':
        pred_boxes, supp_boxes = pred_boxes
        inf_result, gt_boxes, gt_matched = visual_utils.inference_result(
                config.eval_source, args.img_path, pred_boxes, im_info)
    pred_tags = pred_boxes[:, 5].astype(np.int32).flatten()
    pred_tags_name = np.array(config.class_names)[pred_tags]
    # inplace draw
    if config.plot_type == 'my_plot':
        visual_utils.draw_my_boxes(
            image,
            pred_boxes,
            (supp_boxes, inf_result, gt_boxes, gt_matched),
            args,
            line_thick=1, line_color=('red','green'),
            )

    elif config.plot_type == 'plot_supp':
        visual_utils.draw_supp_boxes(
            image,
            pred_boxes,
            supp_boxes,
            args,
            line_thick=1, line_color=('red','green'),
            )

    elif config.plot_type == 'plot_dist':
        # plot dist mask for rcnn_mva
        image = visual_utils.draw_dists(
                image,
                pred_boxes[:, :4],
                pred_boxes[:, 6:],
                config.va_beta,
                scores=pred_boxes[:, 4])
        name = args.img_path.split('/')[-1].split('.')[-2]
        fpath = 'outputs/{}.png'.format(name)
        cv2.imwrite(fpath, image)

    elif config.plot_type == 'normal_plot':
        image = visual_utils.draw_boxes(
                image,
                pred_boxes[:, :4],
                scores=pred_boxes[:, 4],
                tags=pred_tags_name,
                line_thick=1, line_color='white')
        name = args.img_path.split('/')[-1].split('.')[-2]
        fpath = 'outputs/{}.png'.format(name)
        cv2.imwrite(fpath, image)


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
    elif config.test_nms_method == 'normal_nms' and config.plot_type == 'normal_plot':
        if config.plot_data:
                pred_boxes = pred_boxes.reshape(-1, 10)
        else:
            assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
            pred_boxes = pred_boxes.reshape(-1, 6)
        keep = pred_boxes[:, 4] > config.pred_cls_threshold
        pred_boxes = pred_boxes[keep]
        keep = nms_utils.cpu_nms(pred_boxes, config.test_nms)
        pred_boxes = pred_boxes[keep]
    elif config.plot_type != 'normal_plot':
        keep = pred_boxes[:, 4] > config.pred_cls_threshold
        pred_boxes = pred_boxes[keep]
        keep, supp = nms_utils.nms_for_plot(pred_boxes, config.test_nms)
        pre_nms_boxes = pred_boxes
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
    pred_boxes[:, 8:] /= scale
    # vis_keep = pred_boxes[:, 4] > config.visulize_threshold
    vis_keep = pred_boxes[:, 4] >= 0
    pred_boxes = pred_boxes[vis_keep]
    if config.plot_type != 'normal_plot':
        pre_nms_boxes[:, :4] /= scale
        pre_nms_boxes[:, 8:] /= scale
        boxes_supp = supp[:vis_keep.sum()]
        supp_boxes = [pre_nms_boxes[box_supp] for box_supp in boxes_supp]
        pred_boxes = [pred_boxes, supp_boxes]
    return pred_boxes

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
    # args = parser.parse_args()
    args = parser.parse_args(['--model_dir', 'fa_fpn_vpd_kll1e-1_prior_p1_wh',
                                '--resume_weights', '38',
                                '--img_path', './data/CrowdHuman/Images/273275,e99d80007220d4b6.jpg'])
    
    # import libs
    model_root_dir = os.path.join(model_dir, args.model_dir)
    sys.path.insert(0, model_root_dir)
    from config import config
    from network import Network
    inference(args, config, Network)

if __name__ == '__main__':
    run_inference()
