import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
    inf_result, gt_boxes, gt_matched = visual_utils.inference_result_for_set_kl(
                config.eval_source, args.img_path, pred_boxes, im_info)
    pred_tags = pred_boxes[:, 5].astype(np.int32).flatten()
    pred_tags_name = np.array(config.class_names)[pred_tags]
    # inplace draw
    if config.plot_type == 'draw_mip_for_set_kl':
        visual_utils.draw_mip_for_set_kl(
            image,
            pred_boxes,
            (inf_result, gt_boxes, gt_matched),
            args,
            line_thick=1,
            )


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
    elif config.test_nms_method == 'set_kl_nms':
        assert pred_boxes.shape[-1] > 6, "Not EMD Network! Using normal_nms instead."
        top_k = 2
        n = pred_boxes.shape[0]
        pred_boxes = pred_boxes.reshape(-1, 10)
        idents = np.tile(np.arange(n)[:,None], (1, top_k)).reshape(-1, 1)
        pred_boxes = np.hstack((pred_boxes, idents))
        keep = pred_boxes[:, 4] > config.pred_cls_threshold
        pred_boxes = pred_boxes[keep]
        keep = nms_utils.set_cpu_kl_nms(pred_boxes, 0.5)
        pred_boxes = pred_boxes[keep]
        pred_boxes = np.array(sorted(pred_boxes, key=lambda x: x[4], reverse=True))
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
    vis_keep = pred_boxes[:, 4] > config.visulize_threshold
    pred_boxes = pred_boxes[vis_keep]
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
    args = parser.parse_args(['--model_dir', 'rcnn_mip_single_vpd_kll1e-1_prior_p1_xywh',
                                '--resume_weights', '30',
                                '--img_path', './data/CrowdHuman/Images/273271,c9db000d5146c15.jpg'])

    # 273275,720840003a49cf5b:  
    # ('0.999 -0.531', '0.662 -0.489'), 
    # ('0.997 -0.541', '0.664 -0.508'), 
    # ('0.996 -0.500', '0.317 -0.446'), *
    # ('0.985 -0.520', '0.393 -0.479')
    # 282555,715100051f7de22
    # ('0.998 -0.573', '0.349 -0.519')
    # ('0.989 -0.515', '0.544 -0.478')


    # import libs
    model_root_dir = os.path.join(model_dir, args.model_dir)
    sys.path.insert(0, model_root_dir)
    from config import config
    from network import Network
    inference(args, config, Network)

if __name__ == '__main__':
    run_inference()
