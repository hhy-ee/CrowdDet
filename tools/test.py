import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import math
import argparse

import numpy as np
from tqdm import tqdm
import torch
from torch.multiprocessing import Queue, Process

# sys.path.insert(0, '../lib')
# sys.path.insert(0, '../model')
lib_dir = os.path.join(os.path.abspath(__file__).split('tools')[0], 'lib')
model_dir = os.path.join(os.path.abspath(__file__).split('tools')[0], 'model')
sys.path.insert(0, lib_dir)
sys.path.insert(0, model_dir)

from data.CrowdHuman import CrowdHuman
from utils import misc_utils, nms_utils
from evaluate import compute_JI, compute_APMR

def eval_all(args, config, network):
    # model_path
    saveDir = config.model_dir
    evalDir = config.eval_dir
    misc_utils.ensure_dir(evalDir)
    model_file = os.path.join(saveDir, 
            'dump-{}.pth'.format(args.resume_weights))
    assert os.path.exists(model_file)
    # get devices
    str_devices = args.devices
    devices = misc_utils.device_parser(str_devices)
    # load data
    crowdhuman = CrowdHuman(config, if_train=False)
    # multiprocessing
    num_devs = len(devices)
    len_dataset = len(crowdhuman)
    # len_dataset = 5
    num_image = math.ceil(len_dataset / num_devs)
    result_queue = Queue(500)
    procs = []
    all_results = []
    for i in range(num_devs):
        start = i * num_image
        end = min(start + num_image, len_dataset)
        proc = Process(target=inference, args=(
                config, network, model_file, devices[i], crowdhuman, start, end, result_queue))
        proc.start()
        procs.append(proc)
    pbar = tqdm(total=len_dataset, ncols=50)
    for i in range(len_dataset):
        t = result_queue.get()
        all_results.append(t)
        pbar.update(1)
    pbar.close()
    for p in procs:
        p.join()
    fpath = os.path.join(evalDir, 'dump-{}.json'.format(args.resume_weights))
    misc_utils.save_json_lines(all_results, fpath)
    # evaluation
    eval_path = os.path.join(evalDir, 'eval-{}.json'.format(args.resume_weights))
    eval_fid = open(eval_path,'w')
    # res_line, JI = compute_JI.evaluation_all(fpath, 'box')
    # for line in res_line:
    #     eval_fid.write(line+'\n')
    AP, MR = compute_APMR.compute_APMR(fpath, config.eval_source, 'box')
    # line = 'AP:{:.4f}, MR:{:.4f}, JI:{:.4f}.'.format(AP, MR, JI)
    line = 'AP:{:.4f}, MR:{:.4f}.'.format(AP, MR)
    print(line)
    eval_fid.write(line+'\n')
    eval_fid.close()

def eval_all_epoch(args, config, network):
    for epoch_id in range(18, int(args.resume_weights)+1):
        # model_path
        saveDir = config.model_dir
        evalDir = config.eval_dir
        misc_utils.ensure_dir(evalDir)
        model_file = os.path.join(saveDir, 
                'dump-{}.pth'.format(str(epoch_id)))
        assert os.path.exists(model_file)
        # get devices
        str_devices = args.devices
        devices = misc_utils.device_parser(str_devices)
        # load data
        crowdhuman = CrowdHuman(config, if_train=False)
        # multiprocessing
        num_devs = len(devices)
        len_dataset = len(crowdhuman)
        num_image = math.ceil(len_dataset / num_devs)
        result_queue = Queue(500)
        procs = []
        all_results = []
        for i in range(num_devs):
            start = i * num_image
            end = min(start + num_image, len_dataset)
            proc = Process(target=inference, args=(
                    config, network, model_file, devices[i], crowdhuman, start, end, result_queue))
            proc.start()
            procs.append(proc)
        pbar = tqdm(total=len_dataset, ncols=50)
        for i in range(len_dataset):
            t = result_queue.get()
            all_results.append(t)
            pbar.update(1)
        pbar.close()
        for p in procs:
            p.join()
        fpath = os.path.join(evalDir, 'dump-{}.json'.format(str(epoch_id)))
        misc_utils.save_json_lines(all_results, fpath)
        # evaluation
        eval_path = os.path.join(evalDir, 'eval-{}.json'.format(str(epoch_id)))
        eval_fid = open(eval_path,'w')
        # res_line, JI = compute_JI.evaluation_all(fpath, 'box')
        # for line in res_line:
        #     eval_fid.write(line+'\n')
        AP, MR = compute_APMR.compute_APMR(fpath, config.eval_source, 'box')
        # line = 'AP:{:.4f}, MR:{:.4f}, JI:{:.4f}.'.format(AP, MR, JI)
        line = 'AP:{:.4f}, MR:{:.4f}.'.format(AP, MR)
        print(line)
        eval_fid.write(line+'\n')
        eval_fid.close()

def inference(config, network, model_file, device, dataset, start, end, result_queue):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.multiprocessing.set_sharing_strategy('file_system')
    # init model
    net = network()
    net.cuda(device)
    net = net.eval()
    check_point = torch.load(model_file)
    net.load_state_dict(check_point['state_dict'])
    # init data
    dataset.records = dataset.records[start:end]
    data_iter = torch.utils.data.DataLoader(dataset=dataset, shuffle=False)
    # inference
    for (image, gt_boxes, im_info, ID) in data_iter:
        pred_boxes = net(image.cuda(device), im_info.cuda(device))
        scale = im_info[0, 2]
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
            if not config.save_data:
                assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
                pred_boxes = pred_boxes.reshape(-1, 6)
            else:
                pred_boxes = pred_boxes.reshape(-1, pred_boxes.size(1))
            keep = pred_boxes[:, 4] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]
            keep = nms_utils.cpu_nms(pred_boxes, config.test_nms)
            pred_boxes = pred_boxes[keep]
            if config.save_data:
                pred_scores = pred_boxes[:, 4].reshape(-1, 1)
                pred_tags = pred_boxes[:, 5].reshape(-1, 1)
                pred_lstd = pred_boxes[:, 6:]
                save_data(pred_scores, pred_tags, pred_lstd)
            pred_boxes = pred_boxes[:, :6]
        elif config.test_nms_method == 'kl_nms':
            pred_boxes = pred_boxes.reshape(-1, pred_boxes.size(1))
            keep = pred_boxes[:, 4] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]
            keep = nms_utils.cpu_nms(pred_boxes, config.test_nms)
            # keep = nms_utils.cpu_kl_nms(pred_boxes, config.test_nms)
            pred_boxes = pred_boxes[keep]
            pred_boxes = pred_boxes
        elif config.test_nms_method == 'none':
            assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
            pred_boxes = pred_boxes.reshape(-1, 6)
            keep = pred_boxes[:, 4] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]
        else:
            raise ValueError('Unknown NMS method.')
        #if pred_boxes.shape[0] > config.detection_per_image and \
        #    config.test_nms_method != 'none':
        #    order = np.argsort(-pred_boxes[:, 4])
        #    order = order[:config.detection_per_image]
        #    pred_boxes = pred_boxes[order]
        # recovery the scale
        pred_boxes[:, :4] /= scale
        pred_boxes[:, 2:4] -= pred_boxes[:, :2]
        gt_boxes = gt_boxes[0].numpy()
        gt_boxes[:, 2:4] -= gt_boxes[:, :2]
        result_dict = dict(ID=ID[0], height=int(im_info[0, -3]), width=int(im_info[0, -2]),
                dtboxes=boxes_dump(pred_boxes), gtboxes=boxes_dump(gt_boxes))
        result_queue.put_nowait(result_dict)

def boxes_dump(boxes):
    if boxes.shape[-1] == 8:
        result = [{'box':[round(i, 1) for i in box[:4].tolist()],
                   'score':round(float(box[4]), 5),
                   'tag':int(box[5]),
                   'lstd':float(box[6:].mean())} for box in boxes]
    elif boxes.shape[-1] == 7:
        result = [{'box':[round(i, 1) for i in box[:4]],
                   'score':round(float(box[4]), 5),
                   'tag':int(box[5]),
                   'proposal_num':int(box[6])} for box in boxes]
    elif boxes.shape[-1] == 6:
        result = [{'box':[round(i, 1) for i in box[:4].tolist()],
                   'score':round(float(box[4]), 5),
                   'tag':int(box[5])} for box in boxes]
    elif boxes.shape[-1] == 5:
        result = [{'box':[round(i, 1) for i in box[:4]],
                   'tag':int(box[4])} for box in boxes]
    else:
        raise ValueError('Unknown box dim.')
    return result

def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-md', default=None, required=True, type=str)
    parser.add_argument('--resume_weights', '-r', default=None, required=True, type=str)
    parser.add_argument('--devices', '-d', default='0', type=str)
    os.environ['NCCL_IB_DISABLE'] = '1'

    args = parser.parse_args()
    # args = parser.parse_args(['--model_dir', 'fa_fpn_vpd_kll1e-1_prior_p1_wh', 
    #                           '--resume_weights', '38', '-d', '0-1'])

    # import libs
    model_root_dir = os.path.join(model_dir, args.model_dir)
    sys.path.insert(0, model_root_dir)
    from config import config
    from network import Network
    eval_all(args, config, Network)

def save_data(scores, ious, dists):
    import numpy as np
    f = open("./vis_data.txt",'a')
    data = torch.cat([scores, ious, dists], dim=1)
    data = data.detach().cpu().numpy()
    np.savetxt(f, data)
    f.close()
    return 0

if __name__ == '__main__':
    run_test()

