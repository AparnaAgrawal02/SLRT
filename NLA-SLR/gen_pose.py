# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/kennymckormick/pyskl/blob/main/tools/data/custom_2d_skeleton.py
import argparse
from gc import garbage
import os
from unittest.mock import NonCallableMagicMock
import os.path as osp
from torch.utils.data import DataLoader, sampler
import torch
import cv2, time
from functools import partial
import logging

import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from utils.misc import (
    get_logger,
    load_config,
    log_cfg,
    load_checkpoint,
    make_model_dir,
    make_logger, make_writer,
    set_seed,
    symlink_update,
    is_main_process, init_DDP, move_to_device,
    neq_load_customized,
    synchronize,
    merge_pkls
)
import pickle
from dataset.Dataloader import build_dataloader
from mmpose.utils import adapt_mmdet_pipeline
try:
    import mmdet
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this script! ')

try:
    import mmpose
#    from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
    from mmpose.apis import inference_topdown, init_model

except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model` and '
                      '`init_pose_model` form `mmpose.apis`. These apis are '
                      'required in this script! ')

default_mmdet_root = osp.dirname(mmdet.__path__[0])
default_mmpose_root = osp.dirname(mmpose.__path__[0])
default_det_config = (
    f'{default_mmdet_root}/mmdet/configs/faster_rcnn/'
    'faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py')
default_det_ckpt = (
    'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/'
    'faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth')
default_pose_config = (
    f'{default_mmpose_root}/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/'
    'coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py')
default_pose_ckpt = (
    'https://download.openmmlab.com/mmpose/top_down/hrnet/'
    'hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth')


def detection_inference(model, frames):
    results = []
    for frame in frames:
        result = inference_detector(model, frame[0])
        results.append(result)
    return results


def pose_inference(model, frames, det_results):
    garbage_frame = 0
    if det_results is not None:
        assert len(frames) == len(det_results)
        total_frames = len(frames)
        kp = np.zeros((total_frames, 133, 2), dtype=np.float32)
        # bb = np.zeros((total_frames, 5), dtype=np.float32)

        for i, (f, d) in enumerate(zip(frames, det_results)):
            # Align input format
            d = [np.array(x.bboxes.cpu().numpy()[0]) for x in d]
            #print(d[0]["bbox"])
           # print(d[0]["bbox"][None])
            #print(f,d)
            #SAVE FRAME
            #cv2.imwrite("frame"+str(i)+".png",f[0])

            pose = inference_topdown(model, f[0], d, bbox_format='xyxy')
            results = merge_data_samples(pose)
                                         # build the visualizer
            # #print(results)
            # visualizer = VISUALIZERS.build(model.cfg.visualizer)

            # # set skeleton, colormap and joint connection rule
            # visualizer.set_dataset_meta(model.dataset_meta)

            # #img = imread(img_path, channel_order='rgb')

            # # visualize the results
            # visualizer.add_datasample(
            #     'result',
            #     f[0],
            #     data_sample=pose[0],
            #     show=False,
            #     out_file='temp.png')

           # print(pose)
            #breakpoint()
            if pose ==[] :
                # not detect person
                garbage_frame += 1
                continue
            #pose = sorted(pose, key=lambda x:x['bbox'][-1])
            #already sorted
            #keypoints, bbox = pose[-1]['keypoints'], pose[-1]['bbox']
            kp[i] = pose[0].pred_instances.keypoints

            # bb[i] = bbox

    else:
        print(frames.shape)
        d = [{'bbox': np.array([0, 0, frames.shape[2]-1, frames.shape[1]-1])}]
        pose = inference_topdown(model, frames[0], None, bbox_format='xyxy')[0]

    return kp, garbage_frame


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate 2D pose annotations for a custom video dataset')
    # * Both mmdet and mmpose should be installed from source
    parser.add_argument('--mmdet-root', type=str, default=default_mmdet_root)
    parser.add_argument('--mmpose-root', type=str, default=default_mmpose_root)
    parser.add_argument('--det-config', type=str, default=default_det_config)
    parser.add_argument('--det-ckpt', type=str, default=default_det_ckpt)
    parser.add_argument('--pose-config', type=str, default=default_pose_config)
    parser.add_argument('--pose-ckpt', type=str, default=default_pose_ckpt)
    # * Only det boxes with score larger than det_score_thr will be kept
    parser.add_argument('--det-score-thr', type=float, default=0.5)
    # * Only det boxes with large enough sizes will be kept,
    parser.add_argument('--det-area-thr', type=float, default=1600)
    # * Accepted formats for each line in video_list are:
    # * 1. "xxx.mp4" ('label' is missing, the dataset can be used for inference, but not training)
    # * 2. "xxx.mp4 label" ('label' is an integer (category index),
    # * the result can be used for both training & testing)
    # * All lines should take the same format.
    parser.add_argument('--video-list', type=str, help='the list of source videos')
    # * out should ends with '.pkl'
    parser.add_argument('--out', type=str, help='output pickle name')
    parser.add_argument('--tmpdir', type=str, default='./tmp')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--split', type=str, default='train', choices=['train', 'dev', 'test'])
    parser.add_argument('--start_end', nargs='+', type=int, default=None, help='for multi-node')
    parser.add_argument('--from_ckpt', type=int, default=0 ,choices=[0,1])
    parser.add_argument('--img_per_iter', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument("--config", default="configs/det.yaml", type=str, help="Training configuration file (yaml).")
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    args = parser.parse_args()
    return args


def main():
    set_seed(8)
    args = parse_args()
    cfg = load_config(args.config)
    cfg['local_rank'], cfg['world_size'], cfg['device'] = init_DDP()
    model_dir = cfg['training']['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    global logger
    logger = make_logger(model_dir=model_dir, log_file='gen_pose_{}_{}.log'.format(args.split, cfg['local_rank']))

    if cfg['data']['dataset_name'] == 'wlasl':
        path = osp.join('../../data/wlasl_2000', 'keypoints_hrnet_dark_coco_wholebody')
        h, w = 256, 256
    elif cfg['data']['dataset_name'] == 'MSASL_1000':
        path = osp.join('../../data/msasl', 'keypoints_hrnet_dark_coco_wholebody')
        h, w = 256, 256
    elif cfg['data']['dataset_name'] == 'NMFs-CSL':
        path = osp.join('../../data/NMFs-CSL', 'keypoints_hrnet_dark_coco_wholebody')
        h, w = 512, 512
    os.makedirs(path, exist_ok=True)

    dataloader, sampler = build_dataloader(cfg, args.split, is_train=False, val_distributed=True)

    # get existing pkls
    if args.from_ckpt:
        ckpts = {}
        for fname in os.listdir(path):
            if 'pkl' in fname and args.split in fname:
                with open(osp.join(path, fname), 'rb') as f:
                    data = pickle.load(f)
                    ckpts.update(data)
        ckpt_ids = list(ckpts.keys())
        print('num of ckpts: ', len(ckpt_ids))
        if cfg['local_rank'] == 0:
            merge_pkls(path, args.split, True)
        ckpts = {}

    det_model = init_detector(args.det_config, args.det_ckpt, 'cuda')
##HERE
    det_model.cfg=adapt_mmdet_pipeline(det_model.cfg)
#    assert det_model.CLASSES[0] == 'person', 'A detector trained on COCO is required'
    pose_model = init_model(args.pose_config, args.pose_ckpt, 'cuda')
    
    outputs = {}
    save_inte = 1
    for k, batch_data in tqdm(enumerate(dataloader), desc='[Generating keypoints of {:s} of {:s}, {:d} per gpu]'.format(args.split, cfg['data']['dataset_name'], len(dataloader))):
        frames = batch_data['sgn_videos'][0][0].numpy().transpose(0,2,3,1)*255  #[T,H,W,3]
        frames = np.uint8(frames)
        frames = np.split(frames, frames.shape[0], axis=0)
        video_id = batch_data['names'][0]

        if args.from_ckpt and video_id in ckpt_ids:
            if (k+1)%save_inte == 0:
                fname = '{:s}_{:d}_{:d}.pkl'.format(args.split, cfg['local_rank'], k)
                print('save to '+fname)
                with open(os.path.join(path, fname), 'wb') as f:
                    pickle.dump(outputs, f)
                outputs = {}
            continue
        
        det_results = detection_inference(det_model, frames)
        # * Get detection results for human
        #cbreakpoint()
        det_results1 =[]
        frame1 =[]
        #print(det_results)
        for i in range(len(det_results)):
            if det_results[i]!=None or det_results[i].pred_instances.labels.shape!=0:
                #print(det_results[i])
                det_results1.append(det_results[i].pred_instances)
                frame1.append(frames[i])
            elif len(det_results1)!=0 :
                det_results1.append(det_results1[-1].pred_instances)
                frame1.append(frames[i])
        
        print("old frames",len(frames),"new frames",len(frame1),len(det_results))
       # breakpoint()  
        det_results=det_results1
        frames=frame1
        
        #det_results = [x.pred_instances for x in det_results]
        # for i, res in enumerate(det_results):
        #     # * filter boxes with small scores
        #     res = res[res[:, 4] >= args.det_score_thr]
        #     # * filter boxes with small areas
        #     box_areas = (res[:, 3] - res[:, 1]) * (res[:, 2] - res[:, 0])
        #     assert np.all(box_areas >= 0)
        #     res = res[box_areas >= args.det_area_thr]
        #     det_results[i] = res

        for i, res in enumerate(det_results):
            # * filter boxes with small scores
            idxs=res.scores >= args.det_score_thr
           # print(idxs,res.scores)
            res_boxes=res.bboxes[idxs]
            # * filter boxes with small areas
            box_areas = (res_boxes[:, 3] - res_boxes[:, 1]) * (res_boxes[:, 2] - res_boxes[:, 0])
           # print(box_areas)
            #breakpoint()
            box_areas = box_areas.cpu().numpy()
            assert np.all(box_areas >= 0)
            idxs2 = box_areas >= args.det_area_thr
            det_results[i] = res

        pose_results, garbage_frame = pose_inference(pose_model, frames, det_results)

    #     if k%save_inte==0:
    #         # visulize video
    #         fps=15
    #         fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    #         name = "/ssd_scratch/cvit/aparna/vis_res/'+video_id.split('/')[-1]+'_hrnet.mp4"
    #         if not os.path.exists("/ssd_scratch/cvit/aparna/vis_res/"):
    #             os.makedirs("/ssd_scratch/cvit/aparna/vis_res/")
    #         video_writer = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    #         for idx in range(len(frames)):
    #             img = frames[idx][0, ..., ::-1].astype(np.uint8)
    #             # print(img)
    #             cv2.imwrite('temp.png', img)
    #             img = cv2.imread('temp.png')
    #             # bb_x1, bb_y1, bb_x2, bb_y2 = bb_results[idx, :-1]
    #             # cv2.line(f, (int(bb_x1), int(bb_y1)), (int(bb_x1), int(bb_y2)), (0,255,0))
    #             # cv2.line(f, (int(bb_x1), int(bb_y1)), (int(bb_x2), int(bb_y1)), (0,255,0))
    #             # cv2.line(f, (int(bb_x2), int(bb_y2)), (int(bb_x2), int(bb_y1)), (0,255,0))
    #             # cv2.line(f, (int(bb_x2), int(bb_y2)), (int(bb_x1), int(bb_y2)), (0,255,0))
    #             for j in range(133):
    #                 print(pose_results.shape)
    #                 x,y = pose_results[idx, j, :]
    #                 x,y = int(x), int(y)
    #                 cv2.circle(img, (x,y), 1, (0,0,255))
    #             cv2.imwrite('temp.png', img)
    #             img = cv2.imread('temp.png')
    #             video_writer.write(img)

    #             # vis_pose_result(pose_model,
    #             #                 img,
    #             #                 result=[{'keypoints': pose_results[idx]}],
    #             #                 radius=1,
    #             #                 thickness=1,
    #             #                 kpt_score_thr=0.3,
    #             #                 bbox_color='green',
    #             #                 dataset='TopDownCocoWholeBodyDataset',
    #             #                 dataset_info=None,
    #             #                 show=False,
    #             #                 out_file='temp.png')
    #             # f = cv2.imread('temp.png')
    #             # video_writer.write(f)
    #         video_writer.release()
    #         print()
    #    # print("pose_results",pose_results.shape,(batch_data['vlens'][0], 133, 3))
    #     #breakpoint()
    #     #assert pose_results.shape == (batch_data['vlens'][0], 133, 3)
    #     print(pose_results.shape, (batch_data['vlens'][0], 133, 2))
    #     #assert pose_results.shape == (batch_data['vlens'][0], 133, 2)
    #     # np.savez_compressed(fname+'.npz', keypoints=pose_results.astype(np.float16))
        outputs[video_id] = pose_results.astype(np.float32)
        if (k+1)%save_inte == 0:
            if args.start_end is None:
                fname = '{:s}_rank{:d}_{:d}.pkl'.format(args.split, cfg['local_rank'], k)
            else:
                fname = '{:s}_rank{:d}_start{:d}_end{:d}_{:d}.pkl'.format(args.split, cfg['local_rank'], args.start_end[0], args.start_end[1], k)
            print('save to '+fname)
            with open(os.path.join(path, fname), 'wb') as f:
                pickle.dump(outputs, f)
            outputs = {}
    
    if outputs != {}:
        if args.start_end is None:
            fname = '{:s}_rank{:d}_{:d}.pkl'.format(args.split, cfg['local_rank'], k)
        else:
            fname = '{:s}_rank{:d}_start{:d}_end{:d}_{:d}.pkl'.format(args.split, cfg['local_rank'], args.start_end[0], args.start_end[1], k)
        print('save to '+fname)
        with open(os.path.join(path, fname), 'wb') as f:
            pickle.dump(outputs, f)
        outputs = {}

    merge_pkls(path, args.split)


if __name__ == '__main__':
    main()
