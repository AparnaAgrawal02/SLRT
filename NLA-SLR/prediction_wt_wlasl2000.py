from turtle import forward
import warnings, wandb
import pickle
from collections import defaultdict
from modelling.model import build_model
from utils.optimizer import build_optimizer, build_scheduler
warnings.filterwarnings("ignore")
import argparse
import numpy as np
import os, sys
import shutil
import time
import queue
sys.path.append(os.getcwd())#slt dir
import torch
from torch.nn.parallel import DistributedDataParallel as DDP, distributed
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from utils.misc import (
    get_logger,
    load_config,
    log_cfg,
    load_checkpoint,
    make_logger, make_writer,
    set_seed,
    symlink_update,
    is_main_process, init_DDP, move_to_device,
    neq_load_customized,
    synchronize,
)
from utils.metrics import compute_accuracy
from dataset.Dataloader import build_dataloader
from dataset.Dataset import build_dataset
from utils.progressbar import ProgressBar
from copy import deepcopy

def extract_feature(model, val_dataloader, cfg, 
        tb_writer=None, wandb_run=None,
        epoch=None, global_step=None,
        generate_cfg={}, save_dir=None, return_prob=False):  #to-do output_dir
    logger = get_logger()
    logger.info(generate_cfg)
    print()
    vocab = val_dataloader.dataset.vocab
    split = val_dataloader.dataset.split
    cls_num = len(vocab)

    word_emb_tab = []
    if val_dataloader.dataset.word_emb_tab is not None:
        for w in vocab:
            word_emb_tab.append(torch.from_numpy(val_dataloader.dataset.word_emb_tab[w]))
        word_emb_tab = torch.stack(word_emb_tab, dim=0).float().to(cfg['device'])
    else:
        word_emb_tab = None

    if is_main_process() and os.environ.get('enable_pbar', '1') == '1':
        pbar = ProgressBar(n_total=len(val_dataloader), desc=val_dataloader.dataset.split.upper())
    else:
        pbar = None
    if epoch != None:
        logger.info('------------------Evaluation epoch={} {} examples #={}---------------------'.format(epoch, val_dataloader.dataset.split, len(val_dataloader.dataset)))
    elif global_step != None:
        logger.info('------------------Evaluation global step={} {} examples #={}------------------'.format(global_step, val_dataloader.dataset.split, len(val_dataloader.dataset)))
    model.eval()

    folder_path = "/scratch/cvit/aparna/nla_slr_wlasl_generated"
    os.makedirs(folder_path, exist_ok=True)
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            #forward -- loss
            batch = move_to_device(batch, cfg['device'])
            _, features = model(is_train=False, labels=batch['labels'], sgn_videos=batch['sgn_videos'], sgn_keypoints=batch['sgn_keypoints'], epoch=epoch)
            features = features.cpu()
            torch.save(features, os.path.join(folder_path, batch['names'][0]+".pt"))
            # print(batch['names'], features.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SLT baseline Testing")
    parser.add_argument("--config", default="configs/default.yaml", type=str, help="Training configuration file (yaml).")
    parser.add_argument("--save_subdir", default='prediction', type=str)
    parser.add_argument('--ckpt_name', default='best.ckpt', type=str)
    parser.add_argument('--eval_setting', default='origin', type=str)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg['local_rank'], cfg['world_size'], cfg['device'] = init_DDP()
    set_seed(seed=cfg["training"].get("random_seed", 42))
    model_dir = cfg['training']['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    global logger
    logger = make_logger(model_dir=model_dir, log_file='prediction_{}_{}.log'.format(args.eval_setting, cfg['local_rank']))

    dataset = build_dataset(cfg['data'], 'train')
    vocab = dataset.vocab
    cls_num = len(vocab)
    word_emb_tab = []
    if dataset.word_emb_tab is not None:
        for w in vocab:
            word_emb_tab.append(torch.from_numpy(dataset.word_emb_tab[w]))
        word_emb_tab = torch.stack(word_emb_tab, dim=0).float().to(cfg['device'])
    else:
        word_emb_tab = None
    del vocab; del dataset
    model = build_model(cfg, cls_num, word_emb_tab=word_emb_tab)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) 
    #load model
    load_model_path = os.path.join(model_dir,'ckpts',args.ckpt_name)
    if os.path.isfile(load_model_path):
        state_dict = torch.load(load_model_path, map_location='cuda')
        neq_load_customized(model, state_dict['model_state'], verbose=True)
        epoch, global_step = state_dict.get('epoch',0), state_dict.get('global_step',0)
        logger.info('Load model ckpt from '+load_model_path)
    else:
        logger.info(f'{load_model_path} does not exist')
        epoch, global_step = 0, 0
    
    model = DDP(model, 
            device_ids=[cfg['local_rank']], 
            output_device=cfg['local_rank'],
            find_unused_parameters=True)

    for split in ['train']:
        logger.info('Evaluate on {} set'.format(split))
        if args.eval_setting == 'origin':
            dataloader, sampler = build_dataloader(cfg, split, is_train=False, val_distributed=True)
            extract_feature(model=model.module, val_dataloader=dataloader, cfg=cfg, 
                    epoch=epoch, global_step=global_step, 
                    generate_cfg=cfg['testing']['cfg'],
                    save_dir=os.path.join(model_dir,args.save_subdir,split), return_prob=True)
        
        elif args.eval_setting == '3x_pad':
            test_p = ['start', 'end', 'central']
            test_m = ['start_pad', 'end_pad', 'pad']
            all_prob = {}
            for t_p, t_m in zip(test_p, test_m):
                logger.info('----------------------------------crop position: {}----------------------------'.format(t_p))
                new_cfg = deepcopy(cfg)
                new_cfg['data']['transform_cfg']['index_setting'][2] = t_p
                new_cfg['data']['transform_cfg']['index_setting'][3] = t_m
                dataloader, sampler = build_dataloader(new_cfg, split, is_train=False, val_distributed=False)
                extract_feature(model=model.module, val_dataloader=dataloader, cfg=new_cfg, 
                                        epoch=epoch, global_step=global_step, 
                                        generate_cfg=cfg['testing']['cfg'],
                                        save_dir=os.path.join(model_dir,args.save_subdir,split), return_prob=True)
               
