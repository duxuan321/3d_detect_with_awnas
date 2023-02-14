
import argparse
import datetime
import glob
from operator import mod
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
import yaml
from thop import profile
from icecream import ic
import copy
from collections import defaultdict, OrderedDict
from aw_nas import utils
import pickle
import json

import tqdm
import torch.distributed as dist

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--sample_num', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--yaml_file_path', type=str, default=None)
    parser.add_argument('--controller_samples', type=int, default=1)
    parser.add_argument('--epoch_num', type=int, default=1)
    parser.add_argument('--train_predictor', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )
    ic(args.workers, args.batch_size)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    rollout_batch_size = 1
    controller_optimizer = None
    dir_ = "/".join(args.cfg_file.split('/')[:-1]) + '/'
    ic(dir_)

    # ----------------------------- 一些有用的参数 ---------------------------
    model_type = cfg.MODEL.BACKBONE_2D.NAME
    num_filters = np.array(cfg['MODEL']['BACKBONE_2D']['NUM_FILTERS'])
    layer_nums = cfg['MODEL']['BACKBONE_2D']['LAYER_NUMS'] if cfg['MODEL']['BACKBONE_2D'].get("LAYER_NUMS", None) else None
    fix_channel = model.backbone_2d.fix_block_channel if hasattr(model.backbone_2d, "fix_block_channel") else None
    multi_ratio_choice = np.array(model.backbone_2d.mult_ratio_choices)
    kwargs = {
        "model_type": model_type,
        "num_filters": num_filters, 
        "layer_nums": layer_nums,
        "fix_channel": fix_channel,
        "multi_ratio_choice": multi_ratio_choice
    }
    all_sample_size = args.controller_samples * args.epoch_num
    save_path = "/".join(args.ckpt.split('/')[:-2]) + "/evo_" + str(all_sample_size)
    # ----------------------------------------------------------------------
    if args.train_predictor: #训练predictor时置为True
        model.backbone_2d.controller.set_mode("train")
        for ep in range(args.epoch_num):
            if cfg.LOCAL_RANK == 0:
                rollouts = model.backbone_2d.controller.sample(\
                    args.controller_samples, batch_size=rollout_batch_size, **kwargs)
                pickle.dump(rollouts, open(os.path.join(dir_, 'rollouts.pkl'), 'wb'))
            dist.barrier()
            rollouts = pickle.load(open(os.path.join(dir_, 'rollouts.pkl'), 'rb')) # 确保每个线程的rollout一致
            
            for i in range(len(rollouts)):
                ic(ep, i)
                geno = rollouts[i].genotype
                model.gene_type = geno

                final_model = copy.deepcopy(model)
                rollout_temp = final_model.backbone_2d.search_space.rollout_from_genotype(geno)
                final_model.backbone_2d = final_model.backbone_2d.finalize_rollout(rollout_temp)
                final_model.cuda()

                # start evaluation
                result_dict = eval_utils.eval_one_epoch(
                        cfg, final_model, test_loader, epoch_id, logger, dist_test=dist_test,
                        result_dir=eval_output_dir, save_to_file=args.save_to_file, geno=geno, get_flops=True)

                if cfg.LOCAL_RANK == 0:
                    res = []
                    res.append(result_dict['flops'][0] * (-1))
                    res.append(result_dict['Car_3d/moderate_R40'])
                    res.append(result_dict['Pedestrian_3d/moderate_R40'])
                    res.append(result_dict['Cyclist_3d/moderate_R40'])
                    total_score_for_predictor = \
                            (result_dict['Car_3d/moderate_R40'] + result_dict['Pedestrian_3d/moderate_R40'] + result_dict['Cyclist_3d/moderate_R40']) / 300
                    rollouts[i].set_perfs(OrderedDict(zip(['FLOPs', 'Car', 'Pedestrian', 'Cyclist'], res)))
                    rollouts[i].set_perfs(OrderedDict(zip(['reward'], [total_score_for_predictor]))) #!这里设置了reward

            if cfg.LOCAL_RANK == 0:
                controller_loss = model.backbone_2d.controller.step(
                    rollouts, controller_optimizer, perf_name="reward", **kwargs) 

        if cfg.LOCAL_RANK == 0:
            try:
                model.backbone_2d.controller.save(save_path)
            except pickle.PicklingError as e:
                import pdb;pdb.set_trace()

            with torch.no_grad():
                model.backbone_2d.controller.load(save_path)
                with model.backbone_2d.controller.begin_mode("eval"):
                    candidate_num = 30
                    rollouts = model.backbone_2d.controller.sample(candidate_num, batch_size=rollout_batch_size, **kwargs)
                    results = {}
                    for roll in rollouts:
                        if -1 * roll.get_perf("FLOPs") >= 50 or roll.get_perf("reward") < 0.35:
                            continue
                        results[str(roll.arch.values())] = [float(roll.get_perf("reward")), float(roll.get_perf("Car")), float(roll.get_perf("Pedestrian")), float(roll.get_perf("Cyclist")), -1 * roll.get_perf("FLOPs")]
            
            sorted_result = {k:v for k,v in sorted(results.items(), key = lambda kv:kv[1][-1])}
            yaml_file = save_path + "_results.yaml"
            with open(yaml_file, 'w') as f:
                f.write(yaml.dump(sorted_result, allow_unicode=True))

    else:
        with torch.no_grad():
            model.backbone_2d.controller.load(save_path)
            with model.backbone_2d.controller.begin_mode("eval"):
                candidate_num = 30
                rollouts = model.backbone_2d.controller.sample(candidate_num, batch_size=rollout_batch_size, **kwargs)
                results = {}
                for roll in rollouts:
                    if -1 * roll.get_perf("FLOPs") >= 50 or roll.get_perf("reward") < 0.35:
                        continue
                    results[str(roll.arch.values())] = [float(roll.get_perf("reward")), float(roll.get_perf("Car")), float(roll.get_perf("Pedestrian")), float(roll.get_perf("Cyclist")), -1 * roll.get_perf("FLOPs")]
        
        sorted_result = {k:v for k,v in sorted(results.items(), key = lambda kv:kv[1][-1])}
        yaml_file = save_path + "_results.yaml"
        with open(yaml_file, 'w') as f:
            f.write(yaml.dump(sorted_result, allow_unicode=True))
            

        

if __name__ == '__main__':
    main()