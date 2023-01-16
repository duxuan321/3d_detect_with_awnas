
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
from pcdet.models import load_data_to_gpu
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

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, geno=None, get_flops=False):
    ic(dist_test)
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if dist_test and hasattr(model.module, 'backbone_2d') and hasattr(model.module.backbone_2d, 'ctx'):
        model.module.backbone_2d.search_space.on_epoch_start(int(epoch_id))
        assert model.module.gene_type is not None #多线程测试时暂不支持随机arch测试
    elif hasattr(model, 'backbone_2d') and hasattr(model.backbone_2d, 'ctx'):
        model.backbone_2d.search_space.on_epoch_start(int(epoch_id))
        if model.gene_type is None:
            model.gene_type = model.backbone_2d.search_space.random_sample().genotype
    
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if get_flops and cfg.LOCAL_RANK == 0:
        flops, params = profile(model, inputs=(batch_dict,)) #flops单位G，para单位M
        flops /= batch_dict['batch_size']

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)
    
    if get_flops:
        result_dict['flops'] = (flops/1e9, params/1e6)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')

    if geno is None:
        return ret_dict
    else:
        return result_dict


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
    
    #! load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    rollout_batch_size = 1
    controller_optimizer = None
    dir_ = "/".join(args.cfg_file.split('/')[:-1]) + '/'
    group_res_file = os.path.join(dir_, "group_res.txt")
    ic(dir_)

    # ----------------------------- 一些有用的参数 ---------------------------
    model_type = cfg.MODEL.BACKBONE_2D.NAME
    num_filters = np.array(cfg['MODEL']['BACKBONE_2D']['NUM_FILTERS'])
    layer_nums = cfg['MODEL']['BACKBONE_2D']['LAYER_NUMS'] if cfg['MODEL']['BACKBONE_2D'].get("LAYER_NUMS", None) else None
    fix_channel = model.backbone_2d.fix_block_channel if hasattr(model.backbone_2d, "fix_block_channel") else None
    multi_ratio_choice = np.array(model.backbone_2d.mult_ratio_choices)
    all_sample_size = args.controller_samples * args.epoch_num
    save_path = "/".join(args.ckpt.split('/')[:-2]) + "/evo_" + str(all_sample_size)
    kwargs = {
        "model_type": model_type,
        "num_filters": num_filters, 
        "layer_nums": layer_nums,
        "fix_channel": fix_channel,
        "multi_ratio_choice": multi_ratio_choice
    }
    # ----------------------------------------------------------------------
    model.backbone_2d.controller.set_mode("train")
    for ep in range(args.epoch_num):
        ic(ep)
        rollouts = model.backbone_2d.controller.sample(\
            args.controller_samples, batch_size=rollout_batch_size, **kwargs)
        for i, roll in enumerate(rollouts):
            geno = roll.genotype
            model.gene_type = geno

            final_model = copy.deepcopy(model)
            rollout_temp = final_model.backbone_2d.search_space.rollout_from_genotype(geno)
            final_model.backbone_2d = final_model.backbone_2d.finalize_rollout(rollout_temp)
            final_model.cuda()

            # -------------------------------- 计算实际的out channel数值 ------------------------------------
            setattr(roll, "arch_for_pred", OrderedDict())
            if "BEVBackbone" in model_type:
                curr_ratio_choices = np.array(list(roll.arch.values())[:-3].copy())
                if fix_channel:
                    out_channel = num_filters * curr_ratio_choices
                    indices_mapping = num_filters.reshape(-1,1) * multi_ratio_choice.reshape(1,-1)
                else:
                    origin_channel = np.array(sum([(layer_nums[i] + 1) * [num_filters[i]] for i in range(len(layer_nums))], []))
                    out_channel = origin_channel * curr_ratio_choices
                    indices_mapping = origin_channel.reshape(-1,1) * multi_ratio_choice.reshape(1,-1)
                
            elif "MVLidarNet" in model_type:
                curr_ratio_choices = np.array(list(roll.arch.values()).copy())
                out_channel = num_filters * curr_ratio_choices
                indices_mapping = num_filters.reshape(-1,1) * multi_ratio_choice.reshape(1,-1)

            ind = [np.where(indices_mapping[i] == out_channel[i])[0][0] for i in range(len(out_channel))]
            for i, (key,val) in enumerate(roll.arch.items()):
                roll.arch_for_pred[key] = ind[i]
                if i == len(ind) - 1:
                    break
            # ----------------------------------------------------------------------------------------------

            # start evaluation
            result_dict = eval_one_epoch(
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
                roll.set_perfs(OrderedDict(zip(['FLOPs', 'Car', 'Pedestrian', 'Cyclist'], res)))
                roll.set_perfs(OrderedDict(zip(['reward'], [total_score_for_predictor]))) #!这里设置了reward

        if cfg.LOCAL_RANK == 0:
            controller_loss = model.backbone_2d.controller.step(
                rollouts, controller_optimizer, perf_name="reward") 

    if cfg.LOCAL_RANK == 0:
        try:
            model.backbone_2d.controller.save(save_path)
        except pickle.PicklingError as e:
            import pdb;pdb.set_trace()

    # import pdb;pdb.set_trace()
    with torch.no_grad():
        model.backbone_2d.controller.load(save_path)
        with model.backbone_2d.controller.begin_mode("eval"):
        # model.backbone_2d.controller.set_mode("train")
            # for ep in range(args.epoch_num):
            import pdb;pdb.set_trace()
            rollouts = model.backbone_2d.controller.sample(all_sample_size, batch_size=rollout_batch_size, **kwargs)
            results = {}
            for roll in rollouts:
                if -1 * roll.get_perf("FLOPs") >= 50 or roll.get_perf("reward") < 0.35:
                    continue
                results[str(roll.arch.values())] = [float(roll.get_perf("reward")), float(roll.get_perf("Car")), float(roll.get_perf("Pedestrian")), float(roll.get_perf("Cyclist")), -1 * roll.get_perf("FLOPs")]
    
    sorted_result = {k:v for k,v in sorted(results.items(), key = lambda kv:kv[1][-1])}
    yaml_file = save_path + "results.yaml"
    with open(yaml_file, 'w') as f:
        f.write(yaml.dump(sorted_result, allow_unicode=True))
            

        

if __name__ == '__main__':
    main()