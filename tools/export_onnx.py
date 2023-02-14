import _init_path
import argparse
import datetime
import glob
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
import torch.nn as nn
from pcdet.models import load_data_to_gpu


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

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
    parser.add_argument('--eval_ap', action='store_true', default=False, help='whether to evaluate all checkpoints')

    parser.add_argument('--get_workspace', action='store_true', default=False, help='for spconv pretrained model')
    parser.add_argument('--cast_points', action='store_true', default=False, help='for pointnet++ fp16 depoly')
    parser.add_argument('--image_input', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file
    )


data_dict_info = {}

class ExportModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        #self.model.export_onnx(True)

    def forward(self, *args):
        #assert len(args) in [2, 6]
        assert len(args) in [1, 5]
        bev_map = args[0]
        #valid_points = args[1]
        #batch_dict = {'points': points, 'valid_points': valid_points}
        batch_dict = {'bev_map': bev_map}
        if len(args) == 6:
            image = args[2]
            Lidar2Camera = args[3]
            Camera2Image = args[4]
            image_shape = args[5]
            batch_dict['images'] = image
            batch_dict['Lidar2Cam'] = Lidar2Camera
            batch_dict['Cam2Img'] = Camera2Image
            batch_dict['image_shape'] = image_shape

        global data_dict_info
        data_dict_info.update(batch_dict)
        output = self.model(data_dict_info)
        # export_boxes = output['export_boxes']
        # export_boxes = output

        cls_preds = output['export_cls_preds']
        # box_preds = output['export_box_preds']

        # return cls_preds,box_preds
        return output['export_cls_preds'], output["center"], output["center_z"], output["dim"],\
            output["rot"], output["iou"]
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

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    ## 这里开始修改
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    if args.get_workspace: # for spconv and github pretrained
        model.train()
        dataloader = iter(test_loader)
        total = len(test_loader)
        print('*************************** start get spconv workspace ***************************')
        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                load_data_to_gpu(batch_data)
                model(batch_data)
                if (i+1) % 100 == 0:
                    print('*************************** %d / %d ***************************' % (i+1, total))
        print('**************************** %d / %d ***************************' % (i + 1, total))
        print('***************************  get spconv workspace end  ***************************')
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False) # bn层会被改变，用于还原bn

    with torch.no_grad():
        if args.eval_ap:
            eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)


    onnx_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    try:
        onnx_name = args.ckpt[args.ckpt.rindex('/')+1:args.ckpt.rindex('.')] + '_' + args.extra_tag + '.onnx'
    except:
        onnx_name = 'deploy.onnx'
    onnx_path = os.path.join(onnx_dir, onnx_name)

    export_model = ExportModel(model)
    export_model.cuda()

    global data_dict_info
    batch_dict = next(iter(test_loader))
    load_data_to_gpu(batch_dict)
    points = batch_dict['points'].clone()
    print("测试",batch_dict.keys())
    bev_map = batch_dict['bev_map'].clone()
    if args.cast_points: # for point based model and fp16 eval mode
        points_numpy = points.cpu().numpy()
        shape = points_numpy.shape
        temp_path = os.path.join(onnx_dir, 'points_cast_temp.bin')
        points_numpy.tofile(temp_path)
        points = np.fromfile(temp_path, dtype=np.int32).reshape(*shape)
        points = torch.from_numpy(points).int().cuda()
        batch_dict['reinterpret_cast'] = True

    #valid_points = batch_dict['valid_points'].clone()

    #batch_dict.pop('points')
    #batch_dict.pop('valid_points')
    batch_dict.pop('bev_map')
    data_dict_info.update(batch_dict)

    # num_points = points.size(0)
    # new_points = points.new_zeros(int(num_points * 1.5), points.size(1))
    # new_points[:num_points] = points
    # points = new_points.clone().contiguous()
    with torch.no_grad():
        export_model.eval()
        if args.image_input:
            image = data_dict_info['images'].clone()
            Lidar2Camera = data_dict_info['Lidar2Cam'].clone()
            Camera2Image = data_dict_info['Cam2Img'].clone()
            image_shape = data_dict_info['image_shape'].clone()
            if args.cast_points:
                Lidar2Camera_numpy = Lidar2Camera.cpu().numpy()
                shape = Lidar2Camera_numpy.shape
                temp_path = os.path.join(onnx_dir, 'Lidar2Camera_cast_temp.bin')
                Lidar2Camera_numpy.tofile(temp_path)
                Lidar2Camera = np.fromfile(temp_path, dtype=np.int32).reshape(*shape)
                Lidar2Camera = torch.from_numpy(Lidar2Camera).int().cuda()

                Camera2Image_numpy = Camera2Image.cpu().numpy()
                shape = Camera2Image_numpy.shape
                temp_path = os.path.join(onnx_dir, 'Camera2Image_cast_temp.bin')
                Camera2Image_numpy.tofile(temp_path)
                Camera2Image = np.fromfile(temp_path, dtype=np.int32).reshape(*shape)
                Camera2Image = torch.from_numpy(Camera2Image).int().cuda()

            data_dict_info.pop('images')
            data_dict_info.pop('Lidar2Cam')
            data_dict_info.pop('Cam2Img')
            data_dict_info.pop('image_shape')
            torch.onnx.export(export_model, (bev_map, image, Lidar2Camera, Camera2Image, image_shape), onnx_path, verbose=True,
                              training=False, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=9,
                              input_names=['bev_map','image', 'lidar2rect', 'rect2img', 'image_shape'])
        else:
            torch.onnx.export(export_model, (bev_map), onnx_path, verbose=True, training=False,
                              operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=9,
                              input_names=['bev_map'])
            print("导出onnx成功", onnx_path)

if __name__ == '__main__':
    main()
