import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch import nn
import aw_nas
from aw_nas import germ
from aw_nas.ops import get_op, MobileNetV2Block
from aw_nas.utils import feature_level_to_stage_index
from aw_nas.germ.utils import divisor_fn
from aw_nas.controller.evo import ParetoEvoController
from aw_nas.controller.base import BaseController
from icecream import ic 
import functools
from typing import List
from collections import OrderedDict

class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict


def schedule_choice_callback(
    choices: germ.Choices, epoch: int, schedule: List[dict]
) -> None:
    """
    Args:
        choices: instances of Choices
        epoch: int
        schedule: list
            [
                {
                    "epoch": int,
                    "choices": list,
                },
                ...
            ]
    """
    if schedule is None:
        return
    for sch in schedule:
        assert "epoch" in sch and "choices" in sch
        if epoch >= sch["epoch"]:
            choices.choices = sch["choices"]
    print(
        "Epoch: {:>4d}, decision id: {}, choices: {}".format(
            epoch, choices.decision_id, choices.choices
        )
    )


class BEVBackboneSuperNet(germ.GermSuperNet):
    
    def __init__(self, model_cfg, input_channels, 
                search_space,
                num_classes=10,
                depth_choices=[2, 3, 4],
                # strides=[2, 2, 2, 1, 2, 1],
                mult_ratio_choices=(1.0,),
                kernel_sizes=[3, 5, 7],
                # expansion_choices=[2, 3, 4, 6],
                activation="relu",
                pretrained_path=None,
                fix_block_channel=True,
                force_use_ordinal_channel_handler=False,
                schedule_cfg={},
                controller_type=None,
                controller_cfg=None):
        # nn.Module.__init__(self)
        super().__init__(search_space)

        self.depth_choices = depth_choices
        self.kernel_sizes = kernel_sizes
        self.mult_ratio_choices = mult_ratio_choices
        self.fix_block_channel = fix_block_channel

        self.model_cfg = model_cfg
        self.eval_mode = False

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        kernel_choices_cb = functools.partial(
            schedule_choice_callback, schedule=schedule_cfg.get("kernel_sizes")
        )
        width_choices_cb = functools.partial(
            schedule_choice_callback, schedule=schedule_cfg.get("mult_ratio_choices")
        )

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        prev_channels = input_channels
        total_choice_num = len(layer_nums) * len(mult_ratio_choices) if self.fix_block_channel else (sum(layer_nums) + len(layer_nums)) * len(mult_ratio_choices)

        with self.begin_searchable() as ctx:
            for idx in range(num_levels):
                # 每个block内部out channel相同
                if self.fix_block_channel:
                    cur_channels = (germ.Choices(mult_ratio_choices, epoch_callback=width_choices_cb) * num_filters[idx]).apply(divisor_fn)
                cur_layer = []
                for k in range(layer_nums[idx] + 1):
                    # 每个block内部的out channel不同
                    if not self.fix_block_channel:
                        cur_channels = (germ.Choices(mult_ratio_choices, epoch_callback=width_choices_cb) * num_filters[idx]).apply(divisor_fn)
                    cur_layer.extend([germ.SearchableConvBNBlock(
                        ctx, 
                        prev_channels, 
                        cur_channels, 
                        kernel_size=germ.Choices(kernel_sizes, epoch_callback=kernel_choices_cb) if kernel_sizes else 3, 
                        stride=layer_strides[idx] if k==0 else 1,
                        force_use_ordinal_channel_handler=force_use_ordinal_channel_handler),
                        nn.ReLU()])
                    prev_channels = cur_channels
                self.blocks.append(nn.Sequential(*cur_layer))

                stride = upsample_strides[idx]
                if len(upsample_strides) > 0:
                    stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        germ.Searchable_ConvTranspose2d(
                            ctx,
                            cur_channels, (germ.Choices([1]) * num_upsample_filters[idx]).apply(divisor_fn), #todo:是否搜索上采样的out channel？
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False, 
                            force_use_ordinal_channel_handler=force_use_ordinal_channel_handler
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01), #todo:这里是否需要searchable bn？
                        nn.ReLU()
                    ))
                # else:
                #     stride = np.round(1 / stride).astype(np.int)
                #     self.deblocks.append(nn.Sequential(
                #         nn.Conv2d(
                #             num_filters[idx], num_upsample_filters[idx],
                #             stride,
                #             stride=stride, bias=False
                #         ),
                #         nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                #         nn.ReLU()
                #     ))

        device = torch.cuda.current_device()
        controller_cfg['arch_network_cfg']['arch_embedder_cfg']['total_choice_num'] = total_choice_num
        ic(total_choice_num)
        self.controller = BaseController.get_class_(controller_type)(search_space, device, rollout_type="germ", **controller_cfg)

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in
    
    def forward(self, data_dict, test_gene=None):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        
        if not self.training:
            self.eval_mode = True
        else:
            self.ctx.rollout = self.search_space.random_sample()

        if self.eval_mode or test_gene is not None: # 测试和finetune的时候固定网络架构
            self.ctx.rollout = self.search_space.rollout_from_genotype(test_gene)
            self.train()

        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        # ic(self.ctx.rollout.arch)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        if self.eval_mode and self.training:
            self.eval()

        return data_dict
