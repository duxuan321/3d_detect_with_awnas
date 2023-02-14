import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

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

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution with padding"""
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, stride=stride, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()]
    return nn.Sequential(*layers)

class PixelShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, norm_fn, scale=2):
        super(PixelShuffle, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels * scale * scale, 1, stride=1, padding=0, bias=False),
                                    norm_fn(out_channels * scale *scale),
                                    nn.ReLU())
        self.scale = scale
    def forward(self, x):
        x = self.conv(x)
        b = int(x.size(0))
        c = int(x.size(1))
        h = int(x.size(2))
        w = int(x.size(3))
        x = x.view(b, c // self.scale // self.scale, self.scale, self.scale, h, w).permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // self.scale // self.scale, h * self.scale, w * self.scale)
        return x

class PixelShuffle_v2(nn.Module):
    def __init__(self, in_channels, out_channels, norm_fn, scale=2):
        super(PixelShuffle_v2, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels * scale * scale, 1, stride=1, padding=0, bias=False),
                                    norm_fn(out_channels * scale * scale),
                                    nn.ReLU())
        self.scale = scale
        # self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample = torch.nn.PixelShuffle(2)
    def forward(self, x):
        x = self.conv(x)
        b = int(x.size(0))
        c = int(x.size(1))
        h = int(x.size(2))
        w = int(x.size(3))
        # x = x.view(b, c // self.scale // self.scale, self.scale, self.scale, h, w).permute(0, 1, 4, 2, 5, 3).contiguous()
        # x = x.view(b, c // self.scale // self.scale, h * self.scale, w * self.scale)
        # x = x.permute(0, 2, 3, 1).contiguous()
        # x = x.view(b, h, w * self.scale, c // self.scale)
        # x = x.permute(0, 2, 1, 3).contiguous()
        # x = x.view(b, w * self.scale, h * self.scale, c // self.scale // self.scale)
        # x = x.permute(0, 3, 2, 1).contiguous()

        # # 默认是NHWC格式
        # x = x.view(b, h, w * self.scale, c // self.scale)
        # x = x.permute(0, 2, 1, 3).contiguous()
        # x = x.view(b, w * self.scale, h * self.scale, c // self.scale // self.scale)
        # x = x.permute(0, 2, 1, 3).contiguous()

        x = self.upsample(x)

        return x
        

class PixelShuffle_v3(nn.Module):
    def __init__(self, in_channels, out_channels, norm_fn, scale=2):
        super(PixelShuffle_v3, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels * scale * scale, 1, stride=1, padding=0, bias=False),
                                    norm_fn(out_channels * scale * scale),
                                    nn.ReLU())
        self.scale = scale
        # self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        # self.upsample = torch.nn.PixelShuffle(2)
    def forward(self, x):
        x = self.conv(x)
        b = int(x.size(0))
        c = int(x.size(1))
        h = int(x.size(2))
        w = int(x.size(3))
        # x = x.view(b, c // self.scale // self.scale, self.scale, self.scale, h, w).permute(0, 1, 4, 2, 5, 3).contiguous()
        # x = x.view(b, c // self.scale // self.scale, h * self.scale, w * self.scale)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(b, h, w * self.scale, c // self.scale)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b, w * self.scale, h * self.scale, c // self.scale // self.scale)
        x = x.permute(0, 3, 2, 1).contiguous()

        # x = self.upsample(x)

        return x

def upconv(in_channels, out_channels):
    # layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
    #         nn.BatchNorm2d(out_channels),
    #         nn.ReLU()]
    # return nn.Sequential(*layers)

    # return PixelShuffle(in_channels, out_channels, nn.BatchNorm2d)
    return PixelShuffle_v3(in_channels, out_channels, nn.BatchNorm2d)



class MVLidarNetBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        multi_input_channels = self.model_cfg.get('MULTI_INPUT_CHANNELS', [3, 7])
        """
        self.sem = nn.Sequential(conv3x3(multi_input_channels[1], 16),
                                 conv3x3(16, 16),
                                 conv3x3(16, 32, 2),
                                 conv3x3(32, 32))
        """
        self.height = nn.Sequential(conv3x3(multi_input_channels[0], 16),
                                 conv3x3(16, 16),
                                 conv3x3(16, 32, 2),
                                 conv3x3(32, 64))   # 这里改成了32 64，因为暂时没有分割信息
        
        self.block1a = conv3x3(64, 64)
        self.block1b = conv3x3(64, 64, 2)
        self.block2a = conv3x3(64, 128)
        self.block2b = conv3x3(128, 128, 2)
        self.block3a = conv3x3(128, 256)
        self.block3b = conv3x3(256, 256, 2)
        
        self.up1a = upconv(256, 128)
        self.up1c = conv3x3(256, 128)
        
        self.up2a = upconv(128, 64)
        self.up2c = conv3x3(128, 64)

        self.num_bev_features = 64

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        height_feat = data_dict['spatial_features']
        # print(height_feat.shape)
        
        # import skimage
        # jpg = height_feat.detach().cpu().squeeze().numpy()
        # print('-----------------------', jpg.min(), jpg.max())
        # jpg = ((jpg - jpg.min()) / (jpg.max() - jpg.min()) * 255 ).astype(np.ubyte)
        # jpg = jpg.transpose(1,2,0)
        # skimage.io.imsave('./bev.jpg', jpg)

        height_feat = self.height(height_feat)
        
        f_block1a = self.block1a(height_feat)
        f_block1b = self.block1b(f_block1a)
        f_block2a = self.block2a(f_block1b)
        f_block2b = self.block2b(f_block2a)
        f_block3a = self.block3a(f_block2b)
        f_block3b = self.block3b(f_block3a)
        

        f_up1a = self.up1a(f_block3b)
        f_up1b = torch.cat([f_up1a, f_block2b], 1)
        f_up1c = self.up1c(f_up1b)
        
        f_up2a = self.up2a(f_up1c)
        f_up2b = torch.cat([f_up2a, f_block1b], 1)
        f_up2c = self.up2c(f_up2b)

        data_dict['spatial_features_2d'] = f_up2c

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

class MVLidarNetBackboneSuperNet(germ.GermSuperNet):
    
    def __init__(self, model_cfg, input_channels,
                search_space,
                mult_ratio_choices=(1.0,),
                force_use_ordinal_channel_handler=False,
                schedule_cfg={},
                controller_type=None,
                controller_cfg=None):
        # nn.Module.__init__(self)
        super().__init__(search_space)

        self.mult_ratio_choices = mult_ratio_choices

        self.model_cfg = model_cfg
        self.eval_mode = False

        width_choices_cb = functools.partial(
            schedule_choice_callback, schedule=schedule_cfg.get("mult_ratio_choices")
        )

        num_filters = self.model_cfg.NUM_FILTERS #[24, 48, 96, 192, 384, xxx]
        num_levels = len(num_filters) 
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        prev_channels = input_channels
        total_choice_num = num_levels * len(mult_ratio_choices) #todo:check!!! 5个block + up1c的out channel
        block_out_channel = [None] * 5

        with self.begin_searchable() as ctx:
            for idx in range(num_levels - 1): # idx表示第几个block/upconv，此处减去上采样的conv channel
                cur_channels = (germ.Choices(mult_ratio_choices, epoch_callback=width_choices_cb) * num_filters[idx]).apply(divisor_fn)
                block_out_channel[idx] = cur_channels
                
                cur_layer = []
                for k in range(2):# 5个blocks，每个两层，且每个block内部out channel相同
                    cur_layer.extend([germ.SearchableConvBNBlock(
                        ctx, 
                        prev_channels, 
                        cur_channels, 
                        kernel_size=3, 
                        stride=1 if k==0 or idx == 0 else 2,
                        force_use_ordinal_channel_handler=force_use_ordinal_channel_handler), # 除第一个外每个block的第二个layer进行下采样
                        nn.ReLU()])
                    prev_channels = cur_channels
                self.blocks.append(nn.Sequential(*cur_layer))

            # ------------------- 上采样 ----------------------
            up_channels = [None] * 4
            up_channels[0] = block_out_channel[-2]
            # 原本up1c是conv(256, 128)
            up_channels[1] = (germ.Choices(mult_ratio_choices, epoch_callback=width_choices_cb) * num_filters[-1]).apply(divisor_fn)
            up_channels[2] = block_out_channel[-3]
            # 原本up2c是conv(128, 64)
            up_channels[3] = 64

            for i, cur_channels in enumerate(up_channels):
                if i % 2 == 0:
                    self.deblocks.append(nn.Sequential(
                        germ.Searchable_ConvTranspose2d(
                            ctx,
                            prev_channels, 
                            cur_channels, # todo: check out channel, ks, stride
                            kernel_size = 2,
                            stride=2, bias=False, 
                            force_use_ordinal_channel_handler=force_use_ordinal_channel_handler
                        ),
                        germ.SearchableBN(ctx, cur_channels, eps=1e-3, momentum=0.01, force_use_ordinal_channel_handler=force_use_ordinal_channel_handler),
                        nn.ReLU()
                    ))
                else:
                    self.deblocks.append(nn.Sequential(
                        germ.SearchableConvBNBlock(
                            ctx, 
                            prev_channels * 2, 
                            cur_channels, 
                            kernel_size=3, 
                            stride=1,
                            force_use_ordinal_channel_handler=force_use_ordinal_channel_handler),
                            nn.ReLU()))
                prev_channels = cur_channels

        device = torch.cuda.current_device()
        controller_cfg['arch_network_cfg']['arch_embedder_cfg']['total_choice_num'] = total_choice_num
        ic(total_choice_num)
        self.controller = BaseController.get_class_(controller_type)(search_space, device, rollout_type="germ", **controller_cfg)

        self.num_bev_features = 64
    
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
        x = spatial_features

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i == 2:
                block2_out = x
            elif i == 3:
                block3_out = x

        # import pdb;pdb.set_trace()
        f_up1a = self.deblocks[0](x)
        f_up1b = torch.cat([f_up1a, block3_out], 1)
        f_up1c = self.deblocks[1](f_up1b)
        
        f_up2a = self.deblocks[2](f_up1c)
        f_up2b = torch.cat([f_up2a, block2_out], 1)
        f_up2c = self.deblocks[3](f_up2b)

        data_dict['spatial_features_2d'] = f_up2c

        if self.eval_mode and self.training:
            self.eval()

        return data_dict
