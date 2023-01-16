import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from icecream import ic 
import functools
from typing import List
from collections import OrderedDict
from aw_nas.germ.utils import divisor_fn

def make_divisible(v, divisor, min_val=None):
    """
    ref: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

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



class Standalone_MVLidarNet(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        multi_input_channels = self.model_cfg.get('MULTI_INPUT_CHANNELS', [3, 7])
        if self.model_cfg.get('ratio', None) is not None:
            ratio = self.model_cfg.ratio

        num_filters = self.model_cfg.NUM_FILTERS
        num_levels = len(num_filters) 
        block_out_channel = [None] * 5
        prev_channels = input_channels

        '''for idx in range(num_levels - 1): # idx表示第几个block/upconv，此处减去上采样的conv channel
            cur_channels = (num_filters[idx] * ratio[idx]).apply(divisor_fn)
            block_out_channel[idx] = cur_channels
            
            cur_layer = []
            for k in range(2):# 5个blocks，每个两层，且每个block内部out channel相同
                cur_layer.extend([conv3x3(
                    prev_channels, 
                    cur_channels, 
                    stride=1 if k==0 or idx == 0 else 2)])
                prev_channels = cur_channels
            self.blocks.append(nn.Sequential(*cur_layer))

        # ------------------- 上采样 ----------------------
        up_channels = [None] * 4
        up_channels[0] = block_out_channel[-2]
        # 原本up1c是conv(256, 128)
        up_channels[1] = (ratio[-1] * num_filters[-1]).apply(divisor_fn)
        up_channels[2] = block_out_channel[-3]
        # 原本up2c是conv(128, 64)
        up_channels[3] = 64

        for i, cur_channels in enumerate(up_channels):
            if i % 2 == 0:
                self.deblocks.append(nn.Sequential(
                    nn.ConvTranspose2d(
                        prev_channels, 
                        cur_channels,
                        kernel_size = 2,
                        stride=2, bias=False,
                    ),
                    nn.BatchNorm2d(cur_channels),
                    nn.ReLU()
                ))
            else:
                self.deblocks.append(conv3x3(
                        prev_channels * 2, 
                        cur_channels, 
                        stride=1))
            prev_channels = cur_channels
'''
        
        assert len(num_filters) == len(ratio)
        channel_size = [make_divisible(num_filters[i] * ratio[i], 8) for i in range(len(num_filters))]
        self.height = nn.Sequential(conv3x3(multi_input_channels[0], channel_size[0]),
                                 conv3x3(channel_size[0], channel_size[0]),
                                 conv3x3(channel_size[0], channel_size[1], 2),
                                 conv3x3(channel_size[1], channel_size[1]))
        
        self.block1a = conv3x3(channel_size[1], channel_size[2])
        self.block1b = conv3x3(channel_size[2], channel_size[2], 2)
        self.block2a = conv3x3(channel_size[2], channel_size[3])
        self.block2b = conv3x3(channel_size[3], channel_size[3], 2)
        self.block3a = conv3x3(channel_size[3], channel_size[4])
        self.block3b = conv3x3(channel_size[4], channel_size[4], 2)
        
        self.up1a = nn.Sequential(
                nn.ConvTranspose2d(channel_size[4], channel_size[3], 2, stride=2, bias=False),
                nn.BatchNorm2d(channel_size[3], eps=1e-3, momentum=0.01),
                nn.ReLU(),)
        self.up1c = conv3x3(channel_size[3] * 2, channel_size[5])
        
        self.up2a = nn.Sequential(
                nn.ConvTranspose2d(channel_size[5], channel_size[2], 2, stride=2, bias=False),
                nn.BatchNorm2d(channel_size[2], eps=1e-3, momentum=0.01),
                nn.ReLU(),)
        self.up2c = conv3x3(channel_size[2] * 2, 64)

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


class Standalone_Pointpillar(nn.Module):
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
        # c_in_list = [input_channels, *num_filters[:-1]]
        if self.model_cfg.get('ratio', None) is not None:
            ratio = self.model_cfg.ratio
        i = 0
        c_in_list = [input_channels, int(num_filters[0] * ratio[3]), int(num_filters[1] * ratio[9])]
        new_num_upsample_filters = c_in_list[1:] + [int(num_filters[2] * ratio[-1])]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], int(num_filters[idx] * ratio[i]), kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(int(num_filters[idx] * ratio[i]), eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            i += 1
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(int(num_filters[idx] * ratio[i-1]), int(num_filters[idx] * ratio[i]), kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(int(num_filters[idx] * ratio[i]), eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
                i += 1
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            new_num_upsample_filters[idx], num_upsample_filters[idx],
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
