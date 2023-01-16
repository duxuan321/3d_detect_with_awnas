import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution with padding"""
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, stride=stride, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()]
    return nn.Sequential(*layers)

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

        return x

def upconv(in_channels, out_channels):
    return PixelShuffle_v3(in_channels, out_channels, nn.BatchNorm2d)


class YTXBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        self.block1 = nn.Sequential(conv3x3(input_channels, 64),
                                 conv3x3(64, 64,2),
                                 conv3x3(64, 64),
                                 conv3x3(64, 64),
                                 conv3x3(64, 64)
                                 )
    
        self.block2 = nn.Sequential(conv3x3(64, 128,2),
                                 conv3x3(128, 128),
                                 conv3x3(128, 128),
                                 conv3x3(128, 128),
                                 conv3x3(128, 128),
                                 conv3x3(128, 128),
                                 conv3x3(128, 128)
                                 ) 
        
        self.block3 = nn.Sequential(conv3x3(128, 256,2),
                                 conv3x3(256, 256),
                                 conv3x3(256, 256),
                                 conv3x3(256, 256),
                                 conv3x3(256, 256),
                                 conv3x3(256, 256),
                                 conv3x3(256, 256)
                                 ) 
        
        self.up1 = upconv(256, 128)
        self.block4 = nn.Sequential(conv3x3(256, 256),
                                 conv3x3(256, 256),
                                 conv3x3(256, 128),
                                 conv3x3(128, 128),
                                 conv3x3(128, 128)
                                 ) 
        
        self.up2 = upconv(128, 64)
        self.block5 = nn.Sequential(conv3x3(128, 128),
                                 conv3x3(128, 128),
                                 conv3x3(128, 128)
                                 ) 

        self.num_bev_features = 128

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        fea = data_dict['spatial_features']

        fea1 = self.block1(fea)
        fea2 = self.block2(fea1)
        fea3 = self.block3(fea2)

        up1 = self.up1(fea3)
        up1 = torch.cat([up1, fea2], 1)
        up1 = self.block4(up1)

        up2 = self.up2(up1)
        up2 = torch.cat([up2, fea1], 1)
        up2 = self.block5(up2)
        
        data_dict['spatial_features_2d'] = up2

        return data_dict
