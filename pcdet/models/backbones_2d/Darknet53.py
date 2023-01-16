import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

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

class Conv(nn.Module):
	def __init__(self, c_in, c_out, k, s, p, bias=True):
		"""
		自定义一个卷积块，一次性完成卷积+归一化+激活，这在类似于像DarkNet53这样的深层网络编码上可以节省很多代码
		:param c_in: in_channels，输入通道
		:param c_out: out_channels，输出通道
		:param k: kernel_size，卷积核大小
		:param s:  stride，步长
		:param p: padding，边界扩充
		:param bias: …
		"""
		super(Conv, self).__init__()
		self.conv = nn.Sequential(
			#卷积
			nn.Conv2d(c_in, c_out, k, s, p),
			#归一化
			nn.BatchNorm2d(c_out),
			#激活函数
			nn.LeakyReLU(0.1),
		)

	def forward(self, entry):
		return self.conv(entry)

class ConvResidual(nn.Module):
	def __init__(self, c_in):		# converlution * 2 + residual
		"""
		自定义残差单元，只需给出通道数，该单元完成两次卷积，并进行加残差后返回相同维度的特征图
		:param c_in: 通道数
		"""
		c = c_in // 2
		super(ConvResidual, self).__init__()
		self.conv = nn.Sequential(
			Conv(c_in, c, 1, 1, 0),		 # kernel_size = 1进行降通道
			Conv(c, c_in, 3, 1, 1),		 # 再用kernel_size = 3把通道升回去
		)

	def forward(self, entry):
		return entry + self.conv(entry)	 # 加残差，既保留原始信息，又融入了提取到的特征
# 采用 1*1 + 3*3 的形式加深网络深度，加强特征抽象

class Darknet53(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super(Darknet53, self).__init__()
        self.model_cfg = model_cfg
        multi_input_channels = self.model_cfg.get('MULTI_INPUT_CHANNELS', [3, 7])

        self.conv1 = Conv(multi_input_channels[0], 32, 3, 1, 1)			# 一个卷积块 = 1层卷积
        self.conv2 = Conv(32, 64, 3, 2, 1)
        self.conv3_4 = ConvResidual(64)				# 一个残差块 = 2层卷积
        self.conv5 = Conv(64, 128, 3, 2, 1)
        self.conv6_9 = nn.Sequential(				# = 4层卷积
            ConvResidual(128),
            ConvResidual(128),
        )
        self.conv10 = Conv(128, 256, 3, 2, 1)
        self.conv11_26 = nn.Sequential(				# = 16层卷积
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
            ConvResidual(256),
        )
        self.conv27 = Conv(256, 512, 3, 2, 1)
        self.conv28_43 = nn.Sequential(				# = 16层卷积
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
            ConvResidual(512),
        )
        self.conv44 = Conv(512, 1024, 3, 2, 1)
        self.conv45_52 = nn.Sequential(				# = 8层卷积
            ConvResidual(1024),
            ConvResidual(1024),
            ConvResidual(1024),
            ConvResidual(1024) )

        # spp
        self.upconv01 = nn.Sequential(Conv(1024, 512, 1, 1, 0),
                    Conv(512, 1024, 3, 1, 1),
                    Conv(1024, 512, 1, 1, 0)
                )
        self.maxpool1 = nn.MaxPool2d(5,stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(9,stride=1, padding=4)
        self.maxpool3 = nn.MaxPool2d(13,stride=1, padding=6)
        self.upconv02 = nn.Sequential(Conv(2048, 512, 1, 1, 0),
                    Conv(512, 1024, 3, 1, 1),
                    Conv(1024, 512, 1, 1, 0),
                    Conv(512, 256, 1, 1, 0)
                )

        self.up1 = upconv(256,256)
        self.upconv1 = nn.Sequential(Conv(768, 256, 1, 1, 0),
                    Conv(256, 512, 3, 1, 1),
                    Conv(512, 256, 1, 1, 0),
                    Conv(256, 512, 3, 1, 1),
                    Conv(512, 256, 1, 1, 0),
                    Conv(256, 128, 1, 1, 0),
                )
        self.up2 = upconv(128,128)
        self.upconv2 = nn.Sequential(Conv(384, 128, 1, 1, 0),
                    Conv(128, 256, 3, 1, 1),
                    Conv(256, 128, 1, 1, 0),
                    Conv(128, 256, 3, 1, 1),
                    Conv(256, 128, 1, 1, 0),
                    Conv(128, 64, 1, 1, 0),
                    Conv(64, 128, 3, 1, 1),
                    Conv(128, 64, 1, 1, 0)
                )
        # self.up3 = upconv(64,64)
        # self.upconv3 = nn.Sequential(Conv(192, 64, 1, 1, 0),
        #             Conv(64, 128, 3, 1, 1),
        #             Conv(128, 64, 1, 1, 0),
        #             Conv(64, 128, 3, 1, 1),
        #             Conv(128, 64, 1, 1, 0),
        #             Conv(64, 128, 3, 1, 1),
        #             Conv(128, 64, 1, 1, 0),
        #         )
        self.num_bev_features = 64


    def forward(self, data_dict):
        entry = data_dict['spatial_features']

        conv1 = self.conv1(entry)
        conv2 = self.conv2(conv1)
        conv3_4 = self.conv3_4(conv2)
        conv5 = self.conv5(conv3_4)
        conv6_9 = self.conv6_9(conv5)
        conv10 = self.conv10(conv6_9)
        conv11_26 = self.conv11_26(conv10)
        conv27 = self.conv27(conv11_26)
        conv28_43 = self.conv28_43(conv27)
        conv44 = self.conv44(conv28_43)
        conv45_52 = self.conv45_52(conv44)

        # SPP
        pool1 = self.upconv01(conv45_52)
        pool2 = self.maxpool1(pool1)
        pool3 = self.maxpool1(pool1)
        pool4 = self.maxpool1(pool1)
        spp = torch.cat([pool1, pool2,pool3,pool4], 1)
        spp = self.upconv02(spp)

        up1a = self.up1(spp)
        up1a =  torch.cat([up1a, conv28_43], 1)
        up1b = self.upconv1(up1a)

        up2a = self.up2(up1b)
        up2a =  torch.cat([up2a, conv11_26], 1)
        up2b = self.upconv2(up2a)

        # up3a = self.up3(up2b)
        # up3a =  torch.cat([up3a, conv6_9], 1)
        # up3b = self.upconv3(up3a)

        # print("test",up2b.shape)
        data_dict['spatial_features_2d'] = up2b
        return data_dict	# YOLOv3用，所以输出了3次特征

