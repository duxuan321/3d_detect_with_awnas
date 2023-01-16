import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        
        # batch_dict['spatial_features'] = batch_spatial_features.permute(0,1,3,2)
        batch_dict['spatial_features'] = batch_spatial_features
        # print("??",batch_dict['spatial_features'].shape)
        return batch_dict

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution with padding"""
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, stride=stride, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()]
    return nn.Sequential(*layers)

def conv1x1(in_channels, out_channels, stride=1, padding=0):
    """3x3 convolution with padding"""
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=padding, stride=stride, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()]
    return nn.Sequential(*layers)

class BEV_scatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        self.conv1 = nn.Sequential(conv3x3(3, 16),
                                 conv3x3(16, 16),
                                 conv3x3(16, 32),
                                 conv3x3(32, self.num_bev_features))
        # self.conv1 = nn.Sequential(conv1x1(3, 16),
        #                          conv1x1(16, 16),
        #                          conv1x1(16, 32),
        #                          conv1x1(32, 64))
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        # 该模块原本用于mvlidarnet，而该模型的head的宽高和正常是反过来的，现在移植到正常的模型，需要permute
        bev_fea = batch_dict['bev_map'].permute(0,1,3,2)       
        bev_fea = self.conv1(bev_fea)

        batch_dict['spatial_features'] = bev_fea
        # print("??",batch_dict['spatial_features'].shape)
        return batch_dict


class BEV_scatter_reverse(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        self.conv1 = nn.Sequential(conv3x3(3, 16),
                                 conv3x3(16, 16),
                                 conv3x3(16, 32),
                                 conv3x3(32, 64))
        # self.conv1 = nn.Sequential(conv1x1(3, 16),
        #                          conv1x1(16, 16),
        #                          conv1x1(16, 32),
        #                          conv1x1(32, 64))
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        # 这里特征图和正常的相反
        bev_fea = batch_dict['bev_map']     
        bev_fea = self.conv1(bev_fea)

        batch_dict['spatial_features'] = bev_fea
        # print("??",batch_dict['spatial_features'].shape)
        return batch_dict


# 用于将mvlidarnet的BEV特征翻转到正常的顺序
class BEV_mvlidarnet(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size


    def forward(self, batch_dict, **kwargs):
        # 该模块原本用于mvlidarnet，而该模型的head的宽高和正常是反过来的，现在移植到正常的模型，需要permute
        bev_fea = batch_dict['bev_map'].permute(0,1,3,2)   

        batch_dict['spatial_features'] = bev_fea
        # print("??",batch_dict['spatial_features'].shape)
        return batch_dict