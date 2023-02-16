import copy
from this import d
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils
from ...ops.iou3d_nms import iou3d_nms_utils

import torch.nn.functional as F


class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class CenterHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        # new
        self.export_onnx = self.model_cfg.get("EXPORT_ONNX",False)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        ## 扩充项  ##############################################################################################
        self.label_assign_flag = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('LABEL_ASSIGN_FLAG', 'commen')
        self.postprocess = self.model_cfg.POST_PROCESSING.get('POSTPROCESS_TYPE', 'nms')

        self.iou_loss_flag = self.model_cfg.get('WITH_IOU_LOSS', False)
        self.iou_loss_type = self.model_cfg.get('IOU_LOSS_TYPE', 'IOU_HEI')
        self.iou_loss_weight = self.model_cfg.get('IOU_WEIGHT', 1)
        self.iou_aware_flag = self.model_cfg.get('WITH_IOU_AWARE_LOSS', False)
        self.iou_aware_weight = self.model_cfg.get('IOU_AWARE_WEIGHT', 1)
        self.iou_cls_weight = self.model_cfg.get('IOU_CLS_WEIGHT', [0.15,0.5,0.2])
        ############################################################################################################

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            nn.BatchNorm2d(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
                )
            )
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

        # iou loss
        if self.iou_loss_flag:
            self.add_module('iou_loss_func', loss_utils.IOULoss(self.feature_map_stride,self.voxel_size[0]))

        # IOU_AWARE
        if self.iou_aware_flag:
            self.add_module('iou_aware_loss_func', loss_utils.RegLossCenterNet())

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask


    def assign_target_of_single_head_v2(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        版本二：这个版本在中心点附近也分配了一些正例框，这里一共有五个，分别为中心+上下左右
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((5*num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(5*num_max_objs).long()
        mask = gt_boxes.new_zeros(5*num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            # print("测试：",gt_boxes[k])
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

            # centernet_utils.draw_gaussian_to_heatmap_v2(heatmap[cur_class_id], center[k], [dx[k],dy[k]])
            # print("??",heatmap[cur_class_id,int(center[k][1])-5:int(center[k][1])+6,(int(center[k][0])-5):int(center[k][0])+6])

            inds[k*5] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k*5:k*5+5] = 1
            # print(gt_boxes[k, 3:6])

            ret_boxes[k*5:k*5+5, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k*5:k*5+5, 2] = z[k]
            ret_boxes[k*5:k*5+5, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k*5:k*5+5, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k*5:k*5+5, 7] = torch.sin(gt_boxes[k, 6])

            temp_l = gt_boxes[k,3]
            temp_w = gt_boxes[k,4]
            # temp_yaw = abs(gt_boxes[k,-2])
            # if( temp_yaw<0.39 or temp_yaw>2.75 or (temp_yaw>1.18 and temp_yaw<1.96) ):
            #     temp_l = gt_boxes[k,4]
            #     temp_w = gt_boxes[k,3]

            if (self.voxel_size[0] * feature_map_stride > temp_w/4):
                    mask[k*5+1:k*5+3] = 0
                    # print("hello")
            else:
                if (center_int[k, 0] -1 >=0):
                    inds[k*5+1] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0] -1
                    ret_boxes[k*5+1, 0] += 1
                    # ret_boxes[k*5+1, 3:6] = gt_boxes[k, 3:6].log()
                else:
                    mask[k*5+1] = 0

                if (center_int[k, 0] + 1 < feature_map_size[0]):
                    inds[k*5+2] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0] + 1
                    ret_boxes[k*5+2, 0] -= 1
                    # ret_boxes[k*5+2, 3:6] = gt_boxes[k, 3:6].log()
                else:
                    mask[k*5+2] = 0

            if (self.voxel_size[1] * feature_map_stride > temp_l/4):
                    mask[k*5+3:k*5+5] = 0
                    # print("hello kitty")
            else:
                if (center_int[k, 1] -1 >=0):
                    inds[k*5+3] = (center_int[k, 1] - 1) * feature_map_size[0] + center_int[k, 0]
                    ret_boxes[k*5+3, 1] += 1
                    # ret_boxes[k*5+3, 3:6] = gt_boxes[k, 3:6].log()
                else:
                    mask[k*5+3] = 0

                if (center_int[k, 1] +1 < feature_map_size[1]):
                    inds[k*5+4] = (center_int[k, 1] +1 ) * feature_map_size[0] + center_int[k, 0]
                    ret_boxes[k*5+4, 1] -= 1
                    # ret_boxes[k*5+4, 3:6] = gt_boxes[k, 3:6].log()
                else:
                    mask[k*5+4] = 0

            if gt_boxes.shape[1] > 8:
                ret_boxes[k*5:k*5+5, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask

    def assign_target_of_single_head_v3(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        这个版本在中心点附近也分配了一些正例框,这里一共有9个
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        # print("！！！！！！！！！！！！！")
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((9*num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(9*num_max_objs).long()
        inds_center = gt_boxes.new_zeros(9*num_max_objs).long()
        mask = gt_boxes.new_zeros(9*num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())
            # centernet_utils.draw_gaussian_to_heatmap_v2(heatmap[cur_class_id], center[k], [dx[k],dy[k]])
            # print("??",heatmap[cur_class_id,int(center[k][1])-5:int(center[k][1])+6,(int(center[k][0])-5):int(center[k][0])+6])

            inds[k*9] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k*9:k*9+9] = 1
            # print(gt_boxes[k, 3:6])

            ret_boxes[k*9:k*9+9, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k*9:k*9+9, 2] = z[k]
            ret_boxes[k*9:k*9+9, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k*9:k*9+9, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k*9:k*9+9, 7] = torch.sin(gt_boxes[k, 6])

            care_rate = 4
            if (self.voxel_size[0] * feature_map_stride > gt_boxes[k,4]/care_rate):
                    mask[k*9+1:k*9+3] = 0
                    # print("hello")
            else:
                if (center_int[k, 0] -1 >=0):
                    inds[k*9+1] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0] -1
                    ret_boxes[k*9+1, 0] += 1
                    # ret_boxes[k*5+1, 3:6] = gt_boxes[k, 3:6].log()
                else:
                    mask[k*9+1] = 0

                if (center_int[k, 0] + 1 < feature_map_size[0]):
                    inds[k*9+2] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0] + 1
                    ret_boxes[k*9+2, 0] -= 1
                    # ret_boxes[k*5+2, 3:6] = gt_boxes[k, 3:6].log()
                else:
                    mask[k*9+2] = 0

            if (self.voxel_size[1] * feature_map_stride > gt_boxes[k,3]/care_rate):
                    mask[k*9+3:k*9+5] = 0
                    # print("hello")
            else:
                if (center_int[k, 1] -1 >=0):
                    inds[k*9+3] = (center_int[k, 1] - 1) * feature_map_size[0] + center_int[k, 0]
                    ret_boxes[k*9+3, 1] += 1
                    # ret_boxes[k*5+3, 3:6] = gt_boxes[k, 3:6].log()
                else:
                    mask[k*9+3] = 0

                if (center_int[k, 1] +1 < feature_map_size[1]):
                    inds[k*9+4] = (center_int[k, 1] +1 ) * feature_map_size[0] + center_int[k, 0]
                    ret_boxes[k*9+4, 1] -= 1
                    # ret_boxes[k*5+4, 3:6] = gt_boxes[k, 3:6].log()
                else:
                    mask[k*9+4] = 0

            # 加入左上左下右上右下四个label
            if (self.voxel_size[1] * feature_map_stride > gt_boxes[k,3]/care_rate and 
                        self.voxel_size[0] * feature_map_stride > gt_boxes[k,4]/care_rate):
                    mask[k*9+5:k*9+9] = 0
                    # print("hello")
            else:
                if (center_int[k, 0] -1 >=0 and center_int[k, 1] -1 >=0):
                    inds[k*9+5] = (center_int[k, 1] - 1) * feature_map_size[0] + center_int[k, 0] -1
                    ret_boxes[k*9+5, 0] += 1
                    ret_boxes[k*9+5, 1] += 1
                    # ret_boxes[k*5+3, 3:6] = gt_boxes[k, 3:6].log()
                else:
                    mask[k*9+5] = 0

                if (center_int[k, 0] + 1 <feature_map_size[0] and center_int[k, 1] -1 >=0):
                    inds[k*9+6] = (center_int[k, 1] - 1) * feature_map_size[0] + center_int[k, 0] +1
                    ret_boxes[k*9+6, 0] -= 1
                    ret_boxes[k*9+6, 1] += 1
                else:
                    mask[k*9+6] = 0

                if (center_int[k, 0] -1 >=0 and center_int[k, 1] + 1 < feature_map_size[1]):
                    inds[k*9+7] = (center_int[k, 1] + 1) * feature_map_size[0] + center_int[k, 0] -1
                    ret_boxes[k*9+7, 0] += 1
                    ret_boxes[k*9+7, 1] -= 1
                    # ret_boxes[k*5+3, 3:6] = gt_boxes[k, 3:6].log()
                else:
                    mask[k*9+7] = 0

                if (center_int[k, 0] + 1 <feature_map_size[0] and center_int[k, 1] + 1 < feature_map_size[1]):
                    inds[k*9+8] = (center_int[k, 1] + 1) * feature_map_size[0] + center_int[k, 0] +1
                    ret_boxes[k*9+8, 0] -= 1
                    ret_boxes[k*9+8, 1] -= 1
                else:
                    mask[k*9+8] = 0

            if gt_boxes.shape[1] > 8:
                ret_boxes[k*9:k*9+9, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask
    
    def assign_target_of_single_head_v4(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        版本二：这个版本在中心点附近也分配了一些正例框，这里一共有五个，分别为中心+上下左右
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        # print("???V4V4V4V4V")
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((5*num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(5*num_max_objs).long()
        mask = gt_boxes.new_zeros(5*num_max_objs).long()
        ret_boxes_center = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds_center = gt_boxes.new_zeros(num_max_objs).long()
        mask_center = gt_boxes.new_zeros(num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())
            # centernet_utils.draw_gaussian_to_heatmap_v2(heatmap[cur_class_id], center[k], [dx[k],dy[k]])
            # print("??",heatmap[cur_class_id,int(center[k][1])-5:int(center[k][1])+6,(int(center[k][0])-5):int(center[k][0])+6])

            inds[k*5] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k*5:k*5+5] = 1
            inds_center[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask_center[k] = 1
            # print(gt_boxes[k, 3:6])

            ret_boxes[k*5:k*5+5, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k*5:k*5+5, 2] = z[k]
            ret_boxes[k*5:k*5+5, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k*5:k*5+5, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k*5:k*5+5, 7] = torch.sin(gt_boxes[k, 6])

            ret_boxes_center[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes_center[k, 2] = z[k]
            ret_boxes_center[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes_center[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes_center[k, 7] = torch.sin(gt_boxes[k, 6])

            if (self.voxel_size[0] * feature_map_stride > gt_boxes[k,4]/4):
                    mask[k*5+1:k*5+3] = 0
                    # print("hello")
            else:
                if (center_int[k, 0] -1 >=0):
                    inds[k*5+1] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0] -1
                    ret_boxes[k*5+1, 0] += 1
                    # ret_boxes[k*5+1, 3:6] = gt_boxes[k, 3:6].log()
                else:
                    mask[k*5+1] = 0

                if (center_int[k, 0] + 1 < feature_map_size[0]):
                    inds[k*5+2] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0] + 1
                    ret_boxes[k*5+2, 0] -= 1
                    # ret_boxes[k*5+2, 3:6] = gt_boxes[k, 3:6].log()
                else:
                    mask[k*5+2] = 0

            if (self.voxel_size[1] * feature_map_stride > gt_boxes[k,3]/4):
                    mask[k*5+3:k*5+5] = 0
                    # print("hello")
            else:
                if (center_int[k, 1] -1 >=0):
                    inds[k*5+3] = (center_int[k, 1] - 1) * feature_map_size[0] + center_int[k, 0]
                    ret_boxes[k*5+3, 1] += 1
                    # ret_boxes[k*5+3, 3:6] = gt_boxes[k, 3:6].log()
                else:
                    mask[k*5+3] = 0

                if (center_int[k, 1] +1 < feature_map_size[1]):
                    inds[k*5+4] = (center_int[k, 1] +1 ) * feature_map_size[0] + center_int[k, 0]
                    ret_boxes[k*5+4, 1] -= 1
                    # ret_boxes[k*5+4, 3:6] = gt_boxes[k, 3:6].log()
                else:
                    mask[k*5+4] = 0

            if gt_boxes.shape[1] > 8:
                ret_boxes[k*5:k*5+5, 8:] = gt_boxes[k, 7:-1]
                ret_boxes_center[k, 8:] = gt_boxes[k, 7:-1]
        # print("???",inds.shape,mask.shape)
        inds = torch.cat([inds_center,inds])
        mask = torch.cat([mask_center,mask])
        ret_boxes = torch.cat([ret_boxes_center,ret_boxes],dim = 0)
        # print(inds.shape,mask.shape,ret_boxes.shape)

        return heatmap, ret_boxes, inds, mask


    def assign_target_of_single_head_v5(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=100,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        这个版本在中心点附近也分配了一些正例框,这里一共有9个
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        # print("！！！！！！！！！！！！！")
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((9*num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(9*num_max_objs).long()
        inds_center = gt_boxes.new_zeros(9*num_max_objs).long()
        mask = gt_boxes.new_zeros(9*num_max_objs).long()
        ret_boxes_center = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds_center = gt_boxes.new_zeros(num_max_objs).long()
        mask_center = gt_boxes.new_zeros(num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())
            # centernet_utils.draw_gaussian_to_heatmap_v2(heatmap[cur_class_id], center[k], [dx[k],dy[k]])
            # print("??",heatmap[cur_class_id,int(center[k][1])-5:int(center[k][1])+6,(int(center[k][0])-5):int(center[k][0])+6])

            inds[k*9] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k*9:k*9+9] = 1
            inds_center[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask_center[k] = 1
            # print(gt_boxes[k, 3:6])

            ret_boxes[k*9:k*9+9, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k*9:k*9+9, 2] = z[k]
            ret_boxes[k*9:k*9+9, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k*9:k*9+9, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k*9:k*9+9, 7] = torch.sin(gt_boxes[k, 6])

            ret_boxes_center[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes_center[k, 2] = z[k]
            ret_boxes_center[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes_center[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes_center[k, 7] = torch.sin(gt_boxes[k, 6])

            care_rate = 4
            if (self.voxel_size[0] * feature_map_stride > gt_boxes[k,4]/care_rate):
                    mask[k*9+1:k*9+3] = 0
                    # print("hello")
            else:
                if (center_int[k, 0] -1 >=0):
                    inds[k*9+1] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0] -1
                    ret_boxes[k*9+1, 0] += 1
                    # ret_boxes[k*5+1, 3:6] = gt_boxes[k, 3:6].log()
                else:
                    mask[k*9+1] = 0

                if (center_int[k, 0] + 1 < feature_map_size[0]):
                    inds[k*9+2] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0] + 1
                    ret_boxes[k*9+2, 0] -= 1
                    # ret_boxes[k*5+2, 3:6] = gt_boxes[k, 3:6].log()
                else:
                    mask[k*9+2] = 0

            if (self.voxel_size[1] * feature_map_stride > gt_boxes[k,3]/care_rate):
                    mask[k*9+3:k*9+5] = 0
                    # print("hello")
            else:
                if (center_int[k, 1] -1 >=0):
                    inds[k*9+3] = (center_int[k, 1] - 1) * feature_map_size[0] + center_int[k, 0]
                    ret_boxes[k*9+3, 1] += 1
                    # ret_boxes[k*5+3, 3:6] = gt_boxes[k, 3:6].log()
                else:
                    mask[k*9+3] = 0

                if (center_int[k, 1] +1 < feature_map_size[1]):
                    inds[k*9+4] = (center_int[k, 1] +1 ) * feature_map_size[0] + center_int[k, 0]
                    ret_boxes[k*9+4, 1] -= 1
                    # ret_boxes[k*5+4, 3:6] = gt_boxes[k, 3:6].log()
                else:
                    mask[k*9+4] = 0

            # 加入左上左下右上右下四个label
            if (self.voxel_size[1] * feature_map_stride > gt_boxes[k,3]/care_rate and 
                        self.voxel_size[0] * feature_map_stride > gt_boxes[k,4]/care_rate):
                    mask[k*9+5:k*9+9] = 0
                    # print("hello")
            else:
                if (center_int[k, 0] -1 >=0 and center_int[k, 1] -1 >=0):
                    inds[k*9+5] = (center_int[k, 1] - 1) * feature_map_size[0] + center_int[k, 0] -1
                    ret_boxes[k*9+5, 0] += 1
                    ret_boxes[k*9+5, 1] += 1
                    # ret_boxes[k*5+3, 3:6] = gt_boxes[k, 3:6].log()
                else:
                    mask[k*9+5] = 0

                if (center_int[k, 0] + 1 <feature_map_size[0] and center_int[k, 1] -1 >=0):
                    inds[k*9+6] = (center_int[k, 1] - 1) * feature_map_size[0] + center_int[k, 0] +1
                    ret_boxes[k*9+6, 0] -= 1
                    ret_boxes[k*9+6, 1] += 1
                else:
                    mask[k*9+6] = 0

                if (center_int[k, 0] -1 >=0 and center_int[k, 1] + 1 < feature_map_size[1]):
                    inds[k*9+7] = (center_int[k, 1] + 1) * feature_map_size[0] + center_int[k, 0] -1
                    ret_boxes[k*9+7, 0] += 1
                    ret_boxes[k*9+7, 1] -= 1
                    # ret_boxes[k*5+3, 3:6] = gt_boxes[k, 3:6].log()
                else:
                    mask[k*9+7] = 0

                if (center_int[k, 0] + 1 <feature_map_size[0] and center_int[k, 1] + 1 < feature_map_size[1]):
                    inds[k*9+8] = (center_int[k, 1] + 1) * feature_map_size[0] + center_int[k, 0] +1
                    ret_boxes[k*9+8, 0] -= 1
                    ret_boxes[k*9+8, 1] -= 1
                else:
                    mask[k*9+8] = 0

            if gt_boxes.shape[1] > 8:
                ret_boxes[k*9:k*9+9, 8:] = gt_boxes[k, 7:-1]
                ret_boxes_center[k, 8:] = gt_boxes[k, 7:-1]
        
        # print("???",inds.shape,mask.shape)
        inds = torch.cat([inds_center,inds])
        mask = torch.cat([mask_center,mask])
        ret_boxes = torch.cat([ret_boxes_center,ret_boxes],dim = 0)
        # print(inds.shape,mask.shape,ret_boxes.shape)

        return heatmap, ret_boxes, inds, mask

    def assign_targets(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list = [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                if self.label_assign_flag=='commen':
                    heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head(
                        num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                        feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                        num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                        gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                        min_radius=target_assigner_cfg.MIN_RADIUS,
                    )
                elif self.label_assign_flag=='v2':
                    heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head_v2(
                        num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                        feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                        num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                        gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                        min_radius=target_assigner_cfg.MIN_RADIUS,
                    )
                elif self.label_assign_flag=='v3':
                    heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head_v3(
                        num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                        feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                        num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                        gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                        min_radius=target_assigner_cfg.MIN_RADIUS,
                    )
                elif self.label_assign_flag=='v4':
                    heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head_v4(
                        num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                        feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                        num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                        gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                        min_radius=target_assigner_cfg.MIN_RADIUS,
                    )
                elif self.label_assign_flag=='v5':
                    heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head_v5(
                        num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                        feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                        num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                        gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                        min_radius=target_assigner_cfg.MIN_RADIUS,
                    )

                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

            # IOU_AWARE ##################################################################
            if self.iou_aware_flag:
                pred_iou = pred_boxes[:,8:,:,:]
                pred_boxes = pred_boxes[:,:8,:,:]
            else:
                pred_boxes = pred_boxes[:,:8,:,:]
            ##############################################################################

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
            loss += hm_loss + loc_loss

            # 第二种计算损失函数的方法，对dx dy采用多label,其他只计算中心点 ###############################################
            # print("1111",pred_boxes.shape,target_boxes.shape,target_dicts['masks'][idx].shape,target_boxes.shape)
            # reg_loss1 = self.reg_loss_func(
            #     pred_boxes[:,:2], target_dicts['masks'][idx][:,500:], target_dicts['inds'][idx][:,500:], target_boxes[:,500:,:2]
            # )
            # reg_loss2 = self.reg_loss_func(
            #     pred_boxes[:,2:], target_dicts['masks'][idx][:,:500], target_dicts['inds'][idx][:,:500], target_boxes[:,:500,2:]
            # )
            # loc_loss = (reg_loss1 * reg_loss1.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'][:2])).sum()
            # loc_loss += (reg_loss2 * reg_loss2.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'][2:])).sum()
            # loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
            ##############################################################################################################

            ## IOU_AWARE:加入iou aware loss  ###########################################################################################
            if self.iou_aware_flag:
                # 1.解算出检测框，得到gt iou
                batch_size = target_boxes.shape[0]
                batch_center = pred_dict['center']
                batch_center_z = pred_dict['center_z']
                batch_dim = pred_dict['dim'].exp()
                batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
                batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
                batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

                center_ = loss_utils._transpose_and_gather_feat(batch_center,target_dicts['inds'][idx])*self.feature_map_stride*self.voxel_size[0]
                center_z_ = loss_utils._transpose_and_gather_feat(batch_center_z,target_dicts['inds'][idx])
                dim_ = loss_utils._transpose_and_gather_feat(batch_dim,target_dicts['inds'][idx])
                cos_ = loss_utils._transpose_and_gather_feat(batch_rot_cos,target_dicts['inds'][idx])
                sin_ = loss_utils._transpose_and_gather_feat(batch_rot_sin,target_dicts['inds'][idx])
                angle_ = torch.atan2(sin_, cos_)
                final_pred_dicts = torch.cat([center_,center_z_,dim_,angle_],dim = -1)
                final_pred_dicts = final_pred_dicts.view(-1,7)
                final_pred_dicts = final_pred_dicts.detach()  # 这个很重要！！！！！！

                # 解算gt box
                final_target = target_boxes.clone()
                final_target[:,:,:2] *= self.feature_map_stride*self.voxel_size[0]
                final_target[:,:,3:6] = final_target[:,:,3:6].exp()
                final_target[:,:,6:7] = torch.atan2(final_target[:,:,7:8], final_target[:,:,6:7])
                final_target = final_target[:,:,:7].view(-1,7)
                # print(final_pred_dicts.requires_grad,final_target.requires_grad)

                iou_target = iou3d_nms_utils.boxes_iou3d_gpu(final_pred_dicts, final_target)
                iou_target = iou_target[range(iou_target.shape[0]),range(iou_target.shape[0])].view(batch_size,-1,1)
                # print("??",iou_target.shape,iou_target)
                iou_target = 2 * iou_target - 1
                iou_target = iou_target.detach()
                
                # 2.计算iou loss
                iou_aware_loss = self.iou_aware_loss_func(
                    pred_iou, target_dicts['masks'][idx], target_dicts['inds'][idx], iou_target
                )
                iou_aware_loss = iou_aware_loss * 1.0

                # print(loss.shape,iou_aware_loss)
                loss += iou_aware_loss[0]
            ############################################## IOU_AWARE ####################################################################

            ## iou loss ###############################################################################################
            if self.iou_loss_flag:
                iou_loss = self.iou_loss_func(
                    pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
                )
                iou_loss = iou_loss * self.iou_loss_weight

                loss += iou_loss
                tb_dict['iou_loss_head_%d' % idx] = iou_loss.item()
            ##############################################################################################################
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        # IOU_AWARE
        def update_score_with_iou(batch_hm,iou_pre,weight):
            """
                用两段方程来拟合pow函数
            """
            x0 = 0.5
            y0 = pow(0.5,weight)
            y1 = pow(0.5,1-weight)

            update_heatmap = torch.where(batch_hm>0.5, (2.0-2.0*y0)*batch_hm+2*y0-1.0, 2.0*y0*batch_hm)
            update_iou_pre = torch.where(iou_pre>0.5, (2.0-2.0*y1)*iou_pre+2*y1-1.0, 2.0*y1*iou_pre)
            # print(batch_hm.max(),update_heatmap.max())
            return update_heatmap * update_iou_pre

        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()

            ## IOU_AWARE:用iou更新预测分数 ###############################################################################
            if self.iou_aware_flag:
 
                iou_pre = (pred_dict['iou']+1)*0.5
                iou_pre = torch.clamp(iou_pre, min=0, max=1.)
                
                # 基本形式
                # iou_aware_weight = 0.7
                # batch_hm = torch.pow(batch_hm ,iou_aware_weight) * torch.pow(iou_pre ,1-iou_aware_weight)

                # 第一种解码形式
                iou_aware_weight = self.iou_cls_weight
                
                for i in range(len(iou_aware_weight)):
                    batch_hm[:,i:i+1,:,:] = torch.pow(batch_hm[:,i:i+1,:,:] ,iou_aware_weight[i]) * torch.pow(iou_pre ,1-iou_aware_weight[i])

                # 第二种解码形式
                # batch_hm = batch_hm * iou_aware_weight + iou_pre * (1-iou_aware_weight)
                # 第三种解码形式
                # batch_hm = update_score_with_iou(batch_hm,iou_pre,iou_aware_weight)

                # print("???",iou_pre.max())
            #################################################################################################################

            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                if post_process_cfg.NMS_CONFIG.NMS_TYPE != 'circle_nms':
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                    final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                    final_dict['pred_scores'] = selected_scores
                    final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        # print(ret_dict)
        return ret_dict

    def generate_predicted_boxes_v2(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        score_thresh = post_process_cfg.get('SCORE_THRESH', 0.1)

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]

        stride =  self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE',2)
        pre_max_size = post_process_cfg.get('PRE_MAX_SIZE', 40)

        cls_preds = pred_dicts[0]['hm'].sigmoid()

        if self.iou_aware_flag:
            iou_pre = (pred_dicts[0]['iou']+1)*0.5
            iou_pre = torch.clamp(iou_pre, min=0, max=1.)

            iou_aware_weight = [0.1,0.55,0.275]
            for i in range(3):
                cls_preds[:,i:i+1,:,:] = torch.pow(cls_preds[:,i:i+1,:,:] ,iou_aware_weight[i]) * torch.pow(iou_pre ,1-iou_aware_weight[i])

        box_preds = torch.cat([pred_dicts[0]['center'],pred_dicts[0]['center_z'],pred_dicts[0]['dim'].exp(),pred_dicts[0]['rot']],dim=1)

        # print(box_preds.shape)
        ret_dict = postprocessing_v2(cls_preds, box_preds, self.point_cloud_range, self.grid_size, self.num_class, stride, score_thresh, pre_max_size, ret_dict)
        # print(ret_dict)
        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))

        # new:export onnx
        if self.export_onnx:
            print("??????",pred_dicts[0].keys(),self.export_onnx) 
            data_dict["hm"] = pred_dicts[0]['hm']
            data_dict["center"] = pred_dicts[0]['center']
            data_dict["center_z"] = pred_dicts[0]['center_z']
            data_dict["dim"] = pred_dicts[0]['dim']
            data_dict["rot"] = pred_dicts[0]['rot']
            if 'iou' in pred_dicts[0].keys():
                data_dict['iou'] = pred_dicts[0]['iou']
            return data_dict

        if self.training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
            )
            self.forward_ret_dict['target_dicts'] = target_dict

        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if not self.training or self.predict_boxes_when_training:
            if(self.postprocess=="nms"):
                pred_dicts = self.generate_predicted_boxes(
                    data_dict['batch_size'], pred_dicts
                )
            elif(self.postprocess=="maxpooling"):
                pred_dicts = self.generate_predicted_boxes_v2(
                    data_dict['batch_size'], pred_dicts
                )

            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

        return data_dict

    def forward_quant(self, data_dict, pred_dicts_in):
        name = ['hm', 'center', 'center_z', 'dim', 'rot', 'iou']
        pred_dicts = []
        dict = {}
        for i in range(len(name)):
            dict[name[i]] = pred_dicts_in[i]
        pred_dicts.append(dict)
        # import pdb;pdb.set_trace()
        if not self.training or self.predict_boxes_when_training:
            if(self.postprocess=="nms"):
                pred_dicts = self.generate_predicted_boxes(
                    data_dict['batch_size'], pred_dicts
                )
            elif(self.postprocess=="maxpooling"):
                pred_dicts = self.generate_predicted_boxes_v2(
                    data_dict['batch_size'], pred_dicts
                )

            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

        return data_dict


def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)

def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    # ZeroPad = nn.ZeroPad2d(padding=(0, 1, 0, 1)) # 分别为：左、右、上、下
    # heat_temp = ZeroPad(heat)
    # hmax = F.max_pool2d(heat_temp, (kernel, kernel), stride=1, padding=pad)
    
    # print(heat.shape,hmax.shape)
    keep = (hmax == heat).float()

    return heat * keep

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    # 每个类别返回最大的K个值
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_row = (torch.floor_divide(topk_inds, width)).float()
    topk_col = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (torch.floor_divide(topk_ind, K)).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_row = _gather_feat(topk_row.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_col = _gather_feat(topk_col.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_row, topk_col


def decode(hm_cen, cen_offset, direction, z_coor, dim, K=40):
    batch_size, num_classes, height, width = hm_cen.size()

    hm_cen = _nms(hm_cen)
    scores, inds, clses, row, col = _topk(hm_cen, K=K)
    if cen_offset is not None:
        cen_offset = _transpose_and_gather_feat(cen_offset, inds)
        cen_offset = cen_offset.view(batch_size, K, 2)
        col = col.view(batch_size, K, 1) + cen_offset[:, :, 0:1]
        row = row.view(batch_size, K, 1) + cen_offset[:, :, 1:2]
        
    else:
        col = col.view(batch_size, K, 1) + 0.5
        row = row.view(batch_size, K, 1) + 0.5

    direction = _transpose_and_gather_feat(direction, inds)
    direction = direction.view(batch_size, K, 2)
    z_coor = _transpose_and_gather_feat(z_coor, inds)
    z_coor = z_coor.view(batch_size, K, 1)
    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch_size, K, 3)
    clses = clses.view(batch_size, K, 1).float()
    scores = scores.view(batch_size, K, 1)

    # (scores x 1, ys x 1, xs x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
    # (scores-0:1, ys-1:2, xs-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
    # detections: [batch_size, K, 10]
    detections = torch.cat([scores, col, row, z_coor, dim, direction, clses], dim=2)

    return detections

def get_yaw_v2(direction):
    return torch.atan2(direction[:, 1:2],direction[:, 0:1])

def postprocessing_v2(cls_preds, box_preds, point_cloud_range, grid_size, num_class, stride, score_thresh, pre_max_size, ret):
    detections = decode(cls_preds, box_preds[:,:2,:,:],box_preds[:,6:,:,:],box_preds[:,2:3,:,:],box_preds[:,3:6,:,:], K=pre_max_size)

    for i in range(detections.shape[0]):
        index = detections[i,:,0] > score_thresh
        detection = detections[i, index].contiguous()
        
        # 结果解析
        detection[:,1] = detection[:, 1] * stride * (point_cloud_range[3] - point_cloud_range[0]) / grid_size[0] + point_cloud_range[0]  
        detection[:,2] = detection[:, 2] * stride * (point_cloud_range[4] - point_cloud_range[1]) / grid_size[1] + point_cloud_range[1]  
        # detection[:,3] += point_cloud_range[2]

        detection[:,7:8] = get_yaw_v2(detection[:,7:9])
        ret[i]["pred_boxes"] = detection[:, 1:8].contiguous() # x y z l w h r
        ret[i]["pred_scores"] = detection[:, 0].contiguous()
        ret[i]["pred_labels"] = detection[:, -1].int() + 1 # 从1开始
        # print(ret[i])
    
    return ret
