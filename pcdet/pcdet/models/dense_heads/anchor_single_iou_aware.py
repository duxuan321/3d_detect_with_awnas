import numpy as np
import torch.nn as nn
import torch

from ...utils import box_coder_utils, common_utils, loss_utils
from .anchor_head_template import AnchorHeadTemplate
from ...ops.iou3d_nms import iou3d_nms_utils

"""
    基于openpcdstaticgraph工程改的
"""
class IouAwareHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        # IOU_AWARE
        self.conv_iou = nn.Conv2d(
            input_channels, self.num_anchors_per_location,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        #IOU_AWARE
        iou_preds = self.conv_iou(spatial_features_2d) # [N, C, H, W]

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        # IOU_AWARE
        iou_preds = iou_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        # IOU_AWARE
        self.forward_ret_dict['iou_preds'] = iou_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )

            #IOU_AWARE
            # batch_cls_preds = torch.sigmoid(batch_cls_preds)
            # num_anchor = int(batch_cls_preds.size(1))
            # batch_size = int(batch_cls_preds.size(0))
            # iou_preds = iou_preds.view(batch_size, num_anchor, 1)
            # iou_preds = (iou_preds + 1) * 0.5
            # batch_cls_preds = batch_cls_preds * torch.pow(iou_preds, 4.0)

            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

    # 以下都是IOU_AWARE
    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'iou_loss_func',
            getattr(loss_utils, reg_loss_name)()
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        iou_loss, tb_dict_iou = self.get_iou_reg_layer_loss(tb_dict_box)

        tb_dict_box.pop('mask')
        tb_dict_box.pop('iou_target', None)
        tb_dict_box.pop('iou_weights', None)

        tb_dict.update(tb_dict_iou)
        tb_dict.update(tb_dict_box)

        rpn_loss = cls_loss + box_loss + iou_loss

        return rpn_loss, tb_dict

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])

        # ************************** for iou *****************************
        # IOU_AWARE（正确的下面代码不应该注释）
        tb_dict = {}
        iou_weights = reg_weights.clone()
        mask = iou_weights > 0
        if mask.sum() > 0:
            iou_weights = iou_weights[mask].view(1, -1)
            batch_box_preds_temp = self.box_coder.decode_torch(box_preds[mask].view(1, -1, 7), anchors[mask].view(1, -1, 7))
            box_reg_targets_temp  = self.box_coder.decode_torch(box_reg_targets[mask].view(1, -1, 7), anchors[mask].view(1, -1, 7))

            iou_preds_box = batch_box_preds_temp.view(-1, 7)
            iou_gt_box = box_reg_targets_temp.view(-1, 7)
            iou_target = iou3d_nms_utils.boxes_iou3d_align(iou_preds_box, iou_gt_box)

            iou_target = iou_target.view(1, -1, 1)
            iou_target = 2 * iou_target - 1
            tb_dict['iou_target'] = iou_target.detach()
            tb_dict['iou_weights'] = iou_weights
        tb_dict['mask'] = mask
        # ************************** for iou *****************************

        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict.update({
            'rpn_loss_loc': loc_loss.item()
        })

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_iou_reg_layer_loss(self, gt_dict):
        iou_preds = self.forward_ret_dict['iou_preds']
        mask = gt_dict['mask']
        if mask.sum() > 0:
            iou_target = gt_dict['iou_target']
            iou_weights = gt_dict['iou_weights']
            batch_size = int(iou_preds.shape[0])
            num_anchor = int(iou_target.size(1))

            iou_preds = iou_preds.view(batch_size, -1, 1)
            iou_preds = iou_preds[mask].view(1, num_anchor, 1)
            iou_loss_src = self.iou_loss_func(iou_preds, iou_target, weights=iou_weights)  # [N, 1]
            iou_loss = iou_loss_src.sum() / batch_size

            iou_loss = iou_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_weight']

            tb_dict = {
                'rpn_loss_iou': iou_loss.item()
            }
        else:
            tb_dict = {
                'rpn_loss_iou': 0.0
            }
            iou_loss = 0

        return iou_loss, tb_dict