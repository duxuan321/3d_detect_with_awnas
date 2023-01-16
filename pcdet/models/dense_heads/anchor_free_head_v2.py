import numpy as np
import torch.nn as nn
import torch
from .anchor_head_template import AnchorHeadTemplate
import math
import torch.nn.functional as F
"""
基于mvlidarnet的head,采用分离式的head，将中心偏差、高度、维度、方向分别解析
"""

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution with padding"""
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, stride=stride, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()]
    return nn.Sequential(*layers)

class AnchorFreeSingleV2(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.max_objects = self.model_cfg.get('MAX_OBJECTS', 100)
        self.export_onnx = self.model_cfg.get("EXPORT_ONNX",False)

        self.forward_ret_dict = {}
        self.build_losses()
        self.point_cloud_range = point_cloud_range
        self.grid_size = grid_size

        self.shared_conv = conv3x3(input_channels, 64)
        
        self.cls_head = nn.Sequential(conv3x3(64, 64),
                        nn.Conv2d(64,num_class,kernel_size=3, padding=1, stride=1, bias=True))

        self.xy_head = nn.Sequential(conv3x3(64, 64),
                        nn.Conv2d(64,2,kernel_size=3, padding=1, stride=1, bias=True))

        self.z_head = nn.Sequential(conv3x3(64, 64),
                        nn.Conv2d(64,1,kernel_size=3, padding=1, stride=1, bias=True))

        self.dim_head = nn.Sequential(conv3x3(64, 64),
                        nn.Conv2d(64,3,kernel_size=3, padding=1, stride=1, bias=True))

        self.rot_head = nn.Sequential(conv3x3(64, 64),
                        nn.Conv2d(64,2,kernel_size=3, padding=1, stride=1, bias=True))

        self.init_weights()

    def build_losses(self):
        self.add_module(
            'anchor_free_loss',
            Compute_Loss()
        )

    def init_weights(self):
        pi = 0.01
        # nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        # nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        spatial_features_2d = self.shared_conv(spatial_features_2d)

        cls_preds = self.cls_head(spatial_features_2d)
        xy_preds = self.xy_head(spatial_features_2d)
        z_preds = self.z_head(spatial_features_2d)
        dim_preds = self.dim_head(spatial_features_2d)
        rot_preds = self.rot_head(spatial_features_2d)

        if self.export_onnx:
            data_dict["export_cls_preds"] = cls_preds
            data_dict["export_box_preds"] = xy_preds
            return data_dict

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['xy_preds'] = xy_preds
        self.forward_ret_dict['z_preds'] = z_preds
        self.forward_ret_dict['dim_preds'] = dim_preds
        self.forward_ret_dict['rot_preds'] = rot_preds

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes_with_classes=data_dict['gt_boxes'], H=cls_preds.size(2), W=cls_preds.size(3)
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], cls_preds, xy_preds,z_preds,dim_preds,rot_preds
            )

            if self.predict_boxes_when_training:
                raise NotImplementedError
            else:
                data_dict['final_box_dicts'] = pred_dicts

        # print("test:",data_dict.keys())
        return data_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, xy_preds,z_preds,dim_preds,rot_preds):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        score_thresh = post_process_cfg.get('SCORE_THRESH', 0.1)

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]

        h, w = cls_preds.shape[2:4]

        stride =  post_process_cfg.get('STRIDE',4)
        post_process_style = post_process_cfg.get('POST_PROCESS_STYLE', 'anchor_free') 
        pre_max_size = post_process_cfg.get('PRE_MAX_SIZE', 40)
        if post_process_style == 'anchor_free':
            ret_dict = postprocessing_v2(cls_preds, xy_preds,z_preds,dim_preds,rot_preds , self.point_cloud_range, self.grid_size, self.num_class, stride, score_thresh, pre_max_size, ret_dict)
        return ret_dict
    
    # 构建用于计算loss的label.这里有：class_map(3*w*l)、box_head(6*w*l = xs ys w l sinθ cosθ)
    # 其中类别：0--"Pedestrian",1-"Car"/"Van",2-"Cyclist"
    def assign_targets(self, gt_boxes_with_classes, H, W):
        minX = self.point_cloud_range[0]
        maxX = self.point_cloud_range[3]
        minY = self.point_cloud_range[1]
        maxY = self.point_cloud_range[4]
        minZ = self.point_cloud_range[2]
        maxZ = self.point_cloud_range[5]

        bound_size_x = maxX - minX
        bound_size_y = maxY - minY


        batch_size = gt_boxes_with_classes.shape[0]
        gt_classes = gt_boxes_with_classes[:, :, -1]
        gt_boxes = gt_boxes_with_classes[:, :, :-1]
        targets = {
                'cls_head': [],
                'cen_offset': [],   # 2*w*l
                'direction': [],
                'dim': [],
                'z_coor': [],
                'indices_center': [],
                'obj_mask': [],
            }
        for bs in range(batch_size):
            cur_gt = gt_boxes[bs]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1].cpu().numpy()
            cur_gt_classes = gt_classes[bs][:cnt + 1].int().cpu().numpy() # 从1开始

            num_objects = min(cnt + 1, self.max_objects)
            hm_l, hm_w = H, W

            hm_main_center = np.zeros((self.num_class, hm_l, hm_w), dtype=np.float32)     # 1.heat map
            cen_offset = np.zeros((self.max_objects, 2), dtype=np.float32)          # 2. xs ys
            dimension = np.zeros((self.max_objects, 3), dtype=np.float32)           # 3. w l
            direction = np.zeros((self.max_objects, 2), dtype=np.float32)         # 4. sinθ cosθ
            z_coor = np.zeros((self.max_objects, 1), dtype=np.float32)      # 5.z

            # 目标在BEV的坐标，掩码
            indices_center = np.zeros((self.max_objects), dtype=np.int64)
            obj_mask = np.zeros((self.max_objects), dtype=np.uint8)

            # print(cur_gt_classes)
            for k in range(num_objects):
                x, y, z, l, w, h, yaw = cur_gt[k]
                cls_id = int(cur_gt_classes[k]) - 1 
                # Invert yaw angle
                # yaw = -yaw
                if not ((minX <= x <= maxX) and (minY <= y <= maxY) and (minZ <= z <= maxZ)):
                    continue
                if (h <= 0) or (w <= 0) or (l <= 0):
                    continue
                
                # 长宽的单位转化为BEV像素
                bbox_l = l / bound_size_x * hm_l    
                bbox_w = w / bound_size_y * hm_w
                radius = compute_radius((math.ceil(bbox_l), math.ceil(bbox_w)))
                # radius /= 2.0
                radius = max(0, int(radius))
                
                row = (x - minX) / bound_size_x * hm_l  # x --> row (invert to 2D image space)
                col = (y - minY) / bound_size_y * hm_w  # y --> col
                center = np.array([col, row], dtype=np.float32)

                center_int = center.astype(np.int32)

                # Generate heatmaps for main center
                gen_hm_radius(hm_main_center[cls_id], center, radius)
                # Index of the center
                indices_center[k] = center_int[1] * hm_w + center_int[0]

                # targets for center offset
                cen_offset[k] = center - center_int

                # targets for dimension
                dimension[k, 0] = l
                dimension[k, 1] = w
                dimension[k, 2] = h

                z_coor[k] = z - minZ

                # targets for direction
                direction[k, 0] = math.sin(float(yaw))  # im
                direction[k, 1] = math.cos(float(yaw))  # re

                # Generate object masks
                obj_mask[k] = 1


            targets['cls_head'].append(self.to_gpu(hm_main_center, torch.float32).unsqueeze(0))
            targets['cen_offset'].append(self.to_gpu(cen_offset, torch.float32).unsqueeze(0))   # 2*w*l
            targets['direction'].append(self.to_gpu(direction, torch.float32).unsqueeze(0))
            targets['dim'].append(self.to_gpu(dimension, torch.float32).unsqueeze(0))
            targets['z_coor'].append(self.to_gpu(z_coor, torch.float32).unsqueeze(0))
            targets['indices_center'].append(self.to_gpu(indices_center, torch.long).unsqueeze(0))
            targets['obj_mask'].append(self.to_gpu(obj_mask, torch.uint8).unsqueeze(0))
    
        for key, value in targets.items():
            targets[key] = torch.cat(value, 0)

        return {'targets': targets}

    def to_gpu(self, tensor, dtype):
        tensor = torch.from_numpy(tensor).to(dtype).to(self.shared_conv[0].weight.device)
        return tensor

    def get_loss(self):

        cls_preds = self.forward_ret_dict['cls_preds']
        xy_preds = self.forward_ret_dict['xy_preds']
        z_preds = self.forward_ret_dict['z_preds']
        dim_preds = self.forward_ret_dict['dim_preds']
        rot_preds = self.forward_ret_dict['rot_preds']
        targets_map = self.forward_ret_dict['targets']
        
        preds = {'cls_head': cls_preds, 'xy_head': xy_preds,'z_head':z_preds,'dim_head':dim_preds,'rot_head':rot_preds}
        total_loss, tb_dict = self.anchor_free_loss(preds, targets_map)


        return total_loss, tb_dict


def compute_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h


def gen_hm_radius(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    col, row = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(col, radius), min(width - col, radius + 1)
    top, bottom = min(row, radius), min(height - row, radius + 1)

    masked_heatmap = heatmap[row - top:row + bottom, col - left:col + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


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

def _neg_loss(pred, gt, alpha=2, beta=4):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, beta)

    loss = 0
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class L1Loss_Balanced(nn.Module):
    """Balanced L1 Loss
    paper: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    Code refer from: https://github.com/OceanPang/Libra_R-CNN
    """

    def __init__(self, alpha=0.5, gamma=1.5, beta=1.0):
        super(L1Loss_Balanced, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        assert beta > 0
        self.beta = beta

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = self.balanced_l1_loss(pred * mask, target * mask)
        loss = loss.sum() / (mask.sum() + 1e-4)

        return loss

    def balanced_l1_loss(self, pred, target):
        assert pred.size() == target.size() and target.numel() > 0

        diff = torch.abs(pred - target)
        b = math.exp(self.gamma / self.alpha) - 1
        loss = torch.where(diff < self.beta,
                           self.alpha / b * (b * diff + 1) * torch.log(b * diff / self.beta + 1) - self.alpha * diff,
                           self.gamma * diff + self.gamma / b - self.alpha * self.beta)

        return loss


def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)

def to_cpu(tensor):
    return tensor.detach().cpu()

class Compute_Loss(nn.Module):
    def __init__(self):
        super(Compute_Loss, self).__init__()
        self.focal_loss = FocalLoss()
        self.l1_loss = L1Loss()
        self.l1_loss_balanced = L1Loss_Balanced(alpha=0.5, gamma=1.5, beta=1.0)
        self.weight_hm_cen = 1.
        self.weight_z_coor, self.weight_cenoff, self.weight_dim, self.weight_direction = 2., 2., 2., 2.

        self.count = 0
    def forward(self, outputs, tg):
        # tg: targets
        # outputs['cls_head']：class_num l w
        # outputs['box_head']：xs ys z w l h sin cos
        cls_head= _sigmoid(outputs['cls_head'])

        # import skimage
        # jpg = tg['cls_head'].detach().cpu().squeeze().numpy()
        # print('-----------------------', jpg.min(), jpg.max())
        # jpg = ((jpg - jpg.min()) / (jpg.max() - jpg.min()) * 255 ).astype(np.ubyte)
        # jpg = jpg.transpose(1,2,0)
        # skimage.io.imsave('./gt.jpg', jpg)
        # self.count += 1
        # if self.count == 5:
        #     1/0


        l_hm_cen = self.focal_loss(cls_head, tg['cls_head'])
        l_cen_offset = self.l1_loss(_sigmoid(outputs['xy_head']), tg['obj_mask'], tg['indices_center'], tg['cen_offset'])
        l_direction = self.l1_loss(outputs['rot_head'], tg['obj_mask'], tg['indices_center'], tg['direction'])
        # Apply the L1_loss balanced for dimension regression
        l_dim = self.l1_loss_balanced(outputs['dim_head'], tg['obj_mask'], tg['indices_center'], tg['dim'])
        l_z_coor = self.l1_loss_balanced(outputs['z_head'], tg['obj_mask'], tg['indices_center'], tg['z_coor'])

        total_loss = l_hm_cen * self.weight_hm_cen + l_cen_offset * self.weight_cenoff + \
                     l_dim * self.weight_dim + l_direction * self.weight_direction + \
                        l_z_coor * self.weight_z_coor

        loss_stats = {
            'total_loss': to_cpu(total_loss).item(),
            'hm_cen_loss': to_cpu(l_hm_cen).item(), 
            'cen_offset_loss': to_cpu(l_cen_offset).item(),
            'dim_loss': to_cpu(l_dim).item(),
            'direction_loss': to_cpu(l_direction).item(),
            'z_coor_loss': to_cpu(l_z_coor).item(),
        }

        return total_loss, loss_stats

## 后处理第二种方式：采用和SFA3D一样的解析方式
def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
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
    return torch.atan2(direction[:, 0:1], direction[:, 1:2])

def postprocessing_v2(cls_preds, xy_preds,z_preds,dim_preds,rot_preds, point_cloud_range, grid_size, num_class, stride, score_thresh, pre_max_size, ret):
    cls_preds_temp = _sigmoid(cls_preds)
    detections = decode(cls_preds_temp, _sigmoid(xy_preds),rot_preds,z_preds,dim_preds, K=pre_max_size)

    for i in range(detections.shape[0]):
        index = detections[i,:,0] > score_thresh
        detection = detections[i, index].contiguous()
        
        # 结果解析
        preds_col = detection[:, 1].clone()
        preds_row = detection[:, 2].clone()
  
        detection[:,1] = preds_row * stride * (point_cloud_range[3] - point_cloud_range[0]) / grid_size[0] + point_cloud_range[0]  
        detection[:,2] = preds_col * stride * (point_cloud_range[4] - point_cloud_range[1]) / grid_size[1] + point_cloud_range[1]  
        detection[:,3] += point_cloud_range[2]

        detection[:,7:8] = get_yaw_v2(detection[:,7:9])
        ret[i]["pred_boxes"] = detection[:, 1:8].contiguous() # x y z l w h r
        ret[i]["pred_scores"] = detection[:, 0].contiguous()
        ret[i]["pred_labels"] = detection[:, -1].int() + 1 # 从1开始
        # print(ret[i])
    
    return ret



