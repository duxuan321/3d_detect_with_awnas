## 0.安装

pip install -r requirment.txt
python setup.py develop
ln -s /your/data/file ./pcdet/data/kitti

## 1.训练
CUDA_VISIBLE_DEVICES=0,1 bash scripts/dist_train.sh 2 --batch_size 16 --extra_tag with_hei_filter --cfg_file ./cfgs/kitti_models/mvlidarnet.yaml 


## 2.测试
CUDA_VISIBLE_DEVICES=1 python test.py --cfg_file ./cfgs/kitti_models/mvlidarnet.yaml  --batch_size 1 --ckpt ../output/cfgs/kitti_models/mvlidarnet/default/ckpt/checkpoint_epoch_80.pth

## 3.更新
2022/7/7 v1.1

修改了一些BUG，对DBSCAN后处理进行了一些格式化处理，修改了高度过滤


2022/9/27 v2.0

加入了mvlidarnet_center.yaml配置文件，更换了mvldiarnet的head，精度更高，可以接近pointpillar的精度（需要在数据集里加入地面信息）

2022/10/25 v2.1

加入了新的iou loss和centerhead的iou aware分支


## 4.模块的使用
以下模块在配置文件中使用

#### 4.1 手工特征提取代替pillar scatter
```
DATA_PROCESSOR:
    - NAME: mask_points_and_boxes_outside_range
      REMOVE_OUTSIDE_BOXES: True
    
    - NAME: shuffle_points
      SHUFFLE_ENABLED: {
        'train': True,
        'test': False
      }
    
    - NAME: transform_points_to_bev
      VOXEL_SIZE: [0.16, 0.16, 4]
...

MODEL:
    NAME: CenterPoint

    VFE:
        NAME: PlaceHolderVFE

    MAP_TO_BEV:
        NAME: BEV_scatter
        NUM_BEV_FEATURES: 64

    BACKBONE_2D:
        NAME: BaseBEVBackbone
```

        
#### 4.2 center head 的label assignment改进版本
```
MODEL:
    NAME: CenterPoint
    ...

    DENSE_HEAD:
        NAME: CenterHead
        ...

        TARGET_ASSIGNER_CONFIG:
            FEATURE_MAP_STRIDE: 2
            NUM_MAX_OBJS: 500
            GAUSSIAN_OVERLAP: 0.1      
            MIN_RADIUS: 2

            LABEL_ASSIGN_FLAG: v2    # 加入这一行
```

#### 4.3 更换maxpooling后处理(ped 和 cyclist效果交叉，不建议使用)
```
MODEL:
    NAME: CenterPoint
    ...

    DENSE_HEAD:
        NAME: CenterHead
        ...
        POST_PROCESSING:
            POSTPROCESS_TYPE: nms    
            # POSTPROCESS_TYPE: maxpooling    # 第二种后处理方式：maxpooling
```

#### 4.4 iou loss
```
MODEL:
    NAME: CenterPoint

    ...

        WITH_IOU_LOSS: True  # 使用iou loss
        IOU_LOSS_TYPE: GIOU_3D  # 目前有三种IOU损失函数：IOU_HEI、IOU_3D、GIOU_3D
        IOU_WEIGHT: 1   # iou_loss权重
```

    
#### 4.5 使用iou aware(center head)
以下标注IOU_AWARE的都要加上
```
MODEL:
    NAME: CenterPoint
    ...

    DENSE_HEAD:
        NAME: CenterHead 
        CLASS_AGNOSTIC: False

        CLASS_NAMES_EACH_HEAD: [
            ['Car', 'Pedestrian', 'Cyclist']
        ]

        SHARED_CONV_CHANNEL: 64
        USE_BIAS_BEFORE_NORM: True
        NUM_HM_CONV: 2
        SEPARATE_HEAD_CFG:
            HEAD_ORDER: ['center', 'center_z', 'dim', 'rot','iou']  # IOU_AWARE
            # HEAD_ORDER: ['center', 'center_z', 'dim', 'rot']
            HEAD_DICT: {
                'center': {'out_channels': 2, 'num_conv': 2},
                'center_z': {'out_channels': 1, 'num_conv': 2},
                'dim': {'out_channels': 3, 'num_conv': 2},
                'rot': {'out_channels': 2, 'num_conv': 2},

                'iou': {'out_channels': 1, 'num_conv': 2},    # IOU_AWARE
            }
            
            # 扩充项
            # WITH_IOU_LOSS: True  # 使用iou loss
            IOU_LOSS_TYPE: GIOU_3D  # 目前有三种IOU损失函数：IOU_HEI、IOU_3D、GIOU_3D
            IOU_WEIGHT: 1   # iou_loss权重
    
            # WITH_IOU_AWARE_LOSS: True   # IOU_AWARE
            IOU_AWARE_WEIGHT: 1      # IOU_AWARE

        TARGET_ASSIGNER_CONFIG:
            FEATURE_MAP_STRIDE: 2
            NUM_MAX_OBJS: 500
            GAUSSIAN_OVERLAP: 0.1      
            MIN_RADIUS: 2

            # LABEL_ASSIGN_FLAG: v2

        LOSS_CONFIG:
            LOSS_WEIGHTS: {
                'cls_weight': 1.0,
                'loc_weight': 2.0,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }

        POST_PROCESSING:
            POSTPROCESS_TYPE: nms    
            # POSTPROCESS_TYPE: maxpooling    # 第二种后处理方式：maxpooling
            SCORE_THRESH: 0.1
            POST_CENTER_LIMIT_RANGE: [0, -39.68, -3, 69.12, 39.68, 1]
            MAX_OBJ_PER_SAMPLE: 500
            NMS_CONFIG:
                NMS_TYPE: nms_gpu
                NMS_THRESH: 0.7
                NMS_PRE_MAXSIZE: 4096
                NMS_POST_MAXSIZE: 500

```