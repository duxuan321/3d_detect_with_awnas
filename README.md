## Supernet的构建与训练
Supernet是基于awnas中germ.GermSupernet完成的：<br>
```
class MySupernet(germ.GermSuperNet):
  def __init__(self, model_cfg, search_space=GermSearchSpace({})):
	# start to build your network here 
```
初始化函数的参数search_space=GermSearchSpace({}))可以理解为固定写法，这里的search_space会在后面`with self.begin_searchable() as ctx`开始之后确定。

接下来就是最关键的工作：把需要搜索的模块替换成基于awnas中germ模块的Searchable Block。目前已有支持Conv，ConvTranspose2d，BN等常用Searchable Block可以直接使用。如果原本的layer是：

```
cur_layers = [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0, bias=False),
              nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
              nn.ReLU()]
```

现在需要对Conv2d的out channel进行搜索，则替换后写成：

```
with self.begin_searchable() as ctx:
  cur_channels = germ.Choices([xxx]).apply(divisor_fn)
  cur_layer = [germ.SearchableConvBNBlock(ctx, 64, cur_channels, kernel_size=3, stride=2),
               nn.ReLU()]
```

这里面有两个重要参数：ctx是用来记录当前网络结构的重要参数，在Searchable Block初始化必须写在`
with self.begin_searchable() as ctx`下，并在每一个Searchable Block中都传入ctx；定义cur_channels时的[xxx]表示当前channel有哪些选项，注意只有经过germ.Choices包装过才能被正确使用。

完成Supernet的初始化之后，只需要在forward最开始写上：

```
self.ctx.rollout = self.search_space.random_sample()
```

则本次forward的网络架构就是从search space中随机sample的子架构，最后只会更新sample到的子架构的权重。这一部分有两个例子可供参考：
[PointPilllar Based Supernet](https://github.com/duxuan321/3d_detect_with_awnas/blob/e092f93c37da3b43ae1468a93e3fd55575df2388/pcdet/pcdet/models/backbones_2d/base_bev_backbone.py#L157)和[Mvlidarnet Based Supernet](https://github.com/duxuan321/3d_detect_with_awnas/blob/e092f93c37da3b43ae1468a93e3fd55575df2388/pcdet/pcdet/models/backbones_2d/MVLidarNet.py#L115)。训练Supernet的命令如下，这里必须加上use_same_seed以确保每次forward时所有的线程sample到相同的子架构。
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/dist_train.sh 8 --use_same_seed --num_epochs_to_eval 0 --cfg_file ./cfgs/kitti_models/mvlidarnet_center_iou_aware.yaml --extra_tag train_supernet
```

## 子网络的搜索
训练完supernet后即可加载supernet的权重开始搜索。目前支持的random search
，evolution-based search和evolution-based search with predictor，运行evolution-based search with predictor的命令是：
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/dist_pred_search.sh 8 --cfg_file xxx.yaml --ckpt xxx.pth --epoch_num your_epoch_num --controller_samples your_sample_num --train_predictor
```
这样一共会从supernet中sample出`epoch_num * controller_samples`个子网络以权重共享的方式进行推理并保存结果，最后会保存controller模型记录evolution search的结果，一 个predictor模型用于预测架构的精度，和一个yaml文件记录`candidate_num`个表现最好的子网络，`candidate_num`的值可以在[此处](https://github.com/duxuan321/3d_detect_with_awnas/blob/eb9e2798a385dc5150e21d0f091af1d0db668166/tools/parallel_pred_based.py#L202)进行修改。另外，如果已经有controller模型和predictor模型，可以直接加载模型得到结果，把`train_predictor`参数关掉即可。

## StandAlone模型的训练
经过Supernet的训练和子网络的搜索之后可以得到`candidate_num`个表现最好的子网络作为搜索结果，StandAlone模型就是根据这个搜索结果来训练的。以Mvlidarnet Based Supernet为例，假设对其中**6个**blocks的out_channel进行搜索，每个channel的选择范围是[0.5, 0.66, 1.0] * origin_channel。子网络搜索后得到的evo_N_standalone.yaml中会有N= candidate_num条这种格式的数据：

```
odict_values([0.5, 0.66, 1, 1.0, 1, 1]):
- 0.5430261111719045
- 69.83184231146254
- 41.44670118291255
- 51.629289857196255
- 20.693961216
```
这里面odict_values是每个block具体对应的channel，第一行是**predictor预测的分数**，2-4行分别是该子网络在Car、Pedestrian、Cyclist的推理结果，注意这个结果只能做为相对数值的参考，因为子网络是以参数共享的方式进行推理的，实际按照这个架构训出来的StandAlone模型精度会高很多。最后一行是模型的FLOPs。一般来说从中选取predictor预测分数最高的几个架构做StandAlone训练即可，训练过程与正常的pcdet模型没有区别，这里为了方便已经重写了[Standalone_Pointpillar](https://github.com/duxuan321/3d_detect_with_awnas/blob/eb9e2798a385dc5150e21d0f091af1d0db668166/pcdet/models/backbones_2d/base_bev_backbone.py#L157)和[Standalone_MVLidarNet](https://github.com/duxuan321/3d_detect_with_awnas/blob/eb9e2798a385dc5150e21d0f091af1d0db668166/pcdet/models/backbones_2d/MVLidarNet.py#L215)两个模型和对应的配置文件[mvlidarnet_standalone.yaml](https://github.com/duxuan321/3d_detect_with_awnas/blob/main/tools/cfgs/kitti_models/mvlidarnet_standalone.yaml)和[pointpillar_standalone.yaml](https://github.com/duxuan321/3d_detect_with_awnas/blob/main/tools/cfgs/kitti_models/pointpillar_standalone.yaml)，把选出来的架构（比如此处的[0.5, 0.66, 1, 1.0, 1, 1]）填入mvlidarnet_standalone.yaml 中的ratio，运行下面的命令即可：
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/dist_train.sh 8 --cfg_file ./cfgs/kitti_models/mvlidarnet_standalone.yaml --extra_tag standalone_model
```