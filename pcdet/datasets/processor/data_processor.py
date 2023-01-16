from functools import partial

import numpy as np
from skimage import transform
import numba
from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass

# 针对waymo数据集的5个维度信息进行bevmap
@numba.jit
def make_bev_map_norm_waymo(PointCloud, min_heightMap, max_heightMap, mean_intensityMap, mean_elongationMap,countMap,boundary):
    """
        版本三：针对waymo数据集的bev特征提取
    """
    for i in range(PointCloud.shape[0]):
        h = np.int_(PointCloud[i, 0])
        w = np.int_(PointCloud[i, 1])
        # if h >= 0 and h < Height and w >= 0 and w < Width:
        norm_z = (PointCloud[i,2] - boundary[2]) / (boundary[5] - boundary[2]) - 0.5
        if min_heightMap[h, w] > norm_z:
            min_heightMap[h, w] = norm_z 
        if max_heightMap[h, w] < norm_z:
            max_heightMap[h, w] = norm_z

        mean_intensityMap[h, w] += PointCloud[i,3] - 0.5
        mean_elongationMap[h, w] += PointCloud[i,4] - 0.5
        countMap[h, w] += 1

    return min_heightMap, max_heightMap, mean_intensityMap,mean_elongationMap,countMap

@numba.jit
def make_bev_map_norm(PointCloud, min_heightMap, max_heightMap, mean_intensityMap, countMap,boundary):
    """
        版本二：将数据归一化
    """
    for i in range(PointCloud.shape[0]):
        h = np.int_(PointCloud[i, 0])
        w = np.int_(PointCloud[i, 1])
        # if h >= 0 and h < Height and w >= 0 and w < Width:
        norm_z = (PointCloud[i,2] - boundary[2]) / (boundary[5] - boundary[2]) - 0.5
        if min_heightMap[h, w] > norm_z:
            min_heightMap[h, w] = norm_z 
        if max_heightMap[h, w] < norm_z:
            max_heightMap[h, w] = norm_z

        mean_intensityMap[h, w] += PointCloud[i,3] - 0.5
        countMap[h, w] += 1

    return min_heightMap, max_heightMap, mean_intensityMap,countMap

@numba.jit
def make_bev_map(PointCloud, min_heightMap, max_heightMap, mean_intensityMap, countMap):
    """
        版本一：未归一化数据，效果较差
    """
    for i in range(PointCloud.shape[0]):
        h = np.int_(PointCloud[i, 0])
        w = np.int_(PointCloud[i, 1])
        # if h >= 0 and h < Height and w >= 0 and w < Width:
        if min_heightMap[h, w] > PointCloud[i,2]:
            min_heightMap[h, w] = PointCloud[i,2] 
        if max_heightMap[h, w] < PointCloud[i,2]:
            max_heightMap[h, w] = PointCloud[i,2]

        mean_intensityMap[h, w] += PointCloud[i,3]
        countMap[h, w] += 1

    return min_heightMap, max_heightMap, mean_intensityMap,countMap


# 测试扩充维度对结果的影响
@numba.jit
def make_bev_map_norm_more(PointCloud, PointCloud_,min_heightMap, max_heightMap, mean_intensityMap,
                                     max_intensityMap ,countMap,boundary):
    """
        版本四：扩充更多的特征信息
    """
    for i in range(PointCloud.shape[0]):
        h = np.int_(PointCloud[i, 0])
        w = np.int_(PointCloud[i, 1])
        # if h >= 0 and h < Height and w >= 0 and w < Width:
        norm_z = (PointCloud[i,2] - boundary[2]) / (boundary[5] - boundary[2]) - 0.5
        if min_heightMap[h, w] > norm_z:
            min_heightMap[h, w] = norm_z 
        if max_heightMap[h, w] < norm_z:
            max_heightMap[h, w] = norm_z
        if max_intensityMap[h, w]<PointCloud_[i,3]:
            max_intensityMap[h, w] = PointCloud_[i,3]
        
        # mean_xyz[0,h,w] += (PointCloud_[i,0]- boundary[0] - (h+0.5)*0.16)
        # mean_xyz[1,h,w] += (PointCloud_[i,1]- boundary[1] - (w+0.5)*0.16)
        # mean_xyz[2,h,w] += (PointCloud_[i,2]- boundary[2] - 2)
        # mean_xyz[0,h,w] += (PointCloud_[i,0]/(boundary[3]-boundary[0])-0.5)
        # mean_xyz[1,h,w] += (PointCloud_[i,1]/(boundary[4]))
        # mean_xyz[2,h,w] += (PointCloud_[i,2]/(boundary[5]-boundary[2])-0.5)

        mean_intensityMap[h, w] += PointCloud[i,3] - 0.5
        countMap[h, w] += 1

    return min_heightMap, max_heightMap, mean_intensityMap,max_intensityMap,countMap

## MVlidarnet：这里生成的BEV为min_hei max_hei mean_intensity
def makeBEVMap(PointCloud_, boundary, res_x, res_y, BEV_HEIGHT, BEV_WIDTH):
    Height = BEV_HEIGHT + 1
    Width = BEV_WIDTH + 1
    val_flag_1 = np.logical_and(PointCloud_[:, 0] > boundary[0], PointCloud_[:, 0] < boundary[3])
    val_flag_2 = np.logical_and(PointCloud_[:, 1] > boundary[1], PointCloud_[:, 1] < boundary[4])
    val_flag_3 = np.logical_and(PointCloud_[:, 2] > boundary[2], PointCloud_[:, 2] < boundary[5])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    val_flag_merge = np.logical_and(val_flag_merge,val_flag_3)
    PointCloud = PointCloud_[val_flag_merge]

    # Discretize Feature Map
    # PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor((PointCloud[:, 0] - boundary[0]) / res_x))
    PointCloud[:, 1] = np.int_(np.floor((PointCloud[:, 1] - boundary[1]) / res_y))

    min_heightMap = np.zeros((BEV_HEIGHT, BEV_WIDTH),dtype = np.float32) + 99
    max_heightMap = np.zeros((BEV_HEIGHT, BEV_WIDTH),dtype = np.float32) - 99
    mean_intensityMap = np.zeros((BEV_HEIGHT, BEV_WIDTH),dtype = np.float32)  
    max_intensityMap = np.zeros((BEV_HEIGHT, BEV_WIDTH),dtype = np.float32)  # 维度扩充一：最大强度
    occupy = np.zeros((BEV_HEIGHT, BEV_WIDTH),dtype = np.float32)  # 维度扩充二：体素是否占有
    mean_xyz = np.zeros((3,BEV_HEIGHT, BEV_WIDTH),dtype = np.float32)     # 维度扩充三：平均xyz
    countMap = np.zeros((BEV_HEIGHT, BEV_WIDTH),dtype = np.int8)

    # mean_elongationMap = np.zeros((BEV_HEIGHT, BEV_WIDTH),dtype = np.float32)   # 针对waymo
    
    #min_heightMap,max_heightMap,mean_intensityMap,countMap = make_bev_map(PointCloud,min_heightMap,max_heightMap,mean_intensityMap,countMap)
    # min_heightMap,max_heightMap,mean_intensityMap,countMap = make_bev_map_norm(PointCloud,min_heightMap,
    #                             max_heightMap,mean_intensityMap,countMap,boundary)
    # min_heightMap,max_heightMap,mean_intensityMap,mean_elongationMap,countMap = make_bev_map_norm_waymo(PointCloud,min_heightMap,
    #                 max_heightMap,mean_intensityMap,mean_elongationMap,countMap,boundary)
    min_heightMap,max_heightMap,mean_intensityMap,max_intensityMap,countMap = make_bev_map_norm_more(PointCloud,PointCloud_[val_flag_merge],min_heightMap,
                                max_heightMap,mean_intensityMap, max_intensityMap, countMap ,boundary)

    # count_temp = countMap.copy()/130    # 扩充特征
    occupy[countMap>0] = 1    # 扩充特征
    
    countMap[countMap == 0] +=1
    mean_intensityMap = mean_intensityMap/countMap
    mean_xyz = mean_xyz/countMap    # 扩充特征

    min_heightMap[min_heightMap > 98] = 0
    max_heightMap[max_heightMap < -98] = 0
    max_intensityMap -= 0.5

    # RGB_Map = np.stack([min_heightMap, max_heightMap, mean_intensityMap], 0)
    RGB_Map = np.stack([min_heightMap, max_heightMap, mean_intensityMap], 0)

    return RGB_Map

class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)
        
        return data_dict
        
    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    

    def transform_points_to_bev(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_bev, config=config)

        
        points = data_dict['points']
        # print('points', points)
        # print('-----', self.point_cloud_range, self.voxel_size, self.grid_size)
        bev_map = makeBEVMap(points, self.point_cloud_range, self.voxel_size[0], self.voxel_size[1], self.grid_size[0], self.grid_size[1])
        data_dict['bev_map'] = np.expand_dims(bev_map, 0) # (1, 3, h, w)

        return data_dict


    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
