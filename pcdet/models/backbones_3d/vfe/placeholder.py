import torch

from .vfe_template import VFETemplate


class PlaceHolderVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
            **kwargs:

        Returns:
            spatial_features: (n, C, h, w)
        """
        bev_map = batch_dict['bev_map']
        batch_dict['spatial_features'] = bev_map.contiguous()

        return batch_dict
