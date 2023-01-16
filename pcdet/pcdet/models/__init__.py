from collections import namedtuple

import numpy as np
import torch

from thop import profile

from .detectors import build_detector

try:
    import kornia
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')



def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)

        '''from icecream import ic
        assert batch_dict['batch_size'] == 1
        if hasattr(model.backbone_2d, 'ctx'):
            if model.gene_type is None:
                model.gene_type = model.backbone_2d.search_space.random_sample().genotype
            ic(model.gene_type)
            rollout = model.backbone_2d.search_space.rollout_from_genotype(model.gene_type)
            model.backbone_2d = model.backbone_2d.finalize_rollout(rollout)
            model.cuda()
        flops, params = profile(model, inputs=(batch_dict,))
        ic(model)
        ic(flops/1e9, params/1e6) #flops单位G，para单位M
        import pdb;pdb.set_trace()'''

        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
