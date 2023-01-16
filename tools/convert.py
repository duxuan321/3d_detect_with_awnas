from re import T
import torch
import torch.nn as nn
# from nova3d.ops import spconv
# from ..utils import state as nova3d_state
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn import TensorQuantizer
from pytorch_quantization.tensor_quant import QuantDescriptor

def convert_model(model, **kwargs):

    # set quant module
    def _set_module(raw_model, submodule_key, quant_module):
        module_infos = submodule_key.split(".")
        module_info = module_infos[:-1]
        cur_model = raw_model
        for sub in module_info:
            cur_model = getattr(cur_model, sub)
        setattr(cur_model, module_infos[-1], quant_module)
        # TODO: LOAD WEIGHT & BIAS
        
    if not isinstance(model, nn.Module):
        print("model is not valid")
        exit(0)

    conv2d_sign = kwargs.get("Conv2dMode", True)
    conv1d_sign = kwargs.get("Conv1dMode", True)
    linear_sign = kwargs.get("LinearMode", True)
    spconv_sign = kwargs.get("SpconvMode", True)
    convtranspose_sign = kwargs.get("ConvTransposeMode", False)

    if conv2d_sign == conv1d_sign == linear_sign == spconv_sign == False:
        return
    
    # if spconv_sign:
    #     nova3d_state.quantization.available = True

    layer_para = {}
    conv2d_layers, conv1d_layers, linear_layers, spconv_layers, convtranspose_layers = [], [], [], [], []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            conv2d_layers.append(name + "|" + str(module)[6:])
        if isinstance(module, torch.nn.modules.conv.Conv1d):
            conv1d_layers.append(name + "|" + str(module)[6:])
        if isinstance(module, torch.nn.modules.Linear):
            linear_layers.append(name + "|" + str(module)[6:])
        if isinstance(module, torch.nn.modules.conv.ConvTranspose2d):
            convtranspose_layers.append(name + "|" + str(module)[15:])
        # if isinstance(module, spconv.SubMConv3d) or isinstance(module, spconv.SparseConv3d) or isinstance(module, spconv.SparseInverseConv3d):
        #     if spconv_sign:
        #         default_quant_desc_input = QuantDescriptor(calib_method='histogram')  # histogram
        #         default_quant_desc_weight = QuantDescriptor(num_bits=8, axis=(4), calib_method='max')
        #         default_quant_desc_input._fake_quant = True
        #         default_quant_desc_weight._fake_quant = True
        #         input_quantizer = TensorQuantizer(default_quant_desc_input)
        #         weight_quantizer = TensorQuantizer(default_quant_desc_weight)
        #         module.add_module('_input_quantizer', input_quantizer)
        #         module.add_module('_weight_quantizer', weight_quantizer)
        #         # spconv_layers.append(name + "|" + str(module))

    layer_para["conv2d_layers"] = conv2d_layers
    layer_para["conv1d_layers"] = conv1d_layers
    layer_para["linear_layers"] = linear_layers
    layer_para["spconv_layers"] = spconv_layers
    layer_para["convtranspose_layers"] = convtranspose_layers
    for layer_name, layer_infos in layer_para.items():
        if (layer_name == "conv2d_layers") and conv2d_sign:
            for layer_info in layer_infos:
                para = layer_info.split("|")
                layer = "quant_nn.QuantConv2d" + para[-1]
                _set_module(model, para[0], eval(layer))
        if (layer_name == "conv1d_layers") and conv1d_sign:
            for layer_info in layer_infos:
                para = layer_info.split("|")
                layer = "quant_nn.QuantConv1d" + para[-1]
                _set_module(model, para[0], eval(layer))
        if (layer_name == "linear_layers") and linear_sign:
            for layer_info in layer_infos:
                para = layer_info.split("|")
                layer = "quant_nn.QuantLinear" + para[-1]
                _set_module(model, para[0], eval(layer))
        if (layer_name == "convtranspose_layers") and convtranspose_sign:
            for layer_info in layer_infos:
                para = layer_info.split("|")
                layer = "quant_nn.QuantConvTranspose2d" + para[-1]
                _set_module(model, para[0], eval(layer))
        # if (layer_name == "spconv_layers") and spconv_sign:
        #     nova3d_state.quantization.available = True