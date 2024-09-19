r"""
    Replace `conv` with `convq`;
    Replace `Linear` with `LinearQ`
"""

import torch.nn as nn
import torch.jit as jit

__all__ = ['ReplaceModuleTool']


class ReplaceModuleTool(object):
    def __init__(self, model, replace_map, replace_first_layer, **kwargs):
        self.is_first = True
        self.acts = []
        self.convs = []
        self.linears = []
        self.replace_map = replace_map
        self.model = model
        self.replace_first_layer = replace_first_layer
        self.kwargs = kwargs

    def replace(self):
        self.replace_module_recursively(self.model)

    def replace_module_recursively(self, model):
        for module_name in model._modules:
            if isinstance(model._modules[module_name], (nn.Linear, nn.Conv2d)):
                model._modules[module_name] = self.replace_module(model._modules[module_name])
            elif len(model._modules[module_name]._modules) > 0:
                self.replace_module_recursively(model._modules[module_name])
        return model

    def replace_module(self, module_ori):
        if 'Linear' in self.replace_map.keys() and isinstance(module_ori, nn.Linear):
            m = module_ori
            has_bias = m.bias is not None
            linear_map = self.replace_map['Linear']
            Act, Linear = None, None
            if isinstance(linear_map, list) and len(linear_map) == 2:
                [Act, Linear] = linear_map
                assert issubclass(Act, nn.Module)
                assert issubclass(Linear, nn.Linear)
            elif (isinstance(linear_map, list) and len(linear_map) == 1) or issubclass(linear_map, nn.Linear):
                [Linear] = linear_map
            else:
                raise ValueError
            my_fc = Linear(m.in_features, m.out_features, bias=has_bias, **self.kwargs)
            self.linears.append(my_fc)
            fc_st_dict = m.state_dict()
            W = fc_st_dict['weight']
            my_fc.weight.data.copy_(W)
            if has_bias:
                bias = fc_st_dict['bias']
                my_fc.bias.data.copy_(bias)
            if Act is not None:
                my_act = Act(**self.kwargs)
                self.acts.append(my_act)
                my_m = nn.Sequential(*[my_act, my_fc])
            else:
                my_m = my_fc
            my_m.to(m.weight.device)
            del m
            return my_m
        elif 'Conv2d' in self.replace_map.keys() and isinstance(module_ori, nn.Conv2d):
            if self.is_first and not self.replace_first_layer:
                self.is_first = False
                return module_ori
            else:
                m = module_ori
                has_bias = m.bias is not None
                conv_map = self.replace_map['Conv2d']
                Act, Conv = None, None
                if isinstance(conv_map, list) and len(conv_map) == 2:
                    [Act, Conv] = conv_map
                    assert issubclass(Act, nn.Module)
                    assert issubclass(Conv, nn.Conv2d)
                elif (isinstance(conv_map, list) and len(conv_map) == 1) or issubclass(conv_map, nn.Conv2d) or issubclass(conv_map, jit.ScriptModule) :
                    [Conv] = conv_map
                    
                else:
                    raise ValueError
                
                my_conv = Conv(m.in_channels, m.out_channels, m.kernel_size, m.stride,
                               m.padding, m.dilation, groups=m.groups, bias=has_bias,
                               **self.kwargs)
                
                self.convs.append(my_conv)
                conv_st_dict = m.state_dict()
                W = conv_st_dict['weight']
                my_conv.weight.data.copy_(W)
                if has_bias:
                    bias = conv_st_dict['bias']
                    my_conv.bias.data.copy_(bias)
                if Act is not None:
                    my_act = Act(**self.kwargs)
                    if self.is_first:
                        self.is_first = False
                        if my_act.nbits > 0 and my_act.nbits < 8:  # fake quantization
                            my_act.set_bit(8)  # quantize data to 8-bit
                    self.acts.append(my_act)
                    my_m = nn.Sequential(*[my_act, my_conv])
                else:
                    my_m = my_conv
                my_m.to(m.weight.device)
                del m
                return my_m
        else:
            return module_ori
