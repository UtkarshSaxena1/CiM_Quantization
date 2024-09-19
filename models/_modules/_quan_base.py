"""
    Quantized modules: the base class
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.jit as jit

import math
from enum import Enum

__all__ = ['Qmodes',  '_Conv2dQ', '_LinearQ', '_ActQ',
           'truncation', 'get_sparsity_mask', 'FunStopGradient', 'round_pass', 'grad_scale', 'Qmodes_cim', '_Conv2dQCiM']


class Qmodes(Enum):
    layer_wise = 1
    kernel_wise = 2

class Qmodes_cim(Enum):
    column_wise = 1
    bit_wise = 2


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def get_sparsity_mask(param, sparsity):
    bottomk, _ = torch.topk(param.abs().view(-1), int(sparsity * param.numel()), largest=False, sorted=True)
    threshold = bottomk.data[-1]  # This is the largest element from the group of elements that we prune away
    return torch.gt(torch.abs(param), threshold).type(param.type())


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


class FunStopGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, stopGradientMask):
        ctx.save_for_backward(stopGradientMask)
        return weight

    @staticmethod
    def backward(ctx, grad_outputs):
        stopGradientMask, = ctx.saved_tensors
        grad_inputs = grad_outputs * stopGradientMask
        return grad_inputs, None


def log_shift(value_fp):
    value_shift = 2 ** (torch.log2(value_fp).ceil())
    return value_shift


def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def get_quantized_range(num_bits, signed=True):
    if signed:
        n = 2 ** (num_bits - 1)
        return -n, n - 1
    return 0, 2 ** num_bits - 1


def linear_quantize(input, scale_factor, inplace=False):
    if inplace:
        input.mul_(scale_factor).round_()
        return input
    return torch.round(scale_factor * input)


def linear_quantize_clamp(input, scale_factor, clamp_min, clamp_max, inplace=False):
    output = linear_quantize(input, scale_factor, inplace)
    return clamp(output, clamp_min, clamp_max, inplace)


def linear_dequantize(input, scale_factor, inplace=False):
    if inplace:
        input.div_(scale_factor)
        return input
    return input / scale_factor


def truncation(fp_data, nbits=8):
    il = torch.log2(torch.max(fp_data.max(), fp_data.min().abs())) + 1
    il = math.ceil(il - 1e-5)
    qcode = nbits - il
    scale_factor = 2 ** qcode
    clamp_min, clamp_max = get_quantized_range(nbits, signed=True)
    q_data = linear_quantize_clamp(fp_data, scale_factor, clamp_min, clamp_max)
    q_data = linear_dequantize(q_data, scale_factor)
    return q_data, qcode


def get_default_kwargs_q(kwargs_q, layer_type):
    default = {
        'nbits': 4
    }
    
# =============================================================================
#     if isinstance(layer_type, _Conv2dQCiM):
#         default.update({
#             'mode': Qmodes.layer_wise})
# =============================================================================
    if isinstance(layer_type, _Conv2dQCiM) :
        default.update({
            'cimmode': Qmodes_cim.bit_wise})
    
    if isinstance(layer_type, _Conv2dQ) or isinstance(layer_type, _Conv2dQCiM) :
        default.update({
            'mode': Qmodes.layer_wise})
    
    elif isinstance(layer_type, _LinearQ):
        pass
    elif isinstance(layer_type, _ActQ):
        pass
        # default.update({
        #     'signed': 'Auto'})
    else:
        assert NotImplementedError
        return
    for k, v in default.items():
        if k not in kwargs_q:
            kwargs_q[k] = v
            
    return kwargs_q


class _Conv2dQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs_q):
        super(_Conv2dQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            self.register_parameter('alpha_cim', None)
            return
        self.q_mode = kwargs_q['mode']
        if self.q_mode == Qmodes.kernel_wise:
            self.alpha = Parameter(torch.Tensor(out_channels))
        else:  # layer-wise quantization
            self.alpha = Parameter(torch.Tensor(1))
        

            
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.kwargs_q['nbits'] = nbits

    def extra_repr(self):
        s_prefix = super(_Conv2dQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)

class _Conv2dQCiM(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs_q):
        super(_Conv2dQCiM, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits_w = kwargs_q['nbits_w']
        self.nbits_a = kwargs_q['nbits_a']
        self.nbits_alpha = kwargs_q['nbits_alpha']
        
        self.wbitslice = kwargs_q['wbitslice']
        self.abitslice = kwargs_q['abitslice']
        self.xbar = kwargs_q['xbar']
        
        self.adcbits = kwargs_q['adcbits']
        
        if self.nbits_w < 0:
            self.register_parameter('alpha', None)
            self.register_parameter('alpha_cim', None)
            return
        self.q_mode = kwargs_q['mode']
        
        
        
        flattened_dim = in_channels * kernel_size[0] * kernel_size[1]
        
        self.num_xbars = int(math.ceil(flattened_dim/self.xbar))
        
        self.num_bit_slice_weight = int(self.nbits_w/self.wbitslice)
        self.num_bit_slice_act = int(self.nbits_a/self.abitslice)
        self.analog_partial_sums_positive = torch.zeros((1,1,1,1,1,1))
        
        self.binary_mask = torch.ones(self.num_bit_slice_weight, self.num_bit_slice_act)
        
        for i in range(self.num_bit_slice_act):
            for j in range(self.num_bit_slice_weight):
                self.binary_mask[j,i] = ((2**self.abitslice)**i)*((2**self.wbitslice)**j)
            
        self.binary_mask =self.binary_mask.view(1,1,self.binary_mask.shape[0],self.binary_mask.shape[1],1,1)
        self.binary_mask = self.binary_mask.type(torch.int8)
        'binary mask shape [1,1,num_bit_slices_weight, num_bitslices_act,1,1]'
        
        
        if self.adcbits == 1.5 :
            self.alpha_cim = Parameter(torch.ones(1,self.num_xbars, self.num_bit_slice_weight, self.num_bit_slice_act, 1, self.out_channels), requires_grad=True)
        elif self.adcbits == 1:
            self.alpha_cim = Parameter(torch.ones(1,self.num_xbars, self.num_bit_slice_weight, self.num_bit_slice_act, 1, self.out_channels), requires_grad=True)
        else : 
            # ADC Bits  == 0 or ADC_bits > 1.5 have no scale factor
            self.alpha_cim = None
        
        
        #self.alpha_weight = Parameter(torch.ones(self.num_xbars, self.out_channels), requires_grad = True)
        #self.alpha_act = Parameter(torch.ones(self.num_xbars), requires_grad = True)
        self.alpha_weight = Parameter(torch.ones(1), requires_grad = True)
        self.alpha_act = Parameter(torch.ones(1), requires_grad = True)
        # self.alpha_cim = Parameter(torch.ones(1,self.num_xbars, self.num_bit_slice_weight, self.num_bit_slice_act, 1, self.out_channels), requires_grad=True)
        self.temp_buffer = torch.zeros(1,self.num_xbars,1,1, 1,self.out_channels)
         
                   
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('init_state_cim', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.kwargs_q['nbits'] = nbits

    def extra_repr(self):
        s_prefix = super(_Conv2dQCiM, self).extra_repr()
        # if self.alpha is None:
            # return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)
    
    
class _LinearQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs_q):
        super(_LinearQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        s_prefix = super(_LinearQ, self).extra_repr()
        if self.alpha is None:
            return '{}, fake'.format(s_prefix)
        return '{}, {}'.format(s_prefix, self.kwargs_q)


class _ActQ(nn.Module):
    def __init__(self, **kwargs_q):
        super(_ActQ, self).__init__()
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        self.nbits = kwargs_q['nbits']
        if self.nbits < 0:
            self.register_parameter('alpha', None)
            return
        # self.signed = kwargs_q['signed']
        self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('signed', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.kwargs_q['nbits'] = nbits

    def extra_repr(self):
        # s_prefix = super(_ActQ, self).extra_repr()
        if self.alpha is None:
            return 'fake'
        return '{}'.format(self.kwargs_q)
