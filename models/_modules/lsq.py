"""
@inproceedings{
    esser2020learned,
    title={LEARNED STEP SIZE QUANTIZATION},
    author={Steven K. Esser and Jeffrey L. McKinstry and Deepika Bablani and Rathinakumar Appuswamy and Dharmendra S. Modha},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=rkgO66VKDS}
}
    https://quanoview.readthedocs.io/en/latest/_raw/LSQ.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models._modules import _Conv2dQ, Qmodes, _LinearQ, _ActQ, _Conv2dQCiM 
from torch.autograd import Function
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn.parameter import Parameter
import torch.jit as jit
__all__ = ['Conv2dLSQ', 'LinearLSQ', 'ActLSQ', 'Conv2dLSQCiM']

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


def get_analog_partial_sums_signed(x_q, w_q, conv_stride, conv_padding, conv_dilation, act_bits, act_bit_slice, 
                    weight_bits, weight_bit_slice, arr, weight_scaling_factor, act_scaling_factor):
    mapping = 1
    w_int = w_q/weight_scaling_factor
    x_int = x_q/act_scaling_factor

    num_bit_slice_weight = int(weight_bits/weight_bit_slice)
    num_bit_slice_act = int(act_bits/act_bit_slice)
    kernel_size = w_int.shape[2]
    
    
        
    'making input activations'
    x_unf = nn.Unfold(kernel_size = kernel_size,padding = conv_padding, stride = conv_stride)(x_int).transpose(1,2)
    flatdim = x_unf.shape[-1]
    
    x_unf_sliced = slicing_act(x_unf,act_bits, act_bit_slice).transpose(0,1)
    
    'shape of x_unf_sliced = [batch_size, num_bit_slices_weight, flattened_dim_out ,flat_dim]'
    
    'making weight tensors'
    w_unf = w_int.view(w_int.shape[0],-1).t()
    if (mapping == 1):
        w_unf_sliced = slicing_weights_signed(w_unf, weight_bits, weight_bit_slice)
        'shape of w_unf_sliced = [num_bit_slices_weight, flat_dim, out_channels]'
    num_crossbars = math.ceil(flatdim/arr)
    out_channels = w_int.shape[0]
    fold_x =int( (x_int.shape[-1] - w_int.shape[-1] + 2*conv_padding[0])/conv_stride[0] + 1)
    
    out_unf = torch.cuda.FloatTensor(1).type(torch.float32)
    out_unf.resize_((x_unf_sliced.shape[0],num_crossbars,num_bit_slice_weight, num_bit_slice_act, fold_x*fold_x, out_channels))

    
    'matmul weights'
    #num_zeros = 0
    for i in range (int(flatdim/arr)):
        for j in range (num_bit_slice_act):
            for k in range (num_bit_slice_weight):
                temp_x = x_unf_sliced[:,j,:,i*arr:(i+1)*arr]
                temp_w = w_unf_sliced[k,i*arr:(i+1)*arr,:]
                out_unf[:,i,k,j,:,:] = torch.matmul(temp_x,temp_w)
    
    if (flatdim % arr) != 0 :
        for j in range (num_bit_slice_act):
            for k in range (num_bit_slice_weight):
                temp_x = x_unf_sliced[:,j,:,(int(flatdim/arr))*arr:]
                temp_w = w_unf_sliced[k,(int(flatdim/arr))*arr:,:]
                out_unf[:,num_crossbars-1,k,j,:,:] = torch.matmul(temp_x,temp_w)
    'out_unf_positive shape: [batch_size, num crossbars, num_bit_slices_weight, num_bit_slices_act, flattened_dim_output, out_channels]'
    out_unf = out_unf * weight_scaling_factor * act_scaling_factor


    return out_unf

class get_cim_output_signed(Function):
    
    @staticmethod
    def forward(ctx, x, w, conv_stride, conv_padding, conv_dilation, act_bits, act_bit_slice, 
                        weight_bits, weight_bit_slice, adc_bits, arr,  binary_mask, alpha_cim, weight_scaling_factor, act_scaling_factor, stochastic, signed_act):
        'Mapping : '
        '1 : positive and negative weights separate '
        '2 : twos complement mapping of weights'
        x_int = x / act_scaling_factor
        w_int = w / weight_scaling_factor
        ctx.x_int = x_int.type(torch.int8)
        ctx.act_scaling_factor = act_scaling_factor
        ctx.weight_scaling_factor = weight_scaling_factor
        ctx.stride = conv_stride
        ctx.padding = conv_padding
        ctx.dilation = conv_dilation
        ctx.binary_mask = binary_mask
        ctx.alpha_cim = alpha_cim
        ctx.bit_slice_w = weight_bit_slice
        ctx.bit_slice_a = act_bit_slice
        ctx.out_channels = w_int.shape[0]
        ctx.input_size = (x_int.shape[2], x_int.shape[3])        
        ctx.kernel_size = (w_int.shape[2], w_int.shape[3])
        ctx.in_channels = w_int.shape[1]
        mapping = 1
        intermediate_dtype = torch.float32
        num_bit_slice_weight = int(weight_bits/weight_bit_slice)
        ctx.num_bit_slice_weight = num_bit_slice_weight
        num_bit_slice_act = int(act_bits/act_bit_slice)
        ctx.num_bit_slice_act = num_bit_slice_act
        ctx.act_bits = act_bits
        ctx.adc_bits = adc_bits
        kernel_size = w_int.shape[2]
        out_channels = w_int.shape[0]
        fold_x =int( (x_int.shape[-1] - w_int.shape[-1] + 2*conv_padding[0])/conv_stride[0] + 1)
        
        Qp_adc = (2 ** (adc_bits-1)) - 1 
        Qn_adc = -1*(2 ** (adc_bits-1)) 
        if(adc_bits == 1) or (adc_bits == 1.5):
            Qp_adc = 1
            Qn_adc = -1
        ctx.Qp_adc = Qp_adc
        ctx.Qn_adc = Qn_adc

        ctx.stochastic = stochastic
        ctx.signed_act = signed_act
        # ctx.sigmoid_sharpness = sigmoid_sharpness
        if stochastic:
            assert adc_bits == 1.5 # currently only support stochastic quantization with near adcless
        
            
        'making input activations'
        x_unf = nn.Unfold(kernel_size = kernel_size,padding = conv_padding, stride = conv_stride)(x_int).transpose(1,2)
        
        flatdim = x_unf.shape[-1]
        ctx.flatdim = flatdim
        ctx.arr = arr
        if signed_act:
            x_unf_sliced = slicing_act_signed(x_unf, ctx.act_bits, act_bit_slice).transpose(0,1)
        else:
            x_unf_sliced = slicing_act(x_unf,act_bits, act_bit_slice).transpose(0,1).type(intermediate_dtype)
        'shape of x_unf_sliced = [batch_size, num_bit_slices_act, flattened_dim_out ,flat_dim]'
        
        'making weight tensors'
        w_unf = w_int.view(w_int.shape[0],-1).t()
        if (mapping == 1):
            w_unf_sliced = slicing_weights_signed(w_unf, weight_bits, weight_bit_slice).type(intermediate_dtype)
            #var = torch.ones((w_unf_sliced.shape)).to(w_unf_sliced)
            #variation = var.log_normal_(0,0.04)
            #w_unf_sliced = torch.mul(w_unf_sliced,variation)
            
            ctx.w_unf_sliced = w_unf_sliced.type(torch.int8)
            
            
            'shape of w_unf_sliced = [num_bit_slices_weight, flat_dim, out_channels]'
            'shape of w_unf_sliced unsigned = [pos/neg, num_bit_slices_weight, flat_dim, out_channels]'
        
        num_crossbars = math.ceil(flatdim/arr)
        ctx.num_crossbars = num_crossbars
        
        out_unf = torch.cuda.FloatTensor(1).type(torch.float16).cuda()
        out_unf.resize_((x_unf_sliced.shape[0],num_crossbars,num_bit_slice_weight, num_bit_slice_act, fold_x*fold_x, out_channels))
        'matmul weights'
        for i in range (int(flatdim/arr)):
            for j in range (num_bit_slice_act):
                for k in range (num_bit_slice_weight):
                    temp_x = x_unf_sliced[:,j,:,i*arr:(i+1)*arr]
                    temp_w = w_unf_sliced[k,i*arr:(i+1)*arr,:]
                    out_unf[:,i,k,j,:,:] = ((torch.matmul(temp_x,temp_w)))
                    
        if (flatdim % arr) != 0 :
            for s in range(2):
                for j in range (num_bit_slice_act):
                    for k in range (num_bit_slice_weight):
                        temp_x = x_unf_sliced[:,j,:,(int(flatdim/arr))*arr:]
                        temp_w = w_unf_sliced[k,(int(flatdim/arr))*arr:,:]
                        out_unf[:,num_crossbars-1,k,j,:,:] = ((torch.matmul(temp_x,temp_w))) 
            
            'out_unf shape: [batch_size, num crossbars, num_bit_slices_weight, num_bit_slices_act, flattened_dim_output, out_channels]'
            'out_unf_unsigned shape: [batch_size, pos/neg, num crossbars, num_bit_slices_weight, num_bit_slices_act, flattened_dim_output, out_channels]'

        
        
        ctx.ps_int = out_unf

        
        out_unf = out_unf * weight_scaling_factor * act_scaling_factor
        'ADC quantization:'
        if adc_bits == 0:
            #FP ADC
            adc_out = out_unf
        elif adc_bits == 1:
            adc_out = torch.sign(out_unf)
            adc_out = torch.mul(adc_out, alpha_cim)
        elif adc_bits == 1.5 :
            #Near ADCLess
            if stochastic:
                sigmoid_sharpness = 0.01
                ### stochastic sigmoid quantization
                num_iter = 50
                sigmoid_1 = torch.sigmoid((out_unf - 0.5 * alpha_cim)/sigmoid_sharpness)
                sigmoid_2 = torch.sigmoid((out_unf + 0.5 * alpha_cim)/sigmoid_sharpness)
                
                stoch_sigmoid_1 = 0
                stoch_sigmoid_2 = 0
                for i in range(num_iter):
                    stoch_sigmoid_1 += torch.ceil(sigmoid_1 - torch.cuda.FloatTensor(out_unf.size()).uniform_())
                    stoch_sigmoid_2 += torch.ceil(sigmoid_2 - torch.cuda.FloatTensor(out_unf.size()).uniform_())
                
                adc_out =  stoch_sigmoid_1/num_iter + stoch_sigmoid_2/num_iter - 1
                adc_out = torch.round(adc_out).clamp(Qn_adc, Qp_adc)
                adc_out = torch.mul(adc_out, alpha_cim)

            else:
                adc_out = (out_unf)/alpha_cim
                adc_out = torch.round(adc_out).clamp(Qn_adc, Qp_adc)
                adc_out = torch.mul(adc_out, alpha_cim)
        else:
            #Higher precision ADC does not use scale factor
            adc_out = (out_unf)/(weight_scaling_factor * act_scaling_factor)
            adc_out = torch.round(adc_out).clamp(Qn_adc, Qp_adc)
            adc_out = adc_out * weight_scaling_factor * act_scaling_factor

        'shift and add'
        out_sna = torch.sum(torch.mul(adc_out, binary_mask),dim = (1,2,3))
        
        
        
        return out_sna
        

        
        

    @staticmethod
    def backward(ctx, grad_output):
        #x_unf_sliced = ctx.x_unf_sliced.type(torch.float32)
        
        weight_scaling_factor = ctx.weight_scaling_factor
        act_scaling_factor = ctx.act_scaling_factor
        w_unf_sliced = ctx.w_unf_sliced.type(torch.float32)
        'shape of w_unf_sliced = [num_bit_slices_weight num_bit_slice_act, flat_dim, out_channels]'
        
        w_unf_sliced = w_unf_sliced * weight_scaling_factor
        x_int = ctx.x_int.type(torch.float32)
        
        alpha_cim = ctx.alpha_cim
        adc_bits = ctx.adc_bits
        if adc_bits == 1:
            ps_int = ctx.ps_int.type(torch.float32)
            ps = ps_int * weight_scaling_factor * act_scaling_factor
            ps = ps/alpha_cim
        elif adc_bits == 1.5:
            ps_int = ctx.ps_int.type(torch.float32)
            ps = ps_int * weight_scaling_factor * act_scaling_factor
            ps = ps/alpha_cim
        else:
            #adc bits = 0 or > 1.5 do not have scale factor
            ps = ctx.ps_int.type(torch.float32)

        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        kernel_size = ctx.kernel_size
        output_size = ctx.input_size
        out_channels = ctx.out_channels
        in_channels = ctx.in_channels
        num_bit_slice_weight = ctx.num_bit_slice_weight
        num_bit_slice_act = ctx.num_bit_slice_act
        bit_slice_w = ctx.bit_slice_w
        bit_slice_a = ctx.bit_slice_a
        
        Qn_adc = ctx.Qn_adc
        Qp_adc = ctx.Qp_adc
        adc_bits = ctx.adc_bits
        num_crossbars = ctx.num_crossbars
        binary_mask = ctx.binary_mask
        flatdim = ctx.flatdim
        arr = ctx.arr
        
        #
        x_unf = nn.Unfold(kernel_size = kernel_size,padding = padding, stride = stride)(x_int).transpose(1,2)
        if ctx.signed_act:
            x_unf_sliced = slicing_act_signed(x_unf, ctx.act_bits, bit_slice_a).transpose(0,1)
        else:
            x_unf_sliced = slicing_act(x_unf, ctx.act_bits, bit_slice_a).transpose(0,1)
        x_unf_sliced = x_unf_sliced * act_scaling_factor
        'shape of x_unf_sliced = [batch_size, num_bit_slices_weight, num_bit_slices_act, flattened_dim_out ,flat_dim]'

        grad_temp = grad_output.clone()
        'grad_output shape [batch_size, flattened_output_dim, out_channels]'
        
        grad_temp = grad_temp.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,num_crossbars, num_bit_slice_weight, num_bit_slice_act, 1,1)
        'grad temp shape [batch_size, num_crossbars, num_bit_slice_weight, num_bit_slice_act, flatenned_dim_output, out_channels]'
        
        
        'grad for Shift and Add'
        grad_temp = torch.mul(grad_temp, binary_mask)
        grad_output_after_adc = grad_temp.clone()

        'effect of ADC clamping'
        greater = ps.ge(Qp_adc+1e-5)
        lesser = ps.le(Qn_adc-1e-5)
        
        grad_temp[torch.logical_or(greater,lesser)] = 0

        # 'effect of division with alpha and multiplication with beta'
        # if adc_bits == 1.5:
        #     grad_temp = (alpha_cim * grad_temp)/beta_cim 
        
        
        'gradients for scale parameter alpha'
        if adc_bits == 1:
            grad_alpha = torch.sign(ps.clone())
            grad_alpha = torch.mul(grad_alpha,1.0/math.sqrt(ps.numel() * (Qp_adc)))
            grad_alpha = torch.mul(grad_alpha, grad_output_after_adc)
            grad_alpha = torch.sum(grad_alpha,dim = (0,4), keepdim= True)
        elif adc_bits == 1.5:
            grad_alpha = torch.round(ps.clone())
            grad_alpha[greater] = Qp_adc 
            grad_alpha[lesser] = Qn_adc 
            grad_alpha = torch.mul(grad_alpha,1.0/math.sqrt(ps.numel() * (Qp_adc)))
            grad_alpha = torch.mul(grad_alpha, grad_output_after_adc)
            grad_alpha = torch.sum(grad_alpha,dim = (0,4), keepdim= True)
        else :
            grad_alpha = None
        
        grad_input = torch.cuda.FloatTensor(1).resize_(x_unf_sliced.shape[0], num_bit_slice_weight, num_bit_slice_act,x_unf_sliced.shape[2],x_unf_sliced.shape[3])
        grad_weight = torch.cuda.FloatTensor(1).resize_(num_bit_slice_weight,num_bit_slice_act, w_unf_sliced.shape[1],w_unf_sliced.shape[2])
        for i in range (int(flatdim/arr)):
            for j in range (num_bit_slice_act):
                for k in range (num_bit_slice_weight):
                    temp_x = x_unf_sliced.transpose(2,3)[:,j,i*arr:(i+1)*arr,:]
                    temp_w = w_unf_sliced[k,i*arr:(i+1)*arr,:].transpose(0,1)
                    
                    grad_input[:,k,j,:,i*arr:(i+1)*arr] = torch.matmul(grad_temp[:,i,k,j,:,:], temp_w)
                    grad_weight[k,j,i*arr:(i+1)*arr,:] = torch.matmul(temp_x,grad_temp[:,i,k,j,:,:]).sum(0)
            
            'shape grad_temp = [batch, numCrossbar, num_w_bs, num_act_bs, flattened_out_dim, out_channels]'
            'shape grad_input = [batch,num_w_bs, num_act_bs, flattened_out_dim ,flat_dim ]'               
            'shape grad_weight = [num_w_bs, num_act_bs,flat_dim, out_channels] '    
        if (flatdim % arr) != 0 :
            for j in range (num_bit_slice_act):
                for k in range (num_bit_slice_weight):
                    temp_x = x_unf_sliced.transpose(2,3)[:,j,(int(flatdim/arr))*arr:,:]
                    temp_w = w_unf_sliced[k,(int(flatdim/arr))*arr:,:].transpose(0,1)
                    grad_input[:,k,j,:,(int(flatdim/arr))*arr:] = torch.matmul(grad_temp[:,num_crossbars-1,k,j,:,:], temp_w)
                    grad_weight[k,j,(int(flatdim/arr))*arr:,:] = torch.matmul(temp_x,grad_temp[:,num_crossbars-1,k,j,:,:]).sum(0)
        
       
        
        
        'accumulate across act bit slice and weight bit slice for grad_weight '
        grad_weight = torch.sum(grad_weight, dim = 1)
        for i in range(1,num_bit_slice_weight):
            grad_weight[i,:,:] = grad_weight[i,:,:] / (2**(bit_slice_w)) ** i
        
        grad_weight = torch.mean(grad_weight, dim = 0)
        
        'reshape grad_weight'
        grad_weight = grad_weight.transpose(0,1).view(out_channels,in_channels, kernel_size[0], kernel_size[0])
        
        'accumulate across act bit slice and weight bit slice for grad_input'
        grad_input = torch.sum(grad_input, dim = 1)
        for i in range(1,num_bit_slice_act):
            grad_input[:,i,:,:] = grad_input[:,i,:,:] / (2**(bit_slice_a)) ** i
        
        grad_input = torch.mean(grad_input, dim = 1)
        
        
        'backward for unfold is nn.Fold'
        grad_temp_ = grad_input.clone().transpose(1,2)
        
        grad_input = torch.nn.Fold(output_size , kernel_size, dilation, padding,stride)(grad_temp_)
        
        
        
        return grad_input, grad_weight, None, None,  None, None , None, None, None, None, None, None, grad_alpha, None, None, None, None


class Conv2dLSQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=8, **kwargs):
        super(Conv2dLSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits_w)
    
    
    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        
        x_q, act_scaling_factor = x
        # w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            # self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(self.weight.abs().max() * 2)
            self.init_state.fill_(1)
            
        """  
        Implementation according to paper. 
        Feels wrong ...
        When we initialize the alpha as a big number (e.g., self.weight.abs().max() * 2), 
        the clamp function can be skipped.
        Then we get w_q = w / alpha * alpha = w, and $\frac{\partial w_q}{\partial \alpha} = 0$
        As a result, I don't think the pseudo-code in the paper echoes the formula.
       
        Please see jupyter/STE_LSQ.ipynb fo detailed comparison.
        """
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1: 31GB GPU memory (AlexNet w4a4 bs 2048) 17min/epoch
        weight_scaling_factor = grad_scale(self.alpha, g)
        w_q = round_pass((self.weight / weight_scaling_factor).clamp(Qn, Qp))
        # w = w.clamp(Qn, Qp)
        # q_w = round_pass(w)
        # w_q = q_w * alpha
        
        # Method2: 25GB GPU memory (AlexNet w4a4 bs 2048) 32min/epoch
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        # wq = y.transpose(0, 1).reshape(self.weight.shape).detach() + self.weight - self.weight.detach()
        return F.conv2d(x_q, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups) * act_scaling_factor * weight_scaling_factor

def slicing_weights_signed(w_int, bits, bit_slice):
    """
    w_int: signed int to be sliced
    bits: number of bits
    bit_slice : bit width of bit_slice
    """
    tensor = w_int.clone()
    tensor[tensor.le(0)] = 0
    tensor_positive = tensor
    
    tensor = w_int.clone()
    tensor[tensor.ge(0)] = 0
    tensor_negative = -1*tensor
    
    tensor = torch.stack([tensor_positive, tensor_negative])
    tensor = tensor.unsqueeze(1).repeat(1,int(bits/bit_slice),1,1)
    for i in range(1,int(bits/bit_slice)):
        tensor[:,i,:,:] = torch.floor(tensor[:,i,:,:]/(2**bit_slice)**i)
    
    tensor = torch.remainder(tensor, 2**bit_slice)
    tensor = tensor[0] - tensor[1]
    
    
    
    'tensor_sliced : [num_bit_slice, **kwargs]'
    ''
    return tensor.float()

def slicing_act(x_int, bits, bit_slice):
    """
    x_int: unsigned int to be sliced
    bits: number of bits
    bit_slice : bit width of bit_slice
    """
    tensor = x_int.clone()
    tensor = tensor.unsqueeze(0).repeat(int(bits/bit_slice),1,1,1)
    
    for i in range(1,int(bits/bit_slice)):
        tensor[i,:,:,:] = torch.floor(tensor[i,:,:,:]/(2**bit_slice)**i)
    
    tensor = torch.remainder(tensor, 2**bit_slice)
    'LSB = sliced_tensor[0], MSB = sliced_tensor[bits/bit_slice]'
    return tensor.float()    


def slicing_act_signed(w_int, bits, bit_slice):
    """
    w_int: signed int to be sliced
    bits: number of bits
    bit_slice : bit width of bit_slice
    """
    tensor = w_int.clone()
    tensor[tensor.le(0)] = 0
    tensor_positive = tensor
    
    tensor = w_int.clone()
    tensor[tensor.ge(0)] = 0
    tensor_negative = -1*tensor
    
    tensor = torch.stack([tensor_positive, tensor_negative])
    tensor = tensor.unsqueeze(1).repeat(1,int(bits/bit_slice),1,1,1)
    for i in range(1,int(bits/bit_slice)):
        tensor[:,i,:,:] = torch.floor(tensor[:,i,:,:]/(2**bit_slice)**i)
    
    tensor = torch.remainder(tensor, 2**bit_slice)
    tensor = tensor[0] - tensor[1]
    
    
    
    'tensor_sliced : [num_bit_slice, **kwargs]'
    ''
    return tensor.float()
    
class Conv2dLSQCiM(_Conv2dQCiM):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=8, nbits_a=8,nbits_alpha=8, wbitslice=1, abitslice=1, xbar=64, adcbits=6,stochastic_quant=False, **kwargs):

        
        super(Conv2dLSQCiM, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits_w=nbits_w, nbits_a = nbits_a, nbits_alpha=nbits_alpha, wbitslice=wbitslice, abitslice=abitslice, xbar=xbar, adcbits=adcbits, stochastic_quant=stochastic_quant)
        
    
    def forward(self, x):
        Qn_w = -2 ** (self.nbits_w - 1)
        Qp_w = 2 ** (self.nbits_w - 1) - 1
        Qp_adc = 2 ** (self.adcbits-1) -1
        Qn_adc = -2 ** (self.adcbits-1)
        
        if self.adcbits == 1 or self.adcbits == 1.5:
            #Binary/Ternary ps quantization
            Qp_adc = 1
            Qn_adc = -1
        if self.training and self.init_state == 0:
            if x.min() < -1e-5:
                self.signed_act.data.fill_(1)
        
    
        Qn_a = 0
        Qp_a = 2 ** self.nbits_a - 1
        if self.training and self.init_state == 0:
            self.alpha_act.data.copy_(2 * x.abs().mean() / math.sqrt(Qp_a))
            self.alpha_weight.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp_w))
            self.init_state.fill_(1)
        if self.binary_mask.device != x.device:
            self.binary_mask = self.binary_mask.to(x.device)

        # Quantize activations
        ga = 1.0 / math.sqrt(x.numel() * Qp_a)
        act_scaling_factor = grad_scale(self.alpha_act, ga)
        x_q = round_pass((x / act_scaling_factor).clamp(Qn_a, Qp_a)) * act_scaling_factor
        
        
        #Quantize Weights
        gw = 1.0 / math.sqrt(self.weight.numel() * Qp_w)
        weight_scaling_factor = grad_scale(self.alpha_weight, gw)
        w_q = round_pass((self.weight / weight_scaling_factor).clamp(Qn_w, Qp_w)) * weight_scaling_factor

        if (self.training) and (self.init_state_cim == 0) and (self.alpha_cim is not None):
            with torch.no_grad():
                cim_outputs = get_analog_partial_sums_signed(x_q, w_q, self.stride, self.padding, self.dilation, self.nbits_a, self.abitslice, self.nbits_w, self.wbitslice, self.xbar, weight_scaling_factor, act_scaling_factor)
                temp = 2.0*(cim_outputs).abs().mean(dim = (0,4), keepdim = True)/ math.sqrt(Qp_adc)
                temp[temp == 0] = 1.0 * weight_scaling_factor * act_scaling_factor
                self.alpha_cim.data.copy_(temp)
                self.init_state_cim.fill_(1)
        
        #Quantize Alpha
        Qp_alpha = 2 ** self.nbits_alpha - 1
        Qn_alpha = 1
        if self.alpha_cim is not None:
            alpha = self.alpha_cim
            alpha_scale = (alpha.max() - alpha.min())/(Qp_alpha - Qn_alpha)
            alpha_q = round_pass(alpha / alpha_scale).clamp(Qn_alpha, Qp_alpha) * alpha_scale 
        else:
            alpha_q = None
        
        if self.adcbits != 0:
        
            #Get cim outputs
            out = get_cim_output_signed.apply(x_q, w_q, self.stride, self.padding, self.dilation, self.nbits_a, self.abitslice, self.nbits_w, self.wbitslice, self.adcbits, self.xbar,  self.binary_mask, alpha_q,  weight_scaling_factor, act_scaling_factor, self.stochastic_quant, self.signed_act)
            #Reshape to get final layer outputs
            fold_x =int( (x_q.shape[-1] - self.weight.shape[-1] + 2*self.padding[0])/self.stride[0] + 1)
            out = out.transpose(1,2).view(x_q.shape[0],self.out_channels,fold_x,fold_x)
            if self.bias is not None:
                out = out + self.bias
        else:
            out = torch.nn.functional.conv2d(x_q,w_q, self.bias, self.stride, self.padding, self.dilation)
    
        
        return out
    

class LinearLSQ(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, **kwargs):
        super(LinearLSQ, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias, nbits=nbits_w)

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1:
        alpha = grad_scale(self.alpha, g)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        # w = self.weight / alpha
        # w = w.clamp(Qn, Qp)
        # q_w = round_pass(w)
        # w_q = q_w * alpha

        # Method2:
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        return F.linear(x, w_q, self.bias)


class ActLSQ(_ActQ):
    def __init__(self, nbits_a=4, **kwargs):
        super(ActLSQ, self).__init__(nbits=nbits_a)

    def forward(self, x):
        if self.alpha is None:
            return x

        if self.training and self.init_state == 0:
            # The init alpha for activation is very very important as the experimental results shows.
            # Please select a init_rate for activation.
            # self.alpha.data.copy_(x.max() / 2 ** (self.nbits - 1) * self.init_rate)
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            if self.signed == 1:
                Qn = -2 ** (self.nbits - 1)
                Qp = 2 ** (self.nbits - 1) - 1
            else:
                Qn = 0
                Qp = 2 ** self.nbits - 1
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)

        if self.signed == 1:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1

        g = 1.0 / math.sqrt(x.numel() * Qp)
        #print(x)
        # Method1:
        alpha = grad_scale(self.alpha, g)
        x_q = round_pass((x / alpha).clamp(Qn, Qp)) 
        # x = x / alpha
        # x = x.clamp(Qn, Qp)
        # q_x = round_pass(x)
        # x_q = q_x * alpha

        # Method2:
        # x_q = FunLSQ.apply(x, self.alpha, g, Qn, Qp)
        return x_q , alpha
