#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 18:46:22 2022

@author: saxenau
"""


import torch
import sys
sys.path.insert(0, '/home/nano01/a/saxenau/ADCLess/EfficientPyTorch/')
import models._modules as my_nn
from torch.autograd import Function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models._modules import _Conv2dQ, Qmodes, _LinearQ, _ActQ, _Conv2dQCiM 
from torch.autograd import Function
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn.parameter import Parameter
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

collect = {}  

class slicing_act(Function):
    
    @staticmethod
    def forward(ctx, x_int, bits,bit_slice):
        tensor = x_int.clone()
        ctx.bits = bits
        ctx.bit_slice = bit_slice
        tensor = tensor.unsqueeze(0).repeat(int(bits/bit_slice),1,1,1)
        
        for i in range(1,int(bits/bit_slice)):
            tensor[i,:,:,:] = torch.floor(tensor[i,:,:,:]/(2**bit_slice)**i)
        
        tensor = torch.remainder(tensor, 2**bit_slice)
        'LSB = sliced_tensor[0], MSB = sliced_tensor[bits/bit_slice]'
        return tensor    

    @staticmethod
    def backward(ctx, grad_output):
        bits = ctx.bits
        bit_slice = ctx.bit_slice
        num_bit_slice = int(bits/bit_slice) 
        grad_input = grad_output.clone()
        for i in range(1,num_bit_slice):
            grad_input[i,:] = grad_input[i,:] / (2**(bit_slice)) ** i
        
        grad_input = torch.mean(grad_input, dim = 0)
        
        
        return grad_input, None, None

class slicing_weights(Function):
    
    @staticmethod
    def forward(ctx, w_int, bits,bit_slice):
        tensor = w_int.clone()
        tensor[tensor.le(0)] = 0
        tensor_positive = tensor
        ctx.bits = bits
        ctx.bit_slice = bit_slice
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
        return tensor   

    @staticmethod
    def backward(ctx, grad_output):
        bits = ctx.bits
        bit_slice = ctx.bit_slice
        num_bit_slice = int(bits/bit_slice) 
        grad_input = grad_output.clone()
        for i in range(1,num_bit_slice):
            grad_input[i,:] = grad_input[i,:] / (2**(bit_slice)) ** i
        
        grad_input = torch.mean(grad_input, dim = 0)
        
        #print(grad_input)
        return grad_input, None, None


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


class get_adcless_cim_output(Function):
    

    @staticmethod
    def forward(ctx, x_int, w_int, conv_stride, conv_padding, conv_dilation, act_bits, act_bit_slice, 
                        weight_bits, weight_bit_slice, adc_bits, arr,  binary_mask, alpha_cim, beta_cim):
        'Mapping : '
        '1 : positive and negative weights separate '
        '2 : twos complement mapping of weights'
# =============================================================================
#         with profile(activities=[ProfilerActivity.CPU],
#             profile_memory=True, record_shapes=True) as prof:
# =============================================================================
        #print(alpha_cim)
        ctx.x_int = x_int.type(torch.int8)
        ctx.w_int = w_int.type(torch.int8)
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
        intermediate_dtype = torch.float16
        num_bit_slice_weight = int(weight_bits/weight_bit_slice)
        ctx.num_bit_slice_weight = num_bit_slice_weight
        num_bit_slice_act = int(act_bits/act_bit_slice)
        ctx.num_bit_slice_act = num_bit_slice_act
        kernel_size = w_int.shape[2]
        out_channels = w_int.shape[0]
        fold_x =int( (x_int.shape[-1] - w_int.shape[-1] + 2*conv_padding[0])/conv_stride[0] + 1)
        
        Qp_adc = (2 ** (adc_bits-1)) - 1 
        Qn_adc = -1*(2 ** (adc_bits-1)) 
        if(adc_bits == 1):
            Qp_adc = 1
            Qn_adc = -1
        ctx.Qp_adc = Qp_adc
        ctx.Qn_adc = Qn_adc
        
            
        'making input activations'
        x_unf = nn.Unfold(kernel_size = kernel_size,padding = conv_padding, stride = conv_stride)(x_int).transpose(1,2)
        flatdim = x_unf.shape[-1]
        ctx.flatdim = flatdim
        ctx.arr = arr
        x_unf_sliced = slicing_act.apply(x_unf,act_bits, act_bit_slice).transpose(0,1).type(intermediate_dtype)
        x_unf_sliced = x_unf_sliced.unsqueeze(1).repeat(1,num_bit_slice_weight,1,1,1)
        ctx.x_unf_sliced = x_unf_sliced.type(torch.int8)
        
        'shape of x_unf_sliced = [batch_size, num_bit_slices_weight, num_bit_slices_act, flattened_dim_out ,flat_dim]'
        
        'making weight tensors'
        w_unf = w_int.view(w_int.shape[0],-1).t()
        if (mapping == 1):
            w_unf_sliced = slicing_weights.apply(w_unf, weight_bits, weight_bit_slice).type(intermediate_dtype)
            w_unf_sliced = w_unf_sliced.unsqueeze(1).repeat(1,num_bit_slice_act,1,1)
            ctx.w_unf_sliced = w_unf_sliced.type(torch.int8)
            
            'shape of w_unf_sliced = [num_bit_slices_weight num_bit_slice_act, flat_dim, out_channels]'
        
        num_crossbars = math.ceil(flatdim/arr)
        ctx.num_crossbars = num_crossbars
        
        out_unf = torch.cuda.FloatTensor(1).type(torch.float16).cuda()
        out_unf.resize_((x_unf_sliced.shape[0],num_crossbars,num_bit_slice_weight, num_bit_slice_act, fold_x*fold_x, out_channels))
        
        
        'matmul weights'
        
        for i in range (int(flatdim/arr)):
            temp_x = x_unf_sliced[:,:,:,:,i*arr:(i+1)*arr]
            temp_w = w_unf_sliced[:,:,i*arr:(i+1)*arr,:]
            out_unf[:,i,:,:,:,:] = torch.matmul(temp_x,temp_w)
        
        if (flatdim % arr) != 0 :
            temp_x = x_unf_sliced[:,:,:,:,(int(flatdim/arr))*arr:]
            temp_w = w_unf_sliced[:,:,(int(flatdim/arr))*arr:,:]
            out_unf[:,num_crossbars-1,:,:,:,:] = torch.matmul(temp_x,temp_w)
            
            'out_unf_positive shape: [batch_size, num crossbars, num_bit_slices_weight, num_bit_slices_act, flattened_dim_output, out_channels]'
            
        
        'ADC quantization:'
        adc_out = (out_unf - beta_cim)/alpha_cim
        ctx.mask_ge = adc_out.ge(Qp_adc + 1e-5)
        ctx.mask_le = adc_out.le(Qn_adc - 1e-5)
        ctx.num_el = adc_out.numel()
        
        #ctx.scaled_shifted_ps = adc_out

        adc_out = torch.sign(adc_out)
        ctx.ps_sign = adc_out.type(torch.int8)
        adc_out = torch.mul(adc_out, alpha_cim) + beta_cim
                
        
        'shift and add'
        out_sna = torch.sum(torch.mul(adc_out, binary_mask),dim = (1,2,3))
        
        out_sna = out_sna.type(torch.float32)
        
        
        return out_sna
        

        
        

    @staticmethod
    def backward(ctx, grad_output):
# =============================================================================
#         with profile(activities=[ProfilerActivity.CPU],
#             profile_memory=True, record_shapes=True,
#             on_trace_ready=torch.profiler.tensorboard_trace_handler('/home/nano01/a/saxenau/ADCLess/EfficientPyTorch/pytorchprofiler/')) as prof:
# =============================================================================
        x_unf_sliced = ctx.x_unf_sliced.type(torch.float32)
        'shape of x_unf_sliced = [batch_size, num_bit_slices_weight, num_bit_slices_act, flattened_dim_out ,flat_dim]'
        
        w_unf_sliced = ctx.w_unf_sliced.type(torch.float32)
        'shape of w_unf_sliced = [num_bit_slices_weight num_bit_slice_act, flat_dim, out_channels]'
    
        #ps = ctx.scaled_shifted_ps
        mask_ge = ctx.mask_ge
        mask_le = ctx.mask_le
        num_el = ctx.num_el
        ps_sign = ctx.ps_sign.type(torch.float32)
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
        alpha_cim = ctx.alpha_cim
        Qn_adc = ctx.Qn_adc
        Qp_adc = ctx.Qp_adc
        num_crossbars = ctx.num_crossbars
        binary_mask = ctx.binary_mask
        alpha_cim = ctx.alpha_cim
        flatdim = ctx.flatdim
        arr = ctx.arr
        
        grad_temp = grad_output.clone()
        'grad_output shape [batch_size, flattened_output_dim, out_channels]'
        
        grad_temp = grad_temp.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,num_crossbars, num_bit_slice_weight, num_bit_slice_act, 1,1)
        'grad temp shape [batch_size, num_crossbars, num_bit_slice_weight, num_bit_slice_act, flatenned_dim_output, out_channels]'
        
        
        #print(grad_temp.shape)
        'grad for Shift and Add'
        grad_temp = torch.mul(grad_temp, binary_mask)
        grad_output_after_adc = grad_temp.clone()
        #grad_scale = grad_temp.clone()
        'effect of ADC clamping'
        grad_temp[torch.logical_or(mask_ge, mask_le)] = 0
        
        #print(Qn_adc + 1e-5)
        'gradients for scale parameter alpha'
        grad_alpha = torch.mul(ps_sign, grad_output_after_adc)
        
        grad_alpha = torch.mul(grad_alpha,1.0 / math.sqrt(ps_sign.numel() * (Qp_adc)))
        grad_alpha = torch.sum(grad_alpha,dim = (0,4), keepdim= True)
        
        'gradients for shift parameter beta'
        grad_beta = torch.sum(grad_output_after_adc,dim = (0,4), keepdim = True)
        
        
        grad_input = torch.cuda.FloatTensor(1).resize_(x_unf_sliced.shape)
        grad_weight = torch.cuda.FloatTensor(1).resize_(x_unf_sliced.shape[0],w_unf_sliced.shape[0],w_unf_sliced.shape[1],w_unf_sliced.shape[2],w_unf_sliced.shape[3])
        for i in range (int(flatdim/arr)):
           
            grad_input[:,:,:,:,i*arr:(i+1)*arr] = torch.matmul(grad_temp[:,i,:,:,:,:], w_unf_sliced[:,:,i*arr:(i+1)*arr,:].transpose(2,3))
            grad_weight[:,:,:,i*arr:(i+1)*arr,:] = torch.matmul(x_unf_sliced.transpose(3,4)[:,:,:,i*arr:(i+1)*arr,:],grad_temp[:,i,:,:,:,:])
            
            #out_unf[:,i,:,:,:,:] = torch.matmul(temp_x,temp_w)
        
            'shape grad_temp = [batch, numCrossbar, num_w_bs, num_act_bs, flattened_out_dim, out_channels]'
            'shape grad_input = [batch,num_w_bs, num_act_bs, flattened_out_dim ,flat_dim ]'               
            'shape grad_weight = [batch, num_w_bs, num_act_bs,flat_dim, out_channels] '    
        if (flatdim % arr) != 0 :
            
            grad_input[:,:,:,:,(int(flatdim/arr))*arr:] = torch.matmul(grad_temp[:,num_crossbars-1,:,:,:,:], w_unf_sliced[:,:,(int(flatdim/arr))*arr:,:].transpose(2,3))
            grad_weight[:,:,:,(int(flatdim/arr))*arr:,:] = torch.matmul(x_unf_sliced[:,:,:,:,(int(flatdim/arr))*arr:].transpose(3,4),grad_temp[:,num_crossbars-1,:,:,:,:])
        
        'accumulate grad_weight across batch size'
        grad_weight = torch.sum(grad_weight,dim = 0)
        
        'accumulate across act bit slice and weight bit slice for grad_weight '
        grad_weight = torch.sum(grad_weight, dim = 1)
        for i in range(1,num_bit_slice_weight):
            grad_weight[i,:,:] = grad_weight[i,:,:] / (2**(bit_slice_w)) ** i
        
        grad_weight = torch.mean(grad_weight, dim = 0)
        
        'reshape grad_weight'
        grad_weight = grad_weight.transpose(0,1).view(out_channels,in_channels, kernel_size[0], kernel_size[0])
        
        'accumulate across act bit slice and weight bit slice for grad_input'
        grad_input = torch.sum(grad_input, dim = 1)
        for i in range(1,num_bit_slice_weight):
            grad_input[:,i,:,:] = grad_input[:,i,:,:] / (2**(bit_slice_a)) ** i
        
        grad_input = torch.mean(grad_input, dim = 1)
        
        'backward for unfold is nn.Fold'
        grad_temp_ = grad_input.clone().transpose(1,2)
        
        grad_input = torch.nn.Fold(output_size , kernel_size, dilation, padding,stride)(grad_temp_)
            
        
        
        return grad_input, grad_weight, None, None,  None, None , None, None, None, None, None, None, grad_alpha, grad_beta

class get_analog_partial_sums_autograd_ver2(Function):
    

    @staticmethod
    def forward(ctx, x_int, w_int, conv_stride, conv_padding, conv_dilation, act_bits, act_bit_slice, 
                        weight_bits, weight_bit_slice, adc_bits, arr,  binary_mask, alpha_cim, beta_cim):
        'Mapping : '
        '1 : positive and negative weights separate '
        '2 : twos complement mapping of weights'
        ctx.x_int = x_int.type(torch.int8)
        ctx.w_int = w_int.type(torch.int8)
        ctx.stride = conv_stride
        ctx.padding = conv_padding
        ctx.dilation = conv_dilation
        ctx.binary_mask = binary_mask
        ctx.alpha_cim = alpha_cim
        ctx.beta_cim = beta_cim
        ctx.bit_slice_w = weight_bit_slice
        ctx.bit_slice_a = act_bit_slice
        ctx.out_channels = w_int.shape[0]
        ctx.input_size = (x_int.shape[2], x_int.shape[3])        
        ctx.kernel_size = (w_int.shape[2], w_int.shape[3])
        ctx.in_channels = w_int.shape[1]
        mapping = 1
        intermediate_dtype = torch.float16
        num_bit_slice_weight = int(weight_bits/weight_bit_slice)
        ctx.num_bit_slice_weight = num_bit_slice_weight
        num_bit_slice_act = int(act_bits/act_bit_slice)
        ctx.num_bit_slice_act = num_bit_slice_act
        kernel_size = w_int.shape[2]
        out_channels = w_int.shape[0]
        fold_x =int( (x_int.shape[-1] - w_int.shape[-1] + 2*conv_padding[0])/conv_stride[0] + 1)
        
        Qp_adc = (2 ** (adc_bits-1)) - 1 
        Qn_adc = -1*(2 ** (adc_bits-1)) 
        if(adc_bits == 1):
            Qp_adc = 1
            Qn_adc = -1
        ctx.Qp_adc = Qp_adc
        ctx.Qn_adc = Qn_adc
        
            
        'making input activations'
        x_unf = nn.Unfold(kernel_size = kernel_size,padding = conv_padding, stride = conv_stride)(x_int).transpose(1,2)
        flatdim = x_unf.shape[-1]
        ctx.flatdim = flatdim
        ctx.arr = arr
        x_unf_sliced = slicing_act.apply(x_unf,act_bits, act_bit_slice).transpose(0,1).type(intermediate_dtype)
        x_unf_sliced = x_unf_sliced.unsqueeze(1).repeat(1,num_bit_slice_weight,1,1,1)
        ctx.x_unf_sliced = x_unf_sliced.type(torch.int8)
        
        'shape of x_unf_sliced = [batch_size, num_bit_slices_weight, num_bit_slices_act, flattened_dim_out ,flat_dim]'
        
        'making weight tensors'
        w_unf = w_int.view(w_int.shape[0],-1).t()
        if (mapping == 1):
            w_unf_sliced = slicing_weights.apply(w_unf, weight_bits, weight_bit_slice).type(intermediate_dtype)
            w_unf_sliced = w_unf_sliced.unsqueeze(1).repeat(1,num_bit_slice_act,1,1)
            ctx.w_unf_sliced = w_unf_sliced.type(torch.int8)
            
            'shape of w_unf_sliced = [num_bit_slices_weight num_bit_slice_act, flat_dim, out_channels]'
        
        num_crossbars = math.ceil(flatdim/arr)
        ctx.num_crossbars = num_crossbars
        
        out_unf = torch.cuda.FloatTensor(1).type(torch.int8).cuda()
        out_unf.resize_((x_unf_sliced.shape[0],num_crossbars,num_bit_slice_weight, num_bit_slice_act, fold_x*fold_x, out_channels))
        
        
        'matmul weights'
        
        for i in range (int(flatdim/arr)):
            temp_x = x_unf_sliced[:,:,:,:,i*arr:(i+1)*arr]
            temp_w = w_unf_sliced[:,:,i*arr:(i+1)*arr,:]
            out_unf[:,i,:,:,:,:] = torch.matmul(temp_x,temp_w)
        
        if (flatdim % arr) != 0 :
            temp_x = x_unf_sliced[:,:,:,:,(int(flatdim/arr))*arr:]
            temp_w = w_unf_sliced[:,:,(int(flatdim/arr))*arr:,:]
            out_unf[:,num_crossbars-1,:,:,:,:] = torch.matmul(temp_x,temp_w)
            
            'out_unf_positive shape: [batch_size, num crossbars, num_bit_slices_weight, num_bit_slices_act, flattened_dim_output, out_channels]'
            
        ctx.ps_int = out_unf.type(torch.int8)
        'ADC quantization:'
        adc_out = (out_unf - beta_cim)/alpha_cim
        #ctx.ps_int = adc_out.type(torch.float16)
        
        adc_out = torch.round(adc_out).clamp(Qn_adc, Qp_adc)
        adc_out = torch.mul(adc_out, alpha_cim) + beta_cim
                
        
        'shift and add'
        out_sna = torch.sum(torch.mul(adc_out, binary_mask),dim = (1,2,3))
        
        out_sna = out_sna.type(torch.float32)
        
        
        return out_sna
        

        
        

    @staticmethod
    def backward(ctx, grad_output):
        x_unf_sliced = ctx.x_unf_sliced.type(torch.float32)
        'shape of x_unf_sliced = [batch_size, num_bit_slices_weight, num_bit_slices_act, flattened_dim_out ,flat_dim]'
        
        w_unf_sliced = ctx.w_unf_sliced.type(torch.float32)
        'shape of w_unf_sliced = [num_bit_slices_weight num_bit_slice_act, flat_dim, out_channels]'
    
        ps_int = ctx.ps_int
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
        alpha_cim = ctx.alpha_cim
        beta_cim = ctx.beta_cim
        Qn_adc = ctx.Qn_adc
        Qp_adc = ctx.Qp_adc
        num_crossbars = ctx.num_crossbars
        binary_mask = ctx.binary_mask
        flatdim = ctx.flatdim
        arr = ctx.arr
        
        ps = (ps_int - beta_cim)/alpha_cim
        
        grad_temp = grad_output.clone()
        'grad_output shape [batch_size, flattened_output_dim, out_channels]'
        
        grad_temp = grad_temp.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,num_crossbars, num_bit_slice_weight, num_bit_slice_act, 1,1)
        'grad temp shape [batch_size, num_crossbars, num_bit_slice_weight, num_bit_slice_act, flatenned_dim_output, out_channels]'
        
        
        #print(grad_temp.shape)
        'grad for Shift and Add'
        grad_temp = torch.mul(grad_temp, binary_mask)
        grad_output_after_adc = grad_temp.clone()
        #grad_scale = grad_temp.clone()
        'effect of ADC clamping'
        grad_temp[torch.logical_or(ps.ge(Qp_adc+1e-5), ps.le(Qn_adc-1e-5))] = 0
        
        #print(Qn_adc + 1e-5)
        'gradients for scale parameter alpha'
        grad_alpha = torch.round(ps.clone()) - ps.clone()
        grad_alpha[ps.ge(Qp_adc+1e-5)] = Qp_adc 
        grad_alpha[ps.le(Qn_adc-1e-5)] = Qn_adc 
        #grad_alpha = torch.mul(grad_alpha,1.0 / math.sqrt(ps.numel() * (Qp_adc)))
        grad_alpha = torch.mul(grad_alpha, grad_output_after_adc)
        grad_alpha = torch.sum(grad_alpha,dim = (0,4), keepdim= True)
        
        'gradients for shift parameter beta'
        grad_beta = torch.cuda.FloatTensor(1).resize_(grad_output_after_adc.shape)
        grad_beta[:] = 0
        grad_beta[torch.logical_or(ps.ge(Qp_adc+1e-5), ps.le(Qn_adc-1e-5))] = 1.0
        grad_beta =torch.mul(grad_beta, grad_output_after_adc)
        
        grad_beta = torch.sum(grad_beta,dim = (0,4), keepdim = True)
        
        
        grad_input = torch.cuda.FloatTensor(1).resize_(x_unf_sliced.shape)
        grad_weight = torch.cuda.FloatTensor(1).resize_(x_unf_sliced.shape[0],w_unf_sliced.shape[0],w_unf_sliced.shape[1],w_unf_sliced.shape[2],w_unf_sliced.shape[3])
        for i in range (int(flatdim/arr)):
           
            grad_input[:,:,:,:,i*arr:(i+1)*arr] = torch.matmul(grad_temp[:,i,:,:,:,:], w_unf_sliced[:,:,i*arr:(i+1)*arr,:].transpose(2,3))
            grad_weight[:,:,:,i*arr:(i+1)*arr,:] = torch.matmul(x_unf_sliced.transpose(3,4)[:,:,:,i*arr:(i+1)*arr,:],grad_temp[:,i,:,:,:,:])
            
            'shape grad_temp = [batch, numCrossbar, num_w_bs, num_act_bs, flattened_out_dim, out_channels]'
            'shape grad_input = [batch,num_w_bs, num_act_bs, flattened_out_dim ,flat_dim ]'               
            'shape grad_weight = [batch, num_w_bs, num_act_bs,flat_dim, out_channels] '    
        if (flatdim % arr) != 0 :
            
            grad_input[:,:,:,:,(int(flatdim/arr))*arr:] = torch.matmul(grad_temp[:,num_crossbars-1,:,:,:,:], w_unf_sliced[:,:,(int(flatdim/arr))*arr:,:].transpose(2,3))
            grad_weight[:,:,:,(int(flatdim/arr))*arr:,:] = torch.matmul(x_unf_sliced[:,:,:,:,(int(flatdim/arr))*arr:].transpose(3,4),grad_temp[:,num_crossbars-1,:,:,:,:])
        
        'accumulate grad_weight across batch size'
        grad_weight = torch.sum(grad_weight,dim = 0)
        
        'accumulate across act bit slice and weight bit slice for grad_weight '
        grad_weight = torch.sum(grad_weight, dim = 1)
        for i in range(1,num_bit_slice_weight):
            grad_weight[i,:,:] = grad_weight[i,:,:] / (2**(bit_slice_w)) ** i
        
        grad_weight = torch.mean(grad_weight, dim = 0)
        
        'reshape grad_weight'
        grad_weight = grad_weight.transpose(0,1).view(out_channels,in_channels, kernel_size[0], kernel_size[0])
        
        'accumulate across act bit slice and weight bit slice for grad_input'
        grad_input = torch.sum(grad_input, dim = 1)
        for i in range(1,num_bit_slice_weight):
            grad_input[:,i,:,:] = grad_input[:,i,:,:] / (2**(bit_slice_a)) ** i
        
        grad_input = torch.mean(grad_input, dim = 1)
        
        'backward for unfold is nn.Fold'
        grad_temp_ = grad_input.clone().transpose(1,2)
        
        grad_input = torch.nn.Fold(output_size , kernel_size, dilation, padding,stride)(grad_temp_)
            
        
        
        return grad_input, grad_weight, None, None,  None, None , None, None, None, None, None, None, grad_alpha, grad_beta



def get_analog_partial_sums(x_int, w_int, conv_stride, conv_padding, conv_dilation, act_bits, act_bit_slice, 
                    weight_bits, weight_bit_slice,adc_bits, arr, binary_mask, alpha_cim, beta_cim):
    mapping = 1
    intermediate_dtype = torch.float16
    num_bit_slice_weight = int(weight_bits/weight_bit_slice)
    num_bit_slice_act = int(act_bits/act_bit_slice)
    kernel_size = w_int.shape[2]
    Qp_adc = (2 ** (adc_bits-1)) - 1
    Qn_adc = -1*(2 ** (adc_bits-1)) 
    if (adc_bits == 1):
        Qp_adc= 1
        Qn_adc= -1
        
    'making input activations'
    x_unf = nn.Unfold(kernel_size = kernel_size,padding = conv_padding, stride = conv_stride)(x_int).transpose(1,2)
    flatdim = x_unf.shape[-1]
    x_unf.register_hook(save_grad('post_slicing'))
    x_unf_sliced = slicing_act.apply(x_unf,act_bits, act_bit_slice).transpose(0,1).type(intermediate_dtype)
    #x_unf_sliced = x_unf
    x_unf_sliced = x_unf_sliced.unsqueeze(1).repeat(1,num_bit_slice_weight,1,1,1)
    x_unf_sliced.register_hook(save_grad('x_unf_sliced'))
    #del x_unf
    
    'shape of x_unf_sliced = [batch_size, num_bit_slices_weight, num_bit_slices_act, flattened_dim_out ,flat_dim]'
    
    'making weight tensors'
    w_unf = w_int.view(w_int.shape[0],-1).t()
    if (mapping == 1):
        w_unf_sliced = slicing_weights.apply(w_unf, weight_bits, weight_bit_slice).type(intermediate_dtype)
        w_unf_sliced = w_unf_sliced.unsqueeze(1).repeat(1,num_bit_slice_act,1,1)
        #del w_unf
        'shape of w_unf_sliced = [num_bit_slices_weight num_bit_slice_act, flat_dim, out_channels]'
    
    num_crossbars = math.ceil(flatdim/arr)
    out_channels = w_int.shape[0]
    fold_x =int( (x_int.shape[-1] - w_int.shape[-1] + 2*conv_padding[0])/conv_stride[0] + 1)
    
    out_unf = torch.cuda.FloatTensor(1).type(torch.float16).cuda()
    out_unf.resize_((x_unf_sliced.shape[0],num_crossbars,num_bit_slice_weight, num_bit_slice_act, fold_x*fold_x, out_channels))
    'matmul weights'
    
    for i in range (int(flatdim/arr)):
        temp_x = x_unf_sliced[:,:,:,:,i*arr:(i+1)*arr]
        temp_w = w_unf_sliced[:,:,i*arr:(i+1)*arr,:]
        out_unf[:,i,:,:,:,:] = torch.matmul(temp_x,temp_w)
    
   
                         
             
    if (flatdim % arr) != 0 :
        temp_x = x_unf_sliced[:,:,:,:,(int(flatdim/arr))*arr:]
        temp_w = w_unf_sliced[:,:,(int(flatdim/arr))*arr:,:]
        out_unf[:,num_crossbars-1,:,:,:,:] = torch.matmul(temp_x,temp_w)
        

        'out_unf_positive shape: [batch_size, num crossbars, num_bit_slices_weight, num_bit_slices_act, flattened_dim_output, out_channels]'
        
    g = 1.0 / math.sqrt(out_unf.numel() * Qp_adc)
    #print(g)
    #alpha_cim = grad_scale(alpha_cim, g)  
    temp = out_unf
    temp.register_hook(save_grad('grad_after_adc_clamping'))
    adc_out = (out_unf - beta_cim)/alpha_cim
    adc_out = round_pass(adc_out).clamp(Qn_adc, Qp_adc)
    adc_out = torch.mul(adc_out, alpha_cim) + beta_cim
    
    out_sna = torch.sum(torch.mul(adc_out, binary_mask),dim = (1,2,3))
    return out_sna

def get_analog_partial_sums_adcless(x_int, w_int, conv_stride, conv_padding, conv_dilation, act_bits, act_bit_slice, 
                    weight_bits, weight_bit_slice,adc_bits, arr, binary_mask, alpha_cim, beta_cim):
    mapping = 1
    intermediate_dtype = torch.float16
    num_bit_slice_weight = int(weight_bits/weight_bit_slice)
    num_bit_slice_act = int(act_bits/act_bit_slice)
    kernel_size = w_int.shape[2]
    Qp_adc = (2 ** (adc_bits-1)) - 1
    Qn_adc = -1*(2 ** (adc_bits-1)) 
    if (adc_bits == 1):
        Qp_adc= 1
        Qn_adc= -1
        
    'making input activations'
    x_unf = nn.Unfold(kernel_size = kernel_size,padding = conv_padding, stride = conv_stride)(x_int).transpose(1,2)
    flatdim = x_unf.shape[-1]
    x_unf.register_hook(save_grad('post_slicing'))
    x_unf_sliced = slicing_act.apply(x_unf,act_bits, act_bit_slice).transpose(0,1).type(intermediate_dtype)
    #x_unf_sliced = x_unf
    x_unf_sliced = x_unf_sliced.unsqueeze(1).repeat(1,num_bit_slice_weight,1,1,1)
    x_unf_sliced.register_hook(save_grad('x_unf_sliced'))
    #del x_unf
    
    'shape of x_unf_sliced = [batch_size, num_bit_slices_weight, num_bit_slices_act, flattened_dim_out ,flat_dim]'
    
    'making weight tensors'
    w_unf = w_int.view(w_int.shape[0],-1).t()
    if (mapping == 1):
        w_unf_sliced = slicing_weights.apply(w_unf, weight_bits, weight_bit_slice).type(intermediate_dtype)
        w_unf_sliced = w_unf_sliced.unsqueeze(1).repeat(1,num_bit_slice_act,1,1)
        #del w_unf
        'shape of w_unf_sliced = [num_bit_slices_weight num_bit_slice_act, flat_dim, out_channels]'
    
    num_crossbars = math.ceil(flatdim/arr)
    out_channels = w_int.shape[0]
    fold_x =int( (x_int.shape[-1] - w_int.shape[-1] + 2*conv_padding[0])/conv_stride[0] + 1)
    
    out_unf = torch.cuda.FloatTensor(1).type(torch.float16).cuda()
    out_unf.resize_((x_unf_sliced.shape[0],num_crossbars,num_bit_slice_weight, num_bit_slice_act, fold_x*fold_x, out_channels))
    'matmul weights'
    
    for i in range (int(flatdim/arr)):
        temp_x = x_unf_sliced[:,:,:,:,i*arr:(i+1)*arr]
        temp_w = w_unf_sliced[:,:,i*arr:(i+1)*arr,:]
        out_unf[:,i,:,:,:,:] = torch.matmul(temp_x,temp_w)
    
   
                         
             
    if (flatdim % arr) != 0 :
        temp_x = x_unf_sliced[:,:,:,:,(int(flatdim/arr))*arr:]
        temp_w = w_unf_sliced[:,:,(int(flatdim/arr))*arr:,:]
        out_unf[:,num_crossbars-1,:,:,:,:] = torch.matmul(temp_x,temp_w)
        

        'out_unf_positive shape: [batch_size, num crossbars, num_bit_slices_weight, num_bit_slices_act, flattened_dim_output, out_channels]'
        
    g = 1.0 / math.sqrt(out_unf.numel() * Qp_adc)
    #print(g)
    #alpha_cim = grad_scale(alpha_cim, g)  
    temp = out_unf
    temp.register_hook(save_grad('grad_after_adc_clamping'))
    adc_out = (out_unf - beta_cim)/alpha_cim
    adc_out = round_pass(adc_out).clamp(Qn_adc, Qp_adc)
    adc_out = torch.mul(adc_out, alpha_cim) + beta_cim
    
    out_sna = torch.sum(torch.mul(adc_out, binary_mask),dim = (1,2,3))
    return out_sna


grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

X = 32
in_channels =16
out_channels = 32
batch_size = 1
kernel_size = (3,3)
stride = (2,2)
padding = (1,1)
dilation = (1,1)
flattened_dim = in_channels * kernel_size[0] * kernel_size[1]

wprec = 3
actprec = 3
wbitslice = 1
actbitslice = 1
xbar = 64
adcprec =1
num_bit_slice_weight = int(wprec/wbitslice)
num_bit_slice_act = int(actprec / actbitslice)
num_xbars = int(math.ceil(flattened_dim/xbar))

binary_mask = torch.ones(num_bit_slice_weight, num_bit_slice_act)
for i in range(num_bit_slice_act):
    for j in range(num_bit_slice_weight):
        binary_mask[j,i] = ((2**actbitslice)**i)*((2**wbitslice)**j)
    
binary_mask =binary_mask.view(1,1,binary_mask.shape[0],binary_mask.shape[1],1,1).cuda()

alpha_cim = torch.randint(1,3,(1,num_xbars, num_bit_slice_weight, num_bit_slice_act, 1, out_channels)).type(torch.float32).cuda()
alpha_cim.requires_grad = True

beta_cim = torch.randint(1,4,(1,num_xbars, num_bit_slice_weight, num_bit_slice_act, 1, out_channels)).type(torch.float32).cuda()
beta_cim.requires_grad = True

layer_in = torch.randint(0,2**actprec - 1,(batch_size, in_channels, X, X)).type(torch.float32).cuda()
weight = torch.randint(-(2**(wprec-1)), (2**(wprec-1)) - 1, (out_channels, in_channels, kernel_size[0], kernel_size[1])).type(torch.float32).cuda()

weight.requires_grad = True
layer_in.requires_grad = True


out_auto_autograd = get_analog_partial_sums(layer_in, weight, stride, padding, dilation, actprec, actbitslice,
                                            wprec, wbitslice,adcprec, xbar, binary_mask, alpha_cim, beta_cim)
#print(alpha_cim.device)

temp_backward = torch.randn(out_auto_autograd.shape).type(torch.float32).cuda()
out_auto_autograd.backward(temp_backward)

input_grad_auto = layer_in.grad.clone()
#print(input_grad_auto)
weight_grad_auto = weight.grad.clone()
alpha_grad_auto = alpha_cim.grad.clone()
beta_grad_auto = beta_cim.grad.clone()


layer_in.grad = None
weight.grad = None
alpha_cim.grad = None
beta_cim.grad = None

out_manual_autograd = get_analog_partial_sums_autograd_ver2.apply(layer_in, weight, stride, padding, dilation, actprec, actbitslice,
                                            wprec, wbitslice,adcprec, xbar, binary_mask, alpha_cim, beta_cim)

out_manual_autograd.backward(temp_backward)

input_grad_manual = layer_in.grad.clone()
weight_grad_manual = weight.grad.clone()
alpha_grad_manual = alpha_cim.grad.clone()
beta_grad_manual = beta_cim.grad.clone()
print(torch.mean((out_manual_autograd == out_auto_autograd).type(torch.float32)))
print(torch.mean(((input_grad_manual - input_grad_auto).abs() < 0.05).type(torch.float32)))
print(torch.mean(((weight_grad_manual - weight_grad_auto).abs() < 0.05).type(torch.float32)))
print(torch.mean(((alpha_grad_manual - alpha_grad_auto).abs() < 0.0005).type(torch.float32)))
print(torch.mean(((beta_grad_manual - beta_grad_auto).abs() < 0.0005).type(torch.float32)))
#print(weight_grad_manual - weight_grad_auto)