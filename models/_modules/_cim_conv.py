#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 13:09:46 2021

@author: saxenau
"""

import torch.nn as nn
import torch.nn.functional as F
from .quant_utils import *
import torch
import math
from torch.autograd import Function


    


"ADC Quantization strategy:"
"Find the STD of pre ADC value. "



def CiMArray_conv2d(x_int, w_int, b_int, conv_stride, conv_padding, conv_dilation, act_bits, act_bit_slice, 
                    weight_bits, weight_bit_slice, adc_bits, arr):
    'Mapping : '
    '1 : positive and negative weights separate '
    '2 : twos complement mapping of weights'
    mapping = 1
    intermediate_dtype = torch.float32
    num_bit_slice_weight = int(weight_bits/weight_bit_slice)
    num_bit_slice_act = int(act_bits/act_bit_slice)
    
    in_channels = w_int.shape[1]
    kernel_size = w_int.shape[2]
    out_channels = w_int.shape[0]
    fold_x =int( (x_int.shape[-1] - w_int.shape[-1] + 2*conv_padding[0])/conv_stride[0] + 1)
    
    'making input activations'
    x_unf = nn.Unfold(kernel_size = kernel_size,padding = conv_padding, stride = conv_stride)(x_int).transpose(1,2)
    x_unf_sliced = slicing_unsigned_act.apply(x_unf,act_bits, act_bit_slice)
    
    'making weight tensors'
    w_unf = w_int.view(w_int.shape[0],-1).t()
    if (mapping == 1):
        w_unf_sliced_positive, w_unf_sliced_negative = slicing_weights_mapping1.apply(w_unf, weight_bits, weight_bit_slice)
        w_unf_sliced_positive = w_unf_sliced_positive.transpose(0,1).transpose(1,2)
        w_unf_sliced_negative = w_unf_sliced_negative.transpose(0,1).transpose(1,2)
        'shape of w_unf_sliced = [flat_dim, out_channels, num_bit_slices]'
    
    flatdim = x_unf.shape[-1]
    
    'matmul for positive weights'
    out_unf_positive = []
    for i in range (int(flatdim/arr)):
        temp_out_w = []
        for j in range(num_bit_slice_weight):
            temp_out_a = []
            for k in range(num_bit_slice_act):
                temp_x = x_unf_sliced[k,:,:,i*arr:(i+1)*arr].type(intermediate_dtype)
                temp_w = w_unf_sliced_positive[i*arr:(i+1)*arr,:,j].type(intermediate_dtype)
                temp_out_a.append(temp_x.matmul(temp_w).transpose(1,2).type(intermediate_dtype))
            temp_out_a = torch.stack(temp_out_a)
            temp_out_w.append(temp_out_a)
        temp_out_w = torch.stack(temp_out_w)
        out_unf_positive.append(temp_out_w)
    
    if (flatdim % arr) != 0 :
        temp_out_w = []
        for j in range(num_bit_slice_weight):
            temp_out_a = []
            for k in range(num_bit_slice_act):
                temp_x = x_unf_sliced[k,:,:,(int(flatdim/arr))*arr:].type(intermediate_dtype)
                temp_w = w_unf_sliced_positive[(int(flatdim/arr))*arr:,:,j].type(intermediate_dtype)
                temp_out_a.append(temp_x.matmul(temp_w).transpose(1,2).type(intermediate_dtype))
            temp_out_a = torch.stack(temp_out_a)
            temp_out_w.append(temp_out_a)
        
        temp_out_w = torch.stack(temp_out_w)
        out_unf_positive.append(temp_out_w)
    
    out_unf_positive = torch.stack(out_unf_positive)
    'out_unf_positive shape: [num crossbars, num_bit_slices_weight, num_bit_slices_act, batch_size, out_channels, flattened_dim_output]'
    
    
    
    with torch.no_grad():
          adc_scaling_factor_positive = out_unf_positive.max(dim = 5).values
          adc_scaling_factor_positive = adc_scaling_factor_positive.max(dim = 3).values
          #adc_scaling_factor_positive = torch.round(torch.std(out_unf_positive,dim = (3,5)))
          if (adc_bits == 1):
              n = 1
          else:
              n = 2 ** (adc_bits) - 1
          #adc_scaling_factor_positive[adc_scaling_factor_positive.le(n)] = n
      
          #adc_scaling_factor_positive = torch.clamp(adc_scaling_factor_positive, min = 1e-8) /n
          
          adc_scaling_factor_positive = torch.unsqueeze(torch.unsqueeze(adc_scaling_factor_positive, dim = 4),dim = 3)
    
    
    
    
    adc_out_positive = ADCQuantFunction4.apply(out_unf_positive, adc_bits, adc_scaling_factor_positive)
    
    
    
    'matmul for negative weights'
    out_unf_negative = []
    for i in range (int(flatdim/arr)):
        temp_out_w = []
        for j in range(num_bit_slice_weight):
            temp_out_a = []
            for k in range(num_bit_slice_act):
                temp_x = x_unf_sliced[k,:,:,i*arr:(i+1)*arr].type(intermediate_dtype)
                temp_w = w_unf_sliced_negative[i*arr:(i+1)*arr,:,j].type(intermediate_dtype)
                temp_out_a.append(temp_x.matmul(temp_w).transpose(1,2).type(intermediate_dtype))
            temp_out_a = torch.stack(temp_out_a)
            temp_out_w.append(temp_out_a)
        temp_out_w = torch.stack(temp_out_w)
        out_unf_negative.append(temp_out_w)
    
    if (flatdim % arr) != 0 :
        temp_out_w = []
        for j in range(num_bit_slice_weight):
            temp_out_a = []
            for k in range(num_bit_slice_act):
                temp_x = x_unf_sliced[k,:,:,(int(flatdim/arr))*arr:].type(intermediate_dtype)
                
                temp_w = w_unf_sliced_negative[(int(flatdim/arr))*arr:,:,j].type(intermediate_dtype)
                
                temp_out_a.append(temp_x.matmul(temp_w).transpose(1,2).type(intermediate_dtype))
            temp_out_a = torch.stack(temp_out_a)
            temp_out_w.append(temp_out_a)
        temp_out_w = torch.stack(temp_out_w)
        out_unf_negative.append(temp_out_w)
    
    out_unf_negative = torch.stack(out_unf_negative)
    'out_unf_negative shape: [num crossbars, num_bit_slices_weight, num_bit_slices_act, batch_size, out_channels, flattened_dim_output]'
    
    with torch.no_grad():
          adc_scaling_factor_negative = out_unf_negative.max(dim = 5).values
          adc_scaling_factor_negative = adc_scaling_factor_negative.max(dim = 3).values
          #adc_scaling_factor_negative = torch.round(torch.std(out_unf_negative,dim = (3,5)))
          if (adc_bits == 1):
              n = 1
          else:
              n = 2 ** (adc_bits) - 1
          #adc_scaling_factor_negative[adc_scaling_factor_negative.le(n)] = n
      
          #adc_scaling_factor_negative = torch.clamp(adc_scaling_factor_negative, min = 1e-8) / n
          
          adc_scaling_factor_negative = torch.unsqueeze(torch.unsqueeze(adc_scaling_factor_negative,dim=4),dim=3)
   
         
    adc_out_negative = ADCQuantFunction4.apply(out_unf_negative, adc_bits, adc_scaling_factor_negative)
    
    
    
    
    
    binary_mask_act = torch.ones((num_bit_slice_act)).cuda()
    for i in range(num_bit_slice_act):
        binary_mask_act[i] = (2**act_bit_slice)**i
    binary_mask_act = binary_mask_act.view(1,1,binary_mask_act.shape[0],1,1,1)
    
    
    binary_mask_w = torch.ones((num_bit_slice_weight)).cuda()
    for i in range(num_bit_slice_weight):
        binary_mask_w[i] = (2**weight_bit_slice)**i
    binary_mask_w = binary_mask_w.view(1,binary_mask_w.shape[0],1,1,1,1)
    
    out_positive = torch.sum(torch.mul(torch.mul(adc_out_positive, binary_mask_act), binary_mask_w),dim = (0,1,2))
    out_positive = out_positive.view(x_int.shape[0],out_channels,fold_x,fold_x)
    
    out_negative = torch.sum(torch.mul(torch.mul(adc_out_negative, binary_mask_act), binary_mask_w),dim = (0,1,2))
    out_negative = out_negative.view(x_int.shape[0],out_channels,fold_x,fold_x)
    
    out = out_positive - out_negative
    
    b_int_reshaped = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(b_int,1),2),0)
    if b_int is not None:
        out = torch.add(out,b_int_reshaped)

    return out
        






class ADCQuantFunction4(Function):
    """
    Class to quantize the given floating-point values using symmetric quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale=None):
        """
        x: floating point tensor to be quantized
        k: quantization bitwidth
        Note that the current implementation of SymmetricQuantFunction requires pre-calculated scaling factor.
        specified_scale: pre-calculated scaling factor for the tensor x
        """
        # n  = number of steps
        # [0,scale] = range of quantization
        n = 2 ** (k) - 1
        
        if specified_scale is not None:
            scale = specified_scale.clone()
        else:
            raise ValueError("The SymmetricQuantFunction requires a pre-calculated scaling factor")
        scale = torch.clamp(scale, min = torch.round(torch.tensor(n/16)))
        step_size = scale/n
        new_quant_x = torch.round(x/step_size)
        ctx.quant = new_quant_x
        new_quant_x = torch.clamp(new_quant_x , 0 , n)
        new_quant_x = step_size * new_quant_x
        
        
        ctx.n = n
        
       
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):
        quant_input = ctx.quant
        n = ctx.n
        grad_input = grad_output.clone()
        grad_input[quant_input.le(0)] = 0
        grad_input[quant_input.ge(n)] = 0
        
        return grad_input, None, None, None




class slicing_unsigned_act(Function):
    """
    Class to slice the integer tensor
    """

    @staticmethod
    def forward(ctx, x_int, bits,bit_slice):
        """
        x_int: unsigned int to be sliced
        bits: number of bits
        bit_slice : bit width of bit_slice
        """
        tensor = x_int
        sliced_tensor = []
        for i in range(int(bits/bit_slice)):
            sliced_tensor.append(torch.remainder(tensor,2**bit_slice))
            tensor = torch.floor(torch.div(tensor, 2**bit_slice))
        sliced_tensor = torch.stack(sliced_tensor)
        ctx.bits = bits
        ctx.bit_slice = bit_slice
        
        'LSB = sliced_tensor[0], MSB = sliced_tensor[bits/bit_slice]'
        return sliced_tensor

        
        

    @staticmethod
    def backward(ctx, grad_output):
        bits = ctx.bits
        bit_slice = ctx.bit_slice
        
        binary_mask = []
        for i in range(int(bits/bit_slice),0, -1):
            binary_mask.append(torch.tensor((2**bit_slice)**i))
        binary_mask = torch.stack(binary_mask).to(grad_output.device)
        for i in range(len(grad_output.shape)-1):
            binary_mask = torch.unsqueeze(binary_mask,dim =1)
            
        grad_input = grad_output.clone()
        grad_input = torch.div(grad_input, binary_mask)
        grad_input = torch.mean(grad_input, dim=0)
        return grad_input, None, None

    


         


class slicing_unsigned(Function):
    """
    Class to slice the integer tensor
    """

    @staticmethod
    def forward(ctx, x_int, bits,bit_slice):
        """
        x_int: unsigned int to be sliced
        bits: number of bits
        bit_slice : bit width of bit_slice
        """
        tensor = x_int
        sliced_tensor = []
        for i in range(int(bits/bit_slice)):
            sliced_tensor.append(torch.remainder(tensor,2**bit_slice))
            tensor = torch.floor(torch.div(tensor, 2**bit_slice))
        sliced_tensor = torch.stack(sliced_tensor)
        ctx.bits = bits
        ctx.bit_slice = bit_slice
        'LSB = sliced_tensor[0], MSB = sliced_tensor[bits/bit_slice]'
        return sliced_tensor

        
        

    @staticmethod
    def backward(ctx, grad_output):
        bits = ctx.bits
        bit_slice = ctx.bit_slice
        binary_mask = []
        for i in range(int(bits/bit_slice)):
            binary_mask.append(torch.tensor((2**bit_slice)**i))
        binary_mask = torch.stack(binary_mask).to(grad_output.device)
        for i in range(len(grad_output.shape)-1):
            binary_mask = torch.unsqueeze(binary_mask,dim =1)
        
        
            
        grad_input = grad_output.clone()
        
        grad_input = torch.div(grad_input, binary_mask)
        grad_input = torch.mean(grad_input,dim = 0)
        
        return grad_input, None, None
    
class slicing_weights_mapping1(Function):
    """
    Class to slice the integer tensor
    """

    @staticmethod
    def forward(ctx, w_int, bits,bit_slice):
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
        
        tensor_positive_sliced = slicing_unsigned.apply(tensor_positive, bits, bit_slice)
        tensor_negative_sliced = slicing_unsigned.apply(tensor_negative, bits, bit_slice)
        
        ctx.bits = bits
        ctx.bit_slice = bit_slice
        ctx.tensor_positive = tensor_positive
        ctx.tensor_negative = tensor_negative
        
        
        ''
        return tensor_positive_sliced, tensor_negative_sliced

        
        

    @staticmethod
    def backward(ctx, grad_output_positive, grad_output_negative):
        bits = ctx.bits
        bit_slice = ctx.bit_slice
        tensor_positive = ctx.tensor_positive
        tensor_negative = ctx.tensor_negative
        
        binary_mask = []
        for i in range(int(bits/bit_slice)):
            binary_mask.append(torch.tensor((2**bit_slice)**i))
        binary_mask = torch.stack(binary_mask).to(grad_output_positive.device)
        for i in range(len(grad_output_positive.shape)-1):
            binary_mask = torch.unsqueeze(binary_mask,dim =1)
        
        grad_input_positive = grad_output_positive.clone()
        grad_input_positive = torch.div(grad_input_positive, binary_mask).sum(0)
        grad_input_positive[tensor_positive.le(0)] = 0
        grad_input_negative = grad_output_negative.clone()
        grad_input_negative = torch.div(grad_input_negative, binary_mask).sum(0)
        grad_input_negative[tensor_negative.ge(0)] = 0   
        
        grad_input = grad_input_positive - grad_input_negative
        
        
        return grad_input, None, None


class ADCQuantFunction(Function):
    """
    Class to quantize the given floating-point values using symmetric quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale=None):
        """
        x: floating point tensor to be quantized
        k: quantization bitwidth
        Note that the current implementation of SymmetricQuantFunction requires pre-calculated scaling factor.
        specified_scale: pre-calculated scaling factor for the tensor x
        """
        n = 2 ** (k - 1) - 1

        if specified_scale is not None:
            scale = specified_scale
        else:
            raise ValueError("The SymmetricQuantFunction requires a pre-calculated scaling factor")

        zero_point = torch.tensor(0.).cuda()
        new_quant_x = torch.round(1. / scale * x + zero_point)
        #ctx.quant = new_quant_x
        ctx.n = n
        new_quant_x = torch.clamp(new_quant_x, -n - 1, n)
        
        ctx.scale = scale
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):
        #quant_input = ctx.quant
        quant_scale = ctx.n
        scale = ctx.scale
        grad_input = grad_output.clone()
        #grad_input[quant_input.le(-1.0*quant_scale)]
        #grad_input[quant_input.ge(quant_scale)] = 0
        
        return grad_input / scale, None, None, None



































    
class ADCQuantFunction3(Function):
    """
    Class to quantize the given floating-point values using symmetric quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, alpha):
        """
        x: floating point tensor to be quantized
        k: quantization bitwidth
        Note that the current implementation of SymmetricQuantFunction requires pre-calculated scaling factor.
        specified_scale: pre-calculated scaling factor for the tensor x
        """
        n = 2 ** (k) - 1
        alpha = alpha.expand_as(x)
        
        'clamping x in the range [0,alpha]'
        quant_x = x.clone()
        quant_x_greater = (x > alpha).float()
        quant_x_lesser = (x < alpha).float()
        quant_x = torch.add(torch.mul(quant_x, quant_x_lesser), torch.mul(alpha, quant_x_greater))
        
         
        scale = n / alpha
        quant_x = torch.round(scale * quant_x)
        
        quant_x = torch.div(quant_x, scale)
        ctx.alpha = alpha 
        ctx.x = x
        
        return quant_x

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.x
        alpha = ctx.alpha
        alpha = alpha.expand_as(grad_output)
        
        grad_input = grad_output.clone()
        grad_scale = grad_output.clone()
        #grad_input[quant_input.le(-1.0*quant_scale)]
        'grad_inputs outside range are zero'
        grad_input = torch.mul(grad_input , (grad_input < alpha).float())
        
        'grad for scale are zeros within the range'
        grad_scale = torch.mul(grad_scale , (grad_scale > alpha).float())
        
        grad_scale = torch.sum(grad_scale,dim=(1,2,3,5),keepdim=True)
        
        return grad_input, None, 100*grad_scale    
    
    
    
        
    
    
    
    
    
    
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    








class ADCQuantFunction2(Function):
    """
    Class to quantize the given floating-point values using symmetric quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale=None):
        """
        x: floating point tensor to be quantized
        k: quantization bitwidth
        Note that the current implementation of SymmetricQuantFunction requires pre-calculated scaling factor.
        specified_scale: pre-calculated scaling factor for the tensor x
        """
        n = 2 ** (k) - 1

        if specified_scale is not None:
            scale = specified_scale
        else:
            raise ValueError("The SymmetricQuantFunction requires a pre-calculated scaling factor")

        zero_point = torch.tensor(0.).cuda()
        new_quant_x = torch.round(1. / scale * x + zero_point)
        ctx.quant = new_quant_x
        ctx.n = n
        new_quant_x = torch.clamp(new_quant_x, 0, n-1)
        
        ctx.scale = scale
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):
        quant_input = ctx.quant
        quant_scale = ctx.n
        scale = ctx.scale
        grad_input = grad_output.clone()
        #grad_input[quant_input.le(-1.0*quant_scale)]
        grad_input[quant_input.ge(quant_scale)] = 0
        
        return grad_input / scale, None, None, None






    
    



'4b best accuracy : 70.9% STE estimator'
'3b best accuracy : 54.81% STE estimator'



# =============================================================================
# def CiMArray_conv2d(x_int, w_int, b_int, conv_stride, conv_padding, conv_dilation, act_bits, act_bit_slice, weight_bits, weight_bit_slice, adc_bits, arr):
#     
#     
#     'Mapping : '
#     '1 : positive and negative weights separate '
#     '2 : twos complement mapping of weights'
#     mapping = 1
#     
#     
#         
#     in_channels = w_int.shape[1]
#     kernel_size = w_int.shape[2]
#     out_channels = w_int.shape[0]
#     fold_x =int( (x_int.shape[-1] - w_int.shape[-1] + 2*conv_padding[0])/conv_stride[0] + 1)
#     
#     'making input activations'
#     x_unf = nn.Unfold(kernel_size = kernel_size,padding = conv_padding, stride = conv_stride)(x_int).transpose(1,2)
#     x_unf_sliced = slicing_unsigned.apply(x_unf,act_bits, act_bit_slice)
#     
#     'making weight tensors'
#     w_unf = w_int.view(w_int.shape[0],-1).t()
#     
#     if (mapping == 1):
#         w_unf_sliced_positive, w_unf_sliced_negative = slicing_weights_mapping1.apply(w_unf, weight_bits, weight_bit_slice)
#         w_unf_sliced_positive = w_unf_sliced_positive.transpose(0,1).transpose(1,2)
#         w_unf_sliced_negative = w_unf_sliced_negative.transpose(0,1).transpose(1,2)
#         'shape of w_unf_sliced = [flat_dim, out_channels, num_bit_slices]'
#     
#     flatdim = x_unf.shape[-1]
#     
#     'matmul for positive weights'
#     out_unf_positive = []
#     for i in range (int(flatdim/arr)):
#         temp_out = []
#         for j in range(int(weight_bits/weight_bit_slice)):
#             temp_x = x_unf_sliced[:,:,:,i*arr:(i+1)*arr]
#             temp_w = w_unf_sliced_positive[i*arr:(i+1)*arr,:,j]
#             temp_out.append(temp_x.matmul(temp_w).transpose(2,3))
#         temp_out = torch.stack(temp_out)
#         out_unf_positive.append(temp_out)
#     
#     if (flatdim % arr) != 0 :
#         temp_out = []
#         for j in range(int(weight_bits/weight_bit_slice)):
#             temp_x = x_unf_sliced[:,:,:,(int(flatdim/arr))*arr:]
#             temp_w = w_unf_sliced_positive[(int(flatdim/arr))*arr:,:,j]
#             temp_out.append(temp_x.matmul(temp_w).transpose(2,3))
#         temp_out = torch.stack(temp_out)
#         out_unf_positive.append(temp_out)
#     
#     out_unf_positive = torch.stack(out_unf_positive)
#     'out_unf_positive shape: [num crossbars, num_bit_slices_weight, num_bit_slices_act, batch_size, out_channels, flattened_dim_output]'
#     with torch.no_grad():
#         pre_adc_min_positive = out_unf_positive.min(dim = 5).values
#         pre_adc_max_positive = out_unf_positive.max(dim = 5).values
#         
#         adc_scaling_factor_positive = torch.max(torch.stack([pre_adc_min_positive.abs(), pre_adc_max_positive.abs()],dim = 0),dim = 0).values  
#         #adc_scaling_factor_positive = torch.var(out_unf_positive,dim = 5) 
#         
#         if (adc_bits == 1):
#             n = 1
#         else:
#             n = 2 ** (adc_bits - 1) - 1
#         adc_scaling_factor_positive[adc_scaling_factor_positive.le(n)] = n
#     
#         adc_scaling_factor_positive = torch.clamp(adc_scaling_factor_positive, min = 1e-8) /n
#         
#         adc_scaling_factor_positive = torch.unsqueeze(adc_scaling_factor_positive, dim = 5)
#         
#     adc_out_positive = ADCQuantFunction.apply(out_unf_positive, adc_bits, adc_scaling_factor_positive)
#     adc_out_positive = torch.mul(adc_out_positive, adc_scaling_factor_positive)
#     
#     'matmul for negative weights'
#     out_unf_negative = []
#     for i in range (int(flatdim/arr)):
#         temp_out = []
#         for j in range(int(weight_bits/weight_bit_slice)):
#             temp_x = x_unf_sliced[:,:,:,i*arr:(i+1)*arr]
#             temp_w = w_unf_sliced_negative[i*arr:(i+1)*arr,:,j]
#             temp_out.append(temp_x.matmul(temp_w).transpose(2,3))
#         temp_out = torch.stack(temp_out)
#         out_unf_negative.append(temp_out)
#     
#     if (flatdim % arr) != 0 :
#         temp_out = []
#         for j in range(int(weight_bits/weight_bit_slice)):
#             temp_x = x_unf_sliced[:,:,:,(int(flatdim/arr))*arr:]
#             temp_w = w_unf_sliced_negative[(int(flatdim/arr))*arr:,:,j]
#             temp_out.append(temp_x.matmul(temp_w).transpose(2,3))
#         temp_out = torch.stack(temp_out)
#         out_unf_negative.append(temp_out)
#     
#     out_unf_negative = torch.stack(out_unf_negative)
#     'out_unf_negative shape: [num crossbars, num_bit_slices_weight, num_bit_slices_act, batch_size, out_channels, flattened_dim_output]'
#     with torch.no_grad():
#         pre_adc_min_negative = out_unf_negative.min(dim = 5).values
#         pre_adc_max_negative = out_unf_negative.max(dim = 5).values
#         
#         adc_scaling_factor_negative = torch.max(torch.stack([pre_adc_min_negative.abs(), pre_adc_max_negative.abs()],dim = 0),dim = 0).values  
#         #adc_scaling_factor_negative = torch.var(out_unf_negative,dim = 5) 
#     
#         if (adc_bits == 1):
#             n = 1
#         else:
#             n = 2 ** (adc_bits - 1) - 1
#         adc_scaling_factor_negative[adc_scaling_factor_negative.le(n)] = n
#     
#         adc_scaling_factor_negative = torch.clamp(adc_scaling_factor_negative, min = 1e-8) /n
#         
#         adc_scaling_factor_negative = torch.unsqueeze(adc_scaling_factor_negative, dim = 5)
#         
#     adc_out_negative = ADCQuantFunction2.apply(out_unf_negative, adc_bits, adc_scaling_factor_negative)
#     adc_out_negative = torch.mul(adc_out_negative, adc_scaling_factor_negative)
#     
#     
#     
#     binary_mask_act = torch.ones((int(act_bits/act_bit_slice))).cuda()
#     for i in range(int(act_bits/act_bit_slice)):
#         binary_mask_act[i] = (2**act_bit_slice)**i
#     binary_mask_act = binary_mask_act.view(1,1,binary_mask_act.shape[0],1,1,1)
#     
#     
#     binary_mask_w = torch.ones((int(weight_bits/weight_bit_slice))).cuda()
#     for i in range(int(weight_bits/weight_bit_slice)):
#         binary_mask_w[i] = (2**weight_bit_slice)**i
#     binary_mask_w = binary_mask_w.view(1,binary_mask_w.shape[0],1,1,1,1)
#     
#     
#     
#     out_positive = torch.sum(torch.mul(torch.mul(adc_out_positive, binary_mask_act), binary_mask_w),dim = (0,1,2))
#     out_positive = out_positive.view(x_int.shape[0],out_channels,fold_x,fold_x)
#     
#     out_negative = torch.sum(torch.mul(torch.mul(adc_out_negative, binary_mask_act), binary_mask_w),dim = (0,1,2))
#     out_negative = out_negative.view(x_int.shape[0],out_channels,fold_x,fold_x)
#     
#     out = out_positive - out_negative
#     
#     b_int_reshaped = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(b_int,1),2),0)
#     if b_int is not None:
#         out = torch.add(out,b_int_reshaped)
# 
#     return out
# =============================================================================



# =============================================================================
# def CiMArray_conv2d(x_int, w_int, b_int, conv_stride, conv_padding, conv_dilation, arr):
#     
#     act_bits = 8
#     act_bit_slice = 8
#     adc_bits = 4
#     weight_bits = 8
#     weight_bit_slice = 8
#     'Mapping : '
#     '1 : positive and negative weights separate '
#     '2 : twos complement mapping of weights'
#     mapping = 1
#     
#     
#         
#     in_channels = w_int.shape[1]
#     kernel_size = w_int.shape[2]
#     out_channels = w_int.shape[0]
#     fold_x =int( (x_int.shape[-1] - w_int.shape[-1] + 2*conv_padding[0])/conv_stride[0] + 1)
#     
#     'making input activations'
#     x_unf = nn.Unfold(kernel_size = kernel_size,padding = conv_padding, stride = conv_stride)(x_int).transpose(1,2)
#     x_unf_sliced = slicing_unsigned.apply(x_unf,act_bits, act_bit_slice)
#     
#     'making weight tensors'
#     w_unf = w_int.view(w_int.shape[0],-1).t()
#     
#     
#     
#     flatdim = x_unf.shape[-1]
#     
#     'matmul'
#     out_unf = []
#     for i in range (int(flatdim/arr)):
#         temp_x = x_unf_sliced[:,:,:,i*arr:(i+1)*arr]
#         temp_w = w_unf[i*arr:(i+1)*arr,:]
#         temp_out = (temp_x.matmul(temp_w).transpose(2,3))
#         
#         out_unf.append(temp_out)
#     
#     if (flatdim % arr) != 0 :
#         temp_x = x_unf_sliced[:,:,:,(int(flatdim/arr))*arr:]
#         temp_w = w_unf[(int(flatdim/arr))*arr:,:]
#         temp_out = (temp_x.matmul(temp_w).transpose(2,3))
#         
#         out_unf.append(temp_out)
#     
#     out_unf = torch.stack(out_unf)
#     'out_unf_positive shape: [num crossbars, num_bit_slices_act, batch_size, out_channels, flattened_dim_output]'
#     with torch.no_grad():
#         pre_adc_min = out_unf.min(dim = 4).values
#         pre_adc_max = out_unf.max(dim = 4).values
#         
#         adc_scaling_factor = torch.max(torch.stack([pre_adc_min.abs(), pre_adc_max.abs()],dim = 0),dim = 0).values  
#         if (adc_bits == 1):
#             n = 1
#         else:
#             n = 2 ** (adc_bits - 1) - 1
#         
#         adc_scaling_factor = torch.clamp(adc_scaling_factor, min = 1e-8) /n
#         
#         adc_scaling_factor = torch.unsqueeze(adc_scaling_factor, dim = 4)
#     
#     adc_out = ADCQuantFunction.apply(out_unf, adc_bits, adc_scaling_factor)
#     adc_out = torch.mul(adc_out, adc_scaling_factor)
#     
#     
#     
#     binary_mask_act = torch.ones((int(act_bits/act_bit_slice))).cuda()
#     for i in range(int(act_bits/act_bit_slice)):
#         binary_mask_act[i] = (2**act_bit_slice)**i
#     binary_mask_act = binary_mask_act.view(1,binary_mask_act.shape[0],1,1,1)
#     
#     
#     
#     
#     
#     out = torch.sum(torch.mul(adc_out, binary_mask_act),dim = (0,1))
#     out = out.view(x_int.shape[0],out_channels,fold_x,fold_x)
#     
#     b_int_reshaped = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(b_int,1),2),0)
#     if b_int is not None:
#         out = torch.add(out,b_int_reshaped)
# 
#     return out
# =============================================================================


# =============================================================================
# class CiMArray_conv2d(nn.Module):
#     
#     def __init__(self,
#                  act_bits,
#                  act_bit_slice,
#                  weight_bits,
#                  weight_bit_slice,
#                  adc_bits,
#                  arr,
#                  in_channels,
#                  kernel_size,
#                  out_channels):
#         super(CiMArray_conv2d, self).__init__()
# 
#         self.act_bits = act_bits
#         self.act_bit_slice = act_bit_slice
#         self.weight_bits = weight_bits
#         self.weight_bit_slice = weight_bit_slice
#         self.adc_bits = adc_bits
#         self.arr=arr
#         self.flatdim=in_channels*kernel_size*kernel_size
#         self.arr=arr
#         self.num_crossbars = int(self.flatdim/self.arr)
#         self.num_bit_slice_act = act_bits/act_bit_slice
#         self.num_bit_slice_weight = weight_bits/weight_bit_slice
#         self.scaling_factor = nn.Parameter(torch.Tensor((2,self.num_crossbars,self.num_bit_slice_weight,self.num_bit_slice_act,1,out_channels,1)), requires_grad = True)
#         
# 
#     
#     def forward(self, x_int,w_int,b_int,conv_stride,conv_padding,conv_dilation):
#         'Mapping : '
#         '1 : positive and negative weights separate '
#         '2 : twos complement mapping of weights'
#         mapping = 1
#         
#         in_channels = w_int.shape[1]
#         kernel_size = w_int.shape[2]
#         out_channels = w_int.shape[0]
#         fold_x =int( (x_int.shape[-1] - w_int.shape[-1] + 2*conv_padding[0])/conv_stride[0] + 1)
#         
#         'making input activations'
#         x_unf = nn.Unfold(kernel_size = kernel_size,padding = conv_padding, stride = conv_stride)(x_int).transpose(1,2)
#         x_unf_sliced = slicing_unsigned.apply(x_unf,self.act_bits, self.act_bit_slice)
#         
#         'making weight tensors'
#         w_unf = w_int.view(w_int.shape[0],-1).t()
#         if (mapping == 1):
#             w_unf_sliced_positive, w_unf_sliced_negative = slicing_weights_mapping1.apply(w_unf, self.weight_bits, self.weight_bit_slice)
#             w_unf_sliced_positive = w_unf_sliced_positive.transpose(0,1).transpose(1,2)
#             w_unf_sliced_negative = w_unf_sliced_negative.transpose(0,1).transpose(1,2)
#             'shape of w_unf_sliced = [flat_dim, out_channels, num_bit_slices]'
#         
#         flatdim = x_unf.shape[-1]
#         
#         'matmul for positive weights'
#         out_unf_positive = []
#         for i in range (int(flatdim/arr)):
#             temp_out = []
#             for j in range(self.num_bit_slice_weight):
#                 temp_x = x_unf_sliced[:,:,:,i*arr:(i+1)*arr]
#                 temp_w = w_unf_sliced_positive[i*arr:(i+1)*arr,:,j]
#                 temp_out.append(temp_x.matmul(temp_w).transpose(2,3))
#             temp_out = torch.stack(temp_out)
#             out_unf_positive.append(temp_out)
#         
#         if (flatdim % arr) != 0 :
#             temp_out = []
#             for j in range(self.num_bit_slice_weight):
#                 temp_x = x_unf_sliced[:,:,:,(int(flatdim/arr))*arr:]
#                 temp_w = w_unf_sliced_positive[(int(flatdim/arr))*arr:,:,j]
#                 temp_out.append(temp_x.matmul(temp_w).transpose(2,3))
#             temp_out = torch.stack(temp_out)
#             out_unf_positive.append(temp_out)
#         
#         out_unf_positive = torch.stack(out_unf_positive)
#         'out_unf_positive shape: [num crossbars, num_bit_slices_weight, num_bit_slices_act, batch_size, out_channels, flattened_dim_output]'
#         
#         if (adc_bits == 1):
#             n = 1
#         else:
#             n = 2 ** (adc_bits - 1) - 1   
#         
#         adc_out_positive = ADCQuantFunction3.apply(out_unf_positive, adc_bits, self.scaling_factor[0]/n)
#         adc_out_positive = torch.mul(adc_out_positive, adc_scaling_factor_positive)
#         
#         'matmul for negative weights'
#         out_unf_negative = []
#         for i in range (self.num_bit_slice_weight):
#             temp_out = []
#             for j in range():
#                 temp_x = x_unf_sliced[:,:,:,i*arr:(i+1)*arr]
#                 temp_w = w_unf_sliced_negative[i*arr:(i+1)*arr,:,j]
#                 temp_out.append(temp_x.matmul(temp_w).transpose(2,3))
#             temp_out = torch.stack(temp_out)
#             out_unf_negative.append(temp_out)
#         
#         if (flatdim % arr) != 0 :
#             temp_out = []
#             for j in range(self.num_bit_slice_weight):
#                 temp_x = x_unf_sliced[:,:,:,(int(flatdim/arr))*arr:]
#                 temp_w = w_unf_sliced_negative[(int(flatdim/arr))*arr:,:,j]
#                 temp_out.append(temp_x.matmul(temp_w).transpose(2,3))
#             temp_out = torch.stack(temp_out)
#             out_unf_negative.append(temp_out)
#         
#         out_unf_negative = torch.stack(out_unf_negative)
#         'out_unf_negative shape: [num crossbars, num_bit_slices_weight, num_bit_slices_act, batch_size, out_channels, flattened_dim_output]'
#         
#         adc_out_negative = ADCQuantFunction3.apply(out_unf_negative, adc_bits, self.scaling_factor[1]/n)
#         adc_out_negative = torch.mul(adc_out_negative, adc_scaling_factor_negative)
#         
#         binary_mask_act = torch.ones((self.num_bit_slice_act)).cuda()
#         for i in range(self.num_bit_slice_act):
#             binary_mask_act[i] = (2**act_bit_slice)**i
#         binary_mask_act = binary_mask_act.view(1,1,binary_mask_act.shape[0],1,1,1)
#         
#         
#         binary_mask_w = torch.ones((self.num_bit_slice_weight)).cuda()
#         for i in range(self.num_bit_slice_weight):
#             binary_mask_w[i] = (2**weight_bit_slice)**i
#         binary_mask_w = binary_mask_w.view(1,binary_mask_w.shape[0],1,1,1,1)
#         
#         out_positive = torch.sum(torch.mul(torch.mul(adc_out_positive, binary_mask_act), binary_mask_w),dim = (0,1,2))
#         out_positive = out_positive.view(x_int.shape[0],out_channels,fold_x,fold_x)
#         
#         out_negative = torch.sum(torch.mul(torch.mul(adc_out_negative, binary_mask_act), binary_mask_w),dim = (0,1,2))
#         out_negative = out_negative.view(x_int.shape[0],out_channels,fold_x,fold_x)
#         
#         out = out_positive - out_negative
#         
#         b_int_reshaped = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(b_int,1),2),0)
#         if b_int is not None:
#             out = torch.add(out,b_int_reshaped)
#     
#         return out
# =============================================================================
