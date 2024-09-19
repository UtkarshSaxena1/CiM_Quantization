#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 14:36:43 2022

@author: saxenau
"""

#import lsq
import torch
import sys
sys.path.insert(0, '/home/nano01/a/saxenau/ADCLess/EfficientPyTorch/')
import models._modules as my_nn
from torch.autograd import Function
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


error =0
for i in range(1000):
    X = 7
    in_channels =16
    out_channels = 1
    batch_size = 1
    kernel_size = 7
    stride = (1,1)
    padding = (0,0)
    
    
    wprec = 3
    actprec = 3
    wbitslice = 1
    actbitslice = 1
    xbar = 64
    adcprec = 16
    
    
    layer_in = torch.randint(0,2**actprec - 1,(batch_size, in_channels, X, X)).type(torch.float32).cuda()
    weight = torch.randint(-(2**(wprec-1)), (2**(wprec-1)) - 1, (out_channels, in_channels, kernel_size, kernel_size)).cuda()
    
    
    Conv = torch.nn.Conv2d(in_channels, out_channels, (kernel_size,kernel_size),stride, padding,  bias = None).cuda()
    Conv.weight.data.copy_(weight)
    #print(layer_in)
    
    #out = torch.nn.functional.conv2d(layer_in, weight, None, stride, padding)
    #print(out)
    #print(layer_in.type())
    out_conv = Conv(layer_in.type(torch.float32))
    #out_conv = out_conv.view(batch_size, out_channels, -1)
    
    Conv_cim = my_nn.Conv2dLSQCiM(in_channels, out_channels, (kernel_size,kernel_size), stride, padding, 1,1,False, wprec, actprec, wbitslice, actbitslice, xbar, adcprec).cuda()
    Conv_cim.weight.data.copy_(weight)
    #print(layer_in.device)
    out_cim = Conv_cim(layer_in)
    temp_error = torch.sum(torch.abs(out_cim - out_conv))
    error = error + temp_error
    if(temp_error!=0):
        print(out_cim)
        print(out_conv)
        print('--')
        #break
    #print(torch.sum(torch.abs(out_cim - out_conv)))
print(error)

binary_mask = torch.ones(1,1,1,1).cuda()

class slicing_weights_mapping1_ver2(Function):
    """
    Class to slice the integer tensor
    """

    @staticmethod
    def forward(ctx, w_int, bits,bit_slice,binary_mask):
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
        #print(binary_mask.shape)        
        ctx.tensor = tensor
        tensor = tensor.unsqueeze(1).repeat(1,int(bits/bit_slice),1,1)
        tensor = torch.floor(tensor.div(binary_mask))
        #print(tensor.shape)
        tensor = torch.remainder(tensor, 2**bit_slice)
        tensor = tensor[0] - tensor[1]
        #tensor_sliced = slicing_unsigned.apply(tensor, bits, bit_slice)
        ctx.bits = bits
        ctx.bit_slice = bit_slice
        
        ctx.binary_mask = binary_mask
        
        'tensor_sliced : [num_bit_slice, **kwargs]'
        ''
        return tensor

        
        

    @staticmethod
    def backward(ctx, grad_output):
        bits = ctx.bits
        bit_slice = ctx.bit_slice
        tensor = ctx.tensor
        binary_mask = ctx.binary_mask
        
# =============================================================================
#         binary_mask = []
#         for i in range(int(bits/bit_slice)):
#             binary_mask.append(torch.tensor((2**bit_slice)**i))
#         binary_mask = torch.stack(binary_mask).to(grad_output.device)
#         binary_mask = binary_mask.view(1,binary_mask.shape[0],1,1)
# =============================================================================
        
        grad_input = grad_output.clone()
        grad_input = torch.div(grad_input, binary_mask).mean(0,keepdim=False)
        #grad_input[0,tensor[0].le(0)] = 0
        #grad_input[1,tensor[1].ge(0)] = 0
        
        #grad_input = grad_input[0] - grad_input[1]
        
        
        return grad_input, None, None
w_unf = weight.view(weight.shape[0],-1).t()

binary_mask_weight = torch.ones(int(wprec/wbitslice))
for i in range(int(wprec/wbitslice)):
    binary_mask_weight[i] = (2**wbitslice)**i
binary_mask_weight = binary_mask_weight.view(1,binary_mask_weight.shape[0],1,1).cuda()


sliced = slicing_weights_mapping1_ver2.apply(w_unf, wprec, wbitslice, binary_mask_weight)

