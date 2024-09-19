#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 12:13:02 2022

@author: saxenau
"""

import torch
import sys
sys.path.insert(0, '/home/nano01/a/saxenau/ADCLess/EfficientPyTorch/')
import models._modules as my_nn
from torch.autograd import Function
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time


error =0
xdim = 2048
flatdim = 4096
ydim = 1024
arr = 1024
clamp_x = 16

wprec = 3
actprec = 3
wbitslice = 1
actbitslice = 1
xbar = 64
adcprec = 16
num_bs_w = int(wprec/wbitslice)
num_bs_a = int(actprec/actbitslice)

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook




layer_in = torch.randint(0,2**actprec - 1,(num_bs_a, xdim,flatdim)).type(torch.float32).cuda()
layer_in = layer_in.unsqueeze(0).repeat(num_bs_w,1,1,1)
weight = torch.randint(-(2**(wprec-1)), (2**(wprec-1)) - 1, (num_bs_w, flatdim , ydim)).type(torch.float32).cuda()
weight = weight.unsqueeze(1).repeat(1,num_bs_a,1,1)

weight.requires_grad = True
layer_in.requires_grad = True

out_temp = torch.tensor(1).resize_(int(flatdim/arr),num_bs_w, num_bs_a, xdim,ydim).type(torch.float32).cuda()
#out_temp.requires_grad = True
#out_temp.register_hook(save_grad('out_temp'))

for i in range(int(flatdim/arr)):
    
    out_temp[i,:,:,:,:] = torch.matmul(layer_in[:,:,:,i*arr:(i+1)*arr],weight[:,:,i*arr:(i+1)*arr,:])

temp = out_temp 
temp.register_hook(save_grad('after_clamp'))   
out_temp_clamped = torch.clamp(out_temp, -1*clamp_x, clamp_x)
out_temp_clamped.register_hook(save_grad('out_temp_clamped'))
#out_temp_clamped = out_temp_clamped.clamp(-1*clamp_x, clamp_x)

out_f = torch.sum(out_temp_clamped, dim = 0)

grad = torch.ones(out_f.shape).cuda()

out_f.backward(grad)

grad_temp = grad.clone()
grad_temp =  grad_temp.unsqueeze(0).repeat(int(flatdim/arr),1, 1,1,1)
grad_temp[torch.logical_or(out_temp.ge(clamp_x) , out_temp.le(-1*clamp_x))] = 0
print(grad_temp.shape)
grad_temp_sum = torch.mean(grad_temp.clone(), dim = 0)

print(grad_temp_sum.shape)
print(weight.shape)
ingrad = torch.matmul(grad_temp_sum, weight.transpose(2,3))
wgrad = torch.matmul(layer_in.transpose(2,3), grad_temp_sum)

print((ingrad == layer_in.grad).type(torch.float16).mean())

#print(wgrad == weight.grad)

layer_in = torch.randint(0,2**actprec - 1,(num_bs_a, xdim,flatdim)).type(torch.float32).cuda()
layer_in = layer_in.unsqueeze(0).repeat(num_bs_w,1,1,1)
weight = torch.randint(-(2**(wprec-1)), (2**(wprec-1)) - 1, (num_bs_w, flatdim , ydim)).type(torch.float32).cuda()
weight = weight.unsqueeze(1).repeat(1,num_bs_a,1,1)

time_i = time.time()
for i in range(int(flatdim/arr)):
    
    out_temp[i,:,:,:,:] = torch.matmul(layer_in[:,:,:,i*arr:(i+1)*arr],weight[:,:,i*arr:(i+1)*arr,:])

time_f = time.time()
print(time_f - time_i)

time_i = time.time()
for i in range(int(flatdim/arr)):
    for j in range(num_bs_w):
        for k in range (num_bs_a): 
            
            out_temp[i,j,k,:,:] = torch.matmul(layer_in[j,k,:,i*arr:(i+1)*arr],weight[j,k,i*arr:(i+1)*arr,:])

time_f = time.time()
print(time_f - time_i)


# =============================================================================
# =============================================================================
# # a = torch.randn(1024,2048,4096)
# # time_i = time.time()
# # b = a.ge(0)
# # time_f = time.time()
# # print('ge',time_f - time_i)
# # 
# # a = torch.randn(1024,2048,4096)
# # time_i = time.time()
# # b = a>0
# # time_f = time.time()
# # print('>',time_f - time_i)
# # 
# # a = torch.randn(1024,2048,4096)
# # time_i = time.time()
# # b = torch.gt(a,0)
# # time_f = time.time()
# # print('gt',time_f - time_i)
# =============================================================================
# =============================================================================


# =============================================================================
# 
# a = torch.randn(1024,2048,4096)
# b = torch.randn(4096)
# time_i = time.time()
# c=a*b
# time_f = time.time()
# print('gt',time_f - time_i)
# 
# a = torch.randn(1024,2048,4096)
# b = torch.randn(1,1,4096)
# time_i = time.time()
# c=a*b
# time_f = time.time()
# print('gl',time_f - time_i)
# =============================================================================
