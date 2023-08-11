# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 22:20:50 2023

@author: pc
"""
import h5py
import matplotlib.pyplot as plt
from torch import nn
import torch
import numpy as np
from skimage import feature, filters

from utils.model.attentionunet import AttentionUnet
from utils.model.unet import Unet


def get_laplace_kernel(in_chans):
    laplace_kernel = torch.tensor([[0,-1,0], [-1,4,-1], [0,-1,0]]).float().unsqueeze(0).unsqueeze(0)
    return laplace_kernel.repeat(in_chans, 1, 1, 1)
    
def norm(x):
    mean = x.mean()
    std = x.std()
    return (x - mean) / std, mean, std

def get_gaussian_kernel(kernel_size=5, sigma=1, channels=1):
# Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.linspace(-sigma, sigma, kernel_size)
    x_grid, y_grid = torch.meshgrid(x_cord, x_cord)
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = torch.exp(-(x_grid.square() + y_grid.square()) / (2 * sigma * sigma))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, padding=int(kernel_size/2))

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

def get_laplace_kernel(in_chans):
    laplace_kernel = torch.tensor([[0,-1,0], [-1,4,-1], [0,-1,0]]).float().unsqueeze(0).unsqueeze(0)
    laplace_kernel = laplace_kernel.repeat(in_chans, 1, 1, 1)
    laplace_kernel = get_gaussian_kernel(kernel_size=5, sigma=1, channels=in_chans)(laplace_kernel)        
    return laplace_kernel.repeat(in_chans, 1, 1, 1)
    
def get_sobel_kernel_x(in_chans):
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0)
        sobel_kernel_x = sobel_kernel_x.repeat(in_chans, 1, 1, 1)
        sobel_kernel_x = get_gaussian_kernel(kernel_size=5, sigma=1, channels=in_chans)(sobel_kernel_x)
        return sobel_kernel_x

def get_sobel_kernel_y(in_chans):
    sobel_kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).float().unsqueeze(0).unsqueeze(0)
    sobel_kernel_y = sobel_kernel_y.repeat(in_chans, 1, 1, 1)
    sobel_kernel_y = get_gaussian_kernel(kernel_size=5, sigma=1, channels=in_chans)(sobel_kernel_y)
    return sobel_kernel_y

model1 = Unet(1, 1)
model2 = AttentionUnet(1, 1)
model3 = torch.load('C:/Fastmri/home/0.9760_withReconAttention_20epoch/test_Unet/checkpoints/best_model.pt')

sum1 = sum(p.numel() for p in model1.parameters())
sum2 = sum(p.numel() for p in model2.parameters())

print(sum1)
print(sum2)