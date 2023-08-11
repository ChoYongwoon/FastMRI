# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 23:16:12 2023

@author: pc
"""
from torchvision.transforms import GaussianBlur
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Augmenter(nn.Module):
    def __init__(self):
        super(Augmenter, self).__init__()

    def forward(self, image, label, seed):
        # Intensity scaling
        torch.manual_seed(430)
        intensity_factor = 1 + torch.rand(1)
        intensity_factor = intensity_factor.to(image.device)
        augmented_image = image.clone() * intensity_factor
        augmented_label = label.clone() * intensity_factor
            
        '''
        # Noise
        noise_prob = torch.rand(1) * 0.04
        noise_prob = noise_prob.to(augmented_image.device)
        noise = torch.randn_like(augmented_image) * noise_prob
        augmented_image += noise

        # Dropout
        
        dropout_prob = torch.rand(1) * 0.04
        dropout_prob = dropout_prob.to(augmented_image.device)
        drop_mask = torch.rand_like(augmented_image) < dropout_prob
        augmented_image = augmented_image * drop_mask.float()

        # Blur
        if torch.rand(1) < 0.1:
            blur = GaussianBlur(3, 0.6)
            augmented_image = blur(augmented_image)
        '''
        combined_image = torch.cat((image, augmented_image))
        combined_label = torch.cat((label, augmented_label))
        
        return combined_image, combined_label