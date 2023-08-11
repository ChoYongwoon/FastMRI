# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:25:15 2023

@author: pc
"""

from loss_function import SSIMLoss

class MaskedSSIMLoss(SSIMLoss):
    def __init__(self, mask, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = mask
        
        
    def forward(self, X, Y, data_range):
        X_masked = X * self.mask
        Y_masked = Y * self.mask
        return super().forward(X_masked, Y_masked, data_range)