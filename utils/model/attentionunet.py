import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class AttentionUnet(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.laplace = nn.Conv2d(in_chans, in_chans, kernel_size=
                                 3, bias = False)
        self.laplace.weight = nn.Parameter(self.get_laplace_kernel(in_chans), requires_grad=False)

        self.first_block1 = ConvBlock(in_chans, 4)
        self.first_block2 = ConvBlock(in_chans, 4)
        self.first_block3 = ConvBlock(in_chans, 4)
        self.att_block1 = AttentionBlock(4, 4, 4)
        self.att_block2 = AttentionBlock(4, 4, 4)
        self.att_block3 = AttentionBlock(4, 4, 4)
        self.down1 = Down(12,24)
        self.down2 = Down(24,48)
        self.down3 = Down(48,96)
        self.down4 = Down(96,192)
        self.up0 = Up(192,96)
        self.up1 = Up(96, 48)
        self.up2 = Up(48, 24)
        self.up3 = Up(24, 12)
        self.last_block = nn.Conv2d(12, out_chans, kernel_size=1)
        self.drop = nn.Dropout2d(0.1)

    def norm(self, x):
        b, h, w = x.shape
        x = x.view(b, h * w)
        mean = x.mean(dim=1).view(b, 1, 1)
        std = x.std(dim=1).view(b, 1, 1)
        x = x.view(b, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, input):
        input1, mean1, std1 = self.norm(input[:,0])
        input1 = input1.unsqueeze(1)
        input2, mean2, std2 = self.norm(input[:,1])
        input2 = input2.unsqueeze(1)
        input_pad = F.pad(input2,  (1,1,1,1), mode = "replicate")
        input3 = self.laplace(input_pad)
        
        d1 = self.first_block1(input1)
        d2 = self.first_block2(input2)
        d3 = self.first_block3(input3)  
        
        a1 = self.att_block1(d1, d2)
        a2 = self.att_block2(d2, d3)
        a3 = self.att_block3(d1, d3)
        
        d = torch.cat([d1,d2,d3], dim=1)
        m0 = self.down1(d)
        m1 = self.down2(m0)
        m2 = self.down3(m1)
        m3 = self.down4(m2)
        u0 = self.up0(m3, m2)
        u1 = self.up1(u0, m1)
        u2 = self.up2(u1, m0)
        u3 = self.up3(u2, d)
        u4 = self.drop(u3)
        output = self.last_block(u4)
        output = output.squeeze(1)
        output = self.unnorm(output, mean2, std2)

        return output
    
    def get_laplace_kernel(self, in_chans):
        laplace_kernel = torch.tensor([[0,-1,0], [-1,4,-1], [0,-1,0]]).float().unsqueeze(0).unsqueeze(0)
        return laplace_kernel.repeat(in_chans, 1, 1, 1)
        
        

class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class Down(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_chans, out_chans)
        )

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.up = nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_chans, out_chans)

    def forward(self, x, concat_input):
        x = self.up(x)
        concat_output = torch.cat([concat_input, x], dim=1)
        return self.conv(concat_output)
    
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
