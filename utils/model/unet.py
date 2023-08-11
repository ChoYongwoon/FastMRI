import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class Unet(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.laplace = nn.Conv2d(in_chans, in_chans, kernel_size=
                                 3, bias = False)
        self.laplace.weight = nn.Parameter(self.get_laplace_kernel(in_chans), requires_grad=False)
        self.blur = nn.Conv2d(in_chans, in_chans, kernel_size=3, bias = False)
        
        self.gaussian = nn.Conv2d(in_chans, in_chans, kernel_size=3, bias = False)
        self.sobel_x = nn.Conv2d(in_chans, in_chans, kernel_size=3, bias=False)
        self.sobel_y = nn.Conv2d(in_chans, in_chans, kernel_size=3, bias=False)
        self.gaussian.weight = nn.Parameter(self.get_gaussian_kernel(channels=in_chans), requires_grad=False)
        self.sobel_x.weight = nn.Parameter(self.get_sobel_kernel_x(in_chans), requires_grad=False)
        self.sobel_y.weight = nn.Parameter(self.get_sobel_kernel_y(in_chans), requires_grad=False)
        self.attention = AttentionBlock(in_chans, in_chans, 4)
        self.attention5 = AttentionBlock5(in_chans, in_chans, in_chans, in_chans, in_chans, 4)

        self.first_block1 = ConvBlock(in_chans, 4)
        self.first_block2 = ConvBlock(in_chans, 4)
        self.first_block3 = ConvBlock(in_chans, 4)
        self.first_block4 = ConvBlock(in_chans, 4)
        self.first_block5 = ConvBlock(in_chans, 4)
        self.first_block6 = ConvBlock(in_chans, 4)
        self.down1 = Down(24,48)
        self.down2 = Down(48,96)
        self.down3 = Down(96,192)
        self.down4 = Down(192,384)
        
        self.mid = ConvBlock(384, 384)
        self.mid1 = ConvBlock(384, 384)
        
        self.up0 = Up(384,192)
        self.up1 = Up(192, 96)
        self.up2 = Up(96, 48)
        self.up3 = Up(48, 24)
        self.last = ConvBlock(24, 4)
        self.last_block = nn.Conv2d(4, out_chans, kernel_size=1)
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
        input4, mean4, std4 = self.norm(input[:,2])
        input4 = input4.unsqueeze(1)
        input_pad = F.pad(input2,  (2,2,2,2), mode = "replicate")
        input_gaussian = self.gaussian(input_pad)
        input_pad = F.pad(input_gaussian, (1,1,1,1), mode = "replicate")
        input3 = self.laplace(input_pad)
        input3, mean3, std3 = self.norm(input3[:,0])
        input3 = input3.unsqueeze(1)
        sobelx = self.sobel_x(input_pad)
        sobely = self.sobel_y(input_pad)
        sobel = self.attention(sobelx, sobely)
        input5 = self.attention5(input1, input2, input3, input4, sobel)
        
        d1 = self.first_block1(input1)
        d2 = self.first_block2(input2)
        d3 = self.first_block3(input3)      
        d4 = self.first_block4(sobel)
        d5 = self.first_block5(input4)
        d6 = self.first_block6(input5)
        
        d = torch.cat([d1,d2,d3,d4,d5,d6], dim=1)
        m0 = self.down1(d)
        m1 = self.down2(m0)
        m2 = self.down3(m1)
        m3 = self.down4(m2)
        
        m3 = self.mid(m3)
        m3 = self.mid(m3)
        
        u0 = self.up0(m3, m2)
        u1 = self.up1(u0, m1)
        u2 = self.up2(u1, m0)
        u3 = self.up3(u2, d)
        u3 = self.last(u3)
        
        output = self.last_block(u3)
        output = output.squeeze(1)
        output = self.unnorm(output, mean4, std4)

        return output

    def get_gaussian_kernel(self, kernel_size=5, sigma=1, channels=3):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.linspace(-sigma, sigma, kernel_size)
        x_grid, y_grid = torch.meshgrid(x_cord, x_cord)
    
        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = torch.exp(-(x_grid.square() + y_grid.square()) / (2 * sigma * sigma))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    
        return gaussian_kernel
    
    def get_laplace_kernel(self, in_chans):
        laplace_kernel = torch.tensor([[0,-1,0], [-1,4,-1], [0,-1,0]]).float().unsqueeze(0).unsqueeze(0)
        laplace_kernel = laplace_kernel.repeat(in_chans, 1, 1, 1)
        return laplace_kernel.repeat(in_chans, 1, 1, 1)
        
    def get_sobel_kernel_x(self, in_chans):
            sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0)
            sobel_kernel_x = sobel_kernel_x.repeat(in_chans, 1, 1, 1)
            return sobel_kernel_x

    def get_sobel_kernel_y(self, in_chans):
        sobel_kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).float().unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = sobel_kernel_y.repeat(in_chans, 1, 1, 1)
        return sobel_kernel_y
    
class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, padding_mode='replicate'),
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

class AttentionBlock5(nn.Module):
    def __init__(self, F_g, F_l, F_m, F_n, F_o, F_int):
        super(AttentionBlock5, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_y = nn.Sequential(
            nn.Conv2d(F_m, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_z = nn.Sequential(
            nn.Conv2d(F_n, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_w = nn.Sequential(
            nn.Conv2d(F_o, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )


        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x, y, z, w):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        y1 = self.W_y(y)
        z1 = self.W_z(z)
        w1 = self.W_w(w)
        
        psi = self.relu(g1 + x1 + y1 + z1 + w1)
        psi = self.psi(psi)
        return x * psi
    
 