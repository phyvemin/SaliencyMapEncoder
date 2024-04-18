# from torchvision import models
from blocks import *
import torch
import torch.nn as nn

class eeg_encoder(nn.Module):
    def __init__(self, embedding_size=1000, input_width=512, input_height=128, temp_in_channels=1, temp_out_channels=10, temp_kernel_size=(1,33), temp_stride=2, temp_dilation_values=[1, 2, 4, 8, 16],
                space_in_channels=50, space_out_channels=50, space_kernel_sizes=[(128,1), (64,1), (32,1), (16,1)], space_padding_values=[63, 31, 15, 7], space_strides=2, space_dilation_values=1,
                res_in_channels=200, res_out_channels=200, res_kernel_sizes=[3,3], res_strides=[2,1], res_padding_values=1, res_dilation=1, num_res=4):
        super().__init__()

        self.temporal_block = TemporalBlock(temp_in_channels, temp_out_channels, temp_kernel_size, temp_stride, temp_dilation_values)
        self.spatial_block = SpatialBlock(space_in_channels, space_out_channels, space_kernel_sizes, space_padding_values, space_strides, space_dilation_values)
        self.residual_block = ResidualBlock(res_in_channels, res_out_channels, res_kernel_sizes, res_strides, res_padding_values, res_dilation, num_res)

        self.encoder = nn.Sequential(
            self.temporal_block,
            self.spatial_block,
            self.residual_block,
            nn.Conv2d(res_out_channels, res_out_channels, 3),
            nn.ReLU(),
            nn.BatchNorm2d(res_out_channels),
            nn.Dropout2d(0.2),
            nn.Flatten())

        out_size = self.encoder(torch.zeros(1, 1, input_height, input_width)).shape[1]
        # print(out_size)
        self.linear = nn.Sequential(
            nn.Linear(out_size, out_size//2),
            nn.ReLU(),
            nn.Linear(out_size//2, embedding_size)
        )
    
    def forward(self, x):
        out = self.encoder(x)
        out = self.linear(out)
        return out

# eeg = eeg_encoder()
# print(eeg(torch.zeros(1, 10, 128, 512)).shape)
