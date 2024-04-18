import torch
import torch.nn as nn
from einops import rearrange

# kernel_sizes = [(128,1), (64,1), (32,1), (16,1)]
# padding_values = [63, 31, 15, 7]
# dilation_values = [1, 2, 4, 8, 16]

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1,33), stride=2, dilation_values=[1, 2, 4, 8, 16]):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for dilation in dilation_values:
            padding = (kernel_size[1] - 1) // 2 * dilation
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, 
                        out_channels, 
                        kernel_size, 
                        stride=(1,stride), 
                        padding=(0,padding), 
                        dilation=(1,dilation)
                    ),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels),
                    nn.Dropout2d(0.2)))

    def forward(self, x):
        conv_outputs = [conv_layer(x) for conv_layer in self.conv_layers]
        concatenated = torch.cat(conv_outputs, dim=1)
        return concatenated

class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, padding_values, stride=2, dilation=1):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for kernel_size, padding in zip(kernel_sizes, padding_values):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=(stride,1),
                        padding=(padding,0),
                        dilation=(dilation,1),
                    ),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels),
                    nn.Dropout2d(0.2)))
            
    def forward(self, x):
        conv_outputs = [conv_layer(x) for conv_layer in self.conv_layers]
        concatenated = torch.cat(conv_outputs, dim=1)
        return concatenated

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,3], strides=[2,1], padding_values=1, dilation=1, num_res=4):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for i in range(num_res):
            for kernel_size, stride in zip(kernel_sizes, strides):
                self.conv_layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding_values,
                            dilation=dilation,
                        ),
                        nn.ReLU(),
                        nn.BatchNorm2d(out_channels),
                        nn.Dropout2d(0.2)))
            
    def forward(self, x):
        for conv_layer in self.conv_layers:
            residual = x
            x = conv_layer(x)
            # print(x.shape)
            if residual.shape == x.shape:
                x += residual
        return x

# Example usage
# temporal_encoder = TemporalBlock(in_channels=1, out_channels=10, dilation_values=dilation_values)
# spatial_encoder = SpatialBlock(in_channels=50, out_channels=50, kernel_sizes=kernel_sizes, padding_values=padding_values)
# residual = ResidualBlock(in_channels=200, out_channels=200, kernel_sizes=[3,3], strides=[2,1])

# # Assuming your input tensor is of size (batch_size, channels, time_frames)
# input_data = torch.randn(1, 1, 128, 440) # shape (n, b, c, t)
# output = temporal_encoder(input_data)
# # print(output.size())  # Output size should be (50 x 128 x 220)

# output = spatial_encoder(output)
# output = residual(output)
# print(output.size())
