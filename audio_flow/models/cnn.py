import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from einops import rearrange

class Cnn(nn.Module):
    def __init__(self, cnn_dim:int) -> None:
        super().__init__()
        
        self.pre_layer = nn.Conv2d(in_channels=cnn_dim, out_channels=32, kernel_size=5, padding=2) # step = 1 ->> same dimension in/output
        self.conv1 = ConvBlock(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.conv2 = ConvBlock(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.conv3 = ConvBlock(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.post_layer = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, padding=2)

        with torch.no_grad():
            dummy = torch.randn(1, cnn_dim, 64, 64) 
            x = self.forward(dummy)
            self.mlp_in_channels = x.shape[1]

    def forward(self, x: Tensor) -> Tensor:
        """Model.

        b: batch_size
        c: channels_num
        h: height
        w: weight

        Args:
            x: (b, c, h, w)

        Outputs:
            output: (b, c, h, w)
        """

        x = self.pre_layer(x)  # (b, c, h, w)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.post_layer(x)  # (b, c, h/8, w/8)
        x = rearrange(x, 'b c t f -> b (c t f)')  # (b, hw/64)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        r"""Conv block"""
        super(ConvBlock, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        b: batch_size
        c: channels_num
        h: height
        w: weight

        Args:
            x: (b, c, h, w)

        Returns:
            output: (b, c, h, w)
        """
        out = self.conv(F.leaky_relu_(self.bn(x)))
        return out