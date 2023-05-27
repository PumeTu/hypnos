"""
Implementation of the ResNet
References:
- Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)
"""
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """
    Basic Residual Building block for ResNet 18, 34
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int, downsample: bool):
        super().__init__()
        self.conv1 = nn.Conv2d()
        self.bn1 = nn.BatchNorm2d()
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d()
        self.bn2 = nn.BatchNorm2d()
        self.act2 = nn.ReLU()

        self.downsample = downsample

    def forward(self, x):
        return x
    
class BottleneckBlock(nn.Module):
    """
    Bottleneck block for ResNet 50, 101, 152
    """
    expansion: int = 4
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return x

class ResNet(nn.Module):
    """ResNet reimplementation"""
    def __init__(self, ):
        super().__init__()

    def _make_layers(self):
        pass
    
    def forward(self, x):
        return x
