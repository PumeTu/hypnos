"""
Implementation of the ResNet
References:
- Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)
"""
from typing import Optional

import torch
import torch.nn as nn

configs = [18, 34, 50, 101, 152]

class BasicBlock(nn.Module):
    """
    Basic Residual Building block for ResNet 18, 34
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of reduced channels
        stride (int): stride length
    """
    expansion: int = 1
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.act2 = nn.ReLU()

        if stride != 1 or in_channels != self.expansion*out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )
        
    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        x = self.downsample(x) if self.downsample is not None else x
        out = out + x
        out = self.act2(out)
        return out
    
class BottleneckBlock(nn.Module):
    """
    Bottleneck block for ResNet 50, 101, 152
    """
    expansion: int = 4
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.act1 = nn.ReLU() 
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.act2 = nn.ReLU() 
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels*self.expansion)
        self.act3 = nn.ReLU()

        if stride != 1 or in_channels != self.expansion*out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            ) 

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        x = self.downsample(x) if self.downsample is not None else x
        out = out + x
        out = self.act3(x)
        return out

class ResNet(nn.Module):
    """ResNet reimplementation"""
    def __init__(self, depth: int, classification: bool, num_classes: Optional[int]):
        super().__init__()
        assert depth in configs, "Not a supported ResNet config"
        self.depth = depth 
        self.block = {
            18: BasicBlock,
            34: BasicBlock,
            50: BottleneckBlock,
            101: BottleneckBlock,
            152: BottleneckBlock
        }[depth]
        self.num_blocks = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 4],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]            
        }[depth]

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self._build_layers()
        self.layer2 = self._build_layers()
        self.layer3 = self._build_layers()
        self.layer4 = self._build_layers()
        self.fc = nn.Linear(512*self.block.expansion, num_classes) if classification else None

    def _build_layers(self, block, num_blocks, stride):
        pass
    
    def forward(self, x):
        return x
