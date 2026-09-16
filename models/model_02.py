# description
# 使用SMP库提供的UNet网络结构，编码器为ResNet18

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class Model02(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=out_channels,
            activation=None,
        )

    def forward(self, x):
        return self.unet(x)
