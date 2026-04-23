import torch
import torch.nn as nn

# Fig3(a) - Asymmetric convolution pair
class AsymmetricConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()

        pad = k // 2

        self.conv1 = nn.Conv2d(in_ch, out_ch, (k, 1), padding=(pad, 0), bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, (1, k), padding=(0, pad), bias=False)

        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.relu(self.bn(x))
