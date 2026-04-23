import torch
import torch.nn as nn
from .pointwise_conv import PointWiseConv
from .asymmetric_conv import AsymmetricConv
from .dilated_asymmetric_conv import DilatedAsymmetricConv

class EDAModule(nn.Module):
    def __init__(self, in_ch, k=40, dilation=1, use_dilation=False):
        super().__init__()

        self.reduce = PointWiseConv(in_ch, k)

        self.asym1 = AsymmetricConv(k, k)

        if use_dilation:
            self.asym2 = DilatedAsymmetricConv(k, k, dilation=dilation)
        else:
            self.asym2 = AsymmetricConv(k, k)

        self.bn = nn.BatchNorm2d(k)
        self.relu = nn.ReLU(inplace=True)

        self.out_ch = in_ch + k  

    def forward(self, x):
        identity = x

        x = self.reduce(x)
        x = self.asym1(x)
        x = self.asym2(x)

        x = torch.cat([identity, x], dim=1)  
        return self.relu(self.bn(x))
