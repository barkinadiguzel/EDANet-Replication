from .asymmetric_conv import AsymmetricConv
import torch.nn as nn

class DilatedAsymmetricConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, dilation=2):
        super().__init__()

        pad = dilation * (k // 2)

        self.conv1 = nn.Conv2d(in_ch, out_ch, (k, 1),
                               padding=(pad, 0),
                               dilation=(dilation, 1),
                               bias=False)

        self.conv2 = nn.Conv2d(out_ch, out_ch, (1, k),
                               padding=(0, pad),
                               dilation=(1, dilation),
                               bias=False)

        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.relu(self.bn(x))
