import torch
import torch.nn as nn

class DownsamplingBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)

        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.use_pool = out_ch > in_ch

        if self.use_pool:
            self.proj = nn.Conv2d(in_ch, out_ch - in_ch, 1, bias=False)

    def forward(self, x):

        if self.use_pool:
            c = self.conv(x)
            p = self.proj(self.pool(x))
            x = torch.cat([c, p], dim=1)
        else:
            x = self.conv(x)

        return self.relu(self.bn(x))
