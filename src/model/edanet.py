import torch.nn as nn
from ..blocks.downsampling import DownsamplingBlock
from ..modules.eda_block import EDABlock

class EDANet(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()


        self.down1 = DownsamplingBlock(15, 60)
        self.down2 = DownsamplingBlock(60, 260)
        self.down3 = DownsamplingBlock(260, 130)

        self.eda1 = EDABlock(130, num_layers=5, k=40)
        self.eda2 = EDABlock(eda1.out_ch if hasattr(self, "eda1") else 130,
                             num_layers=8,
                             k=40,
                             dilations=[2,2,2,2,4,4,8,8])

        self.proj = nn.Conv2d(self.eda2.out_ch, num_classes, 1)

    def forward(self, x):
        x = self.down1(x)
      
        x = self.down2(x)
      
        x = self.eda1(x)
      
        x = self.down3(x)
      
        x = self.eda2(x)

        x = self.proj(x)

        x = nn.functional.interpolate(x, scale_factor=8, mode="bilinear", align_corners=False)

        return x
