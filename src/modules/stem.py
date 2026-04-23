import torch.nn as nn

class Stem(nn.Module):
    def __init__(self, in_ch=3, out_ch=15):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)
