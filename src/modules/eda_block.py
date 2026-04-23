import torch.nn as nn
from ..blocks.eda_module import EDAModule

class EDABlock(nn.Module):
    def __init__(self, in_ch, num_layers, k=40, dilations=None):
        super().__init__()

        layers = []
        ch = in_ch

        for i in range(num_layers):
            use_dilation = dilations is not None and i < len(dilations)
            d = dilations[i] if use_dilation else 1

            module = EDAModule(ch, k=k, dilation=d, use_dilation=use_dilation)
            layers.append(module)
            ch = module.out_ch

        self.block = nn.Sequential(*layers)
        self.out_ch = ch

    def forward(self, x):
        return self.block(x)
