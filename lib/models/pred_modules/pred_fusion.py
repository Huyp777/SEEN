import torch
from torch import nn
from core.config import config

class PredFusion(nn.Module):
    def __init__(self, cfg):
        super(PredFusion, self).__init__()
        hidden_size = cfg.HIDDEN_SIZE
        self.pred = nn.Conv2d(hidden_size, 1, 1, 1)
        self.Tconv_layers = nn.ModuleList()
        kernel_size = cfg.KERNEL_SIZE
        stride = cfg.STRIDE
        padding = cfg.PADDING
        assert (stride + padding * 2) == kernel_size
        for _ in range(config.TAN.NUM_LAYERS):
            self.Tconv_layers.append(nn.ConvTranspose2d(hidden_size,hidden_size,kernel_size,stride,padding))
            stride *= 2
            padding *= 2
            kernel_size *= 2

    def forward(self, fused_list):
        fused_h = fused_list[0]
        for mini_fused_h, Tconv in zip(fused_list[1:], self.Tconv_layers):
            recov_fused_h = Tconv(mini_fused_h)
            fused_h = fused_h * recov_fused_h
        fused_h = self.pred(fused_h)
        return fused_h