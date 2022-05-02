import torch
from torch import nn
from core.config import config

class FrameAvgPool(nn.Module):

    def __init__(self, cfg):
        super(FrameAvgPool, self).__init__()
        input_size = cfg.INPUT_SIZE
        hidden_size = cfg.HIDDEN_SIZE
        kernel_size = cfg.KERNEL_SIZE
        stride = cfg.STRIDE
        self.vis_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.avg_pools = nn.ModuleList()
        for _ in range(config.TAN.NUM_LAYERS):
            self.avg_pools.append(nn.AvgPool1d(kernel_size, stride))
            stride *= 2
            kernel_size *= 2

    def forward(self, visual_input):
        visual_input = torch.relu(self.vis_conv(visual_input))
        vis_list = []
        for avg_pool in self.avg_pools:
            vis_h = avg_pool(visual_input)
            vis_list.append(vis_h)
        return vis_list

class FrameMaxPool(nn.Module):

    def __init__(self, input_size, hidden_size, stride):
        super(FrameMaxPool, self).__init__()
        self.vis_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.max_pool = nn.MaxPool1d(stride)

    def forward(self, visual_input):
        vis_h = torch.relu(self.vis_conv(visual_input))
        vis_h = self.max_pool(vis_h)
        return vis_h
