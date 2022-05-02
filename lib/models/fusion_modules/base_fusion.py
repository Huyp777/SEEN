import torch
from torch import nn
import torch.nn.functional as F

class BaseFusion(nn.Module):

    def __init__(self, cfg):
        super(BaseFusion, self).__init__()
        self.cfg = cfg
        hidden_size = cfg.HIDDEN_SIZE
        self.vis_conv = nn.Conv2d(hidden_size, hidden_size, 1, 1)
        self.sent_mlp = nn.Linear(1024, hidden_size)

    def forward(self, map_h, map_mask, sent_input):
        map_h = self.vis_conv(map_h)
        sent_h = self.sent_mlp(sent_input)[:,:,None,None]
        fused_h = F.normalize(sent_h * map_h) * map_mask
        return fused_h

