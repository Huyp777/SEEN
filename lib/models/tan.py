from torch import nn
from core.config import config
import models.frame_modules as frame_modules
import models.prop_modules as prop_modules
import models.map_modules as map_modules
import models.atten_modules as atten_modules
import models.cross_modules as cross_modules
import models.fusion_modules as fusion_modules
import models.pred_modules as pred_modules

class TAN(nn.Module):
    def __init__(self):
        super(TAN, self).__init__()

        self.frame_layer = getattr(frame_modules, config.TAN.FRAME_MODULE.NAME)(config.TAN.FRAME_MODULE.PARAMS)

        self.prop_layers = nn.ModuleList()
        self.atten_layers = nn.ModuleList()
        self.cross_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        self.map_layers = nn.ModuleList()

        for _ in range(config.TAN.NUM_LAYERS):
            self.prop_layers.append(getattr(prop_modules, config.TAN.PROP_MODULE.NAME)(config.TAN.PROP_MODULE.PARAMS))
            self.atten_layers.append(getattr(atten_modules, config.TAN.ATTEN_MODULE.NAME)(config.TAN.ATTEN_MODULE.PARAMS))
            self.cross_layers.append(getattr(cross_modules, config.TAN.CROSS_MODULE.NAME)(config.TAN.CROSS_MODULE.PARAMS))
            self.fusion_layers.append(getattr(fusion_modules, config.TAN.FUSION_MODULE.NAME)(config.TAN.FUSION_MODULE.PARAMS))
            self.map_layers.append(getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS))

        self.pred_layer = getattr(pred_modules, config.TAN.PRED_MODULE.NAME)(config.TAN.PRED_MODULE.PARAMS)

    def forward(self, textual_input, textual_mask, visual_input, sent_input):

        vis_list = self.frame_layer(visual_input.transpose(1, 2)) # [x]*16*512*16
        fused_list = []
        map_mask_list = []
        for vis_h, atten_layer, cross_layer, prop_layer, fusion_layer, map_layer in \
            zip(vis_list, self.atten_layers, self.cross_layers, self.prop_layers, self.fusion_layers, self.map_layers):
            vis_h = atten_layer(vis_h) # 16*512*16
            vis_h = cross_layer(vis_h, textual_input, textual_mask) # 16*512*16
            map_h, map_mask = prop_layer(vis_h) #16*512*16*16, 16*1*16*16
            fused_h = fusion_layer(map_h, map_mask, sent_input) #16*512*16*16
            fused_h = map_layer(fused_h, map_mask) #16*512*16*16
            fused_list.append(fused_h)
            map_mask_list.append(map_mask)
        prediction = self.pred_layer(fused_list) * map_mask_list[0] #16*1*16*16
        return prediction, map_mask_list[0]

    def extract_features(self, textual_input, textual_mask, visual_input):
        vis_h = self.frame_layer(visual_input.transpose(1, 2))
        map_h, map_mask = self.prop_layer(vis_h)

        fused_h = self.fusion_layer(textual_input, textual_mask, map_h, map_mask)
        fused_h = self.map_layer(fused_h, map_mask)
        prediction = self.pred_layer(fused_h) * map_mask

        return fused_h, prediction, map_mask
