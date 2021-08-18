import timm
import torch
import torch.nn as nn
from typing import List

from vissl.config import AttrDict
from vissl.models.trunks import register_model_trunk


@register_model_trunk("timm_efficientnet")
class TimmEfficientNet(nn.Module):
    def __init__(self, model_config: AttrDict, model_name: str):
        super(TimmEfficientNet, self).__init__()
        trunk_config = model_config.TRUNK.EFFICIENT_NETS
        trunk_config.update({'num_classes': 0})
        self.backbone = timm.create_model(**trunk_config)

    def forward(self, x: torch.Tensor, out_feat_keys: List[str] = None) -> List[torch.Tensor]:
        return [self.backbone(x)]
