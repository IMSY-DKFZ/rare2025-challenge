import math
import torch
from torch import nn

dinov3_l_config = {
    "model_name": "dinov3_vitl16",
    "embed_dim": 1024,
    "patch_size": 16,
    "num_classes": 0
}

class DinoV3(nn.Module):
    def __init__(self, config=None, lora=False):
        super().__init__()
        self.config = config if config is not None else dinov3_l_config
        self.weight_path = config.model.weight_path
        self.lora = lora
        self._init_model()

    def _init_model(self):
        print(self.weight_path)
        self.model = torch.hub.load("facebookresearch/dinov3", self.config.model.model_name, source='github', weights=self.weight_path)
        self.feat_dim = self.model.embed_dim
        if self.lora:
            for param in self.model.parameters():
                param.requires_grad = False
            self._add_lora_layers()
        self.head = torch.nn.Linear(self.feat_dim, 2)

    def forward(self, x):
        features = self.model(x)
        return self.head(features)

    def _add_lora_layers(self, rank=32, alpha=64):
        def _find_and_replace_linear_layers(module, path=""):
            for name, child in module.named_children():
                new_path = f"{path}.{name}" if path else name
                if isinstance(child, nn.Linear):
                    setattr(module, name, LoRALinear(child, rank, alpha))
                else:
                    _find_and_replace_linear_layers(child, new_path)
        _find_and_replace_linear_layers(self.model)

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        self.scale = alpha / rank
        # add dropout
        self.dropout = nn.Dropout(0.1)

        # Initialize weights
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
    def forward(self, x):
        return self.up(self.dropout(self.down(x))) * self.scale

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=1):
        super().__init__()
        self.linear = linear_layer
        self.in_features = linear_layer.in_features
        self.lora = LoRALayer(
            self.in_features, linear_layer.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)