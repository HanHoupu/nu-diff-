import torch.nn as nn
import math


class LoRALayer(nn.Module):
    """
    只注入到 Q / V；可直接 wrap 在线性层外。
    """

    def __init__(self, orig_linear: nn.Linear, rank: int = 8, alpha: float = 16):
        super().__init__()
        self.orig = orig_linear
        self.rank = rank
        self.lora_A = nn.Parameter(
            orig_linear.weight.new_empty(rank, orig_linear.in_features)
        )
        self.lora_B = nn.Parameter(
            orig_linear.weight.new_empty(orig_linear.out_features, rank)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.scaling = alpha / rank

    def forward(self, x):
        return self.orig(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
