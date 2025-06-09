import torch
import torch.nn as nn
from .lora import LoRALayer

# ← 必须与 DataLoader 中自动选出的 numeric_cols 数量保持一致
NUMERIC_DIM = 7   # Z, A, energy_keV, mass_excess_keV, q_value_keV, e_gamma_keV, intensity

class IncrementalModel(nn.Module):
    def __init__(
        self,
        elem2idx: dict,
        rec2idx: dict,
        numeric_dim: int,
        embed_dim: int = 8,
        rank: int = 8,
        alpha: int = 16,
        train_backbone: bool = True
    ):
        super().__init__()
        # Embedding 层：element + record_type
        self.elem_emb = nn.Embedding(len(elem2idx), embed_dim)
        self.rec_emb  = nn.Embedding(len(rec2idx),  embed_dim)

        # 输入维度 = 数值特征 + 两个 embedding
        self.numeric_dim = numeric_dim
        in_dim = NUMERIC_DIM + embed_dim * 2

        # 主干：两层带 LoRA 的 MLP
        self.fc1 = LoRALayer(nn.Linear(in_dim, 128), rank=rank, alpha=alpha)
        self.act = nn.ReLU()
        self.fc2 = LoRALayer(nn.Linear(128, 64),   rank=rank, alpha=alpha)

        # 任务头：预测一个标量
        self.head = nn.Linear(64, 1)

        # 冻结逻辑：首年微调全部，后续只训 LoRA + head
        if not train_backbone:
            for p in self.parameters():
                p.requires_grad_(False)
            for n, p in self.named_parameters():
                if "lora_" in n or "head" in n:
                    p.requires_grad_(True)

    def forward(self, batch):
        # batch["num"]: [B, numeric_dim]
        # batch["elem"]: [B]; batch["rec"]: [B]
        x_num  = batch["num"]
        x_elem = self.elem_emb(batch["elem"])
        x_rec  = self.rec_emb(batch["rec"])
        h = torch.cat([x_num, x_elem, x_rec], dim=-1)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        return self.head(h).squeeze(-1)

    def training_step(self, batch):
        x, y = batch
        pred = self.forward(x)
        return nn.functional.mse_loss(pred, y)

    def save_lora(self, path: str):
        # 仅保存 LoRA 相关权重
        lora_state = {
            k: v for k, v in self.state_dict().items() if "lora_" in k
        }
        torch.save(lora_state, path)
