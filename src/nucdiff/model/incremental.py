import torch
import torch.nn as nn
from .lora import LoRALayer

# ← Must be consistent with the number of numeric_cols automatically selected in DataLoader
numeric_dim = (
    7  # Z, A, energy_keV, mass_excess_keV, q_value_keV, e_gamma_keV, intensity
)


class IncrementalModel(nn.Module):
    def __init__(
        self,
        elem2idx: dict,
        rec2idx: dict,
        numeric_dim: int,
        embed_dim: int = 8,
        rank: int = 8,
        alpha: int = 16,
        train_backbone: bool = True,
    ):
        super().__init__()
        # Embedding layers: element + record_type
        self.elem_emb = nn.Embedding(len(elem2idx), embed_dim)
        self.rec_emb = nn.Embedding(len(rec2idx), embed_dim)

        # Input dimension = numeric features + two embeddings
        self.numeric_dim = numeric_dim
        in_dim = numeric_dim + embed_dim * 2

        # Backbone: two-layer MLP with LoRA
        self.fc1 = LoRALayer(nn.Linear(in_dim, 128), rank=rank, alpha=alpha)
        self.act = nn.ReLU()
        self.fc2 = LoRALayer(nn.Linear(128, 64), rank=rank, alpha=alpha)

        # Task head: predict a scalar
        self.head = nn.Linear(64, 1)

        # Freezing logic: finetune all in the first year, only train LoRA + head in subsequent years
        if not train_backbone:
            for p in self.parameters():
                p.requires_grad_(False)
            for n, p in self.named_parameters():
                if "lora_" in n or "head" in n:
                    p.requires_grad_(True)

    def forward(self, batch):
        # batch["num"]: [B, numeric_dim]
        # batch["elem"]: [B]; batch["rec"]: [B]
        x_num = batch["num"]
        x_elem = self.elem_emb(batch["elem"])
        x_rec = self.rec_emb(batch["rec"])
        h = torch.cat([x_num, x_elem, x_rec], dim=-1)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        return self.head(h).squeeze(-1)

    def training_step(self, batch):
        x, y = batch
        pred = self.forward(x)
        return nn.functional.mse_loss(pred, y)

    def save_lora(self, path: str):
        # Only save LoRA-related weights
        lora_state = {k: v for k, v in self.state_dict().items() if "lora_" in k}
        torch.save(lora_state, path)
