import torch
import torch.nn as nn
import torch.nn.functional as F
from .lora import LoRALayer

class TransformerModel(nn.Module):
    def __init__(self, cfg, elem2idx, rec2idx):
        super().__init__()
        
        # Config
        self.d_model = cfg.get("d_model", 128)
        self.n_layers = cfg.get("n_layers", 4)
        self.n_heads = cfg.get("n_heads", 4)
        self.ff_dim = cfg.get("ff_dim", 512)
        self.lora_rank = cfg.get("rank", 8)
        self.lora_alpha = cfg.get("alpha", 8)
        
        # Embeddings
        self.elem_embed = nn.Embedding(len(elem2idx), self.d_model)
        self.rec_embed = nn.Embedding(len(rec2idx), self.d_model)
        self.numeric_proj = nn.Linear(1, self.d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.ff_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        
        # Replace Linear layers with LoRA
        self._replace_with_lora()
        
        # Multi-task heads
        self.head_L = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 1)
        )
        self.head_G = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 1)
        )
        self.head_Q = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 1)
        )
        
        # Task weights
        self.task_weights = cfg.get("task_weights", {"L": 1.0, "G": 1.0, "Q": 1.0})
    
    def _replace_with_lora(self):
        """Replace Linear layers in transformer with LoRA"""
        for layer in self.transformer.layers:
            # Feed-forward layers
            layer.linear1 = LoRALayer(
                layer.linear1, rank=self.lora_rank, alpha=self.lora_alpha
            )
            layer.linear2 = LoRALayer(
                layer.linear2, rank=self.lora_rank, alpha=self.lora_alpha
            )
    
    def forward(self, x):
        batch_size = x['element'].shape[0]
        
        # Process discrete features
        elem_emb = self.elem_embed(x['element'])  # [B, seq_len, d_model]
        rec_emb = self.rec_embed(x['record_type'])  # [B, seq_len, d_model]
        
        # Process numeric features
        numeric = x['numeric'].unsqueeze(-1)  # [B, seq_len, 1]
        numeric_emb = self.numeric_proj(numeric)  # [B, seq_len, d_model]
        
        # Combine embeddings
        combined = elem_emb + rec_emb + numeric_emb  # [B, seq_len, d_model]
        
        # Add positional encoding (simple)
        seq_len = combined.size(1)
        pos_encoding = torch.arange(seq_len, device=combined.device).unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
        pos_encoding = pos_encoding / 10000 ** (torch.arange(0, self.d_model, 2, device=combined.device) / self.d_model)
        pos_encoding = torch.cat([torch.sin(pos_encoding), torch.cos(pos_encoding)], dim=-1)  # [1, seq_len, d_model//2*2]
        if pos_encoding.size(-1) < self.d_model:
            pos_encoding = F.pad(pos_encoding, (0, self.d_model - pos_encoding.size(-1)))
        
        combined = combined + pos_encoding  # [B, seq_len, d_model]
        
        # Transformer encoding
        encoded = self.transformer(combined)  # [B, seq_len, d_model]
        
        # CLS pooling (mean pooling)
        pooled = encoded.mean(dim=1)  # [B, d_model]
        
        # Multi-task heads
        y_L = self.head_L(pooled).squeeze(-1)  # [B]
        y_G = self.head_G(pooled).squeeze(-1)  # [B]
        y_Q = self.head_Q(pooled).squeeze(-1)  # [B]
        
        return {"L": y_L, "G": y_G, "Q": y_Q}
    
    def training_step(self, batch):
        x, y = batch
        predictions = self.forward(x)
        
        # Calculate losses for each task
        loss_L = F.l1_loss(predictions["L"], y["L"]) if "L" in y else torch.tensor(0.0, device=predictions["L"].device)
        loss_G = F.l1_loss(predictions["G"], y["G"]) if "G" in y else torch.tensor(0.0, device=predictions["G"].device)
        loss_Q = F.l1_loss(predictions["Q"], y["Q"]) if "Q" in y else torch.tensor(0.0, device=predictions["Q"].device)
        
        # Weighted sum
        total_loss = (
            self.task_weights["L"] * loss_L +
            self.task_weights["G"] * loss_G +
            self.task_weights["Q"] * loss_Q
        )
        
        return total_loss
    
    def save_lora(self, path):
        """Save LoRA parameters"""
        lora_state = {}
        for name, module in self.named_modules():
            if isinstance(module, LoRALayer):
                lora_state[f"{name}.lora_A"] = module.lora_A
                lora_state[f"{name}.lora_B"] = module.lora_B
        torch.save(lora_state, path) 