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
        self.feature_type_embed = nn.Embedding(15, self.d_model)  # 15个特征类型
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
        seq_len = x['numeric'].shape[1]  # 15个数值特征作为序列长度
        
        # Process numeric features as tokens
        numeric = x['numeric'].unsqueeze(-1)  # [B, 15, 1]
        numeric_emb = self.numeric_proj(numeric)  # [B, 15, d_model]
        
        # Process feature type embeddings
        feature_type_emb = self.feature_type_embed(x['feature_type_ids'])  # [B, 15, d_model]
        
        # Process discrete features - expand to match sequence length
        elem_emb = self.elem_embed(x['element'])  # [B, d_model]
        elem_emb = elem_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, 15, d_model]
        
        rec_emb = self.rec_embed(x['record_type'])  # [B, d_model]
        rec_emb = rec_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, 15, d_model]
        
        # Combine all embeddings
        combined = numeric_emb + feature_type_emb + elem_emb + rec_emb  # [B, 15, d_model]
        
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
        
        # Calculate losses for each task - 过滤NaN值
        if "L" in y:
            valid_L = ~torch.isnan(y["L"])
            if valid_L.any():
                loss_L = F.l1_loss(predictions["L"][valid_L], y["L"][valid_L])
            else:
                loss_L = torch.tensor(0.0, device=predictions["L"].device)
        else:
            loss_L = torch.tensor(0.0, device=predictions["L"].device)
            
        if "G" in y:
            valid_G = ~torch.isnan(y["G"])
            if valid_G.any():
                loss_G = F.l1_loss(predictions["G"][valid_G], y["G"][valid_G])
            else:
                loss_G = torch.tensor(0.0, device=predictions["G"].device)
        else:
            loss_G = torch.tensor(0.0, device=predictions["G"].device)
            
        if "Q" in y:
            valid_Q = ~torch.isnan(y["Q"])
            if valid_Q.any():
                loss_Q = F.l1_loss(predictions["Q"][valid_Q], y["Q"][valid_Q])
            else:
                loss_Q = torch.tensor(0.0, device=predictions["Q"].device)
        else:
            loss_Q = torch.tensor(0.0, device=predictions["Q"].device)
        
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