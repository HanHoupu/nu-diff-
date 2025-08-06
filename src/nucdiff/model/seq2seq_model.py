# src/nucdiff/model/seq2seq_model.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """为输入序列添加位置信息。"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] when batch_first=True
        """
        # FIXED: Match input dtype for AMP compatibility  
        seq_len = x.size(1)
        pe = self.pe[:seq_len].transpose(0, 1).to(dtype=x.dtype, device=x.device)  # Match input dtype
        x = x + pe
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    """
    A standard Encoder-Decoder Transformer model, modified for a hybrid output
    (regression for delta_E, classification for discrete tokens).
    """
    def __init__(self,
                 vocab_size: int,
                 encoder_feature_dim: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_encoder_layers: int = 6,
                 n_decoder_layers: int = 6,
                 ff_dim: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # --- Encoder ---
        self.encoder_projection = nn.Linear(encoder_feature_dim, d_model)
        
        # --- Decoder ---
        # Simpler approach: Use full d_model for token embedding, add delta_E separately
        self.embedding = nn.Embedding(vocab_size, d_model) # Full embedding dimension
        self.delta_proj = nn.Linear(1, d_model) # Project delta_E to same dimension
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # --- PyTorch Standard Transformer ---
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        
        # --- Three Output Heads ---
        # Per user's checklist: Head hidden dim >= embedding dim, layers>=2
        hidden_dim = max(d_model * 2, 1024)  # Ensure sufficient capacity
        
        # 1. Token Classification Head (per-position token prediction)
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # Per user's guidance for numerical stability
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # 2. Regression Head (per-position continuous values)
        self.regression_head = nn.Linear(d_model, 1)
        # FIXED: Safe regression head initialization
        with torch.no_grad():
            nn.init.xavier_uniform_(self.regression_head.weight, gain=0.1)  # Smaller gain to prevent NaN
            self.regression_head.bias.data.zero_()    # Start from zero bias
            
        # FIXED: Cache mask to avoid repeated creation
        self._cached_mask = {}  # Cache for different sequence lengths

    def forward(self,
                src: torch.Tensor,
                tgt_tokens: torch.Tensor,
                tgt_delta_e: torch.Tensor,
                tgt_padding_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Two-Head Seq2Seq Transformer.
        Returns: (class_logits, regression_output)
        """
        tgt_len = tgt_tokens.size(1)
        device = tgt_tokens.device
        
        # FIXED: Use cached mask to avoid repeated creation
        cache_key = (tgt_len, device)
        if cache_key not in self._cached_mask:
            self._cached_mask[cache_key] = torch.triu(
                torch.ones(tgt_len, tgt_len, device=device), diagonal=1
            ).bool()
        tgt_mask = self._cached_mask[cache_key]

        # Encoder forward pass
        encoder_src = self.encoder_projection(src).unsqueeze(1) # Shape: [batch, 1, d_model]
        
        # Decoder input preparation
        token_emb = self.embedding(tgt_tokens) # Shape: [batch, seq_len, d_model]
        
        # FIXED: Add token embedding with the projected continuous delta_E value
        delta_e_emb = self.delta_proj(tgt_delta_e.unsqueeze(-1)) # Shape: [batch, seq_len, d_model]
        decoder_input = token_emb + delta_e_emb  # Element-wise addition
        decoder_input = self.pos_encoder(decoder_input)
        
        # Transformer forward pass
        output = self.transformer(
            src=encoder_src,
            tgt=decoder_input,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        
        # 1. Token-level predictions from decoder output
        class_logits = self.classification_head(output)
        regression_output = self.regression_head(output).squeeze(-1) # Remove last dim
        
        return class_logits, regression_output 