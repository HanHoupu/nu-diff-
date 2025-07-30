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
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
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
        # The decoder now takes the concatenation of the previous token embedding
        # and the previous delta_E value as input.
        self.embedding = nn.Embedding(vocab_size, d_model - 1) # Make space for delta_E
        self.delta_proj = nn.Linear(1, 1) # Projection layer for delta_E
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
        
        # --- Hybrid Output Heads ---
        self.classification_head = nn.Linear(d_model, vocab_size)
        # REGRESSION HEAD FIX: Removed ReLU to prevent gradient killing
        self.regression_head = nn.Linear(d_model, 1)

    def forward(self,
                src: torch.Tensor,
                tgt_tokens: torch.Tensor,
                tgt_delta_e: torch.Tensor,
                tgt_padding_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Hybrid Seq2Seq Transformer.
        """
        tgt_len = tgt_tokens.size(1)
        
        # MASKING FIX: Generate the mask dynamically in every forward pass
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt_tokens.device), diagonal=1).bool()

        # Encoder forward pass
        src = self.encoder_projection(src).unsqueeze(1) # Shape: [batch, 1, d_model]
        
        # Decoder input preparation
        token_emb = self.embedding(tgt_tokens) # Shape: [batch, seq_len, d_model - 1]
        
        # Concatenate token embedding with the *projected* continuous delta_E value
        delta_e_emb = self.delta_proj(tgt_delta_e.unsqueeze(-1))
        decoder_input = torch.cat([token_emb, delta_e_emb], dim=-1)
        decoder_input = self.pos_encoder(decoder_input)
        
        # Transformer forward pass
        output = self.transformer(
            src=src,
            tgt=decoder_input,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        
        # Pass transformer output through the two heads
        class_logits = self.classification_head(output)
        regression_output = self.regression_head(output).squeeze(-1) # Remove last dim

        return class_logits, regression_output 