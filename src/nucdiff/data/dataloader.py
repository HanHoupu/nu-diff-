# src/nucdiff/data/dataloader.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Tuple
import json
import torch.nn.functional as F

from .tokenizer import LevelTokenizer

class HybridSeqENSDFDataset(Dataset):
    """
    Prepares data for the hybrid (Regression + Classification) Seq2Seq model.
    """
    def __init__(self, df: pd.DataFrame, tokenizer, encoder_feature_cols: List[str]):
        self.df = df
        self.tokenizer = tokenizer
        self.encoder_feature_cols = encoder_feature_cols
        
        # Pre-convert features to tensors for faster access
        self.encoder_features = torch.tensor(self.df[self.encoder_feature_cols].values, dtype=torch.float32)
        # Pre-load and parse the JSON strings for sequences
        self.delta_e_sequences = [json.loads(s) for s in self.df['delta_e_sequence']]
        
        # Handle different column names for token sequences
        if 'token_sequence' in self.df.columns:
            self.token_sequences = self.df['token_sequence'].tolist()
        elif 'level_sequence' in self.df.columns:
            self.token_sequences = self.df['level_sequence'].tolist()
        else:
            raise ValueError("Neither 'token_sequence' nor 'level_sequence' found in dataframe")
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoder_features = self.encoder_features[idx]
        
        # Token sequence processing
        token_ids = self.tokenizer.encode(self.token_sequences[idx])
        decoder_input_tokens = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_tokens = torch.tensor(token_ids[1:], dtype=torch.long)
        
        # Delta E sequence processing
        delta_e = torch.tensor(self.delta_e_sequences[idx], dtype=torch.float32)
        # The input needs a "start" value for delta_E, we'll prepend a 0.
        # The target is the actual delta_E for each corresponding token.
        decoder_input_delta_e = F.pad(delta_e, (1, 0), "constant", 0)
        # The target delta_E also needs a value for the final [EOS] token, we'll append a 0.
        target_delta_e = F.pad(delta_e, (0, 1), "constant", 0)

        # The length of token and delta_e sequences might differ by 1 after processing.
        # We need to ensure they have the same length for padding and batching.
        min_len = min(len(decoder_input_tokens), len(decoder_input_delta_e))
        
        # NEW: Calculate target sequence length (number of valid tokens including EOS)
        target_length = torch.tensor(len(target_tokens[:min_len]), dtype=torch.float32)

        return {
            'encoder_features': encoder_features,
            'decoder_input_tokens': decoder_input_tokens[:min_len],
            'decoder_input_delta_e': decoder_input_delta_e[:min_len],
            'target_tokens': target_tokens[:min_len],
            'target_delta_e': target_delta_e[:min_len],
            'target_length': target_length,  # NEW: Target sequence length
        }

class PadCollateHybrid:
    """
    Callable class to pad sequences for the hybrid model batch.
    """
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Separate the different data types from the batch
        encoder_features = [item['encoder_features'] for item in batch]
        input_tokens = [item['decoder_input_tokens'] for item in batch]
        input_delta_e = [item['decoder_input_delta_e'] for item in batch]
        target_tokens = [item['target_tokens'] for item in batch]
        target_delta_e = [item['target_delta_e'] for item in batch]
        target_lengths = [item['target_length'] for item in batch]  # NEW: Extract target lengths
        
        # Pad all sequence-like tensors
        padded_input_tokens = pad_sequence(input_tokens, batch_first=True, padding_value=self.pad_id)
        padded_input_delta_e = pad_sequence(input_delta_e, batch_first=True, padding_value=0.0)
        padded_target_tokens = pad_sequence(target_tokens, batch_first=True, padding_value=self.pad_id)
        padded_target_delta_e = pad_sequence(target_delta_e, batch_first=True, padding_value=0.0)
        
        # Stack the non-sequence data
        stacked_encoder_features = torch.stack(encoder_features, dim=0)
        stacked_target_lengths = torch.stack(target_lengths, dim=0)  # NEW: Stack target lengths
        
        # Create the attention mask for the decoder based on token padding
        attention_mask = (padded_input_tokens != self.pad_id)

        return {
            'encoder_features': stacked_encoder_features,
            'decoder_input_tokens': padded_input_tokens,
            'decoder_input_delta_e': padded_input_delta_e,
            'target_tokens': padded_target_tokens,
            'target_delta_e': padded_target_delta_e,
            'target_lengths': stacked_target_lengths,  # NEW: Target sequence lengths
            'attention_mask': attention_mask
        }
