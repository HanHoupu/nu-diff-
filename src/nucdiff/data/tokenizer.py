import pandas as pd
from pathlib import Path
from typing import List, Dict
import json

class LevelTokenizer:
    """
    A simple tokenizer for nuclear level sequences.
    Handles tokenization, vocabulary building, and encoding/decoding.
    - Supports explicit year tokens like "<Y2006>" when present in sequences.
    """
    def __init__(self, pad_token="[PAD]", sos_token="[SOS]", eos_token="[EOS]", unk_token="[UNK]", sep_token=";"):
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.sep_token = sep_token

        # Initialize vocab with special tokens
        self.special_tokens = [pad_token, sos_token, eos_token, unk_token]
        self.token_to_id: Dict[str, int] = {token: i for i, token in enumerate(self.special_tokens)}
        self.id_to_token: Dict[int, str] = {i: token for token, i in self.token_to_id.items()}

        # cache for quick year-token detection
        self._year_prefix = "<Y"
        self._year_suffix = ">"

    def __len__(self):
        return len(self.token_to_id)

    @property
    def pad_id(self):
        return self.token_to_id[self.pad_token]

    @property
    def sos_id(self):
        return self.token_to_id[self.sos_token]

    @property
    def eos_id(self):
        return self.token_to_id[self.eos_token]
        
    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def build_vocab(self, sequences: pd.Series):
        """Builds the vocabulary from a series of sequence strings."""
        all_tokens = set()
        # Split each sequence by the separator to get the actual tokens
        for seq in sequences:
            tokens = seq.split(self.sep_token)
            all_tokens.update(t.strip() for t in tokens if t is not None and t != "")

        # Add new tokens to the vocabulary
        for token in sorted(list(all_tokens)):
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
        print(f"Vocabulary built. Total size: {len(self.token_to_id)} tokens.")

    def encode(self, sequence: str) -> List[int]:
        """Encodes a single sequence string into a list of token IDs."""
        tokens = [t.strip() for t in sequence.split(self.sep_token) if t is not None and t != ""]
        encoded = [self.sos_id]
        for token in tokens:
            # Treat explicit year tokens as atomic tokens
            if token.startswith(self._year_prefix) and token.endswith(self._year_suffix):
                encoded.append(self.token_to_id.get(token, self.token_to_id[self.unk_token]))
            else:
                encoded.append(self.token_to_id.get(token, self.token_to_id[self.unk_token]))
        encoded.append(self.eos_id)
        return encoded

    def decode(self, token_ids: List[int]) -> str:
        """Decodes a list of token IDs back into a string sequence."""
        tokens = []
        for token_id in token_ids:
            if token_id in {self.sos_id, self.eos_id, self.pad_id}:
                continue
            tokens.append(self.id_to_token.get(token_id, self.unk_token))
        
        return self.sep_token.join(tokens)

    def save(self, path: Path):
        """Saves the tokenizer's token_to_id mapping to a file."""
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.token_to_id, f, indent=4)

    @classmethod
    def load(cls, path: Path):
        """Loads a tokenizer from a saved token_to_id file."""
        with open(path, 'r', encoding='utf-8') as f:
            token_to_id = json.load(f)
        
        tokenizer = cls() 
        tokenizer.token_to_id = token_to_id
        tokenizer.id_to_token = {i: token for token, i in token_to_id.items()}
        return tokenizer