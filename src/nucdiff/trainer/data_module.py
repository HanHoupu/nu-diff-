# src/nucdiff/trainer/data_module.py
"""
Lightning DataModule for ENSDF nuclear data.
Replaces manual data_factory logic with standard Lightning pattern.
"""
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd
import torch
from pathlib import Path
from typing import Optional
try:
    from lightning.pytorch.utilities import rank_zero_only
except ImportError:
    from pytorch_lightning.utilities import rank_zero_only

from src.nucdiff.data.dataloader import HybridSeqENSDFDataset, PadCollateHybrid
from src.nucdiff.data.tokenizer import LevelTokenizer
from src.nucdiff.utils.constants import ENCODER_FEATURE_COLS


class ENSDFDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for ENSDF nuclear physics data.
    Handles data loading, tokenization, and DataLoader creation.
    """
    
    def __init__(self, task: str = "SEQ", years_range: str = "2005-2010", seed: int = 412,
                 train_frac: float = 0.20, data_subset_ratio: float = 1.0,
                 batch_size: int = 32, gradient_accumulation_steps: int = 4,
                 num_workers: int = 4, force_rebuild_vocab: bool = False,
                 overfit: bool = False, overfit_samples: int = 5,
                 # Route-2 temporal augmentation (reproducible)
                 epoch_resample: bool = False, k_year_augment: int = 2):
        super().__init__()
        # Create cfg dictionary from flattened parameters
        self.cfg = {
            'task': task,
            'years_range': years_range,
            'seed': seed,
            'train_frac': train_frac,
            'data_subset_ratio': data_subset_ratio,
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'num_workers': num_workers,
            'force_rebuild_vocab': force_rebuild_vocab,
            'epoch_resample': epoch_resample,
            'k_year_augment': int(k_year_augment),
        }
        self.overfit = overfit
        self.overfit_samples = overfit_samples
        
        # Will be initialized in setup()
        self.tokenizer = None
        self.class_weights = None
        self.train_dataset = None
        self.val_dataset = None

        # Temporal augmentation metadata (loaded in setup)
        self.year_index_map = None
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets and tokenizer."""
        if stage == "fit" or stage is None:
            self._setup_data()
    
    def _setup_data(self):
        """Setup tokenizer, datasets, and class weights."""
        # Build tokenizer
        self.tokenizer = self._build_tokenizer()
        
        # Load datasets
        train_df, val_df = self._load_datasets()
        
        # Process class weights
        self.class_weights = self._process_class_weights(train_df)
        
        # FIXED: Use centralized feature column definition
        self.train_dataset = HybridSeqENSDFDataset(train_df, self.tokenizer, ENCODER_FEATURE_COLS)
        self.val_dataset = HybridSeqENSDFDataset(val_df, self.tokenizer, ENCODER_FEATURE_COLS)

        # Load temporal augmentation index map if enabled
        if self.cfg.get("epoch_resample", False):
            self._load_year_index_map(default_dir=Path("preprocessed_data"))
    
    def _build_tokenizer(self) -> LevelTokenizer:
        """Build or load cached tokenizer - integrated logic."""
        cache_path = Path(f"preprocessed_data/tokenizer_{self.cfg['task']}_cache.pkl")
        
        if cache_path.exists() and not self.cfg.get("force_rebuild_vocab", False):
            self._log_info("Loading cached tokenizer")
            return LevelTokenizer.load(cache_path)
        
        self._log_info("Building fresh tokenizer")
        train_df, _ = self._load_raw_datasets()
        tokenizer = LevelTokenizer()
        tokenizer.build_vocab(train_df['token_sequence'].tolist())
        tokenizer.save(cache_path)
        return tokenizer
    
    def _load_datasets(self):
        """Load train/val datasets - integrated logic."""
        train_df, val_df = self._load_raw_datasets()
        
        # Log original dataset size
        self._log_info(f"📂 ORIGINAL DATA SIZE:")
        self._log_info(f"   - Train samples: {len(train_df):,}")
        self._log_info(f"   - Val samples: {len(val_df):,}")
        self._log_info(f"   - Total samples: {len(train_df) + len(val_df):,}")
        
        # Apply data subset ratio for gradual scaling experiments
        data_subset_ratio = self.cfg.get("data_subset_ratio", 1.0)
        if data_subset_ratio < 1.0:
            import numpy as np
            # Set seed for reproducible sampling
            np.random.seed(self.cfg.get("seed", 412))
            
            # Sample subset of training data
            train_size = int(len(train_df) * data_subset_ratio)
            train_indices = np.random.choice(len(train_df), size=train_size, replace=False)
            train_df = train_df.iloc[train_indices].reset_index(drop=True)
            
            # Sample subset of validation data  
            val_size = int(len(val_df) * data_subset_ratio)
            val_indices = np.random.choice(len(val_df), size=val_size, replace=False)
            val_df = val_df.iloc[val_indices].reset_index(drop=True)
            
            self._log_info(f"📊 DATA SUBSET: Using {data_subset_ratio*100:.0f}% of data")
            self._log_info(f"   - Train samples: {len(train_df):,}")
            self._log_info(f"   - Val samples: {len(val_df):,}")
        
        if self.overfit:
            self._log_info(f"🔧 OVERFIT MODE: Using {self.overfit_samples} samples")
            train_df = train_df.head(self.overfit_samples)
            val_df = train_df.copy()  # Use same data for validation in overfit mode
        
        return train_df, val_df
    
    def _load_raw_datasets(self):
        """Load structured data from new state machine processing."""
        structured_path = Path("structured_data") 
        years = self.cfg["years_range"]
        
        # Load all record types from structured_data
        record_files = [
            f"l_records_{years}.csv",  # Level records (largest dataset)
            f"a_records_{years}.csv",  # A records  
            f"n_records_{years}.csv",  # N records
            # Add other record types as needed
        ]
        
        all_data = []
        for file_name in record_files:
            file_path = structured_path / file_name
            if file_path.exists():
                self._log_info(f"Loading {file_name}...")
                df = pd.read_csv(file_path)
                self._log_info(f"   - Loaded {len(df):,} records")
                all_data.append(df)
        
        # Combine all record types
        combined_df = pd.concat(all_data, ignore_index=True)
        self._log_info(f"Combined total: {len(combined_df):,} records")
        
        # 🔧 FIXED: Transform ALL records to sequences FIRST, then split
        self._log_info("Converting all structured data to SEQ format...")
        all_sequences_df = self._transform_structured_to_seq_format(combined_df)
        
        # Now split sequences (not records) based on train_frac
        from sklearn.model_selection import train_test_split
        
        # Try stratified split by nuclide, fall back to random split if stratification fails
        try:
            train_df, val_df = train_test_split(
                all_sequences_df, 
                test_size=(1 - self.cfg["train_frac"]), 
                random_state=self.cfg["seed"],
                stratify=all_sequences_df[['Z_int', 'A_int']].apply(tuple, axis=1)  # Stratify by nuclide
            )
            self._log_info("Used stratified train/val split on sequences")
        except ValueError as e:
            self._log_info(f"Stratified split failed ({e}), using random split on sequences")
            train_df, val_df = train_test_split(
                all_sequences_df, 
                test_size=(1 - self.cfg["train_frac"]), 
                random_state=self.cfg["seed"]
            )
        
        self._log_info(f"CORRECT SPLIT - Train: {len(train_df)}, Val: {len(val_df)}")
        self._log_info(f"   - Train ratio: {len(train_df)/(len(train_df)+len(val_df))*100:.1f}%")
        
        return train_df, val_df
    
    def _transform_structured_to_seq_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform structured_data format to SEQ model-compatible format using preprocess.py logic."""
        self._log_info("Converting structured data to SEQ format...")
        
        import re
        import json
        import numpy as np
        
        # Step 1: Parse nuclide_id to extract Z, A
        # Create complete element symbol to atomic number mapping
        element_to_z = {
            'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'NE': 10,
            'NA': 11, 'MG': 12, 'AL': 13, 'SI': 14, 'P': 15, 'S': 16, 'CL': 17, 'AR': 18, 'K': 19, 'CA': 20,
            'SC': 21, 'TI': 22, 'V': 23, 'CR': 24, 'MN': 25, 'FE': 26, 'CO': 27, 'NI': 28, 'CU': 29, 'ZN': 30,
            'GA': 31, 'GE': 32, 'AS': 33, 'SE': 34, 'BR': 35, 'KR': 36, 'RB': 37, 'SR': 38, 'Y': 39, 'ZR': 40,
            'NB': 41, 'MO': 42, 'TC': 43, 'RU': 44, 'RH': 45, 'PD': 46, 'AG': 47, 'CD': 48, 'IN': 49, 'SN': 50,
            'SB': 51, 'TE': 52, 'I': 53, 'XE': 54, 'CS': 55, 'BA': 56, 'LA': 57, 'CE': 58, 'PR': 59, 'ND': 60,
            'PM': 61, 'SM': 62, 'EU': 63, 'GD': 64, 'TB': 65, 'DY': 66, 'HO': 67, 'ER': 68, 'TM': 69, 'YB': 70,
            'LU': 71, 'HF': 72, 'TA': 73, 'W': 74, 'RE': 75, 'OS': 76, 'IR': 77, 'PT': 78, 'AU': 79, 'HG': 80,
            'TL': 81, 'PB': 82, 'BI': 83, 'PO': 84, 'AT': 85, 'RN': 86, 'FR': 87, 'RA': 88, 'AC': 89, 'TH': 90,
            'PA': 91, 'U': 92, 'NP': 93, 'PU': 94, 'AM': 95, 'CM': 96, 'BK': 97, 'CF': 98, 'ES': 99, 'FM': 100
        }
        
        def parse_nuclide_id(nuclide_id):
            # Example: "100AG" -> A=100, Z=47; "24MG" -> A=24, Z=12
            match = re.match(r'(\d+)([A-Za-z]+)', str(nuclide_id))
            if match:
                A = int(match.group(1))
                element = match.group(2).upper()
                Z = element_to_z.get(element, 1)  # Default to hydrogen if unknown
                return Z, A
            return 1, 1  # Default fallback
        
        df['Z_int'], df['A_int'] = zip(*df['nuclide_id'].apply(parse_nuclide_id))
        
        # Step 2: Parse spin_parity 
        def parse_spin_parity(spin_parity):
            if pd.isna(spin_parity) or spin_parity == '':
                return 1, 1  # Default: twice_J=1, parity=1
            
            try:
                # Handle formats like "1/2+", "3/2-", "0+", etc.
                if '+' in str(spin_parity):
                    spin_part = str(spin_parity).replace('+', '')
                    parity = 2  # positive parity
                elif '-' in str(spin_parity):
                    spin_part = str(spin_parity).replace('-', '')
                    parity = 1  # negative parity  
                else:
                    spin_part = str(spin_parity)
                    parity = 1  # default
                
                if '/' in spin_part:
                    num, den = spin_part.split('/')
                    twice_J = int(num)  # Already twice the spin
                else:
                    twice_J = int(float(spin_part)) * 2  # Convert to twice_J
                
                return twice_J, parity
            except:
                return 1, 1  # Fallback
        
        df['twice_J'], df['parity'] = zip(*df['spin_parity'].apply(parse_spin_parity))
        
        # Step 3: Calculate log10_tau_s from half_life_s
        def safe_log10_tau(half_life_s):
            if pd.isna(half_life_s) or half_life_s <= 0:
                return 0.0
            return np.log10(float(half_life_s))
        
        df['log10_tau_s'] = df['half_life_s'].apply(safe_log10_tau)
        
        # Step 4: Set other required fields
        df['is_isomer'] = ((df['energy_keV'] > 0) & (df['log10_tau_s'] > -6)).astype(int)
        
        # Map record_type to indices
        record_type_map = {'l': 0, 'a': 1, 'n': 2, 'g': 3, 'b': 4, 'p': 5, 'e': 6, 'q': 7}
        df['record_type_idx'] = df['record_type'].map(record_type_map).fillna(0).astype(int)
        
        # Step 5: Group by nuclide and create sequences (following preprocess.py logic) 
        final_sequences = []
        
        for (Z, A), group in df.groupby(['Z_int', 'A_int']):
            if len(group) == 0:
                continue
                
            # Limit sequence length to 256 (from preprocess.py)
            if len(group) > 256:
                group = group.head(256)
            
            # Sort by energy for proper delta_E calculation
            group = group.sort_values('energy_keV')
            
            # Calculate delta_E sequence
            group['delta_E'] = group['energy_keV'].diff().fillna(0)
            
            # Create JP index for same J,P combinations
            group['jp_index'] = group.groupby(['twice_J', 'parity']).cumcount()
            
            # Create discrete tokens (following preprocess.py format)
            group['discrete_token'] = group.apply(
                lambda row: (f"J={row['twice_J']:.0f},"
                           f"P={row['parity']:.0f},"
                           f"T={row['log10_tau_s']:.2f},"
                           f"I={row['is_isomer']},"
                           f"R={row['record_type_idx']},"
                           f"N={row['jp_index']:.0f}"),
                axis=1
            )
            
            # Assemble final sequence
            delta_e_sequence = group['delta_E'].tolist()
            token_sequence = ';'.join(group['discrete_token'])
            latest_year = group['year'].max()
            
            final_sequences.append({
                'Z_int': Z, 'A_int': A, 'year': latest_year,
                'delta_e_sequence': json.dumps(delta_e_sequence),
                'token_sequence': token_sequence
            })
        
        df_sequences = pd.DataFrame(final_sequences)
        self._log_info(f"Created {len(df_sequences)} nuclide sequences from {len(df)} records")
        
        # Step 6: Add engineered features (simplified version)
        df_sequences['asym'] = (df_sequences['A_int'] - 2 * df_sequences['Z_int']) / df_sequences['A_int']
        df_sequences['is_odd_Z'] = (df_sequences['Z_int'] % 2).astype(int)
        df_sequences['is_odd_N'] = ((df_sequences['A_int'] - df_sequences['Z_int']) % 2).astype(int)
        
        # Magic numbers for nuclear physics
        magic_numbers = [2, 8, 20, 28, 50, 82, 126]
        def delta_magic(x):
            return min([abs(x - m) for m in magic_numbers])
        
        df_sequences['delta_magic_z'] = df_sequences['Z_int'].apply(delta_magic)
        df_sequences['delta_magic_n'] = (df_sequences['A_int'] - df_sequences['Z_int']).apply(delta_magic)
        
        # Shell model valence nucleons (simplified)
        df_sequences['valence_p'] = df_sequences['Z_int'] % 8
        df_sequences['valence_n'] = (df_sequences['A_int'] - df_sequences['Z_int']) % 8
        
        # Final column selection
        required_cols = [
            'A_int', 'Z_int', 'asym', 'delta_e_sequence', 'delta_magic_n',
            'delta_magic_z', 'is_odd_N', 'is_odd_Z', 'token_sequence', 
            'valence_n', 'valence_p', 'year'
        ]
        
        df_sequences = df_sequences[required_cols]
        
        self._log_info(f"Transformed to SEQ format: {len(df_sequences)} nuclide sequences")
        return df_sequences
    
    def _process_class_weights(self, train_df: pd.DataFrame) -> torch.Tensor:
        """Process class weights - simplified logic."""
        if self.overfit:
            self._log_info("🔄 Overfit mode: Using uniform weights")
            return torch.ones(len(self.tokenizer))
        
        return torch.ones(len(self.tokenizer))  # Simplified for now
    
    @rank_zero_only
    def _log_info(self, message: str):
        """Log info message (rank 0 only to prevent multi-GPU duplication)."""
        try:
            print(message)
        except UnicodeEncodeError:
            # Fallback for GBK consoles on Windows: strip non-encodable characters
            safe = message.encode("ascii", errors="ignore").decode("ascii", errors="ignore")
            print(safe)

    def _load_year_index_map(self, default_dir: Path):
        """Load year_index_map.json exported by preprocess for reproducible Route-2 augmentation."""
        import json
        map_path = default_dir / "year_index_map.json"
        if not map_path.exists():
            self._log_info(f"year_index_map.json not found at {map_path}. Route-2 augmentation disabled.")
            self.cfg["epoch_resample"] = False
            return
        with open(map_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.year_index_map = meta.get("map", {})
        if not isinstance(self.year_index_map, dict) or len(self.year_index_map) == 0:
            self._log_info("year_index_map is empty. Route-2 augmentation disabled.")
            self.cfg["epoch_resample"] = False
            return
        # Default k from metadata if not explicitly set
        if "k_default" in meta and not self.cfg.get("k_year_augment"):
            self.cfg["k_year_augment"] = int(meta["k_default"])
        self._log_info(f"Loaded year_index_map with {len(self.year_index_map)} nuclides.")

    def _current_epoch_seed(self) -> int:
        """Derive reproducible epoch seed from global seed and Lightning Trainer epoch if available."""
        base = int(self.cfg.get("seed", 412))
        try:
            # When called inside Lightning, self.trainer.current_epoch is available
            epoch = int(getattr(self.trainer, "current_epoch", 0))
        except Exception:
            epoch = 0
        return (base * 1009 + epoch * 9173) % 2_147_483_647

    def _build_epoch_indices(self, dataset: HybridSeqENSDFDataset):
        """For each (Z,A), pick k years (by reproducible RNG) and return concatenated row indices."""
        import numpy as np
        if self.year_index_map is None:
            return list(range(len(dataset.df)))
        rng = np.random.RandomState(self._current_epoch_seed())
        k = max(1, int(self.cfg.get("k_year_augment", 2)))

        # Build mapping from dataset row -> (Z,A,year)
        df = dataset.df
        if not {'Z_int','A_int','year'}.issubset(df.columns):
            # Fallback if columns are missing
            return list(range(len(df)))

        # Group dataset indices by nuclide-year
        # But reuse preprocessed mapping for speed and reproducibility
        selected_indices = []
        # year_index_map keys are "Z_A": { "years": [...], "idx": { "2005":[...], ... } }
        for key, entry in self.year_index_map.items():
            try:
                z_str, a_str = key.split("_")
                z, a = int(z_str), int(a_str)
            except Exception:
                continue
            years = entry.get("years", [])
            if len(years) == 0:
                continue
            chosen_years = rng.choice(years, size=min(k, len(years)), replace=False).tolist()
            idx_map = entry.get("idx", {})
            for y in chosen_years:
                y_str = str(int(y))
                row_ids = idx_map.get(y_str, [])
                selected_indices.extend(row_ids)

        # If nothing selected (shouldn't happen), fallback to all
        if not selected_indices:
            selected_indices = list(range(len(df)))
        return selected_indices

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        if self.cfg.get("epoch_resample", False) and self.year_index_map is not None:
            # Build epoch-specific indices based on original train_df indices from year_index_map.
            # Note: After applying data_subset_ratio, self.train_dataset.df is a subset with reindexed rows.
            # We must remap original indices to current subset positional indices to avoid out-of-bounds.
            raw_indices = self._build_epoch_indices(self.train_dataset)

            # Ensure an original-index column exists for remapping.
            df_current = self.train_dataset.df
            if "_orig_idx" not in df_current.columns:
                # We no longer have the original mapping after prior sampling; create a stable surrogate.
                # Treat current positional indices as the original indices for this epoch to maintain safety.
                df_current = df_current.copy()
                df_current["_orig_idx"] = list(range(len(df_current)))
                self.train_dataset.df = df_current

            # Build map: original index -> current positional index (first occurrence)
            pos_map = {}
            for pos, orig in enumerate(self.train_dataset.df["_orig_idx"].tolist()):
                if orig not in pos_map:
                    pos_map[orig] = pos

            # Remap raw epoch indices into current positional indices; drop those not present
            valid_pos_indices = [pos_map[i] for i in raw_indices if i in pos_map]

            if not valid_pos_indices:
                self._log_info("Route-2 augmentation remap produced empty set; falling back to full train dataset for this epoch.")
                return self._create_dataloader(self.train_dataset, shuffle=True)

            subset_df = self.train_dataset.df.iloc[valid_pos_indices].reset_index(drop=True)
            # Rebuild dataset view (no need to rebuild tokenizer/vocab)
            aug_dataset = HybridSeqENSDFDataset(subset_df, self.tokenizer, ENCODER_FEATURE_COLS)
            return self._create_dataloader(aug_dataset, shuffle=True)

        return self._create_dataloader(self.train_dataset, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return self._create_dataloader(self.val_dataset, shuffle=False)
    
    def _create_dataloader(self, dataset, shuffle: bool) -> DataLoader:
        """Create DataLoader with optimized settings."""
        pad_collate_fn = PadCollateHybrid(pad_id=self.tokenizer.pad_id)
        
        # FIXED: Optimize num_workers with platform compatibility
        num_workers = self.cfg.get("num_workers", 4)
        
        # FIXED: Enable persistent_workers for performance (your environment supports it)
        use_persistent = num_workers > 0  # Enable for all platforms since your setup works
        
        return DataLoader(
            dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=pad_collate_fn,
            pin_memory=True,
            persistent_workers=use_persistent
        )
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
    
    @property
    def pad_id(self) -> int:
        """Get padding token ID."""
        return self.tokenizer.pad_id
    
    @property
    def eos_id(self) -> int:
        """Get EOS token ID."""
        return self.tokenizer.eos_id
    
    @property
    def encoder_feature_dim(self) -> int:
        """Get encoder feature dimension."""
        return len(ENCODER_FEATURE_COLS)  # FIXED: Use actual length instead of hardcoded