#!/usr/bin/env python3
"""
Unified training script for sequence-to-sequence nuclear level prediction.
Powered by PyTorch Lightning.
"""
import torch, warnings, sys, yaml, argparse, logging, json
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
import os

# --- Suppress ALL verbose outputs ---
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
torch.set_float32_matmul_precision('medium')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.nucdiff.model.seq2seq_model import Seq2SeqTransformer
from src.nucdiff.utils.seed import set_seed
from src.nucdiff.data.dataloader import HybridSeqENSDFDataset, PadCollateHybrid
from src.nucdiff.data.tokenizer import LevelTokenizer
from src.nucdiff.utils.callbacks import CustomProgressBar
from src.nucdiff.utils.paths import get_preprocessed_paths, get_run_name

class NucDiffSeq2SeqModule(pl.LightningModule):
    def __init__(self, cfg: dict, vocab_size: int, encoder_feature_dim: int, pad_id: int, delta_e_mean: float, delta_e_std: float):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.pad_id = pad_id
        self.delta_e_mean = delta_e_mean
        self.delta_e_std = delta_e_std
        
        self.model = Seq2SeqTransformer(
            vocab_size=vocab_size,
            encoder_feature_dim=encoder_feature_dim,
            d_model=cfg.get("d_model", 512),
            n_heads=cfg.get("n_heads", 8),
            n_encoder_layers=cfg.get("n_encoder_layers", 6),
            n_decoder_layers=cfg.get("n_decoder_layers", 6),
            ff_dim=cfg.get("ff_dim", 2048),
            dropout=cfg.get("dropout", 0.1)
        )
        self.regr_loss_weight = cfg.get("regr_loss_weight", 1.0)

    def _normalize_delta_e(self, delta_e):
        return (delta_e - self.delta_e_mean) / (self.delta_e_std + 1e-6)

    def forward(self, batch):
        # Scale the delta_E input before passing it to the model
        batch['decoder_input_delta_e_scaled'] = self._normalize_delta_e(batch['decoder_input_delta_e'])
        
        return self.model(
            src=batch['encoder_features'],
            tgt_tokens=batch['decoder_input_tokens'],
            tgt_delta_e=batch['decoder_input_delta_e_scaled'],
            tgt_padding_mask=~batch['attention_mask']
        )

    def _step(self, batch, stage: str):
        class_logits, regr_output_scaled = self(batch)
        target_tokens = batch['target_tokens']
        target_delta_e = batch['target_delta_e']
        attention_mask = batch['attention_mask']
        
        # Scale the regression target for loss calculation
        target_delta_e_scaled = self._normalize_delta_e(target_delta_e)

        # Classification Loss
        loss_class = F.cross_entropy(
            class_logits.view(-1, class_logits.size(-1)),
            target_tokens.view(-1),
            ignore_index=self.pad_id,
            label_smoothing=self.cfg.get("label_smoothing", 0.0)
        )

        # Regression Loss (calculated on scaled values)
        loss_regr = F.mse_loss(
            regr_output_scaled[attention_mask],
            target_delta_e_scaled[attention_mask]
        )
        
        # Total Weighted Loss
        total_loss = loss_class + self.regr_loss_weight * loss_regr
        
        perplexity = torch.exp(loss_class) # Perplexity is derived from classification loss only
        
        # Logging
        self.log(f"{stage}_loss", total_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_loss_class", loss_class, prog_bar=False, on_epoch=True, on_step=False)
        self.log(f"{stage}_loss_regr", loss_regr, prog_bar=False, on_epoch=True, on_step=False)
        # Log perplexity for the callback
        self.log(f"{stage}_perplexity", perplexity, prog_bar=False, on_epoch=True, on_step=False)
        
        if stage == 'train':
            self.log("train_loss_step", total_loss, prog_bar=True, logger=False)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), 
            lr=self.cfg.get("lr_base", 1e-4), 
            weight_decay=self.cfg.get("weight_decay", 0.01)
        )
        return opt

def make_loaders_from_preprocessed(cfg: dict, task: str, debug: bool, overfit: bool):
    """
    Loads preprocessed sequence data and creates PyTorch DataLoaders.
    """
    train_path, val_path = get_preprocessed_paths(cfg, task)
    data_dir = train_path.parent

    logging.info(f"Loading tokenizer from: {data_dir / 'tokenizer_vocab.json'}")
    tokenizer = LevelTokenizer.load(data_dir / 'tokenizer_vocab.json')
    
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    delta_e_scaler_info = metadata.get('delta_e_scaler', {'mean': 0.0, 'std': 1.0})
    delta_e_mean = delta_e_scaler_info['mean']
    delta_e_std = delta_e_scaler_info['std']
    logging.info(f"Loaded delta_E scaler: mean={delta_e_mean:.4f}, std={delta_e_std:.4f}")

    logging.info(f"Loading full training data from: {train_path}")
    train_df = pd.read_feather(train_path)
    logging.info(f"Loading full validation data from: {val_path}")
    val_df = pd.read_feather(val_path)
    
    if overfit:
        logging.warning("--- OVERFITTING MODE ENABLED ---")
        sample_df = train_df.sample(n=100, random_state=cfg.get("seed", 412))
        train_df = sample_df
        val_df = sample_df.copy()
        cfg['dropout'] = 0.0
    elif debug:
        logging.warning("--- DEBUG MODE ENABLED ---")
        train_df = train_df.head(1000)
        val_df = val_df.head(200)

    logging.info(f"Final training set size: {len(train_df):,} nuclides")
    logging.info(f"Final validation set size: {len(val_df):,} nuclides")

    # Define features for the encoder, now with enhanced physics inputs and year
    encoder_feature_cols = [
        'Z_int', 'A_int', 'asym', 
        'is_odd_Z', 'is_odd_N', 
        'delta_magic_z', 'delta_magic_n', 
        'valence_p', 'valence_n',
        'year' # Added year for incremental learning
    ]
    
    train_ds = HybridSeqENSDFDataset(train_df, tokenizer, encoder_feature_cols)
    val_ds = HybridSeqENSDFDataset(val_df, tokenizer, encoder_feature_cols)
    
    pad_collate_fn = PadCollateHybrid(pad_id=tokenizer.pad_id)
    # BUG修复：在Windows上，num_workers > 0 可能会导致 Segmentation Fault
    # 强制设为0以保证稳定性
    workers = 0 
    loader_kw = dict(batch_size=cfg["batch_size"], num_workers=workers, collate_fn=pad_collate_fn)
    
    return (DataLoader(train_ds, shuffle=True, **loader_kw),
            DataLoader(val_ds, shuffle=False, **loader_kw),
            tokenizer,
            delta_e_mean,
            delta_e_std)

def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="NucDiff Seq2Seq Trainer")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--task", type=str, default="SEQ", help="Task to run (must be 'SEQ' for this model).")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    parser.add_argument("--overfit", action="store_true", help="Run in overfitting test mode.")
    args = parser.parse_args()

    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])

    train_loader, val_loader, tokenizer, delta_e_mean, delta_e_std = make_loaders_from_preprocessed(cfg, args.task, args.debug, args.overfit)
    
    module = NucDiffSeq2SeqModule(
        cfg=cfg,
        vocab_size=len(tokenizer),
        encoder_feature_dim=len(train_loader.dataset.encoder_feature_cols),
        pad_id=tokenizer.pad_id,
        delta_e_mean=delta_e_mean,
        delta_e_std=delta_e_std
    )
    
    run_name = get_run_name(cfg, args.task)
    logger = CSVLogger("runs", name=run_name) 

    cb_ckpt = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best-{epoch}-{val_loss:.4f}")
    cb_early = EarlyStopping(monitor="val_loss", patience=cfg.get("early_stop_patience", 10), mode="min")
    cb_lr = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        precision="16-mixed" if cfg.get("use_amp", True) else 32,
        callbacks=[cb_early, cb_ckpt, cb_lr, CustomProgressBar()],
        logger=logger,
        enable_model_summary=True,
        num_sanity_val_steps=0,
        enable_progress_bar=False, # Disable the default progress bar
        accumulate_grad_batches=cfg.get("gradient_accumulation_steps", 1)
    )
    trainer.fit(module, train_loader, val_loader)

if __name__ == "__main__":
    main()
