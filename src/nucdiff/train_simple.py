#!/usr/bin/env python3
"""
Simple training script - bypasses LightningCLI complexity
Uses direct Lightning Trainer with our fixed modules
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelSummary, LearningRateMonitor, EarlyStopping, ModelCheckpoint
)

from src.nucdiff.trainer.lightning_module_slim import NucDiffSeq2SeqModuleSlim  
from src.nucdiff.trainer.data_module import ENSDFDataModule
from src.nucdiff.utils.callbacks import CustomProgressBar

def main():
    # Load config with UTF-8 encoding
    cfg_path = "configs/temporal.yaml"
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # Override max_epochs for quick test
    cfg['max_epochs'] = 10
    
    print(f"🚀 Starting training with config: {cfg_path}")
    print(f"📊 Max epochs: {cfg['max_epochs']}")
    
    # Add missing config keys that DataModule expects
    cfg.setdefault('task', 'SEQ')
    cfg.setdefault('years_range', '2005-2010')
    cfg.setdefault('train_frac', 0.2)
    cfg.setdefault('force_rebuild_vocab', False)
    cfg.setdefault('num_workers', 4)
    
    # Create data module with correct parameters
    data_module = ENSDFDataModule(
        cfg=cfg,
        overfit=False,
        overfit_samples=5
    )
    
    # Setup data to get dimensions
    data_module.setup('fit')
    
    # Create model
    model = NucDiffSeq2SeqModuleSlim(
        cfg=cfg,
        vocab_size=data_module.vocab_size,
        encoder_feature_dim=data_module.encoder_feature_dim,
        pad_id=data_module.pad_id,
        class_weights=data_module.class_weights,
        train_head='both',
        eos_id=data_module.eos_id
    )
    
    # Create callbacks including custom progress bar
    callbacks = [
        CustomProgressBar(),  # Our custom non-spam progress bar
        ModelSummary(max_depth=2),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor='val/loss',
            patience=25,
            min_delta=0.0001,
            mode='min'
        ),
        ModelCheckpoint(
            monitor='val/loss',
            save_top_k=3,
            mode='min',
            filename='nucdiff-{epoch:02d}-{val_loss:.4f}'
        )
    ]
    
    # Create trainer with custom progress bar
    trainer = pl.Trainer(
        max_epochs=cfg['max_epochs'],
        accelerator='auto',
        devices='auto',
        precision=32,
        gradient_clip_val=cfg.get('grad_clip_norm', 1.0),
        log_every_n_steps=cfg.get('log_every', 40),
        enable_checkpointing=True,
        enable_progress_bar=False,  # FIXED: Disable default to use our custom one
        enable_model_summary=False,  # FIXED: Disable to avoid conflicts with our custom ModelSummary
        callbacks=callbacks,
        # FIXED: Additional settings to prevent output conflicts
        logger=True,  # Keep logging enabled
        deterministic=False,  # Allow for better performance
    )
    
    # Start training
    print("🎯 Starting Lightning training...")
    trainer.fit(model, datamodule=data_module)
    
    print("✅ Training completed!")

if __name__ == "__main__":
    main()