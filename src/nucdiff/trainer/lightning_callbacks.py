# src/nucdiff/trainer/lightning_callbacks.py
"""
Lightning callbacks replacing manual diagnostics.
Uses Lightning's built-in monitoring instead of hand-written logging.
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
from typing import Dict, Any


class NuclearPhysicsDiagnostics(Callback):
    """
    Custom callback for nuclear physics specific diagnostics.
    Replaces manual TrainingDiagnostics with Lightning callback pattern.
    """
    
    def __init__(self, print_frequency: int = 100):
        self.print_frequency = print_frequency
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Print diagnostics at validation epoch end."""
        current_epoch = trainer.current_epoch
        
        if current_epoch > 0 and current_epoch % self.print_frequency == 0:
            self._print_diagnostics(trainer, pl_module, current_epoch)
    
    def _print_diagnostics(self, trainer: pl.Trainer, pl_module: pl.LightningModule, epoch: int):
        """Print detailed loss and metrics breakdown."""
        # Get metrics from trainer logs
        metrics = trainer.logged_metrics
        
        print(f"\n=== NUCLEAR PHYSICS DIAGNOSTICS Epoch {epoch} ===")
        
        # Loss breakdown
        if 'val/loss_content' in metrics:
            print(f"📝 Content loss (tokens): {metrics['val/loss_content']:.4f}")
        if 'val/loss_length' in metrics:
            print(f"📏 Length loss (sequence): {metrics['val/loss_length']:.4f}")
        if 'val/loss_regr' in metrics:
            print(f"📊 Regression loss: {metrics['val/loss_regr']:.4f}")
        
        # Metrics
        if 'val/token_accuracy' in metrics:
            token_acc = metrics['val/token_accuracy'].item()
            print(f"🎯 Token accuracy: {token_acc*100:.1f}% (target: >95%)")
        
        if 'val/sequence_similarity' in metrics:
            seq_sim = metrics['val/sequence_similarity'].item()
            print(f"📝 Sequence similarity: {seq_sim*100:.1f}% (avg token match rate)")
        
        if 'val/eos_position_accuracy' in metrics:
            eos_acc = metrics['val/eos_position_accuracy'].item()
            print(f"🔚 EOS position accuracy: {eos_acc*100:.1f}% (correct length)")
        
        if 'val/length_accuracy' in metrics:
            length_acc = metrics['val/length_accuracy'].item()
            print(f"📏 Length accuracy: {length_acc*100:.1f}% (15% relative tolerance)")
        
        # Overfitting diagnosis
        self._print_overfitting_diagnosis(metrics)
    
    def _print_overfitting_diagnosis(self, metrics: Dict[str, torch.Tensor]):
        """Print overfitting diagnosis based on current metrics."""
        if 'val/token_accuracy' not in metrics:
            return
            
        token_acc = metrics['val/token_accuracy'].item()
        seq_sim = metrics.get('val/sequence_similarity', torch.tensor(0.0)).item()
        length_acc = metrics.get('val/length_accuracy', torch.tensor(0.0)).item()
        
        if token_acc < 0.5:
            print("🚨 DIAGNOSIS: Token accuracy very low - model struggling with basic classification")
        elif token_acc > 0.95 and seq_sim > 0.85:
            print("✅ DIAGNOSIS: Excellent performance - high token accuracy and sequence similarity!")
        elif token_acc > 0.8 and seq_sim < 0.6:
            print("⚠️ DIAGNOSIS: Good token accuracy but low sequence similarity - check length prediction")
        elif length_acc < 0.5:
            print("📏 DIAGNOSIS: Length prediction struggling - check length head initialization")


class GradientComponentMonitor(Callback):
    """
    Monitor gradient norms for different model components.
    Replaces manual gradient logging with Lightning callback.
    """
    
    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer):
        """Monitor gradient norms before optimizer step."""
        # Calculate component-wise gradient norms
        total_norm = 0.0
        encoder_norm = 0.0
        cls_head_norm = 0.0
        regr_head_norm = 0.0
        
        for name, p in pl_module.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
                if 'classification_head' in name:
                    cls_head_norm += param_norm.item() ** 2
                elif 'regression_head' in name:
                    regr_head_norm += param_norm.item() ** 2
                else:
                    encoder_norm += param_norm.item() ** 2
        
        # Normalize
        total_norm = total_norm ** 0.5
        encoder_norm = encoder_norm ** 0.5
        cls_head_norm = cls_head_norm ** 0.5
        regr_head_norm = regr_head_norm ** 0.5
        
        # Log to Lightning
        pl_module.log("grad_norm", total_norm, prog_bar=False, logger=True)
        pl_module.log("grad_norm_encoder", encoder_norm, prog_bar=False, logger=True)
        pl_module.log("grad_norm_cls_head", cls_head_norm, prog_bar=False, logger=True)
        pl_module.log("grad_norm_regr_head", regr_head_norm, prog_bar=False, logger=True)
        
        # Calculate gradient ratio for multi-task balance monitoring
        if regr_head_norm > 0 and cls_head_norm > 0:
            grad_ratio = max(cls_head_norm, regr_head_norm) / min(cls_head_norm, regr_head_norm)
            pl_module.log("grad_ratio_cls_regr", grad_ratio, prog_bar=False, logger=True)


def get_lightning_callbacks(cfg: dict) -> list:
    """
    Get all Lightning callbacks to replace manual diagnostics.
    This replaces the entire TrainingDiagnostics class with Lightning-native callbacks.
    """
    callbacks = []
    
    # Custom diagnostics
    callbacks.append(NuclearPhysicsDiagnostics(print_frequency=cfg.get("diagnostic_frequency", 100)))
    
    # Gradient monitoring
    callbacks.append(GradientComponentMonitor())
    
    # Built-in Lightning callbacks
    callbacks.extend([
        pl.callbacks.RichProgressBar(),
        pl.callbacks.ModelSummary(max_depth=2),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        pl.callbacks.EarlyStopping(
            monitor='val/loss',
            patience=cfg.get("early_stop_patience", 15),
            min_delta=cfg.get("min_delta", 0.0001),
            mode='min'
        ),
        pl.callbacks.ModelCheckpoint(
            monitor='val/loss',
            save_top_k=3,
            mode='min',
            filename='nucdiff-{epoch:02d}-{val_loss:.4f}'
        )
    ])
    
    return callbacks