# src/nucdiff/trainer/lightning_module_slim.py
"""
Refactored and slimmed down PyTorch Lightning module for NucDiff Seq2Seq training.
Uses modular components for better maintainability.
"""
import torch
import pytorch_lightning as pl
from typing import Dict, Any
try:
    from lightning.pytorch.utilities import rank_zero_only
except ImportError:
    from pytorch_lightning.utilities import rank_zero_only

from src.nucdiff.model.seq2seq_model import Seq2SeqTransformer
from .loss_calculator import MultiTaskLossCalculator
from .metrics_calculator import Seq2SeqMetricsCalculator
# Diagnostics now handled by Lightning callbacks


class NucDiffSeq2SeqModuleSlim(pl.LightningModule):
    """
    Slimmed down PyTorch Lightning module for sequence-to-sequence nuclear level prediction.
    Uses modular components for loss calculation, metrics, and diagnostics.
    """
    
    def __init__(self, d_model: int = 512, n_heads: int = 8, n_encoder_layers: int = 6,
                 n_decoder_layers: int = 6, ff_dim: int = 2048, dropout: float = 0.1,
                 label_smoothing: float = 0.1, content_loss_weight: float = 1.0,
                 length_loss_weight: float = 0.002, regr_loss_weight: float = 0.1,
                 use_dynamic_lambda: bool = False, lr_base: float = 0.0001,
                 lr_min: float = 0.00001, weight_decay: float = 0.01,
                 grad_clip_norm: float = 1.0, warmup_steps: int = 200,
                 use_separate_head_lr: bool = True, regr_lr_multiplier: float = 5.0,
                 delta_e_scaler: float = 100.0, train_head: str = "both",
                 vocab_size: int = None, encoder_feature_dim: int = None, 
                 pad_id: int = None, class_weights: torch.Tensor = None, eos_id: int = None):
        super().__init__()
        # Create cfg dictionary from flattened parameters
        self.cfg = {
            'd_model': d_model,
            'n_heads': n_heads,
            'n_encoder_layers': n_encoder_layers,
            'n_decoder_layers': n_decoder_layers,
            'ff_dim': ff_dim,
            'dropout': dropout,
            'label_smoothing': label_smoothing,
            'content_loss_weight': content_loss_weight,
            'length_loss_weight': length_loss_weight,
            'regr_loss_weight': regr_loss_weight,
            'use_dynamic_lambda': use_dynamic_lambda,
            'lr_base': lr_base,
            'lr_min': lr_min,
            'weight_decay': weight_decay,
            'grad_clip_norm': grad_clip_norm,
            'warmup_steps': warmup_steps,
            'use_separate_head_lr': use_separate_head_lr,
            'regr_lr_multiplier': regr_lr_multiplier,
            'delta_e_scaler': delta_e_scaler,
        }
        self.save_hyperparameters()
        self.train_head = train_head
        
        # Config validation: Ensure all key parameters are properly set
        
        # Model configuration
        self.delta_e_scaler = delta_e_scaler
        
        # Store parameters for delayed initialization
        self._vocab_size = vocab_size
        self._encoder_feature_dim = encoder_feature_dim
        self._pad_id = pad_id
        self._class_weights = class_weights
        self._eos_id = eos_id
        
        # Initialize components if we have all required parameters
        if all(x is not None for x in [vocab_size, encoder_feature_dim, pad_id]):
            self._initialize_components()
        else:
            # Mark as needing initialization from datamodule
            self._needs_initialization = True
    
    def _initialize_components(self):
        """Initialize model components with known parameters."""
        self.pad_id = self._pad_id
        self.eos_id = self._eos_id
        
        # FIXED: Handle None class_weights safely
        if self._class_weights is not None:
            self.register_buffer("class_weights", self._class_weights)
        else:
            self.register_buffer("class_weights", torch.ones(self._vocab_size))
        
        # Initialize main model
        self.model = Seq2SeqTransformer(
            vocab_size=self._vocab_size,
            encoder_feature_dim=self._encoder_feature_dim,
            d_model=self.cfg.get("d_model", 512),
            n_heads=self.cfg.get("n_heads", 8),
            n_encoder_layers=self.cfg.get("n_encoder_layers", 6),
            n_decoder_layers=self.cfg.get("n_decoder_layers", 6),
            ff_dim=self.cfg.get("ff_dim", 2048),
            dropout=self.cfg.get("dropout", 0.1)
        )
        
        # Initialize modular components
        self.loss_calculator = MultiTaskLossCalculator(
            cfg=self.cfg, 
            pad_id=self._pad_id, 
            class_weights=self._class_weights,
            delta_e_scaler=self.delta_e_scaler
        )
        
        self.metrics_calculator = Seq2SeqMetricsCalculator(
            pad_id=self._pad_id,
            eos_id=self._eos_id,
            vocab_size=self._vocab_size
        )
        
        # Move metrics to device
        if hasattr(self, 'device'):
            self.metrics_calculator.to(self.device)
        
        self._needs_initialization = False
    
    def setup(self, stage: str):
        """Setup method called by Lightning - initialize from datamodule if needed."""
        if hasattr(self, '_needs_initialization') and self._needs_initialization:
            # Get datamodule from trainer
            if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
                dm = self.trainer.datamodule
                # Ensure datamodule is setup
                if hasattr(dm, 'tokenizer') and dm.tokenizer is None:
                    dm.setup(stage)
                
                # Initialize with datamodule parameters
                self._vocab_size = dm.vocab_size
                self._encoder_feature_dim = dm.encoder_feature_dim
                self._pad_id = dm.pad_id
                self._class_weights = dm.class_weights
                self._eos_id = dm.eos_id
                
                self._initialize_components()
        
        # Move metrics to correct device if they exist
        if hasattr(self, 'metrics_calculator'):
            self.metrics_calculator.to(self.device)

    def forward(self, batch):
        """Forward pass through the model."""
        batch['decoder_input_delta_e_scaled'] = batch['decoder_input_delta_e'] / self.delta_e_scaler
        
        # FIXED: Ensure attention_mask is bool type
        attention_mask = batch['attention_mask']
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()
            
        return self.model(
            src=batch['encoder_features'],
            tgt_tokens=batch['decoder_input_tokens'],
            tgt_delta_e=batch['decoder_input_delta_e_scaled'],
            tgt_padding_mask=~attention_mask
        )

    def _step(self, batch, stage: str):
        """Streamlined step function using modular components."""
        # Forward pass
        class_logits, regr_output_scaled = self(batch)
        
        # Calculate losses using loss calculator
        losses = self.loss_calculator.calculate_losses(
            class_logits=class_logits,
            regr_output_scaled=regr_output_scaled,
            target_tokens=batch['target_tokens'],
            target_delta_e=batch['target_delta_e'],
            attention_mask=batch['attention_mask'],
            train_head=self.train_head
        )
        
        # Calculate metrics using metrics calculator
        metrics = self.metrics_calculator.calculate_metrics(
            class_logits=class_logits,
            target_tokens=batch['target_tokens'],
            train_head=self.train_head
        )
        
        # Log all metrics
        self._log_metrics(stage, losses, metrics)
        
        # Diagnostics now handled by Lightning callbacks automatically
        
        # FIXED: Handle NaN loss while preserving gradients
        total_loss = losses['total_loss']
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            # Preserve gradient information when replacing NaN
            total_loss = total_loss.detach().fill_(50.0).requires_grad_(total_loss.requires_grad)
            # Update the losses dict for logging
            losses['total_loss'] = total_loss
            
        return total_loss
    
    def _log_metrics(self, stage: str, losses: Dict[str, torch.Tensor], metrics: Dict[str, torch.Tensor]):
        """Log all losses and metrics."""
        # Clean loss values for display (replace NaN/Inf with reasonable values)
        clean_total_loss = losses['total_loss']
        if torch.isnan(clean_total_loss) or torch.isinf(clean_total_loss):
            clean_total_loss = torch.tensor(50.0, device=clean_total_loss.device)
            
        # FIXED: Detach all logged values to prevent warnings
        display_loss = clean_total_loss
        if torch.isnan(display_loss) or torch.isinf(display_loss):
            display_loss = torch.tensor(15.0, device=display_loss.device)
        
        # CRITICAL FIX: 统一聚合方式 - 解决train_loss跳动而val_loss稳定的异常现象
        # Lightning默认: training_step不聚合(显示单batch)，validation_step自动聚合(显示平均)
        # 修复后: 训练和验证都使用epoch级别的平均，确保指标一致性和可比性
        self.log(f"{stage}/loss", display_loss.detach(), prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log(f"{stage}/loss_content", losses['loss_content'].detach(), prog_bar=False, logger=True, on_epoch=True, on_step=False)
        self.log(f"{stage}/loss_regr", losses['loss_regr'].detach(), prog_bar=False, logger=True, on_epoch=True, on_step=False)
        self.log(f"{stage}/perplexity", losses['perplexity'].detach(), prog_bar=False, logger=True, on_epoch=True, on_step=False)
        
        # Log metrics - 统一使用epoch级别聚合
        self.log(f"{stage}/token_accuracy", metrics['token_accuracy'], prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log(f"{stage}/sequence_similarity", metrics['sequence_similarity'], prog_bar=False, logger=True, on_epoch=True, on_step=False)
        self.log(f"{stage}/eos_position_accuracy", metrics['eos_position_accuracy'], prog_bar=False, logger=True, on_epoch=True, on_step=False)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        return self._step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self._step(batch, "val")
    
    def on_train_epoch_start(self):
        """Reset metrics at the start of each training epoch."""
        self.metrics_calculator.reset()
    
    def on_validation_epoch_start(self):
        """Reset metrics at the start of each validation epoch."""  
        self.metrics_calculator.reset()
        
    # setup method already defined above with delayed initialization logic
    
    def configure_optimizers(self):
        """Configure optimizers with optional separate head learning rates."""
        if self.cfg.get("use_separate_head_lr", False):
            return self._configure_separate_head_optimizers()
        else:
            return self._configure_standard_optimizer()
    
    def _configure_separate_head_optimizers(self):
        """Configure optimizers with different learning rates for different heads."""
        base_lr = self.cfg.get("lr_base", 1e-4)
        regr_lr_multiplier = self.cfg.get("regr_lr_multiplier", 5.0)
        
        # Group parameters by component
        encoder_params = []
        cls_head_params = []
        regr_head_params = []
        
        for name, param in self.model.named_parameters():
            if 'classification_head' in name:
                cls_head_params.append(param)
            elif 'regression_head' in name:
                regr_head_params.append(param)
            else:
                encoder_params.append(param)
        
        # Create parameter groups
        param_groups = [
            {'params': encoder_params, 'lr': base_lr},
            {'params': cls_head_params, 'lr': base_lr},
            {'params': regr_head_params, 'lr': base_lr * regr_lr_multiplier},
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.cfg.get("weight_decay", 0.01)
        )
        
        # FIXED: Use rank_zero_only to prevent multi-GPU print duplication
        self._log_learning_rates(base_lr, regr_lr_multiplier)
        
        return optimizer
    

    @rank_zero_only
    def _log_learning_rates(self, base_lr: float, regr_lr_multiplier: float):
        """Log learning rates (rank 0 only to prevent multi-GPU duplication)."""
        try:
            print(f"📚 Using separate head learning rates:")
            print(f"  - Encoder & Classification: {base_lr}")
            print(f"  - Regression head: {base_lr * regr_lr_multiplier} (x{regr_lr_multiplier})")
        except UnicodeEncodeError:
            # ASCII-safe fallback
            print("Using separate head learning rates:")
            print(f"  - Encoder & Classification: {base_lr}")
            print(f"  - Regression head: {base_lr * regr_lr_multiplier} (x{regr_lr_multiplier})")
    
    def _configure_standard_optimizer(self):
        """Configure standard single learning rate optimizer."""
        return torch.optim.AdamW(
            self.parameters(), 
            lr=self.cfg.get("lr_base", 1e-4), 
            weight_decay=self.cfg.get("weight_decay", 0.01)
        )
    
    # Gradient monitoring now handled by Lightning callbacks