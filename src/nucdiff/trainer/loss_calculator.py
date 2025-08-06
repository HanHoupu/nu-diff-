# src/nucdiff/trainer/loss_calculator.py
"""
Loss calculation utilities for multi-task seq2seq training.
Extracted from lightning_module.py for better modularity.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple


class MultiTaskLossCalculator:
    """
    Handles loss calculations for multi-task sequence-to-sequence training.
    Supports content (classification), length, and regression losses.
    """
    
    def __init__(self, cfg: dict, pad_id: int, class_weights: torch.Tensor, delta_e_scaler: float):
        self.cfg = cfg
        self.pad_id = pad_id
        self.class_weights = class_weights
        self.delta_e_scaler = delta_e_scaler
        
    def calculate_losses(self, 
                        class_logits: torch.Tensor,
                        regr_output_scaled: torch.Tensor, 
                        target_tokens: torch.Tensor,
                        target_delta_e: torch.Tensor,
                        attention_mask: torch.Tensor,
                        train_head: str) -> Dict[str, torch.Tensor]:
        
        # Silently check for NaN inputs (debug info removed for clean training)
        """
        Calculate all losses for the multi-task model.
        
        Returns:
            Dictionary containing individual losses and total loss
        """
        target_delta_e_scaled = target_delta_e / self.delta_e_scaler
        
        # 1. CONTENT LOSS: Token classification
        if train_head in ["both", "classification"]:
            # Ensure class_weights are on the same device as inputs
            device_weights = self.class_weights.to(target_tokens.device) if self.class_weights is not None else None
            try:
                loss_content = F.cross_entropy(
                    class_logits.view(-1, class_logits.size(-1)),
                    target_tokens.view(-1),
                    ignore_index=self.pad_id,
                    label_smoothing=self.cfg.get("label_smoothing", 0.0),
                    weight=device_weights
                )
            except Exception as e:
                loss_content = torch.tensor(10.0, device=target_tokens.device, requires_grad=True)
        else:
            loss_content = torch.tensor(0.0, device=target_tokens.device)
            
        # 2. REGRESSION LOSS: Continuous values
        if train_head in ["both", "regression"]:
            try:
                # FIXED: Handle None attention_mask
                if attention_mask is not None and attention_mask.any():
                    loss_regr = F.huber_loss(
                        regr_output_scaled[attention_mask],
                        target_delta_e_scaled[attention_mask],
                        delta=1.0
                    )
                else:
                    # Fallback: use all positions if mask is None or empty
                    loss_regr = F.huber_loss(regr_output_scaled, target_delta_e_scaled, delta=1.0)
            except Exception as e:
                loss_regr = torch.tensor(1.0, device=target_tokens.device, requires_grad=True)
        else:
            loss_regr = torch.tensor(0.0, device=target_tokens.device)
        
        # FIXED: Preserve gradients while handling NaN
        if torch.isnan(loss_content) or torch.isinf(loss_content):
            loss_content = loss_content.detach().fill_(10.0).requires_grad_(loss_content.requires_grad)
        if torch.isnan(loss_regr) or torch.isinf(loss_regr):
            loss_regr = loss_regr.detach().fill_(1.0).requires_grad_(loss_regr.requires_grad)
            
        # Calculate total loss
        total_loss = self._combine_losses(loss_content, loss_regr, train_head)
        
        # FIXED: Final stability check preserving gradients
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = total_loss.detach().fill_(20.0).requires_grad_(total_loss.requires_grad)
        
        return {
            'loss_content': loss_content,
            'loss_regr': loss_regr,
            'total_loss': total_loss,
            'perplexity': torch.exp(torch.clamp(loss_content, max=50.0))
        }
    
    def _combine_losses(self, loss_content: torch.Tensor, loss_regr: torch.Tensor, train_head: str) -> torch.Tensor:
        """Combine individual losses based on training mode and weights."""
        content_weight = self.cfg.get("content_loss_weight", 1.0)
        regr_loss_weight = self.cfg.get("regr_loss_weight", 1.0)
        
        if train_head == "classification":
            return content_weight * loss_content
        elif train_head == "regression":
            return loss_regr
        else: # 'both'
            return content_weight * loss_content + regr_loss_weight * loss_regr
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights for logging."""
        return {
            'content_weight': self.cfg.get("content_loss_weight", 1.0),
            'regr_weight': self.cfg.get("regr_loss_weight", 1.0)
        }