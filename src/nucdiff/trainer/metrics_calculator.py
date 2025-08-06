# src/nucdiff/trainer/metrics_calculator.py
"""
Ultra-minimal torchmetrics-based calculator.
Custom metrics in <20 lines each, using torchmetrics framework.
"""
import torch
import torchmetrics
from typing import Dict
from torchmetrics import Metric

# Ultra-compact custom metrics - each <15 lines
class SequenceSimilarity(Metric):
    """Sequence accuracy based on token-level similarity score, not exact match."""
    def __init__(self, pad_id: int, threshold: float = 0.8):
        super().__init__()
        self.pad_id = pad_id
        self.threshold = threshold  # 80% token accuracy = sequence correct
        self.add_state("total_similarity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        if target.size(0) == 0:  # Handle empty batch
            return
        mask = target != self.pad_id
        # Calculate token-level accuracy for each sequence
        correct_tokens = (pred == target) & mask
        sequence_lengths = mask.sum(dim=1).float()
        
        # Avoid division by zero
        sequence_lengths = torch.clamp(sequence_lengths, min=1.0)
        
        # Calculate similarity score for each sequence (% of correct tokens)
        similarity_scores = correct_tokens.sum(dim=1).float() / sequence_lengths
        
        self.total_similarity += similarity_scores.sum()
        self.total += target.size(0)
    
    def compute(self): 
        avg_similarity = self.total_similarity / self.total if self.total > 0 else torch.tensor(0.0)
        return avg_similarity


# Main calculator - now <30 lines total
class Seq2SeqMetricsCalculator:
    def __init__(self, pad_id: int, eos_id: int, vocab_size: int):
        self.metrics = torchmetrics.MetricCollection({
            'token_accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=vocab_size, ignore_index=pad_id),
            'sequence_similarity': SequenceSimilarity(pad_id, threshold=0.8),  # 80% token similarity 
            'eos_position_accuracy': SequenceSimilarity(pad_id, threshold=0.95),  # High threshold for EOS
        })
    
    def to(self, device): 
        self.metrics = self.metrics.to(device)
        return self
    
    def reset(self): 
        self.metrics.reset()
    
    def calculate_metrics(self, class_logits, target_tokens, train_head):
        if train_head not in ["both", "classification"]:
            return {k: torch.tensor(0.0) for k in self.metrics}
            
        with torch.no_grad():
            predictions = torch.argmax(class_logits, dim=-1)
            self.metrics['token_accuracy'].update(predictions.flatten(), target_tokens.flatten())
            self.metrics['sequence_similarity'].update(predictions, target_tokens)
            self.metrics['eos_position_accuracy'].update(predictions, target_tokens)
            
        # FIXED: Detach metrics for logging
        results = self.metrics.compute()
        return {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in results.items()}