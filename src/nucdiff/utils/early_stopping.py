#!/usr/bin/env python3
"""
Early stopping mechanism for training
"""
import numpy as np

class EarlyStopper:
    """
    Early stopping mechanism to prevent overfitting
    """
    def __init__(self, patience=10, min_delta=0.01, mode='min'):
        """
        Args:
            patience: Number of epochs to tolerate without improvement
            min_delta: Minimum improvement threshold
            mode: 'min' means lower metric is better, 'max' means higher metric is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.stop_reason = None
        
    def step(self, val_score):
        """
        Check if early stopping is needed
        
        Args:
            val_score: Current validation metric
            
        Returns:
            bool: True means training should stop
        """
        if self.best_score is None:
            self.best_score = val_score
            return False
            
        if self.mode == 'min':
            # Lower metric is better
            if val_score < self.best_score - self.min_delta:
                self.best_score = val_score
                self.counter = 0
            else:
                self.counter += 1
        else:
            # Higher metric is better
            if val_score > self.best_score + self.min_delta:
                self.best_score = val_score
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            self.stop_reason = f"Early stop triggered after {self.patience} epochs without improvement"
            return True
            
        return False
        
    def get_stop_reason(self):
        """Get stop reason"""
        return self.stop_reason
        
    def reset(self):
        """Reset early stopper state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.stop_reason = None 