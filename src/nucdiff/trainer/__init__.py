# src/nucdiff/trainer/__init__.py
"""
Training module for NucDiff Seq2Seq models.
Contains modular components extracted from the monolithic train.py.
"""

# FIXED: Import only existing modules (removed deleted files)
from .lightning_module_slim import NucDiffSeq2SeqModuleSlim
from .data_module import ENSDFDataModule
from .loss_calculator import MultiTaskLossCalculator
from .metrics_calculator import Seq2SeqMetricsCalculator

__all__ = [
    "NucDiffSeq2SeqModuleSlim", 
    "ENSDFDataModule",
    "MultiTaskLossCalculator",
    "Seq2SeqMetricsCalculator"
]