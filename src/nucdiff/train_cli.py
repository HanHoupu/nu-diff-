#!/usr/bin/env python3
"""ULTRA-SIMPLE: 5-line nuclear physics trainer with full Lightning power."""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from lightning.pytorch.cli import LightningCLI
except ImportError:
    from pytorch_lightning.cli import LightningCLI
from src.nucdiff.trainer.lightning_module_slim import NucDiffSeq2SeqModuleSlim  
from src.nucdiff.trainer.data_module import ENSDFDataModule

if __name__ == "__main__":
    LightningCLI(NucDiffSeq2SeqModuleSlim, ENSDFDataModule, seed_everything_default=412)