import os
from pathlib import Path

# D:\VS CODE W\src\nucdiff\utils\paths.py
# .resolve() -> D:\VS CODE W\src\nucdiff\utils\paths.py
# .parents[0] -> D:\VS CODE W\src\nucdiff\utils
# .parents[1] -> D:\VS CODE W\src\nucdiff
# .parents[2] -> D:\VS CODE W\src
# .parents[3] -> D:\VS CODE W (This is the correct project root)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Define paths relative to the project root
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CONFIG_DIR = PROJECT_ROOT / "configs"
RUNS_DIR = PROJECT_ROOT / "runs"
PREPROCESSED_DATA_DIR = PROJECT_ROOT / "preprocessed_data"

# Ensure the output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_preprocessed_paths(cfg: dict, task: str) -> tuple[Path, Path]:
    """Generates consistent, absolute paths for preprocessed train/val files."""
    # Ensure the base output directory exists in the project root
    output_dir = PROJECT_ROOT / "preprocessed_data"
    output_dir.mkdir(exist_ok=True)
    
    seed = cfg["seed"]
    val_frac = 1.0 - cfg.get("train_frac", 0.8) 
    start_year = cfg.get("base_start_year", "NA")
    end_year = cfg.get("base_end_year", "NA")
    
    base_filename = f"task_{task}_years_{start_year}-{end_year}_seed_{seed}_frac_{val_frac:.2f}"
    train_path = output_dir / f"{base_filename}_train.feather"
    val_path = output_dir / f"{base_filename}_val.feather"
    
    return train_path, val_path

def get_run_name(cfg: dict, task: str) -> str:
    """Generates a consistent run name for logging and checkpoints."""
    s_year = cfg["base_start_year"]
    e_year = cfg["base_end_year"]
    return f"pl_{task}_{s_year}-{e_year}" 