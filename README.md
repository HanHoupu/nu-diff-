# NucDiff: Incremental Learning for Nuclear Data

This project implements an incremental learning pipeline using a Transformer-based model to continuously learn from and predict ENSDF nuclear structure data.

## Core Concept: Incremental Learning

The primary technical approach is a two-stage incremental learning strategy:

1.  **Base Model Pre-training**: A foundational Transformer model is first trained on a large, general dataset (e.g., data from 2004-2010). This stage involves training all model parameters to learn the fundamental patterns and physics within the data.
2.  **Incremental Fine-tuning with LoRA**: Once a robust base model is established, it is fine-tuned on newer, incremental data (e.g., data from 2011 onwards). This is done efficiently using **LoRA (Low-Rank Adaptation)**, where the original base model weights are frozen, and only a small number of lightweight "adapter" parameters are trained. This allows the model to absorb new information quickly and cheaply without forgetting its foundational knowledge.

## File-by-File Guide

Here is a breakdown of the key files and directories in this repository.

### `configs/`
This directory holds all configuration files (`.yaml`) that control training and model parameters.

-   `temporal.yaml`: **(Primary Base Model Config)** The main configuration for pre-training the base Transformer model. It defines the model architecture, learning rate, optimizer settings, and data sources.
-   `lora.yaml`: **(Primary LoRA Config)** The configuration for LoRA fine-tuning. It inherits settings from a base config but crucially enables LoRA by setting `use_lora: true` and defines LoRA-specific parameters like `rank`.
-   `base.yaml`, `default.yaml`: Other example or legacy configuration files.

### `src/`
The main source code for the project.

-   **`src/nucdiff/train.py`**: **(Main Training Script)** The unified entry point for all training tasks, powered by PyTorch Lightning. It handles data loading, model initialization, the training/validation loop, and logging.
-   **`src/nucdiff/model/transformer_model.py`**: **(Core Model Definition)** Defines the `TransformerModel` architecture. A critical parameter here is `use_lora`, which acts as the switch between pre-training the full model and performing a LoRA fine-tune.
-   `src/nucdiff/model/lora.py`: Implements the `LoRALayer`, which replaces standard `nn.Linear` layers during fine-tuning to enable low-rank adaptation.
-   `src/nucdiff/data/dataloader.py`: Contains the `ENSDFDataset` class, which handles loading, preprocessing, and batching of the nuclear data from Feather files.
-   `src/nucdiff/utils/`: Contains utility scripts for seeding, callbacks, and path management.

### Root-Level Scripts
Scripts for data preparation and preprocessing.

-   `clean.py`: A script to clean a raw ENSDF data file, filtering for specific record types (Q, L, G).
-   `divide.py`: Splits a single cleaned ENSDF file into separate files based on record type.
-   `check_missing.py`: A utility to check for and report missing values in the processed data tables.

### `runs/`
This is the output directory for all experiments. Each time `train.py` is executed, it creates a new subdirectory here containing:
-   `config.yaml`: A snapshot of the configuration used for that run.
-   `logs/`: CSV logs of metrics like loss, RMSE, R2 over time.
-   `checkpoints/`: Saved model weights (`.ckpt` files), typically storing the best-performing model based on validation loss.

### `test/`
Contains unit and integration tests for different components of the project, such as the dataloader and model forward pass.
