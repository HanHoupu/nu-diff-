import argparse
import json
import logging
import pandas as pd
import lightgbm as lgb
import yaml
from sklearn.metrics import r2_score
from src.nucdiff.utils.paths import get_preprocessed_paths
import os
from pathlib import Path

# Set the project root and change the current working directory
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
logging.info(f"Working directory set to: {os.getcwd()}")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    """
    Train and evaluate a LightGBM model.
    """
    logging.info("Starting LightGBM training process...")
    
    # --- 0. Load Config ---
    try:
        with open(args.config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {args.config_path}")
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {args.config_path}")
        return
    except Exception as e:
        logging.error(f"Error loading YAML configuration: {e}")
        return


    # --- 1. Load Data ---
    try:
        logging.info("Loading preprocessed data and metadata...")
        train_path, val_path = get_preprocessed_paths(cfg, args.task)
        metadata_path = train_path.parent / "metadata.json"

        train_df = pd.read_feather(train_path)
        val_df = pd.read_feather(val_path)

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        numeric_cols = metadata['numeric_cols']
        target_col = cfg['target_col']
        
        logging.info(f"Data loaded successfully. Found {len(numeric_cols)} features.")
        logging.info(f"Features: {numeric_cols}")

    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        logging.error("Please run the preprocessing script first (e.g., `python preprocess.py`)")
        return

    X_train = train_df[numeric_cols]
    y_train = train_df[target_col]
    X_val = val_df[numeric_cols]
    y_val = val_df[target_col]

    if args.mode == 'base':
        base_features = ['z', 'n']
        logging.info("--- BASE MODE ACTIVATED ---")
        # Ensure base features are present in the dataframe columns
        if not all(f in X_train.columns for f in base_features):
             logging.error(f"Base features {base_features} not found in the training data columns: {X_train.columns.tolist()}")
             return
        X_train = X_train[base_features]
        X_val = X_val[base_features]
        numeric_cols = base_features # For feature importance display
        logging.info(f"Using minimal feature set: {base_features}")


    # --- 2. Train Model ---
    logging.info("Training LightGBM model...")
    
    # LightGBM parameters - chosen for a good balance of speed and accuracy
    params = {
        'objective': 'regression_l1',  # MAE loss, often more robust to outliers
        'metric': 'rmse',
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'n_jobs': -1,  # Use all available cores
        'seed': 42,
        'boosting_type': 'gbdt',
    }

    model = lgb.LGBMRegressor(**params)

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=True)])

    # --- 3. Evaluate Model ---
    logging.info("Evaluating model...")
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)

    logging.info("="*30)
    logging.info(f"LightGBM Model Evaluation Complete")
    logging.info(f"Validation R² Score: {r2:.4f}")
    logging.info("="*30)
    
    # --- 4. Feature Importance ---
    logging.info("Top 10 Feature Importances:")
    feature_importance_df = pd.DataFrame({
        'feature': numeric_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance_df.head(10).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a LightGBM model on nuclear data.")
    parser.add_argument('--config_path', type=str, default="configs/base.yaml", help='Path to the YAML configuration file.')
    parser.add_argument('--task', type=str, default="L", help='Task identifier (e.g., "L" or "S_n").')
    parser.add_argument('--mode', type=str, default=None, help='Set to "base" to use only z and n features.')
    
    args = parser.parse_args()
    main(args) 