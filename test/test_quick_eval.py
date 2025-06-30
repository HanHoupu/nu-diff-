import pytest
import torch
import tempfile
import pathlib
from unittest.mock import patch, MagicMock

def test_quick_eval_smoke():
    """Smoke test to ensure quick_eval script doesn't crash with minimal setup"""
    from nucdiff.cli.quick_eval import main
    
    # Mock all the heavy dependencies
    with patch('nucdiff.cli.quick_eval.build_dataset') as mock_build_dataset, \
         patch('nucdiff.cli.quick_eval.TransformerModel') as mock_model_class, \
         patch('nucdiff.cli.quick_eval.evaluate_mae_multi') as mock_mae, \
         patch('nucdiff.cli.quick_eval.evaluate_rmse') as mock_rmse, \
         patch('nucdiff.cli.quick_eval.evaluate_r2') as mock_r2, \
         patch('torch.load') as mock_load, \
         patch('pathlib.Path.exists') as mock_exists, \
         patch('builtins.open') as mock_open, \
         patch('yaml.safe_load') as mock_yaml_load, \
         patch('sys.argv', ['quick_eval.py', '--ckpt', 'test.pt', '--year', '2023']):
        
        # Setup mocks
        mock_exists.return_value = True
        mock_yaml_load.return_value = {
            'batch_size': 32, 'rank': 8, 'alpha': 16, 'embed_dim': 8,
            'train_backbone_first_year': True, 'start_year': 2004,
            'd_model': 128, 'n_layers': 4, 'n_heads': 4, 'ff_dim': 512
        }
        mock_build_dataset.return_value = (None, None, ({}, {}, 10))
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        mock_load.return_value = {}
        mock_mae.return_value = {"L": 0.1, "G": 0.15, "Q": 0.2, "avg": 0.15}
        mock_rmse.return_value = 0.15
        mock_r2.return_value = 0.8
        
        # This should not raise any exceptions
        try:
            main()
        except Exception as e:
            pytest.fail(f"quick_eval main() raised {e} unexpectedly!") 