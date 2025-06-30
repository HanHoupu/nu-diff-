import pytest
import torch
from nucdiff.model import TransformerModel

def test_transformer_smoke():
    """Smoke test to ensure TransformerModel can initialize and forward pass"""
    # Mock config
    cfg = {
        "d_model": 64,  # smaller for testing
        "n_layers": 2,  # fewer layers for testing
        "n_heads": 4,
        "ff_dim": 128,
        "rank": 4,
        "alpha": 8,
        "task_weights": {"L": 1.0, "G": 1.0, "Q": 1.0}
    }
    
    # Mock vocabularies
    elem2idx = {"H": 0, "He": 1, "Li": 2}
    rec2idx = {"L": 0, "G": 1, "Q": 2}
    
    # Initialize model
    model = TransformerModel(cfg, elem2idx, rec2idx)
    
    # Mock input
    batch_size = 4
    seq_len = 10
    x = {
        "element": torch.randint(0, len(elem2idx), (batch_size, seq_len)),
        "record_type": torch.randint(0, len(rec2idx), (batch_size, seq_len)),
        "numeric": torch.randn(batch_size, seq_len)
    }
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Check output structure
    assert "L" in output
    assert "G" in output
    assert "Q" in output
    assert output["L"].shape == (batch_size,)
    assert output["G"].shape == (batch_size,)
    assert output["Q"].shape == (batch_size,)
    
    # Check output values are finite
    assert torch.isfinite(output["L"]).all()
    assert torch.isfinite(output["G"]).all()
    assert torch.isfinite(output["Q"]).all()
    
    print("✓ TransformerModel smoke test passed")

def test_transformer_training_step():
    """Test training step with mock data"""
    cfg = {
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "ff_dim": 128,
        "rank": 4,
        "alpha": 8,
        "task_weights": {"L": 1.0, "G": 1.0, "Q": 1.0}
    }
    
    elem2idx = {"H": 0, "He": 1}
    rec2idx = {"L": 0, "G": 1, "Q": 2}
    
    model = TransformerModel(cfg, elem2idx, rec2idx)
    
    # Mock batch
    batch_size = 2
    seq_len = 5
    x = {
        "element": torch.randint(0, len(elem2idx), (batch_size, seq_len)),
        "record_type": torch.randint(0, len(rec2idx), (batch_size, seq_len)),
        "numeric": torch.randn(batch_size, seq_len)
    }
    y = {
        "L": torch.randn(batch_size),
        "G": torch.randn(batch_size),
        "Q": torch.randn(batch_size)
    }
    
    # Training step
    loss = model.training_step((x, y))
    
    # Check loss is finite and positive
    assert torch.isfinite(loss)
    assert loss > 0
    
    print("✓ TransformerModel training step test passed") 