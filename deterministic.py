"""
Determinism utilities for reproducible experiments.
"""

import random
import numpy as np
import torch
import os

def set_seed(seed=123):
    """
    Set global seeds for reproducible experiments.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # For CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make operations deterministic (may impact performance)
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    print(f"✓ Set seed to {seed} for reproducible results")

def verify_determinism():
    """Verify that the environment is set up for deterministic results."""
    print("=== DETERMINISM VERIFICATION ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Python hash seed: {os.environ.get('PYTHONHASHSEED', 'Not set')}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
    print("=" * 30)
