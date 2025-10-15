"""
Test script to verify the setup works correctly.
"""

import torch
import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import transformer_lens
        print("✓ transformer_lens imported")
    except ImportError as e:
        print(f"✗ transformer_lens import failed: {e}")
        return False
    
    try:
        import transformers
        print("✓ transformers imported")
    except ImportError as e:
        print(f"✗ transformers import failed: {e}")
        return False
    
    try:
        import torch
        print("✓ torch imported")
    except ImportError as e:
        print(f"✗ torch import failed: {e}")
        return False
    
    return True

def test_prompts():
    """Test that our prompts work."""
    print("\nTesting prompts...")
    
    try:
        from prompts import get_all_prompts, get_prompt_counts
        
        prompts = get_all_prompts()
        counts = get_prompt_counts()
        
        print(f"✓ Loaded {sum(counts.values())} prompts:")
        for category, count in counts.items():
            print(f"  - {category}: {count} prompts")
        
        return True
    except Exception as e:
        print(f"✗ Prompts test failed: {e}")
        return False

def test_model_access():
    """Test if we can access the model (without loading it)."""
    print("\nTesting model access...")
    
    try:
        from transformers import AutoTokenizer
        
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        print(f"Checking access to {model_name}...")
        
        # Just try to load the config, not the full model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Model access successful")
        return True
        
    except Exception as e:
        print(f"✗ Model access failed: {e}")
        print("  You may need to:")
        print("  1. Run 'huggingface-cli login'")
        print("  2. Download the model first")
        return False

def test_memory():
    """Test available memory."""
    print("\nTesting memory...")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ CUDA available: {gpu_memory:.1f}GB")
        return True
    else:
        print("ℹ CUDA not available, will use CPU")
        print("  This is fine for the experiment")
        return True

def main():
    """Run all tests."""
    print("Self-Referent Experiment Setup Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_prompts,
        test_model_access,
        test_memory
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Setup is ready.")
    else:
        print("⚠ Some tests failed. Please address the issues above.")

if __name__ == "__main__":
    main()
