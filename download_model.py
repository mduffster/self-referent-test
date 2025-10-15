"""
Script to download Llama-3.1-8B-Instruct model for TransformerLens.
"""

import os
from huggingface_hub import snapshot_download
import torch

def download_llama_model(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", cache_dir=None):
    """
    Download the Llama model from HuggingFace.
    
    Args:
        model_name: HuggingFace model name
        cache_dir: Directory to cache the model (default: ~/.cache/huggingface)
    """
    print(f"Downloading {model_name}...")
    print("This may take a while (model is ~16GB)")
    
    try:
        # Download the model
        local_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True  # Resume if interrupted
        )
        
        print(f"✓ Model downloaded successfully to: {local_dir}")
        return local_dir
        
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        print("\nPossible issues:")
        print("1. Not authenticated with HuggingFace (run 'huggingface-cli login')")
        print("2. Insufficient disk space")
        print("3. Network issues")
        return None

def check_model_access():
    """Check if we can access the model."""
    print("Checking model access...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        
        # Try to load tokenizer first (smaller)
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Cannot access model: {e}")
        return False

def main():
    """Main download function."""
    print("Llama-3.1-8B-Instruct Download Script")
    print("=" * 40)
    
    # Check access first
    if not check_model_access():
        print("\nPlease run 'huggingface-cli login' first to authenticate.")
        return
    
    # Download the model
    model_path = download_llama_model()
    
    if model_path:
        print(f"\n✓ Model ready at: {model_path}")
        print("You can now run the experiment with TransformerLens!")
    else:
        print("\n✗ Download failed. Please check the issues above.")

if __name__ == "__main__":
    main()
