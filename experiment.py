"""
Initial experiment script for self-referent analysis with llama-8B using TransformerLens.
"""

import torch
import transformer_lens
from transformer_lens import HookedTransformer
from prompts import get_all_prompts, get_prompt_counts
from output_manager import OutputManager
import numpy as np
import pandas as pd

def load_model(model_name="mistralai/Mistral-7B-Instruct-v0.1"):
    """
    Load the llama model using TransformerLens.
    
    Args:
        model_name: HuggingFace model name or path to local model
    
    Returns:
        HookedTransformer: Loaded model
    """
    print(f"Loading model: {model_name}")
    
    try:
        model = HookedTransformer.from_pretrained(
            model_name,
            device="cpu",  # Start with CPU, can move to GPU later if needed
            torch_dtype=torch.float32
        )
        print(f"Successfully loaded {model_name}")
        print(f"Model config: {model.cfg}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def test_model_generation(model, test_prompts, output_manager, max_tokens=50):
    """
    Test basic text generation with the model.
    
    Args:
        model: HookedTransformer model
        test_prompts: List of test prompts
        output_manager: OutputManager instance
        max_tokens: Maximum tokens to generate
    """
    print("\nTesting model generation...")
    
    results = []
    
    for i, prompt in enumerate(test_prompts[:6]):  # Test first 6 prompts (2 from each category)
        print(f"\n--- Prompt {i+1}: {prompt} ---")
        try:
            # Generate response
            response = model.generate(
                prompt, 
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True
            )
            print(f"Response: {response}")
            
            # Store result
            result = {
                "prompt_id": i+1,
                "prompt": prompt,
                "response": response,
                "category": "self_referent" if i < 2 else "confounder" if i < 4 else "neutral",
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error generating response: {e}")
            result = {
                "prompt_id": i+1,
                "prompt": prompt,
                "response": f"ERROR: {e}",
                "category": "self_referent" if i < 2 else "confounder" if i < 4 else "neutral",
                "error": str(e)
            }
            results.append(result)
    
    # Save results
    output_manager.save_generation_results(results)
    return results

def analyze_prompts():
    """Analyze the prompt categories and counts."""
    print("=== PROMPT ANALYSIS ===")
    
    prompts = get_all_prompts()
    counts = get_prompt_counts()
    
    for category, prompt_list in prompts.items():
        print(f"\n{category.upper()} PROMPTS ({counts[category]} total):")
        for i, prompt in enumerate(prompt_list[:5]):  # Show first 5
            print(f"  {i+1}. {prompt}")
        if len(prompt_list) > 5:
            print(f"  ... and {len(prompt_list) - 5} more")
    
    print(f"\nTotal prompts: {sum(counts.values())}")

def test_model_loading():
    """Test different model loading options."""
    print("=== MODEL LOADING TEST ===")
    
    # Try different model names/paths
    model_options = [
        "mistralai/Mistral-7B-Instruct-v0.1",  # No auth required
        "meta-llama/Meta-Llama-3.1-8B-Instruct",  # Requires auth
        # Add local paths here if you have them
    ]
    
    for model_name in model_options:
        print(f"\nTrying to load: {model_name}")
        model = load_model(model_name)
        if model is not None:
            print("✓ Model loaded successfully!")
            return model
        else:
            print("✗ Failed to load model")
    
    return None

def main():
    """Main experiment function."""
    print("Self-Referent Experiment Setup")
    print("=" * 40)
    
    # Initialize output manager
    output_manager = OutputManager()
    
    # Analyze prompts
    analyze_prompts()
    
    # Test model loading
    model = test_model_loading()
    
    if model is not None:
        # Test basic generation
        prompts = get_all_prompts()
        test_prompts = prompts["self_referent"][:2] + prompts["confounder"][:2] + prompts["neutral"][:2]
        results = test_model_generation(model, test_prompts, output_manager)
        
        # Save experiment config
        config = {
            "model_name": "mistralai/Mistral-7B-Instruct-v0.1",
            "total_prompts": sum(get_prompt_counts().values()),
            "prompt_counts": get_prompt_counts(),
            "test_prompts_count": len(test_prompts),
            "max_tokens": 50,
            "temperature": 0.7
        }
        output_manager.save_experiment_config(config)
        
        # Create summary
        summary = {
            "experiment_status": "Model loaded and basic generation successful",
            "results_directory": output_manager.run_dir,
            "generation_results": f"{len(results)} prompts tested",
            "next_steps": [
                "Analyze generated responses for self-referent patterns",
                "Set up activation patching experiments", 
                "Compare activations between self-referent and confounder prompts",
                "Identify key attention heads and MLP layers"
            ]
        }
        output_manager.create_summary_report(summary)
        
    else:
        print("\n⚠️  No model could be loaded. Please check:")
        print("   - Model path/name is correct")
        print("   - You have access to the model")
        print("   - Sufficient disk space for model weights")
    
    print(f"\n=== RESULTS SAVED TO: {output_manager.run_dir} ===")
    print("1. Verify model is working with basic generation")
    print("2. Set up activation patching experiments")
    print("3. Analyze self-referent vs confounder activations")
    print("4. Identify key attention heads and MLP layers")

if __name__ == "__main__":
    main()
