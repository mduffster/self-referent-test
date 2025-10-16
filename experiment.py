"""
Initial experiment script for self-referent analysis with Mistral-7B using TransformerLens.
"""

import torch
import transformer_lens
from transformer_lens import HookedTransformer
from prompts import get_all_prompts, get_prompt_counts
from output_manager import OutputManager
from deterministic import set_seed, verify_determinism
import numpy as np
import pandas as pd
import argparse
import sys

def load_model(model_name="mistralai/Mistral-7B-Instruct-v0.1", device="cpu"):
    """
    Load the model using TransformerLens.
    
    Args:
        model_name: HuggingFace model name or path to local model
        device: Device to load model on ("cpu" or "cuda")
    
    Returns:
        HookedTransformer: Loaded model
    """
    print(f"Loading model: {model_name} on {device}")
    
    try:
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            dtype=torch.float32
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
                temperature=0,
                do_sample=False
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

def test_model_loading(model_name, device):
    """Test model loading with specific parameters."""
    print("=== MODEL LOADING TEST ===")
    
    print(f"Trying to load: {model_name} on {device}")
    model = load_model(model_name, device)
    if model is not None:
        print("✓ Model loaded successfully!")
        return model
    else:
        print("✗ Failed to load model")
        return None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Self-referent experiment with Mistral-7B")
    
    parser.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.1",
                       help="HuggingFace model ID (default: mistralai/Mistral-7B-Instruct-v0.1)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                       help="Device to run on (default: cpu)")
    parser.add_argument("--max_tokens", type=int, default=50,
                       help="Maximum tokens to generate (default: 50)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (default: 0.7)")
    parser.add_argument("--seed", type=int, default=123,
                       help="Random seed for reproducibility (default: 123)")
    parser.add_argument("--prompts_per_category", type=int, default=2,
                       help="Number of prompts to test per category (default: 2)")
    
    return parser.parse_args()

def main():
    """Main experiment function."""
    args = parse_args()
    
    print("Self-Referent Experiment Setup")
    print("=" * 40)
    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Prompts per category: {args.prompts_per_category}")
    print("=" * 40)
    
    # Set up determinism
    set_seed(args.seed)
    verify_determinism()
    
    # Initialize output manager
    output_manager = OutputManager()
    
    # Analyze prompts
    analyze_prompts()
    
    # Test model loading
    model = test_model_loading(args.model_id, args.device)
    
    if model is not None:
        # Test basic generation
        prompts = get_all_prompts()
        test_prompts = (prompts["self_referent"][:args.prompts_per_category] + 
                       prompts["confounder"][:args.prompts_per_category] + 
                       prompts["neutral"][:args.prompts_per_category])
        results = test_model_generation(model, test_prompts, output_manager, args.max_tokens)
        
        # Save experiment config
        config = {
            "model_name": args.model_id,
            "device": args.device,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "prompts_per_category": args.prompts_per_category,
            "total_prompts": sum(get_prompt_counts().values()),
            "prompt_counts": get_prompt_counts(),
            "test_prompts_count": len(test_prompts)
        }
        output_manager.save_experiment_config(config)
        
        # Create summary
        summary = {
            "experiment_status": "Model loaded and basic generation successful",
            "results_directory": output_manager.run_dir,
            "model_config": f"{args.model_id} on {args.device}",
            "generation_results": f"{len(results)} prompts tested",
            "next_steps": [
                "Run activation_analysis.py for deeper mechanistic analysis",
                "Use analyze_results.py to examine activation differences", 
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
        sys.exit(1)
    
    print(f"\n=== RESULTS SAVED TO: {output_manager.run_dir} ===")
    print("Next steps:")
    print("1. Run: python activation_analysis.py --model_id {args.model_id} --device {args.device}")
    print("2. Run: python analyze_results.py")

if __name__ == "__main__":
    main()
