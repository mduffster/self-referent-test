"""
Advanced activation analysis for self-referent experiment.
This script extracts and analyzes internal model activations.
"""

import torch
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer
from prompts import get_all_prompts
from output_manager import OutputManager
from deterministic import set_seed, verify_determinism
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import psutil
import gc

def get_memory_info():
    """Get current memory usage information."""
    process = psutil.Process()
    memory_info = process.memory_info()
    system_memory = psutil.virtual_memory()
    
    return {
        'process_memory_gb': memory_info.rss / (1024**3),
        'process_memory_percent': process.memory_percent(),
        'system_memory_used_gb': system_memory.used / (1024**3),
        'system_memory_total_gb': system_memory.total / (1024**3),
        'system_memory_percent': system_memory.percent
    }

def print_memory_status(stage: str):
    """Print current memory status with a stage label."""
    mem_info = get_memory_info()
    print(f"\n🔍 MEMORY STATUS - {stage}")
    print(f"   Process: {mem_info['process_memory_gb']:.2f} GB ({mem_info['process_memory_percent']:.1f}%)")
    print(f"   System:  {mem_info['system_memory_used_gb']:.2f} / {mem_info['system_memory_total_gb']:.2f} GB ({mem_info['system_memory_percent']:.1f}%)")
    
    # Warning if memory usage is high
    if mem_info['system_memory_percent'] > 85:
        print(f"   ⚠️  WARNING: High system memory usage ({mem_info['system_memory_percent']:.1f}%)")
    if mem_info['process_memory_percent'] > 20:
        print(f"   ⚠️  WARNING: High process memory usage ({mem_info['process_memory_percent']:.1f}%)")

def cleanup_memory():
    """Force garbage collection and memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class ActivationAnalyzer:
    """Analyzes model activations for self-referent patterns."""
    
    def __init__(self, model: HookedTransformer, output_manager: OutputManager):
        self.model = model
        self.output_manager = output_manager
        self.activations_cache = {}
        
    def extract_activations(self, prompts: List[str], prompt_categories: List[str]) -> Dict[str, Any]:
        """
        Extract activations for a set of prompts.
        
        Args:
            prompts: List of prompts to analyze
            prompt_categories: List of categories (self_referent, confounder, neutral)
            
        Returns:
            Dictionary containing activation data
        """
        print(f"Extracting activations for {len(prompts)} prompts...")
        print_memory_status("STARTING ACTIVATION EXTRACTION")
        
        # Define which activations to extract
        activation_names = [
            "embed",  # Input embeddings
            "pos_embed",  # Position embeddings
            "ln_final",  # Final layer norm
        ]
        
        # Add all attention and MLP layers
        for layer in range(self.model.cfg.n_layers):  # All layers
            activation_names.extend([
                f"blocks.{layer}.attn.hook_pattern", # Attention patterns (this exists)
                f"blocks.{layer}.mlp.hook_post",     # MLP output
            ])
        
        all_activations = {name: [] for name in activation_names}
        all_prompts = []
        all_categories = []
        
        for i, (prompt, category) in enumerate(zip(prompts, prompt_categories)):
            print(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            # Print memory status every 5 prompts
            if (i + 1) % 5 == 0:
                print_memory_status(f"PROCESSING PROMPT {i+1}/{len(prompts)}")
            
            try:
                # Tokenize the prompt
                tokens = self.model.to_tokens(prompt)
                
                # Run forward pass with hooks to capture activations
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(
                        tokens,
                        names_filter=activation_names,
                        return_type="logits"
                    )
                
                # Store activations for this prompt
                for name in activation_names:
                    if name in cache:
                        # Convert to numpy and store
                        activation = cache[name].cpu().numpy()
                        all_activations[name].append(activation)
                    else:
                        all_activations[name].append(None)
                
                all_prompts.append(prompt)
                all_categories.append(category)
                
                # Clean up memory after each prompt
                if (i + 1) % 5 == 0:
                    cleanup_memory()
                
            except Exception as e:
                print(f"Error processing prompt {i+1}: {e}")
                # Add None values for failed prompts
                for name in activation_names:
                    all_activations[name].append(None)
                all_prompts.append(prompt)
                all_categories.append(category)
        
        print_memory_status("ACTIVATION EXTRACTION COMPLETE")
        cleanup_memory()
        
        return {
            "activations": all_activations,
            "prompts": all_prompts,
            "categories": all_categories,
            "activation_names": activation_names
        }
    
    def analyze_attention_patterns(self, activation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze attention patterns to find self-referent focus."""
        print("Analyzing attention patterns...")
        
        attention_analysis = {}
        
        # Look at attention patterns from different layers
        for layer in range(min(4, self.model.cfg.n_layers)):
            pattern_name = f"blocks.{layer}.attn.hook_pattern"
            
            if pattern_name in activation_data["activations"]:
                patterns = activation_data["activations"][pattern_name]
                categories = activation_data["categories"]
                
                # Analyze patterns by category
                self_referent_patterns = []
                confounder_patterns = []
                neutral_patterns = []
                
                for i, (pattern, category) in enumerate(zip(patterns, categories)):
                    if pattern is not None:
                        if category == "self_referent":
                            self_referent_patterns.append(pattern)
                        elif category == "confounder":
                            confounder_patterns.append(pattern)
                        elif category == "neutral":
                            neutral_patterns.append(pattern)
                
                # Store analysis for this layer
                attention_analysis[f"layer_{layer}"] = {
                    "self_referent_count": len(self_referent_patterns),
                    "confounder_count": len(confounder_patterns),
                    "neutral_count": len(neutral_patterns),
                    "self_referent_patterns": self_referent_patterns,
                    "confounder_patterns": confounder_patterns,
                    "neutral_patterns": neutral_patterns
                }
        
        return attention_analysis
    
    def compare_activations(self, activation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare activations between self-referent and other categories."""
        print("Comparing activations between categories...")
        
        categories = activation_data["categories"]
        activations = activation_data["activations"]
        
        comparison_results = {}
        
        for activation_name, activation_list in activations.items():
            if activation_name == "activation_names":
                continue
                
            # Group activations by category
            self_referent_acts = []
            confounder_acts = []
            neutral_acts = []
            
            for i, (activation, category) in enumerate(zip(activation_list, categories)):
                if activation is not None:
                    if category == "self_referent":
                        self_referent_acts.append(activation)
                    elif category == "confounder":
                        confounder_acts.append(activation)
                    elif category == "neutral":
                        neutral_acts.append(activation)
            
            # Calculate statistics if we have data
            if self_referent_acts and (confounder_acts or neutral_acts):
                # Calculate means safely, handling different array shapes
                def safe_mean(activation_list):
                    if not activation_list:
                        return None
                    try:
                        # Try to concatenate and take mean
                        concatenated = np.concatenate([act.flatten() for act in activation_list])
                        return float(np.mean(concatenated))
                    except:
                        # If concatenation fails, return None
                        return None
                
                comparison_results[activation_name] = {
                    "self_referent_count": len(self_referent_acts),
                    "confounder_count": len(confounder_acts),
                    "neutral_count": len(neutral_acts),
                    "self_referent_mean": safe_mean(self_referent_acts),
                    "confounder_mean": safe_mean(confounder_acts),
                    "neutral_mean": safe_mean(neutral_acts),
                }
        
        return comparison_results
    
    def save_raw_activations(self, activation_data: Dict[str, Any]):
        """Save raw activation data for later analysis."""
        print("Saving raw activation data...")
        
        # Save each activation type separately to avoid huge files
        for activation_name, activation_list in activation_data["activations"].items():
            if activation_name == "activation_names":
                continue
                
            # Create filename for this activation type
            filename = f"raw_{activation_name.replace('.', '_').replace('hook_', '')}.npz"
            
            # Convert activations to a format that can be saved (object array to handle different shapes)
            # Each activation in the list can have different shapes, so we need object dtype
            activations_array = np.empty(len(activation_list), dtype=object)
            for i, activation in enumerate(activation_list):
                activations_array[i] = activation
            
            # Prepare data for saving
            save_data = {
                'activations': activations_array,
                'prompts': np.array(activation_data["prompts"]),
                'categories': np.array(activation_data["categories"]),
                'activation_name': np.array(activation_name)
            }
            
            # Save as compressed numpy file
            filepath = os.path.join(self.output_manager.run_dir, filename)
            np.savez_compressed(filepath, **save_data)
            
            print(f"✓ Saved {activation_name} to {filename}")
    
    def run_analysis(self, num_prompts_per_category: int = 3) -> Dict[str, Any]:
        """Run the complete activation analysis."""
        print("Starting activation analysis...")
        print_memory_status("STARTING ANALYSIS")
        
        # Get prompts
        all_prompts = get_all_prompts()
        
        # Select prompts for analysis
        selected_prompts = []
        selected_categories = []
        
        for category, prompt_list in all_prompts.items():
            for prompt in prompt_list[:num_prompts_per_category]:
                selected_prompts.append(prompt)
                selected_categories.append(category)
        
        print(f"Analyzing {len(selected_prompts)} prompts...")
        
        # Extract activations
        activation_data = self.extract_activations(selected_prompts, selected_categories)
        print_memory_status("AFTER ACTIVATION EXTRACTION")
        
        # Analyze attention patterns
        attention_analysis = self.analyze_attention_patterns(activation_data)
        print_memory_status("AFTER ATTENTION ANALYSIS")
        
        # Compare activations
        comparison_results = self.compare_activations(activation_data)
        print_memory_status("AFTER ACTIVATION COMPARISON")
        
        # Compile results
        results = {
            "activation_data": activation_data,
            "attention_analysis": attention_analysis,
            "comparison_results": comparison_results,
            "num_prompts_analyzed": len(selected_prompts),
            "prompts_per_category": num_prompts_per_category
        }
        
        # Save both processed results and raw data
        self.output_manager.save_activation_data(results)
        print_memory_status("AFTER SAVING PROCESSED DATA")
        
        # Also save raw activations for deeper analysis
        self.save_raw_activations(activation_data)
        print_memory_status("AFTER SAVING RAW DATA")
        
        return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Activation analysis for self-referent experiment")
    
    parser.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.1",
                       help="HuggingFace model ID (default: mistralai/Mistral-7B-Instruct-v0.1)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                       help="Device to run on (default: cpu)")
    parser.add_argument("--prompts_per_category", type=int, default=3,
                       help="Number of prompts per category (default: 3)")
    parser.add_argument("--seed", type=int, default=123,
                       help="Random seed for reproducibility (default: 123)")
    parser.add_argument("--output_type", default="latest", choices=["latest", "latest_base"],
                       help="Output directory type: 'latest' for latest_run, 'latest_base' for latest_base (default: latest)")
    parser.add_argument("--output_dir", default="results_activation_analysis",
                       help="Base output directory (default: results_activation_analysis)")
    
    return parser.parse_args()

def main():
    """Run activation analysis experiment."""
    args = parse_args()
    
    print("Self-Referent Activation Analysis")
    print("=" * 40)
    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print(f"Prompts per category: {args.prompts_per_category}")
    print(f"Seed: {args.seed}")
    print("=" * 40)
    
    # Set up determinism
    set_seed(args.seed)
    verify_determinism()
    
    # Initialize output manager
    use_base = args.output_type == "latest_base"
    output_manager = OutputManager(args.output_dir, use_latest=True, use_base=use_base)
    
    # Load model
    print("Loading model...")
    print_memory_status("BEFORE MODEL LOADING")
    model = HookedTransformer.from_pretrained(
        args.model_id,
        device=args.device,
        dtype=torch.float32
    )
    print("✓ Model loaded successfully")
    print_memory_status("AFTER MODEL LOADING")
    
    # Initialize analyzer
    analyzer = ActivationAnalyzer(model, output_manager)
    
    # Run analysis
    results = analyzer.run_analysis(num_prompts_per_category=args.prompts_per_category)
    
    print(f"\n✓ Analysis complete!")
    print(f"Results saved to: {output_manager.run_dir}")
    print_memory_status("ANALYSIS COMPLETE")
    
    # Print summary
    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"Prompts analyzed: {results['num_prompts_analyzed']}")
    print(f"Layers analyzed: {len([k for k in results['attention_analysis'].keys()])}")
    print(f"Activation types: {len(results['comparison_results'])}")

if __name__ == "__main__":
    main()
