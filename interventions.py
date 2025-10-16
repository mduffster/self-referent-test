"""
Intervention/ablation experiments for self-referent analysis.
This script runs ablations using the same setup as activation_analysis.py.
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

class InterventionAnalyzer:
    """Analyzes model activations for self-referent patterns with interventions."""
    
    def __init__(self, model: HookedTransformer, output_manager: OutputManager):
        self.model = model
        self.output_manager = output_manager
        self.activations_cache = {}
        
    def extract_activations_with_intervention(self, prompts: List[str], prompt_categories: List[str], 
                                            intervention_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract activations for a set of prompts with intervention applied.
        
        Args:
            prompts: List of prompts to analyze
            prompt_categories: List of categories (self_referent, confounder, neutral, third_person)
            intervention_config: Configuration for the intervention
            
        Returns:
            Dictionary containing activation data
        """
        print(f"Extracting activations for {len(prompts)} prompts with intervention...")
        print(f"Intervention config: {intervention_config}")
        
        # Define which activations to extract (same as activation_analysis.py)
        activation_names = [
            "embed",  # Input embeddings
            "pos_embed",  # Position embeddings
            "ln_final",  # Final layer norm
        ]
        
        # Add all attention and MLP layers
        for layer in range(self.model.cfg.n_layers):  # All layers
            activation_names.extend([
                f"blocks.{layer}.attn.hook_pattern", # Attention patterns
                f"blocks.{layer}.mlp.hook_post",     # MLP output
            ])
        
        all_activations = {name: [] for name in activation_names}
        all_prompts = []
        all_categories = []
        
        # Create intervention hook
        intervention_type = intervention_config.get("type", "attention")
        layer = intervention_config.get("layer")
        head = intervention_config.get("head")
        method = intervention_config.get("method", "zero_out")
        
        def create_intervention_hook():
            if intervention_type == "attention" and head is not None:
                def intervention_hook(attn_pattern, hook):
                    if method == "zero_out":
                        attn_pattern[0, head, :, :] = 0.0
                    elif method == "half_out":
                        attn_pattern[0, head, :, :] = attn_pattern[0, head, :, :] * 0.5
                    elif method == "random":
                        seq_len = attn_pattern.shape[-1]
                        random_pattern = torch.randn(seq_len, seq_len)
                        random_pattern = torch.softmax(random_pattern, dim=-1)
                        attn_pattern[0, head, :, :] = random_pattern
                    elif method == "uniform":
                        seq_len = attn_pattern.shape[-1]
                        uniform_pattern = torch.ones(seq_len, seq_len) / seq_len
                        attn_pattern[0, head, :, :] = uniform_pattern
                    return attn_pattern
                return (f"blocks.{layer}.attn.hook_pattern", intervention_hook)
            
            elif intervention_type == "mlp":
                def intervention_hook(mlp_output, hook):
                    if method == "zero_out":
                        mlp_output[0, :, :] = 0.0
                    elif method == "random":
                        mlp_output[0, :, :] = torch.randn_like(mlp_output[0, :, :])
                    return mlp_output
                return (f"blocks.{layer}.mlp.hook_post", intervention_hook)
            
            return None
        
        intervention_hook = create_intervention_hook()
        
        for i, (prompt, category) in enumerate(zip(prompts, prompt_categories)):
            print(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            try:
                # Tokenize the prompt
                tokens = self.model.to_tokens(prompt)
                
                # Run forward pass with hooks to capture activations and intervention
                with torch.no_grad():
                    if intervention_hook:
                        with self.model.hooks([intervention_hook]):
                            _, cache = self.model.run_with_cache(
                                tokens,
                                names_filter=activation_names,
                                return_type="logits"
                            )
                    else:
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
                
            except Exception as e:
                print(f"Error processing prompt {i+1}: {e}")
                # Add None values for failed prompts
                for name in activation_names:
                    all_activations[name].append(None)
                all_prompts.append(prompt)
                all_categories.append(category)
        
        return {
            "activations": all_activations,
            "prompts": all_prompts,
            "categories": all_categories,
            "activation_names": activation_names,
            "intervention_config": intervention_config
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
            activations_array = np.empty(len(activation_list), dtype=object)
            for i, activation in enumerate(activation_list):
                activations_array[i] = activation
            
            # Prepare data for saving
            save_data = {
                'activations': activations_array,
                'prompts': np.array(activation_data["prompts"]),
                'categories': np.array(activation_data["categories"]),
                'activation_name': np.array(activation_name),
                'intervention_config': np.array(str(activation_data["intervention_config"]))
            }
            
            # Save as compressed numpy file
            filepath = os.path.join(self.output_manager.run_dir, filename)
            np.savez_compressed(filepath, **save_data)
            
            print(f"✓ Saved {activation_name} to {filename}")
    
    def get_intervention_directory_name(self, intervention_config: Dict[str, Any]) -> str:
        """Generate directory name based on intervention configuration."""
        intervention_type = intervention_config.get("type", "unknown")
        layer = intervention_config.get("layer")
        head = intervention_config.get("head")
        method = intervention_config.get("method", "zero_out")
        
        if intervention_type == "attention":
            if head is not None:
                return f"sh_{layer}_{head}_{method}"  # single head
            else:
                return f"ah_{layer}_{method}"  # all heads in layer
        elif intervention_type == "mlp":
            return f"mlp_{layer}_{method}"
        else:
            return f"{intervention_type}_{layer}_{method}"
    
    def run_intervention_analysis(self, intervention_config: Dict[str, Any], 
                                 num_prompts_per_category: int = 3) -> Dict[str, Any]:
        """Run activation analysis with a specific intervention."""
        intervention_name = intervention_config.get("name", "unnamed")
        print(f"Starting intervention analysis: {intervention_name}")
        
        # Get prompts
        all_prompts = get_all_prompts()
        
        # Select prompts for analysis
        selected_prompts = []
        selected_categories = []
        
        for category, prompt_list in all_prompts.items():
            for prompt in prompt_list[:num_prompts_per_category]:
                selected_prompts.append(prompt)
                selected_categories.append(category)
        
        print(f"Analyzing {len(selected_prompts)} prompts with intervention...")
        
        # Extract activations with intervention
        activation_data = self.extract_activations_with_intervention(
            selected_prompts, selected_categories, intervention_config
        )
        
        # Analyze attention patterns
        attention_analysis = self.analyze_attention_patterns(activation_data)
        
        # Compare activations
        comparison_results = self.compare_activations(activation_data)
        
        # Compile results
        results = {
            "activation_data": activation_data,
            "attention_analysis": attention_analysis,
            "comparison_results": comparison_results,
            "num_prompts_analyzed": len(selected_prompts),
            "prompts_per_category": num_prompts_per_category,
            "intervention_config": intervention_config
        }
        
        # Save both processed results and raw data
        self.output_manager.save_activation_data(results)
        
        # Also save raw activations for deeper analysis
        self.save_raw_activations(activation_data)
        
        return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Intervention experiments for self-referent analysis")
    
    parser.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.1",
                       help="HuggingFace model ID (default: mistralai/Mistral-7B-Instruct-v0.1)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                       help="Device to run on (default: cpu)")
    parser.add_argument("--prompts_per_category", type=int, default=3,
                       help="Number of prompts per category (default: 3)")
    parser.add_argument("--seed", type=int, default=123,
                       help="Random seed for reproducibility (default: 123)")
    parser.add_argument("--interventions", type=str, default="default",
                       help="Intervention configuration file or 'default' for built-in configs")
    
    return parser.parse_args()

def main():
    """Run intervention experiments."""
    args = parse_args()
    
    print("Self-Referent Intervention Analysis")
    print("=" * 40)
    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print(f"Prompts per category: {args.prompts_per_category}")
    print(f"Seed: {args.seed}")
    print("=" * 40)
    
    # Set up determinism
    set_seed(args.seed)
    verify_determinism()
    
    # Load model
    print("Loading model...")
    model = HookedTransformer.from_pretrained(
        args.model_id,
        device=args.device,
        dtype=torch.float32
    )
    print("✓ Model loaded successfully")
    
    # Load intervention configurations
    intervention_configs = None
    if args.interventions != "default" and os.path.exists(args.interventions):
        import json
        with open(args.interventions, 'r') as f:
            intervention_configs = json.load(f)
        print(f"Loaded intervention configs from: {args.interventions}")
    else:
        # Default configurations
        intervention_configs = [
            {"name": "zero_attn_0_0", "type": "attention", "layer": 0, "head": 0, "method": "zero_out"},
            {"name": "zero_attn_1_5", "type": "attention", "layer": 1, "head": 5, "method": "zero_out"},
        ]
    
    # Run each intervention separately
    all_results = {}
    
    for config in intervention_configs:
        intervention_name = config.get("name", "unnamed")
        print(f"\n{'='*60}")
        print(f"Running intervention: {intervention_name}")
        print(f"Config: {config}")
        print(f"{'='*60}")
        
        # Create specific directory for this intervention
        analyzer = InterventionAnalyzer(model, None)  # We'll set output manager per intervention
        intervention_dir_name = analyzer.get_intervention_directory_name(config)
        intervention_dir = os.path.join("results_activation_analysis", "latest_intervention", intervention_dir_name)
        os.makedirs(intervention_dir, exist_ok=True)
        
        print(f"Output directory: {intervention_dir}")
        
        # Create output manager for this intervention
        output_manager = OutputManager("results_activation_analysis", use_intervention=True)
        output_manager.run_dir = intervention_dir
        
        # Initialize analyzer with output manager
        analyzer = InterventionAnalyzer(model, output_manager)
        
        # Run intervention analysis
        results = analyzer.run_intervention_analysis(config, args.prompts_per_category)
        all_results[intervention_name] = results
    
    print(f"\n✓ Intervention analysis complete!")
    
    # Print summary
    print(f"\n=== INTERVENTION SUMMARY ===")
    print(f"Interventions run: {len(intervention_configs)}")
    for config in intervention_configs:
        intervention_name = config.get("name", "unnamed")
        intervention_dir_name = analyzer.get_intervention_directory_name(config)
        print(f"  {intervention_name} -> results_activation_analysis/latest_intervention/{intervention_dir_name}/")

if __name__ == "__main__":
    main()