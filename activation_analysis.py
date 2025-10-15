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
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
        
        # Define which activations to extract
        activation_names = [
            "embed",  # Input embeddings
            "pos_embed",  # Position embeddings
            "ln_final",  # Final layer norm
        ]
        
        # Add some attention and MLP layers
        for layer in range(min(4, self.model.cfg.n_layers)):  # First 4 layers
            activation_names.extend([
                f"blocks.{layer}.attn.hook_result",  # Attention output
                f"blocks.{layer}.mlp.hook_post",     # MLP output
                f"blocks.{layer}.attn.hook_pattern", # Attention patterns
            ])
        
        all_activations = {name: [] for name in activation_names}
        all_prompts = []
        all_categories = []
        
        for i, (prompt, category) in enumerate(zip(prompts, prompt_categories)):
            print(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
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
            
            # Prepare data for saving
            save_data = {
                'activations': activation_list,
                'prompts': activation_data["prompts"],
                'categories': activation_data["categories"],
                'activation_name': activation_name
            }
            
            # Save as compressed numpy file
            filepath = os.path.join(self.output_manager.run_dir, filename)
            np.savez_compressed(filepath, **save_data)
            
            print(f"✓ Saved {activation_name} to {filename}")
    
    def run_analysis(self, num_prompts_per_category: int = 3) -> Dict[str, Any]:
        """Run the complete activation analysis."""
        print("Starting activation analysis...")
        
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
            "prompts_per_category": num_prompts_per_category
        }
        
        # Save both processed results and raw data
        self.output_manager.save_activation_data(results)
        
        # Also save raw activations for deeper analysis
        self.save_raw_activations(activation_data)
        
        return results

def main():
    """Run activation analysis experiment."""
    print("Self-Referent Activation Analysis")
    print("=" * 40)
    
    # Initialize output manager
    output_manager = OutputManager("results_activation_analysis")
    
    # Load model
    print("Loading model...")
    model = HookedTransformer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        device="cpu",
        torch_dtype=torch.float32
    )
    print("✓ Model loaded successfully")
    
    # Initialize analyzer
    analyzer = ActivationAnalyzer(model, output_manager)
    
    # Run analysis (3 prompts per category = 9 total)
    results = analyzer.run_analysis(num_prompts_per_category=3)
    
    print(f"\n✓ Analysis complete!")
    print(f"Results saved to: {output_manager.run_dir}")
    
    # Print summary
    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"Prompts analyzed: {results['num_prompts_analyzed']}")
    print(f"Layers analyzed: {len([k for k in results['attention_analysis'].keys()])}")
    print(f"Activation types: {len(results['comparison_results'])}")

if __name__ == "__main__":
    main()
