#!/usr/bin/env python3
"""
Automated pipeline runner for model family analysis.
Runs activation analysis, visualization, and comparison for a complete model family.
"""

import json
import argparse
import subprocess
import sys
import os
from pathlib import Path

def load_config():
    """Load the model family configuration."""
    config_path = Path(__file__).parent / "model_family_config.json"
    with open(config_path, 'r') as f:
        return json.load(f)

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed with exit code {result.returncode}")
        sys.exit(1)
    else:
        print(f"\n✅ SUCCESS: {description} completed")

def main():
    parser = argparse.ArgumentParser(description="Run complete analysis pipeline for a model family")
    parser.add_argument("--family", required=True, choices=["llama", "qwen", "mistral"],
                       help="Model family to analyze")
    parser.add_argument("--prompts_per_category", type=int, default=None,
                       help="Number of prompts per category (overrides config default)")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                       help="Device to use (overrides config default)")
    parser.add_argument("--skip_analysis", action="store_true",
                       help="Skip activation analysis (assume data already exists)")
    parser.add_argument("--skip_visualization", action="store_true", 
                       help="Skip visualization (assume figures already exist)")
    parser.add_argument("--skip_comparison", action="store_true",
                       help="Skip comparison (assume comparison already exists)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    family_config = config["model_families"][args.family]
    defaults = config["defaults"]
    
    # Set parameters
    prompts_per_category = args.prompts_per_category or defaults["prompts_per_category"]
    device = args.device or defaults["device"]
    seed = defaults["seed"]
    
    print(f"\n🚀 STARTING PIPELINE FOR {args.family.upper()} FAMILY")
    print(f"   Prompts per category: {prompts_per_category}")
    print(f"   Device: {device}")
    print(f"   Seed: {seed}")
    
    # Step 1: Run activation analysis for base model
    if not args.skip_analysis:
        base_config = family_config["base"]
        cmd = [
            "python", "activation_analysis.py",
            "--model_id", base_config["model_id"],
            "--device", device,
            "--prompts_per_category", str(prompts_per_category),
            "--output_type", "latest_base",  # Always use latest_base for base models
            "--output_dir", base_config["output_dir"],
            "--seed", str(seed)
        ]
        run_command(cmd, f"Base model activation analysis ({args.family})")
        
        # Step 2: Run activation analysis for instruct model
        instruct_config = family_config["instruct"]
        cmd = [
            "python", "activation_analysis.py",
            "--model_id", instruct_config["model_id"],
            "--device", device,
            "--prompts_per_category", str(prompts_per_category),
            "--output_type", "latest",  # Always use latest for instruct models
            "--output_dir", instruct_config["output_dir"],
            "--seed", str(seed)
        ]
        run_command(cmd, f"Instruct model activation analysis ({args.family})")
    
    # Step 3: Visualize base model results
    if not args.skip_visualization:
        base_config = family_config["base"]
        cmd = [
            "python", "visualize_results.py",
            "--output_type", base_config["output_type"],
            "--input_dir", f"{base_config['output_dir']}/latest_base",
            "--model_name", base_config["model_id"],
            "--output_dir", base_config["figures_dir"]
        ]
        run_command(cmd, f"Base model visualization ({args.family})")
        
        # Step 4: Visualize instruct model results
        instruct_config = family_config["instruct"]
        cmd = [
            "python", "visualize_results.py",
            "--output_type", instruct_config["output_type"],
            "--input_dir", f"{instruct_config['output_dir']}/latest_run",
            "--model_name", instruct_config["model_id"],
            "--output_dir", instruct_config["figures_dir"]
        ]
        run_command(cmd, f"Instruct model visualization ({args.family})")
    
    # Step 5: Run comparison
    if not args.skip_comparison:
        comparison_config = family_config["comparison"]
        cmd = [
            "python", "compare_base_instruct.py",
            "--base_dir", comparison_config["base_dir"],
            "--instruct_dir", comparison_config["instruct_dir"],
            "--output_dir", comparison_config["output_dir"],
            "--family", args.family
        ]
        run_command(cmd, f"Base vs Instruct comparison ({args.family})")
    
    print(f"\n🎉 PIPELINE COMPLETE FOR {args.family.upper()} FAMILY!")
    print(f"\nResults saved to:")
    print(f"   Base figures: {family_config['base']['figures_dir']}/")
    print(f"   Instruct figures: {family_config['instruct']['figures_dir']}/")
    print(f"   Comparison: {family_config['comparison']['output_dir']}/")

if __name__ == "__main__":
    main()
