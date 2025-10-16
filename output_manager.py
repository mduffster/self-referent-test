"""
Output management for self-referent experiment results.
"""

import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd

class OutputManager:
    """Manages saving experiment results to various formats."""
    
    def __init__(self, output_dir="results", use_latest=False, use_intervention=False):
        """
        Initialize output manager.
        
        Args:
            output_dir: Directory to save results
            use_latest: If True, use 'latest_run' folder instead of timestamped
            use_intervention: If True, use 'latest_intervention' folder instead of timestamped
        """
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.use_latest = use_latest
        self.use_intervention = use_intervention
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectory
        if use_intervention:
            self.run_dir = os.path.join(output_dir, "latest_intervention")
        elif use_latest:
            self.run_dir = os.path.join(output_dir, "latest_run")
        else:
            self.run_dir = os.path.join(output_dir, f"run_{self.timestamp}")
        
        os.makedirs(self.run_dir, exist_ok=True)
        
        print(f"Output directory: {self.run_dir}")
    
    def save_generation_results(self, results: List[Dict[str, Any]], filename="generation_results.json"):
        """Save text generation results."""
        filepath = os.path.join(self.run_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Generation results saved to: {filepath}")
        return filepath
    
    def save_activation_data(self, activations: Dict[str, Any], filename="activations.json"):
        """Save activation data."""
        filepath = os.path.join(self.run_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        def make_serializable(obj):
            if hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_data = make_serializable(activations)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"✓ Activation data saved to: {filepath}")
        return filepath
    
    def save_attention_data(self, attention_data: Dict[str, Any], filename="attention_patterns.json"):
        """Save attention pattern data."""
        filepath = os.path.join(self.run_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(attention_data, f, indent=2)
        
        print(f"✓ Attention data saved to: {filepath}")
        return filepath
    
    def save_comparison_results(self, results: pd.DataFrame, filename="comparison_results.csv"):
        """Save comparison results as CSV."""
        filepath = os.path.join(self.run_dir, filename)
        results.to_csv(filepath, index=False)
        
        print(f"✓ Comparison results saved to: {filepath}")
        return filepath
    
    def save_experiment_config(self, config: Dict[str, Any], filename="experiment_config.json"):
        """Save experiment configuration."""
        filepath = os.path.join(self.run_dir, filename)
        
        # Add timestamp to config
        config['timestamp'] = self.timestamp
        config['output_dir'] = self.run_dir
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Experiment config saved to: {filepath}")
        return filepath
    
    def create_summary_report(self, results_summary: Dict[str, Any], filename="summary_report.txt"):
        """Create a human-readable summary report."""
        filepath = os.path.join(self.run_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("Self-Referent Experiment Summary Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Output Directory: {self.run_dir}\n\n")
            
            for section, content in results_summary.items():
                f.write(f"{section.upper()}:\n")
                f.write("-" * len(section) + "\n")
                
                if isinstance(content, dict):
                    for key, value in content.items():
                        f.write(f"  {key}: {value}\n")
                elif isinstance(content, list):
                    for item in content:
                        f.write(f"  - {item}\n")
                else:
                    f.write(f"  {content}\n")
                f.write("\n")
        
        print(f"✓ Summary report saved to: {filepath}")
        return filepath

# Example usage and testing
if __name__ == "__main__":
    # Test the output manager
    output = OutputManager()
    
    # Test data
    test_results = [
        {"prompt": "Who are you?", "response": "I am an AI assistant.", "category": "self_referent"},
        {"prompt": "Tell a story", "response": "Once upon a time...", "category": "confounder"}
    ]
    
    output.save_generation_results(test_results)
    
    test_config = {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "prompt_count": 60,
        "categories": ["self_referent", "confounder", "neutral"]
    }
    
    output.save_experiment_config(test_config)
