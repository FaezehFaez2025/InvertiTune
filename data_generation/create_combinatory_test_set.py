#!/usr/bin/env python3
"""
Script to create a combinatory test set by selecting n random samples from each of the four datasets:
- genwiki_hiq
- KELM  
- webnlg20
- CE12000_diverse

The result is a T2G_test.json file with 4*n samples.
"""

import json
import random
import argparse
import os
from pathlib import Path


def load_dataset(file_path):
    """Load a dataset from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Warning: JSON decode error in {file_path}: {e}")
        return []


def sample_dataset(dataset, n_samples, dataset_name):
    """Sample n random samples from a dataset."""
    if len(dataset) == 0:
        print(f"Warning: Empty dataset for {dataset_name}")
        return []
    
    if n_samples > len(dataset):
        print(f"Warning: Requested {n_samples} samples from {dataset_name}, but only {len(dataset)} available. Using all samples.")
        n_samples = len(dataset)
    
    sampled = random.sample(dataset, n_samples)
    print(f"Sampled {len(sampled)} samples from {dataset_name}")
    return sampled


def create_combinatory_test_set(n_samples_per_dataset, output_file="T2G_test.json"):
    """
    Create a combinatory test set by sampling from all four datasets.
    
    Args:
        n_samples_per_dataset (int): Number of samples to take from each dataset
        output_file (str): Output file name
    """
    
    # Define dataset paths
    base_path = Path(__file__).parent.parent / "dataset_investigation" / "datasets"
    
    datasets = {
        "genwiki_hiq": base_path / "genwiki_hiq" / "T2G_test.json",
        "KELM": base_path / "KELM" / "T2G_test.json", 
        "webnlg20": base_path / "webnlg20" / "T2G_test.json",
        "CE12000_diverse": base_path / "CE12000_diverse" / "T2G_test.json"
    }
    
    # Load all datasets
    print("Loading datasets...")
    loaded_datasets = {}
    for name, path in datasets.items():
        data = load_dataset(path)
        loaded_datasets[name] = data
        print(f"Loaded {len(data)} samples from {name}")
    
    # Sample from each dataset
    print(f"\nSampling {n_samples_per_dataset} samples from each dataset...")
    combined_samples = []
    
    for name, dataset in loaded_datasets.items():
        samples = sample_dataset(dataset, n_samples_per_dataset, name)
        combined_samples.extend(samples)
    
    # Save the combined dataset
    print(f"\nSaving combined dataset with {len(combined_samples)} samples to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_samples, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully created combinatory test set with {len(combined_samples)} samples!")
    print(f"Output file: {output_file}")
    
    # Print summary
    print("\nSummary:")
    print(f"- Total samples: {len(combined_samples)}")
    print(f"- Samples per dataset: {n_samples_per_dataset}")
    print(f"- Number of datasets: {len(datasets)}")


def main():
    parser = argparse.ArgumentParser(description="Create a combinatory test set from multiple datasets")
    parser.add_argument(
        "--n_samples", 
        type=int, 
        default=300,
        help="Number of samples to take from each dataset (default: 100)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="T2G_test.json",
        help="Output file name (default: T2G_test.json)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed for reproducibility (default: None)"
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Create the combinatory test set
    create_combinatory_test_set(args.n_samples, args.output)


if __name__ == "__main__":
    main() 