#!/usr/bin/env python3
import os
import argparse
import numpy as np
import ast
from pathlib import Path

def calculate_statistics(dataset_name, partition):
    """
    Calculate statistics for a dataset partition.
    
    Args:
        dataset_name: Name of the dataset folder (e.g., 'webnlg20', 'kelm_sub')
        partition: Partition name (e.g., 'train', 'val', 'test')
    """
    # Get the base directory
    base_dir = Path(__file__).parent
    dataset_dir = base_dir / dataset_name
    
    # Check if dataset directory exists
    if not dataset_dir.exists():
        print(f"Error: Dataset directory '{dataset_name}' not found.")
        return
    
    # Define source and target file paths
    source_file = dataset_dir / f"{partition}.source"
    target_file = dataset_dir / f"{partition}.target"
    
    # Check if files exist
    if not source_file.exists():
        print(f"Error: Source file '{source_file}' not found.")
        return
    
    if not target_file.exists():
        print(f"Error: Target file '{target_file}' not found.")
        return
    
    # Calculate triple statistics from source file
    print(f"\n=== Triple Statistics for {dataset_name}/{partition} ===")
    calculate_triple_statistics(source_file)
    
    # Calculate token statistics from target file
    print(f"\n=== Token Statistics for {dataset_name}/{partition} ===")
    calculate_token_statistics(target_file)

def calculate_triple_statistics(source_file):
    """Calculate statistics for triples in the source file."""
    triples_per_file = []
    
    with open(source_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse the line as a Python list of triples
            try:
                # The format is a list of triples, e.g., [["subject", "predicate", "object"]]
                triple_list = ast.literal_eval(line.strip())
                # Count the number of triples in this line
                triples_per_file.append(len(triple_list))
            except (SyntaxError, ValueError) as e:
                print(f"Warning: Could not parse line: {line.strip()}")
                print(f"Error: {e}")
                triples_per_file.append(0)
    
    # Calculate statistics
    total_files = len(triples_per_file)
    total_triples = sum(triples_per_file)
    avg_triples = total_triples / total_files if total_files > 0 else 0
    max_triples = max(triples_per_file) if triples_per_file else 0
    min_triples = min(triples_per_file) if triples_per_file else 0
    std_dev_triples = np.std(triples_per_file) if triples_per_file else 0
    
    # Print statistics
    print(f"Number of data samples: {total_files}")
    print(f"Total number of triples: {total_triples}")
    print(f"Average number of triples per data sample: {avg_triples:.2f}")
    print(f"Maximum number of triples in a data sample: {max_triples}")
    print(f"Minimum number of triples in a data sample: {min_triples}")
    print(f"Standard deviation of the number of triples per data sample: {std_dev_triples:.2f}")

def calculate_token_statistics(target_file):
    """Calculate statistics for tokens in the target file."""
    tokens_per_file = []
    
    with open(target_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Count tokens in each line
            tokens = line.strip().split()
            tokens_per_file.append(len(tokens))
    
    # Calculate statistics
    total_files = len(tokens_per_file)
    total_tokens = sum(tokens_per_file)
    avg_tokens = total_tokens / total_files if total_files > 0 else 0
    max_tokens = max(tokens_per_file) if tokens_per_file else 0
    min_tokens = min(tokens_per_file) if tokens_per_file else 0
    std_dev_tokens = np.std(tokens_per_file) if tokens_per_file else 0
    
    # Print statistics
    print(f"Number of data samples: {total_files}")
    print(f"Total number of tokens: {total_tokens}")
    print(f"Average number of tokens per data sample: {avg_tokens:.2f}")
    print(f"Maximum number of tokens in a data sample: {max_tokens}")
    print(f"Minimum number of tokens in a data sample: {min_tokens}")
    print(f"Standard deviation of the number of tokens per data sample: {std_dev_tokens:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Calculate statistics for dataset files.")
    parser.add_argument("--dataset", choices=["webnlg20", "kelm_sub"], default="kelm_sub",
                        help="Name of the dataset folder")
    parser.add_argument("--partition", choices=["train", "val", "test"], default="test", 
                        help="Partition name")
    
    args = parser.parse_args()
    calculate_statistics(args.dataset, args.partition)

if __name__ == "__main__":
    main() 