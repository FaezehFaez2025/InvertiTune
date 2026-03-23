#!/usr/bin/env python3
import os
import shutil
import argparse

def copy_results(base_dir="GPT3.5_result_KELMs", iteration=1):
    """
    Copies results by creating a results/original_prediction directory structure
    and copying test_generated_graphs.txt as aggregated_pive_triplets.txt from the specified iteration.
    
    Args:
        base_dir (str): Base directory containing iteration folders
        iteration (int): Iteration number to process
    """
    # Create paths
    iteration_dir = os.path.join(base_dir, f"Iteration{iteration}")
    results_dir = "results"  # Create results folder in current directory (prompt_scripts)
    original_pred_dir = os.path.join(results_dir, "original_prediction")
    
    # Create directory structure
    os.makedirs(original_pred_dir, exist_ok=True)
    
    # Copy test_generated_graphs.txt as aggregated_pive_triplets.txt
    source_file = os.path.join(iteration_dir, "test_generated_graphs.txt")
    dest_file = os.path.join(original_pred_dir, "aggregated_pive_triplets.txt")
    
    # Check if source file exists
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file not found: {source_file}")
    
    # Copy the file with new name
    shutil.copy2(source_file, dest_file)
    print(f"Successfully copied {source_file} to {dest_file}")

def main():
    parser = argparse.ArgumentParser(description="Copy results into a structured directory")
    parser.add_argument("--base_dir", default="GPT3.5_result_KELMs",
                      help="Base directory containing iteration folders")
    parser.add_argument("--iteration", type=int, required=True,
                      help="Iteration number to process")
    
    args = parser.parse_args()
    
    try:
        copy_results(args.base_dir, args.iteration)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main() 