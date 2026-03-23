#!/usr/bin/env python3
import os
import json
import ast
import argparse

def read_predictions(file_path):
    """Read predictions from a file and return list of predictions."""
    predictions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        # Handle both string representation of lists and direct JSON
                        try:
                            pred = ast.literal_eval(line)
                        except:
                            pred = json.loads(line)
                        predictions.append(pred)
                    except:
                        print(f"Warning: Could not parse line in {file_path}: {line}")
                        predictions.append([])  # Add empty list for unparseable lines
                else:
                    predictions.append([])  # Add empty list for empty lines
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    return predictions

def is_empty_prediction(pred):
    """Check if a prediction is empty."""
    return not pred or pred == [] or pred == [[]]

def find_common_triples(results_dir="results"):
    """
    Find triples that are non-empty at the same position across all files.
    
    Args:
        results_dir (str): Directory containing the results
    """
    # Get all subdirectories in results
    subdirs = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            subdirs.append(item_path)
    
    if not subdirs:
        print("No subdirectories found in results directory")
        return
    
    # Dictionary to store predictions from each file
    file_predictions = {}
    
    # Read predictions from all files in all subdirectories
    for subdir in subdirs:
        for filename in os.listdir(subdir):
            if filename.endswith('.txt'):
                file_path = os.path.join(subdir, filename)
                predictions = read_predictions(file_path)
                if predictions is not None:  # Only include files that were read successfully
                    file_predictions[filename] = predictions
                    print(f"Found {len(predictions)} predictions in {filename}")
    
    if not file_predictions:
        print("No files with predictions found")
        return
    
    # Get the length of predictions (should be the same for all files)
    num_samples = len(list(file_predictions.values())[0])
    print(f"Total number of samples: {num_samples}")
    
    # Find positions where all files have non-empty predictions
    common_positions = []
    for i in range(num_samples):
        # Check if all predictions are non-empty at this position
        all_non_empty = True
        for filename, predictions in file_predictions.items():
            if i >= len(predictions) or is_empty_prediction(predictions[i]):
                all_non_empty = False
                break
        
        # If all predictions are non-empty, add to common positions
        if all_non_empty:
            common_positions.append(i)
    
    print(f"Found {len(common_positions)} positions where all files have non-empty predictions")
    
    # Create common_triples directory
    common_triples_dir = os.path.join(results_dir, "common_triples")
    os.makedirs(common_triples_dir, exist_ok=True)
    
    # Save predictions from common positions to each file with the same name
    for filename, predictions in file_predictions.items():
        output_file = os.path.join(common_triples_dir, filename)
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx in common_positions:
                if idx < len(predictions):
                    f.write(json.dumps(predictions[idx]) + '\n')
        print(f"Saved {len(common_positions)} predictions from common positions to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Find predictions that are non-empty at the same position across files")
    parser.add_argument("--results_dir", default="results",
                      help="Directory containing the results")
    
    args = parser.parse_args()
    
    try:
        find_common_triples(args.results_dir)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main() 