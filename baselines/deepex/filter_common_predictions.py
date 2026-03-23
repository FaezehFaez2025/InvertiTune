#!/usr/bin/env python3
import os
import json
import ast
import sys

def read_predictions(file_path):
    """Read predictions from a file and return list of predictions."""
    predictions = []
    try:
        with open(file_path, 'r') as f:
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
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    return predictions

def is_empty_prediction(pred):
    """Check if a prediction is empty."""
    return not pred or pred == [] or pred == [[]]

def main():
    if len(sys.argv) != 2:
        print("Usage: python filter_common_predictions.py <base_directory>")
        print("Example: python filter_common_predictions.py results_KELM/result/controlled_extraction/test")
        sys.exit(1)
    
    # Get base directory from command line argument
    base_dir = sys.argv[1]
    if not os.path.isdir(base_dir):
        print(f"Error: '{base_dir}' is not a valid directory")
        sys.exit(1)
    
    pred_dir = os.path.join(base_dir, "original_prediction")
    if not os.path.isdir(pred_dir):
        print(f"Error: '{pred_dir}' is not a valid directory")
        sys.exit(1)
    
    # Create output directory
    output_dir = os.path.join(base_dir, "common_predictions_after_deepex")
    os.makedirs(output_dir, exist_ok=True)
    
    # Files to process
    files = {
        'chatgpt': 'aggregated_chatgpt_triplets.txt',
        'deepex': 'aggregated_deepex_triplets.txt',
        'finetuned': 'aggregated_finetuned_1.5B_improved_prediction_triplets.txt',
        'finetuned_kelm': 'aggregated_finetuned_1.5B_KELM_improved_prediction_triplets.txt',
        'graphrag': 'aggregated_graphrag_triplets.txt',
        'ground_truth': 'aggregated_ground_truth_triplets.txt',
        'lightrag': 'aggregated_lightrag_triplets.txt',
        'openie6': 'aggregated_openie6_triplets.txt'
    }
    
    # Read all predictions
    predictions = {}
    for name, filename in files.items():
        file_path = os.path.join(pred_dir, filename)
        preds = read_predictions(file_path)
        if preds is None:
            print(f"Error: Could not read predictions from {filename}")
            return
        predictions[name] = preds
    
    # Get the length of predictions (should be the same for all files)
    num_samples = len(predictions['deepex'])
    print(f"Total number of samples: {num_samples}")
    
    # Count cases where DeepEx adds empty predictions
    empty_added_by_deepex = 0
    common_predictions = []
    
    for i in range(num_samples):
        # Check if all other methods have non-empty predictions
        others_non_empty = True
        for name in predictions:
            if name != 'deepex' and is_empty_prediction(predictions[name][i]):
                others_non_empty = False
                break
        
        # If all others are non-empty but DeepEx is empty, count it
        if others_non_empty and is_empty_prediction(predictions['deepex'][i]):
            empty_added_by_deepex += 1
            continue
        
        # Check if all predictions are non-empty
        all_non_empty = True
        for name in predictions:
            if is_empty_prediction(predictions[name][i]):
                all_non_empty = False
                break
        
        # If all predictions are non-empty, add to common predictions
        if all_non_empty:
            common_predictions.append(i)
    
    # Write filtered predictions to new files
    for name, preds in predictions.items():
        output_file = os.path.join(output_dir, files[name])
        with open(output_file, 'w') as f:
            for idx in common_predictions:
                f.write(json.dumps(preds[idx]) + '\n')
    
    print(f"\nAnalysis Results:")
    print(f"New empty predictions added by DeepEx: {empty_added_by_deepex}")
    print(f"Total common non-empty predictions: {len(common_predictions)}")
    print(f"\nFiltered predictions written to: {output_dir}")

if __name__ == "__main__":
    main() 