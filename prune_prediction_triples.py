import json
import re
from pathlib import Path
import argparse

def extract_valid_triples(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_valid_triples = 0
    valid_samples = 0
    invalid_samples = 0
    improved_data = []
    
    for idx, item in enumerate(data, 1):
        # Copy the original item to preserve text and ground_truth
        improved_item = item.copy()
        valid_triples = []
        
        prediction_str = item.get('prediction')
        
        try:
            valid_triple_pattern = r'\["([^"]*)", "([^"]*)", "([^"]*)"\]'

            matches = re.finditer(valid_triple_pattern, prediction_str)
            for match in matches:
                subj, pred, obj = match.groups()
                valid_triples.append([subj, pred, obj])
            
            # Extract remaining content for debugging
            remaining = prediction_str
            for match in re.finditer(valid_triple_pattern, prediction_str):
                remaining = remaining.replace(match.group(0), '[]')
            
            # Clean up the remaining string to remove valid syntax
            remaining = remaining.replace('[],', '')
            remaining = remaining.strip('[]')
            remaining = remaining.strip()
            
            if remaining:
                print(f"Sample {idx}: Found {len(valid_triples)} valid triples")
                print(f"  Problematic content remaining after extracting valid triples:")
                print(remaining)
                invalid_samples += 1
            else:
                print(f"Sample {idx}: All {len(valid_triples)} triples are valid")
                valid_samples += 1
            
            # Manually construct the JSON string for the prediction field
            if not valid_triples:
                improved_item['prediction'] = "[]"
            else:
                triple_strings = []
                for triple in valid_triples:
                    triple_str = f'["{triple[0]}", "{triple[1]}", "{triple[2]}"]'
                    triple_strings.append(triple_str)
                # Add a space between triples for better readability
                improved_item['prediction'] = f"[{', '.join(triple_strings)}]"
            
            improved_data.append(improved_item)
            
            # Update statistics
            total_valid_triples += len(valid_triples)
                
        except Exception as e:
            print(f"\nSample {idx}: Processing error - {str(e)}")
            invalid_samples += 1
            improved_item['prediction'] = "[]"  # Empty list as a JSON string
            improved_data.append(improved_item)
    
    # Save the improved data with proper JSON formatting
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(improved_data, f, ensure_ascii=False, indent=2)
    
    return {
        'total_valid_triples': total_valid_triples,
        'valid_samples': valid_samples,
        'invalid_samples': invalid_samples,
        'output_file': output_path
    }

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract valid triples from prediction JSON file')
    parser.add_argument('--input_filename', type=str, 
                        default='test_predictions_finetuned_1.5B.json',
                        help='Input filename in LLaMA-Factory/results/ directory')
    args = parser.parse_args()
    
    # Base directory for input and output files
    base_dir = 'LLaMA-Factory/results/'
    
    # Construct input and output file paths
    input_file = base_dir + args.input_filename
    
    # Generate output filename by adding '_improved' before the extension
    input_path = Path(args.input_filename)
    output_filename = f"{input_path.stem}_improved{input_path.suffix}"
    output_file = base_dir + output_filename
    
    print(f"Analyzing: {Path(input_file).name}\n")
    print("Processing samples and extracting valid triples...")
    print("=" * 80)
    
    results = extract_valid_triples(input_file, output_file)
    
    print("=" * 80)
    print("\nValidation Summary:")
    print(f"Total valid triples extracted: {results['total_valid_triples']}")
    print(f"Perfect samples: {results['valid_samples']}")
    print(f"Samples with issues: {results['invalid_samples']}")
    
    print(f"\nImproved data saved to: {results['output_file']}")

if __name__ == "__main__":
    main()