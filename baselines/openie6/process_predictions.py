import os
import re
import json
import sys

def process_predictions_file(file_path):
    """Process a single predictions.txt.oie file and extract triples."""
    triples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Find all confidence scores and their corresponding triples
        pattern = r'(\d+\.\d+): \((.*?); (.*?); (.*?)\)'
        matches = re.finditer(pattern, content)
        
        for match in matches:
            confidence, subject, relation, obj = match.groups()
            # Convert to the desired format
            triple = [subject.strip(), relation.strip(), obj.strip()]
            triples.append(triple)
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []
        
    return triples

def main():
    # Check for number of folders argument
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <number_of_folders>")
        sys.exit(1)
    try:
        num_folders = int(sys.argv[1])
    except ValueError:
        print("Error: <number_of_folders> must be an integer.")
        sys.exit(1)

    # Base directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(base_dir, "result/controlled_extraction/test")
    output_dir = os.path.join(test_dir, "original_prediction")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each folder from 1 to num_folders
    all_results = []
    for i in range(1, num_folders + 1):
        folder_path = os.path.join(test_dir, str(i))
        predictions_file = os.path.join(folder_path, "predictions.txt.oie")
        
        print(f"Processing folder {i}...")
        
        if os.path.exists(predictions_file):
            triples = process_predictions_file(predictions_file)
            all_results.append(triples)
        else:
            print(f"Warning: predictions.txt.oie not found in folder {i}")
            all_results.append([])
    
    # Write results to output file
    output_file = os.path.join(output_dir, "aggregated_openie6_triplets.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for triples in all_results:
            # Use ensure_ascii=False to preserve Unicode characters
            f.write(json.dumps(triples, ensure_ascii=False) + '\n')
    
    print(f"\nProcessing complete. Results written to {output_file}")

if __name__ == "__main__":
    main() 