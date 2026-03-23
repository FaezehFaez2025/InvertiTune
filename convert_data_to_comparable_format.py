import json
import os
import ast
import re
import argparse

def process_json_file(input_file, output_file, data_type):
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Write the triplets to the output file - one sample per line
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            if data_type in item:
                # Extract triplets for this sample
                sample_triplets = []

                triplets = ast.literal_eval(item[data_type])
                sample_triplets.extend(triplets)
                
                # Format triplets with quotes and uppercase for subject and object
                formatted_triplets = []
                for triplet in sample_triplets:
                    if len(triplet) == 3:
                        # Clean up any potential extra characters
                        subject = triplet[0].strip('[]"').upper()
                        predicate = triplet[1].strip('[]"')
                        obj = triplet[2].strip('[]"').upper()
                        
                        formatted_triplet = [subject, predicate, obj]
                        formatted_triplets.append(formatted_triplet)
                
                # Write this sample's triplets on a single line
                f.write('[')
                for i, triplet in enumerate(formatted_triplets):
                    f.write(f'["{triplet[0]}", "{triplet[1]}", "{triplet[2]}"]')
                    if i < len(formatted_triplets) - 1:
                        f.write(', ')
                f.write(']\n')

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert model outputs to a suitable format for baseline comparison')
    parser.add_argument('filename', type=str, nargs='?',
                        default='test_predictions_finetuned_1.5B_improved.json',
                        help='Path to the JSON file (relative or absolute; defaults to LLaMA-Factory/results/ if just filename provided)')
    parser.add_argument('--output_dir', '-o', type=str,
                        default='../../result/controlled_extraction/test',
                        help='Directory to save the output file')
    parser.add_argument('--data_type', '-d', type=str, choices=['ground_truth', 'prediction'],
                        default='prediction',
                        help='Type of data to convert')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Define input and output paths
    # Check if filename is absolute or contains a path separator
    if os.path.isabs(args.filename) or os.path.dirname(args.filename):
        input_file = args.filename
    else:
        input_file = os.path.join('LLaMA-Factory/results/', args.filename)
    output_dir = args.output_dir
    
    # Generate output filename based on input filename
    base_name = os.path.splitext(args.filename)[0]
    parts = base_name.split('_')
    if len(parts) >= 3:
        # Skip the first two parts, join the rest
        base_name = '_'.join(parts[2:])
    
    # Include data_type in the output filename
    output_file = os.path.join(output_dir, f'aggregated_{base_name}_{args.data_type}_triplets.txt')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the file
    process_json_file(input_file, output_file, args.data_type)
    print(f"Successfully generated {output_file}")

if __name__ == '__main__':
    main() 
