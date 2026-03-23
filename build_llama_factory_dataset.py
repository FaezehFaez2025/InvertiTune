#!/usr/bin/env python
# coding=utf-8

import os
import json
import argparse
from pathlib import Path

class CustomJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        # Convert to string first
        result = super().encode(obj)
        # Replace escaped quotes with regular quotes
        return result.replace('\\"', '"')

def load_jsonl_data(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def convert_to_llama_format(data, partition):
    """Convert the data to LLaMA-Factory format."""
    llama_data = []
    
    for item in data:
        # Extract text and triples from the original data
        text = item.get('text', '')
        triples = item.get('triples', [])
        
        # Custom format triples to add quotes around elements
        formatted_triples = []
        for triple in triples:
            # Format each triple with quotes around elements
            formatted_triple = f"[\"{triple[0]}\", \"{triple[1]}\", \"{triple[2]}\"]"
            formatted_triples.append(formatted_triple)
        
        # Create the triples list as a string
        triples_str = "[" + ", ".join(formatted_triples) + "]"
        
        # Create the LLaMA format entry
        llama_entry = {
            "instruction": "Generate a knowledge graph, represented as a set of triples, from the input text.",
            "input": text,
            "output": triples_str
        }
        
        llama_data.append(llama_entry)
    
    return llama_data

def main():
    parser = argparse.ArgumentParser(description='Build dataset for LLaMA-Factory from knowledge graph extraction data')
    parser.add_argument('--partition', type=str, choices=['train', 'test'], required=True,
                        help='Data partition to process (train or test)')
    args = parser.parse_args()
    
    # Define paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data' / args.partition
    output_dir = base_dir / 'LLaMA-Factory' / 'data'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine input and output file paths
    input_file = data_dir / 'dataset.jsonl'
    output_file = output_dir / f'T2G_{args.partition}.json'
    
    print(f"Building LLaMA-Factory dataset from {args.partition} data...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Load and process data
    data = load_jsonl_data(input_file)
    llama_data = convert_to_llama_format(data, args.partition)
    
    # Save the processed data using custom encoder
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(llama_data, f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
    
    print(f"Successfully processed {len(data)} examples")
    print(f"LLaMA-Factory dataset saved to {output_file}")

if __name__ == "__main__":
    main() 