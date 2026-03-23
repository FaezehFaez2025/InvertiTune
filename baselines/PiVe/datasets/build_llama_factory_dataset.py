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

def load_source_data(file_path):
    """Load triples from a source file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse the line as a Python list of triples
            try:
                triple_list = eval(line.strip())
                data.append(triple_list)
            except (SyntaxError, ValueError) as e:
                print(f"Warning: Could not parse line: {line.strip()}")
                print(f"Error: {e}")
                data.append([])
    return data

def load_target_data(file_path):
    """Load text from a target file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(line.strip())
    return data

def convert_to_llama_format(source_data, target_data):
    """Convert the data to LLaMA-Factory format."""
    llama_data = []
    
    for triples, text in zip(source_data, target_data):
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
    parser = argparse.ArgumentParser(description='Build dataset for LLaMA-Factory from PiVe datasets')
    parser.add_argument('--dataset', type=str, choices=['webnlg20', 'kelm_sub'], required=True,
                        help='Dataset to process')
    parser.add_argument('--partition', type=str, choices=['train', 'val', 'test'], required=True,
                        help='Data partition to process')
    args = parser.parse_args()
    
    # Define paths
    base_dir = Path(__file__).parent
    dataset_dir = base_dir / args.dataset
    
    # Create dataset-specific output directory using the exact dataset name
    output_dir = base_dir / 'llama_factory_data' / args.dataset
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine input and output file paths
    source_file = dataset_dir / f"{args.partition}.source"
    target_file = dataset_dir / f"{args.partition}.target"
    output_file = output_dir / f"T2G_{args.partition}.json"
    
    print(f"Building LLaMA-Factory dataset from {args.dataset}/{args.partition} data...")
    print(f"Source file: {source_file}")
    print(f"Target file: {target_file}")
    print(f"Output file: {output_file}")
    
    # Check if files exist
    if not source_file.exists():
        print(f"Error: Source file '{source_file}' not found.")
        return
    
    if not target_file.exists():
        print(f"Error: Target file '{target_file}' not found.")
        return
    
    # Load and process data
    source_data = load_source_data(source_file)
    target_data = load_target_data(target_file)
    
    # Verify that source and target have the same number of examples
    if len(source_data) != len(target_data):
        print(f"Error: Source and target have different numbers of examples ({len(source_data)} vs {len(target_data)})")
        return
    
    llama_data = convert_to_llama_format(source_data, target_data)
    
    # Save the processed data using custom encoder
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(llama_data, f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
    
    print(f"Successfully processed {len(llama_data)} examples")
    print(f"LLaMA-Factory dataset saved to {output_file}")

if __name__ == "__main__":
    main() 