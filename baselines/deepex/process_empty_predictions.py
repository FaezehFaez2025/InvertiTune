#!/usr/bin/env python3
"""
Script to process text files and replace empty predictions [] with [["", "", ""]]

Usage:
    python process_empty_predictions.py <filename> [--postfix POSTFIX]

Arguments:
    filename: Name of the .txt file in the test folder (can include subdirectory path)
    --postfix: Optional postfix to add to output filename (default: "_processed")

Examples:
    python process_empty_predictions.py aggregated_chatgpt_triplets.txt
    python process_empty_predictions.py original_prediction/aggregated_chatgpt_triplets.txt --postfix _fixed

The script reads from /root/NeoGraphRAG/result/controlled_extraction/test/<filename>
and writes to /root/NeoGraphRAG/result/controlled_extraction/test/<filename_with_postfix>
"""

import os
import sys
import argparse


def process_file(input_filename, postfix="_processed"):
    """
    Process a text file by replacing empty predictions [] with [["", "", ""]]
    
    Args:
        input_filename (str): Name of the input file (can include subdirectory path)
        postfix (str): Postfix to add to the output filename
    """
    # Define paths
    test_folder = "/root/NeoGraphRAG/result/controlled_extraction/test"
    input_path = os.path.join(test_folder, input_filename)
    
    # Create output filename by inserting postfix before file extension
    # Handle subdirectories properly
    dir_part = os.path.dirname(input_filename)
    base_name = os.path.basename(input_filename)
    name, ext = os.path.splitext(base_name)
    output_filename = os.path.join(dir_part, f"{name}{postfix}{ext}")
    output_path = os.path.join(test_folder, output_filename)
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return False
    
    try:
        # Read and process the file
        with open(input_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
        
        processed_lines = []
        changes_made = 0
        
        for line_num, line in enumerate(lines, 1):
            # Strip whitespace and check if line contains only []
            stripped_line = line.strip()
            
            if stripped_line == '[]':
                # Replace empty prediction with [["", "", ""]]
                processed_line = '[["", "", ""]]\n'
                processed_lines.append(processed_line)
                changes_made += 1
                print(f"Line {line_num}: Replaced [] with [['', '', '']]")
            else:
                # Keep original line
                processed_lines.append(line)
        
        # Write processed content to output file
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.writelines(processed_lines)
        
        print(f"\nProcessing completed!")
        print(f"Input file: {input_path}")
        print(f"Output file: {output_path}")
        print(f"Total lines processed: {len(lines)}")
        print(f"Empty predictions replaced: {changes_made}")
        
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Process text files by replacing empty predictions [] with [['', '', '']]"
    )
    parser.add_argument(
        'filename',
        help='Name of the .txt file in the test folder (can include subdirectory path)'
    )
    parser.add_argument(
        '--postfix',
        default='_processed',
        help='Postfix to add to output filename (default: _processed)'
    )
    
    args = parser.parse_args()
    
    # Validate filename
    if not args.filename.endswith('.txt'):
        print("Warning: File does not have .txt extension")
    
    # Process the file
    success = process_file(args.filename, args.postfix)
    
    if success:
        print("File processed successfully!")
        sys.exit(0)
    else:
        print("File processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
