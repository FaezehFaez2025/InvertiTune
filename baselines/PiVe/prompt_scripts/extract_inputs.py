#!/usr/bin/env python3
"""
Script to extract 'input' fields from T2G_test.json and create a test.target file.
"""

import json
import os

def extract_inputs_to_target():
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to Text_to_KG_SFT directory: prompt_scripts -> PiVe -> baselines -> Text_to_KG_SFT
    text_to_kg_sft_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    # Input file path
    input_file = os.path.join(text_to_kg_sft_dir, "LLaMA-Factory", "data", "T2G_test.json")
    
    # Output file path (in the GPT3.5_result_KELM directory)
    output_file = os.path.join(script_dir, "GPT3.5_result_KELM", "test.target")
    
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    
    try:
        # Read the JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Found {len(data)} entries in the JSON file")
        
        # Extract inputs and write to target file
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, entry in enumerate(data):
                if 'input' in entry:
                    # Replace newlines within the input text with spaces to ensure one line per entry
                    input_text = entry['input'].replace('\n', ' ').replace('\r', ' ')
                    # Write each input as a single line
                    f.write(input_text + '\n')
                else:
                    print(f"Warning: Entry {i} does not have an 'input' field")
        
        print(f"Successfully extracted {len(data)} input texts to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file at {input_file}")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = extract_inputs_to_target()
    if success:
        print("Input extraction completed successfully!")
    else:
        print("Input extraction failed!")
