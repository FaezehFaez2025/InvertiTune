import os
import json
import shutil
import re

def convert_t2g_to_jsonl(input_file, output_dir):
    """
    Convert T2G_test.json to JSONL format with docid, text, and triples fields.
    
    Args:
        input_file (str): Path to the input T2G_test.json file
        output_dir (str): Path to the output directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create output JSONL file
    output_file = os.path.join(output_dir, 'en_test.jsonl')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, sample in enumerate(data, start=1):
            # Extract input text
            text = sample.get('input', '')
            
            # Extract and convert triples
            triples_str = sample.get('output', '[]')
            
            # Parse the string representation of triples
            formatted_triples = []
            
            # The format is like: [[subject, predicate, object], [subject, predicate, object], ...]
            # We need to extract each triple from the string
            
            # First, remove the outer brackets
            if triples_str.startswith('[') and triples_str.endswith(']'):
                triples_str = triples_str[1:-1]
            
            # Use a more robust approach to extract triples
            # Look for patterns like [subject, predicate, object]
            triple_pattern = r'\[(.*?)\]'
            triples_matches = re.findall(triple_pattern, triples_str)
            
            for triple_match in triples_matches:
                # Split by comma and handle potential commas within quotes
                parts = []
                current_part = ""
                in_quotes = False
                
                for char in triple_match:
                    if char == '"' and (len(current_part) == 0 or current_part[-1] != '\\'):
                        in_quotes = not in_quotes
                        current_part += char
                    elif char == ',' and not in_quotes:
                        parts.append(current_part.strip())
                        current_part = ""
                    else:
                        current_part += char
                
                if current_part:
                    parts.append(current_part.strip())
                
                # Clean up the parts (remove quotes)
                parts = [part.strip('"') for part in parts]
                
                if len(parts) == 3:
                    formatted_triple = {
                        "subject": {"surfaceform": parts[0]},
                        "predicate": {"surfaceform": parts[1]},
                        "object": {"surfaceform": parts[2]}
                    }
                    formatted_triples.append(formatted_triple)
            
            # Create the output sample
            output_sample = {
                "docid": str(idx),
                "text": text,
                "triples": formatted_triples
            }
            
            # Write to JSONL file
            f.write(json.dumps(output_sample, ensure_ascii=False) + '\n')
    
    print(f"Conversion complete. Output saved to {output_file}")
    print(f"Processed {len(data)} samples.")

if __name__ == "__main__":
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "LLaMA-Factory", "data", "T2G_test.json")
    
    # Create dataset directory - using relative path
    dataset_name = "controlled_extraction_dataset"
    # Use a relative path to the datasets directory
    datasets_dir = os.path.join("..", "..", "datasets")
    dataset_dir = os.path.join(datasets_dir, dataset_name)
    
    print(f"Creating dataset directory at: {dataset_dir}")
    
    # Convert the data
    convert_t2g_to_jsonl(input_file, dataset_dir) 