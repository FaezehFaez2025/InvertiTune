#!/usr/bin/env python3
import sys
import os

def main(result_dir, num_folders):
    # Ensure the path exists
    if not os.path.exists(result_dir):
        print(f"Error: Directory {result_dir} does not exist")
        sys.exit(1)
        
    # Get output directory
    output_dir = os.path.join(result_dir, "original_prediction")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file
    output_file = os.path.join(output_dir, "aggregated_deepex_triplets.txt")
    
    print(f"Processing {num_folders} folders...")
    
    # Process folders from 1 to num_folders
    with open(output_file, 'w') as outf:
        processed_count = 0
        for folder_num in range(1, num_folders + 1):
            folder_path = os.path.join(result_dir, str(folder_num))
            deepex_file = os.path.join(folder_path, "deepex.txt")
            
            if not os.path.exists(deepex_file):
                print(f"Warning: {deepex_file} not found")
                continue
                
            print(f"Processing folder {folder_num}...")
            try:
                with open(deepex_file, 'r') as f:
                    content = f.read().strip()
                    if content:  # Only write non-empty content
                        outf.write(content + '\n')
                        processed_count += 1
            except Exception as e:
                print(f"Error processing {deepex_file}: {e}")
    
    print(f"Results written to {output_file}")
    print(f"Total folders processed: {processed_count}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python aggregate_deepex.py <result_directory> <num_folders>")
        print("Example: python aggregate_deepex.py results_KELM/result/controlled_extraction/test 1800")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    try:
        num_folders = int(sys.argv[2])
        if num_folders < 1:
            raise ValueError("Number of folders must be positive")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    main(result_dir, num_folders) 