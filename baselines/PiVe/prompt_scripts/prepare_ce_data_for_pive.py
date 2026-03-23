#!/usr/bin/env python3
import os
import argparse
import glob

def process_test_folders(num_folders):
    """
    Process test folders from 1 to num_folders.
    For each folder, read text.txt, remove newlines, and append to test.target.
    
    Args:
        num_folders (int): Number of folders to process
    """
    # Path to the test directory
    test_dir = "result/controlled_extraction/test"
    
    # Path to test.target file
    test_target_file = "GPT3.5_result_KELM/test.target"
    
    # Clear the test.target file first
    print(f"Clearing {test_target_file}...")
    with open(test_target_file, 'w', encoding='utf-8') as f:
        f.write('')
    
    print(f"Processing {num_folders} folders from {test_dir}...")
    
    processed_count = 0
    
    for folder_num in range(1, num_folders + 1):
        folder_path = os.path.join(test_dir, str(folder_num))
        text_file_path = os.path.join(folder_path, "text.txt")
        
        # Check if folder and text.txt exist
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist, skipping...")
            continue
            
        if not os.path.exists(text_file_path):
            print(f"Warning: text.txt not found in {folder_path}, skipping...")
            continue
        
        try:
            # Read the text.txt file
            with open(text_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove all newlines and extra whitespace
            processed_content = ' '.join(content.split())
            
            # Append to test.target file
            with open(test_target_file, 'a', encoding='utf-8') as f:
                f.write(processed_content + '\n')
            
            processed_count += 1
            print(f"Processed folder {folder_num}: {len(processed_content)} characters")
            
        except Exception as e:
            print(f"Error processing folder {folder_num}: {e}")
            continue
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {processed_count} out of {num_folders} folders")
    print(f"Results saved to {test_target_file}")

def main():
    parser = argparse.ArgumentParser(description="Process test folders and create test.target file")
    parser.add_argument("num_folders", type=int, help="Number of folders to process (1 to this number)")
    
    args = parser.parse_args()
    
    if args.num_folders < 1:
        print("Error: Number of folders must be at least 1")
        return
    
    process_test_folders(args.num_folders)

if __name__ == "__main__":
    main() 