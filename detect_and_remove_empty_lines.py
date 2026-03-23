import os
import shutil

# Define paths relative to script location
base_dir = '../../result/controlled_extraction/test'
target_dir = os.path.join(base_dir, 'original_prediction')

# List of files to process
files = [
    'aggregated_chatgpt_triplets.txt',
    'aggregated_finetuned_1.5B_improved_prediction_triplets.txt',
    'aggregated_graphrag_triplets.txt',
    'aggregated_lightrag_triplets.txt',
    'aggregated_ground_truth_triplets.txt',
    'aggregated_deepex_triplets.txt',
    'aggregated_pive_triplets_post_processed.txt',
    'aggregated_openie6_triplets.txt'
]

# Only create directory and copy files if original_prediction doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    # Copy files to original_prediction directory
    for file in files:
        src_path = os.path.join(base_dir, file)
        dst_path = os.path.join(target_dir, file)
        shutil.copy2(src_path, dst_path)
    print("Copied files to original_prediction directory")
else:
    print("original_prediction directory already exists, skipping file copy")

# Read all files and find empty lines
empty_lines = set()
for file in files:
    file_path = os.path.join(base_dir, file)
    with open(file_path, 'r') as f:
        for i, line in enumerate(f, 1):
            if line.strip() == '[]':
                empty_lines.add(i)

# Remove empty lines from all files
for file in files:
    file_path = os.path.join(base_dir, file)
    # Read all lines
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Write back only non-empty lines
    with open(file_path, 'w') as f:
        for i, line in enumerate(lines, 1):
            if i not in empty_lines:
                f.write(line)

print(f"Processed {len(empty_lines)} empty lines from {len(files)} files") 