import os
import random
import shutil
import argparse
import json

# Argument parsing
parser = argparse.ArgumentParser(description='Prepare train and test datasets for fine-tuning.')
parser.add_argument('--data_folder', type=str, default='./data_generation/data',
                    help='Path to the data folder')
parser.add_argument('--source', type=str, default='wikidata', 
                    choices=['wikidata', 'yago', 'kelm_sub', 'webnlg20', 'genwiki_hiq'],
                    help='Data source (wikidata, yago, kelm_sub, webnlg20, or genwiki_hiq)')
parser.add_argument('--train_ratio', type=float, default=0.8,
                    help='Ratio of data to use for training (e.g., 0.8 for 80% train, 20% test)')
parser.add_argument('--triples_postfix', type=str, default='_triples_pruned.txt',
                    help='Postfix for triples files (default: _triples_pruned.txt)')
parser.add_argument('--num_samples', type=int, default=None,
                    help='Number of samples to process. If None, process all samples.')
args = parser.parse_args()

def check_empty_text_files(source_path):
    text_files = [f for f in os.listdir(source_path) if f.endswith('_text.txt')]
    empty_files = []
    
    for file in text_files:
        file_path = os.path.join(source_path, file)
        if os.path.getsize(file_path) == 0:
            empty_files.append(file)
            # Remove the empty text file
            os.remove(file_path)
    
    if empty_files:
        print(f"\nWARNING: Found {len(empty_files)} empty text files that have been deleted.")
        print("Please regenerate these files before proceeding with data preparation.")
        exit(1)
    
    return text_files

# Step 1: Prepare train and test datasets
def prepare_datasets(data_folder, source, train_ratio, triples_postfix, num_samples=None):
    # Path to the source data
    if source in ['wikidata', 'yago']:
        source_path = os.path.join(data_folder, source)
        
        # Delete existing train/test folders if they exist
        train_folder = os.path.join('data', 'train')
        test_folder = os.path.join('data', 'test')
        
        if os.path.exists(train_folder):
            print(f"Deleting existing folder: {train_folder}")
            shutil.rmtree(train_folder)
        
        if os.path.exists(test_folder):
            print(f"Deleting existing folder: {test_folder}")
            shutil.rmtree(test_folder)
        
        # Check for empty text files and get valid files
        text_files = check_empty_text_files(source_path)
        
        # If num_samples is specified, randomly select that many files
        if num_samples is not None and num_samples < len(text_files):
            print(f"Randomly selecting {num_samples} samples from {len(text_files)} total samples.")
            text_files = random.sample(text_files, num_samples)
        
        # Shuffle files for random splitting
        random.shuffle(text_files)
        
        # Split into train and test
        split_index = int(len(text_files) * train_ratio)
        train_files = text_files[:split_index]
        test_files = text_files[split_index:]
        
        # Create train and test folders
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        
        # Function to create dataset.jsonl and copy files for a given set of files
        def process_files(files, output_folder):
            dataset = []
            for file in files:
                base_name = file.replace('_text.txt', '')
                
                # Read text
                with open(os.path.join(source_path, f"{base_name}_text.txt"), 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                # Read triples
                triples = []
                triples_file = os.path.join(source_path, f"{base_name}{triples_postfix}")
                try:
                    with open(triples_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            triple = json.loads(line.strip())
                            triples.append(triple)
                except FileNotFoundError:
                    print(f"Warning: Triples file not found: {triples_file}")
                    continue
                
                # Add to dataset
                dataset.append({
                    "text": text,
                    "triples": triples
                })
                
                # Copy _text.txt and triples files
                shutil.copy2(
                    os.path.join(source_path, f"{base_name}_text.txt"),
                    os.path.join(output_folder, f"{base_name}_text.txt")
                )
                shutil.copy2(
                    triples_file,
                    os.path.join(output_folder, f"{base_name}{triples_postfix}")
                )
            
            # Save dataset.jsonl
            with open(os.path.join(output_folder, 'dataset.jsonl'), 'w', encoding='utf-8') as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Process train and test files
        process_files(train_files, train_folder)
        process_files(test_files, test_folder)
        
        print(f"Data split into train ({len(train_files)} samples) and test ({len(test_files)} samples).")
    
    elif source in ['kelm_sub', 'webnlg20']:
        # Path to the source data
        source_path = os.path.join('./baselines/PiVe/datasets', source)
        
        # Delete existing train/test folders if they exist
        train_folder = os.path.join('data', 'train')
        test_folder = os.path.join('data', 'test')
        
        if os.path.exists(train_folder):
            print(f"Deleting existing folder: {train_folder}")
            shutil.rmtree(train_folder)
        
        if os.path.exists(test_folder):
            print(f"Deleting existing folder: {test_folder}")
            shutil.rmtree(test_folder)
        
        # Create train and test folders
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        
        # Process train and test files
        process_dataset_files(source_path, train_folder, 'train')
        process_dataset_files(source_path, test_folder, 'test')
        
        print(f"Data processed for {source} dataset.")
    
    elif source == 'genwiki_hiq':
        # Path to the source data (GenWiki-HIQ is in a different location)
        source_path = os.path.join('./baselines/PiVe/GenWiki-HIQ')
        
        # Delete existing train/test folders if they exist
        train_folder = os.path.join('data', 'train')
        test_folder = os.path.join('data', 'test')
        
        if os.path.exists(train_folder):
            print(f"Deleting existing folder: {train_folder}")
            shutil.rmtree(train_folder)
        
        if os.path.exists(test_folder):
            print(f"Deleting existing folder: {test_folder}")
            shutil.rmtree(test_folder)
        
        # Create train and test folders
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        
        # Process train and test files
        process_genwiki_dataset_files(source_path, train_folder, 'train')
        process_genwiki_dataset_files(source_path, test_folder, 'test')
        
        print(f"Data processed for {source} dataset.")

def process_dataset_files(source_path, output_folder, split):
    """Process source and target files for a given split (train, val, test)"""
    source_file = os.path.join(source_path, f"{split}.source")
    target_file = os.path.join(source_path, f"{split}.target")
    
    if not os.path.exists(source_file) or not os.path.exists(target_file):
        print(f"Warning: {split} files not found in {source_path}")
        return
    
    dataset = []
    
    # Read source and target files line by line
    with open(source_file, 'r', encoding='utf-8') as f_source, \
         open(target_file, 'r', encoding='utf-8') as f_target:
        
        source_lines = f_source.readlines()
        target_lines = f_target.readlines()
        
        if len(source_lines) != len(target_lines):
            print(f"Warning: Number of lines in {split}.source ({len(source_lines)}) and {split}.target ({len(target_lines)}) do not match.")
            return
        
        for source_line, target_line in zip(source_lines, target_lines):
            source_line = source_line.strip()
            target_line = target_line.strip()
            
            if not source_line or not target_line:
                continue
            
            # Parse the source line (KG triples)
            try:
                triples = json.loads(source_line)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON in {split}.source: {source_line}")
                continue
            
            # Add to dataset
            dataset.append({
                "text": target_line,
                "triples": triples
            })
    
    # Save dataset.jsonl
    with open(os.path.join(output_folder, 'dataset.jsonl'), 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Processed {len(dataset)} samples for {split} split.")

def process_genwiki_dataset_files(source_path, output_folder, split):
    """Process source and target files for GenWiki-HIQ dataset with triple validation"""
    source_file = os.path.join(source_path, f"{split}.source")
    target_file = os.path.join(source_path, f"{split}.target")
    
    if not os.path.exists(source_file) or not os.path.exists(target_file):
        print(f"Warning: {split} files not found in {source_path}")
        return
    
    dataset = []
    skipped_count = 0
    
    # Read source and target files line by line
    with open(source_file, 'r', encoding='utf-8') as f_source, \
         open(target_file, 'r', encoding='utf-8') as f_target:
        
        source_lines = f_source.readlines()
        target_lines = f_target.readlines()
        
        if len(source_lines) != len(target_lines):
            print(f"Warning: Number of lines in {split}.source ({len(source_lines)}) and {split}.target ({len(target_lines)}) do not match.")
            return
        
        for i, (source_line, target_line) in enumerate(zip(source_lines, target_lines)):
            source_line = source_line.strip()
            target_line = target_line.strip()
            
            if not source_line or not target_line:
                skipped_count += 1
                continue
            
            # Parse the source line (KG triples)
            try:
                triples = json.loads(source_line)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON in {split}.source: {source_line}")
                skipped_count += 1
                continue
            
            # Validate and filter triples
            valid_triples = []
            sample_valid = True
            
            for j, triple in enumerate(triples):
                # Check if triple is a list and has exactly 3 elements
                if not isinstance(triple, list) or len(triple) != 3:
                    sample_valid = False
                    break
                
                # Check if all elements can be converted to strings and are not empty
                try:
                    subject = str(triple[0]).strip()
                    predicate = str(triple[1]).strip()
                    obj = str(triple[2]).strip()
                    
                    # Skip if any element is empty
                    if not subject or not predicate or not obj:
                        sample_valid = False
                        break
                    
                    # Add valid triple
                    valid_triples.append([subject, predicate, obj])
                    
                except Exception:
                    sample_valid = False
                    break
            
            # Only add sample if all triples are valid and we have at least one triple
            if sample_valid and valid_triples:
                dataset.append({
                    "text": target_line,
                    "triples": valid_triples
                })
            else:
                skipped_count += 1
    
    # Save dataset.jsonl
    with open(os.path.join(output_folder, 'dataset.jsonl'), 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Processed {len(dataset)} samples for {split} split.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} samples with problematic triples.")

# Step 2: Run the data preparation
prepare_datasets(args.data_folder, args.source, args.train_ratio, args.triples_postfix, args.num_samples)