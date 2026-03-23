import os
import argparse
import numpy as np  # For standard deviation calculation

def calculate_statistics(directory, postfix):
    # Get all files ending with the specified postfix
    files = [f for f in os.listdir(directory) if f.endswith(postfix)]
    
    # Initialize statistics for triples
    total_files = len(files)
    total_triples = 0
    triples_per_file = []  # Store the number of triples in each file
    triples_distribution = {}  # Dictionary to store distribution of triples
    
    # Initialize statistics for tokens
    total_tokens = 0
    tokens_per_file = []  # Store the number of tokens in each file
    tokens_distribution = {}  # Dictionary to store distribution of tokens
    
    # Read each file and count triples and tokens
    for file in files:
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as f:
            triples = f.readlines()
            num_triples = len(triples)
            total_triples += num_triples
            triples_per_file.append(num_triples)
            
            # Update triples distribution dictionary
            if num_triples in triples_distribution:
                triples_distribution[num_triples] += 1
            else:
                triples_distribution[num_triples] = 1
            
            # Count tokens in the file
            num_tokens = 0
            for triple in triples:
                num_tokens += len(triple.strip().split())  # Split by whitespace and count tokens
            total_tokens += num_tokens
            tokens_per_file.append(num_tokens)
            
            # Update tokens distribution dictionary
            if num_tokens in tokens_distribution:
                tokens_distribution[num_tokens] += 1
            else:
                tokens_distribution[num_tokens] = 1
    
    # Calculate statistics for triples
    average_triples = total_triples / total_files if total_files > 0 else 0
    max_triples = max(triples_per_file) if triples_per_file else 0
    min_triples = min(triples_per_file) if triples_per_file else 0
    std_dev_triples = np.std(triples_per_file) if triples_per_file else 0
    
    # Calculate statistics for tokens
    average_tokens = total_tokens / total_files if total_files > 0 else 0
    max_tokens = max(tokens_per_file) if tokens_per_file else 0
    min_tokens = min(tokens_per_file) if tokens_per_file else 0
    std_dev_tokens = np.std(tokens_per_file) if tokens_per_file else 0
    
    # Print statistics for triples
    print("=== Triples Statistics ===")
    print(f"Number of files: {total_files}")
    print(f"Total number of triples: {total_triples}")
    print(f"Average number of triples per file: {average_triples:.2f}")
    print(f"Maximum number of triples in a file: {max_triples}")
    print(f"Minimum number of triples in a file: {min_triples}")
    print(f"Standard deviation of triples per file: {std_dev_triples:.2f}")
    
    # Save triples distribution to a file
    # Remove '.txt' from the postfix to avoid duplication in the filename
    postfix_clean = postfix.replace('.txt', '')
    triples_distribution_filename = f"triples_distribution{postfix_clean}.txt"
    with open(triples_distribution_filename, 'w') as f:
        for num_triples, count in sorted(triples_distribution.items()):
            f.write(f"{count} file(s) with {num_triples} triples\n")
    
    # Print statistics for tokens
    print("\n=== Tokens Statistics ===")
    print(f"Total number of tokens: {total_tokens}")
    print(f"Average number of tokens per file: {average_tokens:.2f}")
    print(f"Maximum number of tokens in a file: {max_tokens}")
    print(f"Minimum number of tokens in a file: {min_tokens}")
    print(f"Standard deviation of tokens per file: {std_dev_tokens:.2f}")
    
    # Save tokens distribution to a file
    tokens_distribution_filename = f"tokens_distribution{postfix_clean}.txt"
    with open(tokens_distribution_filename, 'w') as f:
        for num_tokens, count in sorted(tokens_distribution.items()):
            f.write(f"{count} file(s) with {num_tokens} tokens\n")

def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Calculate statistics for triples and tokens in files with a specified postfix.")
    parser.add_argument('--source', type=str, default='wikidata', choices=['wikidata', 'yago'],
                        help="Source of the data (wikidata or yago). Default is 'wikidata'.")
    parser.add_argument('--postfix', type=str, default='_pruned.txt',
                        help="Postfix of the files to be processed. Default is '_pruned.txt'.")
    
    # Parse arguments first to get the source
    args, remaining_args = parser.parse_known_args()
    
    # Now add the path argument with the default including the source
    parser.add_argument('--path', type=str, 
                        default=os.path.join(script_dir, "data", args.source),
                        help=f"Path to the directory containing the files to analyze.")
    
    # Parse the remaining arguments
    args = parser.parse_args()
    
    source_dir = args.path
    
    # Check if the directory exists
    if not os.path.exists(source_dir):
        print(f"Directory {source_dir} does not exist.")
        return
    
    # Calculate and print statistics
    calculate_statistics(source_dir, args.postfix)

if __name__ == "__main__":
    main()