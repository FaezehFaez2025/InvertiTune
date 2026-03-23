import json
import os
from pathlib import Path
from collections import Counter

def count_triples(triples_str):
    # Parse the string representation of triples
    # Each triple is in the format ["subject", "relation", "object"]
    try:
        triples = json.loads(triples_str)
        return len(triples)
    except:
        return 0

def count_tokens_in_triples(triples_str):
    # Count total tokens across all triples
    try:
        triples = json.loads(triples_str)
        total_tokens = 0
        for triple in triples:
            # Count tokens in subject, relation, and object
            total_tokens += len(triple[0].split())  # subject
            total_tokens += len(triple[1].split())  # relation
            total_tokens += len(triple[2].split())  # object
        return total_tokens
    except:
        return 0

def generate_distribution_files(num_samples):
    # Get the script's directory
    script_dir = Path(__file__).parent
    
    # Paths relative to script directory
    predictions_path = script_dir.parent / "LLaMA-Factory/results/test_predictions_finetuned_1.5B_improved.json"
    distribution_dir = script_dir / "distribution"
    
    # Read the predictions file
    with open(predictions_path, 'r') as f:
        data = json.load(f)
    
    # Initialize counters for both triples and tokens
    gt_triple_counts = []
    pr_triple_counts = []
    gt_token_counts = []
    pr_token_counts = []
    
    for item in data:
        # Count triples
        gt_triple_count = count_triples(item.get("ground_truth", "[]"))
        pr_triple_count = count_triples(item.get("prediction", "[]"))
        gt_triple_counts.append(gt_triple_count)
        pr_triple_counts.append(pr_triple_count)
        
        # Count tokens in triples
        gt_token_count = count_tokens_in_triples(item.get("ground_truth", "[]"))
        pr_token_count = count_tokens_in_triples(item.get("prediction", "[]"))
        gt_token_counts.append(gt_token_count)
        pr_token_counts.append(pr_token_count)
    
    # Create frequency distributions
    gt_triple_dist = Counter(gt_triple_counts)
    pr_triple_dist = Counter(pr_triple_counts)
    gt_token_dist = Counter(gt_token_counts)
    pr_token_dist = Counter(pr_token_counts)
    
    # Create the distribution files
    gt_triple_file = distribution_dir / f"{num_samples}_triples_gt.txt"
    pr_triple_file = distribution_dir / f"{num_samples}_triples_pr.txt"
    gt_token_file = distribution_dir / f"{num_samples}_tokens_gt.txt"
    pr_token_file = distribution_dir / f"{num_samples}_tokens_pr.txt"
    
    # Write the distribution files
    with open(gt_triple_file, 'w') as f:
        for x, y in sorted(gt_triple_dist.items()):
            f.write(f"{x}: {y}\n")
    
    with open(pr_triple_file, 'w') as f:
        for x, y in sorted(pr_triple_dist.items()):
            f.write(f"{x}: {y}\n")
            
    with open(gt_token_file, 'w') as f:
        for x, y in sorted(gt_token_dist.items()):
            f.write(f"{x}: {y}\n")
    
    with open(pr_token_file, 'w') as f:
        for x, y in sorted(pr_token_dist.items()):
            f.write(f"{x}: {y}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate distribution files for visualization')
    parser.add_argument('num_samples', type=int, 
                       help='Number of samples in the dataset. This will be used as a prefix for the output files (e.g., 2000_triples_gt.txt and 2000_tokens_gt.txt)')
    
    args = parser.parse_args()
    generate_distribution_files(args.num_samples) 