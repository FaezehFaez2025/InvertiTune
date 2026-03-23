#!/usr/bin/env python3
"""
Dataset Statistics Analyzer.

This script analyzes the T2G_train.json and T2G_test.json files and reports various statistics
including sample counts, triple counts, token counts, and dataset split percentages.
"""

import json
import argparse
import ast
import statistics
from pathlib import Path


def count_tokens(text):
    """Simple token counting by splitting on whitespace."""
    return len(text.split())


def clean_malformed_quotes(output_str):
    """Clean malformed quotes in the output string.
    
    Keep quotes only when:
    1) [ appears right before "
    2) , appears right after "
    3) , (space+,) appears right before "
    4) ] appears right after "
    
    Remove all other quotes.
    """
    result = []
    i = 0
    while i < len(output_str):
        char = output_str[i]
        
        if char == '"':
            # Check if this quote should be kept
            keep_quote = False
            
            # Check condition 1: [ appears right before "
            if i > 0 and output_str[i-1] == '[':
                keep_quote = True
            
            # Check condition 2: , appears right after "
            elif i < len(output_str) - 1 and output_str[i+1] == ',':
                keep_quote = True
            
            # Check condition 3: , (space+,) appears right before "
            elif i > 1 and output_str[i-2:i] == ', ':
                keep_quote = True
            
            # Check condition 4: ] appears right after "
            elif i < len(output_str) - 1 and output_str[i+1] == ']':
                keep_quote = True
            
            if keep_quote:
                result.append(char)
            # If not keeping the quote, skip it (don't append)
        else:
            result.append(char)
        
        i += 1
    
    return ''.join(result)


def count_triples(output_str):
    """Count the number of triples in the output field."""
    try:
        # Parse the string representation of the list of triples
        triples = ast.literal_eval(output_str)
        return len(triples)
    except (ValueError, SyntaxError):
        # If parsing fails, try cleaning malformed quotes first
        try:
            cleaned_str = clean_malformed_quotes(output_str)
            triples = ast.literal_eval(cleaned_str)
            return len(triples)
        except (ValueError, SyntaxError) as e:
            # If parsing still fails after cleaning, return 0
            print(f"Warning: Could not parse output even after cleaning: {output_str[:100]}...")
            return 0


def analyze_dataset(file_path):
    """Analyze a single dataset file and return statistics."""
    print(f"\nAnalyzing: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Basic statistics
    num_samples = len(data)
    
    # Extract triple counts and token counts
    triple_counts = []
    token_counts = []
    
    for item in data:
        # Count triples from output field
        if 'output' in item:
            triple_count = count_triples(item['output'])
            triple_counts.append(triple_count)
        
        # Count tokens from input field
        if 'input' in item:
            token_count = count_tokens(item['input'])
            token_counts.append(token_count)
    
    # Calculate statistics
    stats = {
        'num_samples': num_samples,
        'triple_stats': {
            'min': min(triple_counts) if triple_counts else 0,
            'max': max(triple_counts) if triple_counts else 0,
            'avg': statistics.mean(triple_counts) if triple_counts else 0,
            'median': statistics.median(triple_counts) if triple_counts else 0,
        },
        'token_stats': {
            'min': min(token_counts) if token_counts else 0,
            'max': max(token_counts) if token_counts else 0,
            'avg': statistics.mean(token_counts) if token_counts else 0,
            'median': statistics.median(token_counts) if token_counts else 0,
        }
    }
    
    return stats


def get_total_samples(data_dir):
    """Get total number of samples across both train and test files."""
    train_path = data_dir / "T2G_train.json"
    test_path = data_dir / "T2G_test.json"
    
    total_samples = 0
    
    if train_path.exists():
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            total_samples += len(train_data)
    
    if test_path.exists():
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            total_samples += len(test_data)
    
    return total_samples


def print_stats(stats, dataset_name, total_samples=None):
    """Print formatted statistics."""
    print(f"\n{'='*50}")
    print(f"STATISTICS FOR {dataset_name.upper()} DATASET")
    print(f"{'='*50}")
    
    print(f"\n📊 SAMPLE COUNT:")
    print(f"   Number of samples: {stats['num_samples']:,}")
    
    if total_samples:
        percentage = (stats['num_samples'] / total_samples) * 100
        print(f"   Percentage of total: {percentage:.2f}%")
    
    print(f"\n🔗 TRIPLE STATISTICS:")
    print(f"   Min triples per KG: {stats['triple_stats']['min']}")
    print(f"   Max triples per KG: {stats['triple_stats']['max']}")
    print(f"   Avg triples per KG: {stats['triple_stats']['avg']:.2f}")
    print(f"   Median triples per KG: {stats['triple_stats']['median']:.2f}")
    
    print(f"\n📝 TOKEN STATISTICS:")
    print(f"   Min tokens per text: {stats['token_stats']['min']}")
    print(f"   Max tokens per text: {stats['token_stats']['max']}")
    print(f"   Avg tokens per text: {stats['token_stats']['avg']:.2f}")
    print(f"   Median tokens per text: {stats['token_stats']['median']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze T2G dataset statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_dataset_stats.py train
  python analyze_dataset_stats.py test
        """
    )
    
    parser.add_argument(
        'dataset',
        choices=['train', 'test'],
        help='Which dataset to analyze: train or test'
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('LLaMA-Factory/data'),
        help='Directory containing the dataset files (default: LLaMA-Factory/data)'
    )
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not args.data_dir.exists():
        print(f"Error: Data directory '{args.data_dir}' does not exist!")
        return 1
    
    # Get total samples for percentage calculation
    total_samples = get_total_samples(args.data_dir)
    print(f"Total samples across all datasets: {total_samples:,}")
    
    if args.dataset == 'train':
        train_path = args.data_dir / "T2G_train.json"
        if train_path.exists():
            train_stats = analyze_dataset(train_path)
            print_stats(train_stats, 'train', total_samples)
        else:
            print(f"Error: {train_path} not found!")
            return 1
    
    elif args.dataset == 'test':
        test_path = args.data_dir / "T2G_test.json"
        if test_path.exists():
            test_stats = analyze_dataset(test_path)
            print_stats(test_stats, 'test', total_samples)
        else:
            print(f"Error: {test_path} not found!")
            return 1
    
    print(f"\n{'='*50}")
    print("Analysis complete!")
    print(f"{'='*50}")
    
    return 0


if __name__ == "__main__":
    exit(main())
