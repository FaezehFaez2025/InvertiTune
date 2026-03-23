#!/usr/bin/env python3
"""
Statistical comparison of two models using Wilcoxon signed-rank test.

Usage:
    python statistical_test.py --model1 model1_per_sample_scores.json --model2 model2_per_sample_scores.json
"""

import json
import argparse
import numpy as np
from scipy.stats import wilcoxon


def compare_models(scores_file1, scores_file2):
    """Compare two models using Wilcoxon signed-rank test."""
    
    # Load scores
    with open(scores_file1, 'r') as f:
        scores1 = json.load(f)
    
    with open(scores_file2, 'r') as f:
        scores2 = json.load(f)
    
    # Verify same number of samples
    if scores1['n_samples'] != scores2['n_samples']:
        print(f"Error: Different number of samples ({scores1['n_samples']} vs {scores2['n_samples']})")
        return
    
    # Extract model names from pred_file paths
    model1_name = scores1['pred_file'].split('/')[-1].replace('.txt', '')
    model2_name = scores2['pred_file'].split('/')[-1].replace('.txt', '')
    
    print("=" * 80)
    print("STATISTICAL COMPARISON (Wilcoxon Signed-Rank Test)")
    print("=" * 80)
    print(f"Model 1: {model1_name}")
    print(f"Model 2: {model2_name}")
    print(f"Samples: {scores1['n_samples']}")
    print("=" * 80)
    
    # Metrics to compare
    metrics_to_compare = [
        ('bleu_precision', 'G-BLEU Precision'),
        ('bleu_recall', 'G-BLEU Recall'),
        ('bleu_f1', 'G-BLEU F1'),
        ('rouge_precision', 'G-ROUGE Precision'),
        ('rouge_recall', 'G-ROUGE Recall'),
        ('rouge_f1', 'G-ROUGE F1'),
        ('bertscore_precision', 'G-BERTScore Precision'),
        ('bertscore_recall', 'G-BERTScore Recall'),
        ('bertscore_f1', 'G-BERTScore F1'),
        ('triple_match_f1', 'Triple Match F1'),
        ('graph_match_accuracy', 'Graph Match Accuracy')
    ]
    
    for metric_key, metric_name in metrics_to_compare:
        model1_scores = np.array(scores1[metric_key])
        model2_scores = np.array(scores2[metric_key])
        
        mean1 = np.mean(model1_scores)
        mean2 = np.mean(model2_scores)
        diff = mean1 - mean2
        
        # Wilcoxon signed-rank test
        stat, p_value = wilcoxon(model1_scores, model2_scores)
        
        # Determine significance
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = ""
        
        # Determine better model
        if p_value < 0.05:
            better = model1_name if diff > 0 else model2_name
        else:
            better = "No difference"
        
        print(f"\n{metric_name:25} | M1: {mean1:.4f}  M2: {mean2:.4f}  Diff: {diff:+.4f}  p: {p_value:.4e} {sig:3} → {better}")
    
    print("\n" + "=" * 80)
    print("Legend: *** p<0.001  ** p<0.01  * p<0.05")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Statistical comparison of two models using Wilcoxon signed-rank test.'
    )
    parser.add_argument('--model1', type=str, required=True,
                        help='Path to first model per-sample scores JSON file')
    parser.add_argument('--model2', type=str, required=True,
                        help='Path to second model per-sample scores JSON file')
    
    args = parser.parse_args()
    
    compare_models(args.model1, args.model2)
