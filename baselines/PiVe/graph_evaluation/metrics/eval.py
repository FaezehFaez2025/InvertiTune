import ast
import argparse
import numpy as np
import json
from scipy.stats import bootstrap
from graph_matching import split_to_edges, get_tokens, get_bleu_rouge, get_bert_score, get_ged, get_triple_match_f1, get_graph_match_accuracy
from tqdm import tqdm


def compute_bootstrap_ci(data, confidence_level=0.95, n_resamples=10000):
    """Compute bootstrap confidence interval for a given metric array."""
    # scipy.stats.bootstrap expects a function that takes the data and returns a statistic
    def statistic(data):
        return np.mean(data)
    
    # Reshape data for bootstrap (needs to be 1D array)
    data_1d = np.array(data).flatten()
    
    # Compute bootstrap CI
    result = bootstrap((data_1d,), statistic, confidence_level=confidence_level, 
                       n_resamples=n_resamples, method='percentile')
    return result.confidence_interval.low, result.confidence_interval.high


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", default=None, type=str, required=True)
    parser.add_argument("--gold_file", default=None, type=str, required=True)
    parser.add_argument("--compute_ged", action="store_true", help="Compute Graph Edit Distance (GED)")
    parser.add_argument("--per_sample", action="store_true", help="Compute metrics per sample (returns lists of scores, one per sample)")

    args = parser.parse_args()
    
    print("Loading gold graphs...")
    gold_graphs = []
    with open(args.gold_file, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            gold_graphs.append(ast.literal_eval(line.strip()))
			
    print("Loading prediction graphs...")
    pred_graphs = []
    with open(args.pred_file, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            #print(line)
            pred_graphs.append(ast.literal_eval(line.strip()))
			
    assert len(gold_graphs) == len(pred_graphs)
    print(f"Evaluating {len(gold_graphs)} samples...")
    
    if args.per_sample:
        # PER-SAMPLE MODE: Compute metrics for each sample individually
        print("Computing metrics per sample...")
        
        # Initialize lists to store per-sample metrics
        triple_match_f1_per_sample = []
        graph_match_accuracy_per_sample = []
        
        # These will be computed per sample in the loop
        precisions_rouge_per_sample = []
        recalls_rouge_per_sample = []
        f1s_rouge_per_sample = []
        precisions_bleu_per_sample = []
        recalls_bleu_per_sample = []
        f1s_bleu_per_sample = []
        precisions_BS_per_sample = []
        recalls_BS_per_sample = []
        f1s_BS_per_sample = []
        
        for i in tqdm(range(len(gold_graphs)), desc="Computing per-sample metrics"):
            # Get single sample as a list
            gold_single = [gold_graphs[i]]
            pred_single = [pred_graphs[i]]
            
            # Compute Triple Match F1 for this sample
            triple_f1 = get_triple_match_f1(gold_single, pred_single)
            triple_match_f1_per_sample.append(triple_f1)
            
            # Compute Graph Match Accuracy for this sample
            graph_acc = get_graph_match_accuracy(pred_single, gold_single)
            graph_match_accuracy_per_sample.append(graph_acc)
            
            # Prepare edges for BLEU/ROUGE/BERTScore
            gold_edges_single = split_to_edges(gold_single)
            pred_edges_single = split_to_edges(pred_single)
            gold_tokens_single, pred_tokens_single = get_tokens(gold_edges_single, pred_edges_single)
            
            # Compute BLEU and ROUGE for this sample
            prec_rouge, rec_rouge, f1_rouge, prec_bleu, rec_bleu, f1_bleu = get_bleu_rouge(
                gold_tokens_single, pred_tokens_single, gold_edges_single, pred_edges_single)
            
            precisions_rouge_per_sample.append(prec_rouge[0])
            recalls_rouge_per_sample.append(rec_rouge[0])
            f1s_rouge_per_sample.append(f1_rouge[0])
            precisions_bleu_per_sample.append(prec_bleu[0])
            recalls_bleu_per_sample.append(rec_bleu[0])
            f1s_bleu_per_sample.append(f1_bleu[0])
            
            # Compute BERTScore for this sample
            prec_BS, rec_BS, f1_BS = get_bert_score(gold_edges_single, pred_edges_single)
            precisions_BS_per_sample.append(prec_BS[0])
            recalls_BS_per_sample.append(rec_BS[0])
            f1s_BS_per_sample.append(f1_BS[0])
        
        # Convert to numpy arrays
        triple_match_f1_per_sample = np.array(triple_match_f1_per_sample)
        graph_match_accuracy_per_sample = np.array(graph_match_accuracy_per_sample)
        precisions_rouge_per_sample = np.array(precisions_rouge_per_sample)
        recalls_rouge_per_sample = np.array(recalls_rouge_per_sample)
        f1s_rouge_per_sample = np.array(f1s_rouge_per_sample)
        precisions_bleu_per_sample = np.array(precisions_bleu_per_sample)
        recalls_bleu_per_sample = np.array(recalls_bleu_per_sample)
        f1s_bleu_per_sample = np.array(f1s_bleu_per_sample)
        precisions_BS_per_sample = np.array(precisions_BS_per_sample)
        recalls_BS_per_sample = np.array(recalls_BS_per_sample)
        f1s_BS_per_sample = np.array(f1s_BS_per_sample)
        
        # Compute bootstrap confidence intervals
        print("Computing bootstrap confidence intervals (95% CI)...")
        triple_f1_ci = compute_bootstrap_ci(triple_match_f1_per_sample)
        graph_acc_ci = compute_bootstrap_ci(graph_match_accuracy_per_sample)
        bleu_prec_ci = compute_bootstrap_ci(precisions_bleu_per_sample)
        bleu_rec_ci = compute_bootstrap_ci(recalls_bleu_per_sample)
        bleu_f1_ci = compute_bootstrap_ci(f1s_bleu_per_sample)
        rouge_prec_ci = compute_bootstrap_ci(precisions_rouge_per_sample)
        rouge_rec_ci = compute_bootstrap_ci(recalls_rouge_per_sample)
        rouge_f1_ci = compute_bootstrap_ci(f1s_rouge_per_sample)
        bs_prec_ci = compute_bootstrap_ci(precisions_BS_per_sample)
        bs_rec_ci = compute_bootstrap_ci(recalls_BS_per_sample)
        bs_f1_ci = compute_bootstrap_ci(f1s_BS_per_sample)
        
        # Print aggregated results with confidence intervals
        print("\n--- Evaluation Results (Averaged from Per-Sample with 95% Bootstrap CI) ---")
        triple_f1_mean = np.mean(triple_match_f1_per_sample)
        print(f'Triple Match F1 Score: {triple_f1_mean:.4f} [{triple_f1_ci[0]:.4f}, {triple_f1_ci[1]:.4f}]\n')
        graph_acc_mean = np.mean(graph_match_accuracy_per_sample)
        print(f'Graph Match F1 Score: {graph_acc_mean:.4f} [{graph_acc_ci[0]:.4f}, {graph_acc_ci[1]:.4f}]\n')
        
        print(f'G-BLEU Precision: {np.mean(precisions_bleu_per_sample):.4f} [{bleu_prec_ci[0]:.4f}, {bleu_prec_ci[1]:.4f}]')
        print(f'G-BLEU Recall: {np.mean(recalls_bleu_per_sample):.4f} [{bleu_rec_ci[0]:.4f}, {bleu_rec_ci[1]:.4f}]')
        print(f'G-BLEU F1: {np.mean(f1s_bleu_per_sample):.4f} [{bleu_f1_ci[0]:.4f}, {bleu_f1_ci[1]:.4f}]\n')
        
        print(f'G-Rouge Precision: {np.mean(precisions_rouge_per_sample):.4f} [{rouge_prec_ci[0]:.4f}, {rouge_prec_ci[1]:.4f}]')
        print(f'G-Rouge Recall Score: {np.mean(recalls_rouge_per_sample):.4f} [{rouge_rec_ci[0]:.4f}, {rouge_rec_ci[1]:.4f}]')
        print(f'G-Rouge F1 Score: {np.mean(f1s_rouge_per_sample):.4f} [{rouge_f1_ci[0]:.4f}, {rouge_f1_ci[1]:.4f}]\n')
        
        print(f'G-BertScore Precision Score: {np.mean(precisions_BS_per_sample):.4f} [{bs_prec_ci[0]:.4f}, {bs_prec_ci[1]:.4f}]')
        print(f'G-BertScore Recall Score: {np.mean(recalls_BS_per_sample):.4f} [{bs_rec_ci[0]:.4f}, {bs_rec_ci[1]:.4f}]')
        print(f'G-BertScore F1 Score: {np.mean(f1s_BS_per_sample):.4f} [{bs_f1_ci[0]:.4f}, {bs_f1_ci[1]:.4f}]\n')
        
        print(f"\nPer-sample metrics computed. Total samples: {len(gold_graphs)}")
        
        # Save per-sample scores to JSON file
        scores_filename = args.pred_file.replace('.txt', '_per_sample_scores.json')
        scores_dict = {
            'pred_file': args.pred_file,
            'gold_file': args.gold_file,
            'n_samples': len(gold_graphs),
            'triple_match_f1': triple_match_f1_per_sample.tolist(),
            'graph_match_accuracy': graph_match_accuracy_per_sample.tolist(),
            'bleu_precision': precisions_bleu_per_sample.tolist(),
            'bleu_recall': recalls_bleu_per_sample.tolist(),
            'bleu_f1': f1s_bleu_per_sample.tolist(),
            'rouge_precision': precisions_rouge_per_sample.tolist(),
            'rouge_recall': recalls_rouge_per_sample.tolist(),
            'rouge_f1': f1s_rouge_per_sample.tolist(),
            'bertscore_precision': precisions_BS_per_sample.tolist(),
            'bertscore_recall': recalls_BS_per_sample.tolist(),
            'bertscore_f1': f1s_BS_per_sample.tolist(),
        }
        with open(scores_filename, 'w') as f:
            json.dump(scores_dict, f, indent=2)
        
    else:
        # ORIGINAL MODE: Compute metrics aggregated over all samples
        # Evaluate Triple Match F1 Score
        print("Computing Triple Match F1 Score...")
        triple_match_f1 = get_triple_match_f1(gold_graphs, pred_graphs)  

        # Evaluate Graph Match Accuracy
        print("Computing Graph Match Accuracy...")
        graph_match_accuracy = get_graph_match_accuracy(pred_graphs, gold_graphs)

        # Compute GED if requested
        if args.compute_ged:
            print("Computing Graph Edit Distance...")
            overall_ged = 0.
            for (gold, pred) in tqdm(zip(gold_graphs, pred_graphs), total=len(gold_graphs), desc="Computing GED"):
                ged = get_ged(gold, pred)
                overall_ged += ged

        # Evaluate for Graph Matching
        print("Preparing for BLEU, ROUGE, and BERTScore evaluation...")
        gold_edges = split_to_edges(gold_graphs)
        pred_edges = split_to_edges(pred_graphs)
        
        gold_tokens, pred_tokens = get_tokens(gold_edges, pred_edges)

        print("Computing BLEU and ROUGE scores...")
        precisions_rouge, recalls_rouge, f1s_rouge, precisions_bleu, recalls_bleu, f1s_bleu = get_bleu_rouge(
            gold_tokens, pred_tokens, gold_edges, pred_edges)

        print("Computing BERTScore...")
        precisions_BS, recalls_BS, f1s_BS = get_bert_score(gold_edges, pred_edges)

        print("\n--- Evaluation Results ---")
        print(f'Triple Match F1 Score: {triple_match_f1:.4f}\n')
        print(f'Graph Match F1 Score: {graph_match_accuracy:.4f}\n')
        
        print(f'G-BLEU Precision: {precisions_bleu.sum() / len(gold_graphs):.4f}')
        print(f'G-BLEU Recall: {recalls_bleu.sum() / len(gold_graphs):.4f}')
        print(f'G-BLEU F1: {f1s_bleu.sum() / len(gold_graphs):.4f}\n')

        print(f'G-Rouge Precision: {precisions_rouge.sum() / len(gold_graphs):.4f}')
        print(f'G-Rouge Recall Score: {recalls_rouge.sum() / len(gold_graphs):.4f}')
        print(f'G-Rouge F1 Score: {f1s_rouge.sum() / len(gold_graphs):.4f}\n')

        print(f'G-BertScore Precision Score: {precisions_BS.sum() / len(gold_graphs):.4f}')
        print(f'G-BertScore Recall Score: {recalls_BS.sum() / len(gold_graphs):.4f}')
        print(f'G-BertScore F1 Score: {f1s_BS.sum() / len(gold_graphs):.4f}\n')

        if args.compute_ged:
            print(f'Graph Edit Distance (GED): {overall_ged / len(gold_graphs):.4f}\n')
