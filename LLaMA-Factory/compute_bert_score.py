import json
import argparse
from bert_score import score
import numpy as np

parser = argparse.ArgumentParser(description='Compute BERTScore between prediction and ground_truth in a JSON file (treating both as plain sentences)')
parser.add_argument('--json_file', type=str, required=True, help='Path to the JSON file containing prediction and ground_truth fields')
args = parser.parse_args()

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    data = load_json(args.json_file)
    preds = []
    refs = []
    for item in data:
        pred_text = item['prediction']
        gt_text = item['ground_truth']
        preds.append(pred_text)
        refs.append(gt_text)
    # Compute BERTScore at the token level for each sample
    P, R, F1 = score(preds, refs, lang='en', verbose=True)
    print(f'Average BERTScore Precision: {P.mean().item():.4f}')
    print(f'Average BERTScore Recall: {R.mean().item():.4f}')
    print(f'Average BERTScore F1: {F1.mean().item():.4f}')

if __name__ == '__main__':
    main() 