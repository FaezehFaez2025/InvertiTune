import ast
import os
import sys

def load_triples(file_path):
    """Load triples from a file and return them as a set of tuples."""
    with open(file_path, "r", encoding="utf-8") as f:
        return set(tuple(ast.literal_eval(line.strip())) for line in f)

def compute_metrics(gt_triples, method_triples, method_name):
    """Compute precision, recall, and F1 score for a given method."""
    true_positives = sum(1 for triple in method_triples if triple in gt_triples)
    false_positives = [triple for triple in method_triples if triple not in gt_triples]

    precision = 100 * true_positives / len(method_triples) if method_triples else 0
    recall = 100 * true_positives / len(gt_triples) if gt_triples else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nMethod: {method_name}")
    print(f"Precision: {round(precision, 2)}%")
    print(f"Recall: {round(recall, 2)}%")
    print(f"F1 Score: {round(f1_score, 2)}%")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {len(false_positives)}")

    if false_positives:
        print("❌ False Positives ❌")
        for item in false_positives:
            print("   ", list(item))

def main():
    """Main function to execute the evaluation."""
    if len(sys.argv) != 2:
        print("Usage: python triple_filtering_performance_evaluation.py <ground_truth_file>")
        sys.exit(1)

    gt_file_name = sys.argv[1]
    data_dir = "data"  # Directory containing the data files

    # Construct paths for the ground truth and method files
    gt_file_path = os.path.join(data_dir, gt_file_name)
    base_name = gt_file_name.replace("_gt.txt", "")

    method_files = {
        "gpt-3.5-turbo": os.path.join(data_dir, f"{base_name}_pruned_gpt-3.5-turbo.txt"),
        "gpt-4o": os.path.join(data_dir, f"{base_name}_pruned_gpt-4o.txt"),
        "V3": os.path.join(data_dir, f"{base_name}_pruned_V3.txt"),
    }

    # Check if the ground truth file exists
    if not os.path.exists(gt_file_path):
        print(f"Ground truth file '{gt_file_path}' not found.")
        sys.exit(1)

    # Load ground truth triples
    gt_triples = load_triples(gt_file_path)

    # Evaluate each method
    for method_name, method_file_path in method_files.items():
        if os.path.exists(method_file_path):
            method_triples = load_triples(method_file_path)
            compute_metrics(gt_triples, method_triples, method_name)
        else:
            print(f"File for method '{method_name}' not found: {method_file_path}")

if __name__ == "__main__":
    main()