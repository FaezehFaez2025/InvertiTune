# Triple Filtering Performance Evaluation

## Description
This script evaluates the performance of different methods (e.g., GPT-3.5, GPT-4, V3) in filtering triples by comparing their outputs against the ground truth. It calculates precision, recall, and 
F1 score for each method and lists any false positives.

## How to Run
1. Ensure the required files are present in the `data` folder (see below).
2. Run the script with the ground truth file as an argument:
   ```bash
   python triple_filtering_performance_evaluation.py <ground_truth_file>
   ```
	Example:
   	```bash
   	python triple_filtering_performance_evaluation.py Q118696959_China_2_hop_0.2_ratio_triples_gt.txt
   	```
## Required Files in `data` Folder
- `<ground_truth_file>`: The ground truth triples file (e.g., `example_gt.txt`).
- `<base_name>_pruned_gpt-3.5-turbo.txt`: Filtered triples from GPT-3.5.
- `<base_name>_pruned_gpt-4o.txt`: Filtered triples from GPT-4.
- `<base_name>_pruned_V3.txt`: Filtered triples from method V3.

Replace `<base_name>` with the prefix of your ground truth file (e.g., `example` for `example_gt.txt`).
