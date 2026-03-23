# InvertiTune

This is the official repository for the [InvertiTune](https://arxiv.org/pdf/2512.03197) paper.

## Environment Setup & Installation

```bash
conda create --name InvertiTun_env python=3.10
```

```bash
conda activate InvertiTun_env
```

```bash
pip install SPARQLWrapper tqdm
```

```bash
# Install PyTorch with CUDA support
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

```bash
# Install remaining packages from PyPI
pip install deepspeed==0.15.0 accelerate==0.34.0 transformers==4.49.0 tokenizers==0.21.0 llamafactory==0.9.3 bert-score==0.3.13 peft==0.15.0 trl==0.9.6 datasets==3.5.0 huggingface-hub==0.33.2 tensorboard
```

---

# pipeline.sh

## Description
This script automates the entire pipeline for fine-tuning and evaluating the model. It handles data preparation, model training, and evaluation in a single run.

## Commands to Run

### Default Dataset (Wikidata) with Default GPU (GPU 0)
```bash
./pipeline.sh
```

## Pipeline Steps
The script will automatically:
1. Prepare the data for the selected dataset
2. Build training and test datasets
3. Prepare data for baselines
4. Run the training process
5. Test both the fine-tuned and original models
6. Compute BERT scores for comparison

# prepare_data.py

## Description
This script prepares training and testing datasets for fine-tuning a model. It:
1. Splits the data into `train` and `test` sets based on a specified ratio.
2. Copies the `_text.txt` and `_triples_pruned.txt` files from the path specified by `--data_folder` into the local `data/train` and `data/test` folders, organizing them into train and test partitions.
3. Creates `dataset.jsonl` files in the `data/train` and `data/test` folders, combining the copied `_text.txt` and `_triples_pruned.txt` files into the format required for fine-tuning:  

   ```json
   {"text": "Example text.", "triples": [["ENTITY", "RELATION", "ENTITY"]]}
   ```
4. Ensures non-ASCII characters (e.g., `ß`) are preserved in `dataset.jsonl`.

## Command to Run
```bash
python prepare_data.py --data_folder ./data_generation/data --source wikidata --train_ratio 0.8
```
- `--data_folder`: Path to the data folder (default: `./data_generation/data`).
- `--source`: Data source (`wikidata`, `yago`, `kelm_sub`, or `webnlg20`, default: `wikidata`).
- `--train_ratio`: Ratio of data to use for training (default: `0.8`).
### with triples_postfix indicated (TBC)
```bash
python prepare_data.py --data_folder ./data_generation/data --source wikidata --train_ratio 0.8 --triples_postfix "_triples.txt"
```

### with specific number of samples
```bash
python prepare_data.py --data_folder ./data_generation/data --source wikidata --train_ratio 0.8 --triples_postfix "_triples.txt" --num_samples 10
```
- `--num_samples`: Number of samples to randomly select from the dataset. If not specified, processes all available samples. Useful for testing or development with smaller datasets.

### Processing KELM dataset
```bash
python prepare_data.py --source kelm_sub
```
- For KELM dataset, the script will look for data in `./baselines/PiVe/datasets/kelm_sub` directory.

### Processing WebNLG20 dataset
```bash
python prepare_data.py --source webnlg20
```
- For WebNLG20 dataset, the script will look for data in `./baselines/PiVe/datasets/webnlg20` directory.

# build_llama_factory_dataset.py

## Description
This script converts the dataset.jsonl files into the format required by LLaMA-Factory for fine-tuning. It:
1. Reads data from the `data/train/dataset.jsonl` or `data/test/dataset.jsonl` file based on the specified partition.
2. Converts each entry into the LLaMA-Factory format with instruction, input, and output fields.
3. Saves the processed data as `T2G_train.json` or `T2G_test.json` in the `LLaMA-Factory/data` directory.

## Command to Run
```bash
# For training data:
python build_llama_factory_dataset.py --partition train
```

```bash
# For test data:
python build_llama_factory_dataset.py --partition test
```
- `--partition`: Specifies which data partition to process (`train` or `test`).

# finetune_qwen2.5.py
```bash
python finetune_qwen2.5.py --num_params 7B --num_epochs 10
```

# test_original_qwen25.py
```bash
python test_original_qwen25.py --num_params 1.5B --dataset_path data/test/dataset.jsonl
```

# prepare_baseline_data.py

## Description
This script prepares data for baseline models by converting the LLaMA-Factory format (T2G_test.json) into a JSONL format that is compatible with the baseline models. It:
1. Reads the T2G_test.json file from the LLaMA-Factory/data directory.
2. Converts each entry into a structured format with docid, text, and triples fields.
3. Saves the processed data as en_test.jsonl in the datasets/controlled_extraction_dataset directory (two levels up from the current path).
4. This converted data is then suitable for use with the process_dataset.py script located two directories up from this path.

## Command to Run
```bash
python prepare_baseline_data.py
```

# prune_prediction_triples.py

## Description
This script processes and cleans the predictions generated by our fine-tuned model (located in the `LLaMA-Factory/results/` directory). The model's predictions may sometimes be incomplete or contain triples that don't exactly follow the expected pattern `["SUBJECT", "PREDICATE", "OBJECT"]`. This script saves the improved predictions to a new file ending with `_improved.json`.

## Command to Run
```bash
python prune_prediction_triples.py
```
- `--input_filename`: Name of the input file in the `LLaMA-Factory/results/` directory (default: `test_predictions_finetuned_1.5B.json`).

# convert_data_to_comparable_format.py

## Description
This script converts the results of the fine-tuned model to an aggregated format that can be easily compared with baseline results. It reads a JSON file containing model predictions from the `LLaMA-Factory/results/` directory, extracts triplets from either the "ground_truth" or "prediction" key (specified by the `--data_type` parameter), formats them with standardized formatting (uppercase subjects and objects, proper quoting), and saves the processed triplets to an aggregated file in the specified output directory for direct comparison with baseline results.

## Command to Run
```bash
python convert_data_to_comparable_format.py
```
- `--data_type` or `-d`: Type of data to convert (`ground_truth` or `prediction`, default: `prediction`).
- `--output_dir` or `-o`: Directory to save the output file (default: `../../result/controlled_extraction/test`).
- `filename`: Name of the JSON file in the `LLaMA-Factory/results/` directory (default: `test_predictions_finetuned_1.5B_improved.json`).

# detect_and_remove_empty_lines.py

## Description
This script handles cases where some baselines (like GraphRAG) fail to generate answers for certain data samples, particularly in datasets like KELM where input texts might be too short. It:
1. Creates a backup of all aggregated triplet files in an `original_prediction` directory (only if it doesn't already exist)
2. Identifies line numbers where any baseline has failed to generate an answer (indicated by empty `[]` entries)
3. Removes those lines from all aggregated files to ensure fair comparison across all baselines
4. This ensures that the evaluation is performed only on samples where all baselines successfully generated answers

## Command to Run
```bash
python detect_and_remove_empty_lines.py
```

# analyze_dataset_stats.py

## Description
This script analyzes the T2G dataset files and reports comprehensive statistics including sample counts, triple counts per knowledge graph, token counts per text, and dataset split percentages.

## Command to Run
```bash
python analyze_dataset_stats.py train
```

```bash
python analyze_dataset_stats.py test
```

# Statistical Comparison (Wilcoxon Signed-Rank Test)

After evaluating models with `--per_sample` flag, you can statistically compare two models using the Wilcoxon signed-rank test. This test determines whether the performance difference between two models is statistically significant across all evaluation metrics (G-BLEU, G-ROUGE, G-BERTScore, etc.).

## Command to Run
```bash
python baselines/PiVe/graph_evaluation/metrics/statistical_test.py --model1 ../../result/controlled_extraction/test/InvertiTune_per_sample_scores.json --model2 ../../result/controlled_extraction/test/Pive_per_sample_scores.json
```
