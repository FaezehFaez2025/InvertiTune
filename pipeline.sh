#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Default dataset source
DATASET_SOURCE="wikidata"
# Default GPU
GPU_ID="0"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET_SOURCE="$2"
      shift 2
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Validate dataset source
if [[ "$DATASET_SOURCE" != "wikidata" && "$DATASET_SOURCE" != "kelm_sub" && "$DATASET_SOURCE" != "webnlg20" && "$DATASET_SOURCE" != "genwiki_hiq" ]]; then
  echo "Error: Dataset source must be either 'wikidata', 'kelm_sub', 'webnlg20', or 'genwiki_hiq'"
  exit 1
fi

# Step 1: Delete the data folder
echo "Step 1: Deleting data folder..."
rm -rf data

# Step 2: Prepare data
echo "Step 2: Preparing data..."
if [[ "$DATASET_SOURCE" == "wikidata" ]]; then
    python prepare_data.py --data_folder ./data_generation/data --source wikidata --train_ratio 0.8 --triples_postfix "_triples.txt" --num_samples 1000
else
    python prepare_data.py --source "$DATASET_SOURCE"
fi

# Step 3: Build training dataset
echo "Step 3: Building training dataset..."
python build_llama_factory_dataset.py --partition train

# Step 4: Build test dataset
echo "Step 4: Building test dataset..."
python build_llama_factory_dataset.py --partition test

# Step 5: Prepare data for baselines
echo "Step 5: Preparing data for baselines..."
python prepare_baseline_data.py

# Step 6: Remove saves folder in LLaMA-Factory
echo "Step 6: Removing saves folder in LLaMA-Factory..."
cd "$SCRIPT_DIR/LLaMA-Factory"
rm -rf saves

# Step 7: Run training
echo "Step 7: Running training..."
CUDA_VISIBLE_DEVICES="$GPU_ID" FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen2.5_full_sft.yaml

# Step 8: Test finetuned model
echo "Step 8: Testing finetuned model..."
cd "$SCRIPT_DIR/LLaMA-Factory"
CUDA_VISIBLE_DEVICES="$GPU_ID" python infer_knowledge_graph.py --num_params 1.5B --use_finetuned

# Step 9: Test original model
echo "Step 9: Testing original model..."
CUDA_VISIBLE_DEVICES="$GPU_ID" python infer_knowledge_graph.py --num_params 1.5B

# Step 10: Compute BERT score for finetuned model
echo "Step 10: Computing BERT score for finetuned model..."
python compute_bert_score.py --json_file results/test_predictions_finetuned_1.5B.json

# Step 11: Compute BERT score for original model
echo "Step 11: Computing BERT score for original model..."
python compute_bert_score.py --json_file results/test_predictions_original_1.5B.json

echo "Pipeline completed!" 