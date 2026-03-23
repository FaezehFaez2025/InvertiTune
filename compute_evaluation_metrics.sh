#!/bin/bash

# Set the base directory to the script's location
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="../../result/controlled_extraction/test"

# Create results directory if it doesn't exist
mkdir -p "$RESULT_DIR"

echo "Evaluating Qwen 1.5B Finetuned..."
python baselines/PiVe/graph_evaluation/metrics/eval.py \
    --pred_file "$RESULT_DIR/aggregated_finetuned_1.5B_improved_prediction_triplets.txt" \
    --gold_file "$RESULT_DIR/aggregated_ground_truth_triplets.txt"
    
echo "Evaluating GraphRAG..."
python baselines/PiVe/graph_evaluation/metrics/eval.py \
    --pred_file "$RESULT_DIR/aggregated_graphrag_triplets.txt" \
    --gold_file "$RESULT_DIR/aggregated_ground_truth_triplets.txt"

echo "Evaluating LightRAG..."
python baselines/PiVe/graph_evaluation/metrics/eval.py \
    --pred_file "$RESULT_DIR/aggregated_lightrag_triplets.txt" \
    --gold_file "$RESULT_DIR/aggregated_ground_truth_triplets.txt"

echo "Evaluating ChatGPT..."
python baselines/PiVe/graph_evaluation/metrics/eval.py \
    --pred_file "$RESULT_DIR/aggregated_chatgpt_triplets.txt" \
    --gold_file "$RESULT_DIR/aggregated_ground_truth_triplets.txt"

echo "Evaluating PiVe..."
python baselines/PiVe/graph_evaluation/metrics/eval.py \
    --pred_file "$RESULT_DIR/aggregated_pive_triplets_post_processed.txt" \
    --gold_file "$RESULT_DIR/aggregated_ground_truth_triplets.txt"

echo "Evaluating OpenIE6..."
python baselines/PiVe/graph_evaluation/metrics/eval.py \
    --pred_file "$RESULT_DIR/aggregated_openie6_triplets.txt" \
    --gold_file "$RESULT_DIR/aggregated_ground_truth_triplets.txt"

echo "Evaluating DeepEx..."
python baselines/PiVe/graph_evaluation/metrics/eval.py \
    --pred_file "$RESULT_DIR/aggregated_deepex_triplets.txt" \
    --gold_file "$RESULT_DIR/aggregated_ground_truth_triplets.txt"

echo "Evaluation complete for all methods!" 