#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$SCRIPT_DIR"

echo "Step 1: Running process_dataset.py..."
cd "$ROOT_DIR"
python process_dataset.py

# Change back to the script directory
cd "$SCRIPT_DIR"

echo "Step 2: Converting finetuned model predictions to comparable format..."
python convert_data_to_comparable_format.py test_predictions_finetuned_1.5B_improved.json -d prediction

echo "Step 3: Converting finetuned model ground truth to comparable format..."
python convert_data_to_comparable_format.py test_predictions_finetuned_1.5B.json -d ground_truth

echo "Step 4: Converting original model predictions to comparable format..."
python convert_data_to_comparable_format.py test_predictions_original_1.5B.json -d prediction

echo "Step 5: Converting original model ground truth to comparable format..."
python convert_data_to_comparable_format.py test_predictions_original_1.5B.json -d ground_truth

echo "Test pipeline completed!" 