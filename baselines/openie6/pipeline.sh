#!/bin/bash

# Base directory for OpenIE6
OPENIE6_DIR="$(pwd)"
RESULT_DIR="$OPENIE6_DIR/result/controlled_extraction/test"

# Check if number of folders is provided
if [ $# -eq 0 ]; then
    echo "Error: Number of folders must be provided as an argument"
    echo "Usage: $0 <number_of_folders>"
    exit 1
fi

NUM_FOLDERS=$1

# Function to process a single folder
process_folder() {
    local folder_num=$1
    local folder_path="$RESULT_DIR/$folder_num"
    
    echo "=========================================="
    echo "Processing folder $folder_num"
    echo "=========================================="
    
    # Check if text.txt exists
    if [ ! -f "$folder_path/text.txt" ]; then
        echo "Error: text.txt not found in folder $folder_num"
        return 1
    fi
    
    # Read text.txt and process it
    echo "Reading and processing text.txt..."
    
    # Create a temporary Python script for sentence tokenization
    cat > "$OPENIE6_DIR/temp_tokenize.py" << 'EOF'
import nltk
import sys
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Read input text
input_path = sys.argv[1]
print(f"Reading from: {input_path}")
print(f"File exists: {os.path.exists(input_path)}")

with open(input_path, 'r') as f:
    text = f.read()

# Tokenize into sentences
sentences = nltk.sent_tokenize(text)

# Write each sentence on a new line
output_path = sys.argv[2]
print(f"Writing to: {output_path}")

with open(output_path, 'w') as f:
    for sentence in sentences:
        if sentence.strip():  # Only write non-empty sentences
            f.write(sentence.strip() + '\n')
EOF

    # Remove existing sentences.txt if it exists
    echo "Removing existing sentences.txt..."
    rm -f "$OPENIE6_DIR/sentences.txt"
    
    # Process the text using Python script
    echo "Tokenizing sentences..."
    docker run --gpus all -v "$OPENIE6_DIR":/workspace openie6:cuda10 \
        python3 /workspace/temp_tokenize.py \
        "/workspace/result/controlled_extraction/test/$folder_num/text.txt" \
        "/workspace/sentences.txt"
    
    # Clean up temporary script
    rm -f "$OPENIE6_DIR/temp_tokenize.py"
    
    # Verify sentences.txt was created and is not empty
    if [ ! -s "$OPENIE6_DIR/sentences.txt" ]; then
        echo "Error: Failed to create sentences.txt or file is empty"
        return 1
    fi
    
    # Run OpenIE6
    echo "Running OpenIE6..."
    if ! docker run --gpus all -v "$OPENIE6_DIR":/workspace openie6:cuda10 \
        python3 run.py --mode splitpredict \
        --inp sentences.txt --out predictions.txt \
        --task oie --gpus 1 \
        --oie_model models/oie_model/epoch=14_eval_acc=0.551_v0.ckpt \
        --conj_model models/conj_model/epoch=28_eval_acc=0.854.ckpt \
        --num_extractions 5; then
        echo "Error: OpenIE6 processing failed for folder $folder_num"
        return 1
    fi
    
    # Check if predictions file was created
    if [ ! -f "$OPENIE6_DIR/predictions.txt.oie" ]; then
        echo "Error: predictions.txt.oie was not created"
        return 1
    fi
    
    # Copy predictions to the corresponding folder
    echo "Copying predictions to folder $folder_num..."
    cp "$OPENIE6_DIR/predictions.txt.oie" "$folder_path/"
    
    echo "Completed processing folder $folder_num"
    echo "=========================================="
    echo ""
    return 0
}

# Main execution
echo "Starting OpenIE6 pipeline processing..."
echo "Processing $NUM_FOLDERS folders in sequence..."

# Process folders 1 through NUM_FOLDERS
for i in $(seq 1 $NUM_FOLDERS); do
    if ! process_folder $i; then
        echo "Failed to process folder $i, continuing with next folder..."
        continue
    fi
done

echo "Pipeline processing completed!" 