#!/bin/bash

# Check if number of samples is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <number_of_samples>"
    echo "Example: $0 2000"
    exit 1
fi

NUM_SAMPLES=$1

# Step 1: Prune prediction triples
echo "Step 1: Pruning prediction triples..."
cd ..
python prune_prediction_triples.py

# Step 2: Extract distribution data
echo -e "\nStep 2: Extracting distribution data..."
cd plotting_utils
python distribution_data_extractor.py $NUM_SAMPLES

# Step 3: Visualize triple distribution
echo -e "\nStep 3: Visualizing triple distribution..."
python distribution_visualizer.py --distribution_type triples

# Step 4: Visualize token distribution
echo -e "\nStep 4: Visualizing token distribution..."
python distribution_visualizer.py --distribution_type tokens

echo -e "\nVisualization pipeline completed!" 