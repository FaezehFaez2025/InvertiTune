# Visualization Pipeline Script

The `visualization_pipeline.sh` script provides a complete workflow for visualizing the distribution of triples and tokens. It takes one argument:

```bash
./visualization_pipeline.sh <number_of_samples>
```

Where `number_of_samples` is the total number of samples (train + test) used to train the model. This number is used to distinguish between models trained on different dataset sizes.

The pipeline performs the following steps in sequence:

1. **Prune prediction triples**: Cleans and validates the predicted triples
2. **Extract distribution data**: Generates distribution files for both triples and tokens
3. **Visualize distributions**: Creates plots for both triple and token distributions

The output includes:
- Distribution files in the `distribution` directory
- Visualization plots in the `plots` directory

Example usage:
```bash
./visualization_pipeline.sh 1000
```

This will generate distribution files with the prefix "1000" to distinguish them from other model variants.

# Distribution Visualizer

A Python utility for visualizing and comparing the distribution of ground truth and prediction triples or tokens across different sample sizes.

## Features

- Parses distribution files in the format `value: count`
- Automatically pairs ground truth and prediction files based on sample size
- Creates KDE (Kernel Density Estimation) plots for visual comparison
- Supports visualization of both triples and tokens
- Supports multiple sample sizes in a single visualization

## Usage

```bash
python distribution_visualizer.py --distribution_type triples
```
```bash
python distribution_visualizer.py --distribution_type tokens
```

### Arguments

- `--distribution_dir`: Directory containing distribution files (default: 'distribution')
- `--output_dir`: Directory to save output plots (default: 'plots')
- `--distribution_type`: Type of distribution to visualize ('triples' or 'tokens', default: 'triples')

## Input File Format

The tool expects pairs of files in the following format:
- Ground truth files: `<sample_size>_<type>_gt.txt` (e.g., `500_triples_gt.txt`, `500_tokens_gt.txt`)
- Prediction files: `<sample_size>_<type>_pr.txt` (e.g., `500_triples_pr.txt`, `500_tokens_pr.txt`)

Each file should contain lines in the format `value: count`, where:
- `value` is the number of triples/tokens per sample
- `count` is the frequency of that value

## Output

The tool generates a PDF file named `<type>_distribution_comparison.pdf` in the specified output directory. The plot includes:
- A subplot for each sample size
- KDE curves for both ground truth (black) and predictions (red)
