# PiVe Datasets

This directory contains datasets of the PiVe baseline.

## Dataset Statistics Calculator

The `calculate_dataset_statistics.py` script calculates statistics for the datasets in this directory. It analyzes triple statistics from source files and token statistics from the corresponding target files. Each target file contains the text representation of the triples in the corresponding source file.

### Triple Statistics
- Number of data samples
- Total number of triples
- Average number of triples per data sample
- Maximum number of triples in a data sample
- Minimum number of triples in a data sample
- Standard deviation of the number of triples per data sample

### Token Statistics
- Number of data samples
- Total number of tokens
- Average number of tokens per data sample
- Maximum number of tokens in a data sample
- Minimum number of tokens in a data sample
- Standard deviation of the number of tokens per data sample

### Usage

Run with default values (kelm_sub/test):
```bash
python calculate_dataset_statistics.py
```

Specify a different dataset:
```bash
python calculate_dataset_statistics.py --dataset webnlg20
```

Specify a different partition:
```bash
python calculate_dataset_statistics.py --partition train
```

Specify both dataset and partition:
```bash
python calculate_dataset_statistics.py --dataset webnlg20 --partition val
```

### Available Options
- Dataset: `webnlg20` or `kelm_sub` (default: `kelm_sub`)
- Partition: `train`, `val`, or `test` (default: `test`)

## LLaMA-Factory Dataset Builder

The `build_llama_factory_dataset.py` script converts PiVe datasets to the LLaMA-Factory format. The output file is saved within the `llama_factory_data` directory.

### Usage

```bash
python build_llama_factory_dataset.py --dataset kelm_sub --partition test
```

### Available Options
- Dataset: `webnlg20` or `kelm_sub` (required)
- Partition: `train`, or `test` (required)

## Datasets

This directory contains the following datasets:
- `webnlg20`: WebNLG 2020 dataset
- `kelm_sub`: KELM subset dataset 