# Dataset Investigation System

A tool for analyzing knowledge graph datasets with statistical visualizations and reports.

## Installation

```bash
pip install -r requirements.txt
```

## Triple Count Distribution Report

To generate a triple count distribution report, run:

```bash
python dataset_analyzer.py --dataset DATASET_NAME --partition PARTITION --report_type triple_count_distribution
```

### Examples:

```bash
python dataset_analyzer.py --dataset CE3000 --partition test --report_type triple_count_distribution
```

![Triple Count Distribution Example](./triple_distribution_CE3000_test.png)

## Token Count Distribution Report

To generate a token count distribution report, run:

```bash
python dataset_analyzer.py --dataset DATASET_NAME --partition PARTITION --report_type token_count_distribution
```

### Examples:

```bash
python dataset_analyzer.py --dataset CE3000 --partition test --report_type token_count_distribution
```

![Token Count Distribution Example](./token_count_distribution_CE3000_test.png)

## Named Entity Distribution Report

To generate a named entity distribution report, run:

```bash
python dataset_analyzer.py --dataset DATASET_NAME --partition PARTITION --report_type named_entity_distribution
```

### Examples:

```bash
python dataset_analyzer.py --dataset CE3000 --partition test --report_type named_entity_distribution
```

This report generates multiple visualizations:

**Main Distribution:**
![Named Entity Distribution](./named_entity_distribution_CE3000_test.png)

**Entity Density Analysis:**
![Entity Density per Token](./named_entity_density_per_token_CE3000_test.png)
![Entity Density per Sentence](./named_entity_density_per_sentence_CE3000_test.png)

**Entity Type Analysis:**
![Entity Type Counts](./entity_types_counts_CE3000_test.png)
![Entity Type Proportions](./entity_types_proportions_CE3000_test.png)

## Cross-Dataset Evaluation Report

To generate a cross-dataset evaluation report, run:

```bash
python dataset_analyzer.py --report_type cross_dataset_evaluation --train_dataset TRAIN_DATASET --test_dataset TEST_DATASET
```

### Examples:

```bash
python dataset_analyzer.py --report_type cross_dataset_evaluation --train_dataset CE3000 --test_dataset webnlg20
```