# Feature Ranking Project

This project uses an autoencoder-based workflow for feature ranking, dimension reduction, classification, regression, and clustering experiments. It was developed for an academic study and is organized so that the experiments can be rerun from the command line.

## What This Project Does

The main workflow is:

1. Load a dataset from `data/raw/`.
2. Preprocess the feature matrix and labels.
3. Train an autoencoder.
4. Rank features using the learned encoder weights.
5. Select the requested feature percentage.
6. Train and evaluate a model using the selected features.
7. Save metrics, selected features, plots, and experiment outputs.

The project also includes separate scripts for large feature-block experiments and paper-style dimension reduction experiments.

## Repository Structure

```text
Feature_Ranking_Project/
├── data/
│   ├── raw/                         # Input datasets
│   ├── autoencoder/                 # Selected feature datasets
│   └── paper_dimension_reduction/   # Generated reduced datasets
├── outputs/                         # Metrics, plots, and experiment results
├── scripts/
│   ├── run_autoencoder.py           # Main experiment script
│   ├── run_block_feature_selection.py
│   ├── generate_paper_dimension_reduction.py
│   ├── evaluate_paper_dimension_reduction.py
│   └── create_*.py                  # Figure generation scripts
├── src/                             # Shared project code
├── requirements.txt
└── Readme.md
```

## Installation

Python 3.13 is recommended.

### macOS / Linux

```bash
cd Feature_Ranking_Project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows

```powershell
cd Feature_Ranking_Project
py -3.13 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check that the main script works:

```bash
python scripts/run_autoencoder.py --help
```

## Dataset Format

All datasets used in this project should be placed in `data/raw/`.

Most datasets are included in the GitHub repository under:

```text
data/raw/
```

The only exception is the Gen Expression dataset, which is not included in the repository because of its large file size. It can be downloaded from the UCI Machine Learning Repository:

[Gene Expression Cancer RNA-Seq Dataset](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq)

After downloading it, place the corresponding data and label files in `data/raw/` using the same naming format used by the project.

The standard format uses two CSV files:

```text
data/raw/example_data.csv
data/raw/example_label.csv
```

- `example_data.csv`: rows are samples, columns are features.
- `example_label.csv`: one label or target value per row.
- The data and label files must have the same number of rows.
- Classification labels may be binary or multiclass.
- Regression labels must be numeric continuous values.

Example:

```text
data/raw/breast_cancer_data.csv
data/raw/breast_cancer_label.csv
```

Run it with:

```bash
python scripts/run_autoencoder.py --dataset-name breast_cancer_data.csv
```

## Basic Usage

### Classification

```bash
python scripts/run_autoencoder.py \
  --dataset-name breast_cancer_data.csv \
  --task classification \
  --feature-percent 20 \
  --random-state 42
```

### Regression

```bash
python scripts/run_autoencoder.py \
  --dataset-name air_data.csv \
  --task regression \
  --feature-percent 30 \
  --random-state 42
```

### Clustering

```bash
python scripts/run_autoencoder.py \
  --dataset-name codon_usage_data.csv \
  --task clustering \
  --feature-percent 60 \
  --save-training-plots
```

To run clustering with a fixed cluster number:

```bash
python scripts/run_autoencoder.py \
  --dataset-name codon_usage_data.csv \
  --task clustering \
  --feature-percent 60 \
  --cluster-k 8 \
  --save-training-plots
```

### Run All Feature Percentages

```bash
python scripts/run_autoencoder.py \
  --dataset-name breast_cancer_data.csv \
  --task classification \
  --feature-percent all \
  --random-state 42
```

## Outputs

Results are saved under `outputs/`.

Common output folders:

```text
outputs/
├── Classification/
│   └── <dataset_name>/
├── Regression/
│   └── <dataset_name>/
├── clustering/
│   └── <dataset_name>/
├── split_datasets/
└── paper_dimension_reduction/
```

Common output files include:

- selected feature lists
- test metric JSON files
- repeated run text files
- confusion matrix, ROC, PR, boxplot, convergence, and clustering figures

## Large Feature-Block Workflow

For very wide datasets, the project includes a block-based feature-selection workflow:

```bash
python scripts/run_block_feature_selection.py \
  --dataset-name gen_expression_data.csv \
  --feature-percent 10 \
  --block-count 10
```

This splits the feature columns into blocks, ranks features inside each block, merges the selected features, and evaluates the final selected feature set.

## Dimension Reduction Workflow

The paper-style dimension reduction workflow is separated into two scripts.

First, generate reduced datasets:

```bash
python scripts/generate_paper_dimension_reduction.py \
  --dataset-name arcene_data.csv \
  --retained-percent all \
  --repeat-runs 50 \
  --base-seed 42
```

Then evaluate the generated reduced datasets:

```bash
python scripts/evaluate_paper_dimension_reduction.py \
  --dataset-name arcene_data
```

Generated reduced datasets are stored under:

```text
data/paper_dimension_reduction/
```

Evaluation results are stored under:

```text
outputs/paper_dimension_reduction/
```

## Reproducing Results

For reproducible experiments:

1. Use the same Python version and `requirements.txt`.
2. Place datasets in `data/raw/` using the expected naming format.
3. Use the same `--random-state` value.
4. Use the same feature percentage and model parameters.
5. Keep previous output folders if you want to reuse existing selected feature lists.
6. Archive or rename previous output folders if you want to force a fresh run.

Repeated runs can be executed with:

```bash
python scripts/run_autoencoder.py \
  --dataset-name breast_cancer_data.csv \
  --task classification \
  --feature-percent 60 \
  --repeat-runs 50 \
  --random-state none
```

`--repeat-runs` performs repeated experiments. It is not automatically the same as 5-fold cross-validation unless a script explicitly implements cross-validation.

## Figure Scripts

Several scripts under `scripts/` generate publication figures from existing outputs:

```bash
python scripts/create_classification_figure.py
python scripts/create_regression_figure_2.py
python scripts/create_cluster_figure_1.py
python scripts/create_cluster_classification_summary_figure.py
```

These scripts expect the required metric and plot files to already exist under `outputs/`.

## Common Problems

### Label file cannot be found

Check that both files exist:

```text
data/raw/<name>_data.csv
data/raw/<name>_label.csv
```

### Data and label row counts do not match

Make sure the data and label files have the same number of rows and do not contain an extra header row.

### TensorFlow cannot be installed

Check that the active environment uses the expected Python version, then reinstall:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### The process is killed or memory is exhausted

Use the block feature-selection workflow, reduce the block size, or run the experiment on a machine with more RAM.
