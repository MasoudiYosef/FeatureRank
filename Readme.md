# Feature Ranking Project

This project uses an autoencoder to rank features and evaluate selected feature subsets on classification, regression, and clustering tasks. It also includes separate workflows for large datasets and autoencoder-based dimension reduction.

The README focuses on the commands needed to install and run the project. For every available option, use:

```bash
python scripts/run_autoencoder.py --help
```

## What the Project Does

The main FeatureRank workflow is:

1. Load and preprocess a dataset.
2. Train an autoencoder without using the target labels.
3. Calculate feature importance from the first encoder layer weights.
4. Select the highest-ranked percentage of original features.
5. Train and evaluate a new model using only the selected features.
6. Save rankings, metrics, histories, and figures under `outputs/`.

FeatureRank keeps original feature columns. The separate dimension-reduction workflow instead creates new latent features from encoder outputs.

## Repository Structure

```text
Feature_Ranking_Project/
|-- data/
|   |-- raw/                         # Input datasets and labels
|   `-- processed/                   # Generated processed data
|-- outputs/
|   |-- autoencoder/                 # Classification and regression results
|   |-- clustering/                  # Clustering results
|   `-- FIGURES/                     # Combined publication figures
|-- scripts/
|   |-- run_autoencoder.py           # Main experiment script
|   |-- run_block_feature_selection.py
|   |-- generate_paper_dimension_reduction.py
|   |-- evaluate_paper_dimension_reduction.py
|   `-- create_*.py                  # Figure-generation scripts
|-- src/                             # Data, preprocessing, model, and utility code
|-- requirements.txt
|-- .python-version
`-- README.md
```

## Installation

### Requirements

- Python 3.13.5
- pip
- A virtual environment is recommended
- GPU is optional; the project can run on CPU

### macOS or Linux

```bash
git clone <REPOSITORY_URL>
cd Feature_Ranking_Project

python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
git clone <REPOSITORY_URL>
cd Feature_Ranking_Project

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Confirm that the main script is available:

```bash
python scripts/run_autoencoder.py --help
```

`<REPOSITORY_URL>` is a placeholder because the public repository URL is not stored in this project.

## Dataset Format

Place datasets in `data/raw/`. The standard format uses two headerless CSV files:

```text
data/raw/example_data.csv
data/raw/example_label.csv
```

- `example_data.csv`: rows are samples and columns are features.
- `example_label.csv`: one label or target value per row.
- The data and label files must have exactly the same number of rows.
- Classification labels may be binary or multiclass.
- Regression labels must be numeric continuous values.

Example:

```text
data/raw/breast_cancer_data.csv
data/raw/breast_cancer_label.csv
```

### Large dataset files

Some datasets used in the experiments may be too large to store directly in the GitHub repository. In that case, the repository can include only the code and README instructions, while the large dataset files are provided through external download links.

After downloading a large dataset, place both files under `data/raw/` using the same naming format:

```text
data/raw/<dataset_name>_data.csv
data/raw/<dataset_name>_label.csv
```

Large dataset links can be listed here:

| Dataset | Data file | Label file |
| --- | --- | --- |
| Arcene | `<ARCENE_DATA_LINK>` | `<ARCENE_LABEL_LINK>` |
| Gen Expression | `<GEN_EXPRESSION_DATA_LINK>` | `<GEN_EXPRESSION_LABEL_LINK>` |
| Carcinom | `<CARCINOM_DATA_LINK>` | `<CARCINOM_LABEL_LINK>` |

Replace the placeholder links with the actual download links before publishing the repository.

The command then uses the data filename:

```bash
python scripts/run_autoencoder.py --dataset-name breast_cancer_data.csv
```

## Basic Usage

### Classification

Select and evaluate the top 20% of ranked features:

```bash
python scripts/run_autoencoder.py \
  --dataset-name breast_cancer_data.csv \
  --task classification \
  --feature-percent 20 \
  --random-state 42
```

Run the same experiment repeatedly and save training plots:

```bash
python scripts/run_autoencoder.py \
  --dataset-name breast_cancer_data.csv \
  --task classification \
  --feature-percent 60 \
  --repeat-runs 50 \
  --random-state 42 \
  --save-training-plots
```

### Regression

```bash
python scripts/run_autoencoder.py \
  --dataset-name air_data.csv \
  --task regression \
  --feature-percent 30 \
  --random-state 42 \
  --save-training-plots
```

Regression outputs include MSE, RMSE, MAE, R-squared, cosine similarity, and Pearson correlation when they can be calculated.

### Clustering

Automatically evaluate the configured range of cluster counts:

```bash
python scripts/run_autoencoder.py \
  --dataset-name codon_usage_data.csv \
  --task clustering \
  --feature-percent 60 \
  --random-state 42 \
  --save-training-plots
```

Use a specific number of clusters:

```bash
python scripts/run_autoencoder.py \
  --dataset-name codon_usage_data.csv \
  --task clustering \
  --feature-percent 60 \
  --cluster-k 8 \
  --random-state 42 \
  --save-training-plots
```

Clustering outputs include silhouette scores, WCSS values, elbow/silhouette plots, cluster assignments, and PCA visualizations.

### Run Multiple Feature Percentages

Use the percentage-sweep option to run the supported feature percentages in sequence:

```bash
python scripts/run_autoencoder.py \
  --dataset-name arcene_data.csv \
  --task classification \
  --feature-percent all \
  --random-state 42
```

If the installed version does not accept `all`, check the current option name with `--help` before running the sweep.

## Outputs

Classification and regression results are normally stored under:

```text
outputs/autoencoder/<dataset_name>/
```

Clustering results are stored under:

```text
outputs/clustering/<dataset_name>/
```

Typical files include:

- `first_layer_W_list.csv`: first-layer weights
- `top_<percentage>_max_abs_features.csv`: ranked and selected features
- `top_<percentage>_test_metrics.json`: evaluation metrics
- training-history CSV files
- accuracy, loss, boxplot, ROC, precision-recall, confusion-matrix, PCA, and clustering figures

The exact files depend on the task and command options.

## Large Datasets

The main script includes feature chunking for very wide datasets. A separate block workflow is also available when the feature matrix is too large to process as one model:

```bash
python scripts/run_block_feature_selection.py \
  --dataset-name arcene_data.csv \
  --target-column target \
  --id-column none \
  --block-size 1000 \
  --feature-percent 20 \
  --random-state 42
```

This workflow divides feature columns into blocks, ranks features inside each block, combines the selected features, and evaluates the merged subset.

For the current command-line options of this script, run:

```bash
python scripts/run_block_feature_selection.py --help
```

## Dimension Reduction Experiment

Dimension reduction is a separate experiment from FeatureRank:

- FeatureRank selects original feature columns.
- Dimension reduction creates new latent features from encoder outputs.

Generate reduced datasets for all configured percentages:

```bash
python scripts/generate_paper_dimension_reduction.py \
  --dataset-name arcene_data.csv \
  --retained-percent all \
  --repeat-runs 50 \
  --base-seed 42
```

Evaluate the generated reduced datasets:

```bash
python scripts/evaluate_paper_dimension_reduction.py \
  --dataset-name arcene_data
```

The generation script trains the reduction model using training data and applies the same encoder to unseen test data. The evaluation script reads the generated datasets and reports their performance. Use the same seeds and evaluation settings when comparing this experiment with FeatureRank.

## Reproducing Results

For a repeatable experiment:

1. Use Python 3.13.5 and install the pinned `requirements.txt`.
2. Use the same dataset files without changing row order or labels.
3. Set `--random-state 42`, or record the seed used in the paper.
4. Use the same task, feature percentage, epoch settings, and number of repeats.
5. Keep the generated JSON metrics and CSV rankings with the experiment record.
6. Record whether CPU or GPU was used because low-level numerical behavior can differ across systems.

`--repeat-runs` performs repeated train/test experiments. It is not automatically a 5-fold cross-validation experiment. Do not describe it as cross-validation unless a dedicated cross-validation workflow was used.

The main workflow may reuse an existing compatible selected-feature file from the dataset output directory. For a completely fresh ranking experiment, archive the previous dataset output directory before running again.

## Creating Figures

The repository contains scripts named `create_*.py` for publication figures. Their required inputs depend on previously generated experiment outputs.

Examples:

```bash
python scripts/create_classification_figure.py
python scripts/create_regression_figure_2.py
python scripts/create_cluster_figure_1.py
```

Open each script and update its dataset configuration section before use. Combined figures are generally saved under `outputs/FIGURES/` or `outputs/autoencoder/`.

## Common Problems

### Data and label row counts do not match

Check that `<name>_data.csv` and `<name>_label.csv` contain the same number of rows and do not contain an extra header row.

### Label file cannot be found

Use the expected pair:

```text
<name>_data.csv
<name>_label.csv
```

Both files must be in `data/raw/`.

### TensorFlow cannot be installed

Confirm that the active virtual environment uses the Python version in `.python-version`, then reinstall dependencies inside that environment.

### The process is killed or memory is exhausted

Use the block feature-selection workflow, reduce the block/chunk size, close other memory-intensive programs, or run on a machine with more RAM.

### Results differ between computers

Confirm the Python version, package versions, dataset files, seed, command options, and CPU/GPU environment. Exact neural-network weights are not guaranteed to be identical across different hardware backends.

## Getting Help

Start with the command help:

```bash
python scripts/run_autoencoder.py --help
```

Then inspect the JSON metrics and console messages under the relevant dataset output directory. They record the task, selected feature count, evaluation metrics, and generated file paths.
