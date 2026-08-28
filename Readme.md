# FeatureRank

FeatureRank is an autoencoder-based feature ranking and evaluation pipeline.
It is designed for high-dimensional datasets, where thousands of input
features must be ranked, reduced, and evaluated without changing the
underlying labels or samples.

The repository supports three downstream tasks:

- classification (binary and multiclass one-vs-rest),
- regression, and
- unsupervised clustering.

The canonical command is [scripts/feature_ranking.py](scripts/feature_ranking.py).
It provides two modes:

~~~
GLOBAL  -> rank the complete feature matrix in one run
DC      -> split features into blocks, rank each block, combine selections,
           and run the final workflow on the combined dataset
~~~

The existing task implementations are reused by both modes. The command line
is intentionally compact; model and training defaults are kept centrally in
[src/config.py](src/config.py).


### Common preprocessing

For every experiment the project:

1. loads the feature/label files,
2. removes the optional ID column,
3. cleans mixed or missing feature values,
4. encodes classification labels when necessary,
5. creates a reproducible train/test split, and
6. scales the feature matrix before model training.

The target column is not used as an input feature. The default split and
training values are defined in src/config.py.

### GLOBAL mode

GLOBAL trains FeatureRank on the complete feature matrix:

~~~
all features
     |
one autoencoder
     |
global feature scores
     |
top-p feature list
     |
task training and evaluation
~~~

### Divide & Combine (DC) mode

DC is intended for very wide datasets. It divides feature columns into evenly
sized blocks; rows are never divided. Each block goes through the existing
FeatureRank workflow independently. The local feature names are translated
back to the original dataset names using the generated mapping file. The
selected features from all blocks are then deduplicated and combined:

~~~
original dataset
       |
split feature columns into N blocks
       |
FeatureRank on every block
       |
select p% from every block
       |
translate local names with feature_block_mapping.csv
       |
combine and deduplicate selected original features
       |
final task workflow on the combined dataset
~~~

The orchestration lives in
[src/divide_combine.py](src/divide_combine.py). Splitting, mapping, and
combining are delegated to the reusable functions in
[scripts/feature_block_dataset_tools.py](scripts/feature_block_dataset_tools.py).
The mathematical ranking and task training code is not duplicated for DC.

## Repository layout

~~~
Feature_Ranking_Project/
├── data/
│   ├── raw/                         # Input feature and label files
│   ├── filtered/                    # Task-level filtered datasets
│   └── paper_dimension_reduction/   # Generated reduced representations
├── outputs/
│   ├── Classification/              # Classification metrics and artifacts
│   ├── Regression/                  # Regression metrics and artifacts
│   ├── Clustering/                  # Clustering metrics and artifacts
│   ├── paper_dimension_reduction/   # Dimension-reduction evaluations
│   └── FIGURES/                     # Publication-oriented figures

├── split_datasets/                  # DC mappings, blocks, and combined data
├── scripts/
│   ├── feature_ranking.py           # Canonical single-entry CLI
│   ├── run_autoencoder.py           # Backward-compatible wrapper
│   ├── feature_block_dataset_tools.py
│   ├── run_block_feature_selection.py
│   ├── generate_paper_dimension_reduction.py
│   └── evaluate_paper_dimension_reduction.py
├── src/
│   ├── classification.py
│   ├── regression.py
│   ├── clustering.py
│   ├── divide_combine.py
│   ├── autoencoder_feature_selection.py
│   ├── workflow.py
│   ├── experiment.py                # Shared numerical/reporting helpers
│   ├── config.py                    # Project defaults
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   └── utils.py
├── requirements.txt
├── requirements-gpu.txt
└── Readme.md
~~~

Older output directories may also be present in an existing checkout. New
canonical task runs use the capitalized outputs/Classification,
outputs/Regression, and outputs/Clustering directories.

### Where to find each responsibility

The task modules expose the readable top-level flows: `classification.py`,
`regression.py`, and `clustering.py`. Dataset loading and train/test preparation
are in `data_loader.py` and `preprocessing.py`; model definitions are in
`models.py`; and the FeatureRank contribution formula plus top-percent selection
are in `autoencoder_feature_selection.py`. `output_paths.py` is the single
source for artifact names and task output directories, while `config.py` holds
project defaults. `experiment.py` is retained for shared model-training helpers
used by task modules and legacy workflows. The standalone
`scripts/simple.py` file is a historical example and is not the production CLI.

### Readability and configuration

The command-line scripts keep only the choices a user normally needs. Detailed
model settings are grouped in small, named configuration objects inside the
specialist workflows (`TrainingSettings`, `SplitOptions`, `CombineOptions`,
and `DocumentAutoencoderSettings`). This keeps function calls short while
leaving every default and random seed explicit and reproducible. The objects do
not introduce a new algorithm; they only carry the same values that the
previous long argument lists carried.

Optional GPU dependencies are listed in requirements-gpu.txt. See
[GPU_SETUP.md](GPU_SETUP.md) for environment-specific instructions.

Verify the installation without starting an experiment:

~~~
python scripts/feature_ranking.py --help
~~~

## Dataset format

Place datasets in data/raw/. The standard project format uses two headerless
files with the same number of rows:

~~~
data/raw/<dataset>_data.csv
data/raw/<dataset>_label.csv
~~~

For example:

~~~
data/raw/breast_cancer_data.csv
data/raw/breast_cancer_label.csv
~~~

The data file contains samples by rows and features by columns. The label file
contains one target value per row. The loader also accepts a normal CSV that
already contains a named target column when used by the block utilities.

Classification targets can be binary or multiclass. Regression targets should
be numeric and continuous. If your target column is not named target, pass it
explicitly with --target-column.

The repository includes several prepared datasets. The Gen Expression dataset
can also be obtained from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq).

## Quick start

Activate the virtual environment first, then run commands from the project
root.

### GLOBAL: default mode

If neither mode flag is supplied, GLOBAL is used:

~~~
python scripts/feature_ranking.py \
  --dataset-name carcinom_data.csv \
  --feature-percent 40
~~~

The explicit form is equivalent:

~~~
python scripts/feature_ranking.py \
  --dataset-name carcinom_data.csv \
  --feature-percent 40 \
  --global
~~~

### DC: split, rank, combine, and evaluate

~~~
python scripts/feature_ranking.py \
  --dataset-name arcene_data.csv \
  --feature-percent 50 \
  --dc \
  --block-count 10
~~~

--global and --dc are mutually exclusive. --block-count applies only to DC
and defaults to 10.

### Classification

~~~
python scripts/feature_ranking.py \
  --dataset-name breast_cancer_data.csv \
  --task classification \
  --feature-percent 20 \
  --random-state 42
~~~

Binary datasets use a binary classifier. For more than two labels, the
existing one-vs-rest implementation trains one binary model per class and
reports the aggregate metrics.

### Regression

~~~
python scripts/feature_ranking.py \
  --dataset-name air_data.csv \
  --task regression \
  --feature-percent 30 \
  --random-state 42
~~~

Regression outputs include MSE, RMSE, MAE, R2, and Pearson correlation where
the corresponding metric is available.

### Clustering

~~~
python scripts/feature_ranking.py \
  --dataset-name codon_usage_data.csv \
  --task clustering \
  --feature-percent 60
~~~

To request a fixed number of clusters:

~~~
python scripts/feature_ranking.py \
  --dataset-name codon_usage_data.csv \
  --task clustering \
  --feature-percent 60 \
  --cluster-k 8
~~~

Without --cluster-k, the existing elbow/silhouette procedure selects the best
candidate from the configured range. Labels, when present, are not used to
train KMeans.

### Several percentages

Use a comma-separated list or all:

~~~
python scripts/feature_ranking.py \
  --dataset-name breast_cancer_data.csv \
  --feature-percent 10,20,40,60
~~~

~~~
python scripts/feature_ranking.py \
  --dataset-name breast_cancer_data.csv \
  --feature-percent all \
  --random-state 42
~~~

all expands to 10%, 20%, ..., 100%.

### Repeated runs

~~~
python scripts/feature_ranking.py \
  --dataset-name breast_cancer_data.csv \
  --feature-percent 60 \
  --repeat-runs 50 \
  --random-state 42
~~~

When multiple runs are requested and a seed is provided, the run seed is
incremented for each repetition. --random-state none disables a fixed seed.
Repeated runs are independent experiments; they are not automatically
five-fold cross-validation.

## Command-line reference

Run python scripts/feature_ranking.py --help for the authoritative list.
The main options are:

| Option | Default | Purpose |
|---|---:|---|
| --dataset-name | breast_cancer_data.csv | Dataset filename in data/raw/ |
| --task | classification | classification, regression, or clustering |
| --feature-percent | 20 | One percentage, comma-separated values, or all |
| --random-state | 42 | Reproducibility seed; use none for random behavior |
| --repeat-runs | 1 | Number of independent runs |
| --target-column | target | Target column name |
| --id-column | ID | Optional ID column; use none when absent |
| --encoding-dim | 8 | Autoencoder latent dimension |
| --cluster-k | unset | Fixed KMeans cluster count |
| --save-details | off | Save additional training-history artifacts |
| --global | default mode | Run normal global FeatureRank |
| --dc | off | Run Divide & Combine |
| --block-count | 10 | Number of feature blocks in DC mode |

Architecture, epochs, batch size, learning rate, early stopping, class
sampling, chunking thresholds, and KMeans ranges are configured in
[src/config.py](src/config.py). Keeping these values centralized makes the
CLI easier to read and keeps experiments consistent.

## Output files

### GLOBAL output

For a classification dataset named arcene_data.csv, the main output folder is:

~~~
outputs/Classification/arcene_data/
├── first_layer_W_list.csv
├── top_50_max_abs_features.csv
├── ORG_*.png / ORG_*.csv
└── metrics/
    ├── ORG_test_metrics.json
    └── top_50_test_metrics.json
~~~

Regression and clustering use the corresponding task directory. Clustering
also writes cluster scores, assignments, PCA figures, and cluster metrics.

### DC output

For arcene_data.csv with ten blocks, DC creates:

~~~
split_datasets/arcene/
├── arcene_block_01.csv ... arcene_block_10.csv
├── feature_block_mapping.csv
└── split_summary.csv

data/raw/
├── arcene_block_01_data.csv ... arcene_block_10_data.csv
├── arcene_block_01_label.csv ... arcene_block_10_label.csv
├── arcene_selected_features_combined_data.csv
└── arcene_selected_features_combined_label.csv

split_datasets/arcene_selected_features_combined.csv
outputs/Classification/arcene_block_01_data/
...
outputs/Classification/arcene_block_10_data/
outputs/Classification/arcene_selected_features_combined_data/
~~~

feature_block_mapping.csv records the original feature, block number, and
local feature index. The combined dataset contains the selected original
features and the target column. Duplicate selections are removed before the
final workflow.

## Backward-compatible and specialist scripts

run_autoencoder.py remains as a compatibility wrapper around
feature_ranking.py:

~~~
python scripts/run_autoencoder.py \
  --dataset-name breast_cancer_data.csv \
  --feature-percent 20
~~~

For the lower-level DC helpers, feature_block_dataset_tools.py exposes split,
combine, and split-and-combine modes. The standalone
run_block_feature_selection.py is a comparison experiment that uses an
explicit --block-size; it is separate from the canonical feature_ranking.py
--dc orchestration.

## Dimension-reduction workflow

Dimension reduction is intentionally separate from FeatureRank selection. The
paper-style workflow exports the encoding-layer representation without a
second autoencoder or feature ranking step.

Generate reduced datasets:

~~~
python scripts/generate_paper_dimension_reduction.py \
  --dataset-name arcene_data.csv \
  --retained-percent all \
  --repeat-runs 50 \
  --base-seed 42
~~~

Evaluate the exported representations:

~~~
python scripts/evaluate_paper_dimension_reduction.py \
  --dataset-name arcene_data
~~~

See [DIMENSION_REDUCTION_WORKFLOW.md](DIMENSION_REDUCTION_WORKFLOW.md) for the
full method and output contract.

Refactor and smoke-test results are recorded in
[REFACTOR_VALIDATION.md](REFACTOR_VALIDATION.md).

## Reproducibility and research use

For comparable results:

1. Use the same Python and dependency versions.
2. Keep the input data and target files unchanged.
3. Record the exact command and --random-state value.
4. Keep src/config.py unchanged, or record any deliberate changes.
5. Compare the generated metric JSON/CSV files from the same mode and task.

Feature ranking files for raw classification runs are regenerated from the
autoencoder trained in the current run. Existing files for the same dataset
and percentage can therefore be overwritten; archive them before rerunning if
the earlier result must be preserved.

For an academic report, include the dataset source, preprocessing choices,
feature percentage, mode (GLOBAL or DC), block count where applicable, random
seed, and the relevant metric files when describing an experiment.

## Troubleshooting

### Dataset or label file not found

Check that both files exist and use the expected naming convention:

~~~
data/raw/<name>_data.csv
data/raw/<name>_label.csv
~~~

Also run commands from the repository root. If the target is not named target,
add --target-column <name>.

### IndexError: single positional indexer is out-of-bounds

This generally indicates an incomplete or incompatible metric artifact from a
previous interrupted run. Preserve any result you need, then remove only the
affected dataset output folder and rerun the command. In DC mode, verify that
the final combined metrics folder exists under the combined dataset name.

### DC cannot find a block selection file

Check that every block completed and that the requested percentage matches the
top_<percent>_max_abs_features.csv files. Also verify that
feature_block_mapping.csv belongs to the same dataset and block count.

### Gen Expression is slow or uses too much memory

Gen Expression has thousands of features and multiclass classification runs
one-vs-rest training for every class. Use DC to process narrower blocks, or
limit CPU threads:

~~~
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/feature_ranking.py \
  --dataset-name gen_expression_data.csv \
  --task classification \
  --feature-percent 20 \
  --random-state 42
~~~

### TensorFlow installation or GPU issue

Confirm that the virtual environment is active and reinstall from the
appropriate requirements file. Consult [GPU_SETUP.md](GPU_SETUP.md) for GPU
configuration and fall back to CPU if the installed TensorFlow build does not
support the local hardware.

## Tests

Run the repository tests from the project root:

~~~
pytest -q
python -m compileall -q scripts src
~~~

The tests cover the compact experiment configuration and the canonical
FeatureRank contribution/ranking formula. A small dataset run is recommended
after changing model or preprocessing defaults.
