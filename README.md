# FeatureRank

FeatureRank is a Python project for ranking and selecting features with an
autoencoder. It is intended for tabular datasets in which using every feature
is expensive or makes a model harder to interpret.

The project supports two selection modes:

1. **GLOBAL**: rank the complete feature matrix in one experiment.
2. **Divide & Combine (DC)**: split wide datasets into feature blocks, rank each
   block, combine the selected features, and run a final evaluation.

The selected features can be evaluated with classification, regression, or
unsupervised clustering.

## Installation (PyPI)

FeatureRank requires Python 3.13. Python 3.14 is not supported by the current
TensorFlow release. Check the Python version first:

```bash
python3 --version
```

Install the published package with:

```bash
pip install FeatureRank
```

<img width="1120" height="706" alt="1" src="https://github.com/user-attachments/assets/a80c2e34-da01-484b-a970-241c31a0b0ec" />


Open Python:

```bash
python3
```

<img width="1110" height="119" alt="2" src="https://github.com/user-attachments/assets/7ff33da0-28bf-4a35-89ed-0f824ef22309" />



Then import FeatureRank at the Python prompt:

```python
>>> import FeatureRank
```

<img width="1094" height="71" alt="3" src="https://github.com/user-attachments/assets/a821ccc1-e6c1-4556-a558-c6235c6f4370" />


The import opens the desktop GUI automatically. Do not type `import
FeatureRank` directly in zsh; it is Python code and must be entered after
starting `python3`.

For local development, run these commands from the repository root instead:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

GPU-specific dependencies are listed in
[`requirements-gpu.txt`](requirements-gpu.txt). Hardware notes are available in
[`GPU_SETUP.md`](GPU_SETUP.md).

The repository metadata is version `0.1.3`. Because PyPI releases cannot be
replaced in place, publish this version (or a newer one) to the existing
`FeatureRank` project when you release the refactor. If an environment still
shows an older FeatureRank version, upgrade it with the command above or use the local
editable installation while developing.

After installation, the package GUI can be opened with:

```bash
FeatureRank
```

`import FeatureRank` is Python code, not a shell command. On a desktop, the
following import opens the GUI automatically:

```python
import FeatureRank
```

For applications that want to decide when the window opens, use:

```python
from FeatureRank import Launch

Launch()
```

On servers or CI without a display, set `FEATURERANK_NO_GUI=1` before
importing the package.

On macOS, some Anaconda installations crash in their native `pythonw`/Tk
bridge. If the import returns without a window, use the Python.org 3.13
interpreter (or update Anaconda's Tk package), then install FeatureRank again
in that interpreter's environment.

If you do not want to install the package, use the equivalent script directly:

```bash
python scripts/FeatureRank.py --help
```

## Local Development

The main entry point is:

```text
scripts/FeatureRank.py
```

After the local installation, the workflow can also be called from Python. The
two required arguments are the dataset name and the percentage to select:

```python
import FeatureRank

FeatureRank.run(
    dataset_name="breast_cancer_data.csv",
    feature_percent=20,
    task="classification",
)
```

The same function accepts `mode="dc"` and `block_count=10` for a Divide &
Combine run.

`dataset_name` may be a file name in `data/raw` or a full path to a CSV/TXT
file. A GUI that runs outside the repository should pass the full path to the
user-selected file.

## GUI Workflow

The GUI is the primary end-user interface. Select a dataset from the dropdown
or choose a CSV/TXT file with **Browse…**, select a feature percentage, choose
`GL` or `DC`, and press **START**. `Block Count` is enabled only for `DC`.

When the GUI opens, the experiment form is ready for these choices:

<img width="791" height="764" alt="4" src="https://github.com/user-attachments/assets/75ce8735-4e16-47e8-a617-ba1ed272b974" />



If the required dataset is not listed, press **Browse…** and select its data
file. For paired datasets, keep the matching label file in the same directory:

<img width="1374" height="739" alt="5" src="https://github.com/user-attachments/assets/2e62d8ce-c769-4b81-85b8-63a67a5efda8" />




While the model runs, the progress bar and log show stages such as loading,
feature ranking, block processing, combining, training, and result creation.
When the run finishes, the summary lists the selected feature count, metric,
execution time, and output directory. **Open Results Folder** opens that
directory in Finder, Explorer, or the Linux file manager.

<img width="722" height="198" alt="6" src="https://github.com/user-attachments/assets/d8831206-81f4-48e0-ac18-501b7de4ba84" />



The selected file can be a normal CSV containing a `target` column, or one of
the project's paired files (`*_data.csv` and `*_label.csv`).

## Using from Terminal

The original parameterized CLI remains available for development and backward
compatibility. To inspect its options:

```bash
python scripts/FeatureRank.py --help
```

A normal classification run is:

```bash
python scripts/FeatureRank.py \
  --dataset-name breast_cancer_data.csv \
  --task classification \
  --feature-percent 20 \
  --random-state 42
```

## GLOBAL Mode

GLOBAL is the default when neither `--global` nor `--dc` is supplied. Its
workflow is:

1. load the dataset and labels,
2. remove the optional ID column and clean the features,
3. encode the target when classification is requested,
4. create the reproducible train/test split and scale the features,
5. train the autoencoder,
6. calculate the FeatureRank score for every feature,
7. select the requested percentage, and
8. evaluate the selected feature set for the requested task.

Run GLOBAL explicitly with:

```bash
python scripts/FeatureRank.py \
  --dataset-name carcinom_data.csv \
  --task classification \
  --feature-percent 40 \
  --global
```

The explicit command above and the same command without `--global` produce the
same mode.

## Divide & Combine (DC) Mode

DC is useful for datasets with a very large number of feature columns. Rows and
labels are preserved while only the feature columns are divided.

The steps are:

1. split the feature columns into `N` blocks,
2. run FeatureRank independently on every block,
3. select the requested percentage from each block,
4. translate local feature names to original names with the mapping file,
5. combine the selected original features,
6. remove duplicate selections,
7. create the combined dataset, and
8. run the final task evaluation on that dataset.

Run DC with ten blocks:

```bash
python scripts/FeatureRank.py \
  --dataset-name arcene_data.csv \
  --task classification \
  --feature-percent 50 \
  --dc \
  --block-count 10
```

`--global` and `--dc` cannot be used together. If `--block-count` is omitted,
the current default is 10.

The orchestration is implemented in
[`src/DivideCombine.py`](src/DivideCombine.py). Splitting, mapping, and
combining reuse
[`scripts/FeatureBlockDatasetTools.py`](scripts/FeatureBlockDatasetTools.py);
the ranking algorithm is not duplicated for DC.



## Tasks

Classification uses a binary classifier for two labels. For multiclass data, the
existing one-vs-rest workflow runs one binary experiment per class and reports
the aggregate metrics.

Regression reports MSE, RMSE, MAE, R², and Pearson correlation where defined.

Clustering evaluates KMeans over the configured `k` range using silhouette
score. A fixed value can be requested with `--cluster-k`.

Examples:

```bash
# Classification
python scripts/FeatureRank.py \
  --dataset-name breast_cancer_data.csv \
  --task classification \
  --feature-percent 20

# Regression
python scripts/FeatureRank.py \
  --dataset-name air_data.csv \
  --task regression \
  --feature-percent 30

# Clustering
python scripts/FeatureRank.py \
  --dataset-name codon_usage_data.csv \
  --task clustering \
  --feature-percent 60
```

## Parameters

The main user-facing parameters are:

| Parameter | Description |
| --- | --- |
| `--dataset-name` | Input dataset name in `data/raw/` |
| `--task` | `classification`, `regression`, or `clustering` |
| `--feature-percent` | Percentage of features to select |
| `--global` | Run GLOBAL mode |
| `--dc` | Run Divide & Combine mode |
| `--block-count` | Number of feature blocks in DC mode |
| `--random-state` | Seed; use `none` for an unseeded run |
| `--target-column` | Target column name |
| `--id-column` | ID column name, or `none` |
| `--cluster-k` | Fixed cluster count for clustering |

For the complete, current list:

```bash
python scripts/FeatureRank.py --help
```

Model architecture, epochs, batch size, learning rate, early stopping, and
other defaults are kept in
[`src/Config.py`](src/Config.py), not repeated in every command.

## Outputs

GLOBAL classification output for `arcene_data.csv` and 50% selection is written
under:

```text
outputs/Classification/arcene_data/
├── first_layer_W_list.csv
├── top_50_max_abs_features.csv
├── ORG_*.csv / ORG_*.png
└── metrics/
    ├── ORG_test_metrics.json
    └── top_50_test_metrics.json
```

The selected feature CSV contains the original feature names and their scores.
Metric JSON/CSV files contain task results. Generated plots include prediction,
confusion-matrix, ROC, precision-recall, or clustering figures as applicable.

DC additionally creates:

```text
split_datasets/arcene/
├── arcene_block_*.csv
├── feature_block_mapping.csv
└── split_summary.csv

data/raw/
├── arcene_block_*_data.csv
├── arcene_block_*_label.csv
├── arcene_selected_features_combined_data.csv
└── arcene_selected_features_combined_label.csv
```

The mapping file links every local block feature to its original feature. The
combined dataset is used by the final evaluation and is also saved under
`split_datasets/`.

## Project Structure

```text
Feature_Ranking_Project/
├── FeatureRank/                      installable package and desktop GUI
│   ├── GUI.py                         Tkinter interface and Launch API
│   └── __init__.py                    public package exports
├── data/raw/                         input feature and label files
├── outputs/                          metrics and generated figures
├── split_datasets/                   DC blocks, mapping, and combined data
├── scripts/
│   ├── FeatureRank.py                main command
│   ├── RunAutoencoder.py             compatibility wrapper
│   ├── FeatureBlockDatasetTools.py   split/combine helpers
│   ├── RunBlockFeatureSelection.py   block comparison workflow
│   └── RunFeatureRankCV.py           leakage-free repeated CV
├── src/
│   ├── Classification.py             classification workflow
│   ├── Regression.py                 regression workflow
│   ├── Clustering.py                 clustering workflow
│   ├── DivideCombine.py              DC orchestration
│   ├── AutoencoderFeatureSelection.py ranking and selection
│   ├── DataLoader.py                 dataset loading
│   ├── Preprocessing.py              cleaning, splitting, and scaling
│   ├── Models.py                     Keras model definitions
│   ├── Workflow.py                   shared task entry point
│   └── Config.py                     project defaults
├── tests/
├── pyproject.toml
├── requirements.txt
└── Readme.md
```

All Python script names use PascalCase. `src/__init__.py` remains unchanged
because it is a special Python package filename.

## Requirements

The main technologies are:

- Python
- TensorFlow / Keras
- pandas
- NumPy
- scikit-learn
- Matplotlib

Exact pinned versions are listed in
[`requirements.txt`](requirements.txt). GPU dependencies are listed separately
in [`requirements-gpu.txt`](requirements-gpu.txt).

## Tests

Run the checks from the repository root:

```bash
python -m pytest -q
python -m pyflakes src scripts tests
python -m compileall -q src scripts tests
```

## Citation

Citation information will be added after publication. When reporting a result,
include the dataset source, task, GLOBAL or DC mode, feature percentage, block
count when applicable, random seed, and the generated metric files.
