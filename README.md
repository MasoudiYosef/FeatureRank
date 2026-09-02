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

FeatureRank requires Python 3.X. Check the Python version first:

```bash
python3 --version
```

Install the published package with:

```bash
pip install FeatureRank
```

<img width="996" height="622" alt="1" src="https://github.com/user-attachments/assets/9a740a4f-7bc1-4b00-a06f-ed4709ecad54" />



Open Python:

```bash
python3
```


<img width="988" height="123" alt="2" src="https://github.com/user-attachments/assets/acda6a62-c687-46a2-925d-74cad86f93b2" />



Then import FeatureRank at the Python prompt:

```python
>>> import FeatureRank
```

<img width="1104" height="668" alt="3" src="https://github.com/user-attachments/assets/db1dc4db-6080-45ac-9d08-d945948dcc2c" />


The import opens the desktop GUI automatically. Do not type `import
FeatureRank` directly in zsh; it is Python code and must be entered after
starting `python3`.


After installation, the package GUI can be opened with:

```bash
FeatureRank
```

`import FeatureRank` is Python code, not a shell command. On a desktop, the
following import opens the GUI automatically:


## GUI Workflow

The GUI is the primary end-user interface. Select a dataset from the dropdown
or choose a CSV/TXT file with **Browse…**, select a feature percentage, choose
`GL` or `DC`, and press **START**. `Block Count` is enabled only for `DC`.

When the GUI opens, the experiment form is ready for these choices:

<img width="763" height="715" alt="4" src="https://github.com/user-attachments/assets/e136bd77-fc05-4533-94df-65e5c5fd1f51" />


If the required dataset is not listed, press **Browse…** and select its data
file. For paired datasets, keep the matching label file in the same directory:


<img width="1492" height="724" alt="5" src="https://github.com/user-attachments/assets/5b5f4d67-f47c-423a-9efd-4da3f537219b" />



While the model runs, the progress bar and log show stages such as loading,
feature ranking, block processing, combining, training, and result creation.
When the run finishes, the summary lists the selected feature count, metric,
execution time, and output directory. **Open Results Folder** opens that
directory in Finder, Explorer, or the Linux file manager.


<img width="852" height="428" alt="6" src="https://github.com/user-attachments/assets/d2908867-882b-4efc-9ff2-49a59cbe27dc" />



The selected file can be a normal CSV containing a `target` column, or one of
the project's paired files (`*_data.csv` and `*_label.csv`).


The explicit command above and the same command without `--global` produce the
same mode.



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


### Result Inspection and Output Access

After an experiment is completed successfully, FeatureRank automatically reports the main execution details in the GUI, including the selected feature count, evaluation metric(s), execution time, and output directory.

The **Log / Output** panel provides a textual summary of the completed experiment and indicates the location of the generated result files. These files include the selected feature subsets, evaluation metrics, and task-specific visualizations.

For convenient inspection, the **Open Results Folder** button opens the corresponding experiment directory directly in the operating system's file manager. This allows users to examine the generated metric files, selected-feature lists, prediction outputs, and evaluation figures without manually navigating to the output directory.

All experiment outputs are stored systematically under the `outputs/` directory according to the selected task and dataset. This organization facilitates result verification, comparison between feature-selection configurations, and reproducibility of the conducted experiments.

<img width="1111" height="634" alt="7" src="https://github.com/user-attachments/assets/25be6055-ca79-4368-ba65-769fcc8205b9" />


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
