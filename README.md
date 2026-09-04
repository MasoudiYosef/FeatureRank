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

<img width="1677" height="215" alt="1" src="https://github.com/user-attachments/assets/a289ac39-aca3-4dcf-8cd7-903c78f016e3" />



Open Python:

```bash
python3
```


<img width="1677" height="137" alt="2" src="https://github.com/user-attachments/assets/4d11c628-59ce-45b7-aa42-5b6756c53f02" />


Then import FeatureRank at the Python prompt:

```python
>>> import FeatureRank
```


<img width="1664" height="946" alt="3" src="https://github.com/user-attachments/assets/9df78d81-b44f-47d3-803c-94289b5ce361" />



The import opens the desktop GUI automatically. Do not type `import
F<img width="1673" height="1041" alt="4" src="https://github.com/user-attachments/assets/d6be4ac4-c3f5-4ba7-80f4-641e62919431" />
eatureRank` directly in zsh; it is Python code and must be entered after
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

1. Select dataset
2. Select task
3. Select feature selection %
4. Select GL or DC
5. If DC → select Block Count
6. Press START
7. Inspect results


When the GUI opens, the experiment form is ready for these choices:


<img width="1673" height="1041" alt="4" src="https://github.com/user-attachments/assets/ce7115fe-32b8-40b3-b792-03ea07ebbd0f" />

If the required dataset is not listed, press **Browse…** and select its data
file. For paired datasets, keep the matching label file in the same directory:


<img width="1664" height="1033" alt="5" src="https://github.com/user-attachments/assets/f5114f13-3916-4ecb-9dfc-8f6e18ebb950" />



While the model runs, the progress bar and log show stages such as loading,
feature ranking, block processing, combining, training, and result creation.
When the run finishes, the summary lists the selected feature count, metric,
execution time, and output directory. **Open Results Folder** opens that
directory in Finder, Explorer, or the Linux file manager.


<img width="1664" height="545" alt="6" src="https://github.com/user-attachments/assets/d6c30c79-dace-4e1f-b601-4dcfe7fce7dd" />



The selected file can be a normal CSV containing a `target` column, or one of
the project's paired files (`*_data.csv` and `*_label.csv`).


The explicit command above and the same command without `--global` produce the
same mode.



### Result Inspection and Output Access

After an experiment is completed successfully, FeatureRank automatically reports the main execution details in the GUI, including the selected feature count, evaluation metric(s), execution time, and output directory.

The **Log / Output** panel provides a textual summary of the completed experiment and indicates the location of the generated result files. These files include the selected feature subsets, evaluation metrics, and task-specific visualizations.

For convenient inspection, the **Open Results Folder** button opens the corresponding experiment directory directly in the operating system's file manager. This allows users to examine the generated metric files, selected-feature lists, prediction outputs, and evaluation figures without manually navigating to the output directory.

All experiment outputs are stored systematically under the `outputs/` directory according to the selected task and dataset. This organization facilitates result verification, comparison between feature-selection configurations, and reproducibility of the conducted experiments.


<img width="1662" height="897" alt="7" src="https://github.com/user-attachments/assets/1949614a-791d-4e70-833b-61567f6ac74a" />


## Citation

If you use FeatureRank in academic research, please cite the associated publication. Citation details will be added upon publication..
