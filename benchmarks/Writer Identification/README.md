# MPHD: Writer Identification Benchmark

This directory provides the official training and evaluation scripts to reproduce the Writer Identification results reported in the MPHD paper (Section 4.3). 

The benchmark evaluates a patch-based, text-independent writer identification protocol using a ResNet-18 backbone, covering both **Closed-Set** and **Open-Set** scenarios.

---

## Prerequisites
1. Download the MPHD dataset from [Zenodo](https://doi.org/10.5281/zenodo.23557640) and extract it.
2. Install the required Python packages:
   ```bash
   pip install torch torchvision opencv-python scikit-learn matplotlib seaborn
   ```

---

## Protocol 1: Closed-Set Writer Identification

In this protocol, all 500 writers are known. The model is trained on patches extracted from the fixed-text images (`Text 1`) and evaluated on patches from the variable-text images (`Text 2`) to ensure text-independence.

### Step 1: Training and Evaluation
1. Open [`mphd_writer_id_benchmark.py`](mphd_writer_id_benchmark.py) and update the `DATA_ROOT` variable to point to your extracted MPHD dataset:
   ```python
   DATA_ROOT = "../MPHD"  # Change to your actual dataset path
   ```
2. Ensure `TEST_ONLY = False` for the first run.
3. Run the script:
   ```bash
   python mphd_writer_id_benchmark.py
   ```
   *The script will train the model for up to 50 epochs and save the best checkpoint as `writer_id_best.pt`. It will then evaluate the model on the `Text 2` test set and print the patch-level and document-level Top-K accuracies.*

*(Note: If you already have the trained `writer_id_best.pt` checkpoint, you can set `TEST_ONLY = True` to skip training and directly evaluate the model.)*

### Reference Results (Table 9)
For reproducibility verification, the expected metrics on the MPHD test set are:

| Level | Top-1 (%) | Top-3 (%) | Top-5 (%) |
| :--- | :--- | :--- | :--- |
| **Patch-level (128x128)** | 56.22 | 73.48 | 79.83 |
| **Document-level (avg prob)** | **95.00** | **98.40** | **99.20** |

---

## Protocol 2: Open-Set Writer Identification

In this protocol, the dataset is evaluated under a realistic scenario where the model must reject unseen writers. We use a rigorous **5-fold rotation protocol**: in each fold, the model is trained on 400 known writers and evaluated on 100 unseen impostors. The reported results are pooled across all 5 folds.

### Step 1: Run the 5-Fold Evaluation
1. Open [`mphd_writer_id_openset_5fold.py`](mphd_writer_id_openset_5fold.py) and update the `DATA_ROOT` variable:
   ```python
   DATA_ROOT = "../MPHD"  # Change to your actual dataset path
   ```
2. Run the script:
   ```bash
   python mphd_writer_id_openset_5fold.py
   ```
   *This script will train 5 different models (one per fold). It will output the Detection & Identification Rate (DIR) and False Alarm Rate (FAR) at three target levels (1%, 5%, 10%). The full per-fold statistics and pooled results will be saved automatically to `openset_5fold_results.json`.*

### Reference Results (Table 10)
For reproducibility verification, the expected pooled open-set results across the 5 folds are:

| Target FAR (%) | Actual FAR (%) | DIR (%) |
| :--- | :--- | :--- |
| **1%** | 0.60 | 28.50 |
| **5%** | 3.40 | 59.20 |
| **10%** | 9.40 | 75.55 |

*(Note: The substantial gap between the closed-set Top-1 accuracy of 95.00% and the open-set DIR of 28.50% at a strict 1% FAR underscores the difficulty of rejecting unseen writers in Persian handwriting.)*
