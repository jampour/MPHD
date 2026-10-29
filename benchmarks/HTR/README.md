# MPHD: Handwriting Text Recognition (HTR) Benchmark

This directory contains the dataset preparation script and the official training/evaluation instructions to reproduce the Handwriting Text Recognition (HTR) benchmark results reported in the MPHD paper (Section 4.1, Table 7).

The benchmark evaluates the **Vision Transformer (HTR-VT)** architecture, which represents the state-of-the-art performance on this dataset.

---

## Part 1: Dataset Preparation & Writer-Level Splits

The provided script [`mphd_htr_prepare_splits.py`](mphd_htr_prepare_splits.py) reproduces the official writer-level train/validation/test splits (70:15:15) with `seed=42`.

### Step 1.1: Run the Preparation Script
1. Open `mphd_htr_prepare_splits.py` and ensure the `CONFIG` points to your extracted MPHD dataset:
   ```python
   DATASET_ROOT = r"../MPHD"     # Path to the root MPHD directory
   OUTPUT_ROOT  = r"./"          # Path where Lines/ and split files will be saved
   ```
2. Run the script:
   ```bash
   python mphd_htr_prepare_splits.py
   ```

### Step 1.2: Organize the Generated Files
The script will generate the `Lines/` folder and the `.ln` split files in your current directory. To make them compatible with the training scripts, move them into the expected directory structure:

```bash
mkdir -p data/persian/
mv Lines/ data/persian/lines/
mv train.ln val.ln test.ln test-t1.ln test-t2.ln data/persian/
```
*The generated splits match exactly Table 7 in the paper: Train = 3,515, Val = 767, Test = 739 (T1 = 403, T2 = 336).*

---

## Part 2: Training and Evaluation

### Prerequisites
1. Clone the official [HTR-VT repository](https://github.com/Intellindust-AI-Lab/HTR-VT) and navigate into it:
   ```bash
   git clone https://github.com/Intellindust-AI-Lab/HTR-VT
   cd HTR-VT
   ```

2. Copy the provided scripts from this directory into the cloned repository exactly as follows:
   * `train_htrvt_base_repro_early.py` ➔ Copy to the **root** directory
   * `test_persian.py` ➔ Copy to the **root** directory
   * `valid.py` ➔ Replace the existing file in the **root** directory
   * `utils/option.py` ➔ Replace the existing file in the **`utils/`** directory
   * `data/dataset.py` ➔ Replace the existing file in the **`data/`** directory

3. Install the required Python packages:
   ```bash
   pip install torch torchvision editdistance tensorboard
   ```

---

### Step 2: Training the Model

#### Why do we use `train_htrvt_base_repro_early.py`?
The original HTR-VT implementation exhibits non-determinism across different hardware and GPU architectures. To ensure 100% reproducibility and fair comparison, our provided script adds strict seeding, deterministic GPU operations, and Early Stopping (patience=10) to prevent overfitting and save computational resources.

#### Training HTR-VT
We train the model using three independent random seeds (`7`, `42`, and `123`). 

**Note on the `PERSIAN` argument:** The provided `option.py` includes a custom subparser for the Persian dataset. The keyword `PERSIAN` at the end of the command tells the script to load the MPHD data paths and class count.

**Seed 123:**
```bash
python train_htrvt_base_repro_early.py --exp-name mphd_htrvt_base_early_seed123 --seed 123 \
  --max-lr 4e-4 --train-bs 16 --val-bs 8 --weight-decay 0.05 \
  --mask-ratio 0.25 --attn-mask-ratio 0.0 --max-span-length 4 \
  --img-size 512 64 --proj 8 --total-iter 100001 --eval-iter 400 \
  --print-iter 100 --warm-up-iter 800 PERSIAN --nb-cls 59
```
*(Repeat the above command by changing `--seed 123` to `42` or `7` and updating the `--exp-name` accordingly).*

---

### Step 3: Evaluation on Test Sets
After training, the best validation checkpoints will be saved in the `./output/[EXP_NAME]/` directory. To evaluate the models on the full test set (`test.ln`), as well as the Fixed (`test-t1.ln`) and Variable (`test-t2.ln`) subsets, run:

```bash
# For HTR-VT (replace with your experiment name)
python test_persian.py mphd_htrvt_base_early_seed123
```
*The evaluation script will output the Test CER and WER for all three splits and save per-line predictions in a `.npz` file for statistical bootstrapping.*

---

## Reference Results
For reproducibility verification, the expected Validation and Test CERs for HTR-VT are approximately:

| Seed | Validation CER (Checkpoint Selection) | Test CER (Full Test Set) |
| :--- | :--- | :--- |
| **123** | **8.42%** | **9.04%** |
| **42** | **9.92%** | **10.45%** |
| **7** | **15.45%** | **16.10%** |

The final aggregated test set results (mean ± std over the 3 seeds) are reported in **Table 7** of the MPHD paper.
