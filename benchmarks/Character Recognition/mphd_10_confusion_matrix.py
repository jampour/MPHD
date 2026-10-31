"""
mphd_10_confusion_matrix.py

Top-N substitution confusion matrix for the MPHD 62-class Persian character
recognition model. Writer-level split via shared .ln files.

Run:
    python mphd_10_confusion_matrix.py
"""

import os
import re
import glob
import random
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ==================== CONFIG ====================
DATA_ROOT       = "../../MPHD2"
TRAIN_LN        = "../train.ln"
VAL_LN          = "../val.ln"
TEST_LN         = "../test.ln"

NUM_CLASSES     = 62
IMG_HEIGHT      = 70
IMG_WIDTH       = 45

MODEL_NAME      = "DenseNet121"     # "DenseNet121" or "ResNet18"
CHECKPOINT_PATH = MODEL_NAME + "/mphd_char62_best.pt"

BATCH_SIZE      = 64
NUM_WORKERS     = 4
RANDOM_SEED     = 42

TOP_N           = 10
OUT_PATH        = "mphd_confusion_matrix_" + MODEL_NAME + ".png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== CHAR MAP ====================
CHAR_MAP = [""] * NUM_CLASSES
CHAR_MAP[0]  = "آ"
CHAR_MAP[1]  = "ا"
CHAR_MAP[2]  = "بـ"
CHAR_MAP[3]  = "ب"
CHAR_MAP[4]  = "پـ"
CHAR_MAP[5]  = "پ"
CHAR_MAP[6]  = "تـ"
CHAR_MAP[7]  = "ت"
CHAR_MAP[8]  = "ثـ"
CHAR_MAP[9]  = "ث"
CHAR_MAP[10] = "جـ"
CHAR_MAP[11] = "ج"
CHAR_MAP[12] = "چـ"
CHAR_MAP[13] = "چ"
CHAR_MAP[14] = "حـ"
CHAR_MAP[15] = "ح"
CHAR_MAP[16] = "خـ"
CHAR_MAP[17] = "خ"
CHAR_MAP[18] = "د"
CHAR_MAP[19] = "ذ"
CHAR_MAP[20] = "ر"
CHAR_MAP[21] = "ز"
CHAR_MAP[22] = "ژ"
CHAR_MAP[23] = "سـ"
CHAR_MAP[24] = "س"
CHAR_MAP[25] = "شـ"
CHAR_MAP[26] = "ش"
CHAR_MAP[27] = "صـ"
CHAR_MAP[28] = "ص"
CHAR_MAP[29] = "ضـ"
CHAR_MAP[30] = "ض"
CHAR_MAP[31] = "ط"
CHAR_MAP[32] = "ظ"
CHAR_MAP[33] = "عـ"
CHAR_MAP[34] = "ـعـ"
CHAR_MAP[35] = "ع"
CHAR_MAP[36] = "غـ"
CHAR_MAP[37] = "ـغـ"
CHAR_MAP[38] = "غ"
CHAR_MAP[39] = "فـ"
CHAR_MAP[40] = "ف"
CHAR_MAP[41] = "قـ"
CHAR_MAP[42] = "ق"
CHAR_MAP[43] = "کـ"
CHAR_MAP[44] = "ک"
CHAR_MAP[45] = "گـ"
CHAR_MAP[46] = "گ"
CHAR_MAP[47] = "لـ"
CHAR_MAP[48] = "ل"
CHAR_MAP[49] = "مـ"
CHAR_MAP[50] = "م"
CHAR_MAP[51] = "نـ"
CHAR_MAP[52] = "ن"
CHAR_MAP[53] = "و"
CHAR_MAP[54] = "هـ"
CHAR_MAP[55] = "ـهـ"
CHAR_MAP[56] = "ـهـ*"   # handwritten variant (no standard Unicode counterpart)
CHAR_MAP[57] = "ـه"
CHAR_MAP[58] = "ه"
CHAR_MAP[59] = "یـ"
CHAR_MAP[60] = "ئـ"
CHAR_MAP[61] = "ی"


def char_label(class_idx: int) -> str:
    if 0 <= class_idx < len(CHAR_MAP) and CHAR_MAP[class_idx]:
        return CHAR_MAP[class_idx]
    return str(class_idx + 1)


# ==================== WRITER-LEVEL SPLIT VIA .ln ====================
CHAR_FILENAME_RE = re.compile(r"_Char_(\d+)", re.IGNORECASE)


def load_writer_ids_from_ln(ln_path):
    writers = set()
    with open(ln_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wid = os.path.basename(line).split("_")[0]
            writers.add(wid)
    return writers


def index_dataset(data_root, num_classes):
    """Return list of (filepath, label, writer_id)."""
    all_files = glob.glob(os.path.join(data_root, "*", "Chars", "*"))
    samples = []
    for fpath in all_files:
        if not fpath.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            continue
        fname = os.path.basename(fpath)
        match = CHAR_FILENAME_RE.search(fname)
        if match is None:
            continue
        label = int(match.group(1)) - 1
        if not (0 <= label < num_classes):
            continue
        wid = fname.split("_")[0]
        samples.append((fpath, label, wid))
    if not samples:
        raise RuntimeError(f"No valid samples found under '{data_root}'.")
    return samples


def split_by_writer_ln(samples, val_ln, test_ln):
    """Split char samples using writer IDs from .ln files. Return (val_set, test_set)."""
    val_w  = load_writer_ids_from_ln(val_ln)
    test_w = load_writer_ids_from_ln(test_ln)
    assert not (val_w & test_w), "writer overlap val/test"

    val_set, test_set = [], []
    for fpath, label, wid in samples:
        if wid in val_w:
            val_set.append((fpath, label))
        elif wid in test_w:
            test_set.append((fpath, label))
    return val_set, test_set


# ==================== DATASET ====================
class MPHDCharacterDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        image = Image.open(fpath).convert("L")
        return self.transform(image), label


def build_eval_transform():
    return T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])


# ==================== MODEL ====================
def build_model(model_name, num_classes, in_channels=1):
    model_name = model_name.lower()
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(in_channels, old_conv.out_channels,
                                kernel_size=old_conv.kernel_size,
                                stride=old_conv.stride,
                                padding=old_conv.padding,
                                bias=(old_conv.bias is not None))
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        old_conv = model.features.conv0
        model.features.conv0 = nn.Conv2d(in_channels, old_conv.out_channels,
                                         kernel_size=old_conv.kernel_size,
                                         stride=old_conv.stride,
                                         padding=old_conv.padding,
                                         bias=(old_conv.bias is not None))
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model_name: '{model_name}'.")
    return model


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_preds)


# ==================== TOP-N SUBSTITUTION MATRIX ====================
def top_n_substitution_matrix(y_true, y_pred, top_n):
    subst_counter = Counter()
    for p, g in zip(y_pred, y_true):
        if p != g:
            subst_counter[(p, g)] += 1

    total_subst = sum(subst_counter.values()) or 1
    most_common = subst_counter.most_common(top_n * 3)

    pred_classes = [p for (p, g), _ in most_common]
    gt_classes   = [g for (p, g), _ in most_common]
    unique_pred = list(dict.fromkeys(pred_classes))[:top_n]
    unique_gt   = list(dict.fromkeys(gt_classes))[:top_n]

    if len(unique_pred) < top_n or len(unique_gt) < top_n:
        all_cls = sorted(set(c for pair in subst_counter for c in pair))
        unique_pred = (unique_pred + [c for c in all_cls if c not in unique_pred])[:top_n]
        unique_gt   = (unique_gt   + [c for c in all_cls if c not in unique_gt])[:top_n]

    matrix = np.zeros((len(unique_pred), len(unique_gt)), dtype=float)
    for i, p in enumerate(unique_pred):
        for j, g in enumerate(unique_gt):
            matrix[i, j] = subst_counter.get((p, g), 0) / total_subst * 100

    return matrix, unique_pred, unique_gt, subst_counter


def plot_matrix(ax, matrix, unique_pred, unique_gt, title):
    x_labels = [f"«{char_label(c)}»" for c in unique_gt]
    y_labels = [f"«{char_label(c)}»" for c in unique_pred]
    sns.heatmap(
        matrix, annot=True, fmt=".1f", cmap="Blues",
        linewidths=1.5, linecolor="white", square=True,
        cbar_kws={"label": "Percentage of substitutions (%)", "shrink": 0.8},
        xticklabels=x_labels, yticklabels=y_labels,
        annot_kws={"size": 11, "weight": "bold", "color": "#1e293b"},
        ax=ax,
    )
    ax.set_title(title, fontsize=13, pad=14, fontweight="bold", color="#0f172a")
    ax.set_xlabel("Ground Truth", fontsize=11, fontweight="bold", labelpad=10, color="#334155")
    ax.set_ylabel("Prediction", fontsize=11, fontweight="bold", labelpad=10, color="#334155")
    ax.tick_params(axis="both", labelsize=11)


# ==================== MAIN ====================
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        print("Checkpoint not found!")
        return

    model = build_model(MODEL_NAME, NUM_CLASSES)
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(DEVICE).eval()
    print("Model loaded.")

    samples = index_dataset(DATA_ROOT, NUM_CLASSES)
    val_samples, test_samples = split_by_writer_ln(samples, VAL_LN, TEST_LN)
    print(f"Validation samples: {len(val_samples)} | Test samples: {len(test_samples)}")

    eval_transform = build_eval_transform()
    val_dataset  = MPHDCharacterDataset(val_samples,  eval_transform)
    test_dataset = MPHDCharacterDataset(test_samples, eval_transform)

    val_loader  = DataLoader(val_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    results = {}
    for split_name, loader in [("Test", test_loader), ("Validation", val_loader)]:
        print(f"Running inference on {split_name} set...")
        y_true, y_pred = predict_all(model, loader, DEVICE)
        matrix, unique_pred, unique_gt, subst_counter = top_n_substitution_matrix(y_true, y_pred, TOP_N)
        print(f"  Total substitutions: {sum(subst_counter.values())}")
        for (p, g), cnt in subst_counter.most_common(10):
            print(f"    '{char_label(p)}' -> '{char_label(g)}' : {cnt}")
        results[split_name] = (matrix, unique_pred, unique_gt)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), dpi=300)

    for ax, split_name in zip(axes, ["Test", "Validation"]):
        matrix, unique_pred, unique_gt = results[split_name]
        plot_matrix(ax, matrix, unique_pred, unique_gt,
                    f"MPHD Character Recognition Confusion (Top-{TOP_N})\n{MODEL_NAME} — {split_name} data")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()