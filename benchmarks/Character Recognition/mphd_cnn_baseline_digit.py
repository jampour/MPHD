"""
mphd_cnn_digit10.py

Benchmark: 10-class Persian digit recognition on MPHD.
Split: writer-level 70:15:15 via shared .ln files (train.ln / val.ln / test.ln).
"""

import os
import re
import random
import glob
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# =========================================================================
# CONFIG
# =========================================================================
DATA_ROOT = "../../MPHD2"
TRAIN_LN  = "../train.ln"
VAL_LN    = "../val.ln"
TEST_LN   = "../test.ln"

NUM_CLASSES = 10
IMG_HEIGHT  = 70
IMG_WIDTH   = 45

MODEL_NAME  = "resnet18"    # densenet121   resnet18
PRETRAINED  = True

BATCH_SIZE    = 64
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
EARLY_STOPPING_PATIENCE = 10

RANDOM_SEED = 42
NUM_WORKERS = 4

CHECKPOINT_PATH       = "./mphd_digit10_best.pt"
CONFUSION_MATRIX_PATH = "./mphd_digit10_cm.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# CHAR MAP (10 classes)
# =========================================================================
CHAR_MAP = {0:"۰", 1:"۱", 2:"۲", 3:"۳", 4:"۴",
            5:"۵", 6:"۶", 7:"۷", 8:"۸", 9:"۹"}

# =========================================================================
# REPRODUCIBILITY
# =========================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================================================================
# DATA INDEXING + WRITER-LEVEL SPLIT
# =========================================================================
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

def index_dataset(data_root, num_classes, label_offset):
    """
    label_offset = 63 for digit recognition (Char_63 -> class 0, ..., Char_72 -> class 9)
    Walk dataset, return list of (filepath, label, writer_id).
    """
    all_files = glob.glob(os.path.join(data_root, "*", "Chars", "*"))
    samples = []
    for fpath in all_files:
        if not fpath.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            continue
        fname = os.path.basename(fpath)
        match = CHAR_FILENAME_RE.search(fname)
        if match is None:
            continue
        cid = int(match.group(1))
        label = cid - label_offset
        if label < 0 or label >= num_classes:
            continue
        wid = fname.split("_")[0]
        samples.append((fpath, label, wid))
    if not samples:
        raise RuntimeError(f"No valid samples found under '{data_root}'.")
    print(f"[index_dataset] Found {len(samples)} images across {num_classes} classes.")
    return samples

def split_by_writer_ln(samples, train_ln, val_ln, test_ln):
    train_w = load_writer_ids_from_ln(train_ln)
    val_w   = load_writer_ids_from_ln(val_ln)
    test_w  = load_writer_ids_from_ln(test_ln)

    assert not (train_w & val_w),  "writer overlap train/val"
    assert not (train_w & test_w), "writer overlap train/test"
    assert not (val_w & test_w),   "writer overlap val/test"

    train_set, val_set, test_set = [], [], []
    for fpath, label, wid in samples:
        if wid in train_w:
            train_set.append((fpath, label))
        elif wid in val_w:
            val_set.append((fpath, label))
        elif wid in test_w:
            test_set.append((fpath, label))

    print(f"[split_by_writer_ln] Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    print(f"[split_by_writer_ln] Writers -> Train: {len(train_w)} | Val: {len(val_w)} | Test: {len(test_w)}")
    return train_set, val_set, test_set

# =========================================================================
# TRANSFORMS
# =========================================================================
def build_transforms(img_height, img_width, train):
    mean, std = [0.5], [0.5]
    if train:
        return T.Compose([
            T.Grayscale(num_output_channels=1),
            T.RandomAffine(degrees=7, translate=(0.06, 0.06),
                           scale=(0.92, 1.08), fill=255),
            T.Resize((img_height, img_width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((img_height, img_width)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

# =========================================================================
# DATASET
# =========================================================================
class MPHDCharacterDataset(Dataset):
    def __init__(self, samples, transform, samples_multiplier=1):
        self.samples = samples
        self.transform = transform
        self.samples_multiplier = max(1, samples_multiplier)
        self.base_len = len(samples)

    def __len__(self):
        return self.base_len * self.samples_multiplier

    def __getitem__(self, idx):
        real_idx = idx % self.base_len
        fpath, label = self.samples[real_idx]
        image = Image.open(fpath).convert("L")
        return self.transform(image), label

def compute_train_multiplier(num_base_train_samples, target_total):
    if num_base_train_samples <= 0:
        return 1
    return max(1, round(target_total / num_base_train_samples))

# =========================================================================
# MODEL
# =========================================================================
def build_model(model_name, num_classes, pretrained, in_channels=1):
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        old_conv = model.conv1
        new_conv = nn.Conv2d(in_channels, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding,
                             bias=(old_conv.bias is not None))
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        model.conv1 = new_conv
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        old_conv = model.features.conv0
        new_conv = nn.Conv2d(in_channels, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding,
                             bias=(old_conv.bias is not None))
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        model.features.conv0 = new_conv
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model_name: '{model_name}'.")

    return model

# =========================================================================
# TRAIN / EVAL
# =========================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def train_model(model, train_loader, val_loader, num_epochs, lr, weight_decay,
                patience, device, checkpoint_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        print(f"Epoch [{epoch:03d}/{num_epochs}] "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f}")
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save(best_state, checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    return model

# =========================================================================
# TEST METRICS + CONFUSION MATRIX
# =========================================================================
@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        preds = model(images).argmax(1).cpu().numpy()
        preds_all.extend(preds.tolist())
        labels_all.extend(labels.numpy().tolist())
    return np.array(labels_all), np.array(preds_all)

def report_test_metrics(y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print("\n===== TEST SET RESULTS =====")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Precision macro : {prec:.4f}")
    print(f"Recall macro    : {rec:.4f}")
    print(f"Macro F1        : {f1:.4f}")
    print("=============================\n")
    return {"accuracy": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1}

def plot_confusion_matrix(y_true, y_pred, num_classes, char_map, save_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    labels = [char_map[i] for i in range(num_classes)]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                xticklabels=labels, yticklabels=labels,
                cbar=True, square=True)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(title)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[plot_confusion_matrix] Saved to '{save_path}'")

# =========================================================================
# MAIN
# =========================================================================
def main():
    set_seed(RANDOM_SEED)
    print(f"Using device: {DEVICE}")

    samples = index_dataset(DATA_ROOT, NUM_CLASSES, label_offset=63)
    train_samples, val_samples, test_samples = split_by_writer_ln(
        samples, TRAIN_LN, VAL_LN, TEST_LN
    )

    train_tf = build_transforms(IMG_HEIGHT, IMG_WIDTH, train=True)
    eval_tf  = build_transforms(IMG_HEIGHT, IMG_WIDTH, train=False)

    train_multiplier = compute_train_multiplier(len(train_samples), 3 * len(train_samples))
    print(f"[main] Train virtual multiplier: x{train_multiplier}")

    train_ds = MPHDCharacterDataset(train_samples, train_tf, samples_multiplier=train_multiplier)
    val_ds   = MPHDCharacterDataset(val_samples,   eval_tf,  samples_multiplier=1)
    test_ds  = MPHDCharacterDataset(test_samples,  eval_tf,  samples_multiplier=1)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model(MODEL_NAME, NUM_CLASSES, PRETRAINED, in_channels=1).to(DEVICE)

    model = train_model(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        patience=EARLY_STOPPING_PATIENCE, device=DEVICE,
        checkpoint_path=CHECKPOINT_PATH,
    )

    y_true, y_pred = predict_all(model, test_loader, DEVICE)
    report_test_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, NUM_CLASSES, CHAR_MAP,
                          CONFUSION_MATRIX_PATH,
                          "MPHD — Confusion Matrix (10 Persian Digit Classes)")
    print("[main] Done.")

if __name__ == "__main__":
    main()