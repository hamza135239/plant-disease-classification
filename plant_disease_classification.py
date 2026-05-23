# -*- coding: utf-8 -*-
"""
=============================================================================
  PLANT DISEASE CLASSIFICATION - ML ASSIGNMENT
  Dataset   : New Plant Diseases Dataset (3 GB, 87,000+ leaf images)
  Source    : https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
  Task      : Multi-class Image Classification (38 disease categories)
  Models    : Logistic Regression, Random Forest, Linear SVM, KNN
  Author    : [Your Name] | Roll No: [Your Roll Number]
=============================================================================

DATASET DOWNLOAD INSTRUCTIONS:
  1. Go to: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
  2. Log in to Kaggle (free) -> click "Download" (~3 GB zip)
  3. Unzip the downloaded file
  4. You will get a folder called "New Plant Diseases Dataset(Augmented)"
     inside it there are two folders: "train" and "valid"
     Each contains 38 sub-folders (one per disease class)
  5. Place the unzipped folder in the same directory as this script
     OR update DATASET_PATH below

  Dataset Size  : ~3 GB (87,000+ color leaf images)
  Classes       : 38 plant disease categories
  Image Format  : JPG/PNG, 256x256 pixels

INSTALLATION:
  pip install pandas numpy scikit-learn matplotlib seaborn Pillow tqdm
=============================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================
import sys
import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
from collections import Counter

# ============================================================================
# CONFIGURATION
# ============================================================================

# Update this path to where you unzipped the dataset
DATASET_PATH = "New Plant Diseases Dataset(Augmented)"  # folder name after unzipping

IMG_SIZE     = 64        # Resize images to 64x64 (memory safe)
MAX_PER_CLASS = 150      # Max images per class (keeps RAM usage safe)
PCA_COMPONENTS = 150     # PCA dimensions after flattening
RANDOM_STATE = 42
TEST_SIZE    = 0.20
OUTPUT_DIR   = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Aesthetics ───────────────────────────────────────────────────────────────
DARK_BG  = "#0D1117"
CARD_BG  = "#161B22"
TEXT_CLR = "#C9D1D9"
ACCENT   = "#58A6FF"

plt.rcParams.update({
    "figure.facecolor" : DARK_BG,
    "axes.facecolor"   : CARD_BG,
    "axes.edgecolor"   : "#30363D",
    "axes.labelcolor"  : TEXT_CLR,
    "xtick.color"      : TEXT_CLR,
    "ytick.color"      : TEXT_CLR,
    "text.color"       : TEXT_CLR,
    "grid.color"       : "#21262D",
    "grid.linestyle"   : "--",
    "grid.alpha"       : 0.5,
    "legend.facecolor" : CARD_BG,
    "legend.edgecolor" : "#30363D",
    "font.family"      : "DejaVu Sans",
    "font.size"        : 10,
})

print("=" * 70)
print("   PLANT DISEASE CLASSIFICATION -- ML ASSIGNMENT")
print("   Dataset: New Plant Diseases Dataset (3 GB, 87K+ Images)")
print("=" * 70)

# ============================================================================
# STEP 1 -- COLLECT THE DATASET
# ============================================================================
print("\n[STEP 1]  Dataset Collection")
print("-" * 50)

def find_dataset():
    """Try to find the dataset folder automatically."""
    possible_names = [
        "New Plant Diseases Dataset(Augmented)",
        "New Plant Diseases Dataset",
        "plant_diseases",
        "dataset",
    ]
    for name in possible_names:
        if os.path.isdir(name):
            return name
    # Search for 'train' folder with 38 subdirectories
    for folder in os.listdir("."):
        if os.path.isdir(folder):
            train_path = os.path.join(folder, "train")
            if os.path.isdir(train_path):
                classes = [d for d in os.listdir(train_path)
                           if os.path.isdir(os.path.join(train_path, d))]
                if len(classes) >= 10:
                    return folder
    return None

found = find_dataset()
if found:
    DATASET_PATH = found
    print(f"    [OK] Dataset found at: {DATASET_PATH}")
else:
    print(f"\n    [!] Dataset folder not found!")
    print(f"""
DOWNLOAD STEPS:
  1. Go to: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
  2. Login -> Download (~3 GB zip file)
  3. Unzip it -> you get a folder like:
       "New Plant Diseases Dataset(Augmented)"
  4. Place that folder in the same directory as this script:
       {os.path.abspath(".")}
  5. Re-run this script
""")
    sys.exit(1)

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VALID_PATH = os.path.join(DATASET_PATH, "valid")

if not os.path.isdir(TRAIN_PATH):
    print(f"    [!] 'train' folder not found inside {DATASET_PATH}")
    sys.exit(1)

all_classes = sorted([d for d in os.listdir(TRAIN_PATH)
                      if os.path.isdir(os.path.join(TRAIN_PATH, d))])
print(f"    Total disease classes : {len(all_classes)}")
print(f"    Dataset path          : {os.path.abspath(DATASET_PATH)}")
print(f"    Dataset size          : ~3 GB (87,000+ images)")
print(f"    Image size used       : {IMG_SIZE}x{IMG_SIZE} pixels")
print(f"    Max images per class  : {MAX_PER_CLASS}")

# Count total images
total_train = sum(
    len(os.listdir(os.path.join(TRAIN_PATH, c))) for c in all_classes
)
print(f"    Total training images : {total_train:,}")

# ============================================================================
# STEP 2 -- PREPROCESS
# ============================================================================
print("\n[STEP 2]  Preprocessing Images ...")
print("-" * 50)
print(f"    Resizing all images to {IMG_SIZE}x{IMG_SIZE} ...")
print(f"    Converting to grayscale feature vectors ...")
print(f"    Applying PCA ({PCA_COMPONENTS} components) ...")

def load_images(train_path, classes, img_size, max_per_class):
    """Load, resize and flatten images into feature matrix."""
    X, y = [], []
    for cls in tqdm(classes, desc="    Loading classes"):
        cls_path = os.path.join(train_path, cls)
        images = [f for f in os.listdir(cls_path)
                  if f.lower().endswith(('.jpg','.jpeg','.png'))]
        images = images[:max_per_class]
        for img_file in images:
            img_path = os.path.join(cls_path, img_file)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((img_size, img_size), Image.LANCZOS)
                arr = np.array(img, dtype=np.float32) / 255.0
                X.append(arr.flatten())
                y.append(cls)
            except Exception:
                continue
    return np.array(X), np.array(y)

t0 = time.time()
X_raw, y_raw = load_images(TRAIN_PATH, all_classes, IMG_SIZE, MAX_PER_CLASS)
print(f"\n    Images loaded    : {len(X_raw):,}")
print(f"    Feature size     : {X_raw.shape[1]:,} (raw pixels per image)")
print(f"    Load time        : {time.time()-t0:.1f}s")

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
num_classes = len(le.classes_)

# ============================================================================
# STEP 3 -- READ THE DATASET (EDA Graphs)
# ============================================================================
print("\n[STEP 3]  Exploratory Data Analysis ...")
print("-" * 50)

# ------------------------------------------------------------------
# Graph 1: Class Distribution (images per disease)
# ------------------------------------------------------------------
class_counts = Counter(y_raw)
sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
cls_names = [c[0].replace("___", "\n").replace("_", " ") for c in sorted_classes]
cls_vals  = [c[1] for c in sorted_classes]

fig, ax = plt.subplots(figsize=(18, 7))
fig.patch.set_facecolor(DARK_BG)
cmap = plt.cm.get_cmap("plasma", len(cls_names))
colors = [cmap(i) for i in range(len(cls_names))]
bars = ax.bar(range(len(cls_names)), cls_vals, color=colors, edgecolor=DARK_BG)
ax.set_xticks(range(len(cls_names)))
ax.set_xticklabels(cls_names, rotation=90, ha="center", fontsize=6)
ax.set_title("Plant Disease Dataset -- Images per Class (38 Categories)",
             fontsize=14, fontweight="bold", color=TEXT_CLR, pad=15)
ax.set_ylabel("Number of Images")
ax.set_xlabel("Disease Category")
ax.grid(axis="y")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_class_distribution.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    [OK] Saved: 01_class_distribution.png")

# ------------------------------------------------------------------
# Graph 2: Sample Images Grid (3x6 = 18 random class samples)
# ------------------------------------------------------------------
fig, axes = plt.subplots(3, 6, figsize=(18, 9))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Sample Leaf Images from Dataset (18 Disease Categories)",
             fontsize=14, fontweight="bold", color=TEXT_CLR)

sample_classes = all_classes[:18]
for ax, cls in zip(axes.flatten(), sample_classes):
    cls_path = os.path.join(TRAIN_PATH, cls)
    imgs = os.listdir(cls_path)
    if imgs:
        img_path = os.path.join(cls_path, imgs[0])
        try:
            img = Image.open(img_path).convert("RGB").resize((128, 128))
            ax.imshow(np.array(img))
        except Exception:
            ax.set_facecolor(CARD_BG)
    ax.set_title(cls.replace("___", "\n").replace("_", " "),
                 fontsize=5, color=TEXT_CLR, pad=2)
    ax.axis("off")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_sample_images.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    [OK] Saved: 02_sample_images.png")

# ------------------------------------------------------------------
# Graph 3: Pixel Intensity Distribution (Fake vs Real-style: healthy vs diseased)
# ------------------------------------------------------------------
healthy_idx = [i for i, lbl in enumerate(y_raw) if "healthy" in lbl.lower()]
diseased_idx = [i for i, lbl in enumerate(y_raw) if "healthy" not in lbl.lower()]

fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor(DARK_BG)

if healthy_idx:
    healthy_mean = X_raw[healthy_idx].mean(axis=0)
    ax.hist(healthy_mean, bins=60, alpha=0.65, color="#2ECC71",
            label=f"Healthy Leaves (n={len(healthy_idx):,})", density=True)
if diseased_idx:
    diseased_mean = X_raw[diseased_idx].mean(axis=0)
    ax.hist(diseased_mean, bins=60, alpha=0.65, color="#E74C3C",
            label=f"Diseased Leaves (n={len(diseased_idx):,})", density=True)

ax.set_title("Pixel Intensity Distribution -- Healthy vs Diseased Leaves",
             fontsize=13, fontweight="bold", color=TEXT_CLR)
ax.set_xlabel("Normalized Pixel Value"); ax.set_ylabel("Density")
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_pixel_intensity.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    [OK] Saved: 03_pixel_intensity.png")

# ============================================================================
# STEP 4 -- IMPLEMENT CLASSIFICATION MODELS
# ============================================================================
print("\n[STEP 4]  Training Classification Models ...")
print("-" * 50)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

# Standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# PCA for dimensionality reduction
print(f"    Applying PCA: {X_train_scaled.shape[1]:,} -> {PCA_COMPONENTS} features ...")
t_pca = time.time()
pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)
explained_var = pca.explained_variance_ratio_.sum() * 100
print(f"    PCA done in {time.time()-t_pca:.1f}s  |  "
      f"Variance explained: {explained_var:.1f}%")
print(f"    Train: {len(X_train_pca):,}  |  Test: {len(X_test_pca):,}")

# ------------------------------------------------------------------
# Graph 4: PCA Explained Variance
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(DARK_BG)
cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
ax.plot(cumvar, color=ACCENT, linewidth=2.5)
ax.fill_between(range(len(cumvar)), cumvar, alpha=0.2, color=ACCENT)
ax.axhline(90, color="#F39C12", linestyle="--", linewidth=1.5, label="90% variance")
ax.axhline(95, color="#E74C3C", linestyle="--", linewidth=1.5, label="95% variance")
ax.set_title("PCA -- Cumulative Explained Variance",
             fontsize=13, fontweight="bold", color=TEXT_CLR)
ax.set_xlabel("Number of Principal Components")
ax.set_ylabel("Cumulative Explained Variance (%)")
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_pca_variance.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    [OK] Saved: 04_pca_variance.png")

# Define models
models = {
    "Logistic Regression" : LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs",
        random_state=RANDOM_STATE
    ),
    "Random Forest"       : RandomForestClassifier(
        n_estimators=200, max_depth=20,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Linear SVM"          : LinearSVC(
        C=1.0, max_iter=2000, random_state=RANDOM_STATE
    ),
    "KNN (k=5)"           : KNeighborsClassifier(
        n_neighbors=5, n_jobs=-1
    ),
}

results = {}
MODEL_COLORS = ["#58A6FF", "#E74C3C", "#2ECC71", "#F39C12"]

for name, model in models.items():
    print(f"\n    Training: {name} ...")
    t_start = time.time()
    model.fit(X_train_pca, y_train)
    y_pred  = model.predict(X_test_pca)
    elapsed = time.time() - t_start

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "model": model, "y_pred": y_pred,
        "acc": acc, "report": report,
        "time": elapsed, "cm": cm,
    }
    print(f"      Accuracy : {acc*100:.2f}%  |  Time: {elapsed:.1f}s")

# ============================================================================
# STEP 5 -- RESULTS IN GRAPHS
# ============================================================================
print("\n[STEP 5]  Generating Result Graphs ...")
print("-" * 50)

model_names = list(results.keys())
model_accs  = [results[n]["acc"]  for n in model_names]
model_times = [results[n]["time"] for n in model_names]

# ------------------------------------------------------------------
# Graph 5: Accuracy Comparison
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(DARK_BG)
bars = ax.bar(model_names, model_accs, color=MODEL_COLORS,
              edgecolor=DARK_BG, linewidth=1.5, width=0.55)
for bar, acc in zip(bars, model_accs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f"{acc*100:.2f}%", ha="center",
            fontsize=12, fontweight="bold", color=TEXT_CLR)
ax.set_ylim(0, 1.10)
ax.set_title("Model Accuracy Comparison\n(Plant Disease Detection -- 3 GB Dataset)",
             fontsize=14, fontweight="bold", color=TEXT_CLR)
ax.set_ylabel("Accuracy"); ax.set_xlabel("Model")
ax.grid(axis="y")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_model_accuracy.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    [OK] Saved: 05_model_accuracy.png")

# ------------------------------------------------------------------
# Graph 6: Confusion Matrix (Best model only -- 38 classes is too large for all 4)
# ------------------------------------------------------------------
best_model = max(results, key=lambda n: results[n]["acc"])
cm = results[best_model]["cm"]

fig, ax = plt.subplots(figsize=(18, 16))
fig.patch.set_facecolor(DARK_BG)
short_labels = [c.split("___")[-1].replace("_", " ")[:15] for c in le.classes_]
sns.heatmap(cm, annot=False, fmt="d", ax=ax,
            cmap="YlOrRd",
            xticklabels=short_labels,
            yticklabels=short_labels,
            linewidths=0.3, linecolor=DARK_BG)
ax.set_title(f"Confusion Matrix -- {best_model} (Best Model)\n"
             f"Accuracy: {results[best_model]['acc']*100:.2f}%",
             fontsize=13, fontweight="bold", color=TEXT_CLR)
ax.set_xlabel("Predicted Label", fontsize=10)
ax.set_ylabel("True Label", fontsize=10)
ax.tick_params(axis="x", rotation=90, labelsize=7)
ax.tick_params(axis="y", rotation=0, labelsize=7)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_confusion_matrix.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    [OK] Saved: 06_confusion_matrix.png")

# ------------------------------------------------------------------
# Graph 7: Top 15 Best & Worst Classified Diseases (F1-Score)
# ------------------------------------------------------------------
f1_scores = {
    le.classes_[i]: results[best_model]["report"].get(le.classes_[i], {}).get("f1-score", 0)
    for i in range(num_classes)
}
f1_sorted = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)

top15    = f1_sorted[:15]
bottom15 = f1_sorted[-15:]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle(f"Top 15 Best vs Worst Classified Diseases (F1-Score) -- {best_model}",
             fontsize=13, fontweight="bold", color=TEXT_CLR)

for ax, data, clr, title in zip(
    axes,
    [top15, bottom15],
    ["#2ECC71", "#E74C3C"],
    ["Top 15 Best Classified", "Top 15 Worst Classified"]
):
    names = [d[0].split("___")[-1].replace("_", " ")[:30] for d in data]
    vals  = [d[1] for d in data]
    y_pos = range(len(names))
    ax.barh(y_pos, vals, color=clr, alpha=0.85, edgecolor=DARK_BG)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_title(title, color=clr, fontsize=12, fontweight="bold")
    ax.set_xlabel("F1-Score"); ax.grid(axis="x")
    for i, v in enumerate(vals):
        ax.text(v+0.01, i, f"{v:.2f}", va="center", fontsize=8, color=TEXT_CLR)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_best_worst_f1.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    [OK] Saved: 07_best_worst_f1.png")

# ------------------------------------------------------------------
# Graph 8: Training Time Comparison
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor(DARK_BG)
bars = ax.bar(model_names, model_times, color=MODEL_COLORS,
              edgecolor=DARK_BG, width=0.5)
for bar, t in zip(bars, model_times):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{t:.1f}s", ha="center", fontsize=12,
            fontweight="bold", color=TEXT_CLR)
ax.set_title("Training Time Comparison",
             fontsize=14, fontweight="bold", color=TEXT_CLR)
ax.set_ylabel("Seconds"); ax.set_xlabel("Model"); ax.grid(axis="y")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_training_time.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    [OK] Saved: 08_training_time.png")

# ------------------------------------------------------------------
# Graph 9: Per-class Accuracy (Heatmap for best model)
# ------------------------------------------------------------------
per_class_acc = cm.diagonal() / cm.sum(axis=1)
fig, ax = plt.subplots(figsize=(18, 4))
fig.patch.set_facecolor(DARK_BG)
short = [c.split("___")[-1].replace("_"," ")[:18] for c in le.classes_]
bars = ax.bar(range(num_classes), per_class_acc,
              color=plt.cm.RdYlGn(per_class_acc), edgecolor=DARK_BG)
ax.set_xticks(range(num_classes))
ax.set_xticklabels(short, rotation=90, fontsize=6)
ax.set_ylim(0, 1.1)
ax.axhline(per_class_acc.mean(), color="white", linestyle="--",
           linewidth=1.5, label=f"Mean: {per_class_acc.mean():.2f}")
ax.set_title(f"Per-Class Accuracy -- {best_model}",
             fontsize=13, fontweight="bold", color=TEXT_CLR)
ax.set_ylabel("Accuracy"); ax.legend(); ax.grid(axis="y")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_per_class_accuracy.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    [OK] Saved: 09_per_class_accuracy.png")

# ------------------------------------------------------------------
# Graph 10: Summary Dashboard
# ------------------------------------------------------------------
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor(DARK_BG)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Accuracy bar
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(range(len(model_names)), model_accs,
               color=MODEL_COLORS, edgecolor=DARK_BG, width=0.6)
ax1.set_xticks(range(len(model_names)))
ax1.set_xticklabels([n.replace(" ","\n") for n in model_names], fontsize=8)
ax1.set_ylim(0, 1.10)
ax1.set_title("Accuracy", fontsize=12, fontweight="bold", color=TEXT_CLR)
ax1.grid(axis="y")
for bar, v in zip(bars, model_accs):
    ax1.text(bar.get_x()+bar.get_width()/2, v+0.01,
             f"{v:.2f}", ha="center", fontsize=9, color=TEXT_CLR)

# Class distribution top 10
ax2 = fig.add_subplot(gs[0, 1:])
top10 = sorted_classes[:10]
t10_names = [c[0].split("___")[-1].replace("_"," ")[:20] for c in top10]
t10_vals  = [c[1] for c in top10]
cmap10 = plt.cm.get_cmap("plasma", 10)
ax2.barh(t10_names, t10_vals,
         color=[cmap10(i) for i in range(10)], edgecolor=DARK_BG)
ax2.invert_yaxis()
ax2.set_title("Top 10 Disease Categories by Image Count",
              fontsize=11, fontweight="bold", color=TEXT_CLR)
ax2.set_xlabel("Number of Images"); ax2.grid(axis="x")

# Per-class accuracy bar (bottom, full width)
ax3 = fig.add_subplot(gs[1, :2])
ax3.bar(range(num_classes), per_class_acc,
        color=plt.cm.RdYlGn(per_class_acc), edgecolor=DARK_BG, width=0.7)
ax3.axhline(per_class_acc.mean(), color="white", linestyle="--",
            linewidth=1.5, label=f"Mean Acc: {per_class_acc.mean():.2f}")
ax3.set_xticks(range(num_classes))
ax3.set_xticklabels(short, rotation=90, fontsize=5)
ax3.set_title(f"Per-Class Accuracy -- {best_model}",
              fontsize=11, fontweight="bold", color=TEXT_CLR)
ax3.set_ylabel("Accuracy"); ax3.legend(fontsize=9); ax3.grid(axis="y")

# Summary text
ax4 = fig.add_subplot(gs[1, 2])
ax4.axis("off")
summary = (
    f"DATASET INFO\n"
    f"{'='*28}\n"
    f"Name    : Plant Diseases\n"
    f"Size    : ~3 GB\n"
    f"Images  : 87,000+\n"
    f"Classes : {num_classes}\n"
    f"Loaded  : {len(X_raw):,}\n"
    f"Train   : {len(X_train):,}\n"
    f"Test    : {len(X_test):,}\n"
    f"PCA     : {PCA_COMPONENTS} components\n"
    f"Var     : {explained_var:.1f}%\n"
    f"{'='*28}\n"
    f"BEST MODEL\n"
    f"{'='*28}\n"
    f"Model   : {best_model}\n"
    f"Accuracy: {results[best_model]['acc']*100:.2f}%\n"
)
ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
         fontsize=9, verticalalignment="top", fontfamily="monospace",
         color=TEXT_CLR,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262D",
                   edgecolor=ACCENT, linewidth=1.5))

fig.suptitle("Plant Disease Classification -- Results Dashboard (3 GB Dataset)",
             fontsize=15, fontweight="bold", color=ACCENT)
plt.savefig(f"{OUTPUT_DIR}/10_dashboard.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    [OK] Saved: 10_dashboard.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("   RESULTS SUMMARY")
print("=" * 70)
print(f"\n{'Model':<22} {'Accuracy':>10} {'Time(s)':>10}")
print("-" * 45)
for name in model_names:
    r = results[name]
    marker = "  <- BEST" if name == best_model else ""
    print(f"{name:<22} {r['acc']*100:>9.2f}% {r['time']:>9.1f}s{marker}")

print(f"\n[BEST] Model    : {best_model}")
print(f"       Accuracy : {results[best_model]['acc']*100:.2f}%")
print(f"\n[INFO] All graphs saved to -> ./{OUTPUT_DIR}/")
print(f"       Total graphs : 10 PNG files")

print("\n" + "=" * 70)
print("   STEP 6 -- PUSH TO GITHUB")
print("=" * 70)
print("""
  git init
  git add .
  git commit -m "Plant Disease Classification - 3GB Dataset ML Assignment"
  git branch -M main
  git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
  git push -u origin main
""")
print("=" * 70)
print("   ASSIGNMENT COMPLETE [DONE]")
print("=" * 70)
