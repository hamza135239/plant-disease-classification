# 🌿 Plant Disease Classification 

> Multi-class image classification of plant leaf images into **38 disease categories** using Machine Learning  
> **Dataset:** New Plant Diseases Dataset (3 GB, 87,000+ images) | **Models:** Logistic Regression · Random Forest · SVM · KNN

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Name** | New Plant Diseases Dataset (Augmented) |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) |
| **Size** | ~3 GB |
| **Images** | 87,000+ leaf images |
| **Classes** | 38 plant disease categories |
| **Format** | JPG/PNG, 256×256 pixels |

**Download:**
1. Go to → https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
2. Download → unzip → place folder `New Plant Diseases Dataset(Augmented)` in this directory

---

## 🛠️ Setup

```bash
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
python plant_disease_classification.py
```

Results are saved to the `results/` folder automatically.

---

## 📋 Steps Covered

| Step | Description |
|------|-------------|
| 1️⃣ | **Collect Dataset** — New Plant Diseases Dataset (3 GB, 87K+ images, 38 classes) |
| 2️⃣ | **Preprocess** — Resize to 64×64, flatten pixels, StandardScaler, PCA (150 components) |
| 3️⃣ | **Read Dataset** — EDA with class distribution, sample images, pixel intensity analysis |
| 4️⃣ | **Classification Models** — Logistic Regression, Random Forest, Linear SVM, KNN |
| 5️⃣ | **Results in Graphs** — Confusion matrix, per-class accuracy, F1 scores, dashboard |
| 6️⃣ | **Open Source** — This GitHub repository |

---

## 📈 Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 45.44% |
| **Random Forest** | **56.93% 🏆** |
| Linear SVM | 48.16% |
| KNN (k=5) | 36.84% |

> **Note:** These results are expected for traditional ML on 38-class image data without deep learning. Random Forest achieved the best performance.

---

## 🖼️ Graphs Produced

| File | Description |
|------|-------------|
| `01_class_distribution.png` | Bar chart of images per disease category |
| `02_sample_images.png` | Sample leaf images from 18 disease categories |
| `03_pixel_intensity.png` | Pixel intensity: healthy vs diseased leaves |
| `04_pca_variance.png` | PCA cumulative explained variance |
| `05_model_accuracy.png` | Accuracy comparison of all 4 models |
| `06_confusion_matrix.png` | 38×38 confusion matrix (best model) |
| `07_best_worst_f1.png` | Top 15 best & worst classified diseases |
| `08_training_time.png` | Training time comparison |
| `09_per_class_accuracy.png` | Per-class accuracy bar chart |
| `10_dashboard.png` | Complete results dashboard |

---

## 📁 Project Structure

```
ML_Assignment/
├── plant_disease_classification.py   # Main assignment script
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── .gitignore                        # Excludes large dataset folder
├── New Plant Diseases Dataset(Augmented)/  # Download from Kaggle (not in repo)
└── results/                          # Auto-generated graphs
    ├── 01_class_distribution.png
    ├── 02_sample_images.png
    ├── ...
    └── 10_dashboard.png
```

---

## 🔧 Technologies Used

- **Python 3.8+**
- **NumPy, Pillow** — Image loading & processing
- **scikit-learn** — PCA, ML models, evaluation metrics
- **matplotlib, seaborn** — Visualization & graphs
- **tqdm** — Progress bar for image loading

---


