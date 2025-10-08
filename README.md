# Asthma Exacerbation Prediction from Wearable Heart Rate

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<USERNAME>/<REPO>/blob/main/notebooks/project.ipynb)
[![nbviewer](https://img.shields.io/badge/view-nbviewer-blue)](https://nbviewer.org/github/<USERNAME>/<REPO>/blob/main/notebooks/project.ipynb)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

**What this is**  
A reproducible ML project that predicts near‑term asthma exacerbation risk using smartwatch‑derived heart‑rate (HR) features within a **temporal prediction framework** (7‑day lookback → 2‑day buffer → 3‑day horizon). The repo includes a polished notebook and a fast‑loading HTML mirror for recruiters.

**Why it matters**  
Shows end‑to‑end DS craft: temporal framing to avoid leakage, subject‑aware splits, engineered circadian/variability features, model selection (logistic vs XGBoost), calibrated evaluation (AUROC, AUPRC), SHAP interpretability, and robust sensitivity analyses.

---

## TL;DR (Outcome)
- **Data**: AAMOS‑00 study (22 adults, smartwatch HR + daily questionnaires). A **small synthetic sample** is provided for demo; full data are restricted.
- **Method**: Temporal windows with lookback/buffer/horizon; aggregated HR stats (means, SD, RMSSD), circadian bins, activity fractions. Models: Logistic Regression & XGBoost.
- **Results (test set)**: Active benchmark stronger; **HR‑only passive** model shows **moderate** discrimination.  
  - Active XGBoost ~ AUROC 0.84, AUPRC 0.65  
  - Passive XGBoost ~ AUROC 0.76, AUPRC 0.49
- **Interpretability**: Active—reliever use & night symptoms; Passive—**HR variability** and **circadian** disruption dominate.
- **Impact**: HR alone is not a replacement for active monitoring, but is a promising **passive digital biomarker** for multimodal systems.

> See `docs/project.html` for an instant, no‑friction preview.

---

## Repo Structure
```
.
├─ README.md
├─ notebooks/
│  └─ project.ipynb              # polished Colab notebook (narrated, restart & run all)
├─ docs/
│  └─ project.html               # exported HTML for GitHub Pages (fast to load)
├─ src/
│  └─ utils.py                   # small helpers (feature eng., metrics, plotting)
├─ figures/
│  ├─ roc_active.png
│  ├─ roc_passive.png
│  ├─ pr_active.png
│  ├─ pr_passive.png
│  └─ shap_top15_passive.png
├─ data/
│  └─ synthetic_demo.csv         # tiny synthetic sample to run the notebook end‑to‑end
├─ requirements.txt
├─ .gitignore
└─ LICENSE
```

### Key Figures (for quick scanning)
Place exported images into `figures/` and they’ll render here:
- **ROC/PR** curves for active vs. passive
- **Top‑15 SHAP** features (passive)
- **Sensitivity** bar chart

---

## Quickstart
```bash
# Clone
git clone https://github.com/<USERNAME>/<REPO>.git
cd <REPO>

# (Optional) create a venv/conda env

# Install minimal deps
pip install -r requirements.txt

# Launch
jupyter lab
# open notebooks/project.ipynb and Run All
```

Or open directly in Colab:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<USERNAME>/<REPO>/blob/main/notebooks/project.ipynb)

---

## Methods in Brief
- **Temporal framing**: 7‑day lookback → 2‑day action buffer → 3‑day horizon to ensure realistic lead time and **no leakage**.
- **Features**: per‑day HR stats (mean/min/max/SD, RMSSD), circadian bins (night/morning/afternoon/evening), step totals, activity fractions; aggregated with mean/median/min/max/range/SD/trend.
- **Feature selection**: near‑zero variance filter; high‑correlation pruning.
- **Models**: Logistic Regression (strong baseline) & XGBoost (non‑linear interactions); randomised hyper‑param search; early stopping on **validation AUPRC**.
- **Splitting**: Chronological, by participant, into train/val/test to prevent window overlap across splits.
- **Metrics**: AUROC + **AUPRC** (class‑imbalance aware). Threshold picked by F1 on validation.
- **Explainability**: SHAP to rank/visualise drivers (direction + magnitude).

---

## Results (headline)
| Model (Test)      | AUROC | AUPRC | Notes |
|-------------------|:-----:|:-----:|------|
| Active – XGBoost  | ~0.84 | ~0.65 | High specificity at chosen threshold |
| Passive – XGBoost | ~0.76 | ~0.49 | HR variability & circadian signals drive performance |

> Exact numbers and curves are in the notebook and figures.

### Sensitivity Highlights
- Stricter “consecutive‑day” outcome ↓ performance (rarer positives).  
- Longer horizons (4–5d) ↓ precision vs. baseline horizon.  
- Per‑user normalisation **hurt** passive performance (individual baselines carry signal).

---

## Data Availability
- The original dataset is **restricted**. This repo ships with a **synthetic demo** enabling full execution of the notebook. Replace with your own data via the interface described in the notebook.

---

## How recruiters should read this
- Start with `docs/project.html` (fast).  
- Skim **TL;DR**, then **Results** and **Key Figures**.  
- Dive into `notebooks/project.ipynb` for methodology and validation details.

---

## Reproducibility
- Deterministic seeds set across libraries.
- `requirements.txt` pins a **minimal** set of versions actually used.
- Notebook is clean room (restart & run all).

---

## How this repo was built (for you to replicate)
1. Export the Colab notebook (`.ipynb`) into `notebooks/`.
2. Export HTML (`File → Print → Save as PDF/HTML` or `jupyter nbconvert --to html`) into `docs/project.html`.
3. Enable **GitHub Pages**: *Settings → Pages → Source: `main` / `/docs`*.
4. Add **Colab** and **nbviewer** badges to this README.
5. Pin the repo on your GitHub profile for visibility.

---

## License
MIT (see `LICENSE`).

## Citation
If you reference this work academically, please cite the associated MSc dissertation or this repository’s release DOI (if minted).
[README_asthma_wearables.md](https://github.com/user-attachments/files/22779759/README_asthma_wearables.md)

