from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score

def _nearest_idx(thresholds, thr):
    if thresholds.size == 0: return 0
    i = np.searchsorted(thresholds, thr, side="left")
    if i == 0: return 0
    if i >= thresholds.size: return thresholds.size - 1
    return i if abs(thresholds[i]-thr) < abs(thresholds[i-1]-thr) else i-1

def plot_pr_roc(y_true, y_prob, thr, title_prefix, save_dir: Path, prefix_file):
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Precision–Recall
    prec, rec, thr_pr = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(rec, prec, label=f"AUPRC={ap:.3f}")
    if thr_pr.size:
        k = _nearest_idx(thr_pr, thr); plt.scatter(rec[k], prec[k], s=40, zorder=3)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{title_prefix} — Precision–Recall")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(save_dir / f"{prefix_file}_PR.png", dpi=300)
    plt.savefig(save_dir / f"{prefix_file}_PR.pdf", dpi=300)
    plt.show()
    thr_pr_pad = np.r_[thr_pr, np.nan]
    n = min(len(rec), len(prec), len(thr_pr_pad))
    pd.DataFrame({"recall": rec[:n], "precision": prec[:n], "threshold": thr_pr_pad[:n]}).to_csv(
        save_dir / f"{prefix_file}_PR.csv", index=False
    )

    # ---- ROC
    fpr, tpr, thr_roc = roc_curve(y_true, y_prob)
    au = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUROC={au:.3f}")
    if thr_roc.size:
        k = _nearest_idx(thr_roc, thr); plt.scatter(fpr[k], tpr[k], s=40, zorder=3)
    plt.plot([0, 1], [0, 1], "--", alpha=0.5)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} — ROC")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(save_dir / f"{prefix_file}_ROC.png", dpi=300)
    plt.savefig(save_dir / f"{prefix_file}_ROC.pdf", dpi=300)
    plt.show()
    thr_roc_pad = np.r_[thr_roc, np.nan]
    m = min(len(fpr), len(tpr), len(thr_roc_pad))
    pd.DataFrame({"fpr": fpr[:m], "tpr": tpr[:m], "threshold": thr_roc_pad[:m]}).to_csv(
        save_dir / f"{prefix_file}_ROC.csv", index=False
    )
