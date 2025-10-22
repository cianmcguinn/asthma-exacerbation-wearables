from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_grouped_bars(
    df: pd.DataFrame,
    title: str,
    *,
    baseline_first: bool = True,
    sort_by: str = "AUROC",
    figsize: Tuple[int, int] = (11, 5),
    save_path: Optional[str | Path] = None,
    save_formats: Iterable[str] = ("png", "pdf"),
    show: bool = True,
):
    """
    Grouped bar chart for sensitivity results with baseline visually distinguished.
    Expected df columns: ['scenario','AUROC','AUPRC'] with baseline at row 0.
    """
    if {"scenario","AUROC","AUPRC"} - set(df.columns):
        raise ValueError("df must contain columns: 'scenario','AUROC','AUPRC'")

    base = df.iloc[[0]]
    rest = df.iloc[1:].sort_values(sort_by, ascending=False) if len(df) > 1 else df.iloc[0:0]
    plot_df = pd.concat([base, rest], ignore_index=True) if baseline_first else df.copy()

    scenarios = plot_df["scenario"].tolist()
    auroc = plot_df["AUROC"].to_numpy()
    auprc = plot_df["AUPRC"].to_numpy()

    x = np.arange(len(scenarios)); w = 0.38
    fig, ax = plt.subplots(figsize=figsize)
    b1 = ax.bar(x - w/2, auroc, w, label="AUROC", alpha=0.95)
    b2 = ax.bar(x + w/2, auprc, w, label="AUPRC", alpha=0.85)

    # hatch the baseline
    if len(b1) > 0:
        b1[0].set_hatch("//"); b2[0].set_hatch("//")
        b1[0].set_alpha(0.75); b2[0].set_alpha(0.65)

    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{h:.2f}", (r.get_x()+r.get_width()/2, h),
                        xytext=(0,3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_title(title); ax.set_ylabel("Score"); ax.set_ylim(0,1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3); ax.legend(loc="upper right")
    ax.set_xticks(x); ax.set_xticklabels(scenarios, rotation=35, ha="right")
    plt.tight_layout()

    if save_path is not None:
        p = Path(save_path); p.parent.mkdir(parents=True, exist_ok=True)
        for ext in save_formats:
            fig.savefig(p.with_suffix(f".{ext}"), dpi=300)

    if show: plt.show()
    else:    plt.close(fig)
    return fig, ax

# Backwards-compatible alias for old name in plots.py
def grouped_bars_sensitivity(df: pd.DataFrame, title: str, stem: str, save_dir: Path, figsize=(11,6)):
    fig, ax = plot_grouped_bars(
        df, title, figsize=figsize, save_path=Path(save_dir) / stem, show=True
    )
    return fig, ax
