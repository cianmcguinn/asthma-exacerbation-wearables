# src/splits.py
from __future__ import annotations
import numpy as np
import pandas as pd

def split_timelines(user_daily: pd.DataFrame, user_col: str, date_col: str,
                    train_ratio: float, val_ratio: float,
                    min_user_windows_train_only: int) -> dict[str, np.ndarray]:
    """
    Chronological per-user split into train/val/test using ratios.
    Users with < min_user_windows_train_only days go to train only.
    """
    df = user_daily.sort_values([user_col, date_col]).reset_index(drop=True)
    idx_train, idx_val, idx_test = [], [], []

    for _, g in df.groupby(user_col, sort=False):
        n = len(g)
        if n == 0:
            continue
        if n < min_user_windows_train_only:
            idx_train += g.index.tolist()
            continue

        t_cut = int(np.floor(train_ratio * n))
        v_cut = int(np.floor((train_ratio + val_ratio) * n))
        idx_train += g.index[:t_cut].tolist()
        idx_val   += g.index[t_cut:v_cut].tolist()
        idx_test  += g.index[v_cut:].tolist()

    return {
        "train": np.array(idx_train, dtype=int),
        "val":   np.array(idx_val,   dtype=int),
        "test":  np.array(idx_test,  dtype=int),
    }

def make_split_masks(n_rows: int, split: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Boolean masks for train/val/test given index dict from split_timelines."""
    all_idx = np.arange(n_rows, dtype=int)
    return {
        "train": np.isin(all_idx, split["train"]),
        "val":   np.isin(all_idx, split["val"]),
        "test":  np.isin(all_idx, split["test"]),
    }

def per_user_split_summary(calendar: pd.DataFrame, masks: dict[str, np.ndarray],
                           user_col: str = "user_key") -> pd.DataFrame:
    """Compact per-user day counts by split (for display)."""
    labels = np.where(masks["train"], "train", np.where(masks["val"], "val", "test"))
    return (
        calendar.assign(split=labels)
                .groupby([user_col, "split"]).size()
                .unstack(fill_value=0)
                .sort_index()
    )
