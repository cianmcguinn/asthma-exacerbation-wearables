# src/temporal.py
from __future__ import annotations
import numpy as np
import pandas as pd

def build_anchors_for_split(calendar_sorted: pd.DataFrame, mask: np.ndarray,
                            lookback: int, buffer: int, horizon: int,
                            user_col: str = "user_key", date_col: str = "date") -> pd.DataFrame:
    """Build anchor days for one split using L/B/H."""
    L, B, H = lookback, buffer, horizon
    if len(mask) != len(calendar_sorted):
        raise ValueError("Mask length must match calendar length.")

    records = []
    for uid, g in calendar_sorted.groupby(user_col, sort=False):
        idx_all = g.index.values
        idx_split = idx_all[mask[idx_all]]
        if idx_split.size == 0:
            continue

        last_pos = idx_split.size - 1
        max_anchor_pos = last_pos - (B + H - 1)
        if max_anchor_pos < L:
            continue

        for pos in range(L, max_anchor_pos + 1):
            anchor_idx = idx_split[pos]
            lb_idx  = idx_split[pos - L : pos]                 # t-L .. t-1
            buf_idx = idx_split[pos : pos + B]                 # t .. t+B-1
            hz_idx  = idx_split[pos + B : pos + B + H]         # t+B .. t+B+H-1

            records.append({
                "user_key": uid,
                "anchor_idx": int(anchor_idx),
                "t_date": int(calendar_sorted.at[anchor_idx, date_col]),
                "lb_start_idx": int(lb_idx[0]),   "lb_end_idx":  int(lb_idx[-1]),
                "buf_start_idx": int(buf_idx[0]), "buf_end_idx": int(buf_idx[-1]),
                "hz_start_idx": int(hz_idx[0]),   "hz_end_idx":  int(hz_idx[-1]),
                "L": L, "B": B, "H": H,
            })

    return pd.DataFrame.from_records(records)

def build_all_anchors(calendar_sorted: pd.DataFrame, masks: dict[str, np.ndarray],
                      lookback: int, buffer: int, horizon: int) -> dict[str, pd.DataFrame]:
    """Build anchors for train/val/test splits."""
    return {
        "train": build_anchors_for_split(calendar_sorted, masks["train"], lookback, buffer, horizon),
        "val":   build_anchors_for_split(calendar_sorted, masks["val"],   lookback, buffer, horizon),
        "test":  build_anchors_for_split(calendar_sorted, masks["test"],  lookback, buffer, horizon),
    }

def validate_anchors(anchors: dict[str, pd.DataFrame], masks: dict[str, np.ndarray]) -> None:
    """Ensure all stored indices lie within their split mask."""
    for split_name, df in anchors.items():
        if df.empty:
            continue
        mask = masks[split_name]
        cols = [c for c in ("lb_start_idx","lb_end_idx","buf_start_idx","buf_end_idx",
                            "hz_start_idx","hz_end_idx","anchor_row_idx","anchor_idx") if c in df.columns]
        idxs = np.concatenate([df[c].to_numpy(int) for c in cols])
        if not mask[idxs].all():
            raise ValueError(f"{split_name}: some anchor indices fall outside the split mask")
