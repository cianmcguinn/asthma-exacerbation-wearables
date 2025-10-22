# src/data_io.py
from __future__ import annotations
# src/data_io.py
from typing import Iterable, Dict
import pandas as pd

try:
    from .outcomes import ensure_inhaler_daily_use
except ImportError:
    from src.outcomes import ensure_inhaler_daily_use


def load_raw(hr_paths: Iterable[str], inhaler_path: str, dq_path: str) -> Dict[str, pd.DataFrame]:
    na_vals = ["NA", "NaN", "NULL", "null", ""]

    # HR
    hr = pd.concat(
        [pd.read_csv(p, low_memory=False, na_values=na_vals, keep_default_na=True) for p in hr_paths],
        ignore_index=True
    )
    hr = hr.rename(columns={"activity_typ": "activity_code"})
    if "time" in hr.columns:
        hr["time"] = hr["time"].astype(str)
    for c in ("hr", "steps", "intensity", "activity_code"):
        if c in hr.columns:
            hr[c] = pd.to_numeric(hr[c], errors="coerce")

    # Inhaler (events or daily)
    inhaler = pd.read_csv(inhaler_path, low_memory=False, na_values=na_vals, keep_default_na=True)

    # DQ (daily questionnaire)
    dq = pd.read_csv(dq_path, low_memory=False, na_values=na_vals, keep_default_na=True)

    # Light standardisation
    for df in (hr, inhaler, dq):
        if "user_key" in df.columns:
            df["user_key"] = df["user_key"].astype(str)
        if "date" in df.columns:
            df["date"] = pd.to_numeric(df["date"], errors="coerce").astype("Int64")

    # Minimal checks
    assert {"user_key", "date"}.issubset(hr.columns), "HR must include user_key and date"

    # Normalize inhaler to daily ['user_key','date','use']
    inh_daily = ensure_inhaler_daily_use(inhaler)

    return {
        "hr": hr,
        "inhaler": inhaler,          # keep original file (events/daily) for reference
        "dq": dq,
        "inh_daily": inh_daily,      # preferred daily table used by both pipelines
        "inhaler_daily": inh_daily,  # alias/back-compat
    }

