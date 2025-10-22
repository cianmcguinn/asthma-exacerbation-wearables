# src/outcomes.py
from __future__ import annotations
import numpy as np, pandas as pd


def ensure_inhaler_daily_use(inh_df: pd.DataFrame) -> pd.DataFrame:
    if inh_df is None or inh_df.empty:
        raise ValueError("Inhaler dataframe is empty/None.")

    df = inh_df.copy()

    # A) already daily with 'use'
    if {"user_key","date","use"}.issubset(df.columns):
        out = df[["user_key","date","use"]].copy()
        out["user_key"] = out["user_key"].astype(str)
        out["date"] = pd.to_numeric(out["date"], errors="coerce").astype("Int64")
        out = out.dropna(subset=["date"]).astype({"date": int})
        out["use"] = pd.to_numeric(out["use"], errors="coerce").fillna(0).astype(int)
        return out

    # B) daily counts
    cnt = next((c for c in ["count","actuations","puffs","num_events","events","n"] if c in df.columns), None)
    if cnt is not None and "user_key" in df.columns:
        if "date" not in df.columns:
            ts_col = next((c for c in ["time","timestamp","datetime","event_time","t"] if c in df.columns), None)
            if ts_col is None:
                raise KeyError("Inhaler needs 'date' or a timestamp column to derive daily 'date'.")
            tmp = df.copy()
            tmp["__dt"] = pd.to_datetime(tmp[ts_col], errors="coerce")
            if tmp["__dt"].isna().all():
                raise ValueError("Inhaler timestamps could not be parsed.")
            base = tmp["__dt"].dt.floor("D").min()
            tmp["date"] = (tmp["__dt"].dt.floor("D") - base).dt.days.astype(int)
            df = tmp

        out = df[["user_key","date",cnt]].copy()
        out["user_key"] = out["user_key"].astype(str)
        out["date"] = pd.to_numeric(out["date"], errors="coerce").astype("Int64")
        out = out.dropna(subset=["date"]).astype({"date": int})
        out["use"] = (pd.to_numeric(out[cnt], errors="coerce").fillna(0) > 0).astype(int)
        return out[["user_key","date","use"]]

    # C) event-level with a timestamp
    ts_col = next((c for c in ["time","timestamp","datetime","event_time","t"] if c in df.columns), None)
    if ts_col is not None and "user_key" in df.columns:
        tmp = df.copy()
        tmp["__dt"] = pd.to_datetime(tmp[ts_col], errors="coerce")
        if tmp["__dt"].isna().all():
            raise ValueError("Inhaler timestamps could not be parsed.")
        base = tmp["__dt"].dt.floor("D").min()
        tmp["date"] = (tmp["__dt"].dt.floor("D") - base).dt.days.astype(int)
        tmp["user_key"] = tmp["user_key"].astype(str)
        daily = (tmp.groupby(["user_key","date"]).size().rename("events").reset_index())
        daily["use"] = (daily["events"] > 0).astype(int)
        return daily[["user_key","date","use"]]

    # D) only user_key + date (your CSV)
    if {"user_key","date"}.issubset(df.columns):
        out = df[["user_key","date"]].copy()
        out["user_key"] = out["user_key"].astype(str)
        out["date"] = pd.to_numeric(out["date"], errors="coerce").astype("Int64")
        out = out.dropna(subset=["date"]).astype({"date": int})
        out = out.drop_duplicates(["user_key","date"]).assign(use=1)
        return out[["user_key","date","use"]]

    raise KeyError("Inhaler data must contain 'user_key' and 'date', or a timestamp, or a daily 'use'/'count'.")



def select_usage_daily(source: str, inh_daily: pd.DataFrame, dq_source: pd.DataFrame) -> pd.DataFrame:
    if source == "inhaler":
        # 1) normalize: events/daily → ['user_key','date','use'] with use=1 when present
        d = ensure_inhaler_daily_use(inh_daily)
        # 2) PAD to dense per-user calendar so missing days become use=0  (matches main pipelines)
        d = _pad_dense_daily(d, "user_key", "date", "use")
        return d

    if source == "questionnaire":
        dq = dq_source.copy()
        dq.columns = [str(c).strip().lower().replace(" ", "_") for c in dq.columns]
        candidates = [
            "reliever_self","daily_relief_inhaler","reliever_today","reliever_use",
            "daily_reliever_inhaler","reliever_inhaler","reliever","reliever_puffs","reliever_taken",
        ]
        col = next((c for c in candidates if c in dq.columns), None)
        if col is None:
            raise ValueError("Reliever-use column not found in questionnaire data.")

        out = dq[["user_key","date",col]].rename(columns={col:"use"})
        out["use"] = pd.to_numeric(out["use"], errors="coerce")

        # Questionnaire stays padded (as you already do)
        out = _pad_dense_daily(out, "user_key", "date", "use")
        return out

    raise ValueError("OUTCOME_SOURCE must be 'inhaler' or 'questionnaire'.")





def compute_personal_baseline_by_train_keys(usage_daily: pd.DataFrame, train_keys_df: pd.DataFrame,
                                            user_col: str = "user_key", date_col: str = "date") -> pd.Series:
    td = usage_daily.merge(train_keys_df[[user_col, date_col]], on=[user_col, date_col], how="inner")
    return td.groupby(user_col)["use"].median()

def build_labels(usage_daily: pd.DataFrame, anchors: pd.DataFrame,
                 user_col: str, date_col: str, baseline_by_user: pd.Series,
                 outcome_def: str, buffer_days: int, horizon_days: int,
                 min_obs_in_hz: int, allow_partial: bool) -> pd.DataFrame:
    df = usage_daily.copy().sort_values([user_col, date_col]).reset_index(drop=True)
    df[user_col] = df[user_col].astype(str)
    df[date_col] = pd.to_numeric(df[date_col], errors="coerce").astype("Int64")

    out_rows = []
    B, H = buffer_days, horizon_days

    for _, a in anchors.iterrows():
        uid = str(a[user_col]); t = int(a["t_date"])
        g = df[df[user_col] == uid].sort_values(date_col).reset_index(drop=True)
        pos_arr = g.index[g[date_col] == t]
        if len(pos_arr) == 0: continue
        pos = int(pos_arr[0])

        start, end = pos + B, pos + B + H - 1
        if allow_partial:
            hz_start, hz_end = start, min(end, len(g) - 1)
            if hz_end < hz_start: continue
        else:
            if end >= len(g): continue
            hz_start, hz_end = start, end

        thresh = baseline_by_user.get(uid, np.nan)
        if np.isnan(thresh): continue

        vals = g.loc[hz_start:hz_end, "use"].astype(float).values
        obs = np.isfinite(vals)
        if obs.sum() < min_obs_in_hz: continue

        if outcome_def == "primary":
            label = int((vals[obs] > thresh).sum() >= 2)
        elif outcome_def == "consecutive":
            elev = np.zeros_like(vals, dtype=bool); elev[obs] = vals[obs] > thresh
            label = int(np.any(elev[:-1] & elev[1:]))
        else:
            raise ValueError("Unknown outcome_def")

        out_rows.append({user_col: uid, date_col: t, "y": label})

    return pd.DataFrame(out_rows, columns=[user_col, date_col, "y"])


def _pad_dense_daily(df: pd.DataFrame, user_col="user_key", date_col="date", use_col="use") -> pd.DataFrame:
    """
    For each user, pad days from min(date) to max(date).
    Missing days get use=0. Assumes 'date' is an integer day index.
    """
    parts = []
    for uid, g in df.groupby(user_col, sort=False):
        if g.empty:
            continue
        dmin, dmax = int(g[date_col].min()), int(g[date_col].max())
        full = pd.DataFrame({user_col: uid, date_col: np.arange(dmin, dmax + 1, dtype=int)})
        gg = full.merge(g[[date_col, use_col]], on=date_col, how="left")
        gg[use_col] = pd.to_numeric(gg[use_col], errors="coerce").fillna(0).astype(int)
        parts.append(gg)
    if not parts:
        return df[[user_col, date_col, use_col]].copy()
    return (pd.concat(parts, ignore_index=True)
            [[user_col, date_col, use_col]]
            .sort_values([user_col, date_col])
            .reset_index(drop=True))


