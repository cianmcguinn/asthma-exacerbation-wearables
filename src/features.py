# src/features.py
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import skew
from typing import Iterable

def daily_passive_features(hr_minute: pd.DataFrame, *, samples_per_day: int, hr_min_coverage: float) -> pd.DataFrame:
    df = hr_minute.copy()

    cov = (
        df.groupby(['user_key','date'])['hr']
          .apply(lambda s: np.isfinite(s).sum() / samples_per_day)
          .rename('hr_coverage')
    )
    valid_days = cov[cov >= hr_min_coverage].reset_index()
    df_valid = df.merge(valid_days[['user_key','date']], on=['user_key','date'], how='inner')

    def count_abrupt(x):
        v = pd.Series(x, dtype='float64').dropna().values
        if v.size < 2: return 0
        return int((np.abs(np.diff(v)) > 5).sum())

    def rmssd(x):
        v = pd.Series(x, dtype='float64').dropna().values
        if v.size < 3: return np.nan
        d = np.diff(v)
        return float(np.sqrt(np.mean(d**2)))

    bins = pd.cut(df_valid['hour'], bins=[-0.1,6,12,18,24],
                  labels=['night','morning','afternoon','evening'])
    df_valid = df_valid.assign(circ_bin=bins)

    def act_cat(code):
        if pd.isna(code): return 'unknown'
        code = int(code)
        if code == 99: return 'sleep'
        if code in (80,89,90,91,92,96): return 'sedentary'
        if code in (1,2,3,4,5,6,82,98): return 'active'
        return 'unknown'
    df_valid['act_cat'] = df_valid.get('activity_code', pd.Series(index=df_valid.index)).apply(act_cat) if 'activity_code' in df_valid else 'unknown'

    agg1 = df_valid.groupby(['user_key','date']).agg(
        mean_hr=('hr','mean'),
        std_hr=('hr','std'),
        min_hr=('hr','min'),
        max_hr=('hr','max'),
        abrupt_hr_changes=('hr', count_abrupt),
        rmssd_hr=('hr', rmssd),
        steps_total=('steps','sum'),
    )

    circ = (df_valid.dropna(subset=['circ_bin'])
                   .groupby(['user_key','date','circ_bin'])['hr'].mean()
                   .unstack('circ_bin'))
    if circ is not None:
        circ = circ.rename(columns=lambda c: f'hr_{c}_mean')

    act_frac = (df_valid.groupby(['user_key','date','act_cat'])['hr'].size()
                        .unstack('act_cat').fillna(0))
    act_frac = act_frac.div(act_frac.sum(axis=1), axis=0).fillna(0).rename(columns=lambda c: f'frac_{c}')

    out_valid = agg1.join(circ, how='left').join(act_frac, how='left').reset_index()

    out = (out_valid.merge(cov.reset_index(), on=['user_key','date'], how='right')
                   .sort_values(['user_key','date']).reset_index(drop=True))
    return out

def forward_fill_limited(df: pd.DataFrame, user_col: str, limit_days: int, feature_cols: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    if feature_cols is None:
        feature_cols = [c for c in out.columns if c not in (user_col, "date", "hr_coverage")]
    out[feature_cols] = out.groupby(user_col, group_keys=False)[feature_cols].apply(lambda g: g.ffill(limit=limit_days))
    return out

def align_passive_to_calendar_and_ffill(passive_daily: pd.DataFrame, calendar: pd.DataFrame,
                                        user_col: str = "user_key", date_col: str = "date",
                                        limit_days: int = 3) -> pd.DataFrame:
    base = calendar[[user_col, date_col]].drop_duplicates().sort_values([user_col, date_col])
    merged = base.merge(passive_daily, on=[user_col, date_col], how="left")
    return forward_fill_limited(merged, user_col=user_col, limit_days=limit_days, feature_cols=None)

def daily_active_features(dq_aligned: pd.DataFrame, *, include_prev1: bool) -> pd.DataFrame:
    cols_min = {"user_key", "date", "obs_day"}
    if not cols_min.issubset(dq_aligned.columns):
        raise ValueError("ACTIVE daily data missing required keys.")

    allowed = {
        "daily_night_symp", "daily_day_symp", "daily_limit_activity",
        "daily_triggers", "preventer_taken", "preventer_delta",
    }
    if include_prev1 and "reliever_prev1" in dq_aligned.columns:
        allowed.add("reliever_prev1")

    keep = ["user_key", "date", "obs_day"] + [c for c in dq_aligned.columns if c in allowed]
    return dq_aligned[keep].sort_values(["user_key", "date"]).reset_index(drop=True)

def rollup_7d_passive(daily_passive: pd.DataFrame,
                      anchors: pd.DataFrame,
                      user_col: str = "user_key",
                      date_col: str = "date",
                      lookback: int = 7) -> pd.DataFrame:
    dp = daily_passive.copy().sort_values([user_col, date_col]).reset_index(drop=True)
    non_feat = {user_col, date_col, "hr_coverage"}
    feat_cols = [c for c in dp.columns if c not in non_feat]

    rows = []
    for _, a in anchors.iterrows():
        uid = a[user_col]; adate = a["t_date"]
        g = dp[dp[user_col] == uid].sort_values(date_col).reset_index(drop=True)
        pos_arr = g.index[g[date_col] == adate]
        if len(pos_arr) == 0: continue
        pos = int(pos_arr[0])

        start, end = pos - lookback, pos - 1
        if start < 0: continue
        win = g.loc[start:end, feat_cols]

        def slope(arr):
            y = np.asarray(arr, dtype=float)
            m = ~np.isnan(y)
            if m.sum() < 2: return np.nan
            x = np.arange(m.sum(), dtype=float)
            return float(np.polyfit(x, y[m], 1)[0])

        roll = {}
        for c in feat_cols:
            arr = win[c].values.astype(float)
            m = ~np.isnan(arr)
            if not m.any():
                roll.update({f"{c}_mean7": np.nan, f"{c}_median7": np.nan, f"{c}_max7": np.nan,
                             f"{c}_min7": np.nan, f"{c}_range7": np.nan, f"{c}_std7": 0.0,
                             f"{c}_skew7": 0.0,    f"{c}_slope7": np.nan, f"{c}_last": np.nan})
            else:
                roll[f"{c}_mean7"]   = float(np.nanmean(arr))
                roll[f"{c}_median7"] = float(np.nanmedian(arr))
                roll[f"{c}_max7"]    = float(np.nanmax(arr))
                roll[f"{c}_min7"]    = float(np.nanmin(arr))
                roll[f"{c}_range7"]  = roll[f"{c}_max7"] - roll[f"{c}_min7"]
                roll[f"{c}_std7"]    = float(np.nanstd(arr, ddof=1)) if m.sum() > 1 else 0.0
                roll[f"{c}_skew7"]   = float(skew(arr[m])) if m.sum() > 2 else 0.0
                roll[f"{c}_slope7"]  = slope(arr)
                roll[f"{c}_last"]    = float(arr[np.where(m)[0][-1]])

        roll[user_col] = uid; roll[date_col] = adate
        rows.append(roll)

    return pd.DataFrame(rows)

def rollup_7d_active(daily_active: pd.DataFrame, anchors_split: pd.DataFrame,
                     user_col: str = "user_key", date_col: str = "date",
                     lookback: int = 7, min_obs_frac: float = 0.5) -> pd.DataFrame:
    da = daily_active.copy().sort_values([user_col, date_col]).reset_index(drop=True)
    if "row_idx" not in da.columns: da["row_idx"] = da.index

    non_feat = {user_col, date_col, "row_idx", "obs_day"}
    feat_cols = [c for c in da.columns if c not in non_feat]
    min_obs_days = int(np.ceil(min_obs_frac * lookback))

    def _slope(arr):
        y = np.asarray(arr, dtype=float)
        m = ~np.isnan(y)
        if m.sum() < 2: return np.nan
        x = np.arange(len(y), dtype=float)[m]
        return float(np.polyfit(x, y[m], 1)[0])

    rows = []
    idx_to_ud = da.set_index("row_idx")[[user_col, date_col]].to_dict("index")

    for _, a in anchors_split.iterrows():
        ridx = a["anchor_row_idx"]; uid = idx_to_ud[ridx][user_col]; t = idx_to_ud[ridx][date_col]
        g = da[da[user_col] == uid].sort_values(date_col).reset_index(drop=True)
        pos = g.index[g[date_col] == t]
        if len(pos) == 0: continue
        pos = int(pos[0])

        start, end = pos - lookback, pos - 1
        if start < 0: continue
        win = g.loc[start:end]
        if "obs_day" in win.columns and int(win["obs_day"].sum()) < min_obs_days:
            continue

        roll = {}
        for c in feat_cols:
            arr = win[c].values.astype(float)
            m = ~np.isnan(arr)
            if not m.any():
                roll.update({f"{c}_mean7": np.nan, f"{c}_median7": np.nan, f"{c}_max7": np.nan,
                             f"{c}_min7": np.nan, f"{c}_range7": np.nan, f"{c}_std7": 0.0,
                             f"{c}_skew7": 0.0,    f"{c}_slope7": np.nan, f"{c}_last": np.nan})
                continue
            roll[f"{c}_mean7"]   = float(np.nanmean(arr))
            roll[f"{c}_median7"] = float(np.nanmedian(arr))
            roll[f"{c}_max7"]    = float(np.nanmax(arr))
            roll[f"{c}_min7"]    = float(np.nanmin(arr))
            roll[f"{c}_range7"]  = roll[f"{c}_max7"] - roll[f"{c}_min7"]
            roll[f"{c}_std7"]    = float(np.nanstd(arr, ddof=1)) if m.sum() > 1 else 0.0
            roll[f"{c}_skew7"]   = float(skew(arr[m])) if m.sum() > 2 else 0.0
            roll[f"{c}_slope7"]  = _slope(arr)
            roll[f"{c}_last"]    = float(arr[m][-1])

        roll[user_col] = uid; roll[date_col] = t
        rows.append(roll)

    return pd.DataFrame(rows)



def assemble_matrices(passive_roll, active_roll, labels, mode, verbose=False):
    key = ["user_key","date"]
    F = passive_roll if mode == "passive" else active_roll
    if F is None:
        raise ValueError(f"features missing for mode={mode}")

    Xy = F.merge(labels, on=key, how="inner", validate="one_to_one")
    dropped_feats = len(F) - len(Xy)
    dropped_lbls  = len(labels) - len(Xy)

    if verbose and (dropped_feats or dropped_lbls):
        print(f"{mode}: dropped_features={dropped_feats}, labels_wo_feat={dropped_lbls}")

    y = Xy.pop("y").astype(int)
    ids = Xy[key].copy()
    feat_cols = [c for c in Xy.columns if c not in key]
    X = Xy[feat_cols].copy()
    return X, y, ids, feat_cols


def variance_filter_cols(X: pd.DataFrame, thr: float) -> list[str]:
    """Return numeric columns with variance > thr (ddof=0)."""
    if X.empty:
        return []
    num = X.select_dtypes(include=[np.number])
    if num.empty:
        return []
    v = num.var(ddof=0)
    return v.index[v > thr].tolist()


def corr_prune_cols(X: pd.DataFrame, cols: list[str], rho: float) -> list[str]:
    """Greedy prune: drop columns with |r| ≥ rho, preferring higher-variance first."""
    if not cols:
        return []
    sub = X[cols]
    order = sub.var(ddof=0).sort_values(ascending=False).index.tolist()
    C = sub[order].corr(min_periods=1).abs()
    keep: list[str] = []
    for c in order:
        if any(C.loc[c, k] >= rho for k in keep if c in C.index and k in C.columns):
            continue
        keep.append(c)
    return keep


def rebalance_train(X: pd.DataFrame, y: pd.Series, method: str | None = None, random_state: int = 0):
    """Rebalance train only. method: None | 'ros' | 'rus' | 'smote'."""
    if method is None:
        return X, y
    n_pos = int(y.sum()); n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return X, y

    rng = np.random.default_rng(random_state)
    pos_idx = y[y == 1].index.to_numpy()
    neg_idx = y[y == 0].index.to_numpy()

    if method == "ros":
        if n_pos < n_neg:
            add = rng.choice(pos_idx, size=(n_neg - n_pos), replace=True)
            new_idx = np.concatenate([neg_idx, pos_idx, add])
        else:
            add = rng.choice(neg_idx, size=(n_pos - n_neg), replace=True)
            new_idx = np.concatenate([pos_idx, neg_idx, add])
        new_idx = rng.permutation(new_idx)
        return X.loc[new_idx].reset_index(drop=True), y.loc[new_idx].reset_index(drop=True)

    if method == "rus":
        if n_pos < n_neg:
            keep = rng.choice(neg_idx, size=n_pos, replace=False)
            new_idx = np.concatenate([keep, pos_idx])
        else:
            keep = rng.choice(pos_idx, size=n_neg, replace=False)
            new_idx = np.concatenate([keep, neg_idx])
        new_idx = rng.permutation(new_idx)
        return X.loc[new_idx].reset_index(drop=True), y.loc[new_idx].reset_index(drop=True)

    if method == "smote":
        try:
            from imblearn.over_sampling import SMOTE
            X_imp = X.fillna(X.median(numeric_only=True))
            sm = SMOTE(random_state=random_state)
            X_res, y_res = sm.fit_resample(X_imp, y)
            return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
        except Exception:
            # Fall back silently to ROS if SMOTE is unavailable
            return rebalance_train(X, y, method="ros", random_state=random_state)

    raise ValueError("method must be None | 'ros' | 'rus' | 'smote'")

def fit_user_scalers(train_df: pd.DataFrame, user_col: str, feature_cols: list[str]):
    """Fit per-user min/max on train; returns {uid: (mins, maxs)}."""
    scalers = {}
    for uid, g in train_df.groupby(user_col):
        scalers[uid] = (g[feature_cols].min(skipna=True), g[feature_cols].max(skipna=True))
    return scalers

def apply_user_scalers(df: pd.DataFrame, user_col: str, feature_cols: list[str], scalers: dict):
    """Apply per-user min–max scaling; preserve NaNs; constant features → 0."""
    out = df.copy()
    for uid, g in out.groupby(user_col):
        if uid not in scalers:
            continue
        mins, maxs = scalers[uid]
        block = g[feature_cols]
        denom = (maxs - mins).replace(0, np.nan)
        scaled = (block - mins) / denom
        scaled = scaled.fillna(0).where(block.notna(), np.nan)
        out.loc[g.index, feature_cols] = scaled
    return out



