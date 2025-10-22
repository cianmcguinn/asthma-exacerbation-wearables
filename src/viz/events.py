from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def hr_trends_around_events(
    passive_daily: pd.DataFrame,
    usage_daily: pd.DataFrame,
    baseline_by_user: pd.Series,
    save_dir: Path | None = None,
    rel_days: range = range(-5, 6),           # t-5 … t+5
    min_days_in_window: int = 6,              # require this many HR days per event window
    dpi: int = 300,
) -> Tuple[pd.DataFrame | None, dict[str, Path]]:
    """
    Compute HR delta trends around exacerbation starts and return (summary_df, saved_paths).
    - Shows nothing by itself; caller can display the returned figure with plt.show().
    - Saves PNG/PDF silently if save_dir is provided.
    """
    # ---- find event starts (first day in any 2-day run over baseline)
    use_df = usage_daily.copy()
    use_df["user_key"] = use_df["user_key"].astype(str)
    use_df["date"]     = pd.to_numeric(use_df["date"], errors="coerce").astype("Int64")
    use_df["use"]      = pd.to_numeric(use_df["use"],  errors="coerce")
    use_df = use_df.merge(baseline_by_user.rename("baseline"),
                          left_on="user_key", right_index=True, how="left")
    use_df["elevated"] = (use_df["use"] > use_df["baseline"]).astype("Int8")

    def _first_day_of_run2(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").reset_index(drop=True)
        starts = (g["elevated"].eq(1) & g["elevated"].shift(-1, fill_value=0).eq(1))
        idx = np.flatnonzero(starts.values)
        keep, last = [], -10**9
        for i in idx:
            if i - last >= 2:
                keep.append(i); last = i
        out = g.loc[keep, ["user_key", "date"]].copy()
        out["event"] = 1
        return out

    events = (
        use_df.dropna(subset=["date"])
              .groupby("user_key", group_keys=False)
              .apply(_first_day_of_run2)
              .reset_index(drop=True)
    )
    if events.empty: return None, {}

    # ---- HR delta (daily mean HR minus per-user median)
    hr_df = passive_daily[["user_key", "date", "mean_hr"]].copy()
    hr_df["user_key"] = hr_df["user_key"].astype(str)
    hr_df["date"]     = pd.to_numeric(hr_df["date"], errors="coerce").astype("Int64")
    hr_df["mean_hr"]  = pd.to_numeric(hr_df["mean_hr"], errors="coerce")
    hr_base = hr_df.groupby("user_key")["mean_hr"].median().rename("hr_baseline")
    hr_df = hr_df.merge(hr_base, left_on="user_key", right_index=True, how="left")
    hr_df["hr_delta"] = hr_df["mean_hr"] - hr_df["hr_baseline"]

    rel = np.array(list(rel_days), dtype=int)
    hr_idx = hr_df.set_index(["user_key", "date"])["hr_delta"]

    rows = []
    for uid, t0 in events[["user_key", "date"]].itertuples(index=False):
        for k in rel:
            rows.append({"user_key": uid, "event_date": int(t0), "rel_day": int(k),
                         "hr_delta": hr_idx.get((uid, int(t0) + int(k)), np.nan)})
    trend = pd.DataFrame(rows)

    ok = (trend.groupby(["user_key", "event_date"])["hr_delta"]
                .apply(lambda s: s.notna().sum() >= int(min_days_in_window)))
    keep_pairs = set(ok[ok].index.tolist())
    trend = trend[trend.set_index(["user_key", "event_date"]).index.isin(keep_pairs)]
    if trend.empty: return None, {}

    trend["hr_delta"] = pd.to_numeric(trend["hr_delta"], errors="coerce")
    summary = (trend.groupby("rel_day")["hr_delta"]
                    .agg(mean=lambda s: float(np.nanmean(s)),
                         n=lambda s: int(np.sum(~np.isnan(s))),
                         sd=lambda s: float(np.nanstd(s, ddof=1)))
                    .reset_index())
    summary["se"] = np.where(summary["n"] > 1, summary["sd"] / np.sqrt(summary["n"]), 0.0)
    summary["lo"] = summary["mean"] - 1.96 * summary["se"]
    summary["hi"] = summary["mean"] + 1.96 * summary["se"]
    summary = summary[["rel_day", "mean", "lo", "hi", "n"]]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(summary["rel_day"], summary["mean"], linewidth=2)
    ax.fill_between(summary["rel_day"], summary["lo"], summary["hi"], alpha=0.2, linewidth=0)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Days relative to exacerbation start (t=0)")
    ax.set_ylabel("Δ Heart rate vs user median (bpm)")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()

    saved: dict[str, Path] = {}
    if save_dir is not None:
        save_dir = Path(save_dir)
        png = save_dir / "Figure4_HR_trends.png"
        pdf = save_dir / "Figure4_HR_trends.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf, dpi=300, bbox_inches="tight")
        saved = {"png": png, "pdf": pdf}

    return summary, saved
