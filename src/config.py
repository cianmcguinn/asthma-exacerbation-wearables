# src/repro.py
from __future__ import annotations
from dataclasses import dataclass
from importlib.metadata import version, PackageNotFoundError
from typing import Iterable, Optional
import sys, random, numpy as np, pandas as pd

# ---- seeds -------------------------------------------------------------
def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

# ---- tables ------------------------------------------------------------
def config_summary(cfg, seed: Optional[int] = None) -> pd.DataFrame:
    rows = [
        ("Seed", seed),
        ("Lookback / Buffer / Horizon (days)", f"{cfg.LOOKBACK} / {cfg.BUFFER} / {cfg.HORIZON}"),
        ("Outcome source", cfg.OUTCOME_SOURCE),
        ("Outcome definition", cfg.OUTCOME_DEF),
        ("Threshold policy", f"{cfg.THRESHOLD_POLICY}" +
         (f" (target={cfg.THRESHOLD_TARGET})" if cfg.THRESHOLD_TARGET else "")),
        ("Imbalance method", cfg.IMBALANCE or "None"),
        ("Scaling", cfg.SCALING or "None"),
        ("HR min coverage (fraction of valid minutes/day)", f"{float(cfg.HR_MIN_COVERAGE):.1f}"),
        ("FFILL limit (passive, days)", cfg.FFILL_LIMIT_DAYS),
    ]
    return pd.DataFrame(rows, columns=["Setting", "Value"])

def env_report_df(packages: Iterable[str] = ("numpy","pandas","scikit-learn","xgboost","matplotlib","scipy","shap")) -> pd.DataFrame:
    rows = [("Python", sys.version.split()[0])]
    for p in packages:
        try:
            rows.append((p, version(p)))
        except PackageNotFoundError:
            rows.append((p, "(not installed)"))
    return pd.DataFrame(rows, columns=["Package","Version"]).sort_values("Package")

def write_min_requirements(df: pd.DataFrame, path: str) -> str:
    # expects the dataframe from env_report_df()
    lines = [f"{pkg}=={ver}" for pkg, ver in df[df["Package"]!="Python"].itertuples(index=False)]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path

# ---- notebook renderer (optional) -------------------------------------
def render_side_by_side(cfg, seed: Optional[int] = None):
    """Inline HTML for Jupyter/Colab; no effect outside notebooks."""
    from IPython.display import HTML, display
    left_html = (config_summary(cfg, seed)
        .style.hide(axis="index").set_caption("Run configuration")
        .set_table_attributes('class="repotbl"')
        .set_properties(subset=["Value"], **{"text-align": "right"})
        .to_html())

    right_html = (env_report_df()
        .style.hide(axis="index").set_caption("Pinned packages")
        .set_table_attributes('class="repotbl"')
        .set_properties(subset=["Version"], **{"font-family":"monospace","text-align":"right"})
        .to_html())

    css = """
    <style>
      .repocont { display:flex; gap:24px; align-items:flex-start; }
      .repocard { border:1px solid #e5e7eb; border-radius:8px; padding:10px; flex:1; min-width:320px; background:#fff; }
      .repotbl  { border-collapse:collapse; width:100%; }
      .repotbl th, .repotbl td { border:1px solid #e5e7eb; padding:6px 8px; }
      .repotbl caption { caption-side: top; text-align:left; font-weight:600; padding:4px 0 8px; }
    </style>
    """
    display(HTML(f"""{css}
    <div class="repocont">
      <div class="repocard">{left_html}</div>
      <div class="repocard">{right_html}</div>
    </div>"""))
