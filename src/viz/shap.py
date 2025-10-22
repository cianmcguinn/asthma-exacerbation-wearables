from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap

def shap_topk_plots(booster, X_test: pd.DataFrame, save_dir: Path, base_name: str, topk: int = 15):
    """Bar + beeswarm; displays inline, saves quietly."""
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    feat_names = X_test.columns.tolist()
    dtest = xgb.DMatrix(X_test, feature_names=feat_names)
    contribs = booster.predict(dtest, pred_contribs=True)
    shap_df  = pd.DataFrame(contribs, columns=feat_names + ["bias"])
    mean_abs = shap_df.drop(columns=["bias"]).abs().mean().sort_values(ascending=False)
    top_feats = mean_abs.head(topk)
    top_feats.to_csv(save_dir / f"{base_name}_SHAP_Top{topk}.csv")

    # --- Bar (static)
    plt.figure(figsize=(6, 4))
    top_feats.iloc[::-1].plot(kind="barh")
    plt.xlabel("Mean |SHAP|"); plt.tight_layout()
    plt.savefig(save_dir / f"{base_name}_SHAP_Bar.png", dpi=300)
    plt.savefig(save_dir / f"{base_name}_SHAP_Bar.pdf", dpi=300)
    plt.show(); plt.close()

    # --- Beeswarm (pin size; avoid flicker in Colab)
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="dot", max_display=topk, show=False)
    plt.gcf().set_size_inches(6, 4)
    plt.tight_layout()
    plt.savefig(save_dir / f"{base_name}_SHAP_Beeswarm.png", dpi=300)
    plt.savefig(save_dir / f"{base_name}_SHAP_Beeswarm.pdf", dpi=300)
    plt.show(); plt.close()

def shap_family_plot(booster, X_test: pd.DataFrame, save_dir: Path, base_name: str):
    """Family breakdown bar; displays inline, saves quietly."""
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    feat_names = X_test.columns.tolist()
    dtest = xgb.DMatrix(X_test, feature_names=feat_names)
    contribs = booster.predict(dtest, pred_contribs=True)
    shap_df  = pd.DataFrame(contribs, columns=feat_names + ["bias"])
    mean_abs = shap_df.drop(columns=["bias"]).abs().mean()

    def feature_family(name: str) -> str:
        if name.startswith(("mean_hr_", "max_hr_", "min_hr_")): return "HR level"
        if name.startswith(("std_hr_", "rmssd_hr_", "abrupt_hr_changes_")): return "HR variability"
        if name.startswith(("hr_night_", "hr_morning_", "hr_afternoon_", "hr_evening_")): return "Circadian HR"
        if name.startswith("frac_"): return "Activity fractions"
        if name.startswith("steps_total_"): return "Steps"
        if name.startswith("hr_coverage_"): return "Coverage"
        return "Other"

    fam_df = mean_abs.rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})
    fam_df["family"] = fam_df["feature"].map(feature_family)
    fam_sum = fam_df.groupby("family", dropna=False)["mean_abs_shap"].sum().sort_values(ascending=False)
    fam_pct = 100 * fam_sum / fam_sum.sum()

    plt.figure(figsize=(6.2, 3.8))
    fam_pct.iloc[::-1].plot(kind="barh")
    plt.xlabel("Contribution to total |SHAP| (%)"); plt.ylabel("Feature family")
    plt.grid(axis="x", alpha=0.3, linewidth=0.5); plt.tight_layout()
    plt.savefig(save_dir / f"{base_name}_SHAP_Family.png", dpi=300)
    plt.savefig(save_dir / f"{base_name}_SHAP_Family.pdf", dpi=300)
    plt.show()

    pd.DataFrame({"family": fam_sum.index, "sum_mean_abs_shap": fam_sum.values, "percent": fam_pct.values}) \
      .to_csv(save_dir / f"{base_name}_SHAP_Family.csv", index=False)
