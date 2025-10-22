# Re-export a convenient surface API
from .curves import plot_pr_roc
from .shap import shap_topk_plots, shap_family_plot
from .bars import plot_grouped_bars, grouped_bars_sensitivity
from .html import chips, cards, styled_table, show_table
from .events import hr_trends_around_events

__all__ = [
    "plot_pr_roc",
    "shap_topk_plots", "shap_family_plot",
    "plot_grouped_bars", "grouped_bars_sensitivity",
    "chips", "cards", "styled_table", "show_table",
    "hr_trends_around_events",
]
