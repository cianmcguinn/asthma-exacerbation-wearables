from __future__ import annotations
from typing import Iterable, Mapping
import pandas as pd
from IPython.display import HTML, display

__all__ = ["chips", "cards", "styled_table", "show_table"]

_CSS = """
<style>
  .chips { display:flex; gap:10px; flex-wrap:wrap; margin:6px 0 12px; }
  .chip  { border:1px solid #e5e7eb; border-radius:9999px; padding:4px 10px; font-size:12.5px; background:#fafafa; }
  .cards { display:flex; gap:12px; flex-wrap:wrap; margin:8px 0 10px; }
  .card  { border:1px solid #e5e7eb; border-radius:10px; padding:10px 12px; min-width:220px; background:#fff; }
  .card h4 { margin:0 0 6px 0; font-size:14px; }
  .card .big { font-size:20px; font-weight:600; }
  .tbl { border-collapse:collapse; width:100%; }
  .tbl th, .tbl td { border:1px solid #e5e7eb; padding:6px 8px; }
  .tbl caption { caption-side: top; text-align:left; font-weight:600; padding:4px 0 8px; }
</style>
"""

def chips(*items: str) -> None:
    html = "".join(f'<span class="chip">{x}</span>' for x in items)
    display(HTML(_CSS + f'<div class="chips">{html}</div>'))

def cards(rows: Iterable[Mapping[str, str]]) -> None:
    parts = []
    for r in rows:
        parts.append(
            f'<div class="card"><h4>{r.get("title","")}</h4>'
            f'<div class="big">{r.get("big","")}</div>'
            f'<div>{r.get("sub","")}</div></div>'
        )
    display(HTML(_CSS + f'<div class="cards">{"".join(parts)}</div>'))

def styled_table(df: pd.DataFrame, title: str) -> None:
    fmt_cols = {c: "{:.1f}" for c in df.columns
                if "percent" in str(c).lower() or "completeness" in str(c).lower()}
    html = (
        df.style.hide(axis="index")
          .set_caption(title)
          .set_table_attributes('class="tbl"')
          .format(fmt_cols)
          .to_html()
    )
    display(HTML(_CSS + html))

def show_table(df: pd.DataFrame, caption: str) -> None:
    html = (
        df.style.hide(axis="index")
          .set_caption(caption)
          .set_table_attributes('class="tbl"')
          .to_html()
    )
    display(HTML(_CSS + html))
