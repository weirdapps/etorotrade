"""Panel-history per-factor alpha backtest — which parameters actually have alpha.

The etoro.csv panel is git-committed daily, so its history is a de-facto point-in-time
panel (~months). For each historical snapshot we compute each factor's directional
cross-sectional rank-z (the SAME transform the live combiner uses) and its forward
return, then measure per-factor / per-cluster forward IC + incremental (spanning) IC.

This covers the panel-native factors (PIT from the CSV) plus price-derived momentum /
realized-vol (PIT from adjusted prices). It deliberately INCLUDES the discarded panel
factors (upside, buy_pct, pe_forward, …) so their discard decisions can be validated.
It excludes the yfinance-.info factors (ev_ebitda, rev_growth) — those are current-only,
not reconstructable point-in-time.

Pure helpers here; the git-history + price I/O + orchestration live in
``scripts/v3_factor_backtest.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade_modules.v3.combine import CLUSTERS, DIRECTION, _rank_z, _sector_demean
from trade_modules.v3.features import _num

# Panel columns backtestable point-in-time -> feature name (for the DIRECTION sign).
# Active (scored) + discarded panel factors, so the discards can be tested too.
BACKTEST_PANEL_FACTORS: dict[str, str] = {
    # active, panel-native (also in the live clusters)
    "PET": "pe_trailing",
    "ROE": "roe",
    "FCF": "fcf",
    "52W": "pct_52w_high",
    "B": "beta",
    "AM": "analyst_mom",
    "EG": "earn_growth",
    # discarded from scoring — tested here to CHECK the discard was right
    "PEF": "pe_forward",
    "P/S": "ps_sector",
    "PEG": "peg",
    "DE": "de",
    "PP": "price_perf",
    "DV": "div_yield",
    "SI": "short_interest",
    "UP%": "upside",
    "%B": "buy_pct",
}

# Which feature names are ACTIVE (in a live cluster) vs discarded.
_ACTIVE_FEATURES = {m for members in CLUSTERS.values() for m in members}


def factor_zscore(
    raw: pd.Series, feature_name: str, sector: pd.Series | None = None, sector_neutral: bool = True
) -> pd.Series:
    """Directional cross-sectional rank-z of ONE raw factor column (one snapshot).

    Mirrors the live combiner: clean → van-der-Waerden rank-z → multiply by the
    factor's DIRECTION (+1 keep / -1 low-is-good) → optional within-sector demean.
    """
    z = _rank_z(_num(pd.Series(raw))) * DIRECTION.get(feature_name, 1)
    if sector_neutral and sector is not None:
        z = _sector_demean(z, pd.Series(sector).reindex(z.index))
    return z


def forward_return_at(prices: pd.DataFrame, as_of, horizon: int) -> pd.Series:
    """Per-ticker forward return from ``as_of`` over ``horizon`` trading bars.

    ``prices`` is a dates×tickers adjusted-close frame. Returns a Series indexed by
    ticker (NaN where the as_of or the +horizon bar is missing). Empty if as_of is
    not in the index or the horizon bar runs past the data.
    """
    idx = prices.index
    if as_of not in idx:
        return pd.Series(dtype=float)
    i = idx.get_loc(as_of)
    j = i + horizon
    if j >= len(idx):
        return pd.Series(dtype=float)
    base = prices.iloc[i]
    fut = prices.iloc[j]
    fr = (fut / base) - 1.0
    return fr.replace([np.inf, -np.inf], np.nan)


def summarize_ic(ic_by_date: pd.Series) -> dict:
    """Summary stats of a per-date IC series: n, mean, std, t-stat, hit-rate."""
    ic = pd.to_numeric(pd.Series(ic_by_date), errors="coerce").dropna()
    n = int(len(ic))
    mean = float(ic.mean()) if n else float("nan")
    sd = float(ic.std(ddof=1)) if n > 1 else float("nan")
    t = mean / (sd / np.sqrt(n)) if (n > 1 and sd and sd > 0) else float("nan")
    hit = float((ic > 0).mean()) if n else float("nan")
    return {"n": n, "mean_ic": mean, "ic_std": sd, "t_stat": t, "hit_rate": hit}


def is_active(feature_name: str) -> bool:
    """True if the feature is in a live scoring cluster (else it is a discarded factor)."""
    return feature_name in _ACTIVE_FEATURES


def beta_neutralize(fwd: pd.Series, beta: pd.Series) -> pd.Series:
    """Cross-sectional residual of forward returns on beta.

    Removes the market-beta component that dominates a trending regime (in a melt-up,
    high-beta names win, which makes any low-beta-favouring factor look anti-predictive
    on RAW returns). Per date: OLS ``r ~ a + b*beta``; return the residual, i.e. the
    alpha net of a name's beta exposure. NaN-safe; <3 clean points -> all-NaN.
    """
    df = pd.DataFrame(
        {"r": pd.to_numeric(fwd, errors="coerce"), "b": pd.to_numeric(beta, errors="coerce")}
    ).dropna()
    if len(df) < 3:
        return pd.Series(np.nan, index=fwd.index)
    x = np.column_stack([np.ones(len(df)), df["b"].to_numpy(dtype=float)])
    coef, *_ = np.linalg.lstsq(x, df["r"].to_numpy(dtype=float), rcond=None)
    resid = df["r"].to_numpy(dtype=float) - x @ coef
    return pd.Series(resid, index=df.index).reindex(fwd.index)


def sector_neutralize(fwd: pd.Series, sector: pd.Series) -> pd.Series:
    """Demean forward returns within sector (removes sector-level moves)."""
    r = pd.to_numeric(fwd, errors="coerce")
    grp = pd.Series(sector).reindex(r.index).fillna("__NA__").astype(str)
    return r - r.groupby(grp).transform("mean")


def ic_by_sector(
    df: pd.DataFrame,
    z_col: str,
    fwd_col: str,
    sector_col: str,
    *,
    date_col: str = "date",
    min_names_per_date: int = 8,
) -> dict[str, dict]:
    """Forward IC of one factor computed WITHIN each sector separately.

    Answers "is this the right metric for this sector?" — e.g. whether P/E predicts
    inside Financials as well as it does inside Technology. For each sector we keep
    only date-slices with at least ``min_names_per_date`` names (a cross-sectional
    rank needs a populated slice), then compute the per-date Spearman IC and average.

    Returns ``{sector: {mean_ic, t_stat, n_dates, n_obs}}`` (thin sectors dropped).
    """
    from trade_modules.validation.xsection_ic import cross_sectional_ic

    out: dict[str, dict] = {}
    for sec, g in df.groupby(sector_col):
        counts = g.groupby(date_col)[z_col].transform("size")
        g2 = g[counts >= min_names_per_date].dropna(subset=[z_col, fwd_col])
        if g2.empty or g2[date_col].nunique() < 1:
            continue
        ic = cross_sectional_ic(g2, z_col, fwd_col, date_col=date_col)
        out[str(sec)] = {
            "mean_ic": ic.get("mean_ic"),
            "t_stat": ic.get("t_stat"),
            "n_dates": ic.get("n_dates"),
            "n_obs": int(len(g2)),
        }
    return out
