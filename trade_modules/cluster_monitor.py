"""Portfolio-level correlated-cluster exposure monitor.

Reuses PortfolioRiskAnalyzer.flag_correlation_clusters to detect groups of
mutually correlated holdings, sums each group's portfolio weight, and raises a
SOFT/HARD alert when a single correlated bloc exceeds the configured caps.

This is a MONITOR, not a sizer: it never changes positions. It exists so a
deliberate concentration bet (e.g. a mega-cap AI cluster) stays visible and
alarmed rather than drifting silently. Caps default to 30%/35% (the rule both
debate camps signed in the 2026-05-30 review).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from trade_modules.portfolio_risk import PortfolioRiskAnalyzer

DEFAULT_THRESHOLD = 0.70  # pairwise |corr| to be "in" a cluster
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_SOFT_PCT = 30.0
DEFAULT_HARD_PCT = 35.0


def compute_cluster_exposure(
    weights: dict[str, float],
    returns: pd.DataFrame,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> list[dict[str, Any]]:
    """Return correlated clusters among holdings with their summed weight.

    weights: {ticker: weight} as fractions (0.15) or percents (15) — normalized
             internally to percent for the output. Format is auto-detected by
             magnitude: sum(|w|) <= 1.5 is treated as fractions (scaled x100).
             A leveraged/short book whose fractions sum above 1.5 will be
             mis-scaled (under-reported) — pass already-percent weights then.
    returns: daily returns DataFrame, one column per held ticker.
    Returns: list of {tickers, combined_weight_pct, avg_correlation}.
    """
    cols = [t for t in returns.columns if t in weights]
    if len(cols) < min_cluster_size:
        return []
    corr = returns[cols].corr()
    clusters = PortfolioRiskAnalyzer().flag_correlation_clusters(
        corr, min_cluster_size=min_cluster_size, threshold=threshold
    )

    total = sum(abs(w) for w in weights.values()) or 1.0
    scale = 100.0 if total <= 1.5 else 1.0  # fractions -> percent

    out: list[dict[str, Any]] = []
    for cl in clusters:
        tickers = cl.get("tickers", [])
        combined = sum(weights.get(t, 0.0) for t in tickers) * scale
        out.append(
            {
                "tickers": tickers,
                "combined_weight_pct": round(combined, 2),
                "avg_correlation": cl.get("avg_correlation"),
            }
        )
    return out


def check_cluster_alerts(
    exposures: list[dict[str, Any]],
    *,
    soft_pct: float = DEFAULT_SOFT_PCT,
    hard_pct: float = DEFAULT_HARD_PCT,
) -> list[dict[str, Any]]:
    """Raise SOFT (>=soft_pct) / HARD (>=hard_pct) alerts per cluster."""
    alerts: list[dict[str, Any]] = []
    for ex in exposures:
        w = ex.get("combined_weight_pct", 0.0)
        level = "HARD" if w >= hard_pct else "SOFT" if w >= soft_pct else None
        if level:
            limit = hard_pct if level == "HARD" else soft_pct
            alerts.append(
                {
                    "level": level,
                    "tickers": ex["tickers"],
                    "combined_weight_pct": w,
                    "limit_pct": limit,
                    "message": (
                        f"{level}: correlated cluster {ex['tickers']} = {w:.1f}% of book "
                        f"(limit {limit:.0f}%)"
                    ),
                }
            )
    return alerts
