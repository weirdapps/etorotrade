"""
Per-Cell Conviction Calibration (CIO v17 op #4)

The v17 review's headline finding was Spearman ρ(conviction, α30 | signal=B)
≈ −0.002 over the aggregate sample. But the aggregate pools across
tier × regime × consensus_band cells. The cell-level rank correlation
might be much stronger: maybe conviction ranks within MEGA-cap BUYs in
RISK_ON but is noise in MICRO-cap SELL signals.

This module slices the historical concordance + forward-returns data
into cells and computes per-cell ρ. Cells with significant ranking
power feed back into the synthesis as a conviction-confidence
multiplier — i.e. when we're scoring a stock that lands in a
high-IC cell, we trust the conviction; in a low-IC cell, we damp it.

This is the most likely path to *immediately* better recommendations
without waiting for the 6-month sample.
"""

import json
import logging
import math
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CELLS_PATH = Path.home() / ".weirdapps-trading" / "committee" / "conviction_cells.json"


def _spearman_rho(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation. Returns None for n<3 or zero variance."""
    n = len(xs)
    if n < 3 or len(ys) != n:
        return None

    def _rank(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n)) / n
    vx = sum((r - mx) ** 2 for r in rx) / n
    vy = sum((r - my) ** 2 for r in ry) / n
    if vx <= 0 or vy <= 0:
        return None
    return round(cov / math.sqrt(vx * vy), 3)


# ── Cell axes ───────────────────────────────────────────────────────────


def _cap_tier(market_cap_billions: float | None) -> str:
    if not market_cap_billions or market_cap_billions <= 0:
        return "UNKNOWN"
    if market_cap_billions >= 500:
        return "MEGA"
    if market_cap_billions >= 100:
        return "LARGE"
    if market_cap_billions >= 10:
        return "MID"
    if market_cap_billions >= 2:
        return "SMALL"
    return "MICRO"


def _consensus_band(buy_pct: float | None) -> str:
    if buy_pct is None:
        return "UNKNOWN"
    if buy_pct >= 90:
        return "EXTREME"  # crowded
    if buy_pct >= 75:
        return "HIGH"
    if buy_pct >= 60:
        return "MODERATE"
    return "LOW"


def _conv_band(conviction: float | None) -> str:
    if conviction is None:
        return "UNKNOWN"
    if conviction >= 70:
        return "≥70"
    if conviction >= 60:
        return "60-69"
    if conviction >= 55:
        return "55-59"
    if conviction >= 45:
        return "45-54"
    return "<45"


def _safe_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _safe_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── Cell aggregation ────────────────────────────────────────────────────


def compute_cells(
    history: Iterable[dict[str, Any]],
    forward_returns: dict[str, dict[str, float]],
    horizon: str = "T+30",
    min_cell_n: int = 5,
) -> dict[str, Any]:
    """
    Compute Spearman ρ(conviction, α) per cell.

    Cells are defined as (signal × tier × regime × consensus_band).
    Cells with fewer than `min_cell_n` observations are reported but
    flagged as low-evidence; the synthesis will fall back to "trust the
    aggregate" for those.

    Args:
        history: iterable of {date, concordance: [...]} (committee_backtester
            output). Each concordance row should carry: ticker, signal,
            conviction, market_cap_str, buy_pct, regime (optional).
        forward_returns: {ticker:date → {T+N_alpha: float}} from
            CommitteeBacktester.
        horizon: which alpha horizon to evaluate.
        min_cell_n: minimum observations to report a non-flagged ρ.

    Returns:
        {
          generated_at, horizon, total_observations,
          aggregate_spearman, aggregate_n,
          cells: {cell_key: {n, spearman, mean_alpha, conviction_quartiles}},
          high_ic_cells: [cell_key, ...],     # |ρ| ≥ 0.3
          low_ic_cells:  [cell_key, ...],     # |ρ| < 0.10
          recommended_action: short text
        }
    """
    alpha_key = f"{horizon}_alpha"
    # cell_key → list of (conviction, alpha)
    cell_pairs: dict[tuple[str, str, str, str], list[tuple[float, float]]] = defaultdict(list)
    all_pairs: list[tuple[float, float]] = []

    for entry in history:
        date_str = entry.get("date", "")
        regime = entry.get("regime", "UNKNOWN")
        for stock in entry.get("concordance", []):
            if not isinstance(stock, dict):
                continue
            ticker = stock.get("ticker", "")
            if not ticker:
                continue
            sig = stock.get("signal", "?")
            conv = _safe_float(stock.get("conviction"))
            if conv is None:
                continue

            # Lookup alpha
            alpha = (forward_returns.get(f"{ticker}:{date_str}", {}) or {}).get(alpha_key)
            if alpha is None:
                continue

            # Cell axes
            cap_str = stock.get("market_cap_str", "") or stock.get("cap", "")
            cap_b = _market_cap_to_billions(cap_str)
            tier = _cap_tier(cap_b)
            consensus = _consensus_band(_safe_float(stock.get("buy_pct")))
            entry_regime = stock.get("regime", regime)

            cell_key = (sig, tier, entry_regime, consensus)
            cell_pairs[cell_key].append((conv, float(alpha)))
            all_pairs.append((conv, float(alpha)))

    # Compute per-cell
    cells_out: dict[str, dict[str, Any]] = {}
    high_ic: list[str] = []
    low_ic: list[str] = []
    for (sig, tier, regime, consensus), pairs in cell_pairs.items():
        n = len(pairs)
        cell_key = f"{sig}|{tier}|{regime}|{consensus}"
        if n < 3:
            cells_out[cell_key] = {
                "n": n,
                "spearman": None,
                "mean_alpha": None,
                "verdict": "INSUFFICIENT",
            }
            continue
        convs, alphas = zip(*pairs, strict=False)
        sp = _spearman_rho(list(convs), list(alphas))
        mean_alpha = round(sum(alphas) / n, 2)
        verdict = "EVIDENCE" if n >= min_cell_n else "LOW_N"
        cells_out[cell_key] = {
            "n": n,
            "spearman": sp,
            "mean_alpha": mean_alpha,
            "verdict": verdict,
        }
        if sp is not None and abs(sp) >= 0.3 and n >= min_cell_n:
            high_ic.append(cell_key)
        elif sp is not None and abs(sp) < 0.10 and n >= min_cell_n:
            low_ic.append(cell_key)

    # Aggregate
    agg_sp = None
    agg_alpha = None
    if all_pairs:
        ac, aa = zip(*all_pairs, strict=False)
        agg_sp = _spearman_rho(list(ac), list(aa))
        agg_alpha = round(sum(aa) / len(aa), 2)

    recommendation = _summarise_recommendation(high_ic, low_ic, agg_sp, len(all_pairs))

    output = {
        "generated_at": datetime.now().isoformat(),
        "horizon": horizon,
        "total_observations": len(all_pairs),
        "aggregate_spearman": agg_sp,
        "aggregate_mean_alpha": agg_alpha,
        "aggregate_n": len(all_pairs),
        "cells": cells_out,
        "high_ic_cells": sorted(high_ic),
        "low_ic_cells": sorted(low_ic),
        "min_cell_n": min_cell_n,
        "recommendation": recommendation,
    }
    return output


def _market_cap_to_billions(s: str) -> float | None:
    if not s:
        return None
    s = s.strip().upper()
    if s.endswith("T"):
        return _safe_float(s[:-1]) and float(s[:-1]) * 1000
    if s.endswith("B"):
        return _safe_float(s[:-1]) and float(s[:-1])
    if s.endswith("M"):
        return _safe_float(s[:-1]) and float(s[:-1]) / 1000
    return _safe_float(s)


def _summarise_recommendation(
    high_ic: list[str],
    low_ic: list[str],
    agg: float | None,
    n: int,
) -> str:
    if n < 50:
        return f"Insufficient evidence yet (n={n}). Re-run weekly until n≥200."
    parts = []
    if high_ic:
        parts.append(
            f"{len(high_ic)} high-IC cells (|ρ|≥0.3) — boost conviction " f"weight in these cells."
        )
    if low_ic:
        parts.append(
            f"{len(low_ic)} low-IC cells (|ρ|<0.10) — damp conviction in "
            f"these cells (treat as a coin flip)."
        )
    if agg is not None:
        parts.append(f"Aggregate ρ={agg:+.3f} (n={n}).")
    if not parts:
        parts.append("No actionable cells yet.")
    return " ".join(parts)


def cell_confidence_multiplier(
    signal: str,
    market_cap_billions: float | None,
    regime: str,
    buy_pct: float | None,
    cells_data: dict[str, Any],
) -> float:
    """
    Resolve the cell containing this stock and return a multiplier in
    [0.5, 1.5] that the synthesis can apply to conviction.

    * High-IC cell (|ρ|≥0.3, sign matches sign of expected impact) → 1.2
    * Low-IC cell (|ρ|<0.10, n≥5) → 0.7
    * Otherwise → 1.0
    """
    if not cells_data or not cells_data.get("cells"):
        return 1.0
    tier = _cap_tier(market_cap_billions)
    consensus = _consensus_band(buy_pct)
    cell_key = f"{signal}|{tier}|{regime}|{consensus}"
    cell = (cells_data.get("cells") or {}).get(cell_key)
    if not cell or cell.get("verdict") == "INSUFFICIENT":
        return 1.0
    sp = cell.get("spearman")
    if sp is None:
        return 1.0
    n = cell.get("n", 0)
    if n < 5:
        return 1.0
    if abs(sp) >= 0.3:
        return 1.2
    if abs(sp) < 0.10:
        return 0.7
    return 1.0


def persist_cells(
    cells_output: dict[str, Any],
    path: Path | None = None,
) -> Path:
    out = path or DEFAULT_CELLS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cells_output, indent=2, default=str))
    return out


def load_cells(path: Path | None = None) -> dict[str, Any] | None:
    p = path or DEFAULT_CELLS_PATH
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return None
