"""Baseline strategy comparison — measure signal engine value-add.

Computes returns of simple, well-known strategies over the same period and
universe as the signal engine, enabling apples-to-apples comparison.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_baselines(
    signal_log_path: Path,
    spy_returns: dict[str, float] | None = None,
    horizons: tuple[int, ...] = (7, 30),
) -> dict[str, Any]:
    """Compute baseline strategy returns from the signal log.

    Baselines (all equal-weight, monthly rebalanced):
    1. All BUY signals: every signal-engine BUY, equal weight
    2. All SELL signals: every SELL (should underperform if signals work)
    3. Top-quintile momentum: top 20% by pct_52w_high
    4. Top-quintile value: top 20% by EXRET (upside x buy%)
    5. Random selection: random 20% of universe (expected return = market)
    """
    entries = _load_signal_log(signal_log_path)
    if not entries:
        return {"status": "no_data", "baselines": {}}

    results = {}

    for h in horizons:
        h_key = f"T+{h}"

        # Group by signal
        buy_returns = [
            e[f"return_{h}d"]
            for e in entries
            if e.get("signal") == "B" and e.get(f"return_{h}d") is not None
        ]
        sell_returns = [
            e[f"return_{h}d"]
            for e in entries
            if e.get("signal") == "S" and e.get(f"return_{h}d") is not None
        ]
        hold_returns = [
            e[f"return_{h}d"]
            for e in entries
            if e.get("signal") == "H" and e.get(f"return_{h}d") is not None
        ]
        all_returns = [e[f"return_{h}d"] for e in entries if e.get(f"return_{h}d") is not None]

        # Top-quintile momentum (by pct_52w_high)
        with_momentum = [
            e
            for e in entries
            if e.get("pct_52w_high") is not None and e.get(f"return_{h}d") is not None
        ]
        if with_momentum:
            with_momentum.sort(key=lambda x: x["pct_52w_high"], reverse=True)
            top_q = with_momentum[: max(1, len(with_momentum) // 5)]
            momentum_returns = [e[f"return_{h}d"] for e in top_q]
        else:
            momentum_returns = []

        # Top-quintile value (by EXRET = upside x buy% / 100)
        with_value = [
            e for e in entries if e.get("exret") is not None and e.get(f"return_{h}d") is not None
        ]
        if with_value:
            with_value.sort(key=lambda x: x["exret"], reverse=True)
            top_q_val = with_value[: max(1, len(with_value) // 5)]
            value_returns = [e[f"return_{h}d"] for e in top_q_val]
        else:
            value_returns = []

        results[h_key] = {
            "buy_signals": _stats(buy_returns, "BUY signals"),
            "sell_signals": _stats(sell_returns, "SELL signals"),
            "hold_signals": _stats(hold_returns, "HOLD signals"),
            "all_universe": _stats(all_returns, "Full universe"),
            "top_quintile_momentum": _stats(momentum_returns, "Top-Q momentum"),
            "top_quintile_value": _stats(value_returns, "Top-Q value"),
        }

    return {"status": "ok", "baselines": results}


def _stats(returns: list[float], name: str) -> dict[str, Any]:
    """Compute summary statistics for a list of returns."""
    if not returns:
        return {"name": name, "count": 0}
    arr = np.array(returns)
    return {
        "name": name,
        "count": len(arr),
        "mean_return": round(float(np.mean(arr)), 2),
        "median_return": round(float(np.median(arr)), 2),
        "std_return": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 2),
        "hit_rate": round(float(np.sum(arr > 0) / len(arr) * 100), 1),
        "min_return": round(float(np.min(arr)), 2),
        "max_return": round(float(np.max(arr)), 2),
        "sharpe_proxy": round(
            float(np.mean(arr) / np.std(arr, ddof=1))
            if len(arr) > 1 and np.std(arr, ddof=1) > 0
            else 0.0,
            3,
        ),
    }


def _load_signal_log(path: Path) -> list[dict[str, Any]]:
    """Load signal log entries with forward returns pre-computed."""
    entries = []
    if not path.exists():
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError:
                continue
    return entries
