"""
Adversarial Debate Effectiveness Scorecard (CIO v17 op #5)

The committee runs Bull vs Bear advocate debates (CIO v27.0). Each
debate emits a `debate_signal` and the synthesis applies waterfall
keys like `debate_strengthen_bull`, `debate_weaken_bull`. We track
*whether* the debate fires but never measure whether the adjustments
produce alpha.

This module builds a scorecard answering: "Did debate-driven
adjustments improve realised α at T+30 vs same-conviction stocks
without debate?"

If the answer is "no" or "marginal", we should consider killing the
adversarial debate path — it's the most expensive component (Round 1 +
Round 2 each consume Opus tokens per contentious stock).
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = (
    Path.home() / ".weirdapps-trading" / "committee" / "debate_scorecard.json"
)


def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _waterfall_value(wf: Dict[str, Any], key: str) -> Optional[int]:
    if not wf:
        return None
    v = wf.get(key)
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def compute_debate_scorecard(
    history: Iterable[Dict[str, Any]],
    forward_returns: Dict[str, Dict[str, float]],
    horizon: str = "T+30",
) -> Dict[str, Any]:
    """
    Aggregate by debate_signal across all historical concordances.

    For each debate signal (STRENGTHEN_BULL, WEAKEN_BULL,
    STRENGTHEN_BEAR, WEAKEN_BEAR, DEADLOCK), report:
      * count
      * mean conviction adjustment applied
      * mean realised α(T+30)
      * hit rate (% positive α for STRENGTHEN_BULL / negative for
        STRENGTHEN_BEAR)
      * comparison vs control (same conviction band, no debate)
    """
    alpha_key = f"{horizon}_alpha"
    by_signal: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    control: List[Dict[str, Any]] = []

    for entry in history:
        date_str = entry.get("date", "")
        for stock in entry.get("concordance", []):
            if not isinstance(stock, dict):
                continue
            ticker = stock.get("ticker", "")
            if not ticker:
                continue
            wf = stock.get("conviction_waterfall") or {}
            alpha = (forward_returns.get(f"{ticker}:{date_str}", {}) or {}).get(alpha_key)
            if alpha is None:
                continue

            # Identify which debate key fired (if any)
            sb = _waterfall_value(wf, "debate_strengthen_bull")
            wb = _waterfall_value(wf, "debate_weaken_bull")
            sbear = _waterfall_value(wf, "debate_strengthen_bear")
            wbear = _waterfall_value(wf, "debate_weaken_bear")

            row = {
                "ticker": ticker,
                "date": date_str,
                "conviction": _safe_float(stock.get("conviction")),
                "alpha": float(alpha),
                "signal": stock.get("signal"),
                "action": stock.get("action"),
            }

            if sb is not None and sb != 0:
                by_signal["STRENGTHEN_BULL"].append({**row, "delta": sb})
            elif wb is not None and wb != 0:
                by_signal["WEAKEN_BULL"].append({**row, "delta": wb})
            elif sbear is not None and sbear != 0:
                by_signal["STRENGTHEN_BEAR"].append({**row, "delta": sbear})
            elif wbear is not None and wbear != 0:
                by_signal["WEAKEN_BEAR"].append({**row, "delta": wbear})
            else:
                control.append(row)

    # Compute scorecard per signal
    out: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "horizon": horizon,
        "control_n": len(control),
        "control_mean_alpha": (
            round(sum(r["alpha"] for r in control) / len(control), 2)
            if control else None
        ),
        "signals": {},
    }

    for sig_name, rows in by_signal.items():
        n = len(rows)
        if n == 0:
            continue
        mean_alpha = sum(r["alpha"] for r in rows) / n
        mean_delta = sum(r["delta"] for r in rows) / n

        # Compute hit rate per signal direction
        if "STRENGTHEN_BULL" in sig_name or "WEAKEN_BEAR" in sig_name:
            # Bullish signal — hit if alpha > 0
            hit_rate = sum(1 for r in rows if r["alpha"] > 0) / n
        else:
            # Bearish signal — hit if alpha < 0
            hit_rate = sum(1 for r in rows if r["alpha"] < 0) / n

        # Compare vs control
        control_alpha = out["control_mean_alpha"] or 0
        excess_alpha = round(mean_alpha - control_alpha, 2)

        out["signals"][sig_name] = {
            "count": n,
            "mean_conviction_delta": round(mean_delta, 2),
            "mean_alpha": round(mean_alpha, 2),
            "hit_rate": round(hit_rate, 3),
            "excess_alpha_vs_control": excess_alpha,
        }

    # Verdict
    out["verdict"] = _summarise_verdict(out["signals"], out["control_n"])
    return out


def _summarise_verdict(signals: Dict[str, Any], control_n: int) -> str:
    if not signals or control_n < 30:
        return "INSUFFICIENT_EVIDENCE — need ≥30 control + 5/signal"

    # Average excess alpha across all signals (weighted)
    total_n = sum(s.get("count", 0) for s in signals.values())
    if total_n == 0:
        return "INSUFFICIENT_EVIDENCE"
    weighted_excess = sum(
        s.get("excess_alpha_vs_control", 0) * s.get("count", 0)
        for s in signals.values()
    ) / total_n

    if abs(weighted_excess) < 0.5:
        return f"NO_EDGE — weighted excess α = {weighted_excess:+.2f}pp. Consider deprecating adversarial debate (Opus cost not justified)."
    elif weighted_excess > 0:
        return f"POSITIVE_EDGE — weighted excess α = {weighted_excess:+.2f}pp. Debate is adding value."
    else:
        return f"NEGATIVE_EDGE — weighted excess α = {weighted_excess:+.2f}pp. Debate adjustments are HARMFUL — flip the sign or disable."


def persist_scorecard(
    scorecard: Dict[str, Any],
    path: Optional[Path] = None,
) -> Path:
    out = path or DEFAULT_OUTPUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scorecard, indent=2, default=str))
    return out


def load_scorecard(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    p = path or DEFAULT_OUTPUT_PATH
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return None
