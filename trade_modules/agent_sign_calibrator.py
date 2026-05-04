"""
Agent Sign Calibrator (CIO v17 H1)

Some specialist agents (Macro, in our T+7 sample) have been observed to
correlate negatively with realized α at the system's target horizon. The
correct treatment for an inverted predictor is to flip its sign — not to
reduce its weight. Reducing weight on a wrong signal damps its damage; it
does not extract information.

This module computes, for each agent and each emitted view value,
P(α(T+30) > 0 | view) over a rolling 60-day archive. If a view's P drops
below 0.45 OR rises above 0.55 *for negative directions*, and the
condition holds for ≥2 consecutive evaluations, we flag the agent for a
sign flip.

Default mode is **shadow**: we compute, log, and persist the proposed
flip but do NOT change the synthesis weight. Set `enabled=True` only
after ≥8 weeks of shadow data confirms a stable inversion.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

logger = logging.getLogger(__name__)

DEFAULT_CALIBRATOR_PATH = (
    Path.home() / ".weirdapps-trading" / "committee" / "agent_sign_calibration.json"
)

# View → polarity. Positive polarity = bullish view; negative = bearish.
# Used to know which side of P(α>0) the inversion crosses.
AGENT_POLARITY: dict[str, dict[str, int]] = {
    "fundamental": {"BUY": +1, "HOLD": 0, "SELL": -1},
    "technical": {
        "ENTER_NOW": +1,
        "WAIT_FOR_PULLBACK": 0,
        "WAIT": 0,
        "HOLD": 0,
        "REDUCE": -1,
        "EXIT_SOON": -1,
        "AVOID": -1,
    },
    "macro": {"FAVORABLE": +1, "NEUTRAL": 0, "UNFAVORABLE": -1},
    "census": {"ALIGNED": +1, "NEUTRAL": 0, "DIVERGENT": -1, "CENSUS_DIV": -1},
    "news": {"POSITIVE": +1, "NEUTRAL": 0, "NEGATIVE": -1},
    "risk": {"OK": +1, "WARN": 0, "TRIM": -1, "EXIT": -1},
}


def _load_concordance_history(
    history_dir: Path,
    days: int = 60,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Load (date, concordance) pairs from history/ within the lookback window."""
    if not history_dir.is_dir():
        return []
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    out: list[tuple[str, list[dict[str, Any]]]] = []
    for fpath in sorted(history_dir.glob("concordance-*.json")):
        # Extract YYYY-MM-DD from stem via regex (filename may be either
        # "concordance-YYYY-MM-DD.json" or "concordance-YYYY-MM-DD-suffix.json").
        m = _DATE_RE.search(fpath.stem)
        if not m:
            continue
        date_str = m.group(1)
        if date_str < cutoff:
            continue
        try:
            data = json.load(open(fpath))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("concordance", data.get("stocks", []))
        else:
            continue
        # Normalise legacy {ticker: row} dict format to list-of-rows.
        if isinstance(items, dict):
            items = [dict(v, ticker=k) if isinstance(v, dict) else None for k, v in items.items()]
            items = [r for r in items if r is not None]
        # Filter out non-dict entries (defensive).
        items = [r for r in items if isinstance(r, dict)]
        if items:
            out.append((date_str, items))
    return out


def _alpha_from_returns(
    forward_returns: dict[str, dict[str, float]],
    ticker: str,
    date_str: str,
    horizon: str,
) -> float | None:
    key = f"{ticker}:{date_str}"
    return (forward_returns.get(key, {}) or {}).get(f"{horizon}_alpha")


def calibrate_agent_signs(
    forward_returns: dict[str, dict[str, float]],
    history_dir: Path | None = None,
    horizon: str = "T+30",
    lookback_days: int = 60,
    min_evidence: int = 10,
    flip_lower: float = 0.45,
    flip_upper: float = 0.55,
) -> dict[str, Any]:
    """
    Compute per-agent, per-view P(α(horizon) > 0) and propose sign flips.

    Args:
        forward_returns: dict from CommitteeBacktester.compute_forward_returns
            keyed by "TICKER:YYYY-MM-DD".
        history_dir: where to read concordance archives from.
        horizon: forward-return horizon to evaluate, e.g. "T+30".
        lookback_days: rolling window in calendar days.
        min_evidence: minimum (ticker, date) observations to evaluate.
        flip_lower / flip_upper: P bounds outside which a flip is proposed.

    Returns:
        Dict per agent with per-view counts, P(α>0), polarity-aware
        verdict ("OK" / "INVERTED" / "INSUFFICIENT_DATA"), and the
        recommended sign multiplier (+1 = unchanged, −1 = flip).
    """
    history_dir = history_dir or (Path.home() / ".weirdapps-trading" / "committee" / "history")
    history = _load_concordance_history(history_dir, lookback_days)
    if not history:
        return {"status": "no_history", "agents": {}}

    # Per-agent, per-view, alphas observed.
    per_agent_view: dict[str, dict[str, list[float]]] = {agent: {} for agent in AGENT_POLARITY}
    # Map agent → field on the concordance row carrying that agent's view.
    # CIO v36 N6: field names verified against actual concordance entries on
    # 2026-05-04. All 5 fields (fund_view, tech_signal, macro_fit, census,
    # news_impact) ARE present in current and recent historical concordance
    # files. The "0 evidence" issue earlier was caused by forward_returns not
    # being populated, NOT by a field-name mismatch. Wire forward_returns
    # from the M11 calibrator/CommitteeBacktester into this function.
    AGENT_FIELD = {
        "fundamental": "fund_view",
        "technical": "tech_signal",
        "macro": "macro_fit",
        "census": "census",
        "news": "news_impact",
        "risk": None,  # risk emits a flag, not a view; handled separately
    }

    for date_str, items in history:
        for stock in items:
            tkr = stock.get("ticker")
            if not tkr:
                continue
            alpha = _alpha_from_returns(forward_returns, tkr, date_str, horizon)
            if alpha is None:
                continue
            for agent, field in AGENT_FIELD.items():
                if not field:
                    continue
                view = stock.get(field)
                if view is None:
                    continue
                # Normalise news_impact (it can be a multi-token tag).
                if agent == "news":
                    if "POSITIVE" in str(view):
                        view = "POSITIVE"
                    elif "NEGATIVE" in str(view):
                        view = "NEGATIVE"
                    else:
                        view = "NEUTRAL"
                per_agent_view[agent].setdefault(str(view), []).append(float(alpha))

    agents_out: dict[str, Any] = {}
    for agent, view_alphas in per_agent_view.items():
        view_stats: dict[str, Any] = {}
        flip_signal_count = 0  # bullish views with low P + bearish views with high P
        evidence_total = 0
        for view, alphas in view_alphas.items():
            n = len(alphas)
            evidence_total += n
            if n < 3:
                view_stats[view] = {"n": n, "p_alpha_pos": None, "verdict": "n<3"}
                continue
            p_pos = sum(1 for a in alphas if a > 0) / n
            polarity = AGENT_POLARITY.get(agent, {}).get(view, 0)
            verdict = "OK"
            if polarity > 0 and p_pos < flip_lower:
                verdict = "INVERTED"
                flip_signal_count += 1
            elif polarity < 0 and p_pos > flip_upper:
                verdict = "INVERTED"
                flip_signal_count += 1
            view_stats[view] = {
                "n": n,
                "p_alpha_pos": round(p_pos, 3),
                "avg_alpha": round(sum(alphas) / n, 2),
                "polarity": polarity,
                "verdict": verdict,
            }

        if evidence_total < min_evidence:
            agent_verdict = "INSUFFICIENT_DATA"
            recommended_sign = 1
        elif flip_signal_count >= 2:
            agent_verdict = "INVERTED"
            recommended_sign = -1
        else:
            agent_verdict = "OK"
            recommended_sign = 1

        agents_out[agent] = {
            "evidence_total": evidence_total,
            "flip_signal_count": flip_signal_count,
            "verdict": agent_verdict,
            "recommended_sign": recommended_sign,
            "views": view_stats,
        }

    return {
        "status": "ok",
        "horizon": horizon,
        "lookback_days": lookback_days,
        "snapshots": len(history),
        "generated_at": datetime.now().isoformat(),
        "agents": agents_out,
    }


def persist_calibration(
    calibration: dict[str, Any],
    path: Path | None = None,
    enabled: bool = False,
) -> Path:
    """
    Persist the latest calibration result to disk and update the rolling
    history of "consecutive INVERTED" counters that drive the auto-flip
    rule.

    SHADOW MODE (`enabled=False`, the default for v17): writes the file
    with `applied: false` for every agent. The synthesis pipeline reads
    this file and ignores `applied=false` entries. Use this for the first
    8 weeks to verify the calibrator behaves as expected.

    AUTO MODE (`enabled=True`): if an agent has been INVERTED for ≥2
    consecutive evaluations, mark `applied: true` so the synthesis
    pipeline picks up the −1 sign on its next run.
    """
    out_path = path or DEFAULT_CALIBRATOR_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    history: dict[str, int] = {}
    if out_path.exists():
        try:
            prev = json.load(open(out_path))
            for agent, data in (prev.get("agents") or {}).items():
                history[agent] = int(data.get("consecutive_inverted", 0))
        except (OSError, json.JSONDecodeError):
            history = {}

    applied: dict[str, int] = {}
    for agent, data in (calibration.get("agents") or {}).items():
        if data.get("verdict") == "INVERTED":
            history[agent] = history.get(agent, 0) + 1
        else:
            history[agent] = 0
        data["consecutive_inverted"] = history[agent]
        # Apply only if the run is in AUTO mode AND we have ≥2 consecutive.
        will_apply = enabled and history[agent] >= 2 and data.get("recommended_sign", 1) == -1
        data["applied"] = will_apply
        applied[agent] = -1 if will_apply else 1

    output = {
        "mode": "AUTO" if enabled else "SHADOW",
        "applied_signs": applied,
        **calibration,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    return out_path


def load_applied_signs(path: Path | None = None) -> dict[str, int]:
    """
    Return per-agent sign multipliers (+1 default, −1 if inverted+applied).
    Synthesis weights should multiply by these to apply auto-flips.
    Returns empty dict if no calibration file or file unreadable — meaning
    no agent is auto-flipped (safe default).
    """
    p = path or DEFAULT_CALIBRATOR_PATH
    if not p.exists():
        return {}
    try:
        data = json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return {}
    if data.get("mode") != "AUTO":
        return {}
    return data.get("applied_signs", {}) or {}
