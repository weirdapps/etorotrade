"""
Rolling Conviction Percentile Thresholds (CIO v17 H4.b)

Replaces fixed `conviction >= 55 → ADD` thresholds with rolling-window
percentile cuts. The empirical headline from CIO v17 was that, within
BUY-signal stocks at T+30, Spearman ρ(conviction, α30) ≈ −0.002 — i.e.
the absolute conviction value carries no rank-order information. Fixed
cuts therefore just produce regime-dependent action counts.

Solution: maintain a rolling distribution of recent convictions per
signal class and translate the threshold "ADD if conviction is in the
top quartile of recent BUYs" into a numeric cut that drifts with the
regime.

State persists to ~/.weirdapps-trading/committee/conviction_thresholds.json
and is updated weekly by the backtest pipeline. Until enough state
exists (≥3 snapshots), the legacy fixed thresholds are used.
"""

import json
import logging
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS_PATH = (
    Path.home() / ".weirdapps-trading" / "committee" / "conviction_thresholds.json"
)

# Legacy fixed defaults — used when no rolling state is available.
LEGACY_THRESHOLDS = {
    "B": {"add_pct": 55, "trim_pct": 0},  # BUY signal: ADD ≥55, never trim
    "H": {"add_pct": 70, "trim_pct": 35},  # HOLD signal: ADD ≥70, TRIM <35
    "S": {"sell_pct": 60},  # SELL signal: SELL ≥60, else TRIM
    "I": {"add_pct": 70, "trim_pct": 35},  # INCONCLUSIVE: same as HOLD
}

# Target percentiles per signal class (the "what fraction is ADD/TRIM").
TARGET_PERCENTILES = {
    "B": {"add_p": 50, "trim_p": 5},  # BUY: top half = ADD, bottom 5% = HOLD floor
    "H": {"add_p": 75, "trim_p": 25},  # HOLD: top quartile = ADD, bottom quartile = TRIM
    "S": {"sell_p": 50},  # SELL: top half by magnitude = SELL
    "I": {"add_p": 75, "trim_p": 25},
}


def _percentile(values: list[float], p: float) -> float | None:
    """Compute the p-th percentile (0-100) of a list of floats."""
    if not values:
        return None
    s = sorted(values)
    if p <= 0:
        return s[0]
    if p >= 100:
        return s[-1]
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def compute_rolling_thresholds(
    history: Iterable[dict[str, Any]],
    lookback_snapshots: int = 8,
    min_per_signal: int = 20,
) -> dict[str, dict[str, float]]:
    """
    Compute per-signal-class action thresholds from the most recent
    `lookback_snapshots` concordance archives.

    Args:
        history: iterable of {"date", "concordance": [...]} dicts (as
            produced by CommitteeBacktester.load_history).
        lookback_snapshots: how many most-recent snapshots to use.
        min_per_signal: minimum stock-records per signal class to compute
            percentiles; below this, fall back to legacy threshold.

    Returns:
        Dict signal → {"add_pct": float, "trim_pct": float} computed from
        rolling percentiles. Always includes a "_meta" key with sample
        sizes and the lookback window for traceability.
    """
    items = list(history)
    items.sort(key=lambda e: e.get("date", ""))
    items = items[-lookback_snapshots:]

    convictions: dict[str, list[float]] = {}
    for entry in items:
        for stock in entry.get("concordance", []):
            sig = stock.get("signal", "?")
            try:
                conv = float(stock.get("conviction", 50))
            except (TypeError, ValueError):
                continue
            convictions.setdefault(sig, []).append(conv)

    out: dict[str, dict[str, float]] = {}
    for sig in ("B", "H", "S", "I"):
        legacy = LEGACY_THRESHOLDS.get(sig, {})
        targets = TARGET_PERCENTILES.get(sig, {})
        sample = convictions.get(sig, [])
        n = len(sample)

        if n < min_per_signal:
            # Insufficient data — keep legacy.
            out[sig] = {**legacy, "n": n, "source": "legacy"}
            continue

        sig_out: dict[str, float] = {"n": n, "source": "rolling"}
        if "add_p" in targets:
            sig_out["add_pct"] = round(
                _percentile(sample, targets["add_p"]) or legacy.get("add_pct", 55), 1
            )
        if "trim_p" in targets:
            sig_out["trim_pct"] = round(
                _percentile(sample, targets["trim_p"]) or legacy.get("trim_pct", 35), 1
            )
        if "sell_p" in targets:
            sig_out["sell_pct"] = round(
                _percentile(sample, targets["sell_p"]) or legacy.get("sell_pct", 60), 1
            )

        out[sig] = sig_out

    out["_meta"] = {
        "snapshots_used": len(items),
        "first_date": items[0].get("date") if items else None,
        "last_date": items[-1].get("date") if items else None,
        "min_per_signal": min_per_signal,
        "lookback_snapshots": lookback_snapshots,
    }
    return out


def persist_thresholds(
    thresholds: dict[str, Any],
    path: Path | None = None,
) -> Path:
    """Persist computed thresholds for the next committee run to consume."""
    out_path = path or DEFAULT_THRESHOLDS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "thresholds": thresholds,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def load_thresholds(
    path: Path | None = None,
    max_age_days: int = 14,
) -> dict[str, dict[str, float]] | None:
    """
    Load the most recent rolling thresholds.

    Returns None if file is missing, unparseable, or older than
    max_age_days — in which case the caller should fall back to legacy
    fixed thresholds (LEGACY_THRESHOLDS). max_age_days defaults to 14
    because we recompute weekly; >14 days stale means the pipeline didn't
    run and the data is no longer trustworthy.
    """
    p = path or DEFAULT_THRESHOLDS_PATH
    if not p.exists():
        return None
    try:
        data = json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return None
    gen = data.get("generated_at")
    if gen:
        try:
            ts = datetime.fromisoformat(gen)
            if datetime.now() - ts > timedelta(days=max_age_days):
                return None
        except ValueError:
            pass
    return data.get("thresholds")


def get_action_thresholds(
    signal: str,
    rolling: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """
    Resolve the action thresholds to use for a given signal class.

    Falls back to legacy when rolling state is missing or low-evidence.
    Always returns at least the legacy keys so the caller can use
    `thr.get("add_pct", 55)` safely.
    """
    legacy = dict(LEGACY_THRESHOLDS.get(signal, {}))
    if not rolling:
        legacy["source"] = "legacy"
        return legacy
    sig_block = rolling.get(signal)
    if not sig_block or sig_block.get("source") == "legacy":
        legacy["source"] = "legacy"
        return legacy
    legacy.update({k: v for k, v in sig_block.items() if k != "source"})
    legacy["source"] = "rolling"
    return legacy
