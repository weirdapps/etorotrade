"""Regime overlay — portfolio-level gross-exposure dial for the risk-first book.

Pure functions only (no I/O): the runner resolves the multiplier (live regime +
state file) and passes a plain float into select_and_construct. Keeps the numpy
core deterministic and lets the historical replay share this exact code.
"""

from __future__ import annotations

import numpy as np

REGIME_EXPOSURE = {"risk_on": 1.00, "neutral": 0.90, "risk_off": 0.65, "crisis": 0.40}
DEFAULT_PERSISTENCE_DAYS = 2
FALLBACK_REGIME = "neutral"


def exposure_for_regime(regime, table=None, fallback=FALLBACK_REGIME) -> float:
    """Regime label -> exposure multiplier. None/unknown -> table[fallback]."""
    table = REGIME_EXPOSURE if table is None else table
    if regime in table:
        return float(table[regime])
    return float(table[fallback])


def confirm_regime(history, persistence_days, fallback=FALLBACK_REGIME) -> str:
    """Hysteresis, computed PURELY from history (oldest->newest).

    Returns the most recent regime that achieved a run of >= persistence_days
    consecutive days anywhere in history; if none did, returns `fallback`.
    """
    if persistence_days < 1:
        persistence_days = 1
    best = fallback
    run_label = None
    run_len = 0
    for label in history:
        if label == run_label:
            run_len += 1
        else:
            run_label, run_len = label, 1
        if run_len >= persistence_days:
            best = label  # oldest->newest: last qualifying run wins => most recent
    return best


def scale_for_regime(weights, multiplier) -> np.ndarray:
    """Scale the whole book by `multiplier` (gross drops, remainder -> cash).
    Multiplier clamped to [0.0, 1.0]."""
    m = min(1.0, max(0.0, float(multiplier)))
    return np.asarray(weights, dtype=float) * m
