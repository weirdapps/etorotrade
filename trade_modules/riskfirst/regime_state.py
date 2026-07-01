"""Runner-side glue for the regime overlay: persistence state + resolution.

Impure (reads the live regime, reads/writes a JSON state file). Kept out of the
pure regime_overlay module so the numpy core and the replay stay I/O-free.
"""

from __future__ import annotations

import json
import os
from datetime import date

from .regime_overlay import (
    DEFAULT_PERSISTENCE_DAYS,
    FALLBACK_REGIME,
    confirm_regime,
    exposure_for_regime,
)

DEFAULT_STATE_PATH = os.path.expanduser("~/.weirdapps-trading/regime/state.json")
_MAX_HISTORY = 10


def _read_state(path):
    try:
        with open(path) as f:
            s = json.load(f)
        return s if isinstance(s, dict) else {"history": []}
    except Exception:
        return {"history": []}


def _write_state(path, state):
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def update_history(history, raw_regime, today, max_history=_MAX_HISTORY):
    """Append [today, raw_regime], one entry per date (same-day overwrite)."""
    hist = [h for h in (history or []) if isinstance(h, (list, tuple)) and len(h) == 2]
    if hist and hist[-1][0] == today:
        hist[-1] = [today, raw_regime]
    else:
        hist.append([today, raw_regime])
    return [list(h) for h in hist[-max_history:]]


def resolve_regime_multiplier(
    *,
    state_path=DEFAULT_STATE_PATH,
    persistence_days=DEFAULT_PERSISTENCE_DAYS,
    regime_fn=None,
    today=None,
    table=None,
):
    """Read live regime, update persistence state, return (multiplier, detail)."""
    if regime_fn is None:
        from trade_modules.regime_detector import get_current_regime

        regime_fn = get_current_regime
    if today is None:
        today = date.today().isoformat()
    try:
        raw = regime_fn()
    except Exception:
        raw = None
    state = _read_state(state_path)
    hist = update_history(state.get("history", []), raw or FALLBACK_REGIME, today)
    labels = [h[1] for h in hist]
    confirmed = confirm_regime(labels, persistence_days)
    mult = exposure_for_regime(confirmed, table)
    state.update({"history": hist, "confirmed": confirmed, "updated": today})
    try:
        _write_state(state_path, state)
    except Exception:
        pass  # state is a best-effort cache; never block a run on it
    return mult, {"raw_regime": raw, "confirmed_regime": confirmed, "applied_multiplier": mult}
