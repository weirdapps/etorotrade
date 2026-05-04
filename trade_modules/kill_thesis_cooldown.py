"""
M10: Kill-thesis cooldown — CIO v36 Empirical Refoundation.

When a kill-thesis fires on a ticker (per existing `kill_thesis` modifier in
committee_synthesis), the legacy behavior is a -15 conviction penalty —
the position can be sized at 0.85× and re-entered the next week. M10 adds
a hard 28-day cooldown: during that window, sizing for the ticker is 0
regardless of conviction.

State persists in a JSONL log so cooldowns survive across runs:
~/.weirdapps-trading/committee/kill_thesis_cooldowns.jsonl

Each line:
  {"ticker": "AAPL", "timestamp": "2026-05-01T12:00:00", "reason": "VIX>40"}

Cooldown logic uses the LATEST trigger per ticker so a re-fire resets the
window, not the original trigger.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_COOLDOWN_LOG = (
    Path.home() / ".weirdapps-trading" / "committee" / "kill_thesis_cooldowns.jsonl"
)
COOLDOWN_DAYS = 28  # 4 weeks


def record_kill_thesis(
    ticker: str,
    reason: str,
    log_path: Path | None = None,
    now: datetime | None = None,
) -> None:
    """Append a kill-thesis trigger to the cooldown log."""
    log_path = Path(log_path or DEFAULT_COOLDOWN_LOG)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ticker": str(ticker),
        "reason": str(reason),
        "timestamp": (now or datetime.now()).isoformat(timespec="seconds"),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info(
        "Recorded kill-thesis cooldown for %s (reason=%s, until=%s)",
        ticker,
        reason,
        (now or datetime.now() + timedelta(days=COOLDOWN_DAYS)).date(),
    )


def _latest_trigger(ticker: str, log_path: Path) -> datetime | None:
    """Return the most recent trigger timestamp for ticker, or None."""
    if not log_path.exists():
        return None
    latest: datetime | None = None
    try:
        with open(log_path) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                if entry.get("ticker") != ticker:
                    continue
                ts_str = entry.get("timestamp")
                if not ts_str:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_str)
                except ValueError:
                    continue
                if latest is None or ts > latest:
                    latest = ts
    except OSError as exc:
        logger.warning("Failed to read cooldown log %s: %s", log_path, exc)
        return None
    return latest


def is_in_cooldown(
    ticker: str,
    log_path: Path | None = None,
    now: datetime | None = None,
    cooldown_days: int = COOLDOWN_DAYS,
) -> bool:
    """Return True if ticker is within cooldown window of its latest trigger."""
    log_path = Path(log_path or DEFAULT_COOLDOWN_LOG)
    latest = _latest_trigger(ticker, log_path)
    if latest is None:
        return False
    expiry = latest + timedelta(days=cooldown_days)
    return (now or datetime.now()) <= expiry


def cooldown_remaining_days(
    ticker: str,
    log_path: Path | None = None,
    now: datetime | None = None,
    cooldown_days: int = COOLDOWN_DAYS,
) -> int:
    """Days until ticker's cooldown expires; 0 if not in cooldown."""
    log_path = Path(log_path or DEFAULT_COOLDOWN_LOG)
    latest = _latest_trigger(ticker, log_path)
    if latest is None:
        return 0
    expiry = latest + timedelta(days=cooldown_days)
    delta = expiry - (now or datetime.now())
    days = max(0, delta.days)
    return days
