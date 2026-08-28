"""Persistent IPO-date cache for the signal pipeline.

``trade_modules.analysis.signals.is_recent_ipo`` probes yfinance for a ticker's
first trading bar (``history(period="max")``) once per ticker, inside the
per-ticker signal loop. Stack-sampled on a 20-ticker run that was 10.73s of
46.1s; across the sharded nightly scan of ~12.8K names it is the better part of
an hour of wall clock, spent re-learning a fact that cannot change.

**An IPO date is permanent.** That is the whole justification for caching it on
disk, and it is also the reason this module persists exactly one thing.

Three probe states, not two
---------------------------
The pre-existing in-memory cache stored ``None`` for two unrelated situations:
the provider genuinely has no history for the ticker, and the fetch failed.
Both then read as "not a recent IPO", which is the strict, fail-closed side of
``sell_criteria``, so the conflation was harmless while it lived only in RAM.

Writing it to disk would not be harmless. A single transient timeout would be
recorded permanently and honoured on every future run, silently pinning that
name's criteria. So the probe reports three states and only one is persisted:

===============  ============================================  ==========
state            meaning                                       persisted?
===============  ============================================  ==========
``found``        a first trading bar came back                 **yes**
``no_data``      the provider answered with an empty history   no
``error``        the fetch raised (network, timeout, throttle) no
===============  ============================================  ==========

``no_data`` is deliberately on the "no" side. yfinance answers a throttle with
an empty frame and no exception, so an empty history is not confirmed negative
knowledge; it is indistinguishable from a rate limit. ``error`` and ``no_data``
stay in memory for the rest of the run (so a name is probed at most once per
run) and are gone when the process exits, which is exactly what makes the next
run retry them.

Storage
-------
JSON at ``yahoofinance/input/ipo_dates.json``, **committed to the repository**,
overridable with ``$ETOROTRADE_IPO_CACHE_PATH``. That directory is where this
repo already keeps machine-generated inputs the scan reads and CI regenerates
(``yfinance_skip.csv``). A committed file is warm on a fresh clone, warm on
every CI runner, reviewable in a diff, and cannot be evicted; an
``actions/cache`` entry is none of those things and expires after 7 idle days.
It is deliberately *not* under ``~/.weirdapps-trading/``, the legacy directory
this estate forbids for new work.

Values are stored as full ISO-8601 naive datetimes rather than dates. The
comparison in ``is_recent_ipo`` is ``first_date > cutoff_datetime``, so dropping
a time component could flip a verdict for a name sitting on the grace boundary.

Every read fails **open**: a missing, truncated, corrupt, mistyped or
future-schema file reads as an empty cache and the probe runs as before. It
never raises and never fails closed.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# --- probe states -----------------------------------------------------------
PROBE_FOUND = "found"
PROBE_NO_DATA = "no_data"
PROBE_ERROR = "error"

# --- storage contract -------------------------------------------------------
SCHEMA_VERSION = 1
"""Bumped whenever the on-disk shape changes. A file whose version is not
exactly this reads as empty, so an old file is never half-understood."""

MAX_ENTRIES = 50_000
"""Hard cap on stored tickers. The eToro universe is ~12.8K, so this is
headroom rather than a working limit; it exists so a runaway caller cannot grow
a tracked repository file without bound. Truncation keeps the lexicographically
first entries, which is deterministic and therefore diff-stable."""

FLUSH_EVERY = 250
"""Write through after this many newly learned dates, so a 5.5h CI shard that
is killed rather than exiting cleanly still banks most of what it learned."""

ENV_CACHE_PATH = "ETOROTRADE_IPO_CACHE_PATH"

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_PATH = _REPO_ROOT / "yahoofinance" / "input" / "ipo_dates.json"


def resolve_cache_path() -> Path:
    """Where the cache lives for this process."""
    override = os.environ.get(ENV_CACHE_PATH)
    if override:
        return Path(override).expanduser()
    return DEFAULT_CACHE_PATH


# ---------------------------------------------------------------------------
# the probe
# ---------------------------------------------------------------------------


def probe_first_trade_date(ticker: str, timeout: int = 5) -> tuple[str, datetime | None]:
    """Ask yfinance for ``ticker``'s first trading bar.

    Returns ``(state, value)`` where state is one of :data:`PROBE_FOUND`,
    :data:`PROBE_NO_DATA` or :data:`PROBE_ERROR`, and ``value`` is a naive
    ``datetime`` only for :data:`PROBE_FOUND`.

    The request is byte-for-byte the one ``is_recent_ipo`` used to make inline
    (``period="max"``, same timeout). Nothing here may be "optimised" into a
    coarser interval: the first bar of a resampled series is the start of the
    period, not the first trading day, and near the 12-month grace boundary
    that difference flips verdicts.
    """
    try:
        import yfinance as yf

        hist = yf.Ticker(ticker).history(period="max", timeout=timeout)
        if hist is None or hist.empty:
            return PROBE_NO_DATA, None
        first_date = hist.index[0].to_pydatetime().replace(tzinfo=None)
        return PROBE_FOUND, first_date
    except Exception as exc:  # noqa: BLE001 - any failure is one thing: unknown
        logger.debug("Ticker %s: failed to auto-detect IPO date: %s", ticker, exc)
        return PROBE_ERROR, None


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------


def _decode_entries(raw: object) -> dict[str, datetime]:
    """Decode a ``dates`` mapping, dropping individual bad entries.

    One unparseable row must not cost the other 12,000.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, datetime] = {}
    for ticker, value in raw.items():
        if not isinstance(ticker, str) or not isinstance(value, str):
            continue
        try:
            parsed = datetime.fromisoformat(value)
        except (ValueError, TypeError):
            continue
        if parsed.tzinfo is not None:
            parsed = parsed.replace(tzinfo=None)
        out[ticker] = parsed
    return out


def _cap(entries: dict[str, datetime]) -> dict[str, datetime]:
    if len(entries) <= MAX_ENTRIES:
        return entries
    logger.warning("IPO date cache holds %d entries, capping at %d", len(entries), MAX_ENTRIES)
    return {k: entries[k] for k in sorted(entries)[:MAX_ENTRIES]}


def read_cache_file(path: Path) -> dict[str, datetime]:
    """Read one cache file. Any problem at all yields ``{}``; never raises."""
    try:
        payload = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001 - corrupt cache must fail OPEN
        logger.warning("IPO date cache at %s unreadable (%s); treating as empty", path, exc)
        return {}

    if not isinstance(payload, dict):
        logger.warning("IPO date cache at %s is not an object; treating as empty", path)
        return {}
    if payload.get("schema_version") != SCHEMA_VERSION:
        logger.warning(
            "IPO date cache at %s has schema_version %r, expected %r; treating as empty",
            path,
            payload.get("schema_version"),
            SCHEMA_VERSION,
        )
        return {}
    return _cap(_decode_entries(payload.get("dates")))


def _write_cache_file(path: Path, entries: dict[str, datetime]) -> bool:
    """Atomically write ``entries``. Returns False on any failure; never raises."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "yfinance Ticker.history(period='max') first bar",
        "note": "Confirmed IPO/first-trade dates only. Failed probes are never recorded.",
        "dates": {k: entries[k].isoformat() for k in sorted(entries)},
    }
    tmp_name: str | None = None
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".ipo_dates.", suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_name, path)
        tmp_name = None
        return True
    except Exception as exc:  # noqa: BLE001 - an unwritable cache is not fatal
        logger.warning("Could not write IPO date cache to %s: %s", path, exc)
        return False
    finally:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


class IpoDateCache:
    """Confirmed first-trade dates, keyed by ticker.

    Only ``put`` writes, and ``put`` accepts nothing but a ``datetime``. There
    is no way to record "I tried and failed" in this object, which is the
    point: the class cannot express the state that must not be persisted.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else resolve_cache_path()
        self._entries: dict[str, datetime] = read_cache_file(self.path)
        self._pending = 0

    # -- reads ----------------------------------------------------------
    def get(self, ticker: str) -> datetime | None:
        return self._entries.get(ticker)

    def tickers(self) -> list[str]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, ticker: object) -> bool:
        return ticker in self._entries

    # -- writes ---------------------------------------------------------
    def put(self, ticker: str, first_trade_date: datetime) -> None:
        """Record a confirmed first-trade date.

        Raises:
            TypeError: for anything that is not a ``datetime``. A failed probe
                has no date, so there is nothing it could legally pass here.
        """
        if not isinstance(first_trade_date, datetime):
            raise TypeError(
                "IpoDateCache.put accepts a datetime only; "
                f"got {type(first_trade_date).__name__}. A failed or empty probe "
                "must not be persisted."
            )
        if first_trade_date.tzinfo is not None:
            first_trade_date = first_trade_date.replace(tzinfo=None)
        if self._entries.get(ticker) == first_trade_date:
            return
        self._entries[ticker] = first_trade_date
        self._pending += 1
        if self._pending >= FLUSH_EVERY:
            self.save()

    def save(self) -> bool:
        """Persist if anything new was learned. Returns True when it wrote.

        Re-reads the file first and unions, so two processes sharing one cache
        (the local CLI and a background run, say) add to each other rather than
        clobbering. On conflict the EARLIER date wins: an earlier first bar
        means a longer history came back, and it is the fail-closed direction,
        since earlier means "not a recent IPO" means strict criteria.
        """
        if not self._pending:
            return False
        merged = dict(read_cache_file(self.path))
        for ticker, value in self._entries.items():
            existing = merged.get(ticker)
            if existing is None or value < existing:
                merged[ticker] = value
        if _write_cache_file(self.path, _cap(merged)):
            self._entries = merged
            self._pending = 0
            return True
        return False


# ---------------------------------------------------------------------------
# module singleton
# ---------------------------------------------------------------------------

_singleton: IpoDateCache | None = None
_atexit_registered = False


def get_cache() -> IpoDateCache:
    """The process-wide cache, created on first use and flushed at exit."""
    global _singleton, _atexit_registered
    if _singleton is None:
        _singleton = IpoDateCache()
        if not _atexit_registered:
            atexit.register(_flush_at_exit)
            _atexit_registered = True
    return _singleton


def reset_cache() -> None:
    """Drop the singleton without saving. Used by tests to simulate a new run."""
    global _singleton
    _singleton = None


def _flush_at_exit() -> None:
    if _singleton is not None:
        try:
            _singleton.save()
        except Exception:  # noqa: BLE001 - never let a cache flush kill a run
            logger.debug("IPO date cache flush at exit failed", exc_info=True)


# ---------------------------------------------------------------------------
# merging shard caches (how the nightly scan accrues the benefit)
# ---------------------------------------------------------------------------


def merge_cache_files(paths, output_path: Path | str) -> int:
    """Union several cache files into ``output_path``; return the entry count.

    Used by the ``daily-signals`` merge job to fold the six read-only scan
    shards' caches back into the committed file. Unreadable inputs are skipped,
    not fatal. Conflicts resolve to the earliest date, so the result does not
    depend on the order the shards happen to be listed in.
    """
    merged: dict[str, datetime] = {}
    for p in paths:
        for ticker, value in read_cache_file(Path(p)).items():
            existing = merged.get(ticker)
            if existing is None or value < existing:
                merged[ticker] = value
    if not merged:
        logger.info("No IPO date entries to merge; leaving %s untouched", output_path)
        return 0
    merged = _cap(merged)
    _write_cache_file(Path(output_path), merged)
    return len(merged)
