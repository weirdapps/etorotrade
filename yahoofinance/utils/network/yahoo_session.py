"""
Rebuild a lost Yahoo session inside one scan.

yfinance keeps one cookie and one crumb for the whole process (``yfinance.data.YfData`` is a
singleton). When Yahoo stops honouring them it answers every request with HTTP 401
"Invalid Crumb". yfinance 1.8.0 logs that error and returns an empty ``Ticker.info``
(``YfConfig.debug.hide_exceptions`` defaults to True), so a scan that lost its session wrote a
blank row for every later name. yfinance's own recovery toggles its cookie strategy once per
request, and the basic strategy reloads the cookie it persisted on disk, so it can keep
presenting the cookie Yahoo stopped accepting.

This module notices the 401 on the call that hit it, discards the session, the cookie, the crumb
and the persisted cookie, and lets that ticker be fetched again. Refreshes are bounded per
process and backed off, so a hard block by Yahoo ends in blank rows, not in a loop.

While one worker waits to rebuild, every other worker that is refused waits behind it, so the
scan pauses through a block instead of writing blank rows. On 2026-11-17 Yahoo refused one
runner for 17 minutes and another for 30; the defaults wait out a block of about 40 minutes, and
a priced read after a rebuild restores the full allowance for the next block.
"""

import functools
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from ...core.logging import get_logger

logger = get_logger(__name__)

#: Rebuilds allowed in a row, with no priced read between them. A healthy scan needs none.
MAX_SESSION_REFRESHES = 12

#: Wait before the first rebuild of a block; each later rebuild waits twice as long as the one
#: before, up to MAX_BACKOFF_SECONDS. Twelve steps wait 2,565s (about 43 minutes) in all.
REFRESH_BACKOFF_SECONDS = 15.0
MAX_BACKOFF_SECONDS = 300.0

#: Most time one process spends waiting on rebuilds, over all blocks. The shard job runs a scan
#: twice under a 330-minute timeout, and a scan takes about 90 minutes without waits.
MAX_WAIT_SECONDS = 3600.0

_call_state = threading.local()


def _is_auth_failure(exc: BaseException) -> bool:
    """True for the HTTP 401 Yahoo returns when it no longer accepts the session's crumb."""
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 401


def install_auth_failure_detector() -> None:
    """Record, per thread, that a yfinance JSON request was refused with HTTP 401.

    ``YfData.get_raw_json`` raises the HTTPError before yfinance's quote scraper catches and
    logs it, so this sees the refusal whatever the logging configuration is. The wrapper only
    records and re-raises: yfinance's behaviour is unchanged. Installing twice is a no-op.
    """
    from yfinance.data import YfData

    original = YfData.get_raw_json
    if getattr(original, "_records_auth_failure", False):
        return

    @functools.wraps(original)
    def get_raw_json(self, url, params=None, timeout=30):
        try:
            return original(self, url, params=params, timeout=timeout)
        except Exception as exc:
            if _is_auth_failure(exc):
                _call_state.auth_failed = True
            raise

    get_raw_json._records_auth_failure = True  # type: ignore[attr-defined]
    YfData.get_raw_json = get_raw_json  # type: ignore[method-assign]


def reset_yfinance_session() -> None:
    """Discard everything yfinance holds for the Yahoo session, so the next request mints anew."""
    from yfinance import cache
    from yfinance._http import new_session
    from yfinance.data import YfData

    # YfData is a singleton: calling it with a session swaps that session into the existing
    # instance (SingletonMeta.__call__ -> _set_session), so the old cookie jar goes with it.
    data = YfData(session=new_session())
    with data._cookie_lock:
        data._cookie = None
        data._crumb = None
    # Storing None deletes the persisted cookie, which _get_cookie_basic would otherwise reload.
    cache.get_cookie_cache().store("curlCffi", None)
    # Responses cached by cache_get may include the refused ones.
    YfData.cache_get.cache_clear()


class YahooSessionRefresher:
    """One session rebuild at a time, a bounded number per process, each backed off."""

    def __init__(
        self,
        max_refreshes: int = MAX_SESSION_REFRESHES,
        backoff_seconds: float = REFRESH_BACKOFF_SECONDS,
        max_backoff_seconds: float = MAX_BACKOFF_SECONDS,
        max_wait_seconds: float = MAX_WAIT_SECONDS,
        reset: Callable[[], None] = reset_yfinance_session,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.max_refreshes = max_refreshes
        self.max_backoff_seconds = max_backoff_seconds
        self.max_wait_seconds = max_wait_seconds
        self.backoff_seconds = backoff_seconds
        self._reset = reset
        self._sleep = sleep
        self._lock = threading.Lock()
        self.generation = 0
        self.refreshes = 0
        self.consecutive = 0  # rebuilds since the last priced read
        self.waited = 0.0

    def refresh(self, seen_generation: int) -> bool:
        """Rebuild the session the caller saw fail; True when the caller should fetch again.

        A caller that failed under a session another caller has already rebuilt fetches again
        without rebuilding, so concurrent failures cost one rebuild, not one per worker.
        """
        with self._lock:
            if self.generation != seen_generation:
                return True
            if self.consecutive >= self.max_refreshes:
                return False
            delay = self.delay_for(self.consecutive)
            if self.waited + delay > self.max_wait_seconds:
                _notice(
                    f"Yahoo refused the session (HTTP 401); the {self.max_wait_seconds:.0f}s "
                    "wait allowance is spent, so later names may come back unpriced"
                )
                return False
            self.refreshes += 1
            self.consecutive += 1
            self.waited += delay
            _notice(
                f"Yahoo refused the session (HTTP 401); rebuilding it in {delay:.0f}s "
                f"(step {self.consecutive}/{self.max_refreshes}, waited {self.waited:.0f}s "
                f"of {self.max_wait_seconds:.0f}s)"
            )
            self._sleep(delay)
            try:
                self._reset()
            except Exception as e:
                _notice(f"Could not rebuild the Yahoo session: {type(e).__name__}: {e}")
                return False
            self.generation += 1
            return True

    def delay_for(self, step: int) -> float:
        """The wait before rebuild number ``step`` (from 0) of one block."""
        return min(self.backoff_seconds * 2**step, self.max_backoff_seconds)

    def mark_healthy(self) -> None:
        """A read was priced: the block is over, so the next one starts from the first step."""
        if self.consecutive:
            with self._lock:
                if self.consecutive:
                    _notice(f"Yahoo accepted the rebuilt session after {self.consecutive} step(s)")
                    self.consecutive = 0


def _notice(message: str) -> None:
    # The scan's log level hides warnings in CI, where these matter most (2026-11-17: a block
    # left no trace in the run log), so print them as well.
    logger.warning(message)
    print(message, file=sys.stderr, flush=True)


_default_refresher: YahooSessionRefresher | None = None
_default_refresher_lock = threading.Lock()


def get_session_refresher() -> YahooSessionRefresher:
    """The process-wide refresher, so the refresh bound holds across providers."""
    global _default_refresher
    with _default_refresher_lock:
        if _default_refresher is None:
            _default_refresher = YahooSessionRefresher()
        return _default_refresher


def fetch_info_with_session_refresh(
    yticker: Any,
    rebuild_ticker: Callable[[], Any],
    refresher: YahooSessionRefresher | None = None,
) -> tuple[Any, dict | None]:
    """Read ``yticker.info``; if Yahoo refused the session, rebuild it and read again.

    Returns the ticker object the info came from, with the info. yfinance either returns an
    empty info after a 401 or raises from a later request of the same read; both count as a
    refused session. When the refresh bound is spent, the last outcome stands as before.
    """
    install_auth_failure_detector()
    refresher = refresher or get_session_refresher()
    while True:
        generation = refresher.generation
        _call_state.auth_failed = False
        try:
            info = yticker.info
        except Exception:
            if not _call_state.auth_failed or not refresher.refresh(generation):
                raise
        else:
            if not _call_state.auth_failed:
                refresher.mark_healthy()
                return yticker, info
            if not refresher.refresh(generation):
                return yticker, info
        yticker = rebuild_ticker()
