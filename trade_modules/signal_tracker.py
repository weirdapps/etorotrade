"""
Signal Tracking System

Logs trading signals with metadata for forward validation.
Allows tracking signal accuracy over time to improve the framework.

P1 Improvement - Implemented from HEDGE_FUND_REVIEW.md recommendations.

Since backtesting is not feasible (no historical target prices available),
this system enables forward validation of signal quality.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading

logger = logging.getLogger(__name__)

# Default storage location
DEFAULT_SIGNAL_LOG_PATH = Path(__file__).parent.parent / "yahoofinance" / "output" / "signal_log.jsonl"

# Lock for thread-safe file operations
_file_lock = threading.Lock()

# SPY price cache (refreshed periodically)
_spy_cache: Dict[str, Any] = {"price": None, "timestamp": None}
_SPY_CACHE_TTL_SECONDS = 300  # 5 minute cache


def _get_spy_price() -> Optional[float]:
    """
    Get current SPY price for benchmark comparison.

    Uses yfinance with caching to avoid excessive API calls.
    Returns None if price cannot be fetched.
    """
    # Check cache validity
    if _spy_cache["timestamp"] is not None:
        cache_age = (datetime.now() - _spy_cache["timestamp"]).total_seconds()
        if cache_age < _SPY_CACHE_TTL_SECONDS and _spy_cache["price"] is not None:
            return _spy_cache["price"]

    try:
        import yfinance as yf
        spy = yf.Ticker("SPY")
        # Use fast_info for quick price lookup
        price = spy.fast_info.get("lastPrice") or spy.fast_info.get("regularMarketPrice")
        if price:
            _spy_cache["price"] = float(price)
            _spy_cache["timestamp"] = datetime.now()
            return _spy_cache["price"]
    except Exception as e:
        logger.debug(f"Failed to fetch SPY price: {e}")

    return None


class SignalRecord:
    """Represents a single trading signal record."""

    def __init__(
        self,
        ticker: str,
        signal: str,
        timestamp: Optional[datetime] = None,
        price_at_signal: Optional[float] = None,
        target_price: Optional[float] = None,
        upside: Optional[float] = None,
        buy_percentage: Optional[float] = None,
        exret: Optional[float] = None,
        market_cap: Optional[float] = None,
        tier: Optional[str] = None,
        region: Optional[str] = None,
        vix_level: Optional[float] = None,
        sector: Optional[str] = None,
        # Additional metrics for comprehensive tracking
        pe_forward: Optional[float] = None,
        pe_trailing: Optional[float] = None,
        peg: Optional[float] = None,
        short_interest: Optional[float] = None,
        roe: Optional[float] = None,
        debt_equity: Optional[float] = None,
        pct_52w_high: Optional[float] = None,
        # Benchmark tracking
        spy_price: Optional[float] = None,
        # Sell trigger details
        sell_triggers: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.ticker = ticker
        self.signal = signal
        self.timestamp = timestamp or datetime.now()
        self.price_at_signal = price_at_signal
        self.target_price = target_price
        self.upside = upside
        self.buy_percentage = buy_percentage
        self.exret = exret
        self.market_cap = market_cap
        self.tier = tier
        self.region = region
        self.vix_level = vix_level
        self.sector = sector
        # Additional metrics
        self.pe_forward = pe_forward
        self.pe_trailing = pe_trailing
        self.peg = peg
        self.short_interest = short_interest
        self.roe = roe
        self.debt_equity = debt_equity
        self.pct_52w_high = pct_52w_high
        # Benchmark
        self.spy_price = spy_price
        self.sell_triggers = sell_triggers or []
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for JSON serialization."""
        return {
            "ticker": self.ticker,
            "signal": self.signal,
            "timestamp": self.timestamp.isoformat(),
            "price_at_signal": self.price_at_signal,
            "target_price": self.target_price,
            "upside": self.upside,
            "buy_percentage": self.buy_percentage,
            "exret": self.exret,
            "market_cap": self.market_cap,
            "tier": self.tier,
            "region": self.region,
            "vix_level": self.vix_level,
            "sector": self.sector,
            # Additional metrics
            "pe_forward": self.pe_forward,
            "pe_trailing": self.pe_trailing,
            "peg": self.peg,
            "short_interest": self.short_interest,
            "roe": self.roe,
            "debt_equity": self.debt_equity,
            "pct_52w_high": self.pct_52w_high,
            # Benchmark
            "spy_price": self.spy_price,
            "sell_triggers": self.sell_triggers,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalRecord":
        """Create SignalRecord from dictionary."""
        return cls(
            ticker=data["ticker"],
            signal=data["signal"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            price_at_signal=data.get("price_at_signal"),
            target_price=data.get("target_price"),
            upside=data.get("upside"),
            buy_percentage=data.get("buy_percentage"),
            exret=data.get("exret"),
            market_cap=data.get("market_cap"),
            tier=data.get("tier"),
            region=data.get("region"),
            vix_level=data.get("vix_level"),
            sector=data.get("sector"),
            # Additional metrics
            pe_forward=data.get("pe_forward"),
            pe_trailing=data.get("pe_trailing"),
            peg=data.get("peg"),
            short_interest=data.get("short_interest"),
            roe=data.get("roe"),
            debt_equity=data.get("debt_equity"),
            pct_52w_high=data.get("pct_52w_high"),
            # Benchmark
            spy_price=data.get("spy_price"),
            sell_triggers=data.get("sell_triggers", []),
            metadata=data.get("metadata", {}),
        )


class SignalTracker:
    """
    Tracks trading signals for forward validation.

    Stores signals in a JSONL (JSON Lines) file for easy appending
    and analysis.
    """

    def __init__(self, log_path: Optional[Path] = None):
        """
        Initialize signal tracker.

        Args:
            log_path: Path to signal log file (default: output/signal_log.jsonl)
        """
        self.log_path = log_path or DEFAULT_SIGNAL_LOG_PATH
        self._ensure_log_file()

    def _ensure_log_file(self) -> None:
        """Ensure log file and parent directories exist."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.touch()

    def log_signal(self, record: SignalRecord) -> bool:
        """
        Log a signal record to the tracking file.

        Args:
            record: SignalRecord to log

        Returns:
            True if logged successfully, False otherwise
        """
        try:
            with _file_lock:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(record.to_dict()) + "\n")
            logger.debug(f"Logged signal: {record.ticker} -> {record.signal}")
            return True
        except Exception as e:
            logger.warning(f"Failed to log signal for {record.ticker}: {e}")
            return False

    def log_signals_batch(self, records: List[SignalRecord]) -> int:
        """
        Log multiple signals in a batch.

        Args:
            records: List of SignalRecord objects

        Returns:
            Number of successfully logged records
        """
        logged = 0
        try:
            with _file_lock:
                with open(self.log_path, "a") as f:
                    for record in records:
                        f.write(json.dumps(record.to_dict()) + "\n")
                        logged += 1
            logger.info(f"Logged {logged} signals to tracker")
        except Exception as e:
            logger.warning(f"Batch logging failed after {logged} records: {e}")
        return logged

    def get_signals(
        self,
        ticker: Optional[str] = None,
        signal_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[SignalRecord]:
        """
        Retrieve signals with optional filtering.

        Args:
            ticker: Filter by ticker symbol
            signal_type: Filter by signal type (B/S/H)
            start_date: Filter signals after this date
            end_date: Filter signals before this date
            limit: Maximum number of records to return

        Returns:
            List of matching SignalRecord objects
        """
        records = []
        try:
            with _file_lock:
                with open(self.log_path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            record = SignalRecord.from_dict(data)

                            # Apply filters
                            if ticker and record.ticker != ticker:
                                continue
                            if signal_type and record.signal != signal_type:
                                continue
                            if start_date and record.timestamp < start_date:
                                continue
                            if end_date and record.timestamp > end_date:
                                continue

                            records.append(record)
                            if len(records) >= limit:
                                break
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.debug(f"Skipping malformed record: {e}")
                            continue
        except FileNotFoundError:
            logger.debug("Signal log file not found")
        except Exception as e:
            logger.warning(f"Error reading signals: {e}")

        return records

    def get_signal_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of logged signals.

        Returns:
            Dictionary with signal statistics
        """
        records = self.get_signals(limit=100000)  # Get all records

        if not records:
            return {"total": 0, "by_signal": {}, "by_tier": {}, "by_region": {}}

        by_signal: Dict[str, int] = {}
        by_tier: Dict[str, int] = {}
        by_region: Dict[str, int] = {}

        for record in records:
            # Count by signal type
            by_signal[record.signal] = by_signal.get(record.signal, 0) + 1
            # Count by tier
            if record.tier:
                by_tier[record.tier] = by_tier.get(record.tier, 0) + 1
            # Count by region
            if record.region:
                by_region[record.region] = by_region.get(record.region, 0) + 1

        stats: Dict[str, Any] = {
            "total": len(records),
            "by_signal": by_signal,
            "by_tier": by_tier,
            "by_region": by_region,
            "date_range": {
                "first": min(r.timestamp for r in records).isoformat(),
                "last": max(r.timestamp for r in records).isoformat(),
            },
        }

        return stats

    def clear_old_signals(self, days_to_keep: int = 90) -> int:
        """
        Remove signals older than specified days.

        Args:
            days_to_keep: Keep signals from the last N days

        Returns:
            Number of signals removed
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=days_to_keep)
        records = self.get_signals(limit=100000)
        kept_records = [r for r in records if r.timestamp >= cutoff]
        removed = len(records) - len(kept_records)

        if removed > 0:
            try:
                with _file_lock:
                    with open(self.log_path, "w") as f:
                        for record in kept_records:
                            f.write(json.dumps(record.to_dict()) + "\n")
                logger.info(f"Removed {removed} old signals, kept {len(kept_records)}")
            except Exception as e:
                logger.warning(f"Failed to clear old signals: {e}")
                return 0

        return removed


# Global tracker instance
_tracker: Optional[SignalTracker] = None


def get_tracker() -> SignalTracker:
    """Get the global signal tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = SignalTracker()
    return _tracker


def log_signal(
    ticker: str,
    signal: str,
    price: Optional[float] = None,
    target: Optional[float] = None,
    upside: Optional[float] = None,
    buy_pct: Optional[float] = None,
    exret: Optional[float] = None,
    market_cap: Optional[float] = None,
    tier: Optional[str] = None,
    region: Optional[str] = None,
    sector: Optional[str] = None,
    # Additional metrics for comprehensive tracking
    pe_forward: Optional[float] = None,
    pe_trailing: Optional[float] = None,
    peg: Optional[float] = None,
    short_interest: Optional[float] = None,
    roe: Optional[float] = None,
    debt_equity: Optional[float] = None,
    pct_52w_high: Optional[float] = None,
    sell_triggers: Optional[List[str]] = None,
    **kwargs,
) -> bool:
    """
    Convenience function to log a signal.

    Args:
        ticker: Stock ticker symbol
        signal: Signal type (B/S/H/I)
        price: Current price at signal time
        target: Analyst target price
        upside: Upside percentage
        buy_pct: Buy percentage from analysts
        exret: Expected return
        market_cap: Market capitalization
        tier: Market cap tier
        region: Geographic region
        sector: Stock sector
        pe_forward: Forward P/E ratio
        pe_trailing: Trailing P/E ratio
        peg: PEG ratio
        short_interest: Short interest percentage
        roe: Return on Equity percentage
        debt_equity: Debt to Equity ratio percentage
        pct_52w_high: Percentage of 52-week high
        sell_triggers: List of sell conditions triggered (for SELL signals)
        **kwargs: Additional metadata

    Returns:
        True if logged successfully
    """
    # Get VIX level if available
    vix_level = None
    try:
        from trade_modules.vix_regime_provider import get_current_vix
        vix_level = get_current_vix()
    except ImportError:
        pass

    # Get SPY price for benchmark comparison
    spy_price = None
    try:
        spy_price = _get_spy_price()
    except Exception:
        pass

    record = SignalRecord(
        ticker=ticker,
        signal=signal,
        price_at_signal=price,
        target_price=target,
        upside=upside,
        buy_percentage=buy_pct,
        exret=exret,
        market_cap=market_cap,
        tier=tier,
        region=region,
        vix_level=vix_level,
        sector=sector,
        # Additional metrics
        pe_forward=pe_forward,
        pe_trailing=pe_trailing,
        peg=peg,
        short_interest=short_interest,
        roe=roe,
        debt_equity=debt_equity,
        pct_52w_high=pct_52w_high,
        # Benchmark
        spy_price=spy_price,
        sell_triggers=sell_triggers or [],
        metadata=kwargs,
    )

    return get_tracker().log_signal(record)


def get_signal_summary() -> Dict[str, Any]:
    """Get summary of logged signals."""
    return get_tracker().get_signal_stats()


# ============================================
# SIGNAL CHANGE DETECTION (P1.3 Enhancement)
# ============================================


class SignalChangeDetector:
    """
    Detect and classify signal changes between runs.

    Used to identify when signals change (especially BUY -> SELL which
    requires immediate action). Implements urgency classification.
    """

    # Signal transition classifications
    SIGNAL_PRIORITY = {"B": 1, "H": 2, "S": 3, "I": 4}

    # Urgency levels for signal transitions
    # (from_signal, to_signal) -> urgency
    CHANGE_URGENCY = {
        ("B", "S"): "CRITICAL",  # BUY -> SELL: Immediate action required
        ("B", "H"): "HIGH",  # BUY -> HOLD: Monitor closely
        ("H", "S"): "MEDIUM",  # HOLD -> SELL: Consider action
        ("S", "B"): "OPPORTUNITY",  # SELL -> BUY: New opportunity
        ("H", "B"): "OPPORTUNITY",  # HOLD -> BUY: New opportunity
        ("S", "H"): "LOW",  # SELL -> HOLD: Stabilizing
        ("I", "B"): "OPPORTUNITY",  # INCONCLUSIVE -> BUY: New signal
        ("I", "S"): "MEDIUM",  # INCONCLUSIVE -> SELL: New concern
        ("B", "I"): "HIGH",  # BUY -> INCONCLUSIVE: Lost coverage
        ("S", "I"): "LOW",  # SELL -> INCONCLUSIVE: Lost coverage
        ("H", "I"): "LOW",  # HOLD -> INCONCLUSIVE: Lost coverage
    }

    def __init__(self, tracker: Optional[SignalTracker] = None):
        """
        Initialize signal change detector.

        Args:
            tracker: SignalTracker instance (uses global if not provided)
        """
        self.tracker = tracker or get_tracker()

    def get_latest_signals_by_date(
        self, date_str: Optional[str] = None
    ) -> Dict[str, SignalRecord]:
        """
        Get the most recent signal for each ticker on a given date.

        Args:
            date_str: Date string (YYYY-MM-DD) or None for today

        Returns:
            Dict mapping ticker -> SignalRecord
        """
        if date_str:
            target_date = datetime.fromisoformat(date_str)
        else:
            target_date = datetime.now()

        start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)

        records = self.tracker.get_signals(start_date=start, end_date=end, limit=10000)

        # Keep latest record per ticker
        latest: Dict[str, SignalRecord] = {}
        for record in records:
            if (
                record.ticker not in latest
                or record.timestamp > latest[record.ticker].timestamp
            ):
                latest[record.ticker] = record

        return latest

    def detect_changes(
        self,
        current_date: str,
        previous_date: str,
    ) -> List[Dict[str, Any]]:
        """
        Compare signals between two dates and identify changes.

        Args:
            current_date: Current date string (YYYY-MM-DD)
            previous_date: Previous date string (YYYY-MM-DD)

        Returns:
            List of change dictionaries with urgency classification
        """
        current = self.get_latest_signals_by_date(current_date)
        previous = self.get_latest_signals_by_date(previous_date)

        changes: List[Dict[str, Any]] = []

        # Check all tickers in both sets
        all_tickers = set(current.keys()) | set(previous.keys())

        for ticker in all_tickers:
            curr_signal = current.get(ticker)
            prev_signal = previous.get(ticker)

            # Track new stocks
            if curr_signal and not prev_signal:
                changes.append(
                    {
                        "ticker": ticker,
                        "previous_signal": None,
                        "current_signal": curr_signal.signal,
                        "urgency": "INFO",
                        "change_type": "NEW",
                        "previous_price": None,
                        "current_price": curr_signal.price_at_signal,
                        "price_change_pct": None,
                        "current_upside": curr_signal.upside,
                        "current_buy_pct": curr_signal.buy_percentage,
                    }
                )
                continue

            # Track removed stocks
            if prev_signal and not curr_signal:
                changes.append(
                    {
                        "ticker": ticker,
                        "previous_signal": prev_signal.signal,
                        "current_signal": None,
                        "urgency": "INFO",
                        "change_type": "REMOVED",
                        "previous_price": prev_signal.price_at_signal,
                        "current_price": None,
                        "price_change_pct": None,
                        "current_upside": None,
                        "current_buy_pct": None,
                    }
                )
                continue

            # Both exist - check for signal change
            if curr_signal.signal != prev_signal.signal:
                urgency = self.CHANGE_URGENCY.get(
                    (prev_signal.signal, curr_signal.signal), "INFO"
                )

                price_change = None
                if (
                    prev_signal.price_at_signal
                    and curr_signal.price_at_signal
                    and prev_signal.price_at_signal > 0
                ):
                    price_change = (
                        (curr_signal.price_at_signal - prev_signal.price_at_signal)
                        / prev_signal.price_at_signal
                        * 100
                    )

                changes.append(
                    {
                        "ticker": ticker,
                        "previous_signal": prev_signal.signal,
                        "current_signal": curr_signal.signal,
                        "urgency": urgency,
                        "change_type": "TRANSITION",
                        "previous_price": prev_signal.price_at_signal,
                        "current_price": curr_signal.price_at_signal,
                        "price_change_pct": price_change,
                        "current_upside": curr_signal.upside,
                        "current_buy_pct": curr_signal.buy_percentage,
                    }
                )

        # Sort by urgency (CRITICAL first)
        urgency_order = {
            "CRITICAL": 0,
            "HIGH": 1,
            "OPPORTUNITY": 2,
            "MEDIUM": 3,
            "LOW": 4,
            "INFO": 5,
        }
        changes.sort(key=lambda x: urgency_order.get(x["urgency"], 99))

        return changes

    def format_change_alert(self, change: Dict[str, Any]) -> str:
        """
        Format a change as a human-readable alert.

        Args:
            change: Change dictionary from detect_changes()

        Returns:
            Formatted alert string
        """
        urgency_emoji = {
            "CRITICAL": "\U0001F6A8",  # Police car light
            "HIGH": "\u26A0\uFE0F",  # Warning sign
            "OPPORTUNITY": "\U0001F7E2",  # Green circle
            "MEDIUM": "\U0001F7E1",  # Yellow circle
            "LOW": "\U0001F535",  # Blue circle
            "INFO": "\u2139\uFE0F",  # Information
        }
        emoji = urgency_emoji.get(change["urgency"], "\U0001F4CA")

        signal_names = {"B": "BUY", "S": "SELL", "H": "HOLD", "I": "INCONCLUSIVE"}

        if change["change_type"] == "NEW":
            curr_name = signal_names.get(
                change["current_signal"], change["current_signal"]
            )
            return f"{emoji} {change['ticker']}: NEW ({curr_name})"

        if change["change_type"] == "REMOVED":
            prev_name = signal_names.get(
                change["previous_signal"], change["previous_signal"]
            )
            return f"{emoji} {change['ticker']}: REMOVED (was {prev_name})"

        prev_name = signal_names.get(
            change["previous_signal"], change["previous_signal"]
        )
        curr_name = signal_names.get(
            change["current_signal"], change["current_signal"]
        )

        price_info = ""
        if change.get("price_change_pct") is not None:
            price_info = f" (price: {change['price_change_pct']:+.1f}%)"

        return f"{emoji} {change['ticker']}: {prev_name} -> {curr_name}{price_info}"

    def get_critical_alerts(
        self,
        current_date: str,
        previous_date: str,
    ) -> List[Dict[str, Any]]:
        """
        Get only CRITICAL and HIGH urgency changes.

        Args:
            current_date: Current date string (YYYY-MM-DD)
            previous_date: Previous date string (YYYY-MM-DD)

        Returns:
            List of critical/high urgency changes only
        """
        all_changes = self.detect_changes(current_date, previous_date)
        return [c for c in all_changes if c["urgency"] in ("CRITICAL", "HIGH")]

    def get_opportunities(
        self,
        current_date: str,
        previous_date: str,
    ) -> List[Dict[str, Any]]:
        """
        Get only OPPORTUNITY urgency changes (new buy signals).

        Args:
            current_date: Current date string (YYYY-MM-DD)
            previous_date: Previous date string (YYYY-MM-DD)

        Returns:
            List of opportunity changes only
        """
        all_changes = self.detect_changes(current_date, previous_date)
        return [c for c in all_changes if c["urgency"] == "OPPORTUNITY"]


def get_signal_changes(current_date: str, previous_date: str) -> List[Dict[str, Any]]:
    """
    Convenience function to get signal changes between two dates.

    Args:
        current_date: Current date (YYYY-MM-DD)
        previous_date: Previous date (YYYY-MM-DD)

    Returns:
        List of signal change dictionaries
    """
    detector = SignalChangeDetector(get_tracker())
    return detector.detect_changes(current_date, previous_date)


def format_signal_changes(changes: List[Dict[str, Any]]) -> List[str]:
    """
    Format a list of signal changes as human-readable alerts.

    Args:
        changes: List of change dictionaries

    Returns:
        List of formatted alert strings
    """
    detector = SignalChangeDetector()
    return [detector.format_change_alert(c) for c in changes]


def get_signal_change_summary(
    current_date: str, previous_date: str
) -> Dict[str, Any]:
    """
    Get a summary of signal changes between two dates.

    Args:
        current_date: Current date (YYYY-MM-DD)
        previous_date: Previous date (YYYY-MM-DD)

    Returns:
        Dictionary with change counts by urgency and type
    """
    detector = SignalChangeDetector(get_tracker())
    changes = detector.detect_changes(current_date, previous_date)

    # Count by urgency
    by_urgency: Dict[str, int] = {}
    for c in changes:
        urgency = c["urgency"]
        by_urgency[urgency] = by_urgency.get(urgency, 0) + 1

    # Count by change type
    by_type: Dict[str, int] = {}
    for c in changes:
        change_type = c["change_type"]
        by_type[change_type] = by_type.get(change_type, 0) + 1

    return {
        "total_changes": len(changes),
        "by_urgency": by_urgency,
        "by_type": by_type,
        "critical_count": by_urgency.get("CRITICAL", 0),
        "high_count": by_urgency.get("HIGH", 0),
        "opportunity_count": by_urgency.get("OPPORTUNITY", 0),
        "changes": changes,
    }
