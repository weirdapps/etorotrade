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
from datetime import datetime
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
