"""
Signal Performance Tracking Module

Enables forward validation of trading signals by tracking price changes
at T+7, T+30, T+90 days after signal generation.

This addresses the critical gap identified in the hedge fund review:
- No backtesting available (historical target prices not available)
- Forward validation is the only way to measure signal quality
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_SIGNAL_LOG = Path(__file__).parent.parent / "yahoofinance" / "output" / "signal_log.jsonl"
DEFAULT_PERFORMANCE_LOG = Path(__file__).parent.parent / "yahoofinance" / "output" / "signal_performance.jsonl"

# Lock for thread-safe file operations
_file_lock = threading.Lock()


@dataclass
class SignalPerformance:
    """Performance data for a single signal."""
    ticker: str
    signal: str
    signal_date: str
    signal_price: Optional[float]
    spy_at_signal: Optional[float]

    # T+7 performance
    price_t7: Optional[float] = None
    return_t7: Optional[float] = None
    spy_return_t7: Optional[float] = None
    alpha_t7: Optional[float] = None

    # T+30 performance
    price_t30: Optional[float] = None
    return_t30: Optional[float] = None
    spy_return_t30: Optional[float] = None
    alpha_t30: Optional[float] = None

    # T+90 performance
    price_t90: Optional[float] = None
    return_t90: Optional[float] = None
    spy_return_t90: Optional[float] = None
    alpha_t90: Optional[float] = None

    # Metadata
    tier: Optional[str] = None
    region: Optional[str] = None
    upside_at_signal: Optional[float] = None
    buy_pct_at_signal: Optional[float] = None
    exret_at_signal: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_current_price(ticker: str) -> Optional[float]:
    """Get current price for a ticker using yfinance."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        price = stock.fast_info.get("lastPrice") or stock.fast_info.get("regularMarketPrice")
        return float(price) if price else None
    except Exception as e:
        logger.debug(f"Failed to get price for {ticker}: {e}")
        return None


def load_signals_needing_followup(
    signal_log_path: Path = DEFAULT_SIGNAL_LOG,
    days_threshold: int = 7
) -> List[Dict]:
    """
    Load signals that need performance follow-up.

    Args:
        signal_log_path: Path to signal log JSONL file
        days_threshold: Minimum days since signal for follow-up

    Returns:
        List of signal records needing follow-up
    """
    signals = []
    cutoff_date = datetime.now() - timedelta(days=days_threshold)

    if not signal_log_path.exists():
        logger.warning(f"Signal log not found: {signal_log_path}")
        return signals

    try:
        with _file_lock:
            with open(signal_log_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        signal_date = datetime.fromisoformat(record["timestamp"])

                        # Only include signals older than threshold that have price data
                        if signal_date < cutoff_date and record.get("price_at_signal"):
                            signals.append(record)
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.debug(f"Skipping malformed record: {e}")
                        continue
    except Exception as e:
        logger.error(f"Error reading signal log: {e}")

    return signals


def capture_performance(
    signal: Dict,
    performance_log_path: Path = DEFAULT_PERFORMANCE_LOG
) -> Optional[SignalPerformance]:
    """
    Capture performance data for a signal.

    Args:
        signal: Original signal record
        performance_log_path: Path to save performance data

    Returns:
        SignalPerformance object if successful
    """
    ticker = signal.get("ticker")
    if not ticker:
        return None

    signal_date = datetime.fromisoformat(signal["timestamp"])
    days_since_signal = (datetime.now() - signal_date).days

    # Get current prices
    current_price = get_current_price(ticker)
    spy_price = get_current_price("SPY")

    if current_price is None:
        logger.debug(f"Could not get current price for {ticker}")
        return None

    signal_price = signal.get("price_at_signal")
    spy_at_signal = signal.get("spy_price")

    # Calculate returns
    if signal_price and signal_price > 0:
        stock_return = (current_price - signal_price) / signal_price * 100
    else:
        stock_return = None

    if spy_at_signal and spy_price and spy_at_signal > 0:
        spy_return = (spy_price - spy_at_signal) / spy_at_signal * 100
    else:
        spy_return = None

    # Calculate alpha
    if stock_return is not None and spy_return is not None:
        alpha = stock_return - spy_return
    else:
        alpha = None

    # Create performance record
    perf = SignalPerformance(
        ticker=ticker,
        signal=signal.get("signal", "?"),
        signal_date=signal["timestamp"],
        signal_price=signal_price,
        spy_at_signal=spy_at_signal,
        tier=signal.get("tier"),
        region=signal.get("region"),
        upside_at_signal=signal.get("upside"),
        buy_pct_at_signal=signal.get("buy_percentage"),
        exret_at_signal=signal.get("exret"),
    )

    # Assign to appropriate time bucket
    if days_since_signal >= 90:
        perf.price_t90 = current_price
        perf.return_t90 = stock_return
        perf.spy_return_t90 = spy_return
        perf.alpha_t90 = alpha
    elif days_since_signal >= 30:
        perf.price_t30 = current_price
        perf.return_t30 = stock_return
        perf.spy_return_t30 = spy_return
        perf.alpha_t30 = alpha
    elif days_since_signal >= 7:
        perf.price_t7 = current_price
        perf.return_t7 = stock_return
        perf.spy_return_t7 = spy_return
        perf.alpha_t7 = alpha

    # Save to performance log
    try:
        with _file_lock:
            performance_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(performance_log_path, "a") as f:
                f.write(json.dumps(perf.to_dict()) + "\n")
    except Exception as e:
        logger.error(f"Failed to save performance data: {e}")

    return perf


def calculate_signal_stats(
    performance_log_path: Path = DEFAULT_PERFORMANCE_LOG
) -> Dict[str, Any]:
    """
    Calculate summary statistics for signal performance.

    Returns:
        Dictionary with performance statistics by signal type
    """
    stats = {
        "buy_signals": {"count": 0, "hit_rate_t30": None, "avg_return_t30": None, "avg_alpha_t30": None},
        "sell_signals": {"count": 0, "hit_rate_t30": None, "avg_return_t30": None, "avg_alpha_t30": None},
        "hold_signals": {"count": 0, "hit_rate_t30": None, "avg_return_t30": None, "avg_alpha_t30": None},
    }

    if not performance_log_path.exists():
        return stats

    buy_returns = []
    buy_alphas = []
    sell_returns = []
    sell_alphas = []

    try:
        with open(performance_log_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    signal = record.get("signal")
                    return_t30 = record.get("return_t30")
                    alpha_t30 = record.get("alpha_t30")

                    if return_t30 is not None:
                        if signal == "B":
                            stats["buy_signals"]["count"] += 1
                            buy_returns.append(return_t30)
                            if alpha_t30 is not None:
                                buy_alphas.append(alpha_t30)
                        elif signal == "S":
                            stats["sell_signals"]["count"] += 1
                            sell_returns.append(return_t30)
                            if alpha_t30 is not None:
                                sell_alphas.append(alpha_t30)
                        elif signal == "H":
                            stats["hold_signals"]["count"] += 1

                except (json.JSONDecodeError, KeyError) as e:
                    continue
    except Exception as e:
        logger.error(f"Error reading performance log: {e}")
        return stats

    # Calculate stats for BUY signals
    if buy_returns:
        stats["buy_signals"]["hit_rate_t30"] = sum(1 for r in buy_returns if r > 0) / len(buy_returns) * 100
        stats["buy_signals"]["avg_return_t30"] = sum(buy_returns) / len(buy_returns)
        if buy_alphas:
            stats["buy_signals"]["avg_alpha_t30"] = sum(buy_alphas) / len(buy_alphas)

    # Calculate stats for SELL signals (hit = negative return)
    if sell_returns:
        stats["sell_signals"]["hit_rate_t30"] = sum(1 for r in sell_returns if r < 0) / len(sell_returns) * 100
        stats["sell_signals"]["avg_return_t30"] = sum(sell_returns) / len(sell_returns)
        if sell_alphas:
            stats["sell_signals"]["avg_alpha_t30"] = sum(sell_alphas) / len(sell_alphas)

    return stats


def run_performance_capture():
    """
    Run performance capture for all signals needing follow-up.

    This should be run as a daily cron job:
    0 18 * * * python -c "from trade_modules.signal_performance import run_performance_capture; run_performance_capture()"
    """
    logger.info("Starting signal performance capture...")

    # Capture for different time horizons
    for days in [7, 30, 90]:
        signals = load_signals_needing_followup(days_threshold=days)
        captured = 0

        for signal in signals:
            result = capture_performance(signal)
            if result:
                captured += 1

        logger.info(f"Captured T+{days} performance for {captured} signals")

    # Print summary stats
    stats = calculate_signal_stats()
    logger.info(f"Performance Summary:")
    logger.info(f"  BUY signals: {stats['buy_signals']['count']} tracked")
    if stats['buy_signals']['hit_rate_t30']:
        logger.info(f"    Hit rate (T+30): {stats['buy_signals']['hit_rate_t30']:.1f}%")
        logger.info(f"    Avg return (T+30): {stats['buy_signals']['avg_return_t30']:.2f}%")
        if stats['buy_signals']['avg_alpha_t30']:
            logger.info(f"    Avg alpha (T+30): {stats['buy_signals']['avg_alpha_t30']:.2f}%")

    logger.info(f"  SELL signals: {stats['sell_signals']['count']} tracked")
    if stats['sell_signals']['hit_rate_t30']:
        logger.info(f"    Hit rate (T+30): {stats['sell_signals']['hit_rate_t30']:.1f}%")
        logger.info(f"    Avg return (T+30): {stats['sell_signals']['avg_return_t30']:.2f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_performance_capture()
