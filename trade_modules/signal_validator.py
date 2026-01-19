"""
Signal Validation and Backtesting System

Analyzes historical signal performance to validate framework effectiveness
and identify improvement opportunities.

Since traditional backtesting is not feasible (no historical target prices),
this system performs forward validation using logged signals.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import statistics

logger = logging.getLogger(__name__)

DEFAULT_SIGNAL_LOG_PATH = Path(__file__).parent.parent / "yahoofinance" / "output" / "signal_log.jsonl"


@dataclass
class ValidationResult:
    """Result of validating a single signal."""
    ticker: str
    signal: str
    signal_date: datetime
    price_at_signal: float
    target_price: Optional[float]
    current_price: Optional[float]
    days_elapsed: int
    price_change_pct: Optional[float]
    hit_target: Optional[bool]
    excess_return: Optional[float]  # Return vs SPY benchmark
    tier: Optional[str]
    region: Optional[str]
    spy_price_at_signal: Optional[float] = None  # SPY price when signal was logged
    spy_current_price: Optional[float] = None  # Current SPY price
    spy_return_pct: Optional[float] = None  # SPY return over same period
    metrics_at_signal: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationSummary:
    """Summary statistics for signal validation."""
    total_signals: int
    validated_signals: int
    hit_rate: float  # % that hit target
    avg_return: float
    median_return: float
    excess_vs_benchmark: float
    by_signal_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_tier: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_region: Dict[str, Dict[str, float]] = field(default_factory=dict)
    improvement_suggestions: List[str] = field(default_factory=list)


class SignalValidator:
    """
    Validates historical signals against actual price performance.

    This is the core backtesting/forward-testing infrastructure.
    """

    def __init__(
        self,
        log_path: Optional[Path] = None,
        benchmark_ticker: str = "SPY"
    ):
        """
        Initialize validator.

        Args:
            log_path: Path to signal log file
            benchmark_ticker: Ticker for benchmark comparison (default SPY)
        """
        self.log_path = log_path or DEFAULT_SIGNAL_LOG_PATH
        self.benchmark_ticker = benchmark_ticker
        self._price_cache: Dict[str, Dict[str, float]] = {}

    def load_signals(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        signal_type: Optional[str] = None,
        tier: Optional[str] = None,
        region: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load signals from log file with optional filtering.

        Args:
            start_date: Filter signals after this date
            end_date: Filter signals before this date
            signal_type: Filter by signal type (B/S/H)
            tier: Filter by market cap tier
            region: Filter by geographic region

        Returns:
            List of signal dictionaries
        """
        signals = []

        if not self.log_path.exists():
            logger.warning(f"Signal log not found: {self.log_path}")
            return signals

        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)

                        # Parse timestamp
                        ts = datetime.fromisoformat(data["timestamp"])

                        # Apply filters
                        if start_date and ts < start_date:
                            continue
                        if end_date and ts > end_date:
                            continue
                        if signal_type and data.get("signal") != signal_type:
                            continue
                        if tier and data.get("tier", "").lower() != tier.lower():
                            continue
                        if region and data.get("region", "").lower() != region.lower():
                            continue

                        data["_timestamp"] = ts
                        signals.append(data)

                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.debug(f"Skipping malformed record: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error loading signals: {e}")

        return signals

    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get current price for ticker.

        Uses caching to minimize API calls.
        """
        if ticker in self._price_cache:
            cache_entry = self._price_cache[ticker]
            # Cache valid for 1 hour
            if (datetime.now() - cache_entry["timestamp"]).seconds < 3600:
                return cache_entry["price"]

        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
                self._price_cache[ticker] = {
                    "price": price,
                    "timestamp": datetime.now()
                }
                return price
        except Exception as e:
            logger.debug(f"Failed to get price for {ticker}: {e}")

        return None

    def validate_signal(
        self,
        signal: Dict[str, Any],
        current_price: Optional[float] = None,
        spy_current_price: Optional[float] = None
    ) -> ValidationResult:
        """
        Validate a single signal against current price.

        Args:
            signal: Signal dictionary from log
            current_price: Current price (fetched if not provided)
            spy_current_price: Current SPY price for benchmark (fetched if not provided)

        Returns:
            ValidationResult with performance metrics
        """
        ticker = signal["ticker"]
        signal_date = signal.get("_timestamp") or datetime.fromisoformat(signal["timestamp"])
        price_at_signal = signal.get("price_at_signal")
        target_price = signal.get("target_price")
        spy_price_at_signal = signal.get("spy_price")

        if current_price is None:
            current_price = self.get_current_price(ticker)

        if spy_current_price is None:
            spy_current_price = self.get_current_price(self.benchmark_ticker)

        days_elapsed = (datetime.now() - signal_date).days

        # Calculate price change
        price_change_pct = None
        hit_target = None
        spy_return_pct = None
        excess_return = None

        if price_at_signal and current_price:
            price_change_pct = ((current_price - price_at_signal) / price_at_signal) * 100

            if target_price:
                # For BUY signals, check if reached target
                if signal["signal"] == "B":
                    hit_target = current_price >= target_price
                # For SELL signals, check if dropped below signal price
                elif signal["signal"] == "S":
                    hit_target = current_price <= price_at_signal

        # Calculate SPY benchmark return and excess return
        if spy_price_at_signal and spy_current_price:
            spy_return_pct = ((spy_current_price - spy_price_at_signal) / spy_price_at_signal) * 100
            if price_change_pct is not None:
                excess_return = price_change_pct - spy_return_pct

        return ValidationResult(
            ticker=ticker,
            signal=signal["signal"],
            signal_date=signal_date,
            price_at_signal=price_at_signal,
            target_price=target_price,
            current_price=current_price,
            days_elapsed=days_elapsed,
            price_change_pct=price_change_pct,
            hit_target=hit_target,
            excess_return=excess_return,
            tier=signal.get("tier"),
            region=signal.get("region"),
            spy_price_at_signal=spy_price_at_signal,
            spy_current_price=spy_current_price,
            spy_return_pct=spy_return_pct,
            metrics_at_signal={
                "upside": signal.get("upside"),
                "buy_percentage": signal.get("buy_percentage"),
                "exret": signal.get("exret"),
                "vix_level": signal.get("vix_level"),
                # Additional metrics
                "pe_forward": signal.get("pe_forward"),
                "pe_trailing": signal.get("pe_trailing"),
                "peg": signal.get("peg"),
                "short_interest": signal.get("short_interest"),
                "roe": signal.get("roe"),
                "debt_equity": signal.get("debt_equity"),
                "pct_52w_high": signal.get("pct_52w_high"),
                "sell_triggers": signal.get("sell_triggers", []),
            }
        )

    def validate_signals_batch(
        self,
        signals: List[Dict[str, Any]],
        min_days: int = 30,
        max_days: int = 180
    ) -> List[ValidationResult]:
        """
        Validate multiple signals with age filtering.

        Args:
            signals: List of signal dictionaries
            min_days: Minimum days since signal (for maturity)
            max_days: Maximum days since signal (for relevance)

        Returns:
            List of ValidationResult objects
        """
        results = []
        now = datetime.now()

        for signal in signals:
            signal_date = signal.get("_timestamp") or datetime.fromisoformat(signal["timestamp"])
            days_elapsed = (now - signal_date).days

            if days_elapsed < min_days or days_elapsed > max_days:
                continue

            result = self.validate_signal(signal)
            results.append(result)

        return results

    def generate_summary(
        self,
        results: List[ValidationResult]
    ) -> ValidationSummary:
        """
        Generate summary statistics from validation results.

        Args:
            results: List of ValidationResult objects

        Returns:
            ValidationSummary with aggregate statistics
        """
        if not results:
            return ValidationSummary(
                total_signals=0,
                validated_signals=0,
                hit_rate=0.0,
                avg_return=0.0,
                median_return=0.0,
                excess_vs_benchmark=0.0
            )

        # Filter to results with price data
        valid_results = [r for r in results if r.price_change_pct is not None]

        # Calculate overall metrics
        returns = [r.price_change_pct for r in valid_results]
        hits = [r for r in valid_results if r.hit_target is True]
        excess_returns = [r.excess_return for r in valid_results if r.excess_return is not None]

        hit_rate = (len(hits) / len(valid_results) * 100) if valid_results else 0
        avg_return = statistics.mean(returns) if returns else 0
        median_return = statistics.median(returns) if returns else 0
        avg_excess_return = statistics.mean(excess_returns) if excess_returns else 0

        # Group by signal type
        by_signal = {}
        for signal_type in ["B", "S", "H"]:
            type_results = [r for r in valid_results if r.signal == signal_type]
            if type_results:
                type_returns = [r.price_change_pct for r in type_results]
                type_hits = [r for r in type_results if r.hit_target is True]
                by_signal[signal_type] = {
                    "count": len(type_results),
                    "hit_rate": len(type_hits) / len(type_results) * 100,
                    "avg_return": statistics.mean(type_returns),
                    "median_return": statistics.median(type_returns)
                }

        # Group by tier
        by_tier = {}
        for tier in ["mega", "large", "mid", "small", "micro"]:
            tier_results = [r for r in valid_results if r.tier and r.tier.lower() == tier]
            if tier_results:
                tier_returns = [r.price_change_pct for r in tier_results]
                tier_hits = [r for r in tier_results if r.hit_target is True]
                by_tier[tier] = {
                    "count": len(tier_results),
                    "hit_rate": len(tier_hits) / len(tier_results) * 100 if tier_results else 0,
                    "avg_return": statistics.mean(tier_returns),
                    "median_return": statistics.median(tier_returns)
                }

        # Group by region
        by_region = {}
        for region in ["us", "eu", "hk"]:
            region_results = [r for r in valid_results if r.region and r.region.lower() == region]
            if region_results:
                region_returns = [r.price_change_pct for r in region_results]
                region_hits = [r for r in region_results if r.hit_target is True]
                by_region[region] = {
                    "count": len(region_results),
                    "hit_rate": len(region_hits) / len(region_results) * 100 if region_results else 0,
                    "avg_return": statistics.mean(region_returns),
                    "median_return": statistics.median(region_returns)
                }

        # Generate improvement suggestions
        suggestions = self._generate_suggestions(by_signal, by_tier, by_region, valid_results)

        return ValidationSummary(
            total_signals=len(results),
            validated_signals=len(valid_results),
            hit_rate=hit_rate,
            avg_return=avg_return,
            median_return=median_return,
            excess_vs_benchmark=avg_excess_return,
            by_signal_type=by_signal,
            by_tier=by_tier,
            by_region=by_region,
            improvement_suggestions=suggestions
        )

    def _generate_suggestions(
        self,
        by_signal: Dict,
        by_tier: Dict,
        by_region: Dict,
        results: List[ValidationResult]
    ) -> List[str]:
        """Generate automated improvement suggestions based on performance."""
        suggestions = []

        # Check BUY signal effectiveness
        if "B" in by_signal:
            buy_stats = by_signal["B"]
            if buy_stats["hit_rate"] < 50:
                suggestions.append(
                    f"BUY hit rate is {buy_stats['hit_rate']:.1f}% - consider tightening criteria"
                )
            if buy_stats["avg_return"] < 0:
                suggestions.append(
                    f"BUY signals averaging {buy_stats['avg_return']:.1f}% return - review upside thresholds"
                )

        # Check SELL signal effectiveness
        if "S" in by_signal:
            sell_stats = by_signal["S"]
            if sell_stats["avg_return"] > 0:
                suggestions.append(
                    f"SELL signals actually gained {sell_stats['avg_return']:.1f}% - may be too aggressive"
                )

        # Check tier performance
        for tier, stats in by_tier.items():
            if stats["count"] >= 10:
                if stats["avg_return"] < -10:
                    suggestions.append(
                        f"{tier.upper()} tier underperforming ({stats['avg_return']:.1f}%) - review thresholds"
                    )

        # Check region performance
        for region, stats in by_region.items():
            if stats["count"] >= 10:
                if stats["hit_rate"] < 40:
                    suggestions.append(
                        f"{region.upper()} region hit rate low ({stats['hit_rate']:.1f}%) - review thresholds"
                    )

        # Analyze metric correlations
        buy_results = [r for r in results if r.signal == "B" and r.price_change_pct is not None]
        if len(buy_results) >= 10:
            # Check if high EXR correlates with success
            high_exret = [r for r in buy_results if r.metrics_at_signal.get("exret", 0) and r.metrics_at_signal["exret"] > 15]
            low_exret = [r for r in buy_results if r.metrics_at_signal.get("exret", 0) and r.metrics_at_signal["exret"] <= 15]

            if high_exret and low_exret:
                high_avg = statistics.mean([r.price_change_pct for r in high_exret])
                low_avg = statistics.mean([r.price_change_pct for r in low_exret])

                if high_avg > low_avg + 5:
                    suggestions.append(
                        f"High EXRET signals outperform by {high_avg - low_avg:.1f}% - consider raising min_exret"
                    )

        if not suggestions:
            suggestions.append("Framework performing within expected parameters - no immediate changes recommended")

        return suggestions

    def generate_report(
        self,
        start_date: Optional[datetime] = None,
        min_days: int = 30,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate comprehensive validation report.

        Args:
            start_date: Start date for signal analysis
            min_days: Minimum days for signal maturity
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        # Load signals
        signals = self.load_signals(start_date=start_date)

        if not signals:
            return "No signals found for validation period"

        # Validate signals
        results = self.validate_signals_batch(signals, min_days=min_days)

        # Generate summary
        summary = self.generate_summary(results)

        # Format report
        report = []
        report.append("=" * 60)
        report.append("SIGNAL VALIDATION REPORT")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 60)
        report.append("")
        report.append(f"Total Signals Analyzed: {summary.total_signals}")
        report.append(f"Signals with Price Data: {summary.validated_signals}")
        report.append(f"Overall Hit Rate: {summary.hit_rate:.1f}%")
        report.append(f"Average Return: {summary.avg_return:.2f}%")
        report.append(f"Median Return: {summary.median_return:.2f}%")
        report.append(f"Excess vs SPY Benchmark: {summary.excess_vs_benchmark:+.2f}%")
        report.append("")

        if summary.by_signal_type:
            report.append("-" * 40)
            report.append("BY SIGNAL TYPE")
            report.append("-" * 40)
            for sig_type, stats in summary.by_signal_type.items():
                label = {"B": "BUY", "S": "SELL", "H": "HOLD"}.get(sig_type, sig_type)
                report.append(f"{label}:")
                report.append(f"  Count: {stats['count']}")
                report.append(f"  Hit Rate: {stats['hit_rate']:.1f}%")
                report.append(f"  Avg Return: {stats['avg_return']:.2f}%")
            report.append("")

        if summary.by_tier:
            report.append("-" * 40)
            report.append("BY MARKET CAP TIER")
            report.append("-" * 40)
            for tier, stats in summary.by_tier.items():
                report.append(f"{tier.upper()}:")
                report.append(f"  Count: {stats['count']}")
                report.append(f"  Hit Rate: {stats['hit_rate']:.1f}%")
                report.append(f"  Avg Return: {stats['avg_return']:.2f}%")
            report.append("")

        if summary.by_region:
            report.append("-" * 40)
            report.append("BY REGION")
            report.append("-" * 40)
            for region, stats in summary.by_region.items():
                report.append(f"{region.upper()}:")
                report.append(f"  Count: {stats['count']}")
                report.append(f"  Hit Rate: {stats['hit_rate']:.1f}%")
                report.append(f"  Avg Return: {stats['avg_return']:.2f}%")
            report.append("")

        report.append("=" * 60)
        report.append("IMPROVEMENT SUGGESTIONS")
        report.append("=" * 60)
        for i, suggestion in enumerate(summary.improvement_suggestions, 1):
            report.append(f"{i}. {suggestion}")
        report.append("")

        report_text = "\n".join(report)

        if output_path:
            output_path.write_text(report_text)
            logger.info(f"Report saved to {output_path}")

        return report_text


def run_validation(
    min_days: int = 30,
    output_path: Optional[str] = None
) -> str:
    """
    Convenience function to run validation.

    Args:
        min_days: Minimum days for signal maturity
        output_path: Optional path to save report

    Returns:
        Validation report string
    """
    validator = SignalValidator()

    path = Path(output_path) if output_path else None
    return validator.generate_report(min_days=min_days, output_path=path)


if __name__ == "__main__":
    # Run validation when executed directly
    import sys

    min_days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    output = sys.argv[2] if len(sys.argv) > 2 else None

    report = run_validation(min_days=min_days, output_path=output)
    print(report)
