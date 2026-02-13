"""
A/B Testing Framework for Trading Thresholds

Provides tools to systematically test and compare different threshold
configurations before deploying to production.

P2 Improvement - Implemented from HEDGE_FUND_REVIEW.md recommendations.

Usage:
    from trade_modules.ab_testing import ThresholdTester

    tester = ThresholdTester()
    results = tester.compare_thresholds(
        market_df,
        threshold_a={'min_upside': 20, 'min_buy_percentage': 75},
        threshold_b={'min_upside': 15, 'min_buy_percentage': 80}
    )
    print(tester.format_comparison(results))
"""

import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ThresholdTester:
    """
    Test and compare different threshold configurations.

    Enables data-driven threshold tuning by comparing signal distributions
    and quality metrics across different configurations.
    """

    def __init__(
        self,
        signal_generator: Optional[Callable[[pd.DataFrame, Dict], pd.DataFrame]] = None,
    ):
        """
        Initialize threshold tester.

        Args:
            signal_generator: Optional custom signal generation function.
                If not provided, uses the default from analysis.signals.
                Signature: (df, thresholds) -> df_with_signals
        """
        self.signal_generator = signal_generator or self._default_signal_generator

    def _default_signal_generator(
        self, df: pd.DataFrame, thresholds: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Default signal generator using existing analysis logic.

        This is a simplified implementation. In production, you would
        integrate with the full signals.py logic.
        """
        result = df.copy()

        # Get threshold values with defaults
        min_upside = thresholds.get("min_upside", 20.0)
        min_buy_pct = thresholds.get("min_buy_percentage", 75.0)
        min_exret = thresholds.get("min_exret", 0.15) * 100  # Convert to %
        max_upside_sell = thresholds.get("max_upside_sell", 5.0)
        min_buy_pct_sell = thresholds.get("min_buy_percentage_sell", 65.0)

        # Find the relevant columns
        upside_col = None
        buy_pct_col = None
        exret_col = None

        for col in df.columns:
            col_lower = col.lower()
            if "upside" in col_lower:
                upside_col = col
            elif "buy" in col_lower and "%" in col:
                buy_pct_col = col
            elif col_lower == "%buy":
                buy_pct_col = col
            elif "exret" in col_lower:
                exret_col = col

        # Apply simple BUY/SELL/HOLD logic
        def classify_signal(row):
            upside = row.get(upside_col, 0) if upside_col else 0
            buy_pct = row.get(buy_pct_col, 0) if buy_pct_col else 0
            exret = row.get(exret_col, 0) if exret_col else 0

            # Parse percentages if needed
            if isinstance(upside, str):
                upside = float(upside.replace("%", "").strip())
            if isinstance(buy_pct, str):
                buy_pct = float(buy_pct.replace("%", "").strip())
            if isinstance(exret, str):
                exret = float(exret.replace("%", "").strip())

            # SELL conditions
            if upside < max_upside_sell or buy_pct < min_buy_pct_sell:
                return "S"

            # BUY conditions
            if upside >= min_upside and buy_pct >= min_buy_pct and exret >= min_exret:
                return "B"

            return "H"

        result["_test_signal"] = result.apply(classify_signal, axis=1)
        return result

    def compare_thresholds(
        self,
        market_df: pd.DataFrame,
        threshold_a: Dict[str, Any],
        threshold_b: Dict[str, Any],
        name_a: str = "Threshold A",
        name_b: str = "Threshold B",
    ) -> Dict[str, Any]:
        """
        Run signal generation with two threshold sets and compare results.

        Args:
            market_df: DataFrame with market data
            threshold_a: First threshold configuration
            threshold_b: Second threshold configuration
            name_a: Label for first configuration
            name_b: Label for second configuration

        Returns:
            Comparison results dictionary
        """
        if market_df.empty:
            return {"error": "Empty DataFrame provided"}

        # Run both configurations
        result_a = self.signal_generator(market_df.copy(), threshold_a)
        result_b = self.signal_generator(market_df.copy(), threshold_b)

        # Calculate metrics for each
        metrics_a = self._calculate_metrics(result_a)
        metrics_b = self._calculate_metrics(result_b)

        # Find signal differences
        signal_col = "_test_signal"
        if signal_col in result_a.columns and signal_col in result_b.columns:
            # Align on ticker
            ticker_col = self._find_ticker_column(result_a)
            if ticker_col:
                result_a = result_a.set_index(ticker_col)
                result_b = result_b.set_index(ticker_col)

                # Find flips
                common_idx = result_a.index.intersection(result_b.index)
                flips_a_to_b = []
                flips_b_to_a = []

                for idx in common_idx:
                    sig_a = result_a.loc[idx, signal_col]
                    sig_b = result_b.loc[idx, signal_col]
                    if sig_a != sig_b:
                        flips_a_to_b.append((idx, sig_a, sig_b))

        return {
            "name_a": name_a,
            "name_b": name_b,
            "threshold_a": threshold_a,
            "threshold_b": threshold_b,
            "metrics_a": metrics_a,
            "metrics_b": metrics_b,
            "signal_flips": flips_a_to_b if "flips_a_to_b" in dir() else [],
            "total_stocks": len(market_df),
        }

    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate quality metrics for a signal run."""
        signal_col = "_test_signal"

        if signal_col not in df.columns:
            return {"error": "No signal column found"}

        # Signal counts
        signal_counts = df[signal_col].value_counts().to_dict()

        # Find metric columns
        upside_col = None
        buy_pct_col = None
        exret_col = None

        for col in df.columns:
            col_lower = col.lower()
            if "upside" in col_lower:
                upside_col = col
            elif "buy" in col_lower:
                buy_pct_col = col
            elif "exret" in col_lower:
                exret_col = col

        # Calculate median quality metrics for BUY signals only
        buy_signals = df[df[signal_col] == "B"]

        metrics = {
            "buy_count": signal_counts.get("B", 0),
            "sell_count": signal_counts.get("S", 0),
            "hold_count": signal_counts.get("H", 0),
            "inconclusive_count": signal_counts.get("I", 0),
            "total": len(df),
        }

        if len(buy_signals) > 0 and upside_col:
            # Parse numeric values
            upside_vals = pd.to_numeric(
                buy_signals[upside_col].astype(str).str.replace("%", ""),
                errors="coerce",
            )
            metrics["median_buy_upside"] = float(upside_vals.median())
            metrics["mean_buy_upside"] = float(upside_vals.mean())

        if len(buy_signals) > 0 and buy_pct_col:
            buy_pct_vals = pd.to_numeric(
                buy_signals[buy_pct_col].astype(str).str.replace("%", ""),
                errors="coerce",
            )
            metrics["median_buy_pct"] = float(buy_pct_vals.median())

        if len(buy_signals) > 0 and exret_col:
            exret_vals = pd.to_numeric(
                buy_signals[exret_col].astype(str).str.replace("%", ""),
                errors="coerce",
            )
            metrics["median_buy_exret"] = float(exret_vals.median())

        return metrics

    def _find_ticker_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the ticker column in a DataFrame."""
        for col in ["ticker", "TICKER", "Ticker", "symbol", "SYMBOL"]:
            if col in df.columns:
                return col
        return None

    def format_comparison(self, results: Dict[str, Any]) -> str:
        """
        Format comparison results as human-readable text.

        Args:
            results: Results from compare_thresholds()

        Returns:
            Formatted string
        """
        if "error" in results:
            return f"Error: {results['error']}"

        lines = [
            "=" * 60,
            "THRESHOLD A/B COMPARISON",
            "=" * 60,
            "",
            f"Configuration A: {results['name_a']}",
            f"Configuration B: {results['name_b']}",
            f"Total stocks analyzed: {results['total_stocks']}",
            "",
            "-" * 40,
            "SIGNAL DISTRIBUTION",
            "-" * 40,
            "",
            f"{'Metric':<25} {'Config A':>12} {'Config B':>12} {'Delta':>10}",
            "-" * 60,
        ]

        metrics_a = results.get("metrics_a", {})
        metrics_b = results.get("metrics_b", {})

        # Compare key metrics
        for metric_name in [
            "buy_count",
            "sell_count",
            "hold_count",
            "median_buy_upside",
            "median_buy_pct",
            "median_buy_exret",
        ]:
            val_a = metrics_a.get(metric_name, 0)
            val_b = metrics_b.get(metric_name, 0)

            if val_a is None:
                val_a = 0
            if val_b is None:
                val_b = 0

            delta = val_b - val_a
            delta_str = f"{delta:+.1f}" if isinstance(delta, float) else f"{delta:+d}"

            lines.append(f"{metric_name:<25} {val_a:>12.1f} {val_b:>12.1f} {delta_str:>10}")

        # Signal flips
        flips = results.get("signal_flips", [])
        if flips:
            lines.extend(
                [
                    "",
                    "-" * 40,
                    f"SIGNAL FLIPS ({len(flips)} stocks)",
                    "-" * 40,
                ]
            )
            for ticker, sig_a, sig_b in flips[:10]:
                signal_map = {"B": "BUY", "S": "SELL", "H": "HOLD", "I": "INC"}
                lines.append(
                    f"  {ticker}: {signal_map.get(sig_a, sig_a)} -> {signal_map.get(sig_b, sig_b)}"
                )
            if len(flips) > 10:
                lines.append(f"  ... and {len(flips) - 10} more")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def run_parameter_sweep(
        self,
        market_df: pd.DataFrame,
        parameter: str,
        values: List[Any],
        base_thresholds: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Sweep a single parameter across multiple values.

        Args:
            market_df: Market data DataFrame
            parameter: Parameter name to vary
            values: List of values to test
            base_thresholds: Base threshold configuration

        Returns:
            List of results for each parameter value
        """
        results = []

        for value in values:
            thresholds = deepcopy(base_thresholds)
            thresholds[parameter] = value

            result_df = self.signal_generator(market_df.copy(), thresholds)
            metrics = self._calculate_metrics(result_df)
            metrics["parameter"] = parameter
            metrics["value"] = value
            results.append(metrics)

        return results

    def format_sweep_results(
        self, results: List[Dict[str, Any]], parameter: str
    ) -> str:
        """
        Format parameter sweep results as a table.

        Args:
            results: Results from run_parameter_sweep()
            parameter: Name of the swept parameter

        Returns:
            Formatted string
        """
        lines = [
            f"Parameter Sweep: {parameter}",
            "=" * 70,
            f"{'Value':>10} {'BUY':>8} {'SELL':>8} {'HOLD':>8} {'Med Upside':>12} {'Med Buy%':>10}",
            "-" * 70,
        ]

        for r in results:
            value = r.get("value", "N/A")
            buy = r.get("buy_count", 0)
            sell = r.get("sell_count", 0)
            hold = r.get("hold_count", 0)
            med_upside = r.get("median_buy_upside", 0) or 0
            med_buy_pct = r.get("median_buy_pct", 0) or 0

            lines.append(
                f"{value:>10} {buy:>8} {sell:>8} {hold:>8} {med_upside:>12.1f}% {med_buy_pct:>10.1f}%"
            )

        return "\n".join(lines)


def compare_threshold_configs(
    market_df: pd.DataFrame,
    config_a: Dict[str, Any],
    config_b: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convenience function to compare two threshold configurations.

    Args:
        market_df: Market data DataFrame
        config_a: First threshold configuration
        config_b: Second threshold configuration

    Returns:
        Comparison results dictionary
    """
    tester = ThresholdTester()
    return tester.compare_thresholds(market_df, config_a, config_b)


def sweep_threshold_parameter(
    market_df: pd.DataFrame,
    parameter: str,
    values: List[Any],
    base_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to sweep a single parameter.

    Args:
        market_df: Market data DataFrame
        parameter: Parameter name to vary
        values: List of values to test
        base_config: Base threshold configuration

    Returns:
        List of results for each parameter value
    """
    if base_config is None:
        base_config = {
            "min_upside": 20.0,
            "min_buy_percentage": 75.0,
            "min_exret": 0.15,
            "max_upside_sell": 5.0,
            "min_buy_percentage_sell": 65.0,
        }

    tester = ThresholdTester()
    return tester.run_parameter_sweep(market_df, parameter, values, base_config)
