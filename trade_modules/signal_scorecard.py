"""
Monthly Signal Scorecard

Evaluates trading signal accuracy at multiple horizons (1M, 3M, 6M)
by comparing predicted signals against actual price movements.
Builds on BacktestEngine infrastructure for data loading and price fetching.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from trade_modules.backtest_engine import BacktestEngine, SIGNAL_LOG_PATH, OUTPUT_DIR

logger = logging.getLogger(__name__)

# Trading days per horizon
HORIZON_DAYS = {
    '1m': 21,
    '3m': 63,
    '6m': 126,
}

SCORECARD_OUTPUT_PATH = OUTPUT_DIR / "signal_scorecard.json"


class SignalScorecard:
    """
    Builds a scorecard measuring signal accuracy at 1M, 3M, 6M horizons.

    Uses BacktestEngine.load_signals() and fetch_price_history() to avoid
    duplicating data loading logic.
    """

    def __init__(
        self,
        signal_log_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        self.output_dir = output_dir or OUTPUT_DIR
        self.engine = BacktestEngine(
            signal_log_path=signal_log_path,
            output_dir=self.output_dir,
            horizons=list(HORIZON_DAYS.values()),
        )

    def generate_scorecard(self, months_back: int = 3) -> Dict[str, Any]:
        """
        Generate a signal scorecard for signals from the last N months.

        Args:
            months_back: How many months of signal history to evaluate.

        Returns:
            Dict with overall, by_tier, by_region breakdowns and calibration alerts.
        """
        # Load signals
        all_signals = self.engine.load_signals()
        if all_signals.empty:
            return self._empty_scorecard(months_back)

        # Filter to date range
        cutoff = datetime.now() - timedelta(days=months_back * 30)
        cutoff_date = cutoff.date()
        signals_df = all_signals[all_signals['date'] >= cutoff_date].copy()

        if signals_df.empty:
            return self._empty_scorecard(months_back)

        # Fetch price data
        tickers = signals_df['ticker'].unique().tolist()
        min_date = signals_df['date'].min() - timedelta(days=5)
        max_date = datetime.now().date()
        price_data, spy_data = self.engine.fetch_price_history(
            tickers, min_date, max_date
        )

        # Backfill missing signal prices
        signals_df = self.engine.backfill_signal_prices(signals_df, price_data)

        # Calculate returns at each horizon
        horizon_results = {}
        for label, trading_days in HORIZON_DAYS.items():
            results = self.engine.calculate_returns(
                signals_df, price_data, spy_data, trading_days
            )
            if not results.empty:
                # Merge tier/region from signals
                results = self._merge_tier_region(results, signals_df)
                horizon_results[label] = results

        # Build scorecard structure
        period_start = signals_df['date'].min()
        period_end = signals_df['date'].max()

        scorecard: Dict[str, Any] = {
            'generated_at': datetime.now().strftime('%Y-%m-%d'),
            'period': f"{period_start} to {period_end}",
            'overall': self._compute_overall(horizon_results),
            'by_tier': self._compute_breakdown(horizon_results, 'tier'),
            'by_region': self._compute_breakdown(horizon_results, 'region'),
            'calibration_alerts': self._generate_alerts(horizon_results),
        }

        # Save to file
        self._save_scorecard(scorecard)

        return scorecard

    def print_scorecard(self, scorecard: Optional[Dict[str, Any]] = None) -> None:
        """Print scorecard to console in a readable format."""
        if scorecard is None:
            # Try to load from file
            if SCORECARD_OUTPUT_PATH.exists():
                with open(SCORECARD_OUTPUT_PATH, 'r') as f:
                    scorecard = json.load(f)
            else:
                print("No scorecard found. Run generate_scorecard() first.")
                return

        print("\n" + "=" * 70)
        print("  SIGNAL SCORECARD")
        print(f"  Period: {scorecard['period']}")
        print(f"  Generated: {scorecard['generated_at']}")
        print("=" * 70)

        # Overall stats
        print("\nOVERALL PERFORMANCE:")
        print("-" * 70)
        for signal_type in ['buy', 'sell', 'hold']:
            stats = scorecard['overall'].get(signal_type, {})
            if not stats or stats.get('count', 0) == 0:
                continue
            print(f"\n  {signal_type.upper()} signals (n={stats['count']}):")
            for horizon in ['1m', '3m', '6m']:
                hr = stats.get(f'hit_rate_{horizon}')
                outperf = stats.get(f'outperformance_rate_{horizon}')
                if hr is not None:
                    outperf_str = f", outperf={outperf:.1f}%" if outperf is not None else ""
                    print(f"    {horizon.upper()}: hit_rate={hr:.1f}%{outperf_str}")

        # By tier
        print("\nBY TIER (BUY signals):")
        print("-" * 70)
        print(f"  {'Tier':<8} {'Count':>6} {'HR 1M':>8} {'HR 3M':>8} {'HR 6M':>8} {'OP 1M':>8}")
        print("  " + "-" * 48)
        for tier in ['mega', 'large', 'mid', 'small', 'micro']:
            tier_data = scorecard['by_tier'].get(tier, {}).get('buy', {})
            if not tier_data or tier_data.get('count', 0) == 0:
                continue
            hr1 = tier_data.get('hit_rate_1m', '-')
            hr3 = tier_data.get('hit_rate_3m', '-')
            hr6 = tier_data.get('hit_rate_6m', '-')
            op1 = tier_data.get('outperformance_rate_1m', '-')
            hr1_s = f"{hr1:.1f}%" if isinstance(hr1, (int, float)) else hr1
            hr3_s = f"{hr3:.1f}%" if isinstance(hr3, (int, float)) else hr3
            hr6_s = f"{hr6:.1f}%" if isinstance(hr6, (int, float)) else hr6
            op1_s = f"{op1:.1f}%" if isinstance(op1, (int, float)) else op1
            print(f"  {tier:<8} {tier_data['count']:>6} {hr1_s:>8} {hr3_s:>8} {hr6_s:>8} {op1_s:>8}")

        # By region
        print("\nBY REGION (BUY signals):")
        print("-" * 70)
        print(f"  {'Region':<8} {'Count':>6} {'HR 1M':>8} {'HR 3M':>8} {'HR 6M':>8}")
        print("  " + "-" * 38)
        for region in ['us', 'eu', 'hk']:
            region_data = scorecard['by_region'].get(region, {}).get('buy', {})
            if not region_data or region_data.get('count', 0) == 0:
                continue
            hr1 = region_data.get('hit_rate_1m', '-')
            hr3 = region_data.get('hit_rate_3m', '-')
            hr6 = region_data.get('hit_rate_6m', '-')
            hr1_s = f"{hr1:.1f}%" if isinstance(hr1, (int, float)) else hr1
            hr3_s = f"{hr3:.1f}%" if isinstance(hr3, (int, float)) else hr3
            hr6_s = f"{hr6:.1f}%" if isinstance(hr6, (int, float)) else hr6
            print(f"  {region:<8} {region_data['count']:>6} {hr1_s:>8} {hr3_s:>8} {hr6_s:>8}")

        # Calibration alerts
        alerts = scorecard.get('calibration_alerts', [])
        if alerts:
            print("\nCALIBRATION ALERTS:")
            print("-" * 70)
            for alert in alerts:
                print(f"  ! {alert}")

        print()

    def _compute_overall(
        self, horizon_results: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Compute overall hit rates by signal type across horizons."""
        overall: Dict[str, Any] = {}
        signal_map = {'B': 'buy', 'S': 'sell', 'H': 'hold'}

        for signal_code, signal_name in signal_map.items():
            stats: Dict[str, Any] = {'count': 0}

            for label, results in horizon_results.items():
                sig_df = results[results['signal'] == signal_code]
                if sig_df.empty:
                    continue

                if stats['count'] == 0:
                    stats['count'] = len(sig_df)

                returns = sig_df['stock_return'].dropna()
                spy_returns = sig_df['spy_return'].dropna()
                alphas = sig_df['alpha'].dropna()
                n = len(returns)

                if n == 0:
                    continue

                stats[f'hit_rate_{label}'] = round(
                    self._hit_rate(returns, signal_code), 1
                )
                stats[f'false_positive_rate_{label}'] = round(
                    self._false_positive_rate(returns, signal_code), 1
                )

                # Outperformance vs SPY
                valid_alpha = alphas[alphas.notna()]
                if len(valid_alpha) > 0 and signal_code == 'B':
                    outperf = float((valid_alpha > 0).mean() * 100)
                    stats[f'outperformance_rate_{label}'] = round(outperf, 1)

                # False negative rate (for SELL/HOLD: stocks that gained >20%)
                if signal_code in ('S', 'H'):
                    fn_rate = float((returns > 20).mean() * 100) if n > 0 else 0.0
                    stats[f'false_negative_rate_{label}'] = round(fn_rate, 1)

            if stats['count'] > 0:
                overall[signal_name] = stats

        return overall

    def _compute_breakdown(
        self,
        horizon_results: Dict[str, pd.DataFrame],
        group_col: str,
    ) -> Dict[str, Any]:
        """Compute hit rates broken down by a grouping column (tier or region)."""
        breakdown: Dict[str, Any] = {}
        signal_map = {'B': 'buy', 'S': 'sell', 'H': 'hold'}

        # Collect all group values across horizons
        all_groups = set()
        for results in horizon_results.values():
            if group_col in results.columns:
                all_groups.update(results[group_col].dropna().unique())

        for group_val in sorted(all_groups):
            group_data: Dict[str, Any] = {}

            for signal_code, signal_name in signal_map.items():
                stats: Dict[str, Any] = {'count': 0}

                for label, results in horizon_results.items():
                    if group_col not in results.columns:
                        continue
                    mask = (results[group_col] == group_val) & (results['signal'] == signal_code)
                    sig_df = results[mask]
                    if sig_df.empty:
                        continue

                    if stats['count'] == 0:
                        stats['count'] = len(sig_df)

                    returns = sig_df['stock_return'].dropna()
                    alphas = sig_df['alpha'].dropna()
                    n = len(returns)
                    if n == 0:
                        continue

                    stats[f'hit_rate_{label}'] = round(
                        self._hit_rate(returns, signal_code), 1
                    )
                    stats[f'false_positive_rate_{label}'] = round(
                        self._false_positive_rate(returns, signal_code), 1
                    )

                    if signal_code == 'B' and len(alphas) > 0:
                        outperf = float((alphas > 0).mean() * 100)
                        stats[f'outperformance_rate_{label}'] = round(outperf, 1)

                    if signal_code in ('S', 'H') and n > 0:
                        fn_rate = float((returns > 20).mean() * 100)
                        stats[f'false_negative_rate_{label}'] = round(fn_rate, 1)

                if stats['count'] > 0:
                    group_data[signal_name] = stats

            if group_data:
                breakdown[str(group_val)] = group_data

        return breakdown

    def _generate_alerts(
        self, horizon_results: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """Generate calibration alerts for underperforming tier/region combos."""
        alerts = []

        # Check 1M horizon for tier+region combos with low hit rates
        results_1m = horizon_results.get('1m', pd.DataFrame())
        if results_1m.empty:
            return alerts

        buy_1m = results_1m[results_1m['signal'] == 'B']
        if buy_1m.empty:
            return alerts

        for (tier, region), gdf in buy_1m.groupby(['tier', 'region']):
            if pd.isna(tier) or pd.isna(region):
                continue
            returns = gdf['stock_return'].dropna()
            n = len(returns)
            if n < 10:
                continue
            hr = float((returns > 0).mean() * 100)
            if hr < 50:
                alerts.append(
                    f"{region.upper()} {tier.upper()} BUY has only {hr:.0f}% "
                    f"hit rate at T+21 (n={n}) - review thresholds"
                )

        # Check SELL signals - alert if hit rate is low
        sell_1m = results_1m[results_1m['signal'] == 'S']
        if not sell_1m.empty:
            for (tier, region), gdf in sell_1m.groupby(['tier', 'region']):
                if pd.isna(tier) or pd.isna(region):
                    continue
                returns = gdf['stock_return'].dropna()
                n = len(returns)
                if n < 10:
                    continue
                hr = float((returns < 0).mean() * 100)
                if hr < 50:
                    alerts.append(
                        f"{region.upper()} {tier.upper()} SELL has only {hr:.0f}% "
                        f"hit rate at T+21 (n={n}) - review thresholds"
                    )

        return alerts

    @staticmethod
    def _hit_rate(returns: pd.Series, signal: str) -> float:
        """Calculate hit rate: % of signals where outcome matched prediction."""
        n = len(returns)
        if n == 0:
            return 0.0
        if signal == 'B':
            return float((returns > 0).mean() * 100)
        elif signal == 'S':
            return float((returns < 0).mean() * 100)
        else:  # H
            return float((returns.abs() < 5).mean() * 100)

    @staticmethod
    def _false_positive_rate(returns: pd.Series, signal: str) -> float:
        """Calculate false positive rate."""
        n = len(returns)
        if n == 0:
            return 0.0
        if signal == 'B':
            # BUY signals that lost money
            return float((returns < 0).mean() * 100)
        elif signal == 'S':
            # SELL signals where stock went up
            return float((returns > 0).mean() * 100)
        else:
            return 0.0

    @staticmethod
    def _merge_tier_region(
        results: pd.DataFrame, signals_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Ensure tier and region columns exist in results from signal data."""
        if 'tier' in results.columns and 'region' in results.columns:
            return results

        # Merge from signals on ticker + signal_date
        signals_lookup = signals_df[['ticker', 'date', 'tier', 'region']].copy()
        signals_lookup['date'] = signals_lookup['date'].astype(str)
        results = results.copy()
        results['_date_str'] = results['signal_date'].astype(str)

        merged = results.merge(
            signals_lookup.rename(columns={'date': '_date_str'}),
            on=['ticker', '_date_str'],
            how='left',
            suffixes=('', '_sig'),
        )

        # Use signal tier/region if missing from results
        if 'tier_sig' in merged.columns:
            merged['tier'] = merged['tier'].fillna(merged['tier_sig'])
            merged['region'] = merged['region'].fillna(merged['region_sig'])
            merged = merged.drop(columns=['tier_sig', 'region_sig'], errors='ignore')

        merged = merged.drop(columns=['_date_str'], errors='ignore')
        return merged

    def _save_scorecard(self, scorecard: Dict[str, Any]) -> None:
        """Save scorecard to JSON file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "signal_scorecard.json"
        with open(output_path, 'w') as f:
            json.dump(scorecard, f, indent=2, default=str)
        print(f"\nScorecard saved to: {output_path}")

    @staticmethod
    def _empty_scorecard(months_back: int) -> Dict[str, Any]:
        """Return an empty scorecard structure."""
        now = datetime.now()
        return {
            'generated_at': now.strftime('%Y-%m-%d'),
            'period': f"{(now - timedelta(days=months_back * 30)).strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}",
            'overall': {},
            'by_tier': {},
            'by_region': {},
            'calibration_alerts': [],
        }


def run_scorecard(months_back: int = 3) -> None:
    """Entry point for scorecard CLI command."""
    scorecard = SignalScorecard()
    result = scorecard.generate_scorecard(months_back=months_back)
    scorecard.print_scorecard(result)
