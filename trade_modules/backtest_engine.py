"""
Backtest Engine for Trading Signal Validation

Performs forward validation of historical trading signals by measuring
actual price changes at T+7 and T+30 trading days after signal generation.

Since Yahoo Finance doesn't provide historical analyst recommendations,
traditional backtesting is impossible. Instead, we use the signal_log.jsonl
accumulated since Jan 14, 2026 to measure how well signals predicted
actual price movements.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Paths
SIGNAL_LOG_PATH = Path(__file__).parent.parent / "yahoofinance" / "output" / "signal_log.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent / "yahoofinance" / "output"
CACHE_PATH = OUTPUT_DIR / ".backtest_price_cache.parquet"
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# Regex for test tickers to exclude
TEST_TICKER_RE = re.compile(r'^(STOCK\d+|BUY\d+|SELL\d+|HOLD\d+|WEAK\d*)$')

# Valid signals for analysis
VALID_SIGNALS = {'B', 'S', 'H'}


class BacktestEngine:
    """
    Forward-validates trading signals against actual price movements.

    Pipeline:
    1. Load signals from signal_log.jsonl
    2. Fetch historical price data via yfinance
    3. Backfill missing signal prices
    4. Calculate returns at T+7 and T+30
    5. Compute statistics by signal/tier/region
    6. Output results to console and CSV
    """

    def __init__(
        self,
        signal_log_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        cache_path: Optional[Path] = None,
        horizons: Optional[List[int]] = None,
    ):
        self.signal_log_path = signal_log_path or SIGNAL_LOG_PATH
        self.output_dir = output_dir or OUTPUT_DIR
        self.cache_path = cache_path or CACHE_PATH
        self.horizons = horizons or [7, 30]

    def run(self) -> None:
        """Execute the full backtest pipeline."""
        print("\n=== BACKTEST: Forward Validation of Trading Signals ===\n")

        # Step 1: Load signals
        signals_df = self.load_signals()
        if signals_df.empty:
            print("No signals found to backtest.")
            return
        print(f"Loaded {len(signals_df)} unique signal-date pairs")

        # Step 2: Fetch price data
        tickers = signals_df['ticker'].unique().tolist()
        min_date = signals_df['date'].min() - timedelta(days=5)
        max_date = datetime.now().date()
        print(f"Fetching price history for {len(tickers)} tickers...")
        price_data, spy_data = self.fetch_price_history(
            tickers, min_date, max_date
        )
        print(f"Price data: {len(price_data.columns)} tickers, {len(price_data)} trading days")

        # Step 3: Backfill missing signal prices
        signals_df = self.backfill_signal_prices(signals_df, price_data)
        has_price = signals_df['price_at_signal'].notna().sum()
        print(f"Signals with price data: {has_price}/{len(signals_df)}")

        # Step 4: Calculate returns for each horizon
        all_results = []
        for horizon in self.horizons:
            print(f"\nCalculating T+{horizon} returns...")
            results = self.calculate_returns(signals_df, price_data, spy_data, horizon)
            if not results.empty:
                results['horizon'] = horizon
                all_results.append(results)
                print(f"  {len(results)} signals with T+{horizon} data")

        if not all_results:
            print("No return data could be calculated.")
            return

        results_df = pd.concat(all_results, ignore_index=True)

        # Step 5: Calculate statistics
        stats = self.calculate_statistics(results_df)

        # Step 6: Output
        self.print_summary(stats)
        self.save_results(results_df, stats)

    def load_signals(self) -> pd.DataFrame:
        """
        Load and deduplicate signals from JSONL log.

        Returns:
            DataFrame with columns: ticker, signal, timestamp, date,
            price_at_signal, tier, region, upside, buy_percentage,
            exret, sell_triggers, and additional metrics.
        """
        if not self.signal_log_path.exists():
            logger.warning(f"Signal log not found: {self.signal_log_path}")
            return pd.DataFrame()

        records = []
        with open(self.signal_log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ticker = data.get('ticker', '')
                signal = data.get('signal', '')

                # Filter test tickers
                if TEST_TICKER_RE.match(ticker):
                    continue

                # Only keep B/S/H signals
                if signal not in VALID_SIGNALS:
                    continue

                records.append(data)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date

        # Deduplicate: keep latest signal per ticker per calendar date
        df = df.sort_values('timestamp')
        df = df.drop_duplicates(subset=['ticker', 'date'], keep='last')

        return df.reset_index(drop=True)

    def fetch_price_history(
        self,
        tickers: List[str],
        start_date,
        end_date,
        batch_size: int = 500,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Bulk-fetch historical closing prices via yfinance.

        Returns:
            Tuple of (price_data DataFrame indexed by date with ticker columns,
                      spy_data Series indexed by date)
        """
        import yfinance as yf

        # Load cache if available
        cached_df = pd.DataFrame()
        if self.cache_path.exists():
            try:
                cached_df = pd.read_parquet(self.cache_path)
                logger.info(f"Loaded price cache: {cached_df.shape}")
            except Exception as e:
                logger.debug(f"Cache load failed: {e}")

        # Determine which tickers need fetching
        if not cached_df.empty:
            cached_tickers = set(cached_df.columns)
            missing_tickers = [t for t in tickers if t not in cached_tickers]
        else:
            missing_tickers = list(tickers)

        # Fetch missing tickers in batches
        new_frames = []
        if missing_tickers:
            for i in range(0, len(missing_tickers), batch_size):
                batch = missing_tickers[i:i + batch_size]
                try:
                    data = yf.download(
                        batch,
                        start=str(start_date),
                        end=str(end_date),
                        group_by='ticker',
                        threads=True,
                        progress=False,
                        auto_adjust=True,
                    )
                    if data.empty:
                        continue

                    # Extract Close prices
                    if len(batch) == 1:
                        # Single ticker: data has simple columns
                        close = data[['Close']].rename(columns={'Close': batch[0]})
                    else:
                        # Multi-ticker: MultiIndex columns
                        close = data.xs('Close', axis=1, level=1) if isinstance(
                            data.columns, pd.MultiIndex
                        ) else data[['Close']]
                    new_frames.append(close)
                except Exception as e:
                    logger.warning(f"Failed to fetch batch {i}-{i + len(batch)}: {e}")

        # Merge with cache
        all_frames = []
        if not cached_df.empty:
            all_frames.append(cached_df)
        all_frames.extend(new_frames)

        if all_frames:
            price_data = pd.concat(all_frames, axis=1)
            # Remove duplicate columns
            price_data = price_data.loc[:, ~price_data.columns.duplicated()]
        else:
            price_data = pd.DataFrame()

        # Save updated cache
        if not price_data.empty and new_frames:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                price_data.to_parquet(self.cache_path)
                logger.info(f"Saved price cache: {price_data.shape}")
            except Exception as e:
                logger.debug(f"Cache save failed: {e}")

        # Fetch SPY separately
        spy_data = pd.Series(dtype=float)
        if 'SPY' in price_data.columns:
            spy_data = price_data['SPY'].dropna()
        else:
            try:
                spy_raw = yf.download(
                    'SPY',
                    start=str(start_date),
                    end=str(end_date),
                    progress=False,
                    auto_adjust=True,
                )
                if not spy_raw.empty:
                    spy_data = spy_raw['Close'].squeeze()
            except Exception as e:
                logger.warning(f"Failed to fetch SPY: {e}")

        # Ensure index is DatetimeIndex
        if not isinstance(price_data.index, pd.DatetimeIndex):
            price_data.index = pd.to_datetime(price_data.index)
        if not isinstance(spy_data.index, pd.DatetimeIndex):
            spy_data.index = pd.to_datetime(spy_data.index)

        return price_data, spy_data

    def backfill_signal_prices(
        self, signals_df: pd.DataFrame, price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fill missing price_at_signal from historical price data.

        For signals where price_at_signal is null, looks up the close price
        on the signal date (or nearest prior trading day).
        """
        df = signals_df.copy()
        mask = df['price_at_signal'].isna()

        if not mask.any() or price_data.empty:
            return df

        for idx in df[mask].index:
            ticker = df.at[idx, 'ticker']
            signal_date = pd.Timestamp(df.at[idx, 'date'])

            if ticker not in price_data.columns:
                continue

            ticker_prices = price_data[ticker].dropna()
            if ticker_prices.empty:
                continue

            # Find nearest trading day on or before signal date
            valid = ticker_prices.index[ticker_prices.index <= signal_date]
            if len(valid) > 0:
                df.at[idx, 'price_at_signal'] = float(ticker_prices.loc[valid[-1]])

        return df

    def calculate_returns(
        self,
        signals_df: pd.DataFrame,
        price_data: pd.DataFrame,
        spy_data: pd.Series,
        horizon: int,
    ) -> pd.DataFrame:
        """
        Calculate actual returns at T+horizon trading days for each signal.

        Uses index-based offset (trading days) not calendar days.
        """
        results = []

        for _, row in signals_df.iterrows():
            ticker = row['ticker']
            signal_price = row.get('price_at_signal')

            if pd.isna(signal_price) or signal_price is None or signal_price <= 0:
                continue

            if ticker not in price_data.columns:
                continue

            signal_date = pd.Timestamp(row['date'])
            ticker_prices = price_data[ticker].dropna()

            # Find signal date position in trading calendar
            valid_dates = ticker_prices.index[ticker_prices.index >= signal_date]
            if len(valid_dates) <= horizon:
                continue  # Not enough future data

            future_date = valid_dates[horizon]
            future_price = float(ticker_prices.loc[future_date])
            stock_return = (future_price - signal_price) / signal_price * 100

            # SPY benchmark
            spy_return = np.nan
            alpha = np.nan
            if not spy_data.empty:
                spy_valid = spy_data.index[spy_data.index >= signal_date]
                if len(spy_valid) > horizon:
                    spy_at_signal_date = spy_valid[0]
                    spy_future_date = spy_valid[horizon]
                    spy_at_signal = float(spy_data.loc[spy_at_signal_date])
                    spy_future = float(spy_data.loc[spy_future_date])
                    if spy_at_signal > 0:
                        spy_return = (spy_future - spy_at_signal) / spy_at_signal * 100
                        alpha = stock_return - spy_return

            results.append({
                'ticker': ticker,
                'signal': row['signal'],
                'signal_date': row['date'],
                'tier': row.get('tier'),
                'region': row.get('region'),
                'price_at_signal': signal_price,
                'future_price': future_price,
                'stock_return': stock_return,
                'spy_return': spy_return,
                'alpha': alpha,
            })

        return pd.DataFrame(results)

    def calculate_statistics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute aggregate statistics from backtest results.

        Groups by signal type, tier, region, and cross-tabs.
        """
        stats: Dict[str, Any] = {}

        for horizon in results_df['horizon'].unique():
            hdf = results_df[results_df['horizon'] == horizon]
            h_key = f"t{horizon}"
            stats[h_key] = {}

            # By signal type
            stats[h_key]['by_signal'] = self._group_stats(hdf, 'signal')

            # By tier
            stats[h_key]['by_tier'] = self._group_stats(hdf, 'tier')

            # By region
            stats[h_key]['by_region'] = self._group_stats(hdf, 'region')

            # Cross-tab: signal x tier
            cross = {}
            for (sig, tier), gdf in hdf.groupby(['signal', 'tier']):
                if pd.isna(tier):
                    continue
                key = f"{tier}_{sig}"
                cross[key] = self._compute_group_stats(gdf, sig)
            stats[h_key]['signal_x_tier'] = cross

        return stats

    def _group_stats(self, df: pd.DataFrame, group_col: str) -> Dict[str, Any]:
        """Compute stats grouped by a column."""
        result = {}
        for group_val, gdf in df.groupby(group_col):
            if pd.isna(group_val):
                continue
            signal = gdf['signal'].mode().iloc[0] if group_col != 'signal' else group_val
            result[str(group_val)] = self._compute_group_stats(gdf, signal)
        return result

    @staticmethod
    def _compute_group_stats(gdf: pd.DataFrame, signal_type: str) -> Dict[str, Any]:
        """Compute hit rate, mean/median return, avg alpha for a group."""
        n = len(gdf)
        returns = gdf['stock_return'].dropna()
        alphas = gdf['alpha'].dropna()

        if signal_type == 'B':
            hits = (returns > 0).sum()
        elif signal_type == 'S':
            hits = (returns < 0).sum()
        else:  # H
            hits = (returns.abs() < 5).sum()

        return {
            'count': n,
            'hit_rate': round(hits / n * 100, 1) if n > 0 else 0,
            'mean_return': round(returns.mean(), 2) if len(returns) > 0 else None,
            'median_return': round(returns.median(), 2) if len(returns) > 0 else None,
            'avg_alpha': round(alphas.mean(), 2) if len(alphas) > 0 else None,
            'low_sample': n < 30,
        }

    def print_summary(self, stats: Dict[str, Any]) -> None:
        """Print formatted summary to console."""
        signal_names = {'B': 'BUY', 'S': 'SELL', 'H': 'HOLD'}

        for h_key in sorted(stats.keys()):
            h_stats = stats[h_key]
            print("\n" + "=" * 60)
            print(f"  {h_key.upper()} PERFORMANCE SUMMARY")
            print("=" * 60)

            # By signal type
            print(f"\n{'Signal':<8} {'Count':>7} {'Hit Rate':>10} {'Mean Ret':>10} {'Med Ret':>10} {'Avg Alpha':>11}")
            print('-' * 58)
            for sig in ['B', 'S', 'H']:
                s = h_stats['by_signal'].get(sig, {})
                if not s:
                    continue
                name = signal_names.get(sig, sig)
                flag = '*' if s.get('low_sample') else ''
                mr = f"{s['mean_return']:.2f}%" if s.get('mean_return') is not None else 'N/A'
                mdr = f"{s['median_return']:.2f}%" if s.get('median_return') is not None else 'N/A'
                aa = f"{s['avg_alpha']:.2f}%" if s.get('avg_alpha') is not None else 'N/A'
                print(f"{name:<8} {s['count']:>7} {s['hit_rate']:>9.1f}% {mr:>10} {mdr:>10} {aa:>11} {flag}")

            # By tier
            if h_stats.get('by_tier'):
                print(f"\n{'Tier':<8} {'Count':>7} {'Hit Rate':>10} {'Mean Ret':>10} {'Avg Alpha':>11}")
                print('-' * 48)
                for tier in ['mega', 'large', 'mid', 'small', 'micro']:
                    s = h_stats['by_tier'].get(tier, {})
                    if not s:
                        continue
                    flag = '*' if s.get('low_sample') else ''
                    mr = f"{s['mean_return']:.2f}%" if s.get('mean_return') is not None else 'N/A'
                    aa = f"{s['avg_alpha']:.2f}%" if s.get('avg_alpha') is not None else 'N/A'
                    print(f"{tier:<8} {s['count']:>7} {s['hit_rate']:>9.1f}% {mr:>10} {aa:>11} {flag}")

            print("\n  * = low sample size (n < 30)")

    def save_results(
        self, results_df: pd.DataFrame, stats: Dict[str, Any]
    ) -> None:
        """Save detailed results and summary to CSV."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Per-signal detail
        detail_path = self.output_dir / "backtest_results.csv"
        results_df.to_csv(detail_path, index=False)
        print(f"\nDetailed results saved to: {detail_path}")

        # Summary table
        summary_rows = []
        for h_key, h_stats in stats.items():
            for group_type in ['by_signal', 'by_tier', 'by_region', 'signal_x_tier']:
                for group_val, s in h_stats.get(group_type, {}).items():
                    summary_rows.append({
                        'horizon': h_key,
                        'group_type': group_type,
                        'group': group_val,
                        **s,
                    })

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = self.output_dir / "backtest_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"Summary saved to: {summary_path}")


class ThresholdAnalyzer:
    """
    Analyzes metric predictiveness and threshold effectiveness.

    Uses backtest results to evaluate how well each metric predicts
    returns and whether current config.yaml thresholds are optimal.
    """

    def __init__(
        self,
        signal_log_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        self.signal_log_path = signal_log_path or SIGNAL_LOG_PATH
        self.config_path = config_path or CONFIG_PATH
        self.output_dir = output_dir or OUTPUT_DIR

    def run(self, results_df: pd.DataFrame) -> None:
        """Run threshold analysis on backtest results."""
        print("\n" + "=" * 60)
        print("  THRESHOLD ANALYSIS REPORT")
        print("=" * 60)

        # Load full signal data with metrics for correlation analysis
        signals_df = self._load_signals_with_metrics()
        if signals_df.empty:
            print("No signal data available for threshold analysis.")
            return

        # Merge signals with T+30 results
        t30_results = results_df[results_df['horizon'] == 30].copy()
        if t30_results.empty:
            print("No T+30 results available for threshold analysis.")
            return

        merged = self._merge_signals_with_results(signals_df, t30_results)
        if merged.empty:
            print("No merged data for analysis.")
            return

        print(f"\nAnalyzing {len(merged)} signals with T+30 returns")

        # 4a: Metric predictiveness
        correlations = self.analyze_metric_predictiveness(merged)

        # 4b: Sell trigger effectiveness
        trigger_stats = self.analyze_sell_triggers(merged)

        # 4c: Threshold recommendations
        suggestions = self.suggest_thresholds(merged)

        # 4d: Regime-stratified analysis
        regime_stats = self.analyze_by_regime(merged)

        # 4e: Output
        self._print_report(correlations, trigger_stats, suggestions, regime_stats)
        self._save_report(correlations, trigger_stats, suggestions, regime_stats)

    def _load_signals_with_metrics(self) -> pd.DataFrame:
        """Load signals from JSONL including all metric columns."""
        if not self.signal_log_path.exists():
            return pd.DataFrame()

        records = []
        with open(self.signal_log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ticker = data.get('ticker', '')
                signal = data.get('signal', '')

                if TEST_TICKER_RE.match(ticker):
                    continue
                if signal not in VALID_SIGNALS:
                    continue

                records.append(data)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df = df.sort_values('timestamp')
        df = df.drop_duplicates(subset=['ticker', 'date'], keep='last')
        return df.reset_index(drop=True)

    @staticmethod
    def _merge_signals_with_results(
        signals_df: pd.DataFrame, results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge signal metrics with calculated returns."""
        # Normalize date columns for merge
        signals_df = signals_df.copy()
        results_df = results_df.copy()
        signals_df['date_str'] = signals_df['date'].astype(str)
        results_df['date_str'] = results_df['signal_date'].astype(str)

        merged = signals_df.merge(
            results_df[['ticker', 'date_str', 'stock_return', 'spy_return', 'alpha']],
            on=['ticker', 'date_str'],
            how='inner',
        )
        return merged

    def analyze_metric_predictiveness(
        self, merged: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Calculate correlation of each metric with T+30 return.

        For BUY signals, a positive correlation means the metric
        correctly predicts higher returns.
        """
        metrics = [
            'upside', 'buy_percentage', 'exret', 'roe', 'debt_equity',
            'pct_52w_high', 'pe_forward', 'pe_trailing', 'sentiment_score',
        ]

        correlations = []
        buy_df = merged[merged['signal'] == 'B']

        for metric in metrics:
            if metric not in merged.columns:
                continue

            series = buy_df[metric].astype(float, errors='ignore')
            returns = buy_df['stock_return']

            # Drop NaN pairs
            valid = series.notna() & returns.notna()
            n = valid.sum()

            if n < 30:
                correlations.append({
                    'metric': metric,
                    'correlation': None,
                    'coverage': n,
                    'total': len(buy_df),
                    'coverage_pct': round(n / len(buy_df) * 100, 1) if len(buy_df) > 0 else 0,
                    'note': 'insufficient data',
                })
                continue

            try:
                corr = float(series[valid].corr(returns[valid]))
            except (ValueError, TypeError):
                corr = None

            # Quintile analysis
            quintile_stats = []
            if corr is not None and n >= 50:
                try:
                    quantiles = pd.qcut(series[valid], 5, labels=False, duplicates='drop')
                    for q in sorted(quantiles.unique()):
                        q_mask = quantiles == q
                        q_returns = returns[valid][q_mask]
                        quintile_stats.append({
                            'quintile': int(q),
                            'count': int(q_mask.sum()),
                            'mean_return': round(float(q_returns.mean()), 2),
                            'hit_rate': round(float((q_returns > 0).mean() * 100), 1),
                        })
                except (ValueError, TypeError):
                    pass

            strength = 'weak'
            if corr is not None:
                abs_corr = abs(corr)
                if abs_corr >= 0.3:
                    strength = 'strong'
                elif abs_corr >= 0.15:
                    strength = 'moderate'

            correlations.append({
                'metric': metric,
                'correlation': round(corr, 3) if corr is not None else None,
                'strength': strength,
                'coverage': int(n),
                'total': len(buy_df),
                'coverage_pct': round(n / len(buy_df) * 100, 1) if len(buy_df) > 0 else 0,
                'quintiles': quintile_stats,
            })

        return correlations

    def analyze_sell_triggers(
        self, merged: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Evaluate effectiveness of each sell trigger type.

        Hit = stock actually declined at T+30 after SELL signal.
        """
        sell_df = merged[merged['signal'] == 'S'].copy()
        if sell_df.empty:
            return []

        # Explode sell_triggers column
        if 'sell_triggers' not in sell_df.columns:
            return []

        trigger_stats = {}
        for _, row in sell_df.iterrows():
            triggers = row.get('sell_triggers', [])
            if not isinstance(triggers, list):
                continue
            stock_return = row.get('stock_return')
            if pd.isna(stock_return):
                continue

            for trigger in triggers:
                trigger = str(trigger).strip()
                # Skip boolean-like entries
                if trigger.lower() in ('true', 'false', ''):
                    continue

                if trigger not in trigger_stats:
                    trigger_stats[trigger] = {'fires': 0, 'hits': 0, 'returns': []}

                trigger_stats[trigger]['fires'] += 1
                if stock_return < 0:
                    trigger_stats[trigger]['hits'] += 1
                trigger_stats[trigger]['returns'].append(stock_return)

        results = []
        for trigger, data in trigger_stats.items():
            n = data['fires']
            if n < 10:
                continue
            hit_rate = data['hits'] / n * 100
            avg_return = np.mean(data['returns'])
            results.append({
                'trigger': trigger,
                'fires': n,
                'hit_rate': round(hit_rate, 1),
                'avg_return': round(avg_return, 2),
                'effectiveness': round(hit_rate * n / 1000, 1),  # hit_rate * frequency
            })

        results.sort(key=lambda x: x['effectiveness'], reverse=True)
        return results

    def suggest_thresholds(
        self, merged: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Compare current config thresholds against data-optimal values.

        Only suggests changes where improvement > 5% hit rate and n >= 50.
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return []

        suggestions = []
        tier_region_combos = [
            ('us', 'mega'), ('us', 'large'), ('us', 'mid'),
            ('us', 'small'), ('us', 'micro'),
            ('eu', 'mega'), ('eu', 'large'), ('eu', 'mid'),
            ('hk', 'mega'), ('hk', 'large'),
        ]

        for region, tier in tier_region_combos:
            config_key = f"{region}_{tier}"
            tier_config = config.get(config_key, {})
            if not tier_config:
                continue

            buy_config = tier_config.get('buy', {})
            sell_config = tier_config.get('sell', {})

            # Filter data for this tier-region
            tier_data = merged[
                (merged['tier'] == tier) & (merged['region'] == region)
            ]

            if len(tier_data) < 50:
                continue

            buy_data = tier_data[tier_data['signal'] == 'B']
            sell_data = tier_data[tier_data['signal'] == 'S']

            # Analyze BUY thresholds
            buy_metrics = {
                'min_upside': 'upside',
                'min_buy_percentage': 'buy_percentage',
                'min_exret': 'exret',
                'min_roe': 'roe',
            }

            for config_key_name, metric_col in buy_metrics.items():
                current_val = buy_config.get(config_key_name)
                if current_val is None or metric_col not in buy_data.columns:
                    continue

                suggestion = self._find_optimal_threshold(
                    buy_data, metric_col, current_val,
                    direction='min', signal='B'
                )
                if suggestion:
                    suggestion['tier_region'] = config_key
                    suggestion['config_key'] = config_key_name
                    suggestion['signal_type'] = 'BUY'
                    suggestions.append(suggestion)

            # Analyze SELL thresholds
            sell_metrics = {
                'max_exret': 'exret',
                'max_upside': 'upside',
            }

            for config_key_name, metric_col in sell_metrics.items():
                current_val = sell_config.get(config_key_name)
                if current_val is None or metric_col not in sell_data.columns:
                    continue

                suggestion = self._find_optimal_threshold(
                    sell_data, metric_col, current_val,
                    direction='max', signal='S'
                )
                if suggestion:
                    suggestion['tier_region'] = config_key
                    suggestion['config_key'] = config_key_name
                    suggestion['signal_type'] = 'SELL'
                    suggestions.append(suggestion)

        return suggestions

    @staticmethod
    def _find_optimal_threshold(
        data: pd.DataFrame,
        metric_col: str,
        current_val: float,
        direction: str,
        signal: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Find the threshold value that maximizes hit rate.

        Args:
            data: DataFrame with metric and stock_return columns
            metric_col: Name of metric column
            current_val: Current config threshold value
            direction: 'min' for >= threshold, 'max' for <= threshold
            signal: 'B' or 'S' - determines what counts as a hit

        Returns:
            Suggestion dict or None if no meaningful improvement found
        """
        valid = data[[metric_col, 'stock_return']].dropna()
        if len(valid) < 50:
            return None

        metric_vals = valid[metric_col].astype(float)
        returns = valid['stock_return']

        # Test range of thresholds
        percentiles = np.percentile(metric_vals, [10, 20, 30, 40, 50, 60, 70, 80, 90])
        best_hr = -1.0
        best_val = current_val

        for test_val in percentiles:
            if direction == 'min':
                mask = metric_vals >= test_val
            else:
                mask = metric_vals <= test_val

            group_returns = returns[mask]
            if len(group_returns) < 30:
                continue

            if signal == 'B':
                hr = float((group_returns > 0).mean() * 100)
            else:
                hr = float((group_returns < 0).mean() * 100)

            if hr > best_hr:
                best_hr = hr
                best_val = round(float(test_val), 1)

        # Calculate current hit rate
        if direction == 'min':
            current_mask = metric_vals >= current_val
        else:
            current_mask = metric_vals <= current_val

        current_returns = returns[current_mask]
        if len(current_returns) < 10:
            current_hr = 0.0
        elif signal == 'B':
            current_hr = float((current_returns > 0).mean() * 100)
        else:
            current_hr = float((current_returns < 0).mean() * 100)

        improvement = best_hr - current_hr

        # Only suggest if improvement > 5% and sufficient sample
        if improvement <= 5.0 or best_val == current_val:
            return None

        return {
            'current_value': current_val,
            'suggested_value': best_val,
            'current_hit_rate': round(current_hr, 1),
            'suggested_hit_rate': round(best_hr, 1),
            'improvement': round(improvement, 1),
            'sample_size': len(valid),
        }

    def analyze_by_regime(
        self, merged: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Analyze signal performance stratified by market regime.

        Returns per-regime stats for BUY and SELL signals.
        """
        if 'regime' not in merged.columns:
            return []

        regime_stats = []
        for regime_val in merged['regime'].dropna().unique():
            regime_df = merged[merged['regime'] == regime_val]

            for signal_type in ['B', 'S']:
                sig_df = regime_df[regime_df['signal'] == signal_type]
                returns = sig_df['stock_return'].dropna()
                n = len(returns)
                if n < 10:
                    continue

                if signal_type == 'B':
                    hits = int((returns > 0).sum())
                else:
                    hits = int((returns < 0).sum())

                alphas = sig_df['alpha'].dropna()

                regime_stats.append({
                    'regime': str(regime_val),
                    'signal': signal_type,
                    'count': n,
                    'hit_rate': round(hits / n * 100, 1),
                    'mean_return': round(float(returns.mean()), 2),
                    'avg_alpha': round(float(alphas.mean()), 2) if len(alphas) > 0 else None,
                })

        return regime_stats

    def _print_report(
        self,
        correlations: List[Dict[str, Any]],
        trigger_stats: List[Dict[str, Any]],
        suggestions: List[Dict[str, Any]],
        regime_stats: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Print threshold analysis report to console."""
        # Metric predictiveness
        print("\nMETRIC PREDICTIVENESS (correlation with T+30 return, BUY signals):")
        print('-' * 65)
        for c in correlations:
            corr_str = f"r={c['correlation']:.3f}" if c['correlation'] is not None else "N/A"
            strength = c.get('strength', c.get('note', ''))
            coverage = f"{c['coverage_pct']:.0f}% coverage"
            print(f"  {c['metric']:<20} {corr_str:<12} ({strength}, {coverage})")

        # Sell trigger effectiveness
        if trigger_stats:
            print("\nSELL TRIGGER EFFECTIVENESS:")
            print('-' * 65)
            print(f"  {'Trigger':<30} {'Fires':>7} {'Hit Rate':>10} {'Avg Ret':>10}")
            print('  ' + '-' * 59)
            for t in trigger_stats[:15]:  # Top 15
                label = 'EFFECTIVE' if t['hit_rate'] >= 65 else (
                    'MODERATE' if t['hit_rate'] >= 55 else 'WEAK'
                )
                print(
                    f"  {t['trigger']:<30} {t['fires']:>7} "
                    f"{t['hit_rate']:>8.1f}%  {t['avg_return']:>8.2f}%  {label}"
                )

        # Threshold suggestions
        if suggestions:
            print("\nTHRESHOLD SUGGESTIONS (>5% hit rate improvement, n>=50):")
            print('-' * 75)
            for s in suggestions:
                print(
                    f"  {s['tier_region']} {s['signal_type']} {s['config_key']}: "
                    f"current={s['current_value']}, suggested={s['suggested_value']}, "
                    f"hit rate +{s['improvement']:.1f}% "
                    f"(n={s['sample_size']})"
                )
        else:
            print("\nNo threshold suggestions (current thresholds are near-optimal or insufficient data).")

        # Regime-stratified performance
        if regime_stats:
            print("\nPERFORMANCE BY MARKET REGIME:")
            print('-' * 65)
            signal_names = {'B': 'BUY', 'S': 'SELL'}
            print(f"  {'Regime':<12} {'Signal':<6} {'Count':>7} {'Hit Rate':>10} {'Mean Ret':>10} {'Avg Alpha':>11}")
            print('  ' + '-' * 58)
            for rs in sorted(regime_stats, key=lambda x: (x['regime'], x['signal'])):
                sig_name = signal_names.get(rs['signal'], rs['signal'])
                aa = f"{rs['avg_alpha']:.2f}%" if rs.get('avg_alpha') is not None else 'N/A'
                print(
                    f"  {rs['regime']:<12} {sig_name:<6} {rs['count']:>7} "
                    f"{rs['hit_rate']:>9.1f}% {rs['mean_return']:>9.2f}% {aa:>11}"
                )

    def _save_report(
        self,
        correlations: List[Dict[str, Any]],
        trigger_stats: List[Dict[str, Any]],
        suggestions: List[Dict[str, Any]],
        regime_stats: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Save threshold analysis report to CSV."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "backtest_threshold_report.csv"

        rows = []

        # Correlations section
        for c in correlations:
            rows.append({
                'section': 'metric_predictiveness',
                'item': c['metric'],
                'value': c.get('correlation'),
                'detail': c.get('strength', c.get('note', '')),
                'coverage': c.get('coverage_pct'),
                'sample_size': c.get('coverage'),
            })

        # Trigger stats section
        for t in trigger_stats:
            rows.append({
                'section': 'sell_trigger_effectiveness',
                'item': t['trigger'],
                'value': t['hit_rate'],
                'detail': f"fires={t['fires']}, avg_return={t['avg_return']}%",
                'coverage': None,
                'sample_size': t['fires'],
            })

        # Suggestions section
        for s in suggestions:
            rows.append({
                'section': 'threshold_suggestion',
                'item': f"{s['tier_region']}_{s['signal_type']}_{s['config_key']}",
                'value': s['suggested_value'],
                'detail': (
                    f"current={s['current_value']}, "
                    f"improvement=+{s['improvement']}%"
                ),
                'coverage': None,
                'sample_size': s['sample_size'],
            })

        # Regime stats section
        for rs in (regime_stats or []):
            rows.append({
                'section': 'regime_performance',
                'item': f"{rs['regime']}_{rs['signal']}",
                'value': rs['hit_rate'],
                'detail': f"mean_return={rs['mean_return']}%, alpha={rs.get('avg_alpha', 'N/A')}",
                'coverage': None,
                'sample_size': rs['count'],
            })

        if rows:
            pd.DataFrame(rows).to_csv(report_path, index=False)
            print(f"Threshold report saved to: {report_path}")


def run_backtest() -> None:
    """Entry point for backtest CLI command."""
    engine = BacktestEngine()
    engine.run()

    # Run threshold analysis using the generated results
    results_path = engine.output_dir / "backtest_results.csv"
    if results_path.exists():
        results_df = pd.read_csv(results_path)
        if not results_df.empty:
            analyzer = ThresholdAnalyzer()
            analyzer.run(results_df)
