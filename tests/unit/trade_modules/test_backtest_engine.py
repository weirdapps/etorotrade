"""
Tests for BacktestEngine and ThresholdAnalyzer.

Uses synthetic data only - no API calls.
"""

import json
import random
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trade_modules.backtest_engine import (
    BacktestEngine,
    ThresholdAnalyzer,
    TEST_TICKER_RE,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def tmp_dir(tmp_path):
    """Create temp directory structure for test outputs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return tmp_path


@pytest.fixture
def signal_log(tmp_dir):
    """Create a synthetic signal log JSONL file."""
    log_path = tmp_dir / "signal_log.jsonl"
    base_date = datetime(2026, 1, 20, 10, 0, 0)

    records = []
    # BUY signals for mega-cap US stocks
    for i, ticker in enumerate(["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]):
        records.append({
            "ticker": ticker,
            "signal": "B",
            "timestamp": (base_date + timedelta(hours=i)).isoformat(),
            "price_at_signal": 150.0 + i * 10,
            "target_price": 200.0 + i * 10,
            "upside": 20.0 + i,
            "buy_percentage": 80.0 + i,
            "exret": 15.0 + i,
            "market_cap": 3e12,
            "tier": "mega",
            "region": "us",
            "roe": 30.0,
            "debt_equity": 50.0,
            "pct_52w_high": 90.0,
            "pe_forward": 25.0,
            "pe_trailing": 28.0,
            "sell_triggers": [],
        })

    # SELL signals for small-cap stocks
    for i, ticker in enumerate(["BADCO", "FAILCO", "DOWNCO"]):
        records.append({
            "ticker": ticker,
            "signal": "S",
            "timestamp": (base_date + timedelta(hours=10 + i)).isoformat(),
            "price_at_signal": 50.0 + i * 5,
            "target_price": 40.0 + i * 5,
            "upside": -10.0 - i,
            "buy_percentage": 30.0 - i,
            "exret": -5.0 - i,
            "market_cap": 5e9,
            "tier": "small",
            "region": "us",
            "roe": 5.0,
            "debt_equity": 200.0,
            "pct_52w_high": 60.0,
            "sell_triggers": ["max_exret", "min_buy_percentage"],
        })

    # HOLD signals
    for i, ticker in enumerate(["HOLDCO", "STEADYCO"]):
        records.append({
            "ticker": ticker,
            "signal": "H",
            "timestamp": (base_date + timedelta(hours=20 + i)).isoformat(),
            "price_at_signal": 100.0,
            "upside": 5.0,
            "buy_percentage": 55.0,
            "exret": 3.0,
            "market_cap": 50e9,
            "tier": "mid",
            "region": "us",
            "sell_triggers": [],
        })

    # Signal with no price (should be backfilled)
    records.append({
        "ticker": "NOPRICE",
        "signal": "B",
        "timestamp": (base_date + timedelta(hours=25)).isoformat(),
        "price_at_signal": None,
        "upside": 15.0,
        "buy_percentage": 75.0,
        "exret": 10.0,
        "market_cap": 1e12,
        "tier": "mega",
        "region": "us",
    })

    # Test ticker (should be filtered)
    records.append({
        "ticker": "STOCK0001",
        "signal": "B",
        "timestamp": (base_date + timedelta(hours=30)).isoformat(),
        "price_at_signal": 100.0,
        "upside": 50.0,
    })

    # INCONCLUSIVE signal (should be filtered)
    records.append({
        "ticker": "INCONCL",
        "signal": "I",
        "timestamp": (base_date + timedelta(hours=31)).isoformat(),
        "price_at_signal": 100.0,
    })

    # Duplicate: same ticker same day, later timestamp (should keep this one)
    records.append({
        "ticker": "AAPL",
        "signal": "H",  # Changed from B to H later in the day
        "timestamp": (base_date + timedelta(hours=2)).isoformat(),
        "price_at_signal": 155.0,
        "upside": 18.0,
        "buy_percentage": 78.0,
        "exret": 13.0,
        "market_cap": 3e12,
        "tier": "mega",
        "region": "us",
    })

    with open(log_path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return log_path


@pytest.fixture
def price_data():
    """Create synthetic price data DataFrame."""
    dates = pd.date_range("2026-01-15", "2026-03-15", freq='B')  # Business days
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
               "BADCO", "FAILCO", "DOWNCO", "HOLDCO", "STEADYCO",
               "NOPRICE", "SPY"]

    np.random.seed(42)
    data = {}
    base_prices = {
        "AAPL": 150, "MSFT": 160, "GOOGL": 170, "AMZN": 180, "NVDA": 190,
        "BADCO": 50, "FAILCO": 55, "DOWNCO": 60,
        "HOLDCO": 100, "STEADYCO": 100,
        "NOPRICE": 200, "SPY": 500,
    }

    for ticker in tickers:
        base = base_prices[ticker]
        # BUY stocks go up, SELL stocks go down, HOLD stays flat
        if ticker in ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "NOPRICE"):
            drift = 0.001  # Upward drift
        elif ticker in ("BADCO", "FAILCO", "DOWNCO"):
            drift = -0.002  # Downward drift
        else:
            drift = 0.0  # Flat

        prices = [base]
        for _ in range(len(dates) - 1):
            change = drift + np.random.normal(0, 0.01)
            prices.append(prices[-1] * (1 + change))
        data[ticker] = prices

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def spy_data(price_data):
    """Extract SPY series from price data."""
    return price_data["SPY"]


@pytest.fixture
def engine(signal_log, tmp_dir):
    """Create a BacktestEngine with test paths."""
    return BacktestEngine(
        signal_log_path=signal_log,
        output_dir=tmp_dir / "output",
        cache_path=tmp_dir / "output" / ".cache.parquet",
        horizons=[7, 30],
    )


# ============================================================
# BacktestEngine Tests
# ============================================================

class TestLoadSignals:
    def test_loads_valid_signals(self, engine):
        df = engine.load_signals()
        # Should have B, S, H signals but not I
        assert set(df['signal'].unique()) <= {'B', 'S', 'H'}
        assert len(df) > 0

    def test_filters_test_tickers(self, engine):
        df = engine.load_signals()
        assert 'STOCK0001' not in df['ticker'].values

    def test_filters_inconclusive(self, engine):
        df = engine.load_signals()
        assert 'I' not in df['signal'].values
        assert 'INCONCL' not in df['ticker'].values

    def test_deduplicates_by_date(self, engine):
        df = engine.load_signals()
        # AAPL had two signals on same day; should keep the later one (H)
        aapl = df[df['ticker'] == 'AAPL']
        assert len(aapl) == 1
        assert aapl.iloc[0]['signal'] == 'H'

    def test_empty_file(self, tmp_dir):
        log_path = tmp_dir / "empty.jsonl"
        log_path.touch()
        eng = BacktestEngine(signal_log_path=log_path)
        df = eng.load_signals()
        assert df.empty

    def test_missing_file(self, tmp_dir):
        eng = BacktestEngine(signal_log_path=tmp_dir / "nonexistent.jsonl")
        df = eng.load_signals()
        assert df.empty


class TestTestTickerRegex:
    @pytest.mark.parametrize("ticker", [
        "STOCK0001", "STOCK9999", "BUY1", "BUY123", "SELL1",
        "SELL99", "HOLD1", "HOLD999", "WEAK", "WEAK1", "WEAK99",
    ])
    def test_filters_test_tickers(self, ticker):
        assert TEST_TICKER_RE.match(ticker) is not None

    @pytest.mark.parametrize("ticker", [
        "AAPL", "MSFT", "STOCKS", "BUYING", "SELLER",
        "HOLDER", "WEAKNESS", "SPY",
    ])
    def test_keeps_real_tickers(self, ticker):
        assert TEST_TICKER_RE.match(ticker) is None


class TestBackfillPrices:
    def test_fills_null_prices(self, engine, price_data):
        signals_df = engine.load_signals()
        before_count = signals_df['price_at_signal'].notna().sum()
        filled = engine.backfill_signal_prices(signals_df, price_data)
        after_count = filled['price_at_signal'].notna().sum()
        assert after_count >= before_count

    def test_noprice_gets_filled(self, engine, price_data):
        signals_df = engine.load_signals()
        noprice = signals_df[signals_df['ticker'] == 'NOPRICE']
        assert noprice.iloc[0]['price_at_signal'] is None or pd.isna(noprice.iloc[0]['price_at_signal'])

        filled = engine.backfill_signal_prices(signals_df, price_data)
        noprice_filled = filled[filled['ticker'] == 'NOPRICE']
        assert noprice_filled.iloc[0]['price_at_signal'] is not None
        assert not pd.isna(noprice_filled.iloc[0]['price_at_signal'])

    def test_preserves_existing_prices(self, engine, price_data):
        signals_df = engine.load_signals()
        aapl_price_before = signals_df[signals_df['ticker'] == 'MSFT'].iloc[0]['price_at_signal']
        filled = engine.backfill_signal_prices(signals_df, price_data)
        aapl_price_after = filled[filled['ticker'] == 'MSFT'].iloc[0]['price_at_signal']
        # Existing price should be preserved (backfill only fills NaN)
        assert aapl_price_after == pytest.approx(aapl_price_before)

    def test_handles_empty_price_data(self, engine):
        signals_df = engine.load_signals()
        result = engine.backfill_signal_prices(signals_df, pd.DataFrame())
        # Should return unchanged
        assert len(result) == len(signals_df)


class TestCalculateReturns:
    def test_buy_signal_returns(self, engine, price_data, spy_data):
        signals_df = engine.load_signals()
        signals_df = engine.backfill_signal_prices(signals_df, price_data)
        results = engine.calculate_returns(signals_df, price_data, spy_data, 7)
        assert not results.empty
        assert 'stock_return' in results.columns
        assert 'spy_return' in results.columns
        assert 'alpha' in results.columns

    def test_alpha_calculation(self, engine, price_data, spy_data):
        signals_df = engine.load_signals()
        signals_df = engine.backfill_signal_prices(signals_df, price_data)
        results = engine.calculate_returns(signals_df, price_data, spy_data, 7)

        for _, row in results.iterrows():
            if not pd.isna(row['alpha']) and not pd.isna(row['spy_return']):
                expected_alpha = row['stock_return'] - row['spy_return']
                assert row['alpha'] == pytest.approx(expected_alpha)

    def test_handles_missing_price_data(self, engine, spy_data):
        signals_df = engine.load_signals()
        # Use empty price data - should return empty
        results = engine.calculate_returns(signals_df, pd.DataFrame(), spy_data, 7)
        assert results.empty

    def test_skips_signals_without_price(self, engine, price_data, spy_data):
        signals_df = engine.load_signals()
        # Don't backfill - NOPRICE should be skipped
        results = engine.calculate_returns(signals_df, price_data, spy_data, 7)
        assert 'NOPRICE' not in results['ticker'].values

    def test_sell_signal_returns(self, engine, price_data, spy_data):
        signals_df = engine.load_signals()
        signals_df = engine.backfill_signal_prices(signals_df, price_data)
        results = engine.calculate_returns(signals_df, price_data, spy_data, 7)
        sell_results = results[results['signal'] == 'S']
        # Sell stocks have downward drift, so most returns should be negative
        assert len(sell_results) > 0


class TestCalculateStatistics:
    def test_statistics_groupby_signal(self, engine, price_data, spy_data):
        signals_df = engine.load_signals()
        signals_df = engine.backfill_signal_prices(signals_df, price_data)
        results = engine.calculate_returns(signals_df, price_data, spy_data, 7)
        results['horizon'] = 7
        stats = engine.calculate_statistics(results)
        assert 't7' in stats
        assert 'by_signal' in stats['t7']

    def test_statistics_groupby_tier(self, engine, price_data, spy_data):
        signals_df = engine.load_signals()
        signals_df = engine.backfill_signal_prices(signals_df, price_data)
        results = engine.calculate_returns(signals_df, price_data, spy_data, 7)
        results['horizon'] = 7
        stats = engine.calculate_statistics(results)
        assert 'by_tier' in stats['t7']

    def test_statistics_groupby_region(self, engine, price_data, spy_data):
        signals_df = engine.load_signals()
        signals_df = engine.backfill_signal_prices(signals_df, price_data)
        results = engine.calculate_returns(signals_df, price_data, spy_data, 7)
        results['horizon'] = 7
        stats = engine.calculate_statistics(results)
        assert 'by_region' in stats['t7']

    def test_low_sample_size_flagging(self):
        """Groups with n<30 should be flagged."""
        small_df = pd.DataFrame({
            'ticker': ['A'] * 10,
            'signal': ['B'] * 10,
            'stock_return': np.random.normal(2, 5, 10),
            'alpha': np.random.normal(1, 3, 10),
            'tier': ['mega'] * 10,
            'region': ['us'] * 10,
            'horizon': [7] * 10,
        })
        eng = BacktestEngine()
        stats = eng.calculate_statistics(small_df)
        buy_stats = stats['t7']['by_signal'].get('B', {})
        assert buy_stats.get('low_sample') is True

    def test_hit_rate_buy(self):
        """BUY hit rate = % of positive returns."""
        df = pd.DataFrame({
            'ticker': ['A', 'B', 'C', 'D'],
            'signal': ['B', 'B', 'B', 'B'],
            'stock_return': [5.0, -2.0, 10.0, 3.0],  # 3 out of 4 positive
            'alpha': [3.0, -4.0, 8.0, 1.0],
            'tier': ['mega'] * 4,
            'region': ['us'] * 4,
            'horizon': [30] * 4,
        })
        stats = BacktestEngine._compute_group_stats(df, 'B')
        assert stats['hit_rate'] == pytest.approx(75.0)
        assert stats['count'] == 4

    def test_hit_rate_sell(self):
        """SELL hit rate = % of negative returns."""
        df = pd.DataFrame({
            'ticker': ['A', 'B', 'C'],
            'signal': ['S', 'S', 'S'],
            'stock_return': [-5.0, -2.0, 3.0],  # 2 out of 3 negative
            'alpha': [-3.0, -4.0, 1.0],
            'tier': ['small'] * 3,
            'region': ['us'] * 3,
            'horizon': [30] * 3,
        })
        stats = BacktestEngine._compute_group_stats(df, 'S')
        assert stats['hit_rate'] == pytest.approx(66.7, abs=0.1)


class TestSaveResults:
    def test_saves_csv_files(self, engine, price_data, spy_data):
        signals_df = engine.load_signals()
        signals_df = engine.backfill_signal_prices(signals_df, price_data)
        results = engine.calculate_returns(signals_df, price_data, spy_data, 7)
        results['horizon'] = 7
        stats = engine.calculate_statistics(results)

        engine.save_results(results, stats)

        assert (engine.output_dir / "backtest_results.csv").exists()
        assert (engine.output_dir / "backtest_summary.csv").exists()

        # Verify contents
        detail = pd.read_csv(engine.output_dir / "backtest_results.csv")
        assert 'ticker' in detail.columns
        assert 'stock_return' in detail.columns

        summary = pd.read_csv(engine.output_dir / "backtest_summary.csv")
        assert 'group_type' in summary.columns
        assert 'hit_rate' in summary.columns


# ============================================================
# ThresholdAnalyzer Tests
# ============================================================

@pytest.fixture
def analyzer(signal_log, tmp_dir):
    """Create ThresholdAnalyzer with test paths."""
    # Create minimal config.yaml
    config_path = tmp_dir / "config.yaml"
    config = {
        "us_mega": {
            "buy": {
                "min_upside": 10,
                "min_buy_percentage": 75,
                "min_exret": 6,
                "min_roe": 8.0,
            },
            "sell": {
                "max_exret": 2,
                "max_upside": 0,
            },
        },
        "us_small": {
            "buy": {
                "min_upside": 15,
                "min_buy_percentage": 80,
                "min_exret": 8,
                "min_roe": 8.0,
            },
            "sell": {
                "max_exret": 3,
                "max_upside": 5,
            },
        },
    }
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    return ThresholdAnalyzer(
        signal_log_path=signal_log,
        config_path=config_path,
        output_dir=tmp_dir / "output",
    )


@pytest.fixture
def merged_data():
    """Create synthetic merged signals+returns for threshold analysis."""
    np.random.seed(42)
    n = 200

    data = {
        'ticker': [f'TICK{i}' for i in range(n)],
        'signal': np.random.choice(['B', 'S', 'H'], n, p=[0.4, 0.3, 0.3]),
        'date': [datetime(2026, 1, 20).date()] * n,
        'timestamp': [datetime(2026, 1, 20).isoformat()] * n,
        'stock_return': np.random.normal(2, 10, n),
        'spy_return': np.random.normal(1, 5, n),
        'alpha': np.random.normal(1, 8, n),
        'upside': np.random.uniform(-5, 40, n),
        'buy_percentage': np.random.uniform(20, 95, n),
        'exret': np.random.uniform(-5, 20, n),
        'roe': np.random.uniform(-10, 50, n),
        'debt_equity': np.random.uniform(0, 300, n),
        'pct_52w_high': np.random.uniform(40, 100, n),
        'pe_forward': np.random.uniform(5, 60, n),
        'pe_trailing': np.random.uniform(5, 80, n),
        'tier': np.random.choice(['mega', 'large', 'mid', 'small'], n),
        'region': ['us'] * n,
        'sell_triggers': [
            random.choice([
                ['max_exret', 'min_buy_percentage'],
                ['low_roe'],
                ['hard_trigger'],
                ['max_upside'],
                [],
            ])
            for _ in range(n)
        ],
    }

    return pd.DataFrame(data)


class TestMetricCorrelation:
    def test_correlation_calculation(self, analyzer, merged_data):
        correlations = analyzer.analyze_metric_predictiveness(merged_data)
        assert len(correlations) > 0
        for c in correlations:
            assert 'metric' in c
            assert 'correlation' in c
            if c['correlation'] is not None:
                assert -1.0 <= c['correlation'] <= 1.0

    def test_known_correlation(self, analyzer):
        """Test with data where upside perfectly predicts return."""
        n = 100
        df = pd.DataFrame({
            'ticker': [f'T{i}' for i in range(n)],
            'signal': ['B'] * n,
            'stock_return': np.arange(n, dtype=float),
            'upside': np.arange(n, dtype=float),  # Perfect correlation
            'buy_percentage': np.random.uniform(70, 90, n),
        })
        correlations = analyzer.analyze_metric_predictiveness(df)
        upside_corr = next(c for c in correlations if c['metric'] == 'upside')
        assert upside_corr['correlation'] is not None
        assert upside_corr['correlation'] > 0.9

    def test_insufficient_data(self, analyzer):
        """Metrics with <30 data points should note insufficient data."""
        small_df = pd.DataFrame({
            'ticker': [f'T{i}' for i in range(10)],
            'signal': ['B'] * 10,
            'stock_return': np.random.normal(0, 5, 10),
            'upside': np.random.normal(15, 5, 10),
        })
        correlations = analyzer.analyze_metric_predictiveness(small_df)
        upside_corr = next(c for c in correlations if c['metric'] == 'upside')
        assert upside_corr.get('note') == 'insufficient data' or upside_corr['correlation'] is None


class TestSellTriggerHitRate:
    def test_trigger_hit_rates(self, analyzer, merged_data):
        # Make sell data with known triggers and returns
        sell_data = merged_data[merged_data['signal'] == 'S'].copy()
        trigger_stats = analyzer.analyze_sell_triggers(merged_data)
        for t in trigger_stats:
            assert 'trigger' in t
            assert 'hit_rate' in t
            assert 0 <= t['hit_rate'] <= 100
            assert t['fires'] >= 10

    def test_filters_boolean_triggers(self, analyzer):
        """Boolean values like 'True'/'False' should be skipped."""
        df = pd.DataFrame({
            'ticker': ['A'] * 20,
            'signal': ['S'] * 20,
            'stock_return': np.random.normal(-2, 5, 20),
            'sell_triggers': [['True', 'max_exret']] * 20,
        })
        results = analyzer.analyze_sell_triggers(df)
        trigger_names = [r['trigger'] for r in results]
        assert 'True' not in trigger_names


class TestThresholdSuggestions:
    def test_no_suggestions_with_small_sample(self, analyzer):
        """No suggestions when sample < 50."""
        small_df = pd.DataFrame({
            'ticker': [f'T{i}' for i in range(20)],
            'signal': ['B'] * 20,
            'stock_return': np.random.normal(5, 3, 20),
            'upside': np.random.uniform(10, 30, 20),
            'buy_percentage': np.random.uniform(70, 90, 20),
            'exret': np.random.uniform(5, 15, 20),
            'roe': np.random.uniform(10, 30, 20),
            'tier': ['mega'] * 20,
            'region': ['us'] * 20,
        })
        suggestions = analyzer.suggest_thresholds(small_df)
        # Should not suggest anything with only 20 data points
        assert len(suggestions) == 0

    def test_suggestion_format(self, analyzer, merged_data):
        suggestions = analyzer.suggest_thresholds(merged_data)
        for s in suggestions:
            assert 'tier_region' in s
            assert 'config_key' in s
            assert 'current_value' in s
            assert 'suggested_value' in s
            assert 'improvement' in s
            assert s['improvement'] > 5.0
            assert s['sample_size'] >= 50


class TestThresholdReport:
    def test_report_saves_csv(self, analyzer, merged_data):
        correlations = analyzer.analyze_metric_predictiveness(merged_data)
        trigger_stats = analyzer.analyze_sell_triggers(merged_data)
        suggestions = analyzer.suggest_thresholds(merged_data)

        analyzer._save_report(correlations, trigger_stats, suggestions)

        report_path = analyzer.output_dir / "backtest_threshold_report.csv"
        assert report_path.exists()

        report = pd.read_csv(report_path)
        assert 'section' in report.columns
        assert 'item' in report.columns
        # Should have at least the correlation entries
        assert len(report[report['section'] == 'metric_predictiveness']) > 0


# Import yaml for config fixture
import yaml
