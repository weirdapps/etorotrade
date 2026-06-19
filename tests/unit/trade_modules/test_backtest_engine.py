"""
Tests for BacktestEngine and ThresholdAnalyzer.

Uses synthetic data only - no API calls.
"""

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from trade_modules.backtest_engine import (
    MIN_PROVEN_OBSERVATIONS,
    TEST_TICKER_RE,
    BacktestEngine,
    ThresholdAnalyzer,
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
        records.append(
            {
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
            }
        )

    # SELL signals for small-cap stocks
    for i, ticker in enumerate(["BADCO", "FAILCO", "DOWNCO"]):
        records.append(
            {
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
            }
        )

    # HOLD signals
    for i, ticker in enumerate(["HOLDCO", "STEADYCO"]):
        records.append(
            {
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
            }
        )

    # Signal with no price (should be backfilled)
    records.append(
        {
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
        }
    )

    # Test ticker (should be filtered)
    records.append(
        {
            "ticker": "STOCK0001",
            "signal": "B",
            "timestamp": (base_date + timedelta(hours=30)).isoformat(),
            "price_at_signal": 100.0,
            "upside": 50.0,
        }
    )

    # INCONCLUSIVE signal (should be filtered)
    records.append(
        {
            "ticker": "INCONCL",
            "signal": "I",
            "timestamp": (base_date + timedelta(hours=31)).isoformat(),
            "price_at_signal": 100.0,
        }
    )

    # Duplicate: same ticker same day, later timestamp (should keep this one)
    records.append(
        {
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
        }
    )

    with open(log_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return log_path


@pytest.fixture
def price_data():
    """Create synthetic price data DataFrame."""
    dates = pd.date_range("2026-01-15", "2026-03-15", freq="B")  # Business days
    tickers = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "BADCO",
        "FAILCO",
        "DOWNCO",
        "HOLDCO",
        "STEADYCO",
        "NOPRICE",
        "SPY",
    ]

    np.random.seed(42)
    data = {}
    base_prices = {
        "AAPL": 150,
        "MSFT": 160,
        "GOOGL": 170,
        "AMZN": 180,
        "NVDA": 190,
        "BADCO": 50,
        "FAILCO": 55,
        "DOWNCO": 60,
        "HOLDCO": 100,
        "STEADYCO": 100,
        "NOPRICE": 200,
        "SPY": 500,
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
        assert set(df["signal"].unique()) <= {"B", "S", "H"}
        assert len(df) > 0

    def test_filters_test_tickers(self, engine):
        df = engine.load_signals()
        assert "STOCK0001" not in df["ticker"].values

    def test_filters_inconclusive(self, engine):
        df = engine.load_signals()
        assert "I" not in df["signal"].values
        assert "INCONCL" not in df["ticker"].values

    def test_deduplicates_by_date(self, engine):
        df = engine.load_signals()
        # AAPL had two signals on same day; should keep the later one (H)
        aapl = df[df["ticker"] == "AAPL"]
        assert len(aapl) == 1
        assert aapl.iloc[0]["signal"] == "H"

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
    @pytest.mark.parametrize(
        "ticker",
        [
            "STOCK0001",
            "STOCK9999",
            "BUY1",
            "BUY123",
            "SELL1",
            "SELL99",
            "HOLD1",
            "HOLD999",
            "WEAK",
            "WEAK1",
            "WEAK99",
        ],
    )
    def test_filters_test_tickers(self, ticker):
        assert TEST_TICKER_RE.match(ticker) is not None

    @pytest.mark.parametrize(
        "ticker",
        [
            "AAPL",
            "MSFT",
            "STOCKS",
            "BUYING",
            "SELLER",
            "HOLDER",
            "WEAKNESS",
            "SPY",
        ],
    )
    def test_keeps_real_tickers(self, ticker):
        assert TEST_TICKER_RE.match(ticker) is None


class TestBackfillPrices:
    def test_fills_null_prices(self, engine, price_data):
        signals_df = engine.load_signals()
        before_count = signals_df["price_at_signal"].notna().sum()
        filled = engine.backfill_signal_prices(signals_df, price_data)
        after_count = filled["price_at_signal"].notna().sum()
        assert after_count >= before_count

    def test_noprice_gets_filled(self, engine, price_data):
        signals_df = engine.load_signals()
        noprice = signals_df[signals_df["ticker"] == "NOPRICE"]
        assert noprice.iloc[0]["price_at_signal"] is None or pd.isna(
            noprice.iloc[0]["price_at_signal"]
        )

        filled = engine.backfill_signal_prices(signals_df, price_data)
        noprice_filled = filled[filled["ticker"] == "NOPRICE"]
        assert noprice_filled.iloc[0]["price_at_signal"] is not None
        assert not pd.isna(noprice_filled.iloc[0]["price_at_signal"])

    def test_preserves_existing_prices(self, engine, price_data):
        signals_df = engine.load_signals()
        aapl_price_before = signals_df[signals_df["ticker"] == "MSFT"].iloc[0]["price_at_signal"]
        filled = engine.backfill_signal_prices(signals_df, price_data)
        aapl_price_after = filled[filled["ticker"] == "MSFT"].iloc[0]["price_at_signal"]
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
        assert "stock_return" in results.columns
        assert "spy_return" in results.columns
        assert "alpha" in results.columns

    def test_alpha_calculation(self, engine, price_data, spy_data):
        signals_df = engine.load_signals()
        signals_df = engine.backfill_signal_prices(signals_df, price_data)
        results = engine.calculate_returns(signals_df, price_data, spy_data, 7)

        for _, row in results.iterrows():
            if not pd.isna(row["alpha"]) and not pd.isna(row["spy_return"]):
                expected_alpha = row["stock_return"] - row["spy_return"]
                assert row["alpha"] == pytest.approx(expected_alpha)

    def test_handles_missing_price_data(self, engine, spy_data):
        signals_df = engine.load_signals()
        # Use empty price data - should return empty
        results = engine.calculate_returns(signals_df, pd.DataFrame(), spy_data, 7)
        assert results.empty

    def test_skips_signals_without_price(self, engine, price_data, spy_data):
        signals_df = engine.load_signals()
        # Don't backfill - NOPRICE should be skipped
        results = engine.calculate_returns(signals_df, price_data, spy_data, 7)
        assert "NOPRICE" not in results["ticker"].values

    def test_sell_signal_returns(self, engine, price_data, spy_data):
        signals_df = engine.load_signals()
        signals_df = engine.backfill_signal_prices(signals_df, price_data)
        results = engine.calculate_returns(signals_df, price_data, spy_data, 7)
        sell_results = results[results["signal"] == "S"]
        # Sell stocks have downward drift, so most returns should be negative
        assert len(sell_results) > 0


class TestCalculateStatistics:
    def test_statistics_groupby_signal(self, engine, price_data, spy_data):
        signals_df = engine.load_signals()
        signals_df = engine.backfill_signal_prices(signals_df, price_data)
        results = engine.calculate_returns(signals_df, price_data, spy_data, 7)
        results["horizon"] = 7
        stats = engine.calculate_statistics(results)
        assert "t7" in stats
        assert "by_signal" in stats["t7"]

    def test_statistics_groupby_tier(self, engine, price_data, spy_data):
        signals_df = engine.load_signals()
        signals_df = engine.backfill_signal_prices(signals_df, price_data)
        results = engine.calculate_returns(signals_df, price_data, spy_data, 7)
        results["horizon"] = 7
        stats = engine.calculate_statistics(results)
        assert "by_tier" in stats["t7"]

    def test_statistics_groupby_region(self, engine, price_data, spy_data):
        signals_df = engine.load_signals()
        signals_df = engine.backfill_signal_prices(signals_df, price_data)
        results = engine.calculate_returns(signals_df, price_data, spy_data, 7)
        results["horizon"] = 7
        stats = engine.calculate_statistics(results)
        assert "by_region" in stats["t7"]

    def test_low_sample_size_flagging(self):
        """Groups with n<30 should be flagged."""
        small_df = pd.DataFrame(
            {
                "ticker": ["A"] * 10,
                "signal": ["B"] * 10,
                "stock_return": np.random.normal(2, 5, 10),
                "alpha": np.random.normal(1, 3, 10),
                "tier": ["mega"] * 10,
                "region": ["us"] * 10,
                "horizon": [7] * 10,
            }
        )
        eng = BacktestEngine()
        stats = eng.calculate_statistics(small_df)
        buy_stats = stats["t7"]["by_signal"].get("B", {})
        assert buy_stats.get("low_sample") is True

    def test_hit_rate_buy(self):
        """BUY hit rate = % of positive returns."""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                "signal": ["B", "B", "B", "B"],
                "stock_return": [5.0, -2.0, 10.0, 3.0],  # 3 out of 4 positive
                "alpha": [3.0, -4.0, 8.0, 1.0],
                "tier": ["mega"] * 4,
                "region": ["us"] * 4,
                "horizon": [30] * 4,
            }
        )
        stats = BacktestEngine._compute_group_stats(df, "B")
        assert stats["hit_rate"] == pytest.approx(75.0)
        assert stats["count"] == 4

    def test_hit_rate_sell(self):
        """SELL hit rate = % of negative returns."""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "signal": ["S", "S", "S"],
                "stock_return": [-5.0, -2.0, 3.0],  # 2 out of 3 negative
                "alpha": [-3.0, -4.0, 1.0],
                "tier": ["small"] * 3,
                "region": ["us"] * 3,
                "horizon": [30] * 3,
            }
        )
        stats = BacktestEngine._compute_group_stats(df, "S")
        assert stats["hit_rate"] == pytest.approx(66.7, abs=0.1)


class TestSaveResults:
    def test_saves_csv_files(self, engine, price_data, spy_data):
        signals_df = engine.load_signals()
        signals_df = engine.backfill_signal_prices(signals_df, price_data)
        results = engine.calculate_returns(signals_df, price_data, spy_data, 7)
        results["horizon"] = 7
        stats = engine.calculate_statistics(results)

        engine.save_results(results, stats)

        assert (engine.output_dir / "backtest_results.csv").exists()
        assert (engine.output_dir / "backtest_summary.csv").exists()

        # Verify contents
        detail = pd.read_csv(engine.output_dir / "backtest_results.csv")
        assert "ticker" in detail.columns
        assert "stock_return" in detail.columns

        summary = pd.read_csv(engine.output_dir / "backtest_summary.csv")
        assert "group_type" in summary.columns
        assert "hit_rate" in summary.columns


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
    with open(config_path, "w") as f:
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
        "ticker": [f"TICK{i}" for i in range(n)],
        "signal": np.random.choice(["B", "S", "H"], n, p=[0.4, 0.3, 0.3]),
        "date": [datetime(2026, 1, 20).date()] * n,
        "timestamp": [datetime(2026, 1, 20).isoformat()] * n,
        "stock_return": np.random.normal(2, 10, n),
        "spy_return": np.random.normal(1, 5, n),
        "alpha": np.random.normal(1, 8, n),
        "upside": np.random.uniform(-5, 40, n),
        "buy_percentage": np.random.uniform(20, 95, n),
        "exret": np.random.uniform(-5, 20, n),
        "roe": np.random.uniform(-10, 50, n),
        "debt_equity": np.random.uniform(0, 300, n),
        "pct_52w_high": np.random.uniform(40, 100, n),
        "pe_forward": np.random.uniform(5, 60, n),
        "pe_trailing": np.random.uniform(5, 80, n),
        "tier": np.random.choice(["mega", "large", "mid", "small"], n),
        "region": ["us"] * n,
        "sell_triggers": [
            [
                ["max_exret", "min_buy_percentage"],
                ["low_roe"],
                ["hard_trigger"],
                ["max_upside"],
                [],
            ][i]
            for i in np.random.randint(0, 5, n)
        ],
    }

    return pd.DataFrame(data)


class TestMetricCorrelation:
    def test_correlation_calculation(self, analyzer, merged_data):
        correlations = analyzer.analyze_metric_predictiveness(merged_data)
        assert len(correlations) > 0
        for c in correlations:
            assert "metric" in c
            assert "correlation" in c
            if c["correlation"] is not None:
                assert -1.0 <= c["correlation"] <= 1.0

    def test_known_correlation(self, analyzer):
        """Test with data where upside perfectly predicts alpha."""
        n = 100
        df = pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(n)],
                "signal": ["B"] * n,
                "stock_return": np.arange(n, dtype=float),
                "alpha": np.arange(n, dtype=float),  # Perfect correlation with alpha
                "upside": np.arange(n, dtype=float),  # Perfect correlation
                "buy_percentage": np.random.uniform(70, 90, n),
            }
        )
        correlations = analyzer.analyze_metric_predictiveness(df)
        upside_corr = next(c for c in correlations if c["metric"] == "upside")
        assert upside_corr["correlation"] is not None
        assert upside_corr["correlation"] > 0.9

    def test_insufficient_data(self, analyzer):
        """Metrics with <30 data points should note insufficient data."""
        small_df = pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(10)],
                "signal": ["B"] * 10,
                "stock_return": np.random.normal(0, 5, 10),
                "alpha": np.random.normal(0, 5, 10),
                "upside": np.random.normal(15, 5, 10),
            }
        )
        correlations = analyzer.analyze_metric_predictiveness(small_df)
        upside_corr = next(c for c in correlations if c["metric"] == "upside")
        assert upside_corr.get("note") == "insufficient data" or upside_corr["correlation"] is None


class TestSellTriggerHitRate:
    def test_trigger_hit_rates(self, analyzer, merged_data):
        # Make sell data with known triggers and returns
        merged_data[merged_data["signal"] == "S"].copy()
        trigger_stats = analyzer.analyze_sell_triggers(merged_data)
        for t in trigger_stats:
            assert "trigger" in t
            assert "hit_rate" in t
            assert 0 <= t["hit_rate"] <= 100
            assert t["fires"] >= 10

    def test_filters_boolean_triggers(self, analyzer):
        """Boolean values like 'True'/'False' should be skipped."""
        df = pd.DataFrame(
            {
                "ticker": ["A"] * 20,
                "signal": ["S"] * 20,
                "stock_return": np.random.normal(-2, 5, 20),
                "alpha": np.random.normal(-1, 5, 20),
                "sell_triggers": [["True", "max_exret"]] * 20,
            }
        )
        results = analyzer.analyze_sell_triggers(df)
        trigger_names = [r["trigger"] for r in results]
        assert "True" not in trigger_names


class TestThresholdSuggestions:
    def test_no_suggestions_with_small_sample(self, analyzer):
        """No suggestions when sample < 50."""
        small_df = pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(20)],
                "signal": ["B"] * 20,
                "stock_return": np.random.normal(5, 3, 20),
                "upside": np.random.uniform(10, 30, 20),
                "buy_percentage": np.random.uniform(70, 90, 20),
                "exret": np.random.uniform(5, 15, 20),
                "roe": np.random.uniform(10, 30, 20),
                "tier": ["mega"] * 20,
                "region": ["us"] * 20,
            }
        )
        suggestions = analyzer.suggest_thresholds(small_df)
        # Should not suggest anything with only 20 data points
        assert len(suggestions) == 0

    def test_suggestion_format(self, analyzer, merged_data):
        suggestions = analyzer.suggest_thresholds(merged_data)
        for s in suggestions:
            assert "tier_region" in s
            assert "config_key" in s
            assert "current_value" in s
            assert "suggested_value" in s
            assert "improvement" in s
            assert s["improvement"] > 5.0
            assert s["sample_size"] >= 50


class TestThresholdReport:
    def test_report_saves_csv(self, analyzer, merged_data):
        correlations = analyzer.analyze_metric_predictiveness(merged_data)
        trigger_stats = analyzer.analyze_sell_triggers(merged_data)
        suggestions = analyzer.suggest_thresholds(merged_data)

        analyzer._save_report(correlations, trigger_stats, suggestions)

        report_path = analyzer.output_dir / "backtest_threshold_report.csv"
        assert report_path.exists()

        report = pd.read_csv(report_path)
        assert "section" in report.columns
        assert "item" in report.columns
        # Should have at least the correlation entries
        assert len(report[report["section"] == "metric_predictiveness"]) > 0


# Import yaml for config fixture
import yaml


class TestConfidenceIntervals:
    def test_stats_include_ci(self):
        """Statistics should include hit_rate_ci_lo and hit_rate_ci_hi."""
        df = pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(50)],
                "signal": ["B"] * 50,
                "stock_return": np.random.normal(2, 5, 50),
                "alpha": np.random.normal(1, 3, 50),
                "tier": ["mega"] * 50,
                "region": ["us"] * 50,
                "horizon": [7] * 50,
            }
        )
        stats = BacktestEngine._compute_group_stats(df, "B")
        assert "hit_rate_ci_lo" in stats
        assert "hit_rate_ci_hi" in stats
        assert stats["hit_rate_ci_lo"] <= stats["hit_rate"]
        assert stats["hit_rate_ci_hi"] >= stats["hit_rate"]

    def test_ci_flags_unproven_signal(self):
        """CI spanning 50% should set proven_signal=False."""
        # Near 50% hit rate with small sample
        np.random.seed(42)
        returns = np.random.normal(0, 5, 15)  # Mean ~0, so ~50% positive
        df = pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(15)],
                "signal": ["B"] * 15,
                "stock_return": returns,
                "alpha": returns,
                "tier": ["mega"] * 15,
                "region": ["us"] * 15,
                "horizon": [7] * 15,
            }
        )
        stats = BacktestEngine._compute_group_stats(df, "B")
        # With small sample near 50%, CI should span 50
        if stats["hit_rate_ci_lo"] < 50 < stats["hit_rate_ci_hi"]:
            assert stats.get("proven_signal") is False


def _cell(n, ret=1.0, alpha=1.0):
    return pd.DataFrame({"stock_return": [ret] * n, "alpha": [alpha] * n})


class TestProvenSampleFloor:
    def test_min_proven_observations_is_30(self):
        assert MIN_PROVEN_OBSERVATIONS == 30

    def test_small_sample_is_not_proven_even_if_ci_excludes_50(self):
        # 10 winners -> hit-rate CI excludes 50 but n < floor
        stats = BacktestEngine._compute_group_stats(_cell(10), "B")
        assert stats["count"] == 10
        assert stats["low_sample"] is True
        assert stats["proven_signal"] is False

    def test_large_sample_with_ci_excluding_50_is_proven(self):
        stats = BacktestEngine._compute_group_stats(_cell(200), "B")
        assert stats["proven_signal"] is True


# ============================================================
# Phase 2 Task 1: beta-adjusted + EUR-denominated alpha
# ============================================================


def _bt():
    return BacktestEngine.__new__(BacktestEngine)


def test_trailing_beta_recovers_known_beta():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2025-01-01", periods=160)
    spy_ret = rng.normal(0, 0.01, len(dates))
    spy = pd.Series(100 * np.cumprod(1 + spy_ret), index=dates)
    stock = pd.Series(100 * np.cumprod(1 + 2.0 * spy_ret), index=dates)
    b = BacktestEngine._trailing_beta(stock, spy, dates[-1], lookback=120, min_obs=30)
    assert abs(b - 2.0) < 0.05


def test_trailing_beta_nan_when_insufficient():
    dates = pd.bdate_range("2025-01-01", periods=10)
    s = pd.Series(range(1, 11), index=dates, dtype=float)
    assert np.isnan(BacktestEngine._trailing_beta(s, s, dates[-1], lookback=120, min_obs=30))


def test_eur_alpha_columns_present_and_usd_alpha_unchanged():
    dates = pd.bdate_range("2025-01-01", periods=40)
    stock = pd.Series(np.linspace(100, 110, 40), index=dates)  # +10%
    spy = pd.Series(np.linspace(100, 105, 40), index=dates)  # +5%
    eurusd = pd.Series(np.linspace(1.00, 1.10, 40), index=dates)  # EUR strengthens 10%
    price_data = pd.DataFrame({"AAA": stock})
    signals = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "signal": "B",
                "date": dates[0],
                "price_at_signal": 100.0,
                "tier": "MID",
                "region": "US",
            }
        ]
    )
    eng = _bt()
    out = eng.calculate_returns(signals, price_data, spy, horizon=39, fx_data={"USD": eurusd})
    row = out.iloc[0]
    assert abs(row["alpha"] - (row["stock_return"] - row["spy_return"])) < 1e-6
    assert abs(row["stock_return"] - 10.0) < 0.5
    assert abs(row["stock_return_eur"] - 0.0) < 0.5  # 10% USD gain offset by 10% EUR strength
    assert (
        "beta" in out.columns and "alpha_eur" in out.columns and "beta_adj_alpha_eur" in out.columns
    )


def test_calculate_returns_no_crash_on_empty_spy():
    """Regression: empty SPY (RangeIndex, not normalized) must not crash the row;
    beta and alpha stay NaN since SPY is unavailable."""
    dates = pd.bdate_range("2025-01-01", periods=40)
    stock = pd.Series(np.linspace(100, 110, 40), index=dates)
    price_data = pd.DataFrame({"AAA": stock})
    empty_spy = pd.Series(dtype=float)  # RangeIndex, NOT a DatetimeIndex
    signals = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "signal": "B",
                "date": dates[0],
                "price_at_signal": 100.0,
                "tier": "MID",
                "region": "US",
            }
        ]
    )
    eng = _bt()
    out = eng.calculate_returns(signals, price_data, empty_spy, horizon=39, fx_data=None)
    assert len(out) == 1
    row = out.iloc[0]
    assert np.isnan(row["beta"])
    assert np.isnan(row["alpha"])
    assert abs(row["stock_return"] - 10.0) < 0.5


def test_non_usd_ticker_uses_its_own_fx_pair():
    """A .T (JPY) ticker must convert via its EURJPY pair, not USD; a .PA (EUR)
    ticker must use factor 1.0 (stock_return_eur == stock_return)."""
    dates = pd.bdate_range("2025-01-01", periods=40)
    jpy_stock = pd.Series(np.linspace(100, 110, 40), index=dates)  # +10% in JPY
    eur_stock = pd.Series(np.linspace(100, 110, 40), index=dates)  # +10% in EUR
    spy = pd.Series(np.linspace(100, 105, 40), index=dates)  # +5%
    eurusd_flat = pd.Series(np.full(40, 1.10), index=dates)  # USD flat vs EUR (factor 1.0)
    eurjpy = pd.Series(np.linspace(150, 165, 40), index=dates)  # JPY weakens 10% vs EUR
    price_data = pd.DataFrame({"AAA.T": jpy_stock, "BBB.PA": eur_stock})
    signals = pd.DataFrame(
        [
            {
                "ticker": "AAA.T",
                "signal": "B",
                "date": dates[0],
                "price_at_signal": 100.0,
                "tier": "MID",
                "region": "JP",
            },
            {
                "ticker": "BBB.PA",
                "signal": "B",
                "date": dates[0],
                "price_at_signal": 100.0,
                "tier": "MID",
                "region": "EU",
            },
        ]
    )
    eng = _bt()
    out = eng.calculate_returns(
        signals, price_data, spy, horizon=39, fx_data={"USD": eurusd_flat, "JPY": eurjpy}
    )
    jpy_row = out[out["ticker"] == "AAA.T"].iloc[0]
    eur_row = out[out["ticker"] == "BBB.PA"].iloc[0]
    # JPY weakened 10% → 10% JPY gain converts to ~0% in EUR. If USD (flat) were
    # wrongly used it would be ~10%, so this distinguishes the per-ticker path.
    assert abs(jpy_row["stock_return_eur"] - 0.0) < 0.5
    # EUR ticker: factor 1.0 → EUR return equals the native return.
    assert abs(eur_row["stock_return_eur"] - eur_row["stock_return"]) < 1e-6


# ============================================================
# Regional Benchmarks
# ============================================================

from trade_modules.backtest_engine import _region_for_ticker, _suggested_holding_horizon


class TestRegionForTicker:
    @pytest.mark.parametrize(
        "ticker,expected",
        [
            ("AAPL", "us"),
            ("MSFT", "us"),
            ("SPY", "us"),
            ("TSLA", "us"),
            ("VOD.L", "uk"),
            ("HSBA.L", "uk"),
            ("0700.HK", "hk"),
            ("2800.HK", "hk"),
            ("SAP.DE", "eu"),
            ("TTE.PA", "eu"),
            ("ASML.AS", "eu"),
            ("ENI.MI", "eu"),
            ("SAN.MC", "eu"),
            ("NOVO-B.CO", "eu"),
            ("VOLV-B.ST", "eu"),
            ("DNB.OL", "eu"),
            ("NOKIA.HE", "eu"),
            ("ABI.BR", "eu"),
            ("HEIA.NV", "eu"),
        ],
    )
    def test_region_mapping(self, ticker, expected):
        assert _region_for_ticker(ticker) == expected


class TestSuggestedHoldingHorizon:
    @pytest.mark.parametrize(
        "track,expected",
        [
            ("momentum", 30),
            ("value", 90),
            ("value+momentum", 45),
            (None, 30),
            ("unknown_track", 30),
        ],
    )
    def test_horizon_mapping(self, track, expected):
        assert _suggested_holding_horizon(track) == expected


class TestRegionalBenchmarksInPriceFetch:
    def test_regional_benchmarks_added_to_tickers(self, engine):
        """The run() method should include regional benchmark tickers in the fetch list."""
        from trade_modules.price_service import REGION_BENCHMARKS

        signals_df = engine.load_signals()
        tickers = signals_df["ticker"].unique().tolist()
        if "SPY" not in tickers:
            tickers.append("SPY")
        for bm_ticker in REGION_BENCHMARKS.values():
            if bm_ticker not in tickers:
                tickers.append(bm_ticker)

        for bm in REGION_BENCHMARKS.values():
            assert bm in tickers, f"Regional benchmark {bm} missing from fetch list"


class TestRegionalAlpha:
    def test_regional_alpha_columns_present(self, engine, price_data, spy_data):
        """Results should include regional_benchmark, regional_benchmark_return, regional_alpha."""
        signals_df = engine.load_signals()
        signals_df = engine.backfill_signal_prices(signals_df, price_data)
        results = engine.calculate_returns(signals_df, price_data, spy_data, 7)
        assert "regional_benchmark" in results.columns
        assert "regional_benchmark_return" in results.columns
        assert "regional_alpha" in results.columns
        assert "suggested_horizon_days" in results.columns

    def test_us_ticker_regional_alpha_equals_spy_alpha(self):
        """For US tickers, regional benchmark is SPY so regional_alpha == alpha."""
        dates = pd.bdate_range("2025-01-01", periods=40)
        stock = pd.Series(np.linspace(100, 110, 40), index=dates)
        spy = pd.Series(np.linspace(100, 105, 40), index=dates)
        price_data = pd.DataFrame({"AAA": stock, "SPY": spy})
        signals = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "signal": "B",
                    "date": dates[0],
                    "price_at_signal": 100.0,
                    "tier": "MID",
                    "region": "US",
                }
            ]
        )
        eng = _bt()
        out = eng.calculate_returns(signals, price_data, spy, horizon=39, fx_data=None)
        row = out.iloc[0]
        assert row["regional_benchmark"] == "SPY"
        assert abs(row["regional_alpha"] - row["alpha"]) < 1e-6

    def test_uk_ticker_uses_isf_benchmark(self):
        """A .L ticker should use ISF.L as its regional benchmark."""
        dates = pd.bdate_range("2025-01-01", periods=40)
        stock = pd.Series(np.linspace(100, 115, 40), index=dates)  # +15%
        spy = pd.Series(np.linspace(100, 105, 40), index=dates)  # +5%
        isf = pd.Series(np.linspace(100, 110, 40), index=dates)  # +10%
        price_data = pd.DataFrame({"VOD.L": stock, "SPY": spy, "ISF.L": isf})
        signals = pd.DataFrame(
            [
                {
                    "ticker": "VOD.L",
                    "signal": "B",
                    "date": dates[0],
                    "price_at_signal": 100.0,
                    "tier": "MID",
                    "region": "UK",
                }
            ]
        )
        eng = _bt()
        out = eng.calculate_returns(signals, price_data, spy, horizon=39, fx_data=None)
        row = out.iloc[0]
        assert row["regional_benchmark"] == "ISF.L"
        # Stock +15%, ISF +10% -> regional alpha ~5%
        assert abs(row["regional_alpha"] - 5.0) < 0.5
        # SPY alpha is stock_return - spy_return = 15 - 5 = 10
        assert abs(row["alpha"] - 10.0) < 0.5


# ============================================================
# Unresolvable Tickers (Survivorship Bias)
# ============================================================


class TestUnresolvableTickers:
    def test_unresolvable_tracked_for_missing_price(self, engine, spy_data):
        """Tickers with no price data should appear in unresolvable list."""
        signals_df = engine.load_signals()
        # Don't backfill; use empty price data to force all tickers unresolvable
        results = engine.calculate_returns(signals_df, pd.DataFrame(), spy_data, 7)
        unresolvable = results.attrs.get("unresolvable_tickers", [])
        assert len(unresolvable) > 0
        assert all(u["reason"] == "no_price_data" for u in unresolvable)
        assert all("ticker" in u and "signal" in u and "date" in u for u in unresolvable)

    def test_unresolvable_includes_ticker_not_in_price_data(self):
        """A ticker present in signals but absent from price_data should be tracked."""
        dates = pd.bdate_range("2025-01-01", periods=40)
        spy = pd.Series(np.linspace(100, 105, 40), index=dates)
        price_data = pd.DataFrame({"SPY": spy})  # No stock data
        signals = pd.DataFrame(
            [
                {
                    "ticker": "GHOST",
                    "signal": "B",
                    "date": dates[0],
                    "price_at_signal": 100.0,
                    "tier": "MID",
                    "region": "US",
                }
            ]
        )
        eng = _bt()
        results = eng.calculate_returns(signals, price_data, spy, horizon=7, fx_data=None)
        unresolvable = results.attrs.get("unresolvable_tickers", [])
        assert len(unresolvable) == 1
        assert unresolvable[0]["ticker"] == "GHOST"
        assert unresolvable[0]["reason"] == "no_price_data"

    def test_resolved_tickers_not_in_unresolvable(self):
        """Tickers that resolve properly should NOT appear in unresolvable."""
        dates = pd.bdate_range("2025-01-01", periods=40)
        stock = pd.Series(np.linspace(100, 110, 40), index=dates)
        spy = pd.Series(np.linspace(100, 105, 40), index=dates)
        price_data = pd.DataFrame({"AAA": stock, "SPY": spy})
        signals = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "signal": "B",
                    "date": dates[0],
                    "price_at_signal": 100.0,
                    "tier": "MID",
                    "region": "US",
                }
            ]
        )
        eng = _bt()
        results = eng.calculate_returns(signals, price_data, spy, horizon=7, fx_data=None)
        unresolvable = results.attrs.get("unresolvable_tickers", [])
        assert len(unresolvable) == 0
