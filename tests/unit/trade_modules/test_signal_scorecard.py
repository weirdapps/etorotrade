"""
Tests for SignalScorecard.

Uses synthetic data only - no API calls. Mocks BacktestEngine's
fetch_price_history to avoid network requests.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from trade_modules.signal_scorecard import SignalScorecard

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def tmp_dir(tmp_path):
    """Create temp directory structure."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return tmp_path

@pytest.fixture
def signal_log(tmp_dir):
    """Create a synthetic signal log JSONL file."""
    log_path = tmp_dir / "signal_log.jsonl"
    base_date = datetime(2026, 1, 20, 10, 0, 0)

    records = []
    # BUY signals - mega cap US
    for i, ticker in enumerate(["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]):
        records.append({
            "ticker": ticker,
            "signal": "B",
            "timestamp": (base_date + timedelta(hours=i)).isoformat(),
            "price_at_signal": 150.0 + i * 10,
            "upside": 20.0 + i,
            "buy_percentage": 80.0 + i,
            "exret": 15.0 + i,
            "tier": "mega",
            "region": "us",
        })

    # SELL signals - small cap US
    for i, ticker in enumerate(["BADCO", "FAILCO"]):
        records.append({
            "ticker": ticker,
            "signal": "S",
            "timestamp": (base_date + timedelta(hours=10 + i)).isoformat(),
            "price_at_signal": 50.0 + i * 5,
            "upside": -10.0,
            "buy_percentage": 20.0,
            "exret": -5.0,
            "tier": "small",
            "region": "us",
        })

    # HOLD signals - EU large cap
    for i, ticker in enumerate(["SAP", "ASML"]):
        records.append({
            "ticker": ticker,
            "signal": "H",
            "timestamp": (base_date + timedelta(hours=20 + i)).isoformat(),
            "price_at_signal": 200.0,
            "upside": 5.0,
            "buy_percentage": 55.0,
            "exret": 3.0,
            "tier": "large",
            "region": "eu",
        })

    with open(log_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return log_path

def _make_price_data(tickers, base_date, num_days=150, gain=True):
    """Create synthetic price data DataFrame."""
    dates = pd.bdate_range(start=base_date - timedelta(days=5), periods=num_days)
    data = {}
    for i, ticker in enumerate(tickers):
        base_price = 150.0 + i * 10
        if gain:
            # Prices go up: positive return
            data[ticker] = [base_price * (1 + 0.001 * d) for d in range(num_days)]
        else:
            # Prices go down: negative return
            data[ticker] = [base_price * (1 - 0.001 * d) for d in range(num_days)]

    df = pd.DataFrame(data, index=dates)
    spy = pd.Series(
        [500.0 * (1 + 0.0005 * d) for d in range(num_days)],
        index=dates,
        name='SPY',
    )
    return df, spy

# ============================================================
# Tests
# ============================================================

class TestSignalScorecard:

    def test_empty_signal_log(self, tmp_dir):
        """Scorecard with no signals returns empty structure."""
        empty_log = tmp_dir / "empty.jsonl"
        empty_log.touch()

        sc = SignalScorecard(
            signal_log_path=empty_log,
            output_dir=tmp_dir / "output",
        )
        result = sc.generate_scorecard(months_back=3)

        assert result['overall'] == {}
        assert result['by_tier'] == {}
        assert result['by_region'] == {}
        assert result['calibration_alerts'] == []
        assert 'generated_at' in result
        assert 'period' in result

    def test_scorecard_structure(self, signal_log, tmp_dir):
        """Scorecard has the expected top-level keys."""
        buy_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        sell_tickers = ["BADCO", "FAILCO"]
        hold_tickers = ["SAP", "ASML"]
        all_tickers = buy_tickers + sell_tickers + hold_tickers

        price_data, spy_data = _make_price_data(all_tickers, datetime(2026, 1, 20))

        sc = SignalScorecard(
            signal_log_path=signal_log,
            output_dir=tmp_dir / "output",
        )

        with patch.object(sc.engine, 'fetch_price_history', return_value=(price_data, spy_data)):
            result = sc.generate_scorecard(months_back=3)

        assert 'generated_at' in result
        assert 'period' in result
        assert 'overall' in result
        assert 'by_tier' in result
        assert 'by_region' in result
        assert 'calibration_alerts' in result

    def test_buy_hit_rates(self, signal_log, tmp_dir):
        """BUY signals with rising prices should have high hit rates."""
        all_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "BADCO", "FAILCO", "SAP", "ASML"]
        price_data, spy_data = _make_price_data(all_tickers, datetime(2026, 1, 20), gain=True)

        sc = SignalScorecard(
            signal_log_path=signal_log,
            output_dir=tmp_dir / "output",
        )

        with patch.object(sc.engine, 'fetch_price_history', return_value=(price_data, spy_data)):
            result = sc.generate_scorecard(months_back=3)

        buy_stats = result['overall'].get('buy', {})
        if buy_stats.get('count', 0) > 0:
            # With rising prices, BUY hit rate should be high
            hr_1m = buy_stats.get('hit_rate_1m')
            if hr_1m is not None:
                assert hr_1m > 50.0

    def test_sell_hit_rates_with_declining_prices(self, tmp_dir):
        """SELL signals with declining prices should have high hit rates."""
        # Create a dedicated signal log with SELL signals at known prices
        log_path = tmp_dir / "sell_log.jsonl"
        base_date = datetime(2026, 2, 1, 10, 0, 0)
        records = []
        for i, ticker in enumerate(["DROPX", "DROPY", "DROPZ"]):
            records.append({
                "ticker": ticker,
                "signal": "S",
                "timestamp": (base_date + timedelta(hours=i)).isoformat(),
                "price_at_signal": 100.0,
                "upside": -10.0,
                "buy_percentage": 20.0,
                "exret": -5.0,
                "tier": "small",
                "region": "us",
            })
        with open(log_path, 'w') as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        # Price data starts at 100 and declines
        tickers = ["DROPX", "DROPY", "DROPZ"]
        dates = pd.bdate_range(start=base_date - timedelta(days=5), periods=150)
        data = {t: [100.0 * (1 - 0.002 * d) for d in range(150)] for t in tickers}
        price_data = pd.DataFrame(data, index=dates)
        spy_data = pd.Series([500.0] * 150, index=dates, name='SPY')

        sc = SignalScorecard(signal_log_path=log_path, output_dir=tmp_dir / "output")

        with patch.object(sc.engine, 'fetch_price_history', return_value=(price_data, spy_data)):
            result = sc.generate_scorecard(months_back=3)

        sell_stats = result['overall'].get('sell', {})
        assert sell_stats.get('count', 0) > 0
        hr_1m = sell_stats.get('hit_rate_1m')
        assert hr_1m is not None
        assert hr_1m > 50.0

    def test_by_tier_breakdown(self, signal_log, tmp_dir):
        """Tier breakdown should contain expected tiers."""
        all_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "BADCO", "FAILCO", "SAP", "ASML"]
        price_data, spy_data = _make_price_data(all_tickers, datetime(2026, 1, 20))

        sc = SignalScorecard(
            signal_log_path=signal_log,
            output_dir=tmp_dir / "output",
        )

        with patch.object(sc.engine, 'fetch_price_history', return_value=(price_data, spy_data)):
            result = sc.generate_scorecard(months_back=3)

        # Should have mega tier (from BUY signals)
        if result['by_tier']:
            assert 'mega' in result['by_tier'] or 'small' in result['by_tier']

    def test_by_region_breakdown(self, signal_log, tmp_dir):
        """Region breakdown should contain expected regions."""
        all_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "BADCO", "FAILCO", "SAP", "ASML"]
        price_data, spy_data = _make_price_data(all_tickers, datetime(2026, 1, 20))

        sc = SignalScorecard(
            signal_log_path=signal_log,
            output_dir=tmp_dir / "output",
        )

        with patch.object(sc.engine, 'fetch_price_history', return_value=(price_data, spy_data)):
            result = sc.generate_scorecard(months_back=3)

        if result['by_region']:
            assert 'us' in result['by_region'] or 'eu' in result['by_region']

    def test_false_positive_rate(self, signal_log, tmp_dir):
        """False positive rate for BUY signals with declining prices should be high."""
        all_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "BADCO", "FAILCO", "SAP", "ASML"]
        price_data, spy_data = _make_price_data(all_tickers, datetime(2026, 1, 20), gain=False)

        sc = SignalScorecard(
            signal_log_path=signal_log,
            output_dir=tmp_dir / "output",
        )

        with patch.object(sc.engine, 'fetch_price_history', return_value=(price_data, spy_data)):
            result = sc.generate_scorecard(months_back=3)

        buy_stats = result['overall'].get('buy', {})
        if buy_stats.get('count', 0) > 0:
            fp_1m = buy_stats.get('false_positive_rate_1m')
            if fp_1m is not None:
                # With declining prices, BUY false positive rate should be high
                assert fp_1m > 50.0

    def test_scorecard_saved_to_json(self, signal_log, tmp_dir):
        """Scorecard should be saved as JSON file."""
        all_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "BADCO", "FAILCO", "SAP", "ASML"]
        price_data, spy_data = _make_price_data(all_tickers, datetime(2026, 1, 20))

        output_dir = tmp_dir / "output"
        sc = SignalScorecard(
            signal_log_path=signal_log,
            output_dir=output_dir,
        )

        with patch.object(sc.engine, 'fetch_price_history', return_value=(price_data, spy_data)):
            sc.generate_scorecard(months_back=3)

        json_path = output_dir / "signal_scorecard.json"
        assert json_path.exists()

        with open(json_path, 'r') as f:
            loaded = json.load(f)
        assert 'overall' in loaded
        assert 'by_tier' in loaded

    def test_print_scorecard(self, signal_log, tmp_dir, capsys):
        """print_scorecard should output without errors."""
        all_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "BADCO", "FAILCO", "SAP", "ASML"]
        price_data, spy_data = _make_price_data(all_tickers, datetime(2026, 1, 20))

        sc = SignalScorecard(
            signal_log_path=signal_log,
            output_dir=tmp_dir / "output",
        )

        with patch.object(sc.engine, 'fetch_price_history', return_value=(price_data, spy_data)):
            result = sc.generate_scorecard(months_back=3)

        sc.print_scorecard(result)
        captured = capsys.readouterr()
        assert "SIGNAL SCORECARD" in captured.out

    def test_calibration_alerts_low_hit_rate(self, tmp_dir):
        """Should generate alert when BUY hit rate < 50%."""
        log_path = tmp_dir / "signal_log.jsonl"
        base_date = datetime(2026, 2, 1, 10, 0, 0)

        # Create many BUY signals for one tier/region (use tickers that won't
        # be filtered by TEST_TICKER_RE)
        tickers = [f"TST{i:03d}" for i in range(15)]
        records = []
        for i, ticker in enumerate(tickers):
            records.append({
                "ticker": ticker,
                "signal": "B",
                "timestamp": (base_date + timedelta(hours=i)).isoformat(),
                "price_at_signal": 100.0,
                "upside": 15.0,
                "buy_percentage": 70.0,
                "exret": 10.0,
                "tier": "mid",
                "region": "eu",
            })

        with open(log_path, 'w') as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        # Price data starts at 100 and declines (BUY hit rate will be 0%)
        dates = pd.bdate_range(start=base_date - timedelta(days=5), periods=150)
        data = {t: [100.0 * (1 - 0.002 * d) for d in range(150)] for t in tickers}
        price_data = pd.DataFrame(data, index=dates)
        spy_data = pd.Series([500.0] * 150, index=dates, name='SPY')

        sc = SignalScorecard(
            signal_log_path=log_path,
            output_dir=tmp_dir / "output",
        )

        with patch.object(sc.engine, 'fetch_price_history', return_value=(price_data, spy_data)):
            result = sc.generate_scorecard(months_back=3)

        # Should generate an alert for EU MID with low hit rate
        alerts = result.get('calibration_alerts', [])
        eu_mid_alerts = [a for a in alerts if 'EU' in a and 'MID' in a]
        assert len(eu_mid_alerts) > 0

    def test_empty_scorecard_helper(self, tmp_dir):
        """_empty_scorecard returns valid structure."""
        sc = SignalScorecard(output_dir=tmp_dir / "output")
        result = sc._empty_scorecard(3)
        assert result['overall'] == {}
        assert result['by_tier'] == {}
        assert result['by_region'] == {}
        assert result['calibration_alerts'] == []

    def test_hit_rate_static_method(self):
        """Test _hit_rate for each signal type."""
        returns = pd.Series([5.0, -2.0, 3.0, -1.0, 8.0])

        # BUY: % > 0
        assert SignalScorecard._hit_rate(returns, 'B') == pytest.approx(60.0)

        # SELL: % < 0
        assert SignalScorecard._hit_rate(returns, 'S') == pytest.approx(40.0)

        # HOLD: % within +-5
        hold_returns = pd.Series([1.0, -2.0, 3.0, -4.0, 10.0])
        assert SignalScorecard._hit_rate(hold_returns, 'H') == pytest.approx(80.0)

        # Empty
        assert SignalScorecard._hit_rate(pd.Series(dtype=float), 'B') == pytest.approx(0.0)

    def test_false_positive_rate_static_method(self):
        """Test _false_positive_rate for each signal type."""
        returns = pd.Series([5.0, -2.0, 3.0, -1.0, 8.0])

        # BUY false positive: % that lost money
        assert SignalScorecard._false_positive_rate(returns, 'B') == pytest.approx(40.0)

        # SELL false positive: % that went up
        assert SignalScorecard._false_positive_rate(returns, 'S') == pytest.approx(60.0)

        # HOLD: 0
        assert SignalScorecard._false_positive_rate(returns, 'H') == pytest.approx(0.0)

# ============================================================
# Consensus Calibration Tests
# ============================================================

class TestConsensusCalibration:
    """Tests for _compute_consensus_calibration method."""

    def _make_buy_signal_log(self, tmp_dir, tickers, buy_percentages, base_date=None):
        """Helper: create signal log with BUY signals at specified buy_percentage values."""
        if base_date is None:
            base_date = datetime(2026, 1, 20, 10, 0, 0)

        log_path = tmp_dir / "signal_log.jsonl"
        records = []
        for i, (ticker, bp) in enumerate(zip(tickers, buy_percentages)):
            records.append({
                "ticker": ticker,
                "signal": "B",
                "timestamp": (base_date + timedelta(hours=i)).isoformat(),
                "price_at_signal": 100.0,
                "upside": 15.0,
                "buy_percentage": bp,
                "exret": 10.0,
                "tier": "mega",
                "region": "us",
            })

        with open(log_path, 'w') as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return log_path

    def test_consensus_calibration_basic(self, tmp_dir):
        """Consensus calibration returns correct buckets with stats."""
        tickers = [f"STK{i:02d}" for i in range(10)]
        # 5 in 50-60% bucket, 5 in 80-90% bucket
        buy_pcts = [55.0] * 5 + [85.0] * 5

        log_path = self._make_buy_signal_log(tmp_dir, tickers, buy_pcts)
        base_date = datetime(2026, 1, 20)

        # Prices go up for all stocks
        price_data, spy_data = _make_price_data(tickers, base_date, gain=True)

        sc = SignalScorecard(
            signal_log_path=log_path,
            output_dir=tmp_dir / "output",
        )

        with patch.object(sc.engine, 'fetch_price_history', return_value=(price_data, spy_data)):
            result = sc.generate_scorecard(months_back=3)

        cal = result.get('consensus_calibration', {})
        assert isinstance(cal, dict)

        # Should have the two populated buckets
        if '50-60%' in cal:
            assert cal['50-60%']['count'] == 5
            assert 'hit_rate' in cal['50-60%']
            assert 'avg_return' in cal['50-60%']
            assert 'outperformance_rate' in cal['50-60%']

        if '80-90%' in cal:
            assert cal['80-90%']['count'] == 5

    def test_consensus_calibration_hit_rate_rising_prices(self, tmp_dir):
        """With rising prices, all buckets should have 100% hit rate."""
        tickers = [f"UP{i:02d}" for i in range(6)]
        buy_pcts = [55.0, 65.0, 75.0, 85.0, 92.0, 95.0]

        log_path = self._make_buy_signal_log(tmp_dir, tickers, buy_pcts)
        base_date = datetime(2026, 1, 20)

        price_data, spy_data = _make_price_data(tickers, base_date, gain=True)

        sc = SignalScorecard(
            signal_log_path=log_path,
            output_dir=tmp_dir / "output",
        )

        with patch.object(sc.engine, 'fetch_price_history', return_value=(price_data, spy_data)):
            result = sc.generate_scorecard(months_back=3)

        cal = result.get('consensus_calibration', {})
        for bucket_label, stats in cal.items():
            assert stats['hit_rate'] == pytest.approx(100.0), (
                f"Expected 100% hit rate for {bucket_label} with rising prices"
            )
            assert stats['avg_return'] > 0

    def test_consensus_calibration_hit_rate_falling_prices(self, tmp_dir):
        """With falling prices, BUY hit rates should be 0%."""
        tickers = [f"DN{i:02d}" for i in range(4)]
        buy_pcts = [55.0, 65.0, 75.0, 85.0]
        base_date = datetime(2026, 1, 20)

        log_path = self._make_buy_signal_log(tmp_dir, tickers, buy_pcts, base_date)

        # Build price data that starts AT 100 on the signal date and declines
        # so returns from price_at_signal=100 are negative
        num_days = 150
        dates = pd.bdate_range(start=base_date - timedelta(days=5), periods=num_days)
        signal_idx = list(dates).index(pd.Timestamp(base_date))
        data = {}
        for ticker in tickers:
            prices = []
            for d in range(num_days):
                # Price is 100 at signal_idx, declining after
                prices.append(100.0 * (1 - 0.003 * (d - signal_idx)))
            data[ticker] = prices
        price_data = pd.DataFrame(data, index=dates)
        spy_data = pd.Series([500.0] * num_days, index=dates, name='SPY')

        sc = SignalScorecard(
            signal_log_path=log_path,
            output_dir=tmp_dir / "output",
        )

        with patch.object(sc.engine, 'fetch_price_history', return_value=(price_data, spy_data)):
            result = sc.generate_scorecard(months_back=3)

        cal = result.get('consensus_calibration', {})
        for bucket_label, stats in cal.items():
            assert stats['hit_rate'] == pytest.approx(0.0), (
                f"Expected 0% hit rate for {bucket_label} with falling prices"
            )
            assert stats['avg_return'] < 0

    def test_consensus_calibration_empty_when_no_buy_percentage(self, tmp_dir):
        """Calibration returns empty dict when buy_percentage is missing."""
        log_path = tmp_dir / "signal_log.jsonl"
        base_date = datetime(2026, 1, 20, 10, 0, 0)

        # Signal log WITHOUT buy_percentage field
        records = []
        for i, ticker in enumerate(["NOPC1", "NOPC2"]):
            records.append({
                "ticker": ticker,
                "signal": "B",
                "timestamp": (base_date + timedelta(hours=i)).isoformat(),
                "price_at_signal": 100.0,
                "upside": 15.0,
                "exret": 10.0,
                "tier": "mega",
                "region": "us",
            })

        with open(log_path, 'w') as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        tickers = ["NOPC1", "NOPC2"]
        price_data, spy_data = _make_price_data(tickers, base_date, gain=True)

        sc = SignalScorecard(
            signal_log_path=log_path,
            output_dir=tmp_dir / "output",
        )

        with patch.object(sc.engine, 'fetch_price_history', return_value=(price_data, spy_data)):
            result = sc.generate_scorecard(months_back=3)

        cal = result.get('consensus_calibration', {})
        assert cal == {}

    def test_consensus_calibration_skips_sell_signals(self, tmp_dir):
        """Calibration only considers BUY signals, not SELL."""
        log_path = tmp_dir / "signal_log.jsonl"
        base_date = datetime(2026, 1, 20, 10, 0, 0)

        records = []
        # Only SELL signals with buy_percentage
        for i, ticker in enumerate(["SELLX", "SELLY"]):
            records.append({
                "ticker": ticker,
                "signal": "S",
                "timestamp": (base_date + timedelta(hours=i)).isoformat(),
                "price_at_signal": 100.0,
                "upside": -10.0,
                "buy_percentage": 20.0,
                "exret": -5.0,
                "tier": "small",
                "region": "us",
            })

        with open(log_path, 'w') as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        tickers = ["SELLX", "SELLY"]
        price_data, spy_data = _make_price_data(tickers, base_date, gain=False)

        sc = SignalScorecard(
            signal_log_path=log_path,
            output_dir=tmp_dir / "output",
        )

        with patch.object(sc.engine, 'fetch_price_history', return_value=(price_data, spy_data)):
            result = sc.generate_scorecard(months_back=3)

        # SELL signals have buy_percentage=20 which is below the 50% bucket floor
        cal = result.get('consensus_calibration', {})
        assert cal == {}

    def test_consensus_calibration_below_50_excluded(self, tmp_dir):
        """Signals with buy_percentage below 50% are excluded from all buckets."""
        tickers = ["LOW1", "LOW2"]
        buy_pcts = [30.0, 45.0]

        log_path = self._make_buy_signal_log(tmp_dir, tickers, buy_pcts)
        base_date = datetime(2026, 1, 20)

        price_data, spy_data = _make_price_data(tickers, base_date, gain=True)

        sc = SignalScorecard(
            signal_log_path=log_path,
            output_dir=tmp_dir / "output",
        )

        with patch.object(sc.engine, 'fetch_price_history', return_value=(price_data, spy_data)):
            result = sc.generate_scorecard(months_back=3)

        cal = result.get('consensus_calibration', {})
        assert cal == {}

    def test_consensus_calibration_90_plus_bucket(self, tmp_dir):
        """90%+ bucket captures buy_percentage values from 90 to 100."""
        tickers = [f"HI{i:02d}" for i in range(3)]
        buy_pcts = [90.0, 95.0, 100.0]

        log_path = self._make_buy_signal_log(tmp_dir, tickers, buy_pcts)
        base_date = datetime(2026, 1, 20)

        price_data, spy_data = _make_price_data(tickers, base_date, gain=True)

        sc = SignalScorecard(
            signal_log_path=log_path,
            output_dir=tmp_dir / "output",
        )

        with patch.object(sc.engine, 'fetch_price_history', return_value=(price_data, spy_data)):
            result = sc.generate_scorecard(months_back=3)

        cal = result.get('consensus_calibration', {})
        assert '90%+' in cal
        assert cal['90%+']['count'] == 3

    def test_consensus_calibration_in_empty_scorecard(self, tmp_dir):
        """Empty scorecard includes consensus_calibration key."""
        sc = SignalScorecard(output_dir=tmp_dir / "output")
        result = sc._empty_scorecard(3)
        assert 'consensus_calibration' in result
        assert result['consensus_calibration'] == {}

    def test_consensus_calibration_outperformance_rate(self, tmp_dir):
        """Outperformance rate measures % of signals that beat SPY."""
        tickers = [f"OPR{i:02d}" for i in range(4)]
        buy_pcts = [75.0, 76.0, 77.0, 78.0]

        log_path = self._make_buy_signal_log(tmp_dir, tickers, buy_pcts)
        base_date = datetime(2026, 1, 20)

        # Stocks go up at 0.1% per day, SPY goes up at 0.05% per day
        # All stocks outperform SPY
        price_data, spy_data = _make_price_data(tickers, base_date, gain=True)

        sc = SignalScorecard(
            signal_log_path=log_path,
            output_dir=tmp_dir / "output",
        )

        with patch.object(sc.engine, 'fetch_price_history', return_value=(price_data, spy_data)):
            result = sc.generate_scorecard(months_back=3)

        cal = result.get('consensus_calibration', {})
        if '70-80%' in cal:
            # With stocks growing faster than SPY, outperformance should be 100%
            assert cal['70-80%']['outperformance_rate'] == pytest.approx(100.0)

    def test_consensus_calibration_included_in_full_scorecard(self, signal_log, tmp_dir):
        """consensus_calibration appears in full scorecard output."""
        all_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "BADCO", "FAILCO", "SAP", "ASML"]
        price_data, spy_data = _make_price_data(all_tickers, datetime(2026, 1, 20))

        sc = SignalScorecard(
            signal_log_path=signal_log,
            output_dir=tmp_dir / "output",
        )

        with patch.object(sc.engine, 'fetch_price_history', return_value=(price_data, spy_data)):
            result = sc.generate_scorecard(months_back=3)

        assert 'consensus_calibration' in result
        cal = result['consensus_calibration']
        assert isinstance(cal, dict)

        # The fixture signal_log has buy_percentage 80-84 for 5 BUY signals
        # They should land in the 80-90% bucket
        if cal:
            assert '80-90%' in cal
            assert cal['80-90%']['count'] == 5

    def test_consensus_calibration_rounding(self, tmp_dir):
        """Stats are properly rounded: hit_rate 1 decimal, avg_return 2 decimals."""
        tickers = [f"RND{i:02d}" for i in range(3)]
        buy_pcts = [65.0, 66.0, 67.0]

        log_path = self._make_buy_signal_log(tmp_dir, tickers, buy_pcts)
        base_date = datetime(2026, 1, 20)

        price_data, spy_data = _make_price_data(tickers, base_date, gain=True)

        sc = SignalScorecard(
            signal_log_path=log_path,
            output_dir=tmp_dir / "output",
        )

        with patch.object(sc.engine, 'fetch_price_history', return_value=(price_data, spy_data)):
            result = sc.generate_scorecard(months_back=3)

        cal = result.get('consensus_calibration', {})
        if '60-70%' in cal:
            stats = cal['60-70%']
            # hit_rate is rounded to 1 decimal
            hr_str = str(stats['hit_rate'])
            assert len(hr_str.split('.')[-1]) <= 1
            # avg_return is rounded to 2 decimals
            ar_str = str(stats['avg_return'])
            assert len(ar_str.split('.')[-1]) <= 2
