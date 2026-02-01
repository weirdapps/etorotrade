#!/usr/bin/env python3
"""
ITERATION 4: Signal Generation Tests for SMALL and MICRO Tiers
Target: Test calculate_action_vectorized() for SMALL ($2B-$10B) and MICRO (<$2B) tiers
"""

import pytest
import pandas as pd

from trade_modules.analysis_engine import calculate_action


class TestSmallUSTierSignals:
    """Test signal generation for SMALL-US tier ($2B-$10B market cap)."""

    @pytest.fixture
    def small_us_base_data(self):
        """Base data for SMALL-US tier testing."""
        return {
            'ticker': 'SNAP',
            'market_cap': 5000000000,  # $5B = SMALL tier
            'region': 'US',
            'analyst_count': 12,
            'total_ratings': 8,
            'pe_forward': 45.0,
            'pe_trailing': 55.0,
        }

    def test_small_us_buy_signal(self, small_us_base_data):
        """BUY signal for SMALL-US tier.

        SMALL-US BUY criteria (from config.yaml):
        - min_upside: 25
        - min_buy_percentage: 85
        - min_exret: 22
        """
        data = small_us_base_data.copy()
        data['upside'] = 28.0
        data['buy_percentage'] = 88.0
        data['EXRET'] = 24.6

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['SNAP', 'BS'] == 'B', "Should generate BUY for SMALL-US"

    def test_small_us_sell_signal(self, small_us_base_data):
        """SELL signal for SMALL-US tier with enhanced scoring.

        Enhanced SELL criteria: hard trigger conditions.
        """
        data = small_us_base_data.copy()
        data['upside'] = -15.0            # Severe negative upside (hard trigger)
        data['buy_percentage'] = 45.0     # Below 55% threshold
        data['EXRET'] = -6.75             # Negative EXRET

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['SNAP', 'BS'] == 'S', "Should SELL with severe negative upside + weak sentiment"

    def test_small_us_hold_signal(self, small_us_base_data):
        """HOLD signal for SMALL-US tier."""
        data = small_us_base_data.copy()
        data['upside'] = 20.0
        data['buy_percentage'] = 82.0
        data['EXRET'] = 16.4

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['SNAP', 'BS'] == 'H', "Should HOLD for SMALL-US"


class TestMinimumMarketCapFilter:
    """Test minimum market cap filtering (config.yaml universal_thresholds.min_market_cap)."""

    def test_micro_cap_inconclusive_below_1b(self):
        """Stocks below $1B market cap should be INCONCLUSIVE (universal filter).

        Universal threshold: min_market_cap: 1000000000 ($1B)
        """
        df = pd.DataFrame([{
            'ticker': 'MICROCAP',
            'market_cap': 800000000,  # $800M < $1B minimum
            'analyst_count': 15,
            'total_ratings': 10,
            'upside': 50.0,              # Would be BUY otherwise
            'buy_percentage': 90.0,
            'EXRET': 45.0,
        }]).set_index('ticker')

        result = calculate_action(df)

        assert result.loc['MICROCAP', 'BS'] == 'I', "Should be INCONCLUSIVE below $1B market cap"

    def test_micro_cap_just_above_threshold(self):
        """Stocks just above $1B minimum should process normally."""
        df = pd.DataFrame([{
            'ticker': 'SMALLCAP',
            'market_cap': 1100000000,  # $1.1B > $1B minimum
            'analyst_count': 12,
            'total_ratings': 8,
            'upside': 28.0,
            'buy_percentage': 88.0,
            'EXRET': 24.6,
        }]).set_index('ticker')

        result = calculate_action(df)

        # Should not be filtered out, will get a real signal
        assert result.loc['SMALLCAP', 'BS'] in ['B', 'H', 'S'], "Should get valid signal above $1B"


class TestMultiRegionProcessing:
    """Test processing DataFrames with multiple regions simultaneously."""

    def test_mixed_regions_in_single_dataframe(self):
        """Process US, EU, and HK tickers in single DataFrame."""
        df = pd.DataFrame([
            # US MEGA
            {
                'ticker': 'AAPL',
                'market_cap': 3000000000000,
                'analyst_count': 20,
                'total_ratings': 15,
                'upside': 10.0,
                'buy_percentage': 70.0,
                'EXRET': 7.0,
            },
            # EU LARGE
            {
                'ticker': 'SAP.DE',
                'market_cap': 150000000000,
                'analyst_count': 18,
                'total_ratings': 12,
                'upside': 15.0,
                'buy_percentage': 75.0,
                'EXRET': 11.25,
            },
            # HK MID
            {
                'ticker': 'BIDU.HK',
                'market_cap': 55000000000,
                'analyst_count': 13,
                'total_ratings': 8,
                'upside': 28.0,
                'buy_percentage': 82.0,
                'EXRET': 23.0,
            },
        ]).set_index('ticker')

        result = calculate_action(df)

        # All should get valid signals based on region-specific thresholds
        assert 'BS' in result.columns
        assert len(result) == 3
        assert result.loc['AAPL', 'BS'] in ['B', 'H', 'S', 'I']
        assert result.loc['SAP.DE', 'BS'] in ['B', 'H', 'S', 'I']
        assert result.loc['BIDU.HK', 'BS'] in ['B', 'H', 'S', 'I']


class TestBatchProcessing:
    """Test vectorized processing of large batches."""

    def test_batch_of_50_tickers(self):
        """Process 50 tickers efficiently with vectorized operations."""
        data = []
        for i in range(50):
            data.append({
                'ticker': f'TICKER{i}',
                'market_cap': 100000000000 + (i * 10000000000),
                'analyst_count': 15 + (i % 5),
                'total_ratings': 10 + (i % 3),
                'upside': 10 + (i % 15),
                'buy_percentage': 60 + (i % 30),
                'EXRET': 5 + (i % 20),
            })

        df = pd.DataFrame(data).set_index('ticker')
        result = calculate_action(df)

        assert len(result) == 50, "Should process all 50 tickers"
        assert 'BS' in result.columns
        # All should have valid signals
        assert result['BS'].isin(['B', 'H', 'S', 'I']).all()
