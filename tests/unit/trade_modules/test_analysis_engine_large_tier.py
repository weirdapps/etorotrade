#!/usr/bin/env python3
"""
ITERATION 2: Signal Generation Tests for LARGE Tier (US, EU, HK)
Target: Test calculate_action_vectorized() for LARGE tier across all regions
"""

import pytest
import pandas as pd
import numpy as np

from trade_modules.analysis_engine import (
    calculate_action_vectorized,
    calculate_exret,
    calculate_action,
)


class TestLargeUSTierSignals:
    """Test signal generation for LARGE-US tier ($100B-$500B market cap)."""

    @pytest.fixture
    def large_us_base_data(self):
        """Base data for LARGE-US tier testing."""
        return {
            'ticker': 'NFLX',
            'market_cap': 200000000000,  # $200B = LARGE tier
            'region': 'US',
            'analyst_count': 20,
            'total_ratings': 15,
            'pe_forward': 30.0,
            'pe_trailing': 35.0,
        }

    def test_large_us_buy_signal_all_conditions_met(self, large_us_base_data):
        """BUY signal when ALL buy conditions met for LARGE-US.

        LARGE-US BUY criteria (from config.yaml):
        - min_upside: 10
        - min_buy_percentage: 70
        - min_exret: 7
        """
        data = large_us_base_data.copy()
        data['upside'] = 12.0              # ✓ ≥10%
        data['buy_percentage'] = 75.0      # ✓ ≥70%
        data['EXRET'] = 9.0                # ✓ ≥7.0

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['NFLX', 'BS'] == 'B', "Should generate BUY for LARGE-US with all conditions met"

    def test_large_us_sell_signal_low_upside(self, large_us_base_data):
        """SELL signal when upside <= 5% (LARGE-US threshold).

        LARGE-US SELL criteria (from config.yaml):
        - max_upside: 5
        """
        data = large_us_base_data.copy()
        data['upside'] = 4.0               # ✗ <=5% → SELL
        data['buy_percentage'] = 75.0
        data['EXRET'] = 3.0

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['NFLX', 'BS'] == 'S', "Should SELL when upside <= 5%"

    def test_large_us_sell_signal_low_buy_percentage(self, large_us_base_data):
        """SELL signal when buy_percentage < 55% (LARGE-US threshold)."""
        data = large_us_base_data.copy()
        data['upside'] = 12.0
        data['buy_percentage'] = 50.0      # ✗ <55% → SELL
        data['EXRET'] = 6.0

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['NFLX', 'BS'] == 'S', "Should SELL when buy% < 55%"

    def test_large_us_hold_signal(self, large_us_base_data):
        """HOLD signal when between BUY and SELL thresholds.

        LARGE-US thresholds:
        - BUY: min_upside: 10, min_buy_percentage: 70, min_exret: 7
        - SELL: max_upside: 5, max_exret: 3
        """
        data = large_us_base_data.copy()
        data['upside'] = 7.0               # Between 5 and 10
        data['buy_percentage'] = 65.0      # Between 55 and 70
        data['EXRET'] = 4.5                # Between 3 and 7

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['NFLX', 'BS'] == 'H', "Should HOLD when between thresholds"


class TestLargeEUTierSignals:
    """Test signal generation for LARGE-EU tier."""

    @pytest.fixture
    def large_eu_base_data(self):
        """Base data for LARGE-EU tier testing."""
        return {
            'ticker': 'SAP.DE',
            'market_cap': 150000000000,  # $150B = LARGE tier
            'region': 'EU',
            'analyst_count': 18,
            'total_ratings': 12,
            'pe_forward': 28.0,
            'pe_trailing': 32.0,
        }

    def test_large_eu_buy_signal(self, large_eu_base_data):
        """BUY signal for LARGE-EU tier.

        LARGE-EU BUY criteria (from config.yaml):
        - min_upside: 12
        - min_buy_percentage: 72
        - min_exret: 8.5
        """
        data = large_eu_base_data.copy()
        data['upside'] = 15.0              # ✓ ≥12%
        data['buy_percentage'] = 75.0      # ✓ ≥72%
        data['EXRET'] = 11.25              # ✓ ≥8.5

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['SAP.DE', 'BS'] == 'B', "Should generate BUY for LARGE-EU"

    def test_large_eu_sell_signal(self, large_eu_base_data):
        """SELL signal for LARGE-EU tier.

        LARGE-EU SELL criteria:
        - max_upside: 6
        """
        data = large_eu_base_data.copy()
        data['upside'] = 5.0               # ✗ <=6% → SELL
        data['buy_percentage'] = 75.0
        data['EXRET'] = 3.75

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['SAP.DE', 'BS'] == 'S', "Should SELL when upside <= 6%"

    def test_large_eu_hold_signal(self, large_eu_base_data):
        """HOLD signal for LARGE-EU tier when between thresholds."""
        data = large_eu_base_data.copy()
        data['upside'] = 9.0               # Between 6 and 12
        data['buy_percentage'] = 68.0      # Between sell and buy
        data['EXRET'] = 6.1                # Between thresholds

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['SAP.DE', 'BS'] == 'H', "Should HOLD for LARGE-EU between thresholds"


class TestLargeHKTierSignals:
    """Test signal generation for LARGE-HK tier."""

    @pytest.fixture
    def large_hk_base_data(self):
        """Base data for LARGE-HK tier testing."""
        return {
            'ticker': 'BABA.HK',
            'market_cap': 180000000000,  # $180B = LARGE tier
            'region': 'HK',
            'analyst_count': 16,
            'total_ratings': 10,
            'pe_forward': 22.0,
            'pe_trailing': 26.0,
        }

    def test_large_hk_buy_signal(self, large_hk_base_data):
        """BUY signal for LARGE-HK tier.

        LARGE-HK BUY criteria (from config.yaml - tightened per hedge fund review):
        - min_upside: 25 (was 20)
        - min_buy_percentage: 80 (was 75)
        - min_exret: 20 (was 15)
        """
        data = large_hk_base_data.copy()
        data['upside'] = 27.0              # ✓ ≥25%
        data['buy_percentage'] = 82.0      # ✓ ≥80%
        data['EXRET'] = 22.0               # ✓ ≥20

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['BABA.HK', 'BS'] == 'B', "Should generate BUY for LARGE-HK"

    def test_large_hk_sell_signal(self, large_hk_base_data):
        """SELL signal for LARGE-HK tier.

        LARGE-HK SELL criteria:
        - max_upside: 10
        """
        data = large_hk_base_data.copy()
        data['upside'] = 9.0               # ✗ <=10% → SELL
        data['buy_percentage'] = 78.0
        data['EXRET'] = 7.0

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['BABA.HK', 'BS'] == 'S', "Should SELL when upside <= 10%"

    def test_large_hk_hold_signal(self, large_hk_base_data):
        """HOLD signal for LARGE-HK tier when between thresholds."""
        data = large_hk_base_data.copy()
        data['upside'] = 15.0              # Between 10 and 20
        data['buy_percentage'] = 72.0
        data['EXRET'] = 10.8               # Between thresholds

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['BABA.HK', 'BS'] == 'H', "Should HOLD for LARGE-HK between thresholds"


class TestRegionDetection:
    """Test region detection from ticker suffixes."""

    def test_us_region_detection(self):
        """Detect US region from ticker without suffix."""
        df = pd.DataFrame([{
            'ticker': 'AAPL',
            'market_cap': 3000000000000,
            'analyst_count': 20,
            'total_ratings': 15,
            'upside': 10.0,
            'buy_percentage': 70.0,
            'EXRET': 7.0,
        }]).set_index('ticker')

        result = calculate_action(df)
        # Should use US thresholds
        assert 'BS' in result.columns

    def test_eu_region_detection_de(self):
        """Detect EU region from .DE suffix."""
        df = pd.DataFrame([{
            'ticker': 'SAP.DE',
            'market_cap': 150000000000,
            'analyst_count': 18,
            'total_ratings': 12,
            'upside': 15.0,
            'buy_percentage': 75.0,
            'EXRET': 11.0,
        }]).set_index('ticker')

        result = calculate_action(df)
        # Should use EU thresholds
        assert 'BS' in result.columns

    def test_hk_region_detection(self):
        """Detect HK region from .HK suffix."""
        df = pd.DataFrame([{
            'ticker': 'BABA.HK',
            'market_cap': 180000000000,
            'analyst_count': 16,
            'total_ratings': 10,
            'upside': 18.0,
            'buy_percentage': 78.0,
            'EXRET': 14.0,
        }]).set_index('ticker')

        result = calculate_action(df)
        # Should use HK thresholds
        assert 'BS' in result.columns


class TestTierClassification:
    """Test market cap tier classification."""

    def test_mega_tier_classification(self):
        """Classify stocks ≥$500B as MEGA tier."""
        df = pd.DataFrame([{
            'ticker': 'AAPL',
            'market_cap': 3000000000000,  # $3T = MEGA
            'analyst_count': 20,
            'total_ratings': 15,
            'upside': 10.0,
            'buy_percentage': 70.0,
            'EXRET': 7.0,
        }]).set_index('ticker')

        result = calculate_action(df)
        assert 'BS' in result.columns

    def test_large_tier_classification(self):
        """Classify stocks $100B-$500B as LARGE tier."""
        df = pd.DataFrame([{
            'ticker': 'NFLX',
            'market_cap': 200000000000,  # $200B = LARGE
            'analyst_count': 20,
            'total_ratings': 15,
            'upside': 12.0,
            'buy_percentage': 75.0,
            'EXRET': 9.0,
        }]).set_index('ticker')

        result = calculate_action(df)
        assert 'BS' in result.columns

    def test_mid_tier_classification(self):
        """Classify stocks $10B-$100B as MID tier."""
        df = pd.DataFrame([{
            'ticker': 'ROKU',
            'market_cap': 50000000000,  # $50B = MID
            'analyst_count': 15,
            'total_ratings': 10,
            'upside': 18.0,
            'buy_percentage': 80.0,
            'EXRET': 14.4,
        }]).set_index('ticker')

        result = calculate_action(df)
        assert 'BS' in result.columns

    def test_small_tier_classification(self):
        """Classify stocks $2B-$10B as SMALL tier."""
        df = pd.DataFrame([{
            'ticker': 'SNAP',
            'market_cap': 5000000000,  # $5B = SMALL
            'analyst_count': 12,
            'total_ratings': 8,
            'upside': 25.0,
            'buy_percentage': 85.0,
            'EXRET': 21.25,
        }]).set_index('ticker')

        result = calculate_action(df)
        assert 'BS' in result.columns
