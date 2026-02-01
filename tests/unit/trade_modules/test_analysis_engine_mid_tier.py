#!/usr/bin/env python3
"""
ITERATION 3: Signal Generation Tests for MID Tier (US, EU, HK)
Target: Test calculate_action_vectorized() for MID tier ($10B-$100B) across all regions
"""

import pytest
import pandas as pd

from trade_modules.analysis_engine import calculate_action


class TestMidUSTierSignals:
    """Test signal generation for MID-US tier ($10B-$100B market cap)."""

    @pytest.fixture
    def mid_us_base_data(self):
        """Base data for MID-US tier testing."""
        return {
            'ticker': 'ROKU',
            'market_cap': 50000000000,  # $50B = MID tier
            'region': 'US',
            'analyst_count': 15,
            'total_ratings': 10,
            'pe_forward': 35.0,
            'pe_trailing': 40.0,
        }

    def test_mid_us_buy_signal(self, mid_us_base_data):
        """BUY signal for MID-US tier.

        MID-US BUY criteria (from config.yaml):
        - min_upside: 15
        - min_buy_percentage: 75
        - min_exret: 12
        """
        data = mid_us_base_data.copy()
        data['upside'] = 18.0
        data['buy_percentage'] = 80.0
        data['EXRET'] = 14.4

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['ROKU', 'BS'] == 'B', "Should generate BUY for MID-US"

    def test_mid_us_sell_signal(self, mid_us_base_data):
        """SELL signal for MID-US tier with enhanced scoring.

        Enhanced SELL criteria: hard trigger conditions.
        """
        data = mid_us_base_data.copy()
        data['upside'] = -10.0            # Severe negative upside (hard trigger)
        data['buy_percentage'] = 50.0     # Below 55% threshold
        data['EXRET'] = -5.0              # Negative EXRET

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['ROKU', 'BS'] == 'S', "Should SELL with severe negative upside + weak sentiment"

    def test_mid_us_hold_signal(self, mid_us_base_data):
        """HOLD signal for MID-US tier."""
        data = mid_us_base_data.copy()
        data['upside'] = 11.0             # Between 7.5 and 15
        data['buy_percentage'] = 72.0
        data['EXRET'] = 7.9

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['ROKU', 'BS'] == 'H', "Should HOLD for MID-US"


class TestMidEUTierSignals:
    """Test signal generation for MID-EU tier."""

    @pytest.fixture
    def mid_eu_base_data(self):
        """Base data for MID-EU tier testing."""
        return {
            'ticker': 'ASML.DE',
            'market_cap': 45000000000,
            'region': 'EU',
            'analyst_count': 14,
            'total_ratings': 9,
            'pe_forward': 32.0,
            'pe_trailing': 38.0,
        }

    def test_mid_eu_buy_signal(self, mid_eu_base_data):
        """BUY signal for MID-EU tier.

        MID-EU BUY criteria:
        - min_upside: 20
        - min_buy_percentage: 75
        - min_exret: 15
        """
        data = mid_eu_base_data.copy()
        data['upside'] = 22.0
        data['buy_percentage'] = 78.0
        data['EXRET'] = 17.2

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['ASML.DE', 'BS'] == 'B', "Should generate BUY for MID-EU"

    def test_mid_eu_sell_signal(self, mid_eu_base_data):
        """SELL signal for MID-EU tier with enhanced scoring.

        Enhanced SELL criteria: hard trigger conditions.
        """
        data = mid_eu_base_data.copy()
        data['upside'] = -8.0             # Negative upside (hard trigger component)
        data['buy_percentage'] = 45.0     # Below 55% threshold
        data['EXRET'] = -3.6              # Negative EXRET

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['ASML.DE', 'BS'] == 'S', "Should SELL with negative upside + weak sentiment"

    def test_mid_eu_hold_signal(self, mid_eu_base_data):
        """HOLD signal for MID-EU tier."""
        data = mid_eu_base_data.copy()
        data['upside'] = 15.0             # Between 10 and 20
        data['buy_percentage'] = 73.0
        data['EXRET'] = 10.95

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['ASML.DE', 'BS'] == 'H', "Should HOLD for MID-EU"


class TestMidHKTierSignals:
    """Test signal generation for MID-HK tier."""

    @pytest.fixture
    def mid_hk_base_data(self):
        """Base data for MID-HK tier testing."""
        return {
            'ticker': 'BIDU.HK',
            'market_cap': 55000000000,
            'region': 'HK',
            'analyst_count': 13,
            'total_ratings': 8,
            'pe_forward': 20.0,
            'pe_trailing': 24.0,
        }

    def test_mid_hk_buy_signal(self, mid_hk_base_data):
        """BUY signal for MID-HK tier.

        MID-HK BUY criteria (tightened per hedge fund review):
        - min_upside: 30 (was 25)
        - min_buy_percentage: 92 (tightened for HK)
        - min_exret: 27.6 (30 * 92 / 100)
        """
        data = mid_hk_base_data.copy()
        data['upside'] = 35.0              # ✓ ≥30%
        data['buy_percentage'] = 94.0      # ✓ ≥92%
        data['EXRET'] = 32.9               # ✓ ≥27.6

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['BIDU.HK', 'BS'] == 'B', "Should generate BUY for MID-HK"

    def test_mid_hk_sell_signal(self, mid_hk_base_data):
        """SELL signal for MID-HK tier with enhanced scoring.

        Enhanced SELL criteria: hard trigger conditions.
        """
        data = mid_hk_base_data.copy()
        data['upside'] = -12.0            # Severe negative upside (hard trigger)
        data['buy_percentage'] = 50.0     # Below 55% threshold
        data['EXRET'] = -6.0              # Negative EXRET

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['BIDU.HK', 'BS'] == 'S', "Should SELL with severe negative upside + weak sentiment"

    def test_mid_hk_hold_signal(self, mid_hk_base_data):
        """HOLD signal for MID-HK tier."""
        data = mid_hk_base_data.copy()
        data['upside'] = 19.0             # Between 12.5 and 25
        data['buy_percentage'] = 78.0
        data['EXRET'] = 14.8

        df = pd.DataFrame([data]).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['BIDU.HK', 'BS'] == 'H', "Should HOLD for MID-HK"
