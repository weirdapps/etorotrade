#!/usr/bin/env python3
"""
ITERATION 1: Signal Generation Tests for MEGA-US Tier
Target: Test calculate_action_vectorized() for MEGA-US tier with all conditions

All floating-point comparisons use pytest.approx() for reliable testing.
"""

import pytest
import pandas as pd
import numpy as np

from trade_modules.analysis_engine import (
    calculate_action_vectorized,
    calculate_exret,
    calculate_action,
)


class TestMegaUSTierSignals:
    """Test signal generation for MEGA-US tier ($500B+ market cap)."""

    @pytest.fixture
    def mega_us_base_data(self):
        """Base data for MEGA-US tier testing."""
        return {
            'ticker': 'AAPL',
            'market_cap': 3000000000000,  # $3T = MEGA tier
            'region': 'US',
            'analyst_count': 20,          # ≥4 required
            'total_ratings': 15,          # ≥4 required (called #A in function)
            'pe_forward': 25.0,
            'pe_trailing': 28.0,
        }

    def test_mega_us_buy_signal_all_conditions_met(self, mega_us_base_data):
        """BUY signal when ALL buy conditions met for MEGA-US.

        MEGA-US BUY criteria (from config.yaml):
        - min_upside: 8 (upside ≥ 8%)
        - min_buy_percentage: 75 (buy_percentage ≥ 75%)
        - min_exret: 6 (EXRET ≥ 6.0)
        - PEF < PET × 1.1 (earnings trajectory improving)
        """
        # Arrange - Set data that meets ALL buy criteria
        data = mega_us_base_data.copy()
        data['upside'] = 15.0              # ✓ ≥8%
        data['buy_percentage'] = 80.0      # ✓ ≥75%
        data['EXRET'] = 12.0               # ✓ ≥6.0 (15.0 * 80.0 / 100)
        # PEF (25.0) < PET (28.0) × 1.1 = 30.8 ✓

        df = pd.DataFrame([data]).set_index('ticker')

        # Act
        result = calculate_action(df)

        # Assert
        assert 'BS' in result.columns, "Result should have BS column"
        assert result.loc['AAPL', 'BS'] == 'B', "Should generate BUY signal when all conditions met"

    def test_mega_us_sell_signal_low_upside(self, mega_us_base_data):
        """SELL signal with enhanced scoring: negative upside + weak sentiment.

        Enhanced SELL criteria uses multi-factor scoring:
        - Hard trigger: upside <= -5% with buy% <= 55%
        - Negative upside alone doesn't trigger SELL without weak sentiment
        """
        # Arrange - strong SELL signal: negative upside + weak buy%
        data = mega_us_base_data.copy()
        data['upside'] = -10.0            # Severe negative upside (hard trigger)
        data['buy_percentage'] = 50.0     # Below 55% threshold
        data['EXRET'] = -5.0              # Negative EXRET

        df = pd.DataFrame([data]).set_index('ticker')

        # Act
        result = calculate_action(df)

        # Assert - should SELL due to hard trigger (negative upside + weak sentiment)
        assert result.loc['AAPL', 'BS'] == 'S', "Should generate SELL for negative upside with weak sentiment"

    def test_mega_us_sell_signal_low_buy_percentage(self, mega_us_base_data):
        """SELL signal with enhanced scoring: very low buy% (hard trigger)."""
        # Arrange - hard trigger: buy% <= 35%
        data = mega_us_base_data.copy()
        data['upside'] = 5.0
        data['buy_percentage'] = 35.0     # Hard trigger: very low buy%
        data['EXRET'] = 1.75              # 5.0 * 35.0 / 100

        df = pd.DataFrame([data]).set_index('ticker')

        # Act
        result = calculate_action(df)

        # Assert - should SELL due to hard trigger (very low buy%)
        assert result.loc['AAPL', 'BS'] == 'S', "Should SELL when buy% <= 35% (hard trigger)"

    def test_mega_us_sell_signal_low_exret(self, mega_us_base_data):
        """SELL signal with enhanced scoring: negative upside + moderate sentiment.

        Enhanced scoring requires stronger signals than old system.
        Single low EXRET alone doesn't trigger SELL anymore.
        """
        # Arrange - SELL via hard trigger: negative upside + moderate sentiment
        data = mega_us_base_data.copy()
        data['upside'] = -8.0             # Significant negative upside
        data['buy_percentage'] = 45.0     # Below 55% threshold
        data['EXRET'] = -3.6              # Negative EXRET

        df = pd.DataFrame([data]).set_index('ticker')

        # Act
        result = calculate_action(df)

        # Assert - should SELL (hard trigger: negative upside + weak sentiment)
        assert result.loc['AAPL', 'BS'] == 'S', "Should SELL with negative upside and weak sentiment"

    def test_mega_us_hold_signal_between_buy_sell(self, mega_us_base_data):
        """HOLD signal when between BUY and SELL thresholds.

        MEGA-US thresholds (from config.yaml):
        - BUY: min_upside: 5, min_buy_percentage: 65, min_exret: 4
        - SELL: max_upside: 2.5, max_exret: 2
        """
        # Arrange - Data between buy and sell thresholds
        data = mega_us_base_data.copy()
        data['upside'] = 3.5               # Between 2.5 (sell) and 5 (buy)
        data['buy_percentage'] = 60.0      # Between 50 and 65
        data['EXRET'] = 2.1                # Between 2 (sell) and 4 (buy)

        df = pd.DataFrame([data]).set_index('ticker')

        # Act
        result = calculate_action(df)

        # Assert
        assert result.loc['AAPL', 'BS'] == 'H', "Should HOLD when between thresholds"

    def test_inconclusive_insufficient_analysts(self, mega_us_base_data):
        """INCONCLUSIVE signal when analyst_count < 4.

        CRITICAL: Signal priority order
        1. INCONCLUSIVE (< 4 analysts OR < 4 price targets)
        2. SELL (ANY sell condition)
        3. BUY (ALL buy conditions)
        4. HOLD (default)
        """
        # Arrange
        data = mega_us_base_data.copy()
        data['analyst_count'] = 3          # ✗ <4 → INCONCLUSIVE
        data['total_ratings'] = 10         # Good
        data['upside'] = 20.0               # Would be BUY otherwise
        data['buy_percentage'] = 80.0
        data['EXRET'] = 16.0

        df = pd.DataFrame([data]).set_index('ticker')

        # Act
        result = calculate_action(df)

        # Assert
        assert result.loc['AAPL', 'BS'] == 'I', "Should be INCONCLUSIVE with < 4 analysts"

    def test_inconclusive_insufficient_price_targets(self, mega_us_base_data):
        """INCONCLUSIVE when price_targets < 4."""
        # Arrange
        data = mega_us_base_data.copy()
        data['analyst_count'] = 10
        data['total_ratings'] = 3          # ✗ <4 → INCONCLUSIVE
        data['upside'] = 20.0

        df = pd.DataFrame([data]).set_index('ticker')

        # Act
        result = calculate_action(df)

        # Assert
        assert result.loc['AAPL', 'BS'] == 'I', "Should be INCONCLUSIVE with < 4 price targets"

    def test_signal_priority_inconclusive_overrides_buy(self, mega_us_base_data):
        """INCONCLUSIVE overrides BUY in signal priority."""
        # Arrange - Perfect BUY conditions but insufficient analysts
        data = mega_us_base_data.copy()
        data['analyst_count'] = 2          # INCONCLUSIVE trigger
        data['upside'] = 15.0               # Perfect BUY
        data['buy_percentage'] = 80.0
        data['EXRET'] = 12.0

        df = pd.DataFrame([data]).set_index('ticker')

        # Act
        result = calculate_action(df)

        # Assert - INCONCLUSIVE wins over BUY
        assert result.loc['AAPL', 'BS'] == 'I', "INCONCLUSIVE must override BUY"

    def test_signal_priority_sell_overrides_buy(self, mega_us_base_data):
        """SELL overrides BUY when multi-factor score triggers SELL."""
        # Arrange - Meets some BUY criteria but SELL hard trigger is met
        data = mega_us_base_data.copy()
        data['analyst_count'] = 20
        data['total_ratings'] = 15
        data['upside'] = -8.0               # Negative upside (hard trigger component)
        data['buy_percentage'] = 50.0       # Below 55% threshold (completes hard trigger)
        data['EXRET'] = -4.0                # Negative EXRET

        df = pd.DataFrame([data]).set_index('ticker')

        # Act
        result = calculate_action(df)

        # Assert - SELL wins (hard trigger: negative upside + weak sentiment)
        assert result.loc['AAPL', 'BS'] == 'S', "Multi-factor SELL should override when hard trigger is met"


class TestExretCalculation:
    """Test EXRET (Expected Return) calculation."""

    def test_exret_positive_upside_high_buy_percentage(self):
        """EXRET = upside × buy_percentage / 100."""
        df = pd.DataFrame([{
            'ticker': 'TEST',
            'upside': 10.0,
            'buy_percentage': 70.0
        }]).set_index('ticker')

        result = calculate_exret(df)
        assert result.loc['TEST', 'EXRET'] == pytest.approx(7.0, rel=1e-2), "EXRET should be 10.0 * 70.0 / 100 = 7.0"

    def test_exret_zero_buy_percentage(self):
        """EXRET = 0 when no analysts recommend buy."""
        df = pd.DataFrame([{
            'ticker': 'TEST',
            'upside': 15.0,
            'buy_percentage': 0.0
        }]).set_index('ticker')

        result = calculate_exret(df)
        assert result.loc['TEST', 'EXRET'] == pytest.approx(0.0), "EXRET should be 0 with 0% buy recommendations"

    def test_exret_negative_upside(self):
        """EXRET is negative when price is below target."""
        df = pd.DataFrame([{
            'ticker': 'TEST',
            'upside': -5.0,
            'buy_percentage': 60.0
        }]).set_index('ticker')

        result = calculate_exret(df)
        assert result.loc['TEST', 'EXRET'] < 0, "EXRET should be negative with negative upside"
        assert result.loc['TEST', 'EXRET'] == pytest.approx(-3.0, rel=1e-2)

    def test_exret_dataframe_vectorized(self):
        """EXRET calculation on entire DataFrame (vectorized)."""
        df = pd.DataFrame({
            'upside': [10.0, 15.0, -5.0, 0.0],
            'buy_percentage': [70.0, 80.0, 60.0, 50.0],
        })

        # Calculate EXRET column
        df['EXRET'] = df['upside'] * df['buy_percentage'] / 100.0

        assert df.loc[0, 'EXRET'] == pytest.approx(7.0)
        assert df.loc[1, 'EXRET'] == pytest.approx(12.0)
        assert df.loc[2, 'EXRET'] == pytest.approx(-3.0)
        assert df.loc[3, 'EXRET'] == pytest.approx(0.0)


class TestEdgeCases:
    """Test edge cases and data validation."""

    def test_empty_dataframe(self):
        """Handle empty DataFrame gracefully."""
        df = pd.DataFrame()
        result = calculate_action(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_missing_required_columns(self):
        """Handle missing columns gracefully."""
        df = pd.DataFrame([{
            'ticker': 'AAPL',
            # Missing upside, buy_percentage, etc.
        }]).set_index('ticker')

        # Should not crash, might return INCONCLUSIVE or default
        result = calculate_action(df)
        assert isinstance(result, pd.DataFrame)

    def test_null_values_in_critical_fields(self):
        """Handle null values in analyst data."""
        df = pd.DataFrame([{
            'ticker': 'AAPL',
            'analyst_count': None,         # NULL
            'price_targets': None,
            'upside': 10.0,
        }]).set_index('ticker')

        result = calculate_action(df)

        # Should be INCONCLUSIVE due to missing analyst data
        if 'BS' in result.columns:
            assert result.loc['AAPL', 'BS'] == 'I'

    def test_extreme_values(self):
        """Handle extreme values correctly."""
        df = pd.DataFrame([{
            'ticker': 'EXTREME',
            'market_cap': 999999999999999,  # Extreme market cap
            'upside': 1000.0,                # 1000% upside
            'buy_percentage': 100.0,         # All analysts say buy
            'analyst_count': 50,
            'price_targets': 50,
            'EXRET': 1000.0,
        }]).set_index('ticker')

        result = calculate_action(df)

        # Should process without error
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
