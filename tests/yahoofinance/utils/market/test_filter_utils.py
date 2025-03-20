"""
Tests for the filter_utils module in yahoofinance.utils.market.filter_utils.
"""

import pandas as pd
import pytest
import numpy as np
from yahoofinance.utils.market import (
    filter_buy_opportunities,
    filter_sell_candidates,
    filter_hold_candidates,
    filter_risk_first_buy_opportunities,
    prepare_dataframe_for_filtering,
    apply_confidence_threshold,
    create_buy_filter,
    create_sell_filter,
)
from yahoofinance.core.config import TRADING_CRITERIA

# Test data
@pytest.fixture
def sample_market_data():
    """Create sample market data for testing filtering functions."""
    # Create a DataFrame with test data
    data = {
        'ticker': ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB', 'TSLA', 'NFLX', 'NVDA', 'JPM', 'BAC'],
        'company': ['Apple Inc', 'Microsoft Corp', 'Alphabet Inc', 'Amazon.com Inc', 'Meta Platforms', 
                    'Tesla Inc', 'Netflix Inc', 'NVIDIA Corp', 'JPMorgan Chase', 'Bank of America'],
        'market_cap': [2.5e12, 2.4e12, 1.8e12, 1.7e12, 1.0e12, 8.0e11, 2.5e11, 1.2e12, 5.0e11, 3.0e11],
        'price': [150.0, 300.0, 2500.0, 3500.0, 350.0, 800.0, 600.0, 250.0, 150.0, 40.0],
        'target_price': [180.0, 350.0, 3000.0, 4000.0, 400.0, 600.0, 500.0, 300.0, 175.0, 45.0],
        'upside': [20.0, 16.7, 20.0, 14.3, 14.3, -25.0, -16.7, 20.0, 16.7, 12.5],
        'analyst_count': [30, 28, 25, 35, 30, 25, 20, 30, 15, 12],
        'buy_percentage': [85, 80, 90, 75, 70, 50, 60, 85, 70, 65],
        'total_ratings': [40, 35, 30, 45, 40, 30, 25, 35, 20, 15],
        'beta': [1.2, 1.0, 1.3, 1.5, 1.8, 2.5, 1.9, 1.7, 1.4, 1.6],
        'pe_trailing': [30.0, 35.0, 25.0, 60.0, 25.0, 100.0, 40.0, 75.0, 12.0, 10.0],
        'pe_forward': [25.0, 30.0, 20.0, 50.0, 20.0, 150.0, 35.0, 60.0, 10.0, 9.0],
        'peg_ratio': [1.5, 2.0, 1.0, 2.5, 1.8, 4.0, 2.2, 2.8, 1.2, 1.0],
        'dividend_yield': [0.5, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 2.5, 2.0],
        'short_float_pct': [0.8, 0.5, 0.7, 1.0, 1.5, 5.0, 2.5, 0.9, 0.6, 0.7]
    }
    
    return pd.DataFrame(data)

def test_prepare_dataframe_for_filtering(sample_market_data):
    """Test prepare_dataframe_for_filtering function."""
    # Act
    result = prepare_dataframe_for_filtering(sample_market_data)
    
    # Assert
    assert 'peg_ratio_numeric' in result.columns
    assert 'short_float_pct_numeric' in result.columns
    assert 'pe_forward_numeric' in result.columns
    assert 'pe_trailing_numeric' in result.columns
    assert 'peg_missing' in result.columns
    assert 'si_missing' in result.columns
    assert 'EXRET' in result.columns
    
    # Calculate expected EXRET manually
    expected_values = sample_market_data['upside'] * sample_market_data['buy_percentage'] / 100
    
    # Compare the values using numpy testing
    np.testing.assert_allclose(result['EXRET'].values, expected_values.values)

def test_apply_confidence_threshold(sample_market_data):
    """Test apply_confidence_threshold function."""
    # Arrange
    min_analyst_count = TRADING_CRITERIA["COMMON"]["MIN_ANALYST_COUNT"]
    min_ratings_count = TRADING_CRITERIA["COMMON"]["MIN_RATINGS_COUNT"]
    
    # Act
    result = apply_confidence_threshold(sample_market_data)
    
    # Assert
    assert len(result) == len(sample_market_data[
        (sample_market_data['analyst_count'] >= min_analyst_count) &
        (sample_market_data['total_ratings'] >= min_ratings_count)
    ])
    
    # Check that all stocks in result meet the criteria
    assert (result['analyst_count'] >= min_analyst_count).all()
    assert (result['total_ratings'] >= min_ratings_count).all()

def test_filter_buy_opportunities(sample_market_data):
    """Test filter_buy_opportunities function."""
    # Act
    result = filter_buy_opportunities(sample_market_data)
    
    # Assert - Check that AAPL and GOOG are included (meet buy criteria)
    # and TSLA and NFLX are excluded (negative upside)
    assert 'AAPL' in result['ticker'].values
    assert 'GOOG' in result['ticker'].values
    assert 'TSLA' not in result['ticker'].values
    assert 'NFLX' not in result['ticker'].values
    
    # Check that all stocks meet buy criteria
    min_upside = TRADING_CRITERIA["BUY"]["MIN_UPSIDE"]
    min_buy_pct = TRADING_CRITERIA["BUY"]["MIN_BUY_PERCENTAGE"]
    
    assert (result['upside'] >= min_upside).all()
    assert (result['buy_percentage'] >= min_buy_pct).all()

def test_filter_sell_candidates(sample_market_data):
    """Test filter_sell_candidates function."""
    # Act
    result = filter_sell_candidates(sample_market_data)
    
    # Assert - Check that TSLA is included (negative upside)
    # and AAPL is excluded (meets buy criteria)
    assert 'TSLA' in result['ticker'].values
    assert 'AAPL' not in result['ticker'].values
    
    # Check that TSLA meets sell criteria (negative upside)
    tsla_row = result[result['ticker'] == 'TSLA']
    assert tsla_row['upside'].values[0] < 0

def test_filter_hold_candidates(sample_market_data):
    """Test filter_hold_candidates function."""
    # Act
    result = filter_hold_candidates(sample_market_data)
    
    # Check the stocks in result are neither buy nor sell
    buy_opportunities = filter_buy_opportunities(sample_market_data)
    sell_candidates = filter_sell_candidates(sample_market_data)
    
    # Assert - Hold stocks should not appear in buy or sell lists
    for ticker in result['ticker'].values:
        assert ticker not in buy_opportunities['ticker'].values
        assert ticker not in sell_candidates['ticker'].values

def test_filter_risk_first_buy_opportunities(sample_market_data):
    """Test filter_risk_first_buy_opportunities function."""
    # Act
    result = filter_risk_first_buy_opportunities(sample_market_data)
    
    # Get sell candidates
    sell_candidates = filter_sell_candidates(sample_market_data)
    
    # Assert - No stock should appear in both lists
    for ticker in result['ticker'].values:
        assert ticker not in sell_candidates['ticker'].values
    
    # Check that all stocks meet buy criteria
    min_upside = TRADING_CRITERIA["BUY"]["MIN_UPSIDE"]
    min_buy_pct = TRADING_CRITERIA["BUY"]["MIN_BUY_PERCENTAGE"]
    
    assert (result['upside'] >= min_upside).all()
    assert (result['buy_percentage'] >= min_buy_pct).all()

def test_create_buy_filter(sample_market_data):
    """Test create_buy_filter function."""
    # Arrange
    prepared_df = prepare_dataframe_for_filtering(sample_market_data)
    
    # Act
    buy_filter = create_buy_filter(prepared_df)
    
    # Assert
    # Check that AAPL and GOOG are included
    aapl_index = prepared_df[prepared_df['ticker'] == 'AAPL'].index[0]
    goog_index = prepared_df[prepared_df['ticker'] == 'GOOG'].index[0]
    tsla_index = prepared_df[prepared_df['ticker'] == 'TSLA'].index[0]
    
    assert buy_filter.iloc[aapl_index]
    assert buy_filter.iloc[goog_index]
    assert not buy_filter.iloc[tsla_index]

def test_create_sell_filter(sample_market_data):
    """Test create_sell_filter function."""
    # Arrange
    prepared_df = prepare_dataframe_for_filtering(sample_market_data)
    
    # Act
    sell_filter = create_sell_filter(prepared_df)
    
    # Assert
    # Check that TSLA is included and AAPL is excluded
    aapl_index = prepared_df[prepared_df['ticker'] == 'AAPL'].index[0]
    tsla_index = prepared_df[prepared_df['ticker'] == 'TSLA'].index[0]
    
    assert not sell_filter.iloc[aapl_index]
    assert sell_filter.iloc[tsla_index]

def test_filters_are_mutually_exclusive(sample_market_data):
    """Test that buy, sell, and hold filters are mutually exclusive."""
    # Get the prepared dataframe with confidence threshold applied
    prepared_df = apply_confidence_threshold(sample_market_data)
    
    # Create filters
    buy_filter = create_buy_filter(prepared_df)
    sell_filter = create_sell_filter(prepared_df)
    hold_filter = ~buy_filter & ~sell_filter
    
    # Assert - Each stock should be in exactly one category
    for i in range(len(prepared_df)):
        # Count how many filters are True for this stock
        true_count = int(buy_filter.iloc[i]) + int(sell_filter.iloc[i]) + int(hold_filter.iloc[i])
        # Each stock should be in exactly one category
        assert true_count == 1, f"Stock {prepared_df.iloc[i]['ticker']} is in {true_count} categories"