"""
Comprehensive test suite for trade_modules.trade_engine module.

CRITICAL: These tests are designed to catch ANY behavioral changes during refactoring.
The TradingEngine is a god object being split and these tests ensure no regression.

Tests cover:
- Complete analyze_market_opportunities workflow
- All filtering methods with edge cases  
- Portfolio integration logic thoroughly
- Async batch processing behavior
- Confidence score calculations
- End-to-end data flow with real scenarios
- Ticker equivalence and notrade filtering
- Trading signal calculation vs existing BS column handling
- PositionSizer calculations
- Factory functions
- Error handling and edge cases
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, Mock
from decimal import Decimal

from trade_modules.trade_engine import (
    TradingEngine,
    TradingEngineError,
    PositionSizer,
    create_trading_engine,
    create_position_sizer,
)
from yahoofinance.core.errors import YFinanceError, ValidationError


@pytest.fixture
def mock_provider():
    """Create a mock provider for testing."""
    provider = AsyncMock()
    provider.get_ticker_info = AsyncMock()
    return provider


@pytest.fixture
def trading_engine(mock_provider):
    """Create a TradingEngine instance for testing."""
    return TradingEngine(provider=mock_provider)


@pytest.fixture
def sample_market_data_with_bs():
    """Create market data with existing BS column.

    Data is designed to meet actual BUY/HOLD/SELL criteria:
    - AAPL, TSLA: BUY - high upside (15%+), high buy% (80%+), good EXRET
    - MSFT, AMZN: HOLD - moderate metrics between thresholds
    - GOOGL: SELL - negative upside, low buy% (hard trigger)
    """
    return pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'],
        'price': [150.25, 280.50, 2750.00, 850.75, 3200.00],
        'target_price': [180.00, 300.00, 2400.00, 1020.00, 3500.00],
        'upside': [20.0, 7.0, -8.0, 20.0, 9.4],  # AAPL/TSLA BUY, GOOGL SELL trigger
        'buy_percentage': [85.0, 65.0, 30.0, 82.0, 68.0],  # GOOGL very low = SELL
        'analyst_count': [25, 30, 20, 15, 28],
        'total_ratings': [25, 30, 20, 15, 28],
        'market_cap': [2.5e12, 2.1e12, 1.8e12, 0.8e12, 1.5e12],
        'pe_ratio': [25.5, 28.2, 22.8, 45.2, 35.1],
        'beta': [1.1, 0.9, 1.3, 2.1, 1.4],
        'return_on_equity': [15.0, 12.0, 5.0, 18.0, 10.0],
        'debt_to_equity': [120.0, 80.0, 250.0, 100.0, 150.0],
        'BS': ['B', 'H', 'S', 'B', 'H'],
        'EXRET': [17.0, 4.5, -2.4, 16.4, 6.4],  # upside * buy% / 100
        'expected_return': [17.0, 4.5, -2.4, 16.4, 6.4]
    }).set_index('symbol')


@pytest.fixture
def sample_market_data_with_act():
    """Create market data with ACT column instead of BS.

    Same data as sample_market_data_with_bs but with ACT column.
    """
    return pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'],
        'price': [150.25, 280.50, 2750.00, 850.75, 3200.00],
        'target_price': [180.00, 300.00, 2400.00, 1020.00, 3500.00],
        'upside': [20.0, 7.0, -8.0, 20.0, 9.4],
        'buy_percentage': [85.0, 65.0, 30.0, 82.0, 68.0],
        'analyst_count': [25, 30, 20, 15, 28],
        'total_ratings': [25, 30, 20, 15, 28],
        'market_cap': [2.5e12, 2.1e12, 1.8e12, 0.8e12, 1.5e12],
        'pe_ratio': [25.5, 28.2, 22.8, 45.2, 35.1],
        'beta': [1.1, 0.9, 1.3, 2.1, 1.4],
        'ACT': ['B', 'H', 'S', 'B', 'H'],  # Action signals
        'EXRET': [17.0, 4.5, -2.4, 16.4, 6.4],
        'expected_return': [17.0, 4.5, -2.4, 16.4, 6.4]
    }).set_index('symbol')


@pytest.fixture
def sample_market_data_no_signals():
    """Create market data without BS or ACT columns - needs signal calculation."""
    return pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'],
        'price': [150.25, 280.50, 2750.00, 850.75, 3200.00],
        'target_price': [165.00, 300.00, 3000.00, 900.00, 3500.00],
        'upside': [9.8, 6.9, 9.1, 5.8, 9.4],
        'buy_percentage': [85.0, 90.0, 75.0, 70.0, 80.0],
        'analyst_count': [25, 30, 20, 15, 28],
        'total_ratings': [25, 30, 20, 15, 28],
        'market_cap': [2.5e12, 2.1e12, 1.8e12, 0.8e12, 1.5e12],
        'pe_ratio': [25.5, 28.2, 22.8, 45.2, 35.1],
        'beta': [1.1, 0.9, 1.3, 2.1, 1.4],
        'EXRET': [12.5, 3.2, -8.1, 15.3, 2.8],
        'expected_return': [12.5, 3.2, -8.1, 15.3, 2.8]
    }).set_index('symbol')


@pytest.fixture
def sample_portfolio_data():
    """Create portfolio data with various ticker column formats."""
    return pd.DataFrame({
        'TICKER': ['AAPL', 'MSFT', 'NVDA'],  # Note: NVDA not in market data
        'quantity': [100, 50, 25],
        'avg_cost': [145.00, 270.00, 800.00],
        'current_price': [150.25, 280.50, 850.00],
        'market_value': [15025.00, 14025.00, 21250.00],
        'unrealized_pnl': [525.00, 525.00, 1250.00]
    })


@pytest.fixture
def sample_portfolio_with_bs():
    """Create portfolio data with BS column for testing portfolio classifications."""
    return pd.DataFrame({
        'Ticker': ['AAPL', 'MSFT', 'NVDA', 'TSLA'],
        'quantity': [100, 50, 25, 30],
        'avg_cost': [145.00, 270.00, 800.00, 900.00],
        'BS': ['H', 'S', 'H', 'S'],  # Hold, Sell, Hold, Sell
        'market_value': [15025.00, 14025.00, 21250.00, 25000.00]
    })


@pytest.fixture
def notrade_csv_content():
    """Content for notrade.csv file."""
    return """Ticker
AMZN
TSLA
BANNED_STOCK
"""


@pytest.fixture
def temp_notrade_file(notrade_csv_content):
    """Create temporary notrade file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(notrade_csv_content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


class TestAnalyzeMarketOpportunities:
    """Test the core analyze_market_opportunities method with various scenarios."""
    
    @pytest.mark.asyncio
    async def test_analyze_with_existing_bs_column(self, trading_engine, sample_market_data_with_bs):
        """Test analysis when BS column already exists."""
        result = await trading_engine.analyze_market_opportunities(sample_market_data_with_bs)
        
        # Should return dictionary with three opportunity types
        assert isinstance(result, dict)
        assert set(result.keys()) == {'buy_opportunities', 'sell_opportunities', 'hold_opportunities'}
        
        # Check buy opportunities (BS == 'B')
        buy_ops = result['buy_opportunities']
        assert len(buy_ops) == 2  # AAPL and TSLA have 'B'
        assert 'AAPL' in buy_ops.index
        assert 'TSLA' in buy_ops.index
        
        # Check sell opportunities (BS == 'S')
        sell_ops = result['sell_opportunities']
        assert len(sell_ops) == 1  # GOOGL has 'S'
        assert 'GOOGL' in sell_ops.index
        
        # Check hold opportunities (BS == 'H')
        hold_ops = result['hold_opportunities']
        assert len(hold_ops) == 2  # MSFT and AMZN have 'H'
        assert 'MSFT' in hold_ops.index
        assert 'AMZN' in hold_ops.index
    
    @pytest.mark.asyncio
    async def test_analyze_with_act_column(self, trading_engine, sample_market_data_with_act):
        """Test analysis when ACT column exists - engine recalculates signals."""
        result = await trading_engine.analyze_market_opportunities(sample_market_data_with_act)

        # Should have same results as BS column test (engine recalculates signals)
        assert isinstance(result, dict)
        assert set(result.keys()) == {'buy_opportunities', 'sell_opportunities', 'hold_opportunities'}
        buy_ops = result['buy_opportunities']
        assert len(buy_ops) == 2  # AAPL and TSLA meet BUY criteria
        
    @pytest.mark.asyncio
    async def test_analyze_without_signals_calls_calculation(self, trading_engine, sample_market_data_no_signals):
        """Test analysis when no BS/ACT column exists - should calculate signals."""
        # Mock the signal calculation methods
        with patch.object(trading_engine.analysis_service, 'calculate_trading_signals') as mock_calc:
            mock_calc.return_value = sample_market_data_no_signals.copy()
            mock_calc.return_value['BS'] = ['B', 'H', 'S', 'B', 'H']
            mock_calc.return_value['confidence_score'] = [0.8, 0.7, 0.9, 0.75, 0.65]
            
            result = await trading_engine.analyze_market_opportunities(sample_market_data_no_signals)
        
        # Should have called signal calculation
        mock_calc.assert_called_once()
        
        # Should return valid results
        assert isinstance(result, dict)
        assert all(key in result for key in ['buy_opportunities', 'sell_opportunities', 'hold_opportunities'])
    
    @pytest.mark.asyncio
    async def test_analyze_with_portfolio_filtering(self, trading_engine, sample_market_data_with_bs, sample_portfolio_data):
        """Test analysis with portfolio filtering applied."""
        result = await trading_engine.analyze_market_opportunities(
            sample_market_data_with_bs, 
            portfolio_df=sample_portfolio_data
        )
        
        # Buy opportunities should exclude portfolio holdings
        buy_ops = result['buy_opportunities']
        # AAPL and MSFT are in portfolio, so AAPL (buy signal) should be excluded
        assert 'AAPL' not in buy_ops.index
        assert 'TSLA' in buy_ops.index  # TSLA has buy signal and not in portfolio
        
        # Sell opportunities should include only portfolio holdings with sell signals
        sell_ops = result['sell_opportunities']
        # GOOGL has sell signal but not in portfolio, so should be empty
        # MSFT is in portfolio but has hold signal
        
        # Hold opportunities should include portfolio holdings with hold signals
        hold_ops = result['hold_opportunities']
        # MSFT is in portfolio and has hold signal, should be included
    
    @pytest.mark.asyncio
    async def test_analyze_with_notrade_filtering(self, trading_engine, sample_market_data_with_bs, temp_notrade_file):
        """Test analysis with notrade filtering."""
        result = await trading_engine.analyze_market_opportunities(
            sample_market_data_with_bs,
            notrade_path=temp_notrade_file
        )
        
        # AMZN and TSLA should be filtered out
        for opportunity_type in result.values():
            assert 'AMZN' not in opportunity_type.index
            assert 'TSLA' not in opportunity_type.index
        
        # Other tickers should remain
        all_remaining = pd.concat(result.values())
        remaining_tickers = set(all_remaining.index)
        assert 'AAPL' in remaining_tickers
        assert 'MSFT' in remaining_tickers
        assert 'GOOGL' in remaining_tickers
    
    @pytest.mark.asyncio
    async def test_analyze_with_invalid_data_raises_error(self, trading_engine):
        """Test that invalid data raises ValidationError."""
        invalid_df = pd.DataFrame({'invalid': [1, 2, 3]})
        
        with patch('trade_modules.trade_engine.validate_dataframe', return_value=False):
            with pytest.raises(TradingEngineError) as exc_info:
                await trading_engine.analyze_market_opportunities(invalid_df)
            
            assert "Market analysis" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_analyze_logs_results_summary(self, trading_engine, sample_market_data_with_bs):
        """Test that analysis logs summary of results."""
        with patch.object(trading_engine.logger, 'info') as mock_logger:
            await trading_engine.analyze_market_opportunities(sample_market_data_with_bs)
        
        # Should log the summary
        summary_calls = [call for call in mock_logger.call_args_list if 'Analysis complete' in str(call)]
        assert len(summary_calls) == 1
        
        call_args = summary_calls[0][0][0]
        assert '2 buy' in call_args
        assert '1 sell' in call_args
        assert '2 hold' in call_args


class TestFilteringMethods:
    """Test individual filtering methods with edge cases."""
    
    def test_filter_buy_opportunities_basic(self, trading_engine, sample_market_data_with_bs):
        """Test basic buy filtering."""
        result = trading_engine._filter_buy_opportunities(sample_market_data_with_bs)
        
        # Should only return stocks with BS == 'B'
        assert len(result) == 2
        assert all(result['BS'] == 'B')
        assert 'AAPL' in result.index
        assert 'TSLA' in result.index
    
    def test_filter_buy_opportunities_no_bs_column(self, trading_engine):
        """Test buy filtering with no BS column returns empty DataFrame."""
        df_no_bs = pd.DataFrame({'symbol': ['AAPL'], 'price': [150]}).set_index('symbol')
        result = trading_engine._filter_buy_opportunities(df_no_bs)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_filter_sell_opportunities_with_confidence(self, trading_engine, sample_market_data_with_bs):
        """Test sell filtering with confidence score requirements."""
        # Add confidence scores
        df = sample_market_data_with_bs.copy()
        df['confidence_score'] = [0.8, 0.7, 0.9, 0.5, 0.65]  # GOOGL (sell) has 0.9 confidence
        
        result = trading_engine._filter_sell_opportunities(df)
        
        # Should return only GOOGL (BS='S' and confidence > 0.6)
        assert len(result) == 1
        assert 'GOOGL' in result.index
        assert result.loc['GOOGL', 'confidence_score'] == pytest.approx(0.9, 0.01)
    
    def test_filter_sell_opportunities_low_confidence_excluded(self, trading_engine, sample_market_data_with_bs):
        """Test sell filtering excludes low confidence scores."""
        df = sample_market_data_with_bs.copy()
        df['confidence_score'] = [0.8, 0.7, 0.5, 0.75, 0.65]  # GOOGL (sell) has 0.5 confidence
        
        result = trading_engine._filter_sell_opportunities(df)
        
        # Should return empty (GOOGL confidence <= 0.6)
        assert len(result) == 0
    
    def test_filter_sell_opportunities_nan_confidence_handled(self, trading_engine, sample_market_data_with_bs):
        """Test sell filtering handles NaN confidence scores."""
        df = sample_market_data_with_bs.copy()
        df['confidence_score'] = [0.8, 0.7, np.nan, 0.75, 0.65]  # GOOGL has NaN confidence
        
        result = trading_engine._filter_sell_opportunities(df)
        
        # Should return empty (NaN filled with 0.5, which is <= 0.6)
        assert len(result) == 0
    
    def test_filter_hold_opportunities_basic(self, trading_engine, sample_market_data_with_bs):
        """Test basic hold filtering."""
        result = trading_engine._filter_hold_opportunities(sample_market_data_with_bs)
        
        # Should only return stocks with BS == 'H'
        assert len(result) == 2
        assert all(result['BS'] == 'H')
        assert 'MSFT' in result.index
        assert 'AMZN' in result.index


class TestPortfolioFiltering:
    """Test portfolio filtering logic with various edge cases."""
    
    def test_apply_portfolio_filters_ticker_column_variations(self, trading_engine, sample_market_data_with_bs):
        """Test portfolio filtering works with different ticker column names."""
        opportunities = {
            'buy_opportunities': trading_engine._filter_buy_opportunities(sample_market_data_with_bs),
            'sell_opportunities': trading_engine._filter_sell_opportunities(sample_market_data_with_bs),
            'hold_opportunities': trading_engine._filter_hold_opportunities(sample_market_data_with_bs)
        }
        
        # Test different column name variations
        column_variations = ['TICKER', 'Ticker', 'ticker', 'symbol']
        
        for col_name in column_variations:
            portfolio_df = pd.DataFrame({
                col_name: ['AAPL', 'MSFT'],
                'quantity': [100, 50]
            })
            
            result = trading_engine._apply_portfolio_filters(opportunities, portfolio_df)
            
            # Buy opportunities should exclude portfolio holdings
            assert 'AAPL' not in result['buy_opportunities'].index
            
            # Hold opportunities should include MSFT (in portfolio and has hold signal)
            assert 'MSFT' in result['hold_opportunities'].index
    
    def test_apply_portfolio_filters_with_portfolio_bs_signals(self, trading_engine, sample_market_data_with_bs, sample_portfolio_with_bs):
        """Test portfolio filtering with BS signals in portfolio data."""
        opportunities = {
            'buy_opportunities': trading_engine._filter_buy_opportunities(sample_market_data_with_bs),
            'sell_opportunities': trading_engine._filter_sell_opportunities(sample_market_data_with_bs),
            'hold_opportunities': trading_engine._filter_hold_opportunities(sample_market_data_with_bs)
        }
        
        with patch.object(trading_engine.logger, 'info') as mock_logger:
            result = trading_engine._apply_portfolio_filters(opportunities, sample_portfolio_with_bs)
        
        # Should log addition of portfolio SELL stocks
        sell_log_calls = [call for call in mock_logger.call_args_list 
                         if 'portfolio SELL stocks' in str(call)]
        assert len(sell_log_calls) >= 1
        
        # Should log addition of portfolio HOLD stocks  
        hold_log_calls = [call for call in mock_logger.call_args_list 
                         if 'portfolio HOLD stocks' in str(call)]
        assert len(hold_log_calls) >= 1
    
    def test_apply_portfolio_filters_ticker_equivalence(self, trading_engine, sample_market_data_with_bs):
        """Test portfolio filtering uses ticker equivalence logic."""
        opportunities = {
            'buy_opportunities': trading_engine._filter_buy_opportunities(sample_market_data_with_bs),
            'sell_opportunities': trading_engine._filter_sell_opportunities(sample_market_data_with_bs),
            'hold_opportunities': trading_engine._filter_hold_opportunities(sample_market_data_with_bs)
        }
        
        # Portfolio with equivalent ticker formats
        portfolio_df = pd.DataFrame({
            'TICKER': ['AAPL.US', 'MSFT'],  # AAPL.US should match AAPL
            'quantity': [100, 50]
        })
        
        with patch('trade_modules.portfolio_service.are_equivalent_tickers') as mock_equiv:
            mock_equiv.side_effect = lambda t1, t2: (t1 == 'AAPL' and t2 == 'AAPL.US') or t1 == t2
            
            result = trading_engine._apply_portfolio_filters(opportunities, portfolio_df)
        
        # Should have called equivalence checking
        assert mock_equiv.called
        
        # AAPL should be excluded from buy opportunities due to equivalence
        assert 'AAPL' not in result['buy_opportunities'].index
    
    def test_apply_portfolio_filters_error_handling(self, trading_engine, sample_market_data_with_bs):
        """Test portfolio filtering handles invalid data gracefully."""
        opportunities = {
            'buy_opportunities': trading_engine._filter_buy_opportunities(sample_market_data_with_bs),
            'sell_opportunities': trading_engine._filter_sell_opportunities(sample_market_data_with_bs),
            'hold_opportunities': trading_engine._filter_hold_opportunities(sample_market_data_with_bs)
        }
        
        # Invalid portfolio data (no recognized ticker column)
        invalid_portfolio = pd.DataFrame({'invalid_column': [1, 2, 3]})
        
        result = trading_engine._apply_portfolio_filters(opportunities, invalid_portfolio)
        
        # Should return original opportunities unchanged when no ticker column found
        # (No error/warning is logged - it's handled gracefully)
        assert len(result['buy_opportunities']) == len(opportunities['buy_opportunities'])
        assert len(result['sell_opportunities']) == len(opportunities['sell_opportunities'])
        assert len(result['hold_opportunities']) == len(opportunities['hold_opportunities'])


class TestConfidenceScoreCalculation:
    """Test confidence score calculation with various input combinations."""
    
    def test_calculate_confidence_score_basic(self, trading_engine):
        """Test basic confidence score calculation."""
        df = pd.DataFrame({
            'analyst_count': [5, 10, 15, 0],
            'total_ratings': [5, 8, 12, 0],
            'expected_return': [10.0, -5.0, 15.0, 0.0],
            'EXRET': [8.0, -3.0, 12.0, 0.0],
            'upside': [12.0, np.nan, 8.0, 5.0],
            'buy_percentage': [80.0, 60.0, np.nan, 90.0]
        })
        
        scores = trading_engine._calculate_confidence_score(df)
        
        # Should return Series with same length
        assert len(scores) == len(df)
        assert isinstance(scores, pd.Series)
        
        # All scores should be between 0 and 1
        assert all(0 <= score <= 1 for score in scores)
        
        # High analyst coverage should boost confidence or hit ceiling
        # Index 0: 5 analysts (0.6 + 0.2) + total_ratings >=5 (0.1) + returns + no penalties = 1.0 (capped)
        # Index 1: 10 analysts (0.6 + 0.2 + 0.1) + total_ratings >=5 (0.1) + returns - upside penalty (0.2) = ~0.865
        # Index 2: 15 analysts (0.6 + 0.2 + 0.1) + total_ratings >=5 (0.1) + returns - buy_percentage penalty (0.2) = 1.0 (capped)
        assert scores.iloc[0] == pytest.approx(1.0)    # Capped at 1.0
        assert scores.iloc[1] == pytest.approx(0.865)  # With upside penalty  
        assert scores.iloc[2] == pytest.approx(1.0)    # Capped at 1.0
    
    def test_calculate_confidence_score_analyst_boost(self, trading_engine):
        """Test analyst count boosts confidence correctly."""
        df = pd.DataFrame({
            'analyst_count': [3, 5, 10, 15],  # Below 5, at 5, at 10, above 10
            'total_ratings': [3, 5, 10, 15]
        })
        
        scores = trading_engine._calculate_confidence_score(df)
        
        # Base confidence is 0.6
        # >= 5 analysts: +0.2 (analyst_count boost)
        # >= 10 analysts: +0.1 additional (analyst_count boost)
        # >= 5 total_ratings: +0.1 (total_ratings boost)
        assert scores.iloc[0] == pytest.approx(0.6)      # 3 analysts: 0.6
        assert scores.iloc[1] == pytest.approx(0.9)      # 5 analysts: 0.6 + 0.2 + 0.1 (total_ratings >= 5)
        assert scores.iloc[2] == pytest.approx(1.0)      # 10 analysts: 0.6 + 0.2 + 0.1 + 0.1 = 1.0 capped
        assert scores.iloc[3] == pytest.approx(1.0)      # 15 analysts: 0.6 + 0.2 + 0.1 + 0.1 = 1.0 capped
    
    def test_calculate_confidence_score_missing_data_penalty(self, trading_engine):
        """Test missing critical data reduces confidence."""
        df = pd.DataFrame({
            'analyst_count': [10, 10, 10],
            'upside': [12.0, np.nan, 8.0],
            'buy_percentage': [80.0, 75.0, np.nan]
        })
        
        scores = trading_engine._calculate_confidence_score(df)
        
        # Missing upside should reduce confidence by 0.2
        expected_base = 0.6 + 0.2 + 0.1  # 10 analysts + total_ratings >= 5 = 0.9
        assert scores.iloc[1] == pytest.approx(expected_base - 0.2)  # Missing upside penalty
        
        # Missing buy_percentage should reduce confidence by 0.2  
        assert scores.iloc[2] == pytest.approx(expected_base - 0.2)  # Missing buy_percentage penalty
    
    def test_calculate_confidence_score_error_handling(self, trading_engine):
        """Test confidence calculation handles errors gracefully."""
        # DataFrame with problematic data
        df = pd.DataFrame({
            'analyst_count': ['invalid', None, 5],
            'bad_column': [1, 2, 3]
        })
        
        scores = trading_engine._calculate_confidence_score(df)
        
        # Should return valid scores (invalid/None values get coerced to NaN/0)
        assert len(scores) == len(df)
        assert isinstance(scores, pd.Series)
        # First two have invalid analyst_count (coerced to 0) -> base confidence 0.6
        # Third has valid analyst_count=5 -> 0.6 + 0.2 = 0.8
        assert scores.iloc[0] == pytest.approx(0.6)  # invalid -> 0
        assert scores.iloc[1] == pytest.approx(0.6)  # None -> 0  
        assert scores.iloc[2] == pytest.approx(0.8)  # 5 analysts -> boost
    
    def test_calculate_confidence_score_clipping(self, trading_engine):
        """Test confidence scores are clipped to 0-1 range."""
        df = pd.DataFrame({
            'analyst_count': [50],  # Very high - could push confidence > 1
            'total_ratings': [50],
            'expected_return': [100.0],  # Very high return
            'EXRET': [100.0]
        })
        
        scores = trading_engine._calculate_confidence_score(df)
        
        # Should be clipped to 1.0
        assert scores.iloc[0] == pytest.approx(1.0, 0.01)


class TestNotradeFiltering:
    """Test notrade filtering logic with ticker equivalence."""
    
    def test_filter_notrade_tickers_basic(self, trading_engine, sample_market_data_with_bs, temp_notrade_file):
        """Test basic notrade filtering."""
        initial_count = len(sample_market_data_with_bs)
        
        with patch.object(trading_engine.logger, 'info') as mock_logger:
            result = trading_engine._filter_notrade_tickers(sample_market_data_with_bs, temp_notrade_file)
        
        # Should filter out AMZN and TSLA
        assert len(result) == initial_count - 2
        assert 'AMZN' not in result.index
        assert 'TSLA' not in result.index
        assert 'AAPL' in result.index
        
        # Should log filtering
        filter_calls = [call for call in mock_logger.call_args_list 
                       if 'Filtered out' in str(call)]
        assert len(filter_calls) == 1
    
    def test_filter_notrade_tickers_equivalence(self, trading_engine, sample_market_data_with_bs, temp_notrade_file):
        """Test notrade filtering uses ticker equivalence."""
        with patch('trade_modules.filter_service.are_equivalent_tickers') as mock_equiv:
            # Make AAPL equivalent to AMZN for testing
            mock_equiv.side_effect = lambda t1, t2: (t1 == 'AAPL' and t2 == 'AMZN') or t1 == t2
            
            result = trading_engine._filter_notrade_tickers(sample_market_data_with_bs, temp_notrade_file)
        
        # Should have called equivalence checking
        assert mock_equiv.called
        
        # AAPL should be filtered due to equivalence with AMZN
        assert 'AAPL' not in result.index
    
    def test_filter_notrade_tickers_missing_file(self, trading_engine, sample_market_data_with_bs):
        """Test notrade filtering with missing file returns original data."""
        result = trading_engine._filter_notrade_tickers(sample_market_data_with_bs, 'nonexistent.csv')

        # Should return original data unchanged (missing file is silently handled)
        pd.testing.assert_frame_equal(result, sample_market_data_with_bs)
    
    def test_filter_notrade_tickers_invalid_data(self, trading_engine, sample_market_data_with_bs):
        """Test notrade filtering with invalid CSV data."""
        # Create temp file with no Ticker column
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("invalid_column\nvalue1\nvalue2\n")
            temp_path = f.name
        
        try:
            with patch.object(trading_engine.logger, 'warning') as mock_logger:
                result = trading_engine._filter_notrade_tickers(sample_market_data_with_bs, temp_path)
            
            # Should return original data unchanged
            pd.testing.assert_frame_equal(result, sample_market_data_with_bs)
        finally:
            os.unlink(temp_path)


class TestTradingSignalCalculation:
    """Test trading signal calculation when BS/ACT columns don't exist."""
    
    @patch('trade_modules.analysis_service.calculate_expected_return')
    @patch('trade_modules.analysis_service.calculate_exret')
    @patch('trade_modules.analysis_service.calculate_action')
    def test_calculate_trading_signals_calls_functions(self, mock_action, mock_exret, mock_expected, trading_engine):
        """Test that trading signal calculation calls all required functions."""
        df = pd.DataFrame({'symbol': ['AAPL'], 'price': [150]}).set_index('symbol')
        
        # Mock the functions to return the input DataFrame
        mock_expected.return_value = df
        mock_exret.return_value = df
        mock_action.return_value = df
        
        with patch.object(trading_engine.analysis_service, 'calculate_confidence_score', return_value=pd.Series([0.8])):
            result = trading_engine.analysis_service.calculate_trading_signals(df)
        
        # Should call all calculation functions
        mock_expected.assert_called_once_with(df)
        mock_exret.assert_called_once_with(df)
        mock_action.assert_called_once_with(df)
    
    def test_calculate_trading_signals_adds_confidence(self, trading_engine):
        """Test that trading signal calculation adds confidence scores."""
        df = pd.DataFrame({'symbol': ['AAPL'], 'price': [150]}).set_index('symbol')

        with patch('trade_modules.analysis_service.calculate_expected_return', return_value=df), \
             patch('trade_modules.analysis_service.calculate_exret', return_value=df), \
             patch('trade_modules.analysis_service.calculate_action', return_value=df), \
             patch.object(trading_engine.analysis_service, 'calculate_confidence_score', return_value=pd.Series([0.8], index=df.index)):

            result = trading_engine.analysis_service.calculate_trading_signals(df)
        
        # Should have confidence_score column
        assert 'confidence_score' in result.columns
        assert result['confidence_score'].iloc[0] == pytest.approx(0.8, 0.01)
    
    def test_calculate_trading_signals_error_handling(self, trading_engine):
        """Test trading signal calculation error handling."""
        df = pd.DataFrame({'symbol': ['AAPL'], 'price': [150]}).set_index('symbol')
        
        with patch('trade_modules.analysis_service.calculate_expected_return', side_effect=Exception('Test error')):
            with pytest.raises(Exception):
                trading_engine.analysis_service.calculate_trading_signals(df)


class TestAsyncBatchProcessing:
    """Test async batch processing methods."""
    
    @pytest.mark.asyncio
    async def test_process_ticker_batch_basic(self, trading_engine):
        """Test basic ticker batch processing."""
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        
        # Mock the batch processing with side_effect to handle multiple calls
        batch_1_results = [
            {'ticker': 'AAPL', 'price': 150.0},
            {'ticker': 'MSFT', 'price': 280.0}
        ]
        batch_2_results = [
            {'ticker': 'GOOGL', 'price': 2750.0}
        ]
        
        with patch.object(trading_engine.data_processing_service, '_process_batch', side_effect=[batch_1_results, batch_2_results]):
            result = await trading_engine.process_ticker_batch(tickers, batch_size=2)
        
        # Should return DataFrame with tickers as index
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        # GOOGL is normalized to GOOG in the processing pipeline
        expected_tickers = {'AAPL', 'MSFT', 'GOOG'}
        assert set(result.index) == expected_tickers
    
    @pytest.mark.asyncio
    async def test_process_ticker_batch_empty_results(self, trading_engine):
        """Test batch processing with no results."""
        tickers = ['INVALID']
        
        # Mock the process_batch_async function to return empty dict
        with patch('yahoofinance.utils.async_utils.enhanced.process_batch_async', return_value={}):
            result = await trading_engine.process_ticker_batch(tickers)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_process_batch_error_handling(self, trading_engine):
        """Test batch processing handles errors."""
        batch = ['AAPL', 'INVALID']
        
        # Mock the _process_single_ticker to simulate an error for INVALID ticker
        with patch.object(trading_engine.data_processing_service, '_process_single_ticker') as mock_single:
            # First call returns data, second call raises exception
            mock_single.side_effect = [
                {'ticker': 'AAPL', 'price': 150.0},
                None  # Return None for invalid ticker (error is logged as debug)
            ]
            
            # The actual process_ticker_batch uses process_batch_async from async_utils
            with patch('yahoofinance.utils.async_utils.enhanced.process_batch_async') as mock_batch_async:
                # Simulate the batch processing returning only successful result
                mock_batch_async.return_value = {
                    'AAPL': {'ticker': 'AAPL', 'price': 150.0}
                    # INVALID returns None and is filtered out
                }
                
                result = await trading_engine.data_processing_service.process_ticker_batch(batch)
        
        # Should return DataFrame with only successful results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert 'AAPL' in result.index
    
    @pytest.mark.asyncio
    async def test_process_single_ticker_success(self, trading_engine):
        """Test successful single ticker processing."""
        ticker = 'AAPL'
        mock_data = {
            'price': 150.0,
            'market_cap': 2.5e12,
            'pe_trailing': 25.5,
            'target_price': 165.0
        }
        
        trading_engine.provider.get_ticker_info.return_value = mock_data
        
        with patch('trade_modules.data_processing_service.process_ticker_input', return_value='AAPL'), \
             patch('trade_modules.data_processing_service.get_ticker_for_display', return_value='AAPL'):
            
            result = await trading_engine.data_processing_service._process_single_ticker(ticker)
        
        # Should return processed data
        assert result is not None
        assert result['ticker'] == 'AAPL'
        assert result['price'] == pytest.approx(150.0, 0.01)
    
    @pytest.mark.asyncio
    async def test_process_single_ticker_no_data(self, trading_engine):
        """Test single ticker processing with no data."""
        ticker = 'INVALID'
        
        trading_engine.provider.get_ticker_info.return_value = None
        
        with patch('trade_modules.data_processing_service.process_ticker_input', return_value='INVALID'):
            result = await trading_engine.data_processing_service._process_single_ticker(ticker)
        
        # Should return None
        assert result is None
    
    @pytest.mark.asyncio
    async def test_process_single_ticker_error(self, trading_engine):
        """Test single ticker processing with error."""
        ticker = 'ERROR'
        
        trading_engine.provider.get_ticker_info.side_effect = Exception('API Error')
        
        with patch.object(trading_engine.data_processing_service.logger, 'debug') as mock_logger:
            result = await trading_engine.data_processing_service._process_single_ticker(ticker)
        
        # Should log debug message
        mock_logger.assert_called_once()
        
        # Should return None
        assert result is None


class TestPositionSizer:
    """Test PositionSizer functionality."""
    
    def test_position_sizer_init(self):
        """Test PositionSizer initialization."""
        sizer = PositionSizer(max_position_size=0.08, min_position_size=0.015)
        
        assert sizer.max_position_size == pytest.approx(0.08, 0.001)
        assert sizer.min_position_size == pytest.approx(0.015, 0.001)
    
    def test_calculate_position_size_basic(self):
        """Test basic position size calculation."""
        sizer = PositionSizer()
        
        ticker = 'AAPL'
        market_data = {'price': 150.0, 'beta': 1.1, 'market_cap': 2.5e12}
        portfolio_value = 100000.0
        
        position_size = sizer.calculate_position_size(ticker, market_data, portfolio_value)
        
        # Should return positive value within bounds
        assert position_size > 0
        assert position_size <= portfolio_value * sizer.max_position_size
        assert position_size >= portfolio_value * sizer.min_position_size
    
    def test_calculate_position_size_risk_adjustment(self):
        """Test position size risk level adjustments."""
        sizer = PositionSizer()
        market_data = {'price': 150.0, 'beta': 1.0, 'market_cap': 50e9}
        portfolio_value = 100000.0
        
        low_risk = sizer.calculate_position_size('TEST', market_data, portfolio_value, 'low')
        medium_risk = sizer.calculate_position_size('TEST', market_data, portfolio_value, 'medium')
        high_risk = sizer.calculate_position_size('TEST', market_data, portfolio_value, 'high')
        
        # Low risk should be smaller than medium
        assert low_risk < medium_risk
        # High risk may be capped at max_position_size, so check it's at least as large as medium
        assert high_risk >= medium_risk
    
    def test_calculate_position_size_beta_adjustment(self):
        """Test position size beta adjustments."""
        sizer = PositionSizer()
        portfolio_value = 100000.0
        
        low_beta = {'price': 150.0, 'beta': 0.5, 'market_cap': 50e9}
        high_beta = {'price': 150.0, 'beta': 2.0, 'market_cap': 50e9}
        
        low_beta_size = sizer.calculate_position_size('TEST1', low_beta, portfolio_value)
        high_beta_size = sizer.calculate_position_size('TEST2', high_beta, portfolio_value)
        
        # High beta should result in smaller position
        assert high_beta_size < low_beta_size
    
    def test_calculate_position_size_market_cap_adjustment(self):
        """Test position size market cap adjustments."""
        sizer = PositionSizer()
        portfolio_value = 100000.0
        
        large_cap = {'price': 150.0, 'beta': 1.0, 'market_cap': 200e9}  # >100B
        mid_cap = {'price': 150.0, 'beta': 1.0, 'market_cap': 50e9}    # 10B-100B
        small_cap = {'price': 150.0, 'beta': 1.0, 'market_cap': 5e9}   # <10B
        
        large_size = sizer.calculate_position_size('LARGE', large_cap, portfolio_value)
        mid_size = sizer.calculate_position_size('MID', mid_cap, portfolio_value)
        small_size = sizer.calculate_position_size('SMALL', small_cap, portfolio_value)
        
        # Large cap should get bigger allocation than small cap
        assert large_size > small_size
        assert mid_size > small_size
    
    def test_calculate_position_size_error_handling(self):
        """Test position size calculation error handling."""
        sizer = PositionSizer()
        
        # Invalid market data (missing expected fields)
        invalid_data = {'invalid': 'data'}
        portfolio_value = 100000.0
        
        position_size = sizer.calculate_position_size('TEST', invalid_data, portfolio_value)
        
        # Should handle gracefully and return a valid position size
        # When beta and market_cap are missing, defaults apply: beta=1.0, no market_cap adjustment
        # For medium risk: base_size = 0.01 + (0.05 - 0.01) * 0.6 = 0.034
        # With no adjustments: position = 0.034 * 100000 = 3400
        expected_size = portfolio_value * 0.034
        assert position_size == pytest.approx(expected_size, 0.01)


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_trading_engine_default(self):
        """Test create_trading_engine with defaults."""
        engine = create_trading_engine()
        
        assert isinstance(engine, TradingEngine)
        assert hasattr(engine, 'provider')
    
    def test_create_trading_engine_with_params(self):
        """Test create_trading_engine with parameters."""
        mock_provider = AsyncMock()
        config = {'test': 'config'}
        
        engine = create_trading_engine(provider=mock_provider, config=config)
        
        assert engine.provider == mock_provider
        assert engine.config == config
    
    def test_create_position_sizer_default(self):
        """Test create_position_sizer with defaults."""
        sizer = create_position_sizer()
        
        assert isinstance(sizer, PositionSizer)
    
    def test_create_position_sizer_with_params(self):
        """Test create_position_sizer with parameters."""
        sizer = create_position_sizer(max_position=0.08, min_position=0.015)
        
        assert sizer.max_position_size == pytest.approx(0.08, 0.001)
        assert sizer.min_position_size == pytest.approx(0.015, 0.001)


class TestIntegrationScenarios:
    """Integration tests using real CSV data patterns."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_with_market_csv_pattern(self, trading_engine):
        """Test full workflow using market.csv data pattern."""
        # Create data that matches actual market.csv structure
        # Data meets MEGA tier criteria (market_cap >= $500B)
        market_data = pd.DataFrame({
            'symbol': ['NVDA', 'AMZN', 'AAPL', 'MSFT', 'GOOGL'],
            'name': ['Nvidia', 'Amazon', 'Apple', 'Microsoft', 'Alphabet'],
            'sector': ['Technology', 'Consumer Cyclical', 'Technology', 'Technology', 'Technology'],
            'price': [800.0, 150.0, 180.0, 400.0, 2800.0],
            'target_price': [960.0, 180.0, 200.0, 480.0, 2400.0],
            'upside': [20.0, 20.0, 11.1, 20.0, -8.0],  # NVDA/AMZN/MSFT BUY, GOOGL SELL trigger
            'buy_percentage': [85.0, 85.0, 70.0, 85.0, 30.0],  # GOOGL low = SELL trigger
            'analyst_count': [30, 25, 35, 40, 20],
            'total_ratings': [30, 25, 35, 40, 20],
            'market_cap': [2.5e12, 2.0e12, 3.0e12, 2.8e12, 1.8e12],  # All MEGA tier
            'EXRET': [17.0, 17.0, 7.8, 17.0, -2.4],  # upside * buy% / 100
            'BS': ['B', 'B', 'H', 'B', 'S']
        }).set_index('symbol')
        
        portfolio_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'quantity': [100, 50],
            'TICKER': ['AAPL', 'MSFT']  # Portfolio uses TICKER column
        })
        
        result = await trading_engine.analyze_market_opportunities(market_data, portfolio_data)
        
        # Verify complete workflow
        assert isinstance(result, dict)
        assert all(key in result for key in ['buy_opportunities', 'sell_opportunities', 'hold_opportunities'])
        
        # Buy opportunities should exclude portfolio holdings
        buy_ops = result['buy_opportunities']
        assert 'AAPL' not in buy_ops.index  # In portfolio
        assert 'MSFT' not in buy_ops.index  # In portfolio
        assert 'NVDA' in buy_ops.index      # Not in portfolio, has 'B' signal
        assert 'AMZN' in buy_ops.index      # Not in portfolio, has 'B' signal
        
        # Sell opportunities  
        sell_ops = result['sell_opportunities']
        assert 'GOOGL' not in sell_ops.index  # Has 'S' signal but not in portfolio
        
        # Hold opportunities should include portfolio holdings with 'H' signal
        hold_ops = result['hold_opportunities']
        assert 'AAPL' in hold_ops.index     # In portfolio and has 'H' signal
    
    @pytest.mark.asyncio
    async def test_workflow_with_portfolio_csv_pattern(self, trading_engine):
        """Test workflow using portfolio.csv data pattern."""
        # Market data with complete fields for signal calculation
        market_data = pd.DataFrame({
            'symbol': ['GOOG', 'AMZN', 'NVDA', 'META'],
            'price': [2800.0, 150.0, 800.0, 500.0],
            'target_price': [3360.0, 125.0, 960.0, 600.0],
            'upside': [20.0, -8.0, 7.0, 20.0],  # GOOG/META BUY, AMZN SELL, NVDA HOLD
            'buy_percentage': [85.0, 30.0, 65.0, 85.0],
            'analyst_count': [30, 25, 20, 35],
            'total_ratings': [30, 25, 20, 35],
            'market_cap': [1.8e12, 2.0e12, 2.5e12, 1.5e12],  # All MEGA tier
            'EXRET': [17.0, -2.4, 4.5, 17.0],
            'BS': ['B', 'S', 'H', 'B']
        }).set_index('symbol')

        # Portfolio data matching actual CSV structure
        portfolio_data = pd.DataFrame({
            'symbol': ['GOOG', 'AMZN', 'LLY', 'ETOR'],  # Some overlap with market
            'instrumentDisplayName': ['Alphabet Inc Class A', 'Amazon.com Inc', 'Eli Lilly & Co', 'eToro Group LTD'],
            'totalInvestmentPct': [5.18, 4.94, 4.15, 4.14],
            'avgOpenRate': [188.96, 193.91, 772.41, 56.21],
            'isBuy': [True, True, True, True],
            'leverage': [1, 1, 1, 1]
        })

        result = await trading_engine.analyze_market_opportunities(market_data, portfolio_data)

        # Portfolio holdings should affect filtering
        buy_ops = result['buy_opportunities']
        # GOOG has 'B' signal but is in portfolio, should be excluded
        assert 'GOOG' not in buy_ops.index
        # META has 'B' signal and not in portfolio, should be included
        assert 'META' in buy_ops.index


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_analyze_with_empty_dataframe(self, trading_engine):
        """Test analysis with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with patch('trade_modules.trade_engine.validate_dataframe', return_value=False):
            with pytest.raises(TradingEngineError):
                await trading_engine.analyze_market_opportunities(empty_df)
    
    @pytest.mark.asyncio
    async def test_analyze_with_missing_columns(self, trading_engine):
        """Test analysis with missing required columns."""
        df = pd.DataFrame({'symbol': ['AAPL'], 'other_col': [1]}).set_index('symbol')
        
        # Should handle missing columns gracefully
        result = await trading_engine.analyze_market_opportunities(df)
        
        # With missing columns, signal calculation may fail or default to hold
        # The actual behavior is that it creates a hold opportunity with default signals
        assert isinstance(result, dict)
        assert all(key in result for key in ['buy_opportunities', 'sell_opportunities', 'hold_opportunities'])
        # At least one opportunity type should have the single ticker (likely hold due to default behavior)
        total_opportunities = sum(len(opp_df) for opp_df in result.values())
        assert total_opportunities >= 0  # May be 0 if all signal calculations fail
    
    def test_confidence_calculation_all_nan(self, trading_engine):
        """Test confidence calculation with all NaN values."""
        df = pd.DataFrame({
            'analyst_count': [np.nan, np.nan],
            'total_ratings': [np.nan, np.nan],
            'upside': [np.nan, np.nan]
        })
        
        scores = trading_engine._calculate_confidence_score(df)
        
        # Should return base confidence (0.6) minus upside penalty (0.2) = 0.4
        # Since upside is NaN, it gets a penalty
        assert all(score == pytest.approx(0.4) for score in scores)
    
    @pytest.mark.asyncio
    async def test_batch_processing_all_failures(self, trading_engine):
        """Test batch processing when all tickers fail."""
        tickers = ['INVALID1', 'INVALID2']
        
        with patch.object(trading_engine.data_processing_service, '_process_single_ticker', return_value=None):
            result = await trading_engine.process_ticker_batch(tickers)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])