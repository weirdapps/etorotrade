"""
Behavioral validation tests using actual CSV data from the project.

These tests load real CSV files and validate the exact current behavior
of TradingEngine to catch any regressions during refactoring.
"""

import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

from trade_modules.trade_engine import TradingEngine


@pytest.fixture
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture
def input_data_path(project_root):
    """Get the input data directory path."""
    return project_root / "yahoofinance" / "input"


@pytest.fixture
def market_csv_data(input_data_path):
    """Load actual market.csv data."""
    market_path = input_data_path / "market.csv"
    if market_path.exists():
        return pd.read_csv(market_path).set_index('symbol')
    else:
        # Fallback data if file doesn't exist
        return pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'name': ['Apple Inc.', 'Microsoft', 'Alphabet Inc.'],
            'sector': ['Technology', 'Technology', 'Technology']
        }).set_index('symbol')


@pytest.fixture
def portfolio_csv_data(input_data_path):
    """Load actual portfolio.csv data."""
    portfolio_path = input_data_path / "portfolio.csv"
    if portfolio_path.exists():
        return pd.read_csv(portfolio_path)
    else:
        # Fallback data if file doesn't exist
        return pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'instrumentDisplayName': ['Apple Inc.', 'Microsoft'],
            'totalInvestmentPct': [5.0, 4.0]
        })


@pytest.fixture
def notrade_csv_data(input_data_path):
    """Load actual notrade.csv data."""
    notrade_path = input_data_path / "notrade.csv"
    if notrade_path.exists():
        return pd.read_csv(notrade_path)
    else:
        # Fallback data if file doesn't exist
        return pd.DataFrame({
            'symbol': ['BANNED1', 'BANNED2']
        })


@pytest.fixture
def enhanced_market_data(market_csv_data):
    """Enhance market data with trading signals and financial data."""
    # Add required columns for TradingEngine analysis
    enhanced = market_csv_data.copy()
    
    # Add price data (mock realistic values)
    n_stocks = len(enhanced)
    enhanced['price'] = np.random.uniform(50, 500, n_stocks)
    enhanced['target_price'] = enhanced['price'] * np.random.uniform(1.05, 1.25, n_stocks)
    enhanced['upside'] = ((enhanced['target_price'] - enhanced['price']) / enhanced['price'] * 100)
    
    # Add analyst data
    enhanced['analyst_count'] = np.random.randint(10, 40, n_stocks)
    enhanced['total_ratings'] = enhanced['analyst_count']
    enhanced['buy_percentage'] = np.random.uniform(60, 95, n_stocks)
    
    # Add financial metrics
    enhanced['market_cap'] = np.random.uniform(1e9, 3e12, n_stocks)
    enhanced['beta'] = np.random.uniform(0.5, 2.5, n_stocks)
    enhanced['pe_ratio'] = np.random.uniform(10, 50, n_stocks)
    enhanced['dividend_yield'] = np.random.uniform(0, 5, n_stocks)
    
    # Add trading signals based on upside
    enhanced['BS'] = np.where(enhanced['upside'] > 15, 'B',
                              np.where(enhanced['upside'] < 5, 'S', 'H'))
    
    # Add expected return and EXRET
    enhanced['expected_return'] = enhanced['upside'] * 0.8
    enhanced['EXRET'] = enhanced['expected_return'] + np.random.normal(0, 2, n_stocks)
    
    return enhanced


class TestBehavioralValidation:
    """Validate exact current behavior of TradingEngine."""
    
    @pytest.mark.asyncio
    async def test_analyze_market_baseline_behavior(self, enhanced_market_data):
        """Test analyze_market_opportunities baseline behavior."""
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        # Record the exact behavior with current implementation
        result = await engine.analyze_market_opportunities(enhanced_market_data)
        
        # Validate structure
        assert isinstance(result, dict)
        assert set(result.keys()) == {'buy_opportunities', 'sell_opportunities', 'hold_opportunities'}
        
        # Each result should be a DataFrame
        for key, df in result.items():
            assert isinstance(df, pd.DataFrame)
        
        # Total opportunities should equal input (all stocks categorized)
        total_results = sum(len(df) for df in result.values())
        assert total_results == len(enhanced_market_data)
        
        # No overlapping stocks between categories
        all_indices = []
        for df in result.values():
            all_indices.extend(df.index.tolist())
        assert len(all_indices) == len(set(all_indices))
        
        # Buy opportunities should only contain BS == 'B'
        buy_signals = enhanced_market_data.loc[result['buy_opportunities'].index, 'BS']
        assert all(signal == 'B' for signal in buy_signals)
        
        # Sell opportunities should only contain BS == 'S'
        if len(result['sell_opportunities']) > 0:
            sell_signals = enhanced_market_data.loc[result['sell_opportunities'].index, 'BS']
            assert all(signal == 'S' for signal in sell_signals)
        
        # Hold opportunities should only contain BS == 'H'
        if len(result['hold_opportunities']) > 0:
            hold_signals = enhanced_market_data.loc[result['hold_opportunities'].index, 'BS']
            assert all(signal == 'H' for signal in hold_signals)
    
    @pytest.mark.asyncio
    async def test_portfolio_filtering_baseline_behavior(self, enhanced_market_data, portfolio_csv_data):
        """Test portfolio filtering baseline behavior."""
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        # Get baseline behavior without portfolio
        result_no_portfolio = await engine.analyze_market_opportunities(enhanced_market_data)
        
        # Get behavior with portfolio
        result_with_portfolio = await engine.analyze_market_opportunities(
            enhanced_market_data, 
            portfolio_df=portfolio_csv_data
        )
        
        # Portfolio filtering should reduce buy opportunities
        # (portfolio holdings should be excluded from buy)
        assert len(result_with_portfolio['buy_opportunities']) <= len(result_no_portfolio['buy_opportunities'])
        
        # Extract portfolio symbols (handle different column names)
        portfolio_symbols = set()
        for col in ['symbol', 'TICKER', 'Ticker', 'ticker']:
            if col in portfolio_csv_data.columns:
                portfolio_symbols.update(portfolio_csv_data[col].dropna())
        
        # Portfolio symbols should not appear in buy opportunities
        buy_symbols = set(result_with_portfolio['buy_opportunities'].index)
        portfolio_in_market = portfolio_symbols.intersection(set(enhanced_market_data.index))
        portfolio_in_buy = portfolio_in_market.intersection(buy_symbols)
        assert len(portfolio_in_buy) == 0, f"Portfolio symbols {portfolio_in_buy} found in buy opportunities"
    
    @pytest.mark.asyncio
    async def test_notrade_filtering_baseline_behavior(self, enhanced_market_data, notrade_csv_data, input_data_path):
        """Test notrade filtering baseline behavior."""
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        # Create temporary notrade file
        notrade_path = input_data_path / "test_notrade.csv"
        
        # Add some market symbols to notrade for testing
        test_notrade = notrade_csv_data.copy()
        if 'symbol' not in test_notrade.columns and 'Ticker' in test_notrade.columns:
            test_notrade = test_notrade.rename(columns={'Ticker': 'symbol'})
        
        # Add a few market symbols to notrade list
        market_symbols = enhanced_market_data.index.tolist()[:2]
        test_notrade_symbols = pd.DataFrame({'Ticker': market_symbols})
        test_notrade_symbols.to_csv(notrade_path, index=False)
        
        try:
            # Test with notrade filtering
            result = await engine.analyze_market_opportunities(
                enhanced_market_data,
                notrade_path=str(notrade_path)
            )
            
            # Notrade symbols should not appear in any results
            all_result_symbols = set()
            for df in result.values():
                all_result_symbols.update(df.index)
            
            for symbol in market_symbols:
                assert symbol not in all_result_symbols, f"Notrade symbol {symbol} found in results"
            
            # Total results should be reduced
            total_results = sum(len(df) for df in result.values())
            expected_total = len(enhanced_market_data) - len(market_symbols)
            assert total_results == expected_total
            
        finally:
            # Cleanup
            if notrade_path.exists():
                notrade_path.unlink()
    
    @pytest.mark.asyncio
    async def test_confidence_score_baseline_behavior(self, enhanced_market_data):
        """Test confidence score calculation baseline behavior."""
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        # Test confidence calculation directly
        confidence_scores = engine._calculate_confidence_score(enhanced_market_data)
        
        # Should return Series with same length as input
        assert isinstance(confidence_scores, pd.Series)
        assert len(confidence_scores) == len(enhanced_market_data)
        
        # All scores should be between 0 and 1
        assert all(0 <= score <= 1 for score in confidence_scores)
        
        # Scores should vary based on analyst_count
        high_analyst_mask = enhanced_market_data['analyst_count'] >= 30
        low_analyst_mask = enhanced_market_data['analyst_count'] < 15
        
        if high_analyst_mask.any() and low_analyst_mask.any():
            high_analyst_scores = confidence_scores[high_analyst_mask].mean()
            low_analyst_scores = confidence_scores[low_analyst_mask].mean()
            assert high_analyst_scores > low_analyst_scores
    
    @pytest.mark.asyncio
    async def test_act_column_conversion_baseline_behavior(self, enhanced_market_data):
        """Test ACT column conversion to BS column baseline behavior."""
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        # Create data with ACT column instead of BS
        act_data = enhanced_market_data.copy()
        act_data['ACT'] = act_data['BS']
        act_data = act_data.drop(columns=['BS'])
        
        # Test conversion
        with patch.object(engine.logger, 'info') as mock_logger:
            result = await engine.analyze_market_opportunities(act_data)
        
        # Should log ACT column usage
        log_calls = [call for call in mock_logger.call_args_list 
                    if 'Using ACT column values as BS column' in str(call)]
        assert len(log_calls) == 1
        
        # Results should be identical to BS column behavior
        bs_result = await engine.analyze_market_opportunities(enhanced_market_data)
        
        # Same number of opportunities in each category
        assert len(result['buy_opportunities']) == len(bs_result['buy_opportunities'])
        assert len(result['sell_opportunities']) == len(bs_result['sell_opportunities'])
        assert len(result['hold_opportunities']) == len(bs_result['hold_opportunities'])
    
    @pytest.mark.asyncio
    async def test_missing_bs_column_calculation_baseline_behavior(self, enhanced_market_data):
        """Test trading signal calculation when BS column is missing."""
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        # Remove BS column to trigger calculation
        no_bs_data = enhanced_market_data.drop(columns=['BS'])
        
        # Mock the calculation functions
        with patch('trade_modules.trade_engine.calculate_expected_return') as mock_expected, \
             patch('trade_modules.trade_engine.calculate_exret') as mock_exret, \
             patch('trade_modules.trade_engine.calculate_action') as mock_action:
            
            # Set up mocks to return data with BS column
            mock_expected.return_value = no_bs_data
            mock_exret.return_value = no_bs_data  
            mock_action_result = no_bs_data.copy()
            mock_action_result['BS'] = ['B'] * len(no_bs_data)
            mock_action.return_value = mock_action_result
            
            result = await engine.analyze_market_opportunities(no_bs_data)
        
        # Should call all calculation functions
        mock_expected.assert_called_once()
        mock_exret.assert_called_once()
        mock_action.assert_called_once()
        
        # Should return valid results
        assert isinstance(result, dict)
        total_results = sum(len(df) for df in result.values())
        assert total_results == len(no_bs_data)
    
    def test_position_sizer_baseline_behavior(self):
        """Test PositionSizer baseline calculation behavior."""
        from trade_modules.trade_engine import PositionSizer
        
        sizer = PositionSizer(max_position_size=0.05, min_position_size=0.01)
        
        # Test with standard inputs
        market_data = {
            'price': 150.0,
            'beta': 1.2,
            'market_cap': 2.5e12
        }
        portfolio_value = 100000.0
        
        # Test different risk levels
        low_risk = sizer.calculate_position_size('TEST', market_data, portfolio_value, 'low')
        medium_risk = sizer.calculate_position_size('TEST', market_data, portfolio_value, 'medium')
        high_risk = sizer.calculate_position_size('TEST', market_data, portfolio_value, 'high')
        
        # Validate risk level ordering
        assert low_risk < medium_risk < high_risk
        
        # All should be within bounds
        min_position = portfolio_value * sizer.min_position_size
        max_position = portfolio_value * sizer.max_position_size
        
        assert min_position <= low_risk <= max_position
        assert min_position <= medium_risk <= max_position
        assert min_position <= high_risk <= max_position
        
        # Test beta adjustment
        high_beta_data = market_data.copy()
        high_beta_data['beta'] = 2.5
        
        low_beta_data = market_data.copy()
        low_beta_data['beta'] = 0.5
        
        high_beta_position = sizer.calculate_position_size('HIGH_BETA', high_beta_data, portfolio_value)
        low_beta_position = sizer.calculate_position_size('LOW_BETA', low_beta_data, portfolio_value)
        
        # High beta should result in smaller position
        assert high_beta_position < low_beta_position
    
    @pytest.mark.asyncio
    async def test_error_handling_baseline_behavior(self, enhanced_market_data):
        """Test error handling baseline behavior."""
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        # Test with invalid data
        with patch('trade_modules.trade_engine.validate_dataframe', return_value=False):
            with pytest.raises(Exception):  # Should raise TradingEngineError
                await engine.analyze_market_opportunities(enhanced_market_data)
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        with patch('trade_modules.trade_engine.validate_dataframe', return_value=True):
            result = await engine.analyze_market_opportunities(empty_df)
            
            # Should return empty results
            assert all(len(df) == 0 for df in result.values())
        
        # Test confidence calculation error handling
        problematic_data = pd.DataFrame({
            'analyst_count': ['invalid', None, 'bad_data'],
            'total_ratings': [None, 'invalid', 'also_bad']
        })
        
        with patch.object(engine.logger, 'warning') as mock_logger:
            scores = engine._calculate_confidence_score(problematic_data)
        
        # Should log warning and return default scores
        mock_logger.assert_called_once()
        assert all(score == 0.7 for score in scores)


class TestBehavioralRegression:
    """Regression tests to catch specific behavioral changes."""
    
    @pytest.mark.asyncio
    async def test_sell_opportunities_confidence_filtering(self, enhanced_market_data):
        """Test that sell opportunities are filtered by confidence score."""
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        # Create data with sell signals but varying confidence
        sell_data = enhanced_market_data.copy()
        sell_data['BS'] = 'S'  # All sell signals
        sell_data['confidence_score'] = [0.5, 0.7, 0.9, 0.4, 0.8]  # Mix of high/low confidence
        
        result = await engine.analyze_market_opportunities(sell_data)
        
        sell_ops = result['sell_opportunities']
        
        # Only high confidence (> 0.6) should be included
        if len(sell_ops) > 0:
            assert all(sell_ops['confidence_score'] > 0.6)
        
        # Low confidence should be excluded
        expected_excluded = (sell_data['confidence_score'] <= 0.6).sum()
        expected_included = len(sell_data) - expected_excluded
        assert len(sell_ops) == expected_included
    
    @pytest.mark.asyncio
    async def test_portfolio_ticker_equivalence_checking(self, enhanced_market_data):
        """Test that portfolio filtering uses ticker equivalence checking."""
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        # Create portfolio with ticker variants
        portfolio_data = pd.DataFrame({
            'symbol': ['AAPL.US', 'MSFT'],  # AAPL.US should match AAPL
            'TICKER': ['AAPL.US', 'MSFT'],
            'quantity': [100, 50]
        })
        
        # Ensure market data has AAPL (without .US)
        if 'AAPL' not in enhanced_market_data.index:
            # Add AAPL to market data
            aapl_data = enhanced_market_data.iloc[0].copy()
            aapl_data.name = 'AAPL'
            enhanced_market_data = pd.concat([enhanced_market_data, aapl_data.to_frame().T])
        
        # Set AAPL to have buy signal
        enhanced_market_data.loc['AAPL', 'BS'] = 'B'
        
        with patch('trade_modules.trade_engine.are_equivalent_tickers') as mock_equiv:
            # Mock equivalence: AAPL == AAPL.US
            mock_equiv.side_effect = lambda t1, t2: (
                (t1 == 'AAPL' and t2 == 'AAPL.US') or 
                (t1 == 'AAPL.US' and t2 == 'AAPL') or
                t1 == t2
            )
            
            result = await engine.analyze_market_opportunities(
                enhanced_market_data,
                portfolio_df=portfolio_data
            )
        
        # Should have called equivalence checking
        assert mock_equiv.called
        
        # AAPL should be excluded from buy opportunities due to equivalence with AAPL.US
        buy_ops = result['buy_opportunities']
        assert 'AAPL' not in buy_ops.index
    
    @pytest.mark.asyncio
    async def test_logging_behavior_consistency(self, enhanced_market_data):
        """Test that logging behavior is consistent."""
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        with patch.object(engine.logger, 'info') as mock_info:
            result = await engine.analyze_market_opportunities(enhanced_market_data)
        
        # Should log analysis summary
        summary_calls = [call for call in mock_info.call_args_list 
                        if 'Analysis complete' in str(call)]
        assert len(summary_calls) == 1
        
        # Summary should contain counts
        summary_msg = summary_calls[0][0][0]
        assert 'buy' in summary_msg
        assert 'sell' in summary_msg  
        assert 'hold' in summary_msg
        
        # Counts should match actual results
        buy_count = len(result['buy_opportunities'])
        sell_count = len(result['sell_opportunities'])
        hold_count = len(result['hold_opportunities'])
        
        assert str(buy_count) in summary_msg
        assert str(sell_count) in summary_msg
        assert str(hold_count) in summary_msg


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])