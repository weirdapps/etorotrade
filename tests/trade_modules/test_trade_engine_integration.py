"""
Integration tests for TradingEngine using real CSV data patterns.

These tests use actual CSV file structures to ensure the TradingEngine
behaves correctly with real-world data scenarios.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, AsyncMock

from trade_modules.trade_engine import TradingEngine, create_trading_engine


@pytest.fixture
def real_market_csv_data():
    """Create market data matching actual market.csv structure."""
    return pd.DataFrame({
        'symbol': ['NVDA', 'SHW', 'AMZN', 'AMGN', 'CRM', 'HON', 'GS', 'AAPL', 'NKE', 'GOOGL'],
        'name': [
            'Nvidia',
            'Sherwin-Williams', 
            'Amazon.com, Inc.amazon.com',
            'Amgen Inc.',
            'Salesforce, Inc.',
            'Honeywell International Inc.',
            'The Goldman Sachs Group, Inc.',
            'Apple Inc.',
            'Nike, Inc.',
            'Alphabet Inc.'
        ],
        'sector': [
            'Technology', 'Basic Materials', 'Consumer Cyclical', 'Healthcare',
            'Technology', 'Industrials', 'Financial Services', 'Technology',
            'Consumer Cyclical', 'Technology'
        ],
        'subSector': [
            'Semiconductors', 'Chemicals - Specialty', 'Specialty Retail',
            'Drug Manufacturers - General', 'Software - Application', 'Conglomerates',
            'Financial - Capital Markets', 'Consumer Electronics',
            'Apparel - Footwear & Accessories', 'Internet Content & Information'
        ],
        'headQuarter': [
            'Santa Clara, CA', 'Cleveland, OH', 'Seattle, WA', 'Thousand Oaks, CA',
            'San Francisco, CA', 'Charlotte, NC', 'New York, NY', 'Cupertino, CA',
            'Beaverton, OR', 'Mountain View, CA'
        ],
        'founded': [
            '1993-04-05', '1866-01-01', '1994-07-05', '1980-04-08',
            '1999-02-03', '1906-01-01', '1869-01-01', '1976-04-01',
            '1964-01-25', '1998-09-04'
        ]
    }).set_index('symbol')


@pytest.fixture 
def real_portfolio_csv_data():
    """Create portfolio data matching actual portfolio.csv structure."""
    return pd.DataFrame({
        'symbol': ['GOOG', 'AMZN', 'LLY', 'ETOR', 'NVDA', 'META', 'MSFT', '0700.HK', 'DTE.DE', 'UBER'],
        'instrumentDisplayName': [
            'Alphabet Inc Class A', 'Amazon.com Inc', 'Eli Lilly & Co', 'eToro Group LTD',
            'NVIDIA Corporation', 'Meta Platforms Inc', 'Microsoft', 'Tencent', 
            'Deutsche Telekom AG', 'Uber Technologies Inc'
        ],
        'instrumentId': [6434, 1005, 1567, 12200, 1137, 1003, 1004, 2339, 1207, 1020],
        'numPositions': [9, 18, 5, 7, 17, 5, 10, 5, 2, 8],
        'totalInvestmentPct': [5.186, 4.940, 4.148, 4.143, 3.734, 3.551, 3.526, 3.111, 3.111, 3.042],
        'avgOpenRate': [188.96, 193.91, 772.41, 56.21, 117.01, 729.32, 450.78, 530.45, 31.09, 73.45],
        'totalNetProfit': [81.51, 621.12, -45.39, -118.31, 2579.20, 32.21, 345.43, 79.44, 0.64, -89.23],
        'totalNetProfitPct': [15.72, 125.74, -10.94, -28.56, 690.81, 9.07, 97.96, 25.53, 0.20, -12.15],
        'isBuy': [True, True, True, True, True, True, True, True, True, True],
        'leverage': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'instrumentTypeId': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        'exchangeId': [4, 4, 5, 4, 4, 4, 4, 21, 6, 4],
        'exchangeName': ['NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'HKEX', 'Xetra', 'NASDAQ']
    })


@pytest.fixture
def real_notrade_csv_data():
    """Create notrade data matching actual notrade.csv structure."""
    return pd.DataFrame({
        'symbol': ['0883.HK', 'ATYM.L', 'CSH.DE', 'CWC.DE', 'EOAN.DE'],
        'company': [
            'CNOOC', 'Atalaya Mining Copper, S.A.', 'CENIT Aktiengesellschaft',
            'CEWE Stiftung & Co. KGaA', 'E.ON SE'
        ],
        'price': [17.88, 365.0, 7.25, 97.5, 12.45],
        'target_price': [22.37, 488.33, 16.74, 139.0, 15.20],
        'upside': [25.10, 33.79, 130.90, 42.56, 22.09],
        'analyst_count': [18.0, 6.0, 5.0, 6.0, 12.0],
        'buy_percentage': [94.44, 85.71, 100.0, 100.0, 83.33],
        'total_ratings': [18.0, 7.0, 5.0, 6.0, 12.0],
        'A': ['A', 'A', 'A', 'A', 'A'],
        'EXRET': [23.70, 28.96, 130.90, 42.56, 22.09],
        'beta': [0.742, 1.642, 0.725, 0.989, 1.234],
        'pe_trailing': [5.57, 24.33, 20.14, 10.71, 8.92],
        'pe_forward': [5.59, 6.76, 8.63, 11.16, 7.45],
        'dividend_yield': [8.28, 1.69, 0.55, 2.67, 5.12],
        'exchange': ['HK', 'L', 'DE', 'DE', 'DE']
    })


@pytest.fixture
def market_data_with_prices():
    """Market data enhanced with price and trading signal data."""
    return pd.DataFrame({
        'symbol': ['NVDA', 'AMZN', 'AAPL', 'MSFT', 'GOOGL', 'META', 'TSLA'],
        'name': ['Nvidia', 'Amazon', 'Apple', 'Microsoft', 'Alphabet', 'Meta', 'Tesla'],
        'price': [800.50, 145.75, 185.30, 410.25, 2850.00, 515.80, 245.90],
        'target_price': [950.00, 170.00, 205.00, 470.00, 3100.00, 580.00, 280.00],
        'upside': [18.7, 16.6, 10.6, 14.6, 8.8, 12.4, 13.9],
        'buy_percentage': [88.0, 82.0, 76.0, 85.0, 72.0, 79.0, 68.0],
        'analyst_count': [32, 28, 35, 38, 25, 24, 22],
        'total_ratings': [32, 28, 35, 38, 25, 24, 22],
        'market_cap': [1.9e12, 1.5e12, 2.8e12, 3.1e12, 1.8e12, 1.3e12, 0.8e12],
        'beta': [1.6, 1.2, 1.1, 0.9, 1.1, 1.4, 2.0],
        'pe_ratio': [65.2, 45.8, 28.5, 32.1, 24.8, 22.9, 85.4],
        'dividend_yield': [0.0, 0.0, 0.5, 0.7, 0.0, 0.0, 0.0],
        'BS': ['B', 'B', 'H', 'B', 'S', 'H', 'B'],
        'EXRET': [15.2, 12.8, 5.3, 11.4, -3.2, 7.1, 9.8],
        'expected_return': [15.2, 12.8, 5.3, 11.4, -3.2, 7.1, 9.8]
    }).set_index('symbol')


@pytest.fixture
def temp_csv_files(real_market_csv_data, real_portfolio_csv_data, real_notrade_csv_data):
    """Create temporary CSV files with real data structures."""
    temp_dir = tempfile.mkdtemp()
    
    # Create market.csv
    market_path = os.path.join(temp_dir, 'market.csv')
    real_market_csv_data.to_csv(market_path)
    
    # Create portfolio.csv  
    portfolio_path = os.path.join(temp_dir, 'portfolio.csv')
    real_portfolio_csv_data.to_csv(portfolio_path, index=False)
    
    # Create notrade.csv
    notrade_path = os.path.join(temp_dir, 'notrade.csv')
    real_notrade_csv_data.to_csv(notrade_path, index=False)
    
    yield {
        'market': market_path,
        'portfolio': portfolio_path, 
        'notrade': notrade_path,
        'temp_dir': temp_dir
    }
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


class TestRealDataIntegration:
    """Integration tests using real CSV data structures."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_real_csv_structure(self, market_data_with_prices, real_portfolio_csv_data):
        """Test complete workflow with real CSV structures."""
        # Create engine
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        # Analyze market with portfolio
        result = await engine.analyze_market_opportunities(
            market_data_with_prices,
            portfolio_df=real_portfolio_csv_data
        )
        
        # Verify structure
        assert isinstance(result, dict)
        assert set(result.keys()) == {'buy_opportunities', 'sell_opportunities', 'hold_opportunities'}
        
        # Check portfolio filtering behavior
        buy_ops = result['buy_opportunities']
        sell_ops = result['sell_opportunities']
        hold_ops = result['hold_opportunities']
        
        # Stocks in portfolio should be excluded from buy opportunities
        portfolio_symbols = set(real_portfolio_csv_data['symbol'])
        buy_symbols = set(buy_ops.index)
        
        # NVDA, AMZN, MSFT, META are in portfolio and should be excluded from buy
        assert not buy_symbols.intersection({'NVDA', 'AMZN', 'MSFT', 'META'})
        
        # TSLA has buy signal and not in portfolio, should be in buy opportunities
        assert 'TSLA' in buy_symbols
        
        # Portfolio stocks with hold signals should be in hold opportunities
        assert 'AAPL' in hold_ops.index or 'META' in hold_ops.index  # Depending on signal matching
    
    @pytest.mark.asyncio
    async def test_notrade_filtering_real_data(self, market_data_with_prices, temp_csv_files):
        """Test notrade filtering with real CSV data."""
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        # Add some notrade symbols to market data for testing
        market_with_notrade = market_data_with_prices.copy()
        notrade_data = pd.DataFrame({
            'symbol': ['0883.HK', 'EOAN.DE'],
            'price': [17.88, 12.45],
            'BS': ['B', 'S'],
            'upside': [25.0, 22.0],
            'analyst_count': [18, 12]
        }).set_index('symbol')
        
        market_with_notrade = pd.concat([market_with_notrade, notrade_data])
        
        result = await engine.analyze_market_opportunities(
            market_with_notrade,
            notrade_path=temp_csv_files['notrade']
        )
        
        # Verify notrade symbols are filtered out
        all_results = pd.concat(result.values())
        assert '0883.HK' not in all_results.index
        assert 'EOAN.DE' not in all_results.index
        
        # Regular symbols should remain
        assert 'NVDA' in all_results.index
        assert 'TSLA' in all_results.index
    
    @pytest.mark.asyncio
    async def test_portfolio_column_variations(self, market_data_with_prices):
        """Test portfolio filtering with different column name variations."""
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        # Test different ticker column names
        column_variations = [
            ('TICKER', ['NVDA', 'AAPL', 'MSFT']),
            ('Ticker', ['NVDA', 'AAPL', 'MSFT']),
            ('ticker', ['NVDA', 'AAPL', 'MSFT']),
            ('symbol', ['NVDA', 'AAPL', 'MSFT'])
        ]
        
        for col_name, tickers in column_variations:
            portfolio_df = pd.DataFrame({
                col_name: tickers,
                'quantity': [100, 50, 75],
                'value': [80000, 9265, 30768]
            })
            
            result = await engine.analyze_market_opportunities(
                market_data_with_prices,
                portfolio_df=portfolio_df
            )
            
            # Verify portfolio filtering works regardless of column name
            buy_ops = result['buy_opportunities']
            
            # Portfolio stocks should be excluded from buy opportunities
            for ticker in tickers:
                if ticker in market_data_with_prices.index:
                    assert ticker not in buy_ops.index, f"Portfolio stock {ticker} should be excluded from buy ops"
    
    @pytest.mark.asyncio
    async def test_act_column_handling_real_scenario(self, real_market_csv_data):
        """Test ACT column is properly converted to BS column."""
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        # Add ACT column instead of BS
        market_with_act = real_market_csv_data.copy()
        market_with_act['ACT'] = ['B', 'H', 'B', 'S', 'H', 'B', 'S', 'B', 'H', 'S']
        market_with_act['price'] = [800, 350, 145, 475, 280, 220, 410, 185, 165, 2850]
        market_with_act['upside'] = [15.0, 8.5, 16.6, -2.1, 12.3, 18.9, -5.2, 10.6, 9.8, 8.8]
        market_with_act['analyst_count'] = [30, 15, 28, 20, 25, 18, 22, 35, 12, 25]
        
        with patch.object(engine.logger, 'info') as mock_logger:
            result = await engine.analyze_market_opportunities(market_with_act)
        
        # Verify ACT column usage is logged
        log_calls = [call for call in mock_logger.call_args_list 
                    if 'Using ACT column values as BS column' in str(call)]
        assert len(log_calls) == 1
        
        # Verify filtering works with converted ACT->BS column
        buy_ops = result['buy_opportunities']
        sell_ops = result['sell_opportunities']
        hold_ops = result['hold_opportunities']
        
        # Count expected results based on ACT values
        expected_buys = sum(1 for act in market_with_act['ACT'] if act == 'B')
        expected_sells = sum(1 for act in market_with_act['ACT'] if act == 'S')
        expected_holds = sum(1 for act in market_with_act['ACT'] if act == 'H')
        
        assert len(buy_ops) == expected_buys
        assert len(sell_ops) == expected_sells
        assert len(hold_ops) == expected_holds
    
    @pytest.mark.asyncio
    async def test_confidence_score_with_real_data_patterns(self, market_data_with_prices):
        """Test confidence score calculation with realistic data patterns."""
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        # Analyze without existing BS column to trigger confidence calculation
        market_no_signals = market_data_with_prices.drop(columns=['BS'])
        
        with patch.object(engine, '_calculate_trading_signals') as mock_calc:
            # Mock the signal calculation to return data with confidence scores
            mock_result = market_no_signals.copy()
            mock_result['BS'] = ['B', 'B', 'H', 'B', 'S', 'H', 'B']
            mock_result['confidence_score'] = [0.9, 0.8, 0.7, 0.85, 0.75, 0.65, 0.7]
            mock_calc.return_value = mock_result
            
            result = await engine.analyze_market_opportunities(market_no_signals)
        
        # Verify signal calculation was called
        mock_calc.assert_called_once()
        
        # Verify confidence scores affect sell filtering
        sell_ops = result['sell_opportunities']
        # Only GOOGL has 'S' signal, and confidence of 0.75 > 0.6, so should be included
        assert len(sell_ops) == 1
        assert sell_ops.iloc[0]['confidence_score'] > 0.6
    
    @pytest.mark.asyncio
    async def test_empty_data_scenarios(self):
        """Test handling of empty or minimal data scenarios."""
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        # Test with empty market data
        empty_market = pd.DataFrame().reindex_like(pd.DataFrame({'symbol': []}).set_index('symbol'))
        
        with patch('trade_modules.trade_engine.validate_dataframe', return_value=True):
            result = await engine.analyze_market_opportunities(empty_market)
        
        # Should return empty opportunities
        assert all(len(df) == 0 for df in result.values())
        
        # Test with single row
        single_row = pd.DataFrame({
            'symbol': ['AAPL'],
            'price': [180.0],
            'BS': ['B']
        }).set_index('symbol')
        
        result = await engine.analyze_market_opportunities(single_row)
        
        # Should handle single row correctly
        assert len(result['buy_opportunities']) == 1
        assert 'AAPL' in result['buy_opportunities'].index
    
    @pytest.mark.asyncio 
    async def test_international_ticker_handling(self, market_data_with_prices):
        """Test handling of international ticker formats."""
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        # Add international tickers
        intl_data = pd.DataFrame({
            'symbol': ['0700.HK', 'ASML.AS', 'SAP.DE', 'NESN.SW'],
            'name': ['Tencent', 'ASML', 'SAP', 'Nestle'],
            'price': [530.45, 785.20, 198.85, 104.50],
            'BS': ['B', 'H', 'S', 'B'],
            'upside': [25.5, 8.2, -5.1, 12.8],
            'analyst_count': [22, 18, 25, 20]
        }).set_index('symbol')
        
        combined_data = pd.concat([market_data_with_prices, intl_data])
        
        # Portfolio with international holdings
        intl_portfolio = pd.DataFrame({
            'symbol': ['0700.HK', 'ASML.AS'],
            'TICKER': ['0700.HK', 'ASML.AS'],
            'quantity': [100, 25]
        })
        
        result = await engine.analyze_market_opportunities(
            combined_data,
            portfolio_df=intl_portfolio
        )
        
        # Verify international tickers are handled correctly
        buy_ops = result['buy_opportunities']
        hold_ops = result['hold_opportunities']
        sell_ops = result['sell_opportunities']
        
        # 0700.HK has 'B' signal but is in portfolio, should be excluded from buy
        assert '0700.HK' not in buy_ops.index
        
        # ASML.AS has 'H' signal and is in portfolio, should be in hold
        assert 'ASML.AS' in hold_ops.index
        
        # SAP.DE has 'S' signal and not in portfolio, should be in sell (if confidence allows)
        # NESN.SW has 'B' signal and not in portfolio, should be in buy
        assert 'NESN.SW' in buy_ops.index


class TestDataValidationIntegration:
    """Test data validation with real-world data scenarios."""
    
    @pytest.mark.asyncio
    async def test_missing_required_columns_handling(self):
        """Test graceful handling when required columns are missing."""
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        # Market data missing price column
        incomplete_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'name': ['Apple', 'Microsoft'],
            'BS': ['B', 'H']
        }).set_index('symbol')
        
        with patch('trade_modules.trade_engine.validate_dataframe', return_value=True):
            result = await engine.analyze_market_opportunities(incomplete_data)
        
        # Should complete without error
        assert isinstance(result, dict)
        assert all(key in result for key in ['buy_opportunities', 'sell_opportunities', 'hold_opportunities'])
    
    @pytest.mark.asyncio
    async def test_mixed_data_types_handling(self):
        """Test handling of mixed data types in columns."""
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        # Data with mixed types and missing values
        mixed_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
            'price': [180.5, 'N/A', 2850.0, None],
            'analyst_count': [35, 38.0, 'Unknown', 22],
            'upside': [10.6, np.nan, 8.8, '13.9%'],
            'BS': ['B', 'H', 'S', 'B']
        }).set_index('symbol')
        
        result = await engine.analyze_market_opportunities(mixed_data)
        
        # Should handle mixed types gracefully
        assert isinstance(result, dict)
        
        # Verify filtering still works despite data quality issues
        buy_ops = result['buy_opportunities']
        assert 'AAPL' in buy_ops.index
        assert 'TSLA' in buy_ops.index
    
    @pytest.mark.asyncio
    async def test_duplicate_ticker_handling(self):
        """Test handling of duplicate tickers in data."""
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        # Market data with duplicate tickers
        duplicate_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'AAPL', 'GOOGL'],  # AAPL appears twice
            'price': [180.0, 410.0, 182.0, 2850.0],
            'BS': ['B', 'H', 'S', 'B']
        }).set_index('symbol')
        
        # pandas will keep the last occurrence by default
        result = await engine.analyze_market_opportunities(duplicate_data)
        
        # Should handle duplicates without error
        assert isinstance(result, dict)
        
        # AAPL should appear with the last value ('S' signal)
        sell_ops = result['sell_opportunities']
        buy_ops = result['buy_opportunities']
        
        # AAPL with 'S' signal should be in sell opportunities (last occurrence)
        assert 'AAPL' in sell_ops.index or 'AAPL' not in buy_ops.index


class TestPerformanceIntegration:
    """Test performance characteristics with realistic data sizes."""
    
    @pytest.mark.asyncio
    async def test_large_dataset_performance(self):
        """Test performance with large dataset (realistic market size)."""
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        # Create large dataset (simulating ~1000 stocks)
        n_stocks = 1000
        large_data = pd.DataFrame({
            'symbol': [f'STOCK{i:04d}' for i in range(n_stocks)],
            'price': rng.uniform(10, 1000, n_stocks),
            'target_price': rng.uniform(15, 1200, n_stocks),
            'upside': rng.uniform(-20, 50, n_stocks),
            'analyst_count': rng.integers(5, 40, n_stocks),
            'total_ratings': rng.integers(5, 40, n_stocks),
            'BS': rng.choice(['B', 'H', 'S'], n_stocks)
        }).set_index('symbol')
        
        import time
        start_time = time.perf_counter()
        
        result = await engine.analyze_market_opportunities(large_data)
        
        end_time = time.perf_counter()
        
        # Should complete in reasonable time (< 2 seconds for 1000 stocks)
        processing_time = end_time - start_time
        assert processing_time < 2.0, f"Processing took {processing_time:.2f}s, expected < 2.0s"
        
        # Verify results are correct
        total_opportunities = sum(len(df) for df in result.values())
        assert total_opportunities == n_stocks  # All stocks should be categorized
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self):
        """Test async batch processing performance."""
        mock_provider = AsyncMock()
        engine = TradingEngine(provider=mock_provider)
        
        # Mock ticker data response
        mock_provider.get_ticker_info.return_value = {
            'price': 150.0,
            'market_cap': 2.5e12,
            'target_price': 165.0,
            'upside': 10.0
        }
        
        # Test batch processing with moderate size
        tickers = [f'TICKER{i:03d}' for i in range(100)]
        
        with patch('trade_modules.trade_engine.process_ticker_input', side_effect=lambda x: x), \
             patch('trade_modules.trade_engine.get_ticker_for_display', side_effect=lambda x: x):
            
            import time
            start_time = time.perf_counter()
            
            result = await engine.process_ticker_batch(tickers, batch_size=20)
            
            end_time = time.perf_counter()
        
        # Should complete batch processing efficiently
        processing_time = end_time - start_time
        assert processing_time < 5.0, f"Batch processing took {processing_time:.2f}s, expected < 5.0s"
        
        # Should return results for all tickers
        assert len(result) == len(tickers)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])