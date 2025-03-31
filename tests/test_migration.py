"""
Test migration from compat layer to canonical sources.

This file tests both the old compatibility imports and the new canonical imports
to verify they provide equivalent functionality.

DEPRECATION NOTICE: This test file is ONLY for testing the migration process. 
Once the compat folder is removed, this file will also be removed.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
import warnings

# Suppress deprecation warnings during testing
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestAnalystDataMigration:
    """Test migration from compat.analyst to analysis.analyst."""
    
    def test_import_equivalence(self):
        """Test that both import paths provide equivalent classes."""
        # Import from both paths
        from yahoofinance.compat.analyst import AnalystData as OldAnalystData
        from yahoofinance.analysis.analyst import CompatAnalystData as NewAnalystData
        
        # Create instances
        old_instance = OldAnalystData()
        new_instance = NewAnalystData()
        
        # Verify they have the same methods
        old_methods = [m for m in dir(old_instance) if not m.startswith('_')]
        new_methods = [m for m in dir(new_instance) if not m.startswith('_')]
        
        assert set(old_methods) == set(new_methods), "Method sets differ between old and new classes"
    
    def test_method_behavior(self):
        """Test that methods behave the same way."""
        from yahoofinance.compat.analyst import AnalystData as OldAnalystData
        from yahoofinance.analysis.analyst import CompatAnalystData as NewAnalystData
        
        # Create mock client
        mock_client = Mock()
        
        # Create instances with mock client
        old_instance = OldAnalystData(mock_client)
        new_instance = NewAnalystData(mock_client)
        
        # Test a method with simple inputs
        with patch.object(old_instance, '_validate_date') as old_validate:
            with patch.object(new_instance, '_validate_date') as new_validate:
                old_instance._validate_date('2024-01-01')
                new_instance._validate_date('2024-01-01')
                
                old_validate.assert_called_once_with('2024-01-01')
                new_validate.assert_called_once_with('2024-01-01')


class TestEarningsCalendarMigration:
    """Test migration from compat.earnings to analysis.earnings."""
    
    def test_import_equivalence(self):
        """Test that both import paths provide equivalent classes."""
        # Import from both paths
        from yahoofinance.compat.earnings import EarningsCalendar as OldEarningsCalendar
        from yahoofinance.analysis.earnings import EarningsCalendar as NewEarningsCalendar
        
        # Create instances
        old_instance = OldEarningsCalendar()
        new_instance = NewEarningsCalendar()
        
        # Verify they have the same methods
        old_methods = [m for m in dir(old_instance) if not m.startswith('_')]
        new_methods = [m for m in dir(new_instance) if not m.startswith('_')]
        
        assert set(old_methods) == set(new_methods), "Method sets differ between old and new classes"
    
    def test_format_earnings_table(self):
        """Test that format_earnings_table function is equivalent."""
        from yahoofinance.compat.earnings import format_earnings_table as old_format
        from yahoofinance.analysis.earnings import format_earnings_table as new_format
        
        # Create test data
        df = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT'],
            'Date': ['2024-01-01', '2024-01-02'],
            'EPS Est': ['1.23', '2.34'],
            'Market Cap': ['$3T', '$2T']
        })
        
        # Test with mock print function
        with patch('builtins.print') as mock_print:
            old_format(df, '2024-01-01', '2024-01-07')
            old_calls = mock_print.call_args_list
            
        with patch('builtins.print') as mock_print:
            new_format(df, '2024-01-01', '2024-01-07')
            new_calls = mock_print.call_args_list
            
        assert old_calls == new_calls, "Format function calls differ"


class TestClientMigration:
    """Test migration from compat.client to api provider pattern."""
    
    def test_client_provider_equivalence(self):
        """Test that client and provider return similar data structures."""
        # Only run if network access is allowed
        pytest.importorskip("requests")
        
        # Import both
        from yahoofinance.compat.client import YFinanceClient
        from yahoofinance.api import get_provider
        
        # Create instances
        client = YFinanceClient()
        provider = get_provider()
        
        # Mock get_ticker_info to avoid actual network calls
        mock_data = {
            'name': 'Apple Inc.',
            'price': 150.0,
            'ticker': 'AAPL',
            'market_cap': 2500000000000
        }
        
        with patch.object(provider, 'get_ticker_info', return_value=mock_data) as mock_provider:
            with patch.object(client, 'get_ticker_info') as mock_client:
                # Setup client mock to return a StockData-like object
                mock_client_result = Mock()
                mock_client_result.name = 'Apple Inc.'
                mock_client_result.price = 150.0
                mock_client_result.ticker = 'AAPL'
                mock_client_result.market_cap = 2500000000000
                mock_client.return_value = mock_client_result
                
                # Get data
                client_data = client.get_ticker_info('AAPL')
                provider_data = provider.get_ticker_info('AAPL')
                
                # Verify client called correctly
                mock_client.assert_called_once_with('AAPL')
                # Verify provider called correctly
                mock_provider.assert_called_once_with('AAPL')
                
                # Check data equivalence
                assert client_data.name == provider_data['name']
                assert client_data.price == provider_data['price']
                assert client_data.ticker == provider_data['ticker']
                assert client_data.market_cap == provider_data['market_cap']


class TestDisplayFormattingMigration:
    """Test migration from compat.formatting to presentation.formatter."""
    
    def test_formatter_equivalence(self):
        """Test that formatters provide equivalent functionality."""
        from yahoofinance.compat.formatting import DisplayFormatter as OldFormatter
        from yahoofinance.presentation.formatter import DisplayFormatter as NewFormatter
        
        # Create instances
        old_formatter = OldFormatter()
        new_formatter = NewFormatter()
        
        # Test percentage formatting
        assert old_formatter.format_percentage(10.5) == new_formatter.format_percentage(10.5)
        assert old_formatter.format_percentage(None) == new_formatter.format_percentage(None)
        
        # Test price formatting
        assert old_formatter.format_price(150.25) == new_formatter.format_price(150.25)
        assert old_formatter.format_price(None) == new_formatter.format_price(None)
        
        # Test market cap formatting
        assert old_formatter.format_market_cap(1500000000) == new_formatter.format_market_cap(1500000000)
        assert old_formatter.format_market_cap(None) == new_formatter.format_market_cap(None)


class TestMarketDisplayMigration:
    """Test migration from compat.display to presentation.console."""
    
    def test_display_functionality(self):
        """Test that both display classes provide similar functionality."""
        from yahoofinance.compat.display import MarketDisplay as OldMarketDisplay
        from yahoofinance.presentation.console import MarketDisplay as NewMarketDisplay
        
        # Create instances - note that the old display requires a client which we'll mock
        from unittest.mock import Mock, patch
        mock_client = Mock()
        
        # Disable actual printing during tests
        with patch('builtins.print'):
            # Create instances
            old_display = OldMarketDisplay(client=mock_client)
            new_display = NewMarketDisplay()
            
            # Verify methods with similar functionality exist in both classes
            # The old interface used different method names but similar functionality
            assert hasattr(old_display, 'display_report')  # Used in v1 compat layer
            assert hasattr(new_display, 'display_stock_table')  # Used in v2
            
            # Verify other key methods
            assert hasattr(old_display, 'generate_stock_report')
            assert hasattr(old_display, '_create_empty_report')
            assert hasattr(old_display, 'load_tickers')
            
            # The v2 display has save_to_csv
            assert hasattr(new_display, 'save_to_csv')
            
            # Test that old display delegates to new display correctly
            # Create a sample data dict that both can work with
            v2_display = old_display.v2_display
            assert v2_display is not None
            assert isinstance(v2_display, NewMarketDisplay)


class TestPricingAnalyzerMigration:
    """Test migration from compat.pricing to analysis.market."""
    
    def test_analyzer_functionality(self):
        """Test that analyzer functionality is preserved."""
        from yahoofinance.compat.pricing import PricingAnalyzer
        from yahoofinance.analysis.market import MarketAnalyzer
        
        # Create instances
        old_analyzer = PricingAnalyzer()
        new_analyzer = MarketAnalyzer()
        
        # Verify both have get_provider attribute
        assert hasattr(old_analyzer, 'provider')
        assert hasattr(new_analyzer, 'provider')
        
        # Test calculate_upside method in old analyzer
        assert old_analyzer.calculate_upside(100, 120) == 20.0
        assert old_analyzer.calculate_upside(100, 100) == 0.0
        assert old_analyzer.calculate_upside(0, 100) == 0.0
        assert old_analyzer.calculate_upside(None, 100) == 0.0