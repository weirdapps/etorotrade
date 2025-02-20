import pytest
from unittest.mock import patch, mock_open
import pandas as pd
from pathlib import Path
from yahoofinance.cons import (
    get_sp500_constituents,
    get_ftse100_constituents,
    get_cac40_constituents,
    get_dax_constituents,
    get_nikkei225_constituents,
    get_hangseng_constituents,
    save_constituents_to_csv,
    main
)

def test_get_sp500_constituents_success():
    """Test successful retrieval of S&P 500 constituents."""
    with patch('pandas.read_html') as mock_read_html:
        mock_read_html.return_value = [pd.DataFrame({'Symbol': ['AAPL', 'MSFT', 'GOOGL']})]
        result = get_sp500_constituents()
        assert isinstance(result, list)
        assert len(result) == 3
        assert 'AAPL' in result
        assert 'MSFT' in result
        assert 'GOOGL' in result

def test_get_sp500_constituents_failure():
    """Test handling of failure in S&P 500 constituents retrieval."""
    with patch('pandas.read_html', side_effect=Exception('Connection error')):
        result = get_sp500_constituents()
        assert isinstance(result, list)
        assert len(result) == 0

def test_get_ftse100_constituents():
    """Test retrieval of FTSE 100 constituents."""
    result = get_ftse100_constituents()
    assert isinstance(result, list)
    assert len(result) == 80  # Known number of FTSE 100 constituents
    assert 'HSBA.L' in result  # Check for a known constituent
    assert all(symbol.endswith('.L') for symbol in result)  # All symbols should end with .L

def test_get_cac40_constituents():
    """Test retrieval of CAC 40 constituents."""
    result = get_cac40_constituents()
    assert isinstance(result, list)
    assert len(result) == 39  # Known number of CAC 40 constituents
    assert 'AIR.PA' in result  # Check for a known constituent
    assert all(symbol.endswith('.PA') for symbol in result)  # All symbols should end with .PA

def test_get_dax_constituents():
    """Test retrieval of DAX constituents."""
    result = get_dax_constituents()
    assert isinstance(result, list)
    assert len(result) == 32  # Known number of DAX constituents
    assert 'SAP.DE' in result  # Check for a known constituent
    assert all(symbol.endswith('.DE') for symbol in result)  # All symbols should end with .DE

def test_get_nikkei225_constituents():
    """Test retrieval of Nikkei 225 constituents."""
    result = get_nikkei225_constituents()
    assert isinstance(result, list)
    assert len(result) == 215  # Actual number of Nikkei 225 constituents in the list
    assert '7203.T' in result  # Check for Toyota's symbol
    assert all(symbol.endswith('.T') for symbol in result)  # All symbols should end with .T

def test_get_hangseng_constituents():
    """Test retrieval of Hang Seng constituents."""
    result = get_hangseng_constituents()
    assert isinstance(result, list)
    assert len(result) == 67  # Known number of Hang Seng constituents
    assert '0700.HK' in result  # Check for Tencent's symbol
    assert all(symbol.endswith('.HK') for symbol in result)  # All symbols should end with .HK

def test_save_constituents_to_csv(tmp_path):
    """Test saving constituents to CSV file."""
    # Create a temporary directory for testing
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AAPL']  # Include duplicate to test deduplication
    
    # Create input directory
    input_dir = tmp_path / 'input'
    input_dir.mkdir(exist_ok=True)
    
    with patch('yahoofinance.cons.Path') as mock_path:
        mock_path.return_value.parent = tmp_path
        save_constituents_to_csv(test_symbols)
        
        # Check if file was created
        csv_path = input_dir / 'cons.csv'
        assert csv_path.exists()
        
        # Read and verify contents
        df = pd.read_csv(csv_path)
        assert len(df) == 3  # Should be 3 unique symbols
        assert list(df['Symbol']) == ['AAPL', 'GOOGL', 'MSFT']  # Should be sorted

def test_main(tmp_path):
    """Test main function."""
    # Mock all constituent getters
    # Create input directory
    input_dir = tmp_path / 'input'
    input_dir.mkdir(exist_ok=True)
    
    with patch('yahoofinance.cons.get_sp500_constituents', return_value=['AAPL', 'MSFT']), \
         patch('yahoofinance.cons.get_ftse100_constituents', return_value=['HSBA.L']), \
         patch('yahoofinance.cons.get_cac40_constituents', return_value=['AIR.PA']), \
         patch('yahoofinance.cons.get_dax_constituents', return_value=['SAP.DE']), \
         patch('yahoofinance.cons.get_nikkei225_constituents', return_value=['7203.T']), \
         patch('yahoofinance.cons.get_hangseng_constituents', return_value=['0700.HK']), \
         patch('yahoofinance.cons.Path') as mock_path:
        
        mock_path.return_value.parent = tmp_path
        main()
        
        # Verify CSV was created with all constituents
        csv_path = input_dir / 'cons.csv'
        assert csv_path.exists()
        
        df = pd.read_csv(csv_path)
        assert len(df) == 7  # One from each index plus one extra from S&P 500
        assert set(df['Symbol']) == {'AAPL', 'MSFT', 'HSBA.L', 'AIR.PA', 'SAP.DE', '7203.T', '0700.HK'}