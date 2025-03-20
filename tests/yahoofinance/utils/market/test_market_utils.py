import unittest
from yahoofinance.utils.market_utils import is_us_ticker, normalize_hk_ticker

class TestMarketUtils(unittest.TestCase):
    """Test market utility functions for ticker validation and normalization."""
    
    def test_is_us_ticker(self):
        """Test US ticker detection."""
        # Standard US tickers without exchange suffix
        us_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB']
        for ticker in us_tickers:
            with self.subTest(ticker=ticker):
                self.assertTrue(is_us_ticker(ticker), f"Should identify {ticker} as US ticker")
        
        # US tickers with explicit US suffix
        us_suffix_tickers = ['AAPL.US', 'MSFT.US', 'AMZN.US']
        for ticker in us_suffix_tickers:
            with self.subTest(ticker=ticker):
                self.assertTrue(is_us_ticker(ticker), f"Should identify {ticker} as US ticker")
        
        # Special case US tickers with dot notation
        special_us_tickers = ['BRK.A', 'BRK.B', 'BF.A', 'BF.B']
        for ticker in special_us_tickers:
            with self.subTest(ticker=ticker):
                self.assertTrue(is_us_ticker(ticker), f"Should identify {ticker} as US ticker")
        
        # Non-US tickers with exchange suffix
        non_us_tickers = [
            '0700.HK',    # Hong Kong
            'BP.L',       # London
            'SAN.MC',     # Madrid
            'AIR.PA',     # Paris
            'BMW.DE',     # Germany
            'NESN.SW',    # Switzerland
            'NOVO-B.CO',  # Copenhagen
            'MAERSK-B.CO' # Copenhagen with dash
        ]
        for ticker in non_us_tickers:
            with self.subTest(ticker=ticker):
                self.assertFalse(is_us_ticker(ticker), f"Should identify {ticker} as non-US ticker")
        
        # Edge cases
        edge_cases = {
            'ABC.DEF.GHI': False,  # Multiple dots
            None: False,           # None input
            '': False,             # Empty string
            '123': False,          # Numeric string
            'ABC-DEF': False       # Dash but no exchange suffix
        }
        for ticker, expected in edge_cases.items():
            with self.subTest(ticker=ticker):
                try:
                    result = is_us_ticker(ticker)
                    self.assertEqual(result, expected, f"Unexpected result for {ticker}")
                except Exception as e:
                    if expected is not False:  # If we expected something other than False
                        self.fail(f"Unexpected exception for {ticker}: {str(e)}")
    
    def test_normalize_hk_ticker(self):
        """Test Hong Kong ticker normalization."""
        # Test cases for HK ticker normalization
        test_cases = [
            # Original, Expected
            ('0700.HK', '0700.HK'),     # Remove leading zero, pad to 4 digits
            ('00700.HK', '0700.HK'),    # Remove leading zeros, pad to 4 digits
            ('000700.HK', '0700.HK'),   # Remove leading zeros, pad to 4 digits
            ('0007000.HK', '7000.HK'),  # Remove leading zeros, already 4+ digits
            ('01234.HK', '1234.HK'),    # Remove leading zero, already 4 digits
            ('9988.HK', '9988.HK'),     # No leading zeros, already 4 digits
            ('09988.HK', '9988.HK'),    # Remove leading zero, already 4 digits
            ('009988.HK', '9988.HK'),   # Remove leading zeros, already 4 digits
            ('0700-A.HK', '0700-A.HK'), # Remove leading zero, pad to 4 digits
            ('12.HK', '0012.HK'),       # No leading zeros, pad to 4 digits
            ('012.HK', '0012.HK'),      # Remove leading zero, pad to 4 digits
            ('0.HK', '0000.HK'),        # Single zero becomes 0000
        ]
        
        for original, expected in test_cases:
            with self.subTest(original=original):
                self.assertEqual(normalize_hk_ticker(original), expected)
        
        # Test non-HK tickers (should return unchanged)
        non_hk_tickers = ['AAPL', 'MSFT.US', 'BP.L']
        for ticker in non_hk_tickers:
            with self.subTest(ticker=ticker):
                self.assertEqual(normalize_hk_ticker(ticker), ticker)
        
        # Edge cases
        edge_cases = {
            None: None,
            '': '',
            '123': '123',
            '0.HK': '0000.HK',        # Zero should become 0000
            '0000000000.HK': '0000.HK' # All zeros should become 0000
        }
        for ticker, expected in edge_cases.items():
            with self.subTest(ticker=ticker):
                try:
                    result = normalize_hk_ticker(ticker)
                    self.assertEqual(result, expected)
                except Exception as e:
                    self.fail(f"Unexpected exception for {ticker}: {str(e)}")

if __name__ == '__main__':
    unittest.main()