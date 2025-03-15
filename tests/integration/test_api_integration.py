"""
Integration tests for API client and rate limiting.

These tests verify that the API client and rate limiting components
work together correctly in realistic scenarios.
"""

import pytest
import time
from unittest.mock import patch, Mock

from yahoofinance.client import YFinanceClient
from yahoofinance.utils.rate_limiter import global_rate_limiter, AdaptiveRateLimiter
from yahoofinance.errors import RateLimitError, APIError


@pytest.mark.integration
@pytest.mark.network
def test_client_with_rate_limiting():
    """Test that client uses rate limiting correctly."""
    # Create client with custom rate limiter for testing
    test_limiter = AdaptiveRateLimiter(window_size=5, max_calls=10)
    
    with patch('yahoofinance.client.global_rate_limiter', test_limiter):
        with patch('yahoofinance.client.yf.Ticker') as mock_ticker:
            # Configure mock ticker
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = {'regularMarketPrice': 150.0}
            mock_ticker.return_value = mock_ticker_instance
            
            client = YFinanceClient()
            
            # Make multiple API calls
            for _ in range(5):
                client.get_ticker_info('AAPL')
            
            # Verify calls were tracked by rate limiter
            assert len(test_limiter.calls) == 5


@pytest.mark.integration
@pytest.mark.network
def test_rate_limited_retries():
    """Test retry behavior with rate limiting."""
    # Create client with custom rate limiter for testing
    test_limiter = AdaptiveRateLimiter(window_size=5, max_calls=10)
    
    with patch('yahoofinance.client.global_rate_limiter', test_limiter):
        with patch('yahoofinance.client.yf.Ticker') as mock_ticker:
            # Configure mock ticker to fail twice then succeed
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = {'regularMarketPrice': 150.0}
            
            call_count = 0
            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise RateLimitError("Too many requests")
                return mock_ticker_instance
            
            mock_ticker.side_effect = side_effect
            
            # Create client and make API call
            client = YFinanceClient()
            
            # Patch sleep to make test run faster
            with patch('time.sleep'):
                result = client.get_ticker_info('AAPL')
                
                # Verify result after retries
                assert result is not None
                assert call_count == 3  # Original + 2 retries


@pytest.mark.integration
@pytest.mark.network
def test_batch_processing_with_rate_limiting():
    """Test batch processing with rate limiting."""
    # Create client with custom rate limiter for testing
    test_limiter = AdaptiveRateLimiter(window_size=5, max_calls=10)
    
    with patch('yahoofinance.client.global_rate_limiter', test_limiter):
        with patch('yahoofinance.client.yf.Ticker') as mock_ticker:
            # Configure mock ticker
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = {'regularMarketPrice': 150.0}
            mock_ticker.return_value = mock_ticker_instance
            
            client = YFinanceClient()
            
            # Create batch of tickers
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
            
            # Track sleep calls
            sleep_calls = []
            
            def record_sleep(seconds):
                sleep_calls.append(seconds)
            
            # Mock sleep to make test run faster and record calls
            with patch('time.sleep', side_effect=record_sleep):
                # Process batch
                results = []
                for ticker in tickers:
                    results.append(client.get_ticker_info(ticker))
                
                # Verify results
                assert len(results) == 4
                assert all(r is not None for r in results)
                
                # Verify rate limiting was applied
                assert len(sleep_calls) > 0


@pytest.mark.integration
@pytest.mark.network
def test_error_recovery_integration():
    """Test error recovery with rate limiting."""
    # Create client with custom rate limiter for testing
    test_limiter = AdaptiveRateLimiter(window_size=5, max_calls=10)
    
    with patch('yahoofinance.client.global_rate_limiter', test_limiter):
        with patch('yahoofinance.client.yf.Ticker') as mock_ticker:
            # Configure mock ticker to simulate various errors
            call_count = 0
            
            def side_effect(ticker):
                nonlocal call_count
                call_count += 1
                
                if ticker == 'ERROR':
                    raise APIError("API Error")
                elif ticker == 'RATE_LIMIT':
                    raise RateLimitError("Too many requests")
                else:
                    mock_instance = Mock()
                    mock_instance.info = {'regularMarketPrice': 150.0}
                    return mock_instance
            
            mock_ticker.side_effect = side_effect
            
            client = YFinanceClient()
            
            # Mock sleep to make test run faster
            with patch('time.sleep'):
                # Test successful request
                result1 = client.get_ticker_info('AAPL')
                assert result1 is not None
                
                # Test API error (should be retried a few times then fail)
                with pytest.raises(APIError):
                    client.get_ticker_info('ERROR')
                
                # Test rate limit error
                with pytest.raises(RateLimitError):
                    # Mock max_retries to 1 for faster test
                    with patch.object(client, 'max_retries', 1):
                        client.get_ticker_info('RATE_LIMIT')
                
                # Verify rate limiter tracked errors
                assert len(test_limiter.errors) > 0
                assert 'RATE_LIMIT' in test_limiter.error_counts
                
                # Verify higher delay for problematic ticker
                normal_delay = test_limiter.get_delay()
                error_delay = test_limiter.get_delay('RATE_LIMIT')
                assert error_delay > normal_delay