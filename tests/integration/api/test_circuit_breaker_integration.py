"""
Integration tests for the circuit breaker pattern implementation.

This module tests the integration of circuit breakers across various components:
1. Synchronous circuit breaker with API providers
2. Asynchronous circuit breaker with enhanced async providers
3. State persistence and recovery across sessions
4. Circuit breaker with enhanced async rate limiting
"""

import pytest
import json
import os
import time
import asyncio
import aiohttp
import tempfile
from unittest.mock import patch, AsyncMock, MagicMock, mock_open

from yahoofinance_v2.utils.network.circuit_breaker import (
    CircuitBreaker, AsyncCircuitBreaker, CircuitState, CircuitOpenError,
    get_circuit_breaker, get_async_circuit_breaker, reset_all_circuits,
    circuit_protected, async_circuit_protected
)
from yahoofinance_v2.utils.async_utils.enhanced import (
    AsyncRateLimiter, enhanced_async_rate_limited, retry_async_with_backoff
)
from yahoofinance_v2.api.providers.enhanced_async_yahoo_finance import EnhancedAsyncYahooFinanceProvider
from yahoofinance_v2.core.errors import APIError, ValidationError, RateLimitError, NetworkError


@pytest.fixture
def temp_state_file():
    """Create a temporary file for circuit breaker state"""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def circuit_breaker(temp_state_file):
    """Create a test circuit breaker with temporary state file"""
    return CircuitBreaker(
        name="test_integration",
        failure_threshold=3,
        failure_window=10,
        recovery_timeout=5,
        success_threshold=2,
        half_open_allow_percentage=50,
        max_open_timeout=60,
        enabled=True,
        state_file=temp_state_file
    )


@pytest.fixture
def async_circuit_breaker(temp_state_file):
    """Create a test async circuit breaker with temporary state file"""
    return AsyncCircuitBreaker(
        name="test_async_integration",
        failure_threshold=3,
        failure_window=10,
        recovery_timeout=5,
        success_threshold=2,
        half_open_allow_percentage=50,
        max_open_timeout=60,
        enabled=True,
        state_file=temp_state_file
    )


@pytest.fixture
async def enhanced_provider(temp_state_file):
    """Create an enhanced provider with circuit breaker using temp state file"""
    # Patch the config to use our temp state file
    with patch("yahoofinance_v2.core.config.CIRCUIT_BREAKER", {
        "FAILURE_THRESHOLD": 3,
        "FAILURE_WINDOW": 10,
        "RECOVERY_TIMEOUT": 5,
        "SUCCESS_THRESHOLD": 2,
        "HALF_OPEN_ALLOW_PERCENTAGE": 50,
        "MAX_OPEN_TIMEOUT": 60,
        "ENABLED": True,
        "STATE_FILE": temp_state_file
    }):
        provider = EnhancedAsyncYahooFinanceProvider(
            max_retries=1,
            retry_delay=0.01,
            max_concurrency=2,
            enable_circuit_breaker=True
        )
        yield provider
        await provider.close()


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker pattern"""
    
    def test_state_persistence_across_instances(self, temp_state_file):
        """Test that circuit state persists across different instances"""
        # Create initial circuit and trip it
        circuit1 = CircuitBreaker(
            name="persistence_test",
            failure_threshold=3,
            state_file=temp_state_file
        )
        
        # Initially circuit should be closed
        assert circuit1.state == CircuitState.CLOSED
        
        # Trip the circuit
        for _ in range(circuit1.failure_threshold):
            circuit1.record_failure()
        
        assert circuit1.state == CircuitState.OPEN
        
        # Create second circuit with the same state file
        circuit2 = CircuitBreaker(
            name="persistence_test",
            failure_threshold=3,
            state_file=temp_state_file
        )
        
        # Second circuit should load the OPEN state
        assert circuit2.state == CircuitState.OPEN
        
        # Reset the circuit
        circuit2.reset()
        assert circuit2.state == CircuitState.CLOSED
        
        # Create third circuit, it should see the CLOSED state
        circuit3 = CircuitBreaker(
            name="persistence_test",
            failure_threshold=3,
            state_file=temp_state_file
        )
        
        assert circuit3.state == CircuitState.CLOSED
    
    def test_multiple_circuits_in_same_file(self, temp_state_file):
        """Test multiple circuits using the same state file"""
        # Create two circuits with different names but same state file
        circuit_a = CircuitBreaker(
            name="circuit_a",
            failure_threshold=3,
            state_file=temp_state_file
        )
        
        circuit_b = CircuitBreaker(
            name="circuit_b", 
            failure_threshold=3,
            state_file=temp_state_file
        )
        
        # Trip circuit A
        for _ in range(circuit_a.failure_threshold):
            circuit_a.record_failure()
        
        assert circuit_a.state == CircuitState.OPEN
        assert circuit_b.state == CircuitState.CLOSED  # B should still be closed
        
        # Create new instances
        circuit_a2 = CircuitBreaker(
            name="circuit_a",
            failure_threshold=3,
            state_file=temp_state_file
        )
        
        circuit_b2 = CircuitBreaker(
            name="circuit_b",
            failure_threshold=3,
            state_file=temp_state_file
        )
        
        # Verify states loaded correctly
        assert circuit_a2.state == CircuitState.OPEN
        assert circuit_b2.state == CircuitState.CLOSED
    
    def test_circuit_protected_decorator(self, circuit_breaker):
        """Test the circuit_protected decorator integration"""
        call_count = 0
        
        # Function to protect with circuit breaker
        @circuit_protected("test_integration")
        def test_function(success=True):
            nonlocal call_count
            call_count += 1
            if not success:
                raise ValueError("Test failure")
            return "success"
        
        # Replace get_circuit_breaker to return our test circuit
        with patch("yahoofinance_v2.utils.network.circuit_breaker.get_circuit_breaker", 
                   return_value=circuit_breaker):
            
            # Test successful execution
            result = test_function(success=True)
            assert result == "success"
            assert call_count == 1
            assert circuit_breaker.state == CircuitState.CLOSED
            
            # Trip the circuit with failed calls
            with pytest.raises(ValueError):
                test_function(success=False)
            
            with pytest.raises(ValueError):
                test_function(success=False)
            
            with pytest.raises(ValueError):
                test_function(success=False)
            
            assert call_count == 4  # 1 success + 3 failures
            assert circuit_breaker.state == CircuitState.OPEN
            
            # Next call should be blocked by circuit breaker
            with pytest.raises(CircuitOpenError):
                test_function(success=True)
            
            # Call count shouldn't increase as function wasn't executed
            assert call_count == 4
    
    @pytest.mark.asyncio
    async def test_async_circuit_protected_decorator(self, async_circuit_breaker):
        """Test the async_circuit_protected decorator integration"""
        call_count = 0
        
        # Async function to protect with circuit breaker
        @async_circuit_protected("test_async_integration")
        async def test_async_function(success=True):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Small delay
            if not success:
                raise ValueError("Test async failure")
            return "async success"
        
        # Replace get_async_circuit_breaker to return our test circuit
        with patch("yahoofinance_v2.utils.network.circuit_breaker.get_async_circuit_breaker", 
                   return_value=async_circuit_breaker):
            
            # Test successful execution
            result = await test_async_function(success=True)
            assert result == "async success"
            assert call_count == 1
            assert async_circuit_breaker.state == CircuitState.CLOSED
            
            # Trip the circuit with failed calls
            with pytest.raises(ValueError):
                await test_async_function(success=False)
            
            with pytest.raises(ValueError):
                await test_async_function(success=False)
            
            with pytest.raises(ValueError):
                await test_async_function(success=False)
            
            assert call_count == 4  # 1 success + 3 failures
            assert async_circuit_breaker.state == CircuitState.OPEN
            
            # Next call should be blocked by circuit breaker
            with pytest.raises(CircuitOpenError):
                await test_async_function(success=True)
            
            # Call count shouldn't increase as function wasn't executed
            assert call_count == 4
    
    @pytest.mark.asyncio
    async def test_enhanced_async_rate_limited_with_circuit_breaker(self, async_circuit_breaker):
        """Test integration of enhanced_async_rate_limited with circuit breaker"""
        rate_limiter = AsyncRateLimiter(
            window_size=1,
            max_calls=10,
            base_delay=0.01
        )
        call_count = 0
        
        # Create function with the decorator
        @enhanced_async_rate_limited(
            circuit_name="test_async_integration",
            max_retries=1,
            rate_limiter=rate_limiter
        )
        async def test_function(success=True):
            nonlocal call_count
            call_count += 1
            if not success:
                raise ValueError("Test failure in enhanced function")
            return "enhanced success"
        
        # Mock the circuit breaker access
        with patch("yahoofinance_v2.utils.network.circuit_breaker.get_async_circuit_breaker", 
                   return_value=async_circuit_breaker):
            
            # Test successful execution
            result = await test_function(success=True)
            assert result == "enhanced success"
            assert call_count == 1
            
            # Trip the circuit with failures
            for _ in range(async_circuit_breaker.failure_threshold):
                with pytest.raises(ValueError):
                    await test_function(success=False)
            
            # Circuit should be open now
            assert async_circuit_breaker.state == CircuitState.OPEN
            
            # Next call should fail with translated CircuitOpenError
            with pytest.raises(CircuitOpenError):
                await test_function(success=True)
    
    @pytest.mark.asyncio
    async def test_retry_async_with_backoff_and_circuit_breaker(self, async_circuit_breaker):
        """Test integration of retry_async_with_backoff with circuit breaker"""
        call_count = 0
        
        # Create test async function
        async def test_function(success=True):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            if not success:
                raise ValueError("Test failure in retry function")
            return "retry success"
        
        # Mock sleep to avoid actual delays
        with patch("asyncio.sleep", AsyncMock()), \
             patch("yahoofinance_v2.utils.network.circuit_breaker.get_async_circuit_breaker", 
                   return_value=async_circuit_breaker):
            
            # Test successful execution with retries
            result = await retry_async_with_backoff(
                test_function,
                max_retries=2,
                base_delay=0.01,
                circuit_name="test_async_integration"
            )
            assert result == "retry success"
            assert call_count == 1
            
            # Reset counter
            call_count = 0
            
            # Test with a failing function - should retry
            with pytest.raises(ValueError):
                await retry_async_with_backoff(
                    test_function,
                    success=False,
                    max_retries=2,
                    base_delay=0.01,
                    circuit_name="test_async_integration"
                )
            
            # Should have attempted 1 + 2 retries = 3 times
            assert call_count == 3
            
            # Trip the circuit by calling directly
            for _ in range(async_circuit_breaker.failure_threshold):
                async_circuit_breaker.record_failure()
            
            # Reset counter
            call_count = 0
            
            # Next retry should immediately fail with CircuitOpenError (no retries)
            with pytest.raises(CircuitOpenError):
                await retry_async_with_backoff(
                    test_function,
                    max_retries=2,
                    base_delay=0.01,
                    circuit_name="test_async_integration"
                )
            
            # Function should not have been called at all
            assert call_count == 0
    
    @pytest.mark.asyncio
    async def test_enhanced_provider_with_circuit_breaker(self, enhanced_provider, temp_state_file):
        """Test EnhancedAsyncYahooFinanceProvider with circuit breaker integration"""
        provider = enhanced_provider
        
        # Mock successful fetch
        mock_success_response = {
            "quoteSummary": {
                "result": [{
                    "price": {
                        "regularMarketPrice": {"raw": 150.25},
                        "longName": "Test Company",
                        "shortName": "TEST",
                        "marketCap": {"raw": 2000000000},
                        "currency": "USD"
                    }
                }]
            }
        }
        
        # Create a new circuit breaker that uses our temp file
        circuit = get_async_circuit_breaker("yahoofinance_api")
        
        # Reset the circuit to ensure clean state
        circuit.reset()
        
        # Mock the fetch_json method to first succeed then fail
        fetch_mock = AsyncMock()
        fetch_mock.side_effect = [
            mock_success_response,  # First call succeeds
            APIError("Test API error 1"),  # Next calls fail
            APIError("Test API error 2"),
            APIError("Test API error 3"),
        ]
        
        with patch.object(provider, '_fetch_json', fetch_mock), \
             patch.object(provider, 'get_insider_transactions', AsyncMock(return_value=[])):
            
            # First call should succeed
            result = await provider.get_ticker_info("AAPL")
            assert result["symbol"] == "AAPL"
            assert result["current_price"] == 150.25
            
            # Next calls should fail and trip the circuit
            for _ in range(3):
                with pytest.raises(APIError):
                    await provider.get_ticker_info("AAPL")
            
            # Circuit should be open now
            assert circuit.state == CircuitState.OPEN
            
            # Next call should fail with a translated CircuitOpenError
            with pytest.raises(APIError) as excinfo:
                await provider.get_ticker_info("AAPL")
            
            # Verify error translation
            assert "currently unavailable" in str(excinfo.value)
            assert excinfo.value.status_code == 503
            assert excinfo.value.retry_after > 0
            
            # Function should only have been called 4 times (1 success + 3 failures)
            assert fetch_mock.call_count == 4
            
            # Create a new provider instance
            new_provider = EnhancedAsyncYahooFinanceProvider(enable_circuit_breaker=True)
            
            # The circuit should still be open for the new provider
            new_circuit = get_async_circuit_breaker("yahoofinance_api")
            assert new_circuit.state == CircuitState.OPEN
            
            # Clean up
            await new_provider.close()
    
    @pytest.mark.asyncio
    async def test_circuit_recovery_after_timeout(self, enhanced_provider):
        """Test circuit recovery after timeout period"""
        provider = enhanced_provider
        
        # Mock successful fetch
        mock_success_response = {
            "quoteSummary": {
                "result": [{
                    "price": {
                        "regularMarketPrice": {"raw": 150.25},
                        "longName": "Test Company",
                        "shortName": "TEST",
                        "marketCap": {"raw": 2000000000},
                        "currency": "USD"
                    }
                }]
            }
        }
        
        # Get the circuit breaker
        circuit = get_async_circuit_breaker("yahoofinance_api")
        
        # Reset the circuit to ensure clean state
        circuit.reset()
        
        # Trip the circuit
        for _ in range(circuit.failure_threshold):
            circuit.record_failure()
        
        assert circuit.state == CircuitState.OPEN
        
        # Mock time to simulate recovery timeout passed
        with patch('time.time') as mock_time:
            # Set time to be after recovery timeout
            mock_time.return_value = circuit.last_state_change + circuit.recovery_timeout + 1
            
            # Mock successful fetch
            with patch.object(provider, '_fetch_json', AsyncMock(return_value=mock_success_response)), \
                 patch.object(provider, 'get_insider_transactions', AsyncMock(return_value=[])):
                
                # Request should be allowed through in HALF_OPEN state
                result = await provider.get_ticker_info("AAPL")
                
                # Verify result
                assert result["symbol"] == "AAPL"
                assert result["current_price"] == 150.25
                
                # Circuit should be in HALF_OPEN state
                assert circuit.state == CircuitState.HALF_OPEN
                
                # Success count should be incremented
                assert circuit.success_count == 1
                
                # Another successful call
                result = await provider.get_ticker_info("MSFT")
                
                # After success_threshold successes, circuit should close
                assert circuit.state == CircuitState.CLOSED