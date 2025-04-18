"""
Unit tests for the async circuit breaker pattern implementation.

This module contains tests for the AsyncCircuitBreaker class and related utilities,
verifying correct async execution, state transitions, and protection against
cascading failures in asynchronous code.
"""

import json

from yahoofinance.core.errors import YFinanceError, APIError, ValidationError, DataError
from yahoofinance.utils.error_handling import translate_error, enrich_error_context, with_retry, safe_operation
import os
import time
import asyncio
import pytest
from unittest.mock import patch, mock_open, MagicMock, call, AsyncMock

from yahoofinance.utils.network.circuit_breaker import (
    AsyncCircuitBreaker,
    CircuitState,
    CircuitOpenError,
    get_async_circuit_breaker,
    async_circuit_protected
)


@pytest.fixture
def circuit_breaker_config():
    return {
        "name": "test_async_circuit",
        "failure_threshold": 3,
        "failure_window": 10,
        "recovery_timeout": 5,
        "success_threshold": 2,
        "half_open_allow_percentage": 50,
        "max_open_timeout": 60,
        "enabled": True,
        "state_file": os.path.join(os.path.dirname(__file__), "test_async_circuit_breaker_state.json")
    }


@pytest.fixture
def async_circuit_breaker(circuit_breaker_config):
    """Create a test async circuit breaker with controlled configuration"""
    # Ensure the state file doesn't exist or is empty
    if os.path.exists(circuit_breaker_config["state_file"]):
        os.remove(circuit_breaker_config["state_file"])
    
    circuit = AsyncCircuitBreaker(**circuit_breaker_config)
    yield circuit
    
    # Clean up
    if os.path.exists(circuit_breaker_config["state_file"]):
        os.remove(circuit_breaker_config["state_file"])


@pytest.mark.asyncio
async def test_async_circuit_breaker_execute_async_success(async_circuit_breaker):
    """Test executing an async function with circuit breaker (success case)."""
    mock_func = AsyncMock(return_value="async_success")
    
    result = await async_circuit_breaker.execute_async(mock_func, "arg1", kwarg1="value1")
    
    assert result == "async_success"
    mock_func.assert_called_once_with("arg1", kwarg1="value1")
    assert async_circuit_breaker.total_successes == 1
    assert async_circuit_breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_async_circuit_breaker_execute_async_failure(async_circuit_breaker):
    """Test executing an async function with circuit breaker (failure case)."""
    mock_func = AsyncMock(side_effect=ValueError("async test error"))
    
    with pytest.raises(ValueError, match="async test error"):
        await async_circuit_breaker.execute_async(mock_func, "arg1", kwarg1="value1")
    
    mock_func.assert_called_once_with("arg1", kwarg1="value1")
    assert async_circuit_breaker.total_failures == 1
    assert async_circuit_breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_async_circuit_breaker_execute_async_open_circuit(async_circuit_breaker):
    """Test executing an async function with an open circuit."""
    # Trip the circuit
    for _ in range(async_circuit_breaker.failure_threshold):
        async_circuit_breaker.record_failure()
    
    assert async_circuit_breaker.state == CircuitState.OPEN
    
    # Try to execute when circuit is open
    mock_func = AsyncMock(return_value="async_success")
    
    with pytest.raises(CircuitOpenError) as exc_info:
        await async_circuit_breaker.execute_async(mock_func, "arg1", kwarg1="value1")
    
    assert "test_async_circuit is OPEN" in str(exc_info.value)
    assert exc_info.value.circuit_name == "test_async_circuit"
    assert exc_info.value.circuit_state == CircuitState.OPEN.value
    assert "metrics" in exc_info.value.__dict__
    mock_func.assert_not_called()


@pytest.mark.asyncio
async def test_async_circuit_breaker_trip_and_recovery(async_circuit_breaker):
    """Test async circuit breaker trip and recovery cycle."""
    # Mock async functions
    success_func = AsyncMock(return_value="success")
    failure_func = AsyncMock(side_effect=ValueError("error"))
    
    # Step 1: Trip the circuit with failures
    for _ in range(async_circuit_breaker.failure_threshold):
        with pytest.raises(ValueError):
            await async_circuit_breaker.execute_async(failure_func)
    
    assert async_circuit_breaker.state == CircuitState.OPEN
    
    # Step 2: Time passes, circuit goes to half-open
    with patch('time.time') as mock_time:
        mock_time.return_value = async_circuit_breaker.last_state_change + async_circuit_breaker.recovery_timeout + 1
        
        # Force circuit to check state (it happens on next execute_async call)
        async_circuit_breaker.get_state()
        
        assert async_circuit_breaker.state == CircuitState.HALF_OPEN
        
        # Try with deterministic random to allow first request
        with patch('random.randint', return_value=1):  # Ensure request is allowed
            # Step 3: Record successes to close circuit
            for _ in range(async_circuit_breaker.success_threshold):
                result = await async_circuit_breaker.execute_async(success_func)
                assert result == "success"
            
            assert async_circuit_breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_async_circuit_protected_decorator():
    """Test async_circuit_protected decorator."""
    # Create mock async function for the decorator
    mock_func = AsyncMock(return_value="async_success")
    decorated_func = async_circuit_protected("test_async_decorator")(mock_func)
    
    # Mock get_async_circuit_breaker to return a controlled circuit
    mock_circuit = MagicMock()
    mock_circuit.execute_async = AsyncMock(return_value="success_from_async_circuit")
    
    with patch("yahoofinance.utils.network.circuit_breaker.get_async_circuit_breaker", 
               return_value=mock_circuit):
        result = await decorated_func("arg1", kwarg1="value1")
    
    assert result == "success_from_async_circuit"
    mock_circuit.execute_async.assert_called_once()
    # The first argument to execute_async should be the function
    assert mock_circuit.execute_async.call_args[0][0] == mock_func


@pytest.mark.asyncio
async @with_retry 
def test_get_async_circuit_breaker():
    """Test get_async_circuit_breaker creates and returns async circuit breakers."""
    # Clear existing circuits
    with patch.dict("yahoofinance.utils.network.circuit_breaker._circuit_breakers", {}):
        circuit1 = get_async_circuit_breaker("test_async_get_1")
        circuit2 = get_async_circuit_breaker("test_async_get_2")
        circuit1_again = get_async_circuit_breaker("test_async_get_1")
        
        assert circuit1.name == "test_async_get_1"
        assert circuit2.name == "test_async_get_2"
        assert circuit1 is circuit1_again  # Should return the same instance
        assert isinstance(circuit1, AsyncCircuitBreaker)


@pytest.mark.asyncio
async def test_async_circuit_breaker_integration_with_enhanced_async():
    """Test integration of AsyncCircuitBreaker with enhanced async module."""
    # Create a simplified test that verifies the basic functionality

    # Create an AsyncCircuitBreaker instance
    circuit_config = {
        "name": "test_simplified_integration",
        "failure_threshold": 3,
        "failure_window": 10,
        "recovery_timeout": 5,
        "success_threshold": 2,
        "half_open_allow_percentage": 50,
        "max_open_timeout": 60,
        "enabled": True,
        "state_file": os.path.join(os.path.dirname(__file__), "test_simplified_circuit_state.json")
    }
    
    # Ensure clean state for test
    if os.path.exists(circuit_config["state_file"]):
        os.remove(circuit_config["state_file"])
    
    # Create circuit breaker
    circuit = AsyncCircuitBreaker(**circuit_config)
    
    try:
        # Create test functions
        success_func = AsyncMock(return_value="success")
        failure_func = AsyncMock(side_effect=ValueError("test error"))
        
        # Test successful call
        result = await circuit.execute_async(success_func)
        assert result == "success"
        assert circuit.total_successes == 1
        assert circuit.state == CircuitState.CLOSED
        
        # Trip the circuit
        for _ in range(circuit.failure_threshold):
            try:
                await circuit.execute_async(failure_func)
            except ValueError:
                pass  # Expected exception
        
        # Circuit should now be open
        assert circuit.state == CircuitState.OPEN
        assert circuit.total_failures >= circuit.failure_threshold
        
        # Test that circuit rejects requests when open
        with pytest.raises(CircuitOpenError):
            await circuit.execute_async(success_func)
        
        # Verify that success_func was not called
        assert success_func.call_count == 1  # Only called once before circuit opened
        
        # Test half-open state transition
        with patch('time.time') as mock_time:
            # Simulate time passing for recovery timeout
            mock_time.return_value = circuit.last_state_change + circuit.recovery_timeout + 1
            
            # Force state check
            circuit.get_state()
            
            # Should be in half-open state now
            assert circuit.state == CircuitState.HALF_OPEN
            
            # Allow first request in half-open state by mocking the random choice
            with patch('random.randint', return_value=1):
                result = await circuit.execute_async(success_func)
                assert result == "success"
                
                # Record enough successes to close circuit
                for _ in range(circuit.success_threshold - 1):
                    await circuit.execute_async(success_func)
                
                # Circuit should be closed now
                assert circuit.state == CircuitState.CLOSED
    finally:
        # Clean up
        if os.path.exists(circuit_config["state_file"]):
            os.re@with_retry(max_retries=3, retry_delay=1.0, backoff_factor=2.0)
def test_async_circuit_breaker_multiple_concurrent_requests(ync def test_async_circuit_breaker_multiple_concurrent_requests(async_circuit_breaker):
    """Test circuit breaker with multiple concurrent requests."""
    # Create test async functions
    async def success_task():
        await asyncio.sleep(0.01)
        return "success"
    
    async def fail_task():
        await asyncio.sleep(0.01)
        raise ValueError("failure")
    
    # Run multiple successful tasks concurrently
    tasks = [async_circuit_breaker.execute_async(success_task) for _ in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    
    assert all(result == "success" for result in results)
    assert async_circuit_breaker.total_successes == 5
    assert async_circuit_breaker.state == CircuitState.CLOSED
    
    # Reset for next test
    async_circuit_breaker.reset()
    
    # Trip the circuit with concurrent failures
    failure_threshold = async_circuit_breaker.failure_threshold
    
    # Need to handle exceptions in gather to prevent test failure
    tasks = [async_circuit_breaker.execute_async(fail_task) for _ in range(failure_threshold + 2)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # All tasks should have raised ValueError except possibly the last ones
    # that encountered an already-open circuit
    assert all(isinstance(r, (ValueError, CircuitOpenError)) for r in results)
    
    # Circuit should be open now
    assert async_circuit_breaker.state == CircuitState.OPEN
    
    # Verify some failures were recorded (at least up to threshold)
    assert async_circuit_breaker.total_failures >= failure_threshold