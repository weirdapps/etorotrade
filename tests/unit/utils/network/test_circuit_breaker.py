"""
Unit tests for the circuit breaker pattern implementation.

This module contains tests for the CircuitBreaker class and related utilities,
verifying correct state transitions, failure counting, and protection against
cascading failures.
"""

import json
import os
import time
import pytest
from unittest.mock import patch, mock_open, MagicMock, call

from yahoofinance_v2.utils.network.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitOpenError,
    get_circuit_breaker,
    reset_all_circuits,
    circuit_protected
)


@pytest.fixture
def circuit_breaker_config():
    return {
        "name": "test_circuit",
        "failure_threshold": 3,
        "failure_window": 10,
        "recovery_timeout": 5,
        "success_threshold": 2,
        "half_open_allow_percentage": 50,
        "max_open_timeout": 60,
        "enabled": True,
        "state_file": "/tmp/test_circuit_breaker_state.json"
    }


@pytest.fixture
def circuit_breaker(circuit_breaker_config):
    """Create a test circuit breaker with controlled configuration"""
    # Ensure the state file doesn't exist or is empty
    if os.path.exists(circuit_breaker_config["state_file"]):
        os.remove(circuit_breaker_config["state_file"])
    
    circuit = CircuitBreaker(**circuit_breaker_config)
    yield circuit
    
    # Clean up
    if os.path.exists(circuit_breaker_config["state_file"]):
        os.remove(circuit_breaker_config["state_file"])


def test_circuit_breaker_initial_state(circuit_breaker):
    """Test that circuit breaker initializes in CLOSED state."""
    assert circuit_breaker.state == CircuitState.CLOSED
    assert circuit_breaker.failure_count == 0
    assert circuit_breaker.success_count == 0
    assert circuit_breaker.total_failures == 0
    assert circuit_breaker.total_successes == 0
    assert circuit_breaker.total_requests == 0


def test_circuit_breaker_record_failure(circuit_breaker):
    """Test recording failures increases failure count."""
    circuit_breaker.record_failure()
    assert circuit_breaker.failure_count == 1
    assert circuit_breaker.total_failures == 1
    assert circuit_breaker.consecutive_failures == 1
    assert circuit_breaker.total_requests == 1
    assert circuit_breaker.state == CircuitState.CLOSED  # Still closed


def test_circuit_breaker_record_success(circuit_breaker):
    """Test recording successes increases success count."""
    circuit_breaker.record_success()
    assert circuit_breaker.success_count == 0  # Only incremented in HALF_OPEN state
    assert circuit_breaker.total_successes == 1
    assert circuit_breaker.consecutive_failures == 0
    assert circuit_breaker.total_requests == 1
    assert circuit_breaker.state == CircuitState.CLOSED


def test_circuit_breaker_trip_open(circuit_breaker):
    """Test circuit breaker opens after threshold failures."""
    # Record failures up to threshold
    for _ in range(circuit_breaker.failure_threshold):
        circuit_breaker.record_failure()
    
    # Circuit should be open now
    assert circuit_breaker.state == CircuitState.OPEN
    assert circuit_breaker.total_failures == circuit_breaker.failure_threshold
    assert circuit_breaker.consecutive_failures == circuit_breaker.failure_threshold


def test_circuit_breaker_half_open_after_timeout(circuit_breaker):
    """Test circuit transitions to half-open after timeout."""
    # Trip the circuit
    for _ in range(circuit_breaker.failure_threshold):
        circuit_breaker.record_failure()
    
    assert circuit_breaker.state == CircuitState.OPEN
    
    # Mock the time to be after recovery timeout
    original_time = circuit_breaker.last_state_change
    with patch('time.time') as mock_time:
        mock_time.return_value = original_time + circuit_breaker.recovery_timeout + 1
        
        # Check if request should be allowed - this should transition to HALF_OPEN
        allowed = circuit_breaker._should_allow_request()
        
        # First request after timeout should be allowed
        assert allowed is True
        assert circuit_breaker.state == CircuitState.HALF_OPEN


def test_circuit_breaker_close_after_success_threshold(circuit_breaker):
    """Test circuit closes after success threshold in half-open state."""
    # Trip the circuit
    for _ in range(circuit_breaker.failure_threshold):
        circuit_breaker.record_failure()
    
    # Transition to half-open
    with patch('time.time') as mock_time:
        mock_time.return_value = circuit_breaker.last_state_change + circuit_breaker.recovery_timeout + 1
        circuit_breaker._should_allow_request()
    
    assert circuit_breaker.state == CircuitState.HALF_OPEN
    
    # Record successful requests to meet threshold
    for _ in range(circuit_breaker.success_threshold):
        circuit_breaker.record_success()
    
    # Circuit should be closed now
    assert circuit_breaker.state == CircuitState.CLOSED
    assert circuit_breaker.failure_count == 0  # Failures reset


def test_circuit_breaker_reopen_on_failure_in_half_open(circuit_breaker):
    """Test circuit reopens on failure in half-open state."""
    # Trip the circuit
    for _ in range(circuit_breaker.failure_threshold):
        circuit_breaker.record_failure()
    
    # Transition to half-open
    with patch('time.time') as mock_time:
        mock_time.return_value = circuit_breaker.last_state_change + circuit_breaker.recovery_timeout + 1
        circuit_breaker._should_allow_request()
    
    assert circuit_breaker.state == CircuitState.HALF_OPEN
    
    # Record a failure in half-open state
    circuit_breaker.record_failure()
    
    # Circuit should be open again
    assert circuit_breaker.state == CircuitState.OPEN


def test_circuit_breaker_force_reset_after_max_timeout(circuit_breaker):
    """Test circuit force resets after max timeout."""
    # Trip the circuit
    for _ in range(circuit_breaker.failure_threshold):
        circuit_breaker.record_failure()
    
    assert circuit_breaker.state == CircuitState.OPEN
    
    # Mock time to be after max open timeout
    with patch('time.time') as mock_time:
        mock_time.return_value = circuit_breaker.last_state_change + circuit_breaker.max_open_timeout + 1
        
        # Get state should force reset due to timeout
        state = circuit_breaker.get_state()
        
        assert state == CircuitState.HALF_OPEN
        assert circuit_breaker.success_count == 0  # Reset


def test_circuit_breaker_manual_reset(circuit_breaker):
    """Test manually resetting the circuit breaker."""
    # Trip the circuit
    for _ in range(circuit_breaker.failure_threshold):
        circuit_breaker.record_failure()
    
    assert circuit_breaker.state == CircuitState.OPEN
    
    # Manually reset
    circuit_breaker.reset()
    
    assert circuit_breaker.state == CircuitState.CLOSED
    assert circuit_breaker.failure_count == 0
    assert circuit_breaker.failure_timestamps == []
    assert circuit_breaker.consecutive_failures == 0


def test_circuit_breaker_clean_old_failures(circuit_breaker):
    """Test cleaning old failures outside the window."""
    # Create failures at different times
    base_time = time.time()
    failure_times = [base_time - 20, base_time - 15, base_time - 5, base_time]
    
    # Temporarily patch _should_trip to prevent auto cleaning during test setup
    with patch.object(circuit_breaker, '_should_trip', return_value=False):
        with patch('time.time') as mock_time:
            for failure_time in failure_times:
                mock_time.return_value = failure_time
                circuit_breaker.record_failure()
            
            # Verify all 4 failures were recorded
            assert len(circuit_breaker.failure_timestamps) == 4
            assert circuit_breaker.failure_count == 4
    
    # Now manually clean old failures (window is 10 seconds)
    with patch('time.time') as mock_time:
        mock_time.return_value = failure_times[-1] + 1
        circuit_breaker._clean_old_failures()
    
    # Only the last 2 failures should remain (within 10 second window)
    assert circuit_breaker.failure_count == 2


def test_circuit_breaker_allow_percentage_in_half_open(circuit_breaker):
    """Test that only a percentage of requests are allowed in half-open state."""
    # Trip the circuit
    for _ in range(circuit_breaker.failure_threshold):
        circuit_breaker.record_failure()
    
    # Transition to half-open
    with patch('time.time') as mock_time:
        mock_time.return_value = circuit_breaker.last_state_change + circuit_breaker.recovery_timeout + 1
        circuit_breaker._should_allow_request()
    
    assert circuit_breaker.state == CircuitState.HALF_OPEN
    
    # Test with a deterministic random function
    with patch('random.randint') as mock_random:
        # Allow requests where random returns <= half_open_allow_percentage
        mock_random.return_value = circuit_breaker.half_open_allow_percentage
        assert circuit_breaker._should_allow_request() is True
        
        # Deny requests where random returns > half_open_allow_percentage
        mock_random.return_value = circuit_breaker.half_open_allow_percentage + 1
        assert circuit_breaker._should_allow_request() is False


def test_circuit_breaker_disabled(circuit_breaker_config):
    """Test that disabled circuit breaker always allows requests."""
    config = circuit_breaker_config.copy()
    config["enabled"] = False
    disabled_circuit = CircuitBreaker(**config)
    
    # Trip the circuit
    for _ in range(disabled_circuit.failure_threshold):
        disabled_circuit.record_failure()
    
    # Even though failures exceed threshold, requests should be allowed
    assert disabled_circuit._should_allow_request() is True


def test_circuit_breaker_execute_success(circuit_breaker):
    """Test executing a function with circuit breaker (success case)."""
    mock_func = MagicMock(return_value="success")
    
    result = circuit_breaker.execute(mock_func, "arg1", kwarg1="value1")
    
    assert result == "success"
    mock_func.assert_called_once_with("arg1", kwarg1="value1")
    assert circuit_breaker.total_successes == 1
    assert circuit_breaker.state == CircuitState.CLOSED


def test_circuit_breaker_execute_failure(circuit_breaker):
    """Test executing a function with circuit breaker (failure case)."""
    mock_func = MagicMock(side_effect=ValueError("test error"))
    
    with pytest.raises(ValueError, match="test error"):
        circuit_breaker.execute(mock_func, "arg1", kwarg1="value1")
    
    mock_func.assert_called_once_with("arg1", kwarg1="value1")
    assert circuit_breaker.total_failures == 1
    assert circuit_breaker.state == CircuitState.CLOSED


def test_circuit_breaker_execute_open_circuit(circuit_breaker):
    """Test executing a function with an open circuit."""
    # Trip the circuit
    for _ in range(circuit_breaker.failure_threshold):
        circuit_breaker.record_failure()
    
    assert circuit_breaker.state == CircuitState.OPEN
    
    # Try to execute when circuit is open
    mock_func = MagicMock(return_value="success")
    
    with pytest.raises(CircuitOpenError) as exc_info:
        circuit_breaker.execute(mock_func, "arg1", kwarg1="value1")
    
    assert "test_circuit is OPEN" in str(exc_info.value)
    assert exc_info.value.circuit_name == "test_circuit"
    assert exc_info.value.circuit_state == CircuitState.OPEN.value
    assert "metrics" in exc_info.value.__dict__
    mock_func.assert_not_called()


def test_circuit_breaker_save_load_state(circuit_breaker, circuit_breaker_config):
    """Test saving and loading circuit state."""
    # Set up some state
    circuit_breaker.record_failure()
    circuit_breaker.record_success()
    circuit_breaker.total_requests = 2
    
    # Mock open to capture the saved state
    mock_data = {}
    
    with patch("builtins.open", mock_open()) as mock_file, \
         patch("json.dump") as mock_dump, \
         patch("json.load", return_value={circuit_breaker.name: {
             "state": CircuitState.CLOSED.value,
             "failure_count": 1,
             "success_count": 0,
             "last_state_change": time.time(),
             "last_failure_time": time.time(),
             "last_success_time": time.time(),
             "total_failures": 1,
             "total_successes": 1,
             "total_requests": 2,
             "failure_timestamps": [time.time()],
             "consecutive_failures": 0
         }}):
        
        # Save state
        circuit_breaker._save_state()
        
        # Load state with a new circuit
        new_circuit = CircuitBreaker(**circuit_breaker_config)
        
        # Verify state was loaded
        assert new_circuit.failure_count == 1
        assert new_circuit.total_failures == 1
        assert new_circuit.total_successes == 1
        assert new_circuit.total_requests == 2


def test_circuit_protected_decorator():
    """Test circuit_protected decorator."""
    # Create mock function for the decorator
    mock_func = MagicMock(return_value="success")
    decorated_func = circuit_protected("test_decorator")(mock_func)
    
    # Mock get_circuit_breaker to return a controlled circuit
    mock_circuit = MagicMock()
    mock_circuit.execute.return_value = "success_from_circuit"
    
    with patch("yahoofinance_v2.utils.network.circuit_breaker.get_circuit_breaker", 
               return_value=mock_circuit):
        result = decorated_func("arg1", kwarg1="value1")
    
    assert result == "success_from_circuit"
    mock_circuit.execute.assert_called_once()
    # The first argument to execute should be the function
    assert mock_circuit.execute.call_args[0][0] == mock_func


def test_get_circuit_breaker():
    """Test get_circuit_breaker creates and returns circuit breakers."""
    # Clear existing circuits
    with patch.dict("yahoofinance_v2.utils.network.circuit_breaker._circuit_breakers", {}):
        circuit1 = get_circuit_breaker("test_get_1")
        circuit2 = get_circuit_breaker("test_get_2")
        circuit1_again = get_circuit_breaker("test_get_1")
        
        assert circuit1.name == "test_get_1"
        assert circuit2.name == "test_get_2"
        assert circuit1 is circuit1_again  # Should return the same instance


def test_reset_all_circuits():
    """Test reset_all_circuits resets all circuit breakers."""
    # Create mock circuits
    mock_circuit1 = MagicMock()
    mock_circuit2 = MagicMock()
    
    # Patch the circuits registry
    with patch.dict("yahoofinance_v2.utils.network.circuit_breaker._circuit_breakers", 
                   {"circuit1": mock_circuit1, "circuit2": mock_circuit2}):
        
        reset_all_circuits()
        
        mock_circuit1.reset.assert_called_once()
        mock_circuit2.reset.assert_called_once()


def test_circuit_breaker_metrics(circuit_breaker):
    """Test getting circuit breaker metrics."""
    # Set up some state
    circuit_breaker.record_failure()
    circuit_breaker.record_success()
    
    # Get metrics
    metrics = circuit_breaker.get_metrics()
    
    assert metrics["name"] == "test_circuit"
    assert metrics["state"] == CircuitState.CLOSED.value
    assert metrics["enabled"] is True
    assert metrics["current_failure_count"] == 1
    assert metrics["total_failures"] == 1
    assert metrics["total_successes"] == 1
    assert metrics["total_requests"] == 2
    assert metrics["failure_rate"] == 50.0  # 1 failure out of 2 requests
    assert "time_in_current_state" in metrics
    assert metrics["consecutive_failures"] == 0