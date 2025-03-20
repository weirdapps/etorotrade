"""
Circuit breaker pattern implementation for API calls.

This module provides a circuit breaker implementation to prevent
cascading failures when an API or service is experiencing issues.
The circuit breaker will automatically stop requests to the service
when failure rates exceed thresholds, and will gradually allow
requests through during recovery.
"""

import os
import json
import time
import logging
import threading
import random
from enum import Enum
from typing import Dict, Any, Optional, Callable, TypeVar, List, Tuple
from functools import wraps
from datetime import datetime, timedelta

from ...core.config import CIRCUIT_BREAKER

# Type variables for generic function signatures
T = TypeVar('T')
R = TypeVar('R')

# Set up logging
logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "CLOSED"      # Normal operation, requests flow through
    OPEN = "OPEN"          # Circuit is open, requests are blocked
    HALF_OPEN = "HALF_OPEN"  # Testing if service has recovered


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.
    
    The circuit breaker tracks failures and will open (block requests)
    when the failure rate exceeds a threshold. After a timeout period,
    it will allow some requests through to test if the service has
    recovered.
    """
    
    def __init__(self, 
                 name: str,
                 failure_threshold: int = None,
                 failure_window: int = None,
                 recovery_timeout: int = None,
                 success_threshold: int = None,
                 half_open_allow_percentage: int = None,
                 max_open_timeout: int = None,
                 enabled: bool = None,
                 state_file: str = None):
        """
        Initialize the circuit breaker.
        
        Args:
            name: Unique name for this circuit breaker
            failure_threshold: Number of failures to trip the circuit
            failure_window: Time window in seconds to count failures
            recovery_timeout: Time in seconds before allowing test requests
            success_threshold: Number of successes to close the circuit
            half_open_allow_percentage: Percentage of requests to allow through in half-open state
            max_open_timeout: Maximum time circuit can stay open before forced reset
            enabled: Whether the circuit breaker is enabled
            state_file: File to persist circuit state
        """
        self.name = name
        self.failure_threshold = failure_threshold or CIRCUIT_BREAKER["FAILURE_THRESHOLD"]
        self.failure_window = failure_window or CIRCUIT_BREAKER["FAILURE_WINDOW"]
        self.recovery_timeout = recovery_timeout or CIRCUIT_BREAKER["RECOVERY_TIMEOUT"]
        self.success_threshold = success_threshold or CIRCUIT_BREAKER["SUCCESS_THRESHOLD"]
        self.half_open_allow_percentage = half_open_allow_percentage or CIRCUIT_BREAKER["HALF_OPEN_ALLOW_PERCENTAGE"]
        self.max_open_timeout = max_open_timeout or CIRCUIT_BREAKER["MAX_OPEN_TIMEOUT"]
        self.enabled = enabled if enabled is not None else CIRCUIT_BREAKER["ENABLED"]
        self.state_file = state_file or CIRCUIT_BREAKER["STATE_FILE"]
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.failure_timestamps: List[float] = []
        self.success_count = 0
        self.last_state_change = time.time()
        self.last_failure_time = 0
        self.last_success_time = 0
        self.total_failures = 0
        self.total_successes = 0
        self.total_requests = 0
        self.consecutive_failures = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Load previous state if available
        self._load_state()
        
        logger.debug(f"Initialized circuit breaker '{name}' in {self.state.value} state")
    
    def _load_state(self) -> None:
        """Load circuit breaker state from file if it exists"""
        if not os.path.exists(self.state_file):
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)
                
            # Only load state for this specific circuit breaker
            if self.name in state_data:
                circuit_data = state_data[self.name]
                
                # Check if state is still valid (not too old)
                last_change = circuit_data.get('last_state_change', 0)
                if time.time() - last_change < self.max_open_timeout:
                    self.state = CircuitState(circuit_data.get('state', CircuitState.CLOSED.value))
                    self.failure_count = circuit_data.get('failure_count', 0)
                    self.success_count = circuit_data.get('success_count', 0)
                    self.last_state_change = last_change
                    self.last_failure_time = circuit_data.get('last_failure_time', 0)
                    self.last_success_time = circuit_data.get('last_success_time', 0)
                    self.total_failures = circuit_data.get('total_failures', 0)
                    self.total_successes = circuit_data.get('total_successes', 0)
                    self.total_requests = circuit_data.get('total_requests', 0)
                    self.failure_timestamps = circuit_data.get('failure_timestamps', [])
                    self.consecutive_failures = circuit_data.get('consecutive_failures', 0)
                else:
                    # Reset to closed if state is too old
                    self.state = CircuitState.CLOSED
                    
            logger.debug(f"Loaded circuit breaker state for '{self.name}': {self.state.value}")
        except Exception as e:
            logger.warning(f"Failed to load circuit breaker state: {str(e)}")
    
    def _save_state(self) -> None:
        """Save circuit breaker state to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            
            # Load existing state file
            state_data = {}
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
            
            # Update with current state
            state_data[self.name] = {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_state_change': self.last_state_change,
                'last_failure_time': self.last_failure_time,
                'last_success_time': self.last_success_time,
                'total_failures': self.total_failures,
                'total_successes': self.total_successes,
                'total_requests': self.total_requests,
                'failure_timestamps': self.failure_timestamps,
                'consecutive_failures': self.consecutive_failures
            }
            
            # Save updated state
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
            logger.debug(f"Saved circuit breaker state for '{self.name}'")
        except Exception as e:
            logger.warning(f"Failed to save circuit breaker state: {str(e)}")
    
    def _clean_old_failures(self) -> None:
        """Remove failures outside the current window"""
        now = time.time()
        window_start = now - self.failure_window
        
        # Keep only failures within the window
        with self.lock:
            recent_failures = [t for t in self.failure_timestamps if t >= window_start]
            self.failure_timestamps = recent_failures
            self.failure_count = len(recent_failures)
    
    def _should_trip(self) -> bool:
        """Check if circuit should trip open based on recent failures"""
        # Clean old failures and then check threshold
        self._clean_old_failures()
        return self.failure_count >= self.failure_threshold
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        time_since_trip = time.time() - self.last_state_change
        return time_since_trip >= self.recovery_timeout
    
    def _should_close(self) -> bool:
        """Check if circuit should close based on successful test requests"""
        return self.success_count >= self.success_threshold
    
    def _should_allow_request(self) -> bool:
        """Determine if a request should be allowed through"""
        with self.lock:
            if not self.enabled:
                return True
            
            if self.state == CircuitState.CLOSED:
                return True
                
            if self.state == CircuitState.OPEN:
                # Check if we should attempt reset
                if self._should_attempt_reset():
                    logger.info(f"Circuit '{self.name}' transitioning from OPEN to HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    self.last_state_change = time.time()
                    self._save_state()
                    # Allow this request through as a test
                    return True
                return False
                
            if self.state == CircuitState.HALF_OPEN:
                # Only allow a percentage of requests through
                return random.randint(1, 100) <= self.half_open_allow_percentage
                
            # Default to closed for safety
            return True
    
    def record_success(self) -> None:
        """Record a successful operation"""
        with self.lock:
            self.last_success_time = time.time()
            self.total_successes += 1
            self.total_requests += 1
            self.consecutive_failures = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                
                if self._should_close():
                    logger.info(f"Circuit '{self.name}' closing after {self.success_count} successful operations")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.failure_timestamps.clear()
                    self.last_state_change = time.time()
            
            self._save_state()
    
    def record_failure(self) -> None:
        """Record a failed operation"""
        with self.lock:
            now = time.time()
            self.last_failure_time = now
            self.failure_timestamps.append(now)
            self.failure_count += 1
            self.total_failures += 1
            self.total_requests += 1
            self.consecutive_failures += 1
            self.success_count = 0
            
            # Handle state transitions
            if self.state == CircuitState.CLOSED and self._should_trip():
                logger.warning(
                    f"Circuit '{self.name}' tripping OPEN after {self.failure_count} "
                    f"failures in {self.failure_window}s window"
                )
                self.state = CircuitState.OPEN
                self.last_state_change = now
                
            elif self.state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit '{self.name}' reopening after test failure")
                self.state = CircuitState.OPEN
                self.last_state_change = now
            
            self._save_state()
    
    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        with self.lock:
            # Check for forced reset due to max timeout
            if (self.state == CircuitState.OPEN and 
                time.time() - self.last_state_change > self.max_open_timeout):
                logger.info(
                    f"Circuit '{self.name}' force reset after exceeding "
                    f"maximum open timeout of {self.max_open_timeout}s"
                )
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = time.time()
                self.success_count = 0
                self._save_state()
            # Check if enough time has passed to transition from OPEN to HALF_OPEN
            elif (self.state == CircuitState.OPEN and
                  self._should_attempt_reset()):
                logger.info(f"Circuit '{self.name}' transitioning from OPEN to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self.last_state_change = time.time()
                self._save_state()
            
            return self.state
    
    def reset(self) -> None:
        """Manually reset the circuit to closed state"""
        with self.lock:
            logger.info(f"Circuit '{self.name}' manually reset to CLOSED state")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.failure_timestamps.clear()
            self.success_count = 0
            self.last_state_change = time.time()
            self.consecutive_failures = 0
            self._save_state()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        with self.lock:
            current_state = self.get_state()
            now = time.time()
            
            metrics = {
                "name": self.name,
                "state": current_state.value,
                "enabled": self.enabled,
                "current_failure_count": self.failure_count,
                "success_count": self.success_count,
                "total_failures": self.total_failures,
                "total_successes": self.total_successes,
                "total_requests": self.total_requests,
                "failure_rate": round(self.total_failures / max(1, self.total_requests) * 100, 2),
                "time_in_current_state": round(now - self.last_state_change, 2),
                "consecutive_failures": self.consecutive_failures,
                "circuit_age": round(now - self.last_state_change, 2),
                "last_failure_ago": round(now - self.last_failure_time, 2) if self.last_failure_time > 0 else None,
                "last_success_ago": round(now - self.last_success_time, 2) if self.last_success_time > 0 else None,
                "time_until_reset": round(
                    self.recovery_timeout - (now - self.last_state_change), 2
                ) if current_state == CircuitState.OPEN else None,
                "recovery_percentage": self.half_open_allow_percentage if current_state == CircuitState.HALF_OPEN else None
            }
            
            return metrics
    
    def execute(self, func, *args, **kwargs):
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function if successful
            
        Raises:
            CircuitOpenError: If circuit is open
            Original exception: If function call fails
        """
        if not self._should_allow_request():
            raise CircuitOpenError(
                f"{self.name} is OPEN - request rejected",
                circuit_name=self.name,
                circuit_state=self.state.value,
                metrics=self.get_metrics()
            )
        
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise e


class CircuitOpenError(Exception):
    """Exception raised when a circuit is open and rejects a request"""
    
    def __init__(self, message: str, circuit_name: str, circuit_state: str, metrics: Dict[str, Any]):
        """
        Initialize circuit open error.
        
        Args:
            message: Error message
            circuit_name: Name of the circuit breaker
            circuit_state: Current state of the circuit
            metrics: Circuit metrics
        """
        self.circuit_name = circuit_name
        self.circuit_state = circuit_state
        self.metrics = metrics
        self.retry_after = metrics.get("time_until_reset", 300)  # Default 5 minutes
        super().__init__(message)


class AsyncCircuitBreaker(CircuitBreaker):
    """Asynchronous version of the circuit breaker"""
    
    async def execute_async(self, func, *args, **kwargs):
        """
        Execute an async function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function if successful
            
        Raises:
            CircuitOpenError: If circuit is open
            Original exception: If function call fails
        """
        if not self._should_allow_request():
            raise CircuitOpenError(
                f"{self.name} is OPEN - request rejected",
                circuit_name=self.name,
                circuit_state=self.state.value,
                metrics=self.get_metrics()
            )
        
        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise e


# Global registry of circuit breakers
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_circuit_breakers_lock = threading.RLock()

def get_circuit_breaker(name: str) -> CircuitBreaker:
    """
    Get or create a circuit breaker by name.
    
    Args:
        name: Name of the circuit breaker
        
    Returns:
        Circuit breaker instance
    """
    with _circuit_breakers_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name=name)
        return _circuit_breakers[name]

def get_async_circuit_breaker(name: str) -> AsyncCircuitBreaker:
    """
    Get or create an async circuit breaker by name.
    
    Args:
        name: Name of the circuit breaker
        
    Returns:
        Async circuit breaker instance
    """
    with _circuit_breakers_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = AsyncCircuitBreaker(name=name)
        return _circuit_breakers[name]  # type: ignore

def reset_all_circuits() -> None:
    """Reset all circuit breakers to closed state"""
    with _circuit_breakers_lock:
        for circuit in _circuit_breakers.values():
            circuit.reset()
            
    # Clear the state file to ensure a clean slate
    state_file = CIRCUIT_BREAKER["STATE_FILE"]
    if os.path.exists(state_file):
        try:
            with open(state_file, 'w') as f:
                f.write('{}')
            logger.info(f"Cleared circuit breaker state file: {state_file}")
        except Exception as e:
            logger.warning(f"Failed to clear circuit breaker state file: {str(e)}")

def get_all_circuits() -> Dict[str, Dict[str, Any]]:
    """
    Get metrics for all circuit breakers.
    
    Returns:
        Dictionary of circuit breaker metrics keyed by name
    """
    with _circuit_breakers_lock:
        return {name: circuit.get_metrics() for name, circuit in _circuit_breakers.items()}

def circuit_protected(circuit_name: str):
    """
    Decorator for protecting functions with a circuit breaker.
    
    Args:
        circuit_name: Name of the circuit breaker to use
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            circuit = get_circuit_breaker(circuit_name)
            return circuit.execute(func, *args, **kwargs)
        return wrapper
    return decorator

def async_circuit_protected(circuit_name: str):
    """
    Decorator for protecting async functions with a circuit breaker.
    
    Args:
        circuit_name: Name of the circuit breaker to use
        
    Returns:
        Decorated async function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            circuit = get_async_circuit_breaker(circuit_name)
            return await circuit.execute_async(func, *args, **kwargs)
        return wrapper
    return decorator