"""
Circuit breaker pattern implementation for API calls.

This module provides a circuit breaker implementation to prevent
cascading failures when an API or service is experiencing issues.
The circuit breaker will automatically stop requests to the service
when failure rates exceed thresholds, and will gradually allow
requests through during recovery.
"""

import json
import os
import secrets
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from ...core.config import CIRCUIT_BREAKER
from ...core.errors import APIError, DataError, ValidationError, YFinanceError
from ...core.logging import get_logger


# Type variables for generic function signatures
T = TypeVar("T")
R = TypeVar("R")

# Set up logging
logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "CLOSED"  # Normal operation, requests flow through
    OPEN = "OPEN"  # Circuit is open, requests are blocked
    HALF_OPEN = "HALF_OPEN"  # Testing if service has recovered


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    The circuit breaker tracks failures and will open (block requests)
    when the failure rate exceeds a threshold. After a timeout period,
    it will allow some requests through to test if the service has
    recovered.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = None,
        failure_window: int = None,
        recovery_timeout: int = None,
        success_threshold: int = None,
        half_open_allow_percentage: int = None,
        max_open_timeout: int = None,
        timeout: float = None,
        enabled: bool = None,
        state_file: str = None,
    ):
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
            timeout: Maximum execution time for protected functions in seconds
            enabled: Whether the circuit breaker is enabled
            state_file: File to persist circuit state
        """
        self.name = name
        # Use sensible defaults if CIRCUIT_BREAKER global config is not available or incomplete
        try:
            from ...core.config import CIRCUIT_BREAKER as config
        except ImportError:
            config = {}
        
        self.failure_threshold = failure_threshold or config.get("FAILURE_THRESHOLD", 5)
        self.failure_window = failure_window or config.get("FAILURE_WINDOW", 60)
        self.recovery_timeout = recovery_timeout or config.get("RECOVERY_TIMEOUT", 300)
        self.success_threshold = success_threshold or config.get("SUCCESS_THRESHOLD", 3)
        self.half_open_allow_percentage = (
            half_open_allow_percentage or config.get("HALF_OPEN_ALLOW_PERCENTAGE", 10)
        )
        self.max_open_timeout = max_open_timeout or config.get("MAX_OPEN_TIMEOUT", 1800)
        self.timeout = timeout or config.get("TIMEOUT", 10.0)
        self.enabled = enabled if enabled is not None else config.get("ENABLED", True)
        # Use secure temporary directory location
        import tempfile
        import os
        secure_temp_dir = tempfile.gettempdir()
        default_state_file = os.path.join(secure_temp_dir, f"circuit_breaker_{name}.json")
        self.state_file = state_file or config.get("STATE_FILE", default_state_file)

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
        if not self.state_file or not os.path.exists(self.state_file):
            # Ensure directory exists if state file is specified
            if self.state_file:
                os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            return

        try:
            with open(self.state_file, "r") as f:
                try:
                    state_data = json.load(f)
                except json.JSONDecodeError:
                    # File exists but is not valid JSON, create empty state data
                    state_data = {}

            # Only load state for this specific circuit breaker
            if self.name in state_data:
                circuit_data = state_data[self.name]

                # Check if state is still valid (not too old)
                last_change = circuit_data.get("last_state_change", 0)
                if time.time() - last_change < self.max_open_timeout:
                    state_value = circuit_data.get("state", CircuitState.CLOSED.value)
                    try:
                        self.state = CircuitState(state_value)
                    except ValueError:
                        # Invalid state value, default to CLOSED
                        self.state = CircuitState.CLOSED

                    self.failure_count = circuit_data.get("failure_count", 0)
                    self.success_count = circuit_data.get("success_count", 0)
                    self.last_state_change = last_change
                    self.last_failure_time = circuit_data.get("last_failure_time", 0)
                    self.last_success_time = circuit_data.get("last_success_time", 0)
                    self.total_failures = circuit_data.get("total_failures", 0)
                    self.total_successes = circuit_data.get("total_successes", 0)
                    self.total_requests = circuit_data.get("total_requests", 0)
                    self.failure_timestamps = circuit_data.get("failure_timestamps", [])
                    self.consecutive_failures = circuit_data.get("consecutive_failures", 0)
                else:
                    # Reset to closed if state is too old
                    self.state = CircuitState.CLOSED

            logger.debug(f"Loaded circuit breaker state for '{self.name}': {self.state.value}")
        except YFinanceError as e:
            logger.warning(f"Failed to load circuit breaker state: {str(e)}")
            # Ensure we default to CLOSED state for safety
            self.state = CircuitState.CLOSED

    def _save_state(self) -> None:
        """Save circuit breaker state to file"""
        if not self.state_file:
            return

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

            # Load existing state file
            state_data = {}
            if os.path.exists(self.state_file):
                try:
                    with open(self.state_file, "r") as f:
                        try:
                            state_data = json.load(f)
                        except json.JSONDecodeError:
                            # File exists but is not valid JSON, create empty state data
                            state_data = {}
                except YFinanceError as e:
                    logger.warning(f"Error reading state file: {str(e)}, creating new state file")

            # Update with current state
            state_data[self.name] = {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_state_change": self.last_state_change,
                "last_failure_time": self.last_failure_time,
                "last_success_time": self.last_success_time,
                "total_failures": self.total_failures,
                "total_successes": self.total_successes,
                "total_requests": self.total_requests,
                "failure_timestamps": self.failure_timestamps,
                "consecutive_failures": self.consecutive_failures,
            }

            # Save updated state
            with open(self.state_file, "w") as f:
                json.dump(state_data, f, indent=2)

            logger.debug(f"Saved circuit breaker state for '{self.name}'")
        except YFinanceError as e:
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
                # Use a more deterministic approach based on request count
                with self.lock:
                    self.total_requests += 1
                    # Allow exactly the configured percentage of requests through
                    # by using modulo on the request count
                    return (self.total_requests % 100) < self.half_open_allow_percentage

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
                    logger.info(
                        f"Circuit '{self.name}' closing after {self.success_count} successful operations"
                    )
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
            if (
                self.state == CircuitState.OPEN
                and time.time() - self.last_state_change > self.max_open_timeout
            ):
                logger.info(
                    f"Circuit '{self.name}' force reset after exceeding "
                    f"maximum open timeout of {self.max_open_timeout}s"
                )
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = time.time()
                self.success_count = 0
                self._save_state()
            # Check if enough time has passed to transition from OPEN to HALF_OPEN
            elif self.state == CircuitState.OPEN and self._should_attempt_reset():
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
                "last_failure_ago": (
                    round(now - self.last_failure_time, 2) if self.last_failure_time > 0 else None
                ),
                "last_success_ago": (
                    round(now - self.last_success_time, 2) if self.last_success_time > 0 else None
                ),
                "time_until_reset": (
                    round(self.recovery_timeout - (now - self.last_state_change), 2)
                    if current_state == CircuitState.OPEN
                    else None
                ),
                "recovery_percentage": (
                    self.half_open_allow_percentage
                    if current_state == CircuitState.HALF_OPEN
                    else None
                ),
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
            CircuitBreakerError: If function exceeds timeout
            Original exception: If function call fails
        """
        if not self._should_allow_request():
            raise CircuitOpenError(
                f"{self.name} is OPEN - request rejected",
                circuit_name=self.name,
                circuit_state=self.state.value,
                metrics=self.get_metrics(),
            )

        try:
            start_time = time.time()
            result = func(*args, **kwargs)

            # Check if execution exceeded timeout
            if self.timeout and time.time() - start_time > self.timeout:
                self.record_failure()
                raise CircuitBreakerError("Circuit breaker timeout")

            self.record_success()
            return result
        except YFinanceError as e:
            self.record_failure()
            raise e


class CircuitBreakerError(Exception):
    """Exception raised when a circuit breaker operation fails"""

    def __init__(self, message: str, circuit_name: str = None, details: Dict[str, Any] = None):
        """
        Initialize circuit breaker error.

        Args:
            message: Error message
            circuit_name: Name of the circuit breaker (optional)
            details: Additional error details (optional)
        """
        self.circuit_name = circuit_name
        self.details = details or {}
        super().__init__(message)


class CircuitOpenError(Exception):
    """Exception raised when a circuit is open and rejects a request"""

    def __init__(
        self, message: str, circuit_name: str, circuit_state: str, metrics: Dict[str, Any]
    ):
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
            CircuitBreakerError: If function exceeds timeout
            Original exception: If function call fails
        """
        if not self._should_allow_request():
            raise CircuitOpenError(
                f"{self.name} is OPEN - request rejected",
                circuit_name=self.name,
                circuit_state=self.state.value,
                metrics=self.get_metrics(),
            )

        try:
            start_time = time.time()
            result = await func(*args, **kwargs)

            # Check if execution exceeded timeout
            if self.timeout and time.time() - start_time > self.timeout:
                self.record_failure()
                raise CircuitBreakerError("Circuit breaker timeout")

            self.record_success()
            return result
        except YFinanceError as e:
            self.record_failure()
            raise e


# Circuit breaker registry for dependency injection
class CircuitBreakerRegistry:
    """Registry for managing circuit breaker instances with dependency injection support."""
    
    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the circuit breaker registry.
        
        Args:
            default_config: Default configuration for circuit breakers
        """
        from ...core.config import CIRCUIT_BREAKER
        self.default_config = default_config or CIRCUIT_BREAKER
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def get_circuit_breaker(self, name: str, config: Optional[Dict[str, Any]] = None) -> CircuitBreaker:
        """
        Get or create a circuit breaker by name.
        
        Args:
            name: Name of the circuit breaker
            config: Optional configuration overrides
            
        Returns:
            Circuit breaker instance
        """
        with self._lock:
            if name not in self._circuit_breakers:
                # Merge default config with provided config
                final_config = self.default_config.copy()
                if config:
                    final_config.update(config)
                
                # Map configuration keys to constructor parameters
                # Handle missing keys gracefully for backward compatibility
                constructor_args = {
                    "failure_threshold": final_config.get("FAILURE_THRESHOLD"),
                    "failure_window": final_config.get("FAILURE_WINDOW", 60),  # Default to 60 if missing
                    "recovery_timeout": final_config.get("RECOVERY_TIMEOUT"),
                    "success_threshold": final_config.get("SUCCESS_THRESHOLD", 3),  # Default to 3 if missing
                    "half_open_allow_percentage": final_config.get("HALF_OPEN_ALLOW_PERCENTAGE", 10),  # Default to 10 if missing
                    "max_open_timeout": final_config.get("MAX_OPEN_TIMEOUT", 1800),  # Default to 30 min if missing
                    "timeout": final_config.get("TIMEOUT", 10.0),  # Default to 10s if missing
                    "enabled": final_config.get("ENABLED", True),  # Default to True if missing
                    "state_file": final_config.get("STATE_FILE"),
                }
                
                # Remove None values
                constructor_args = {k: v for k, v in constructor_args.items() if v is not None}
                
                self._circuit_breakers[name] = CircuitBreaker(name=name, **constructor_args)
                logger.debug(f"Created circuit breaker '{name}' with config: {constructor_args}")
            
            return self._circuit_breakers[name]
    
    def create_circuit_breaker(self, name: str, config: Optional[Dict[str, Any]] = None) -> CircuitBreaker:
        """
        Create a new circuit breaker instance (not cached).
        
        Args:
            name: Name of the circuit breaker
            config: Configuration for the circuit breaker
            
        Returns:
            New CircuitBreaker instance
        """
        final_config = self.default_config.copy()
        if config:
            final_config.update(config)
        
        # Map configuration keys to constructor parameters
        # Handle missing keys gracefully for backward compatibility
        constructor_args = {
            "failure_threshold": final_config.get("FAILURE_THRESHOLD"),
            "failure_window": final_config.get("FAILURE_WINDOW", 60),  # Default to 60 if missing
            "recovery_timeout": final_config.get("RECOVERY_TIMEOUT"),
            "success_threshold": final_config.get("SUCCESS_THRESHOLD", 3),  # Default to 3 if missing
            "half_open_allow_percentage": final_config.get("HALF_OPEN_ALLOW_PERCENTAGE", 10),  # Default to 10 if missing
            "max_open_timeout": final_config.get("MAX_OPEN_TIMEOUT", 1800),  # Default to 30 min if missing
            "timeout": final_config.get("TIMEOUT", 10.0),  # Default to 10s if missing
            "enabled": final_config.get("ENABLED", True),  # Default to True if missing
            "state_file": final_config.get("STATE_FILE"),
        }
        
        # Remove None values
        constructor_args = {k: v for k, v in constructor_args.items() if v is not None}
        
        return CircuitBreaker(name=name, **constructor_args)
    
    def get_all_circuits(self) -> Dict[str, CircuitBreaker]:
        """
        Get all registered circuit breakers.
        
        Returns:
            Dictionary of all circuit breaker instances
        """
        with self._lock:
            return self._circuit_breakers.copy()
    
    def clear_circuits(self) -> None:
        """Clear all circuit breaker instances (useful for testing)."""
        with self._lock:
            self._circuit_breakers.clear()
            logger.debug("Cleared all circuit breaker instances")


# Create a default circuit breaker registry
_default_circuit_breaker_registry = CircuitBreakerRegistry()

# Global registry of circuit breakers (for backward compatibility)
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_circuit_breakers_lock = threading.RLock()


def get_circuit_breaker(name: str) -> CircuitBreaker:
    """
    Get or create a circuit breaker by name.
    
    This function provides backward compatibility by delegating to the default registry.

    Args:
        name: Name of the circuit breaker

    Returns:
        Circuit breaker instance
    """
    return _default_circuit_breaker_registry.get_circuit_breaker(name)


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

        # Ensure the returned circuit breaker is of the correct type
        # If we have a CircuitBreaker but need AsyncCircuitBreaker, replace it
        if not isinstance(_circuit_breakers[name], AsyncCircuitBreaker):
            logger.warning(
                f"Converting circuit breaker '{name}' from CircuitBreaker to AsyncCircuitBreaker"
            )
            # Copy state from the existing circuit breaker
            old_circuit = _circuit_breakers[name]
            new_circuit = AsyncCircuitBreaker(
                name=name,
                failure_threshold=old_circuit.failure_threshold,
                failure_window=old_circuit.failure_window,
                recovery_timeout=old_circuit.recovery_timeout,
                success_threshold=old_circuit.success_threshold,
                half_open_allow_percentage=old_circuit.half_open_allow_percentage,
                max_open_timeout=old_circuit.max_open_timeout,
                timeout=getattr(old_circuit, "timeout", None),
                enabled=old_circuit.enabled,
                state_file=old_circuit.state_file,
            )
            # Copy dynamic state
            new_circuit.state = old_circuit.state
            new_circuit.failure_count = old_circuit.failure_count
            new_circuit.failure_timestamps = old_circuit.failure_timestamps.copy()
            new_circuit.success_count = old_circuit.success_count
            new_circuit.last_state_change = old_circuit.last_state_change
            new_circuit.last_failure_time = old_circuit.last_failure_time
            new_circuit.last_success_time = old_circuit.last_success_time
            new_circuit.total_failures = old_circuit.total_failures
            new_circuit.total_successes = old_circuit.total_successes
            new_circuit.total_requests = old_circuit.total_requests
            new_circuit.consecutive_failures = old_circuit.consecutive_failures

            # Replace in registry
            _circuit_breakers[name] = new_circuit

        return _circuit_breakers[name]  # type: ignore


def reset_all_circuits() -> None:
    """Reset all circuit breakers to closed state"""
    with _circuit_breakers_lock:
        # First reset all circuit breakers
        for circuit in _circuit_breakers.values():
            circuit.reset()

        # Clear the cache to ensure new calls get fresh instances
        _circuit_breakers.clear()

    # Clear the state file to ensure a clean slate
    state_file = CIRCUIT_BREAKER["STATE_FILE"]
    if os.path.exists(state_file):
        try:
            with open(state_file, "w") as f:
                f.write("{}")
            logger.info(f"Cleared circuit breaker state file: {state_file}")
        except YFinanceError as e:
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
