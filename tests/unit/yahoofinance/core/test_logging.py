"""
Tests for yahoofinance/core/logging.py

This module tests logging configuration and utilities.
"""

import pytest
import logging
import tempfile
import os
from unittest.mock import MagicMock, patch

from yahoofinance.core.logging import (
    YFinanceErrorFilter,
    suppress_yfinance_noise,
    setup_logging,
    configure_logging,
    get_logger,
    DEFAULT_FORMAT,
    DEBUG_FORMAT,
    CONSOLE_FORMAT,
)


@pytest.fixture(autouse=True)
def cleanup_handlers():
    """Clean up logging handlers after each test."""
    yield
    # Remove all handlers from root logger
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()


class TestYFinanceErrorFilter:
    """Tests for YFinanceErrorFilter class."""

    def test_filter_allows_normal_messages(self):
        """Test that normal messages are allowed through."""
        filter_obj = YFinanceErrorFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Normal log message",
            args=(),
            exc_info=None,
        )
        assert filter_obj.filter(record) is True

    def test_filter_blocks_delisted_messages(self):
        """Test that delisting messages are blocked."""
        filter_obj = YFinanceErrorFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="AAPL: Possibly delisted",
            args=(),
            exc_info=None,
        )
        assert filter_obj.filter(record) is False

    def test_filter_blocks_earnings_messages(self):
        """Test that earnings date messages are blocked."""
        filter_obj = YFinanceErrorFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="No earnings dates found for AAPL",
            args=(),
            exc_info=None,
        )
        assert filter_obj.filter(record) is False

    def test_filter_blocks_http_errors(self):
        """Test that HTTP error messages are blocked."""
        filter_obj = YFinanceErrorFilter()

        error_messages = [
            "HTTP error 404",
            "HTTP Error 400",
            "http error 403",
            "HTTP ERROR 500",
        ]

        for msg in error_messages:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg=msg,
                args=(),
                exc_info=None,
            )
            assert filter_obj.filter(record) is False, f"Should block: {msg}"

    def test_filter_blocks_connection_errors(self):
        """Test that connection error messages are blocked."""
        filter_obj = YFinanceErrorFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Connection error while fetching data",
            args=(),
            exc_info=None,
        )
        assert filter_obj.filter(record) is False

    def test_filter_blocks_timeout_errors(self):
        """Test that timeout error messages are blocked."""
        filter_obj = YFinanceErrorFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Timeout error occurred",
            args=(),
            exc_info=None,
        )
        assert filter_obj.filter(record) is False

    def test_filter_handles_missing_msg(self):
        """Test filter handles records without msg attribute."""
        filter_obj = YFinanceErrorFilter()
        record = MagicMock()
        del record.msg  # Remove msg attribute
        # Should not raise, should allow through
        result = filter_obj.filter(record)
        assert result is True


class TestSuppressYfinanceNoise:
    """Tests for suppress_yfinance_noise function."""

    def test_applies_filter_to_yfinance_logger(self):
        """Test that filter is applied to yfinance logger."""
        suppress_yfinance_noise()

        yf_logger = logging.getLogger("yfinance")
        filter_applied = any(
            isinstance(f, YFinanceErrorFilter) for f in yf_logger.filters
        )
        assert filter_applied

    def test_applies_filter_to_urllib3_logger(self):
        """Test that filter is applied to urllib3 logger."""
        suppress_yfinance_noise()

        urllib_logger = logging.getLogger("urllib3")
        filter_applied = any(
            isinstance(f, YFinanceErrorFilter) for f in urllib_logger.filters
        )
        assert filter_applied

    def test_idempotent_filter_application(self):
        """Test that calling multiple times doesn't duplicate filters."""
        # Get initial filter count
        yf_logger = logging.getLogger("yfinance")
        initial_count = len([f for f in yf_logger.filters if isinstance(f, YFinanceErrorFilter)])

        # Call twice
        suppress_yfinance_noise()
        suppress_yfinance_noise()

        final_count = len([f for f in yf_logger.filters if isinstance(f, YFinanceErrorFilter)])

        # Should only have one filter (might be 1 or same as initial if already applied)
        assert final_count <= initial_count + 1


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_with_default_level(self):
        """Test setup with default INFO level."""
        setup_logging()
        root_logger = logging.getLogger()
        assert root_logger.level <= logging.INFO

    def test_setup_with_debug_level(self):
        """Test setup with DEBUG level."""
        setup_logging(log_level=logging.DEBUG)
        # Should not raise

    def test_setup_with_string_level(self):
        """Test setup with string level."""
        setup_logging(log_level="WARNING")
        # Should not raise

    def test_setup_with_log_file(self):
        """Test setup with log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            setup_logging(log_file=log_file)

            # Log something
            logger = logging.getLogger(__name__)
            logger.info("Test message")

            # Check file was created (may be buffered)
            # Just verify no error was raised

    def test_setup_without_console(self):
        """Test setup without console output."""
        setup_logging(console=False)
        # Should not raise

    def test_setup_with_custom_format(self):
        """Test setup with custom format."""
        custom_format = "%(name)s - %(message)s"
        setup_logging(log_format=custom_format)
        # Should not raise


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_with_debug_mode(self):
        """Test configure with debug mode enabled."""
        configure_logging(debug=True)
        # Should configure debug-level logging

    def test_configure_with_console_level(self):
        """Test configure with separate console level."""
        configure_logging(level=logging.DEBUG, console_level=logging.INFO)
        # Should not raise


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger(__name__)
        assert isinstance(logger, logging.Logger)

    def test_get_logger_with_module_name(self):
        """Test get_logger with module name."""
        logger = get_logger("test.module.name")
        assert logger.name == "test.module.name"

    def test_get_logger_caches_instances(self):
        """Test that get_logger returns same instance for same name."""
        logger1 = get_logger("same.name")
        logger2 = get_logger("same.name")
        assert logger1 is logger2


class TestLoggingConstants:
    """Tests for logging constants."""

    def test_default_format_valid(self):
        """Test DEFAULT_FORMAT is a valid format string."""
        assert "%(asctime)s" in DEFAULT_FORMAT
        assert "%(levelname)s" in DEFAULT_FORMAT
        assert "%(message)s" in DEFAULT_FORMAT

    def test_debug_format_includes_location(self):
        """Test DEBUG_FORMAT includes file location."""
        assert "%(filename)s" in DEBUG_FORMAT
        assert "%(lineno)d" in DEBUG_FORMAT

    def test_console_format_concise(self):
        """Test CONSOLE_FORMAT is concise."""
        assert "%(message)s" in CONSOLE_FORMAT
        # Should not have timestamp for brevity
        assert "asctime" not in CONSOLE_FORMAT


class TestLoggingIntegration:
    """Integration tests for logging."""

    def test_logger_can_log_all_levels(self):
        """Test that logger can log at all levels."""
        setup_logging(log_level=logging.DEBUG)
        logger = get_logger("test.all.levels")

        # Should not raise at any level
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

    def test_filter_integrates_with_logging(self):
        """Test that filter works in actual logging flow."""
        # Setup logging and apply filter
        suppress_yfinance_noise()

        # Get yfinance logger
        yf_logger = logging.getLogger("yfinance")
        yf_logger.setLevel(logging.WARNING)

        # Create a handler to capture logs
        captured_logs = []
        class CaptureHandler(logging.Handler):
            def emit(self, record):
                captured_logs.append(record.getMessage())

        capture_handler = CaptureHandler()
        yf_logger.addHandler(capture_handler)

        try:
            # Log a message that should be filtered
            yf_logger.warning("Possibly delisted")

            # Log a message that should pass
            yf_logger.warning("Normal warning")

            # Check only normal warning was captured
            assert "Possibly delisted" not in captured_logs
            assert "Normal warning" in captured_logs
        finally:
            yf_logger.removeHandler(capture_handler)
