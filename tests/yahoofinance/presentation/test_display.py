from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from yahoofinance.core.errors import YFinanceError
from yahoofinance.presentation.console import MarketDisplay
from yahoofinance.presentation.formatter import DisplayConfig, DisplayFormatter


@pytest.fixture
def mock_provider():
    provider = Mock()
    return provider


@pytest.fixture
def display(mock_provider):
    with patch("yahoofinance.api.get_provider", return_value=mock_provider):
        display = MarketDisplay(provider=mock_provider)
        return display


def test_init_default():
    # Test with default parameters
    display = MarketDisplay()
    assert display.provider is None
    assert display.formatter is not None


def test_init_custom():
    config = DisplayConfig()
    mock_provider = Mock()
    formatter = DisplayFormatter(compact_mode=True)

    display = MarketDisplay(provider=mock_provider, formatter=formatter, config=config)
    assert display.provider is mock_provider
    assert display.formatter is formatter
    assert display.config is config
