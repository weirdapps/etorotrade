from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest

import yahoofinance.analysis.holders as holders


@pytest.fixture
def sample_major_holders():
    return pd.DataFrame(
        {"Value": [0.00519, 0.73502, 0.73886, 4004.0]},
        index=[
            "insidersPercentHeld",
            "institutionsPercentHeld",
            "institutionsFloatPercentHeld",
            "institutionsCount",
        ],
    )


@pytest.fixture
def sample_institutional_holders():
    return pd.DataFrame(
        {
            "Holder": ["Vanguard Group Inc", "Blackrock Inc."],
            "Shares": [68187418, 51149202],
            "Date Reported": [pd.Timestamp("2023-06-30"), pd.Timestamp("2023-06-30")],
            "pctHeld": [0.0936, 0.0702],
            "Value": [20347807405, 15263433368],
        }
    )


def test_format_percentage():
    assert holders.format_percentage(0.0519) == "5.19%"
    assert holders.format_percentage(4004) == "4,004"


def test_format_billions():
    assert holders.format_billions(20347807405) == "$20.35B"
    assert holders.format_billions(15263433368) == "$15.26B"


@patch("yfinance.Ticker")
def test_analyze_holders_with_data(
    mock_ticker, sample_major_holders, sample_institutional_holders, capsys
):
    # Configure mock
    mock_ticker_instance = Mock()
    mock_ticker_instance.major_holders = sample_major_holders
    mock_ticker_instance.institutional_holders = sample_institutional_holders
    mock_ticker_instance.info = {"sharesOutstanding": 1000000}
    mock_ticker.return_value = mock_ticker_instance

    # Test the analyze_holders function
    holders.analyze_holders("MCD")
    captured = capsys.readouterr()
    output = captured.out

    # Verify the output contains expected information
    assert "Analyzing MCD:" in output
    assert "Insiders Percentheld: 0.52%" in output
    assert "Institutions Percentheld: 73.50%" in output
    assert "Vanguard Group Inc" in output
    assert "Blackrock Inc." in output
    assert "$20.35B" in output
    assert "$15.26B" in output


@patch("yfinance.Ticker")
def test_analyze_holders_no_data(mock_ticker, capsys):
    # Configure mock with no data
    mock_ticker_instance = Mock()
    mock_ticker_instance.major_holders = None
    mock_ticker_instance.institutional_holders = None
    mock_ticker_instance.info = {"sharesOutstanding": 1000000}
    mock_ticker.return_value = mock_ticker_instance

    # Test the analyze_holders function
    holders.analyze_holders("INVALID")
    captured = capsys.readouterr()
    output = captured.out

    # Verify the output indicates no data available
    assert "Analyzing INVALID:" in output
    assert "No institutional holders information available" in output


@patch("builtins.input")
def test_main_single_ticker(mock_input, capsys):
    # Mock user entering one ticker and then quitting
    mock_input.side_effect = ["AAPL", "q"]

    with patch("yahoofinance.analysis.holders.analyze_holders") as mock_analyze:
        holders.main()
        mock_analyze.assert_called_once_with("AAPL")


@patch("builtins.input")
def test_main_multiple_tickers(mock_input, capsys):
    # Mock user entering multiple tickers and then quitting
    mock_input.side_effect = ["AAPL,MSFT,GOOGL", "q"]

    with patch("yahoofinance.analysis.holders.analyze_holders") as mock_analyze:
        holders.main()
        assert mock_analyze.call_count == 3
        mock_analyze.assert_any_call("AAPL")
        mock_analyze.assert_any_call("MSFT")
        mock_analyze.assert_any_call("GOOGL")


@patch("builtins.input")
def test_main_error_handling(mock_input, capsys):
    # Mock user entering invalid input and then quitting
    mock_input.side_effect = ["", "q"]

    with pytest.raises((ValueError, SystemExit)):
        holders.main()

    captured = capsys.readouterr()
    assert "Please enter at least one ticker" in captured.out
