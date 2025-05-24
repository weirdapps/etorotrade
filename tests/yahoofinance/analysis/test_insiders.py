from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from yahoofinance.analysis.insiders import InsiderAnalyzer, InsiderSummary, InsiderTransaction
from yahoofinance.utils.error_handling import with_retry


# Sample transaction data for testing
SAMPLE_TRANSACTIONS = [
    {
        "name": "John Doe",
        "title": "CEO",
        "date": "2024-01-15",
        "transaction_type": "Buy",
        "shares": 1000,
        "value": 100000.0,
        "share_price": 100.0,
    },
    {
        "name": "Jane Smith",
        "title": "CFO",
        "date": "2024-01-16",
        "transaction_type": "Sell",
        "shares": 500,
        "value": 50000.0,
        "share_price": 100.0,
    },
    {
        "name": "Bob Johnson",
        "title": "Director",
        "date": "2024-01-17",
        "transaction_type": "Buy",
        "shares": 200,
        "value": 20000.0,
        "share_price": 100.0,
    },
]


@pytest.fixture
def mock_provider():
    provider = Mock()
    provider.get_insider_transactions = Mock(return_value=SAMPLE_TRANSACTIONS)
    return provider


@pytest.fixture
def async_mock_provider():
    provider = Mock()

    # Create a spy to track calls
    call_tracker = {"called": False}

    async def mock_get_insider_transactions(*args, **kwargs):
        call_tracker["called"] = True
        call_tracker["args"] = args
        call_tracker["kwargs"] = kwargs
        return SAMPLE_TRANSACTIONS

    provider.get_insider_transactions = mock_get_insider_transactions
    provider.call_tracker = call_tracker
    return provider


@pytest.fixture
def analyzer(mock_provider):
    return InsiderAnalyzer(mock_provider)


@pytest.fixture
def async_analyzer(async_mock_provider):
    analyzer = InsiderAnalyzer(async_mock_provider)
    # Force is_async to True for testing
    analyzer.is_async = True
    return analyzer


class TestInsiderTransaction:
    def test_insider_transaction_init(self):
        """Test InsiderTransaction initialization"""
        transaction = InsiderTransaction(
            name="John Doe",
            title="CEO",
            date="2024-01-15",
            transaction_type="Buy",
            shares=1000,
            value=100000.0,
            share_price=100.0,
        )

        assert transaction.name == "John Doe"
        assert transaction.title == "CEO"
        assert transaction.date == "2024-01-15"
        assert transaction.transaction_type == "Buy"
        assert transaction.shares == 1000
        assert transaction.value == pytest.approx(100000.0, abs=1e-9)
        assert transaction.share_price == pytest.approx(100.0, abs=1e-9)


class TestInsiderSummary:
    def test_insider_summary_init(self):
        """Test InsiderSummary initialization"""
        summary = InsiderSummary()

        assert summary.transactions == []
        assert summary.buy_count == 0
        assert summary.sell_count == 0
        assert summary.total_buy_value == pytest.approx(0.0, abs=1e-9)
        assert summary.total_sell_value == pytest.approx(0.0, abs=1e-9)
        assert summary.net_value == pytest.approx(0.0, abs=1e-9)
        assert summary.net_share_count == 0
        assert summary.average_buy_price is None
        assert summary.average_sell_price is None


class TestInsiderAnalyzer:
    def test_init(self, mock_provider):
        """Test analyzer initialization"""
        analyzer = InsiderAnalyzer(mock_provider)
        assert analyzer.provider == mock_provider
        assert not analyzer.is_async

    def test_get_transactions(self, analyzer, mock_provider):
        """Test get_transactions method"""
        with patch("yahoofinance.analysis.insiders.is_us_ticker", return_value=True):
            # Mock _process_transactions_data to return a controlled result
            expected_summary = InsiderSummary(
                transactions=[],
                buy_count=2,
                sell_count=1,
                total_buy_value=120000.0,
                total_sell_value=50000.0,
                net_value=70000.0,
                net_share_count=700,
            )

            with patch.object(
                analyzer, "_process_transactions_data", return_value=expected_summary
            ):
                result = analyzer.get_transactions("AAPL")

                mock_provider.get_insider_transactions.assert_called_once()
                assert isinstance(result, InsiderSummary)
                assert result.buy_count == 2
                assert result.sell_count == 1
                assert result.total_buy_value == pytest.approx(120000.0, abs=1e-9)
                assert result.total_sell_value == pytest.approx(50000.0, abs=1e-9)
                assert result.net_value == pytest.approx(70000.0, abs=1e-9)
                assert result.net_share_count == 700

    def test_get_transactions_non_us_ticker(self, analyzer):
        """Test get_transactions for non-US ticker"""
        with patch("yahoofinance.analysis.insiders.is_us_ticker", return_value=False):
            result = analyzer.get_transactions("9988.HK")

            assert isinstance(result, InsiderSummary)
            assert len(result.transactions) == 0
            assert result.buy_count == 0
            assert result.sell_count == 0

    @pytest.mark.asyncio
    async def test_get_transactions_async(self, async_analyzer, async_mock_provider):
        """Test get_transactions_async method"""
        with patch("yahoofinance.analysis.insiders.is_us_ticker", return_value=True):
            # Mock _process_transactions_data to return a controlled result
            expected_summary = InsiderSummary(
                transactions=[],
                buy_count=2,
                sell_count=1,
                total_buy_value=120000.0,
                total_sell_value=50000.0,
                net_value=70000.0,
                net_share_count=700,
            )

            with patch.object(
                async_analyzer, "_process_transactions_data", return_value=expected_summary
            ):
                result = await async_analyzer.get_transactions_async("AAPL")

                # Verify the method was called using our tracker
                assert async_mock_provider.call_tracker["called"]
                assert isinstance(result, InsiderSummary)
                assert result.buy_count == 2
                assert result.sell_count == 1
                assert result.total_buy_value == pytest.approx(120000.0, abs=1e-9)
                assert result.total_sell_value == pytest.approx(50000.0, abs=1e-9)
                assert result.net_value == pytest.approx(70000.0, abs=1e-9)
                assert result.net_share_count == 700

    def test_get_transactions_batch(self, analyzer):
        """Test get_transactions_batch method"""
        with patch(
            "yahoofinance.analysis.insiders.is_us_ticker",
            side_effect=lambda x: x in ["AAPL", "MSFT"],
        ):
            result = analyzer.get_transactions_batch(["AAPL", "MSFT", "9988.HK"])

            assert len(result) == 3
            assert "AAPL" in result
            assert "MSFT" in result
            assert "9988.HK" in result
            assert isinstance(result["AAPL"], InsiderSummary)
            assert isinstance(result["MSFT"], InsiderSummary)
            assert isinstance(result["9988.HK"], InsiderSummary)
            assert len(result["9988.HK"].transactions) == 0

    @pytest.mark.asyncio
    async def test_get_transactions_batch_async(self, async_analyzer):
        """Test get_transactions_batch_async method"""
        with patch(
            "yahoofinance.analysis.insiders.is_us_ticker",
            side_effect=lambda x: x in ["AAPL", "MSFT"],
        ):
            result = await async_analyzer.get_transactions_batch_async(["AAPL", "MSFT", "9988.HK"])

            assert len(result) == 3
            assert "AAPL" in result
            assert "MSFT" in result
            assert "9988.HK" in result
            assert isinstance(result["AAPL"], InsiderSummary)
            assert isinstance(result["MSFT"], InsiderSummary)
            assert isinstance(result["9988.HK"], InsiderSummary)
            assert len(result["9988.HK"].transactions) == 0

    def test_analyze_insider_sentiment(self, analyzer):
        """Test analyze_insider_sentiment method"""
        with patch("yahoofinance.analysis.insiders.is_us_ticker", return_value=True):
            result = analyzer.analyze_insider_sentiment("AAPL")

            assert isinstance(result, dict)
            assert "sentiment" in result
            assert "confidence" in result
            assert "buy_count" in result
            assert "sell_count" in result
            assert "net_value" in result
            assert result["sentiment"] == "BULLISH"
            assert result["confidence"] == "MEDIUM"

    @pytest.mark.asyncio
    async def test_analyze_insider_sentiment_async(self, async_analyzer):
        """Test analyze_insider_sentiment_async method"""
        with patch("yahoofinance.analysis.insiders.is_us_ticker", return_value=True):
            # Create a mock summary
            mock_summary = InsiderSummary(
                transactions=[],
                buy_count=4,
                sell_count=1,
                total_buy_value=1500000.0,
                total_sell_value=200000.0,
                net_value=1300000.0,
                net_share_count=10000,
            )

            # Patch the get_transactions_async method
            with patch.object(async_analyzer, "get_transactions_async", return_value=mock_summary):
                result = await async_analyzer.analyze_insider_sentiment_async("AAPL")

                assert isinstance(result, dict)
                assert "sentiment" in result
                assert "confidence" in result
                assert "buy_count" in result
                assert "sell_count" in result
                assert "net_value" in result
                # The values depend on the _calculate_sentiment_metrics method's implementation
                # Just ensuring they have valid values
                assert result["sentiment"] in ["BULLISH", "BEARISH", "NEUTRAL"]
                assert result["confidence"] in ["HIGH", "MEDIUM", "LOW"]
