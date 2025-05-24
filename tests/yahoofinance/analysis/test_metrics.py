from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from yahoofinance.analysis.metrics import PriceData, PriceTarget, PricingAnalyzer


# Sample ticker data for testing
SAMPLE_TICKER_INFO = {
    "price": 150.0,
    "change": 2.5,
    "change_percent": 1.69,
    "volume": 75000000,
    "average_volume": 70000000,
    "target_price": 180.0,
    "median_target_price": 175.0,
    "highest_target_price": 200.0,
    "lowest_target_price": 150.0,
    "upside": 20.0,
    "analyst_count": 32,
    "pe_ratio": 25.5,
    "forward_pe": 22.3,
    "peg_ratio": 1.5,
    "price_to_book": 15.2,
    "price_to_sales": 6.8,
    "ev_to_ebitda": 18.4,
    "dividend_yield": 0.65,
    "dividend_rate": 0.92,
    "ex_dividend_date": "2024-01-05",
    "earnings_growth": 12.5,
    "revenue_growth": 8.2,
    "beta": 1.2,
    "short_percent": 0.8,
    "market_cap": 2500000000000,
    "market_cap_fmt": "2.5T",
    "enterprise_value": 2700000000000,
    "float_shares": 16000000000,
    "shares_outstanding": 16500000000,
    "high_52week": 175.0,
    "low_52week": 125.0,
    "from_high": -14.29,
    "from_low": 20.0,
}


@pytest.fixture
def mock_provider():
    provider = Mock()
    provider.get_ticker_info = Mock(return_value=SAMPLE_TICKER_INFO)
    provider.batch_get_ticker_info = Mock(
        return_value={
            "AAPL": SAMPLE_TICKER_INFO,
            "MSFT": SAMPLE_TICKER_INFO,
        }
    )
    return provider


@pytest.fixture
def async_mock_provider():
    provider = Mock()

    # Create a spy to track calls
    call_tracker = {"called": False}

    async def mock_get_ticker_info(*args, **kwargs):
        call_tracker["called"] = True
        call_tracker["args"] = args
        call_tracker["kwargs"] = kwargs
        return SAMPLE_TICKER_INFO

    async def mock_batch_get_ticker_info(*args, **kwargs):
        call_tracker["batch_called"] = True
        call_tracker["batch_args"] = args
        call_tracker["batch_kwargs"] = kwargs
        return {
            "AAPL": SAMPLE_TICKER_INFO,
            "MSFT": SAMPLE_TICKER_INFO,
        }

    provider.get_ticker_info = mock_get_ticker_info
    provider.batch_get_ticker_info = mock_batch_get_ticker_info
    provider.call_tracker = call_tracker
    return provider


@pytest.fixture
def analyzer(mock_provider):
    return PricingAnalyzer(mock_provider)


@pytest.fixture
def async_analyzer(async_mock_provider):
    analyzer = PricingAnalyzer(async_mock_provider)
    # Force is_async to True for testing
    analyzer.is_async = True
    return analyzer


class TestPriceData:
    """Tests for PriceData class"""

    def test_price_data_init(self):
        """Test PriceData initialization"""
        data = PriceData(
            price=150.0,
            change=2.5,
            change_percent=1.69,
            volume=75000000,
            average_volume=70000000,
            volume_ratio=1.07,
            high_52week=175.0,
            low_52week=125.0,
            from_high=-14.29,
            from_low=20.0,
        )

        assert data.price == pytest.approx(150.0, abs=1e-9)
        assert data.change == pytest.approx(2.5, abs=1e-9)
        assert data.change_percent == pytest.approx(1.69, abs=1e-9)
        assert data.volume == 75000000
        assert data.average_volume == 70000000
        assert data.volume_ratio == pytest.approx(1.07, abs=1e-9)
        assert data.high_52week == pytest.approx(175.0, abs=1e-9)
        assert data.low_52week == pytest.approx(125.0, abs=1e-9)
        assert data.from_high == pytest.approx(-14.29, abs=1e-9)
        assert data.from_low == pytest.approx(20.0, abs=1e-9)


class TestPriceTarget:
    """Tests for PriceTarget class"""

    def test_price_target_init(self):
        """Test PriceTarget initialization"""
        target = PriceTarget(
            average=180.0, median=175.0, high=200.0, low=150.0, upside=20.0, analyst_count=32
        )

        assert target.average == pytest.approx(180.0, abs=1e-9)
        assert target.median == pytest.approx(175.0, abs=1e-9)
        assert target.high == pytest.approx(200.0, abs=1e-9)
        assert target.low == pytest.approx(150.0, abs=1e-9)
        assert target.upside == pytest.approx(20.0, abs=1e-9)
        assert target.analyst_count == 32


class TestPricingAnalyzer:
    """Tests for PricingAnalyzer class"""

    def test_init(self, mock_provider):
        """Test analyzer initialization"""
        analyzer = PricingAnalyzer(mock_provider)
        assert analyzer.provider == mock_provider
        assert not analyzer.is_async

    def test_process_price_data(self, analyzer):
        """Test _process_price_data method"""
        result = analyzer._process_price_data(SAMPLE_TICKER_INFO)

        assert isinstance(result, PriceData)
        assert result.price == pytest.approx(150.0, abs=1e-9)
        assert result.change == pytest.approx(2.5, abs=1e-9)
        assert result.change_percent == pytest.approx(1.69, abs=1e-9)
        assert result.volume == 75000000
        assert result.average_volume == 70000000
        assert result.volume_ratio == pytest.approx(75000000 / 70000000, abs=1e-9)
        assert result.high_52week == pytest.approx(175.0, abs=1e-9)
        assert result.low_52week == pytest.approx(125.0, abs=1e-9)
        assert result.from_high == pytest.approx(-14.29, abs=1e-9)
        assert result.from_low == pytest.approx(20.0, abs=1e-9)

    def test_process_price_target(self, analyzer):
        """Test _process_price_target method"""
        result = analyzer._process_price_target(SAMPLE_TICKER_INFO)

        assert isinstance(result, PriceTarget)
        assert result.average == pytest.approx(180.0, abs=1e-9)
        assert result.median == pytest.approx(175.0, abs=1e-9)
        assert result.high == pytest.approx(200.0, abs=1e-9)
        assert result.low == pytest.approx(150.0, abs=1e-9)
        assert result.upside == pytest.approx(20.0, abs=1e-9)
        assert result.analyst_count == 32

    def test_get_price_data(self, analyzer, mock_provider):
        """Test get_price_data method"""
        result = analyzer.get_price_data("AAPL")

        mock_provider.get_ticker_info.assert_called_once_with("AAPL")
        assert isinstance(result, PriceData)
        assert result.price == pytest.approx(150.0, abs=1e-9)
        assert result.volume_ratio is not None

    def test_get_price_target(self, analyzer, mock_provider):
        """Test get_price_target method"""
        result = analyzer.get_price_target("AAPL")

        mock_provider.get_ticker_info.assert_called_once_with("AAPL")
        assert isinstance(result, PriceTarget)
        assert result.average == pytest.approx(180.0, abs=1e-9)
        assert result.analyst_count == 32

    def test_get_all_metrics(self, analyzer, mock_provider):
        """Test get_all_metrics method"""
        result = analyzer.get_all_metrics("AAPL")

        mock_provider.get_ticker_info.assert_called_once_with("AAPL")
        assert isinstance(result, dict)
        assert result["price"] == pytest.approx(150.0, abs=1e-9)
        assert result["pe_ratio"] == pytest.approx(25.5, abs=1e-9)
        assert result["peg_ratio"] == pytest.approx(1.5, abs=1e-9)
        assert result["beta"] == pytest.approx(1.2, abs=1e-9)
        assert result["market_cap"] == pytest.approx(2500000000000, abs=1e-9)

    def test_get_metrics_batch(self, analyzer, mock_provider):
        """Test get_metrics_batch method"""
        result = analyzer.get_metrics_batch(["AAPL", "MSFT"])

        mock_provider.batch_get_ticker_info.assert_called_once_with(["AAPL", "MSFT"])
        assert isinstance(result, dict)
        assert "AAPL" in result
        assert "MSFT" in result
        assert isinstance(result["AAPL"], dict)
        assert result["AAPL"]["price"] == pytest.approx(150.0, abs=1e-9)

    @pytest.mark.asyncio
    async def test_get_price_data_async(self, async_analyzer, async_mock_provider):
        """Test get_price_data_async method"""
        with patch.object(
            async_analyzer, "_process_price_data", return_value=PriceData(price=150.0)
        ):
            result = await async_analyzer.get_price_data_async("AAPL")

            assert async_mock_provider.call_tracker["called"]
            assert isinstance(result, PriceData)
            assert result.price == pytest.approx(150.0, abs=1e-9)

    @pytest.mark.asyncio
    async def test_get_price_target_async(self, async_analyzer, async_mock_provider):
        """Test get_price_target_async method"""
        with patch.object(
            async_analyzer, "_process_price_target", return_value=PriceTarget(average=180.0)
        ):
            result = await async_analyzer.get_price_target_async("AAPL")

            assert async_mock_provider.call_tracker["called"]
            assert isinstance(result, PriceTarget)
            assert result.average == pytest.approx(180.0, abs=1e-9)

    @pytest.mark.asyncio
    async def test_get_all_metrics_async(self, async_analyzer, async_mock_provider):
        """Test get_all_metrics_async method"""
        with patch.object(async_analyzer, "_extract_all_metrics", return_value={"price": 150.0}):
            result = await async_analyzer.get_all_metrics_async("AAPL")

            assert async_mock_provider.call_tracker["called"]
            assert isinstance(result, dict)
            assert result["price"] == pytest.approx(150.0, abs=1e-9)

    @pytest.mark.asyncio
    async def test_get_metrics_batch_async(self, async_analyzer, async_mock_provider):
        """Test get_metrics_batch_async method"""
        with patch.object(async_analyzer, "_extract_metrics", return_value={"price": 150.0}):
            result = await async_analyzer.get_metrics_batch_async(["AAPL", "MSFT"])

            assert async_mock_provider.call_tracker["batch_called"]
            assert isinstance(result, dict)
            assert "AAPL" in result
            assert "MSFT" in result
            assert result["AAPL"]["price"] == pytest.approx(150.0, abs=1e-9)
