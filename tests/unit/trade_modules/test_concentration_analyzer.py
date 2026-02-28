"""Tests for the concentration analyzer module."""

import pandas as pd
import pytest

from trade_modules.concentration_analyzer import (
    ConcentrationWarning,
    analyze_concentration,
    format_concentration_warnings,
    get_diversification_score,
    _infer_region_from_ticker,
)


class TestConcentrationWarning:
    """Tests for ConcentrationWarning class."""

    def test_init(self):
        w = ConcentrationWarning("sector", "Technology", 0.5, 5, 10, ["AAPL", "MSFT"])
        assert w.warning_type == "sector"
        assert w.concentrated_value == "Technology"
        assert w.percentage == 0.5
        assert w.count == 5
        assert w.total == 10
        assert w.tickers == ["AAPL", "MSFT"]

    def test_str(self):
        w = ConcentrationWarning("sector", "Technology", 0.5, 5, 10, ["AAPL"])
        s = str(w)
        assert "SECTOR CONCENTRATION" in s
        assert "5/10" in s
        assert "Technology" in s

    def test_to_dict(self):
        w = ConcentrationWarning("region", "US", 0.7, 7, 10, ["AAPL", "MSFT"])
        d = w.to_dict()
        assert d["warning_type"] == "region"
        assert d["concentrated_value"] == "US"
        assert d["percentage"] == 0.7
        assert d["count"] == 7
        assert d["total"] == 10
        assert d["tickers"] == ["AAPL", "MSFT"]


class TestAnalyzeConcentration:
    """Tests for analyze_concentration function."""

    def _make_df(self, tickers, sectors, regions, signals):
        return pd.DataFrame({
            "ticker": tickers,
            "sector": sectors,
            "region": regions,
            "BS": signals,
        })

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        assert analyze_concentration(df) == []

    def test_missing_signal_column(self):
        df = pd.DataFrame({"ticker": ["AAPL"], "sector": ["Tech"]})
        assert analyze_concentration(df) == []

    def test_too_few_signals(self):
        df = self._make_df(["AAPL", "MSFT"], ["Tech", "Tech"], ["US", "US"], ["B", "S"])
        # Only 1 BUY signal, below min_signals=3
        assert analyze_concentration(df) == []

    def test_no_concentration(self):
        df = self._make_df(
            ["AAPL", "MSFT", "JNJ", "XOM"],
            ["Tech", "Tech", "Health", "Energy"],
            ["US", "US", "US", "US"],
            ["B", "B", "B", "B"],
        )
        # 50% Tech, 25% Health, 25% Energy - max_sector=0.40 triggers for Tech
        warnings = analyze_concentration(df, max_sector_concentration=0.60)
        sector_warnings = [w for w in warnings if w.warning_type == "sector"]
        assert len(sector_warnings) == 0

    def test_sector_concentration_detected(self):
        df = self._make_df(
            ["AAPL", "MSFT", "GOOGL", "JNJ"],
            ["Tech", "Tech", "Tech", "Health"],
            ["US", "US", "US", "US"],
            ["B", "B", "B", "B"],
        )
        warnings = analyze_concentration(df, max_sector_concentration=0.40)
        sector_warnings = [w for w in warnings if w.warning_type == "sector"]
        assert len(sector_warnings) == 1
        assert sector_warnings[0].concentrated_value == "Tech"
        assert sector_warnings[0].count == 3
        assert sector_warnings[0].total == 4
        assert set(sector_warnings[0].tickers) == {"AAPL", "MSFT", "GOOGL"}

    def test_region_concentration_detected(self):
        df = self._make_df(
            ["AAPL", "MSFT", "GOOGL", "SAP.DE"],
            ["Tech", "Tech", "Tech", "Tech"],
            ["US", "US", "US", "EU"],
            ["B", "B", "B", "B"],
        )
        warnings = analyze_concentration(
            df, max_sector_concentration=1.0, max_region_concentration=0.60
        )
        region_warnings = [w for w in warnings if w.warning_type == "region"]
        assert len(region_warnings) == 1
        assert region_warnings[0].concentrated_value == "US"

    def test_filters_by_signal(self):
        df = self._make_df(
            ["AAPL", "MSFT", "GOOGL", "JNJ", "XOM"],
            ["Tech", "Tech", "Tech", "Health", "Energy"],
            ["US", "US", "US", "US", "US"],
            ["B", "B", "B", "S", "S"],
        )
        # Only 3 BUY signals, all Tech - should trigger
        warnings = analyze_concentration(df, max_sector_concentration=0.40)
        sector_warnings = [w for w in warnings if w.warning_type == "sector"]
        assert len(sector_warnings) == 1

    def test_region_inferred_from_ticker(self):
        """When no region column, regions inferred from ticker suffixes."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOGL", "0700.HK"],
            "sector": ["Tech", "Tech", "Tech", "Tech"],
            "BS": ["B", "B", "B", "B"],
        })
        warnings = analyze_concentration(
            df, max_sector_concentration=1.0, max_region_concentration=0.50
        )
        region_warnings = [w for w in warnings if w.warning_type == "region"]
        assert len(region_warnings) == 1
        assert region_warnings[0].concentrated_value == "US"


class TestInferRegionFromTicker:
    """Tests for _infer_region_from_ticker."""

    def test_us_ticker(self):
        assert _infer_region_from_ticker("AAPL") == "US"

    def test_hk_ticker(self):
        assert _infer_region_from_ticker("0700.HK") == "HK"

    def test_eu_tickers(self):
        assert _infer_region_from_ticker("SAP.DE") == "EU"
        assert _infer_region_from_ticker("HSBA.L") == "EU"
        assert _infer_region_from_ticker("TTE.PA") == "EU"
        assert _infer_region_from_ticker("ASML.AS") == "EU"

    def test_empty(self):
        assert _infer_region_from_ticker("") == "unknown"

    def test_case_insensitive(self):
        assert _infer_region_from_ticker("sap.de") == "EU"


class TestFormatConcentrationWarnings:
    """Tests for format_concentration_warnings."""

    def test_empty_warnings(self):
        assert format_concentration_warnings([]) == ""

    def test_single_warning(self):
        w = ConcentrationWarning("sector", "Technology", 0.75, 3, 4, ["AAPL", "MSFT", "GOOGL"])
        result = format_concentration_warnings([w])
        assert "CONCENTRATION WARNINGS" in result
        assert "Technology" in result
        assert "AAPL" in result

    def test_many_tickers_truncated(self):
        tickers = [f"T{i}" for i in range(15)]
        w = ConcentrationWarning("sector", "Tech", 0.5, 15, 30, tickers)
        result = format_concentration_warnings([w])
        assert "... and 5 more" in result


class TestGetDiversificationScore:
    """Tests for get_diversification_score."""

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        score, desc = get_diversification_score(df)
        assert score == 0.0
        assert desc == "No data"

    def test_too_few_signals(self):
        df = pd.DataFrame({"BS": ["B"], "sector": ["Tech"]})
        score, desc = get_diversification_score(df)
        assert score == 100.0
        assert "Too few" in desc

    def test_perfectly_concentrated(self):
        df = pd.DataFrame({
            "BS": ["B", "B", "B"],
            "sector": ["Tech", "Tech", "Tech"],
        })
        score, desc = get_diversification_score(df)
        assert score == 0.0
        assert "concentrated" in desc.lower()

    def test_well_diversified(self):
        df = pd.DataFrame({
            "BS": ["B"] * 10,
            "sector": ["Tech", "Health", "Finance", "Energy", "Consumer",
                       "Industrial", "Materials", "Utilities", "Real Estate", "Telecom"],
        })
        score, desc = get_diversification_score(df)
        assert score >= 70
        assert "diversified" in desc.lower()

    def test_moderately_diversified(self):
        df = pd.DataFrame({
            "BS": ["B"] * 6,
            "sector": ["Tech", "Tech", "Tech", "Health", "Finance", "Energy"],
        })
        score, desc = get_diversification_score(df)
        assert 30 <= score < 70
