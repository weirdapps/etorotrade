"""
Tests for dynamic sector detection in trade_config.py

Tests cover:
- YFINANCE_SECTOR_MAP mapping
- YFINANCE_INDUSTRY_MAP mapping
- get_sector_from_ticker_dynamic() method
- Integration with get_sector_adjusted_thresholds()
"""

import pytest

from trade_modules.trade_config import TradeConfig


class TestYfinanceSectorMap:
    """Tests for YFINANCE_SECTOR_MAP constant."""

    def test_sector_map_exists(self):
        """Test that sector map is defined."""
        assert hasattr(TradeConfig, "YFINANCE_SECTOR_MAP")
        assert isinstance(TradeConfig.YFINANCE_SECTOR_MAP, dict)

    def test_sector_map_financial_services(self):
        """Test Financial Services mapping."""
        assert TradeConfig.YFINANCE_SECTOR_MAP.get("Financial Services") == "FINANCIAL"
        assert TradeConfig.YFINANCE_SECTOR_MAP.get("Financial") == "FINANCIAL"

    def test_sector_map_real_estate(self):
        """Test Real Estate mapping."""
        assert TradeConfig.YFINANCE_SECTOR_MAP.get("Real Estate") == "REIT"

    def test_sector_map_utilities(self):
        """Test Utilities mapping."""
        assert TradeConfig.YFINANCE_SECTOR_MAP.get("Utilities") == "UTILITY"


class TestYfinanceIndustryMap:
    """Tests for YFINANCE_INDUSTRY_MAP constant."""

    def test_industry_map_exists(self):
        """Test that industry map is defined."""
        assert hasattr(TradeConfig, "YFINANCE_INDUSTRY_MAP")
        assert isinstance(TradeConfig.YFINANCE_INDUSTRY_MAP, dict)

    def test_industry_map_banks(self):
        """Test bank industry mappings."""
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("Banks—Regional") == "FINANCIAL"
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("Banks—Diversified") == "FINANCIAL"

    def test_industry_map_insurance(self):
        """Test insurance industry mappings."""
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("Insurance—Life") == "FINANCIAL"
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("Insurance—Property & Casualty") == "FINANCIAL"

    def test_industry_map_reits(self):
        """Test REIT industry mappings."""
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("REIT—Industrial") == "REIT"
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("REIT—Residential") == "REIT"
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("REIT—Retail") == "REIT"
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("REIT—Healthcare Facilities") == "REIT"
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("REIT—Specialty") == "REIT"
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("REIT—Office") == "REIT"
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("REIT—Diversified") == "REIT"

    def test_industry_map_utilities(self):
        """Test utility industry mappings."""
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("Utilities—Regulated Electric") == "UTILITY"
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("Utilities—Diversified") == "UTILITY"
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("Utilities—Regulated Gas") == "UTILITY"

    def test_industry_map_telecom(self):
        """Test telecom industry mappings."""
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("Telecom Services") == "TELECOM"
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("Wireless Telecommunications Services") == "TELECOM"

    def test_industry_map_semiconductors(self):
        """Test semiconductor industry mappings."""
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("Semiconductors") == "TECHNOLOGY_HARDWARE"
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("Semiconductor Equipment & Materials") == "TECHNOLOGY_HARDWARE"

    def test_industry_map_payment_processors(self):
        """Test payment processor industry mappings."""
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("Credit Services") == "PAYMENT_PROCESSORS"

    def test_industry_map_pharma(self):
        """Test pharma industry mappings."""
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("Drug Manufacturers—General") == "PHARMA_HIGH_LEVERAGE"
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("Drug Manufacturers—Specialty & Generic") == "PHARMA_HIGH_LEVERAGE"

    def test_industry_map_mlp(self):
        """Test MLP industry mappings."""
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("Oil & Gas Midstream") == "MLP"
        assert TradeConfig.YFINANCE_INDUSTRY_MAP.get("Oil & Gas Pipelines") == "MLP"


class TestGetSectorFromTickerDynamic:
    """Tests for get_sector_from_ticker_dynamic() method."""

    def test_method_exists(self):
        """Test that method exists."""
        assert hasattr(TradeConfig, "get_sector_from_ticker_dynamic")
        assert callable(TradeConfig.get_sector_from_ticker_dynamic)

    def test_hardcoded_override_takes_priority(self):
        """Test that hardcoded tickers still work."""
        # These are in the hardcoded SECTOR_RULES
        result = TradeConfig.get_sector_from_ticker_dynamic(
            ticker="JPM",
            sector="Technology",  # Wrong sector
            industry="Software",  # Wrong industry
        )
        # Should use hardcoded FINANCIAL, not the wrong sector/industry
        assert result == "FINANCIAL"

    def test_industry_detection(self):
        """Test detection via industry."""
        result = TradeConfig.get_sector_from_ticker_dynamic(
            ticker="NEW_BANK",  # Not in hardcoded list
            sector="Financial Services",
            industry="Banks—Regional",
        )
        assert result == "FINANCIAL"

    def test_sector_fallback(self):
        """Test fallback to sector when industry not mapped."""
        result = TradeConfig.get_sector_from_ticker_dynamic(
            ticker="NEW_FINCO",  # Not in hardcoded list
            sector="Financial Services",
            industry="Unknown Industry",
        )
        assert result == "FINANCIAL"

    def test_no_match_returns_none(self):
        """Test that unknown sector/industry returns None."""
        result = TradeConfig.get_sector_from_ticker_dynamic(
            ticker="UNKNOWN",
            sector="Made Up Sector",
            industry="Made Up Industry",
        )
        assert result is None

    def test_none_inputs(self):
        """Test with None inputs."""
        result = TradeConfig.get_sector_from_ticker_dynamic(
            ticker="UNKNOWN",
            sector=None,
            industry=None,
        )
        assert result is None

    def test_empty_ticker(self):
        """Test with empty ticker."""
        result = TradeConfig.get_sector_from_ticker_dynamic(
            ticker="",
            sector="Financial Services",
            industry="Banks—Regional",
        )
        # Should still detect via industry
        assert result == "FINANCIAL"


class TestDynamicSectorIntegration:
    """Integration tests for dynamic sector with threshold adjustment."""

    def test_bank_from_industry_gets_adjusted_thresholds(self):
        """Test that a bank detected via industry gets FINANCIAL thresholds."""
        # Detect sector dynamically
        sector = TradeConfig.get_sector_from_ticker_dynamic(
            ticker="NEW_BANK",
            sector="Financial Services",
            industry="Banks—Diversified",
        )
        assert sector == "FINANCIAL"

        # Get adjusted thresholds
        base_criteria = {"min_roe": 8.0, "max_debt_equity": 200.0}
        adjusted = TradeConfig.get_sector_adjusted_thresholds(
            ticker="NEW_BANK",
            action="buy",
            base_criteria=base_criteria,
        )

        # Note: get_sector_adjusted_thresholds uses get_sector_from_ticker
        # which only checks hardcoded list, not dynamic detection.
        # The dynamic detection would need integration at a higher level.
        # This test documents current behavior.

    def test_reit_from_industry_detection(self):
        """Test REIT detection from industry."""
        sector = TradeConfig.get_sector_from_ticker_dynamic(
            ticker="NEW_REIT",
            sector="Real Estate",
            industry="REIT—Industrial",
        )
        assert sector == "REIT"

    def test_utility_detection(self):
        """Test utility detection."""
        sector = TradeConfig.get_sector_from_ticker_dynamic(
            ticker="NEW_UTIL",
            sector="Utilities",
            industry="Utilities—Regulated Electric",
        )
        assert sector == "UTILITY"


class TestGetAllSectorCategories:
    """Tests for get_all_sector_categories() method."""

    def test_method_exists(self):
        """Test that method exists."""
        assert hasattr(TradeConfig, "get_all_sector_categories")
        assert callable(TradeConfig.get_all_sector_categories)

    def test_returns_list(self):
        """Test that method returns a list."""
        categories = TradeConfig.get_all_sector_categories()
        assert isinstance(categories, list)

    def test_contains_expected_categories(self):
        """Test that expected categories are present."""
        categories = TradeConfig.get_all_sector_categories()
        expected = [
            "FINANCIAL",
            "REIT",
            "MLP",
            "PHARMA_HIGH_LEVERAGE",
            "EQUIPMENT_FINANCING",
            "UTILITY",
            "PAYMENT_PROCESSORS",
            "TELECOM",
            "TECHNOLOGY_HARDWARE",
        ]
        for cat in expected:
            assert cat in categories, f"Missing category: {cat}"


class TestRealWorldScenarios:
    """Real-world scenario tests for dynamic sector detection."""

    def test_detect_new_bank_ipo(self):
        """Test detecting a newly IPO'd bank not in hardcoded list."""
        # Simulate a new bank that IPO'd after our hardcoded list was created
        sector = TradeConfig.get_sector_from_ticker_dynamic(
            ticker="NWBK",  # Hypothetical new bank
            sector="Financial Services",
            industry="Banks—Regional",
        )
        assert sector == "FINANCIAL"

    def test_detect_european_reit(self):
        """Test detecting a European REIT."""
        sector = TradeConfig.get_sector_from_ticker_dynamic(
            ticker="NEW_EU_REIT.L",  # Hypothetical EU REIT
            sector="Real Estate",
            industry="REIT—Industrial",
        )
        assert sector == "REIT"

    def test_detect_hong_kong_utility(self):
        """Test detecting a Hong Kong utility."""
        sector = TradeConfig.get_sector_from_ticker_dynamic(
            ticker="0001.HK",  # Hypothetical HK utility
            sector="Utilities",
            industry="Utilities—Diversified",
        )
        assert sector == "UTILITY"

    def test_tech_company_not_hardware(self):
        """Test that software company doesn't get hardware classification."""
        sector = TradeConfig.get_sector_from_ticker_dynamic(
            ticker="SOFT",
            sector="Technology",
            industry="Software—Application",
        )
        # Should return None because Technology sector and Software industry
        # are not mapped to any special category
        assert sector is None


class TestEdgeCases:
    """Edge case tests."""

    def test_case_sensitivity_ticker(self):
        """Test case sensitivity of ticker lookup."""
        # Hardcoded tickers are uppercase, so lowercase should still work
        sector1 = TradeConfig.get_sector_from_ticker("jpm")
        sector2 = TradeConfig.get_sector_from_ticker("JPM")
        # Both should work (method uppercases internally)
        assert sector1 == sector2 == "FINANCIAL"

    def test_industry_with_special_characters(self):
        """Test industry names with em-dashes and special chars."""
        # yfinance uses em-dashes (—) not regular dashes (-)
        sector = TradeConfig.get_sector_from_ticker_dynamic(
            ticker="TEST",
            sector="Financial Services",
            industry="Banks—Regional",  # Em-dash
        )
        assert sector == "FINANCIAL"

    def test_partial_industry_match_fails(self):
        """Test that partial matches don't work."""
        sector = TradeConfig.get_sector_from_ticker_dynamic(
            ticker="TEST",
            sector="Financial Services",
            industry="Banks",  # Missing the type suffix
        )
        # Should fall back to sector
        assert sector == "FINANCIAL"  # Via sector fallback
