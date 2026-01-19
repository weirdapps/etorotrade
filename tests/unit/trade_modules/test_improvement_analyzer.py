"""Tests for the Improvement Analyzer module."""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import pandas as pd

from trade_modules.improvement_analyzer import (
    ImprovementAnalyzer,
    MetricEffectiveness,
    DataQualityReport,
    ImprovementSuggestion,
    SuggestionsDocument,
    run_analysis,
)


class TestImprovementAnalyzer:
    """Tests for ImprovementAnalyzer class."""

    @pytest.fixture
    def temp_market_csv(self):
        """Create temporary market.csv for testing."""
        data = {
            "TKR": ["AAPL", "MSFT", "GOOG", "TSLA", "NVDA", "AMD", "INTC", "META"],
            "NAME": ["Apple", "Microsoft", "Alphabet", "Tesla", "Nvidia", "AMD", "Intel", "Meta"],
            "CAP": ["3.8T", "3.4T", "2.0T", "800B", "4.5T", "200B", "150B", "1.5T"],
            "BS": ["B", "B", "H", "S", "B", "H", "S", "H"],
            "UP%": ["15%", "20%", "5%", "-10%", "25%", "8%", "-5%", "10%"],
            "%B": ["90%", "85%", "70%", "40%", "95%", "75%", "45%", "80%"],
            "EXR": ["13.5%", "17.0%", "3.5%", "-4.0%", "23.8%", "6.0%", "-2.3%", "8.0%"],
            "PEF": ["28", "32", "25", "200", "45", "35", "15", "22"],
            "PEG": ["2.5", "2.0", "1.8", "8.0", "1.5", "1.2", "1.0", "1.6"],
            "SI": ["0.8%", "0.5%", "1.0%", "3.0%", "0.3%", "2.0%", "4.0%", "1.5%"],
            "DE": ["100%", "50%", "20%", "40%", "15%", "30%", "80%", "25%"],
            "ROE": ["150%", "40%", "30%", "5%", "100%", "15%", "10%", "25%"],
            "FCF": ["5%", "4%", "3%", "1%", "2%", "2.5%", "3.5%", "4%"],
            "52W": ["95%", "90%", "85%", "60%", "98%", "80%", "50%", "88%"],
            "#A": ["30", "35", "25", "40", "45", "20", "25", "30"],
            "TGT": ["200", "350", "180", "200", "250", "180", "40", "400"],
        }
        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = Path(f.name)
        yield temp_path
        temp_path.unlink()

    def test_load_market_data(self, temp_market_csv):
        """Test loading market data."""
        analyzer = ImprovementAnalyzer(market_csv_path=temp_market_csv)
        df = analyzer.load_market_data()

        assert len(df) == 8
        assert "TKR" in df.columns
        assert "BS" in df.columns

    def test_analyze_metric_effectiveness(self, temp_market_csv):
        """Test metric effectiveness analysis."""
        analyzer = ImprovementAnalyzer(market_csv_path=temp_market_csv)
        results = analyzer.analyze_metric_effectiveness()

        assert len(results) > 0
        assert all(isinstance(r, MetricEffectiveness) for r in results)

        # Check that results are sorted by ratio
        ratios = [r.ratio for r in results if r.ratio > 0]
        assert ratios == sorted(ratios, reverse=True)

    def test_analyze_data_quality(self, temp_market_csv):
        """Test data quality analysis."""
        analyzer = ImprovementAnalyzer(market_csv_path=temp_market_csv)
        report = analyzer.analyze_data_quality()

        assert isinstance(report, DataQualityReport)
        assert report.total_stocks == 8
        assert report.inconclusive_count >= 0
        assert len(report.by_tier) > 0

    def test_generate_suggestions(self, temp_market_csv):
        """Test suggestion generation."""
        analyzer = ImprovementAnalyzer(market_csv_path=temp_market_csv)
        metrics = analyzer.analyze_metric_effectiveness()
        data_quality = analyzer.analyze_data_quality()
        suggestions = analyzer.generate_suggestions(metrics, data_quality)

        assert isinstance(suggestions, list)
        assert all(isinstance(s, ImprovementSuggestion) for s in suggestions)

        # Suggestions should be sorted by priority
        priorities = [s.priority for s in suggestions]
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        priority_nums = [priority_order.get(p, 3) for p in priorities]
        assert priority_nums == sorted(priority_nums)

    def test_generate_key_insights(self, temp_market_csv):
        """Test key insights generation."""
        analyzer = ImprovementAnalyzer(market_csv_path=temp_market_csv)
        metrics = analyzer.analyze_metric_effectiveness()
        data_quality = analyzer.analyze_data_quality()
        suggestions = analyzer.generate_suggestions(metrics, data_quality)
        insights = analyzer.generate_key_insights(metrics, data_quality, suggestions)

        assert isinstance(insights, list)
        assert len(insights) > 0
        assert all(isinstance(i, str) for i in insights)

    def test_generate_suggestions_document(self, temp_market_csv):
        """Test full document generation."""
        analyzer = ImprovementAnalyzer(market_csv_path=temp_market_csv)
        doc = analyzer.generate_suggestions_document()

        assert isinstance(doc, SuggestionsDocument)
        assert doc.generated_at is not None
        assert len(doc.metric_analysis) > 0
        assert doc.data_quality is not None
        assert isinstance(doc.suggestions, list)
        assert len(doc.key_insights) > 0

    def test_save_document(self, temp_market_csv):
        """Test document saving."""
        analyzer = ImprovementAnalyzer(market_csv_path=temp_market_csv)
        doc = analyzer.generate_suggestions_document()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_path = Path(f.name)

        try:
            saved_path = analyzer.save_document(doc, output_path)
            assert saved_path.exists()
            content = saved_path.read_text()
            assert "# Automated Improvement Suggestions" in content
            assert "Executive Summary" in content
        finally:
            output_path.unlink()


class TestMetricEffectiveness:
    """Tests for MetricEffectiveness dataclass."""

    def test_creation(self):
        """Test MetricEffectiveness creation."""
        metric = MetricEffectiveness(
            metric_name="EXR",
            buy_median=15.0,
            sell_median=5.0,
            hold_median=10.0,
            ratio=3.0,
            is_effective=True,
            recommendation="Keep current thresholds",
        )

        assert metric.metric_name == "EXR"
        assert abs(metric.ratio - 3.0) < 0.01
        assert metric.is_effective is True


class TestDataQualityReport:
    """Tests for DataQualityReport dataclass."""

    def test_creation(self):
        """Test DataQualityReport creation."""
        report = DataQualityReport(
            total_stocks=5000,
            inconclusive_count=2500,
            inconclusive_rate=50.0,
            by_tier={"MEGA": {"total": 50, "inconclusive": 5, "rate": 10.0}},
            by_region={"US": {"total": 3000, "inconclusive": 1500, "rate": 50.0}},
            primary_causes=[("Low analyst count", 1500, 60.0)],
            recommendations=["Improve data sourcing"],
        )

        assert report.total_stocks == 5000
        assert abs(report.inconclusive_rate - 50.0) < 0.01


class TestImprovementSuggestion:
    """Tests for ImprovementSuggestion dataclass."""

    def test_creation(self):
        """Test ImprovementSuggestion creation."""
        suggestion = ImprovementSuggestion(
            category="metric",
            priority="HIGH",
            title="Remove PEG",
            description="PEG has no predictive power",
            evidence="Ratio: 1.0x",
            action="Remove from criteria",
            expected_impact="Reduce noise",
        )

        assert suggestion.priority == "HIGH"
        assert suggestion.category == "metric"
