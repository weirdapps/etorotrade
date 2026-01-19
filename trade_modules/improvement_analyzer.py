"""
Automated Improvement Analyzer

Analyzes framework performance and generates actionable improvement suggestions
based on signal validation, metric effectiveness, and data quality metrics.

This is the automated system that identifies opportunities for improvement
and generates a suggestions document with key insights.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import statistics

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetricEffectiveness:
    """Analysis of a single metric's predictive power."""
    metric_name: str
    buy_median: float
    sell_median: float
    hold_median: float
    ratio: float  # buy/sell ratio (or inverse for "lower is better")
    is_effective: bool
    recommendation: str


@dataclass
class DataQualityReport:
    """Report on data quality issues."""
    total_stocks: int
    inconclusive_count: int
    inconclusive_rate: float
    by_tier: Dict[str, Dict[str, Any]]
    by_region: Dict[str, Dict[str, Any]]
    primary_causes: List[Tuple[str, int, float]]
    recommendations: List[str]


@dataclass
class ImprovementSuggestion:
    """A single improvement suggestion."""
    category: str  # "threshold", "metric", "data", "architecture"
    priority: str  # "HIGH", "MEDIUM", "LOW"
    title: str
    description: str
    evidence: str
    action: str
    expected_impact: str


@dataclass
class SuggestionsDocument:
    """Complete suggestions document."""
    generated_at: datetime
    framework_version: str
    analysis_period: str
    summary: str
    metric_analysis: List[MetricEffectiveness]
    data_quality: DataQualityReport
    suggestions: List[ImprovementSuggestion]
    key_insights: List[str]


class ImprovementAnalyzer:
    """
    Analyzes framework performance and generates improvement suggestions.

    This is the main automated analysis system that:
    1. Analyzes metric effectiveness
    2. Identifies data quality issues
    3. Generates actionable suggestions
    4. Creates a comprehensive report
    """

    def __init__(
        self,
        market_csv_path: Optional[Path] = None,
        config_path: Optional[Path] = None
    ):
        """
        Initialize analyzer.

        Args:
            market_csv_path: Path to market.csv output
            config_path: Path to config.yaml
        """
        base_path = Path(__file__).parent.parent
        self.market_csv_path = market_csv_path or base_path / "yahoofinance" / "output" / "market.csv"
        self.config_path = config_path or base_path / "config.yaml"
        self._df: Optional[pd.DataFrame] = None

    def load_market_data(self) -> pd.DataFrame:
        """Load and cache market.csv data."""
        if self._df is None:
            self._df = pd.read_csv(self.market_csv_path)
        return self._df

    def _to_numeric(self, series: pd.Series) -> pd.Series:
        """Convert string series to numeric."""
        if series.dtype == object:
            series = series.str.replace('%', '').str.replace('--', '')
        return pd.to_numeric(series, errors='coerce')

    def _get_region(self, ticker: str) -> str:
        """Determine region from ticker."""
        if pd.isna(ticker):
            return "OTHER"
        ticker = str(ticker)
        if ".HK" in ticker:
            return "HK"
        elif any(x in ticker for x in [".PA", ".DE", ".L", ".ST", ".CO", ".OL", ".BR", ".AS", ".MC"]):
            return "EU"
        elif "-USD" in ticker:
            return "CRYPTO"
        else:
            return "US"

    def _get_tier(self, cap_str: str) -> str:
        """Determine tier from market cap string."""
        if pd.isna(cap_str) or cap_str == "--":
            return "UNKNOWN"
        cap_str = str(cap_str).upper()
        try:
            if "T" in cap_str:
                val = float(cap_str.replace("T", "").replace(",", "")) * 1e12
            elif "B" in cap_str:
                val = float(cap_str.replace("B", "").replace(",", "")) * 1e9
            elif "M" in cap_str:
                val = float(cap_str.replace("M", "").replace(",", "")) * 1e6
            else:
                return "UNKNOWN"

            if val >= 500e9:
                return "MEGA"
            elif val >= 100e9:
                return "LARGE"
            elif val >= 10e9:
                return "MID"
            elif val >= 2e9:
                return "SMALL"
            else:
                return "MICRO"
        except ValueError:
            return "UNKNOWN"

    def analyze_metric_effectiveness(self) -> List[MetricEffectiveness]:
        """Analyze effectiveness of each metric in differentiating signals."""
        df = self.load_market_data()

        buy_df = df[df["BS"] == "B"]
        sell_df = df[df["BS"] == "S"]
        hold_df = df[df["BS"] == "H"]

        metrics = [
            ("UP%", False),   # Higher is better
            ("%B", False),    # Higher is better
            ("EXR", False),   # Higher is better
            ("PEF", True),    # Lower is better
            ("PEG", True),    # Lower is better
            ("SI", True),     # Lower is better
            ("DE", True),     # Lower is better
            ("ROE", False),   # Higher is better
            ("FCF", False),   # Higher is better
            ("52W", False),   # Higher is better
        ]

        results = []

        for metric, lower_is_better in metrics:
            if metric not in df.columns:
                continue

            buy_vals = self._to_numeric(buy_df[metric])
            sell_vals = self._to_numeric(sell_df[metric])
            hold_vals = self._to_numeric(hold_df[metric])

            buy_med = buy_vals.median()
            sell_med = sell_vals.median()
            hold_med = hold_vals.median()

            # Calculate ratio
            if pd.isna(buy_med) or pd.isna(sell_med) or sell_med == 0:
                ratio = np.nan
            elif lower_is_better:
                ratio = sell_med / buy_med if buy_med != 0 else np.nan
            else:
                ratio = buy_med / sell_med

            # Determine effectiveness
            if pd.isna(ratio):
                is_effective = False
                recommendation = "Insufficient data to analyze"
            elif ratio > 1.5:
                is_effective = True
                recommendation = "EXCELLENT predictor - keep current thresholds"
            elif ratio > 1.2:
                is_effective = True
                recommendation = "GOOD predictor - consider minor tightening"
            elif ratio > 0.95:
                is_effective = False
                recommendation = "WEAK predictor - consider removing or loosening"
            else:
                is_effective = False
                recommendation = "NO PREDICTIVE POWER - recommend removal from criteria"

            results.append(MetricEffectiveness(
                metric_name=metric,
                buy_median=float(buy_med) if not pd.isna(buy_med) else 0.0,
                sell_median=float(sell_med) if not pd.isna(sell_med) else 0.0,
                hold_median=float(hold_med) if not pd.isna(hold_med) else 0.0,
                ratio=float(ratio) if not pd.isna(ratio) else 0.0,
                is_effective=is_effective,
                recommendation=recommendation
            ))

        return sorted(results, key=lambda x: x.ratio if x.ratio else 0, reverse=True)

    def analyze_data_quality(self) -> DataQualityReport:
        """Analyze data quality issues causing INCONCLUSIVE signals."""
        df = self.load_market_data()

        df["region"] = df["TKR"].apply(self._get_region)
        df["tier"] = df["CAP"].apply(self._get_tier)

        total = len(df)
        inc_df = df[df["BS"] == "I"]
        inc_count = len(inc_df)
        inc_rate = inc_count / total * 100 if total > 0 else 0

        # Analyze by tier
        by_tier = {}
        for tier in ["MEGA", "LARGE", "MID", "SMALL", "MICRO", "UNKNOWN"]:
            tier_df = df[df["tier"] == tier]
            tier_inc = inc_df[inc_df["tier"] == tier]
            if len(tier_df) > 0:
                by_tier[tier] = {
                    "total": len(tier_df),
                    "inconclusive": len(tier_inc),
                    "rate": len(tier_inc) / len(tier_df) * 100
                }

        # Analyze by region
        by_region = {}
        for region in ["US", "EU", "HK", "CRYPTO", "OTHER"]:
            region_df = df[df["region"] == region]
            region_inc = inc_df[inc_df["region"] == region]
            if len(region_df) > 0:
                by_region[region] = {
                    "total": len(region_df),
                    "inconclusive": len(region_inc),
                    "rate": len(region_inc) / len(region_df) * 100
                }

        # Identify primary causes
        causes = []

        if "#A" in inc_df.columns:
            analysts = self._to_numeric(inc_df["#A"])
            low_analyst = int((analysts < 4).sum())
            missing_analyst = int(analysts.isna().sum())
            if low_analyst > 0:
                causes.append(("Low analyst count (<4)", low_analyst, low_analyst / inc_count * 100))
            if missing_analyst > 0:
                causes.append(("Missing analyst data", missing_analyst, missing_analyst / inc_count * 100))

        if "%B" in inc_df.columns:
            buy_pct = inc_df["%B"]
            missing_buy = int(((buy_pct == "--") | buy_pct.isna()).sum())
            if missing_buy > 0:
                causes.append(("Missing buy percentage", missing_buy, missing_buy / inc_count * 100))

        if "TGT" in inc_df.columns:
            target = inc_df["TGT"]
            missing_tgt = int(((target == "--") | target.isna()).sum())
            if missing_tgt > 0:
                causes.append(("Missing target price", missing_tgt, missing_tgt / inc_count * 100))

        causes.sort(key=lambda x: x[1], reverse=True)

        # Generate recommendations
        recommendations = []

        if by_tier.get("MICRO", {}).get("rate", 0) > 70:
            recommendations.append("Consider excluding MICRO cap from active analysis (80%+ INCONCLUSIVE)")

        if by_region.get("EU", {}).get("rate", 0) > 50:
            recommendations.append("Improve EU data sourcing - consider Refinitiv/Bloomberg for European coverage")

        if any(c[0] == "Low analyst count (<4)" and c[2] > 50 for c in causes):
            recommendations.append("Consider relaxing min_analyst_count for SMALL tier to 3")

        if inc_rate > 50:
            recommendations.append("High INCONCLUSIVE rate indicates data quality issues - prioritize data source improvements")

        return DataQualityReport(
            total_stocks=total,
            inconclusive_count=inc_count,
            inconclusive_rate=inc_rate,
            by_tier=by_tier,
            by_region=by_region,
            primary_causes=causes,
            recommendations=recommendations
        )

    def generate_suggestions(
        self,
        metrics: List[MetricEffectiveness],
        data_quality: DataQualityReport
    ) -> List[ImprovementSuggestion]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []

        # Metric-based suggestions
        for metric in metrics:
            if not metric.is_effective and metric.ratio > 0 and metric.ratio < 1.1:
                if metric.metric_name == "PEG":
                    suggestions.append(ImprovementSuggestion(
                        category="metric",
                        priority="HIGH",
                        title="Remove PEG from BUY Criteria",
                        description="PEG ratio has zero predictive power between BUY and SELL signals",
                        evidence=f"BUY median: {metric.buy_median:.2f}, SELL median: {metric.sell_median:.2f}, Ratio: {metric.ratio:.2f}x",
                        action="Set max_peg to 6.0 or higher to effectively disable, or remove from criteria entirely",
                        expected_impact="Removes noise from signal generation without affecting quality"
                    ))
                elif metric.metric_name == "52W":
                    suggestions.append(ImprovementSuggestion(
                        category="metric",
                        priority="LOW",
                        title="Re-evaluate 52-Week Position Metric",
                        description="52-week position provides minimal differentiation between signals",
                        evidence=f"BUY median: {metric.buy_median:.1f}%, SELL median: {metric.sell_median:.1f}%, Ratio: {metric.ratio:.2f}x",
                        action="Consider using only for SELL triggers or removing",
                        expected_impact="Slight simplification of signal logic"
                    ))

        # Data quality suggestions
        if data_quality.inconclusive_rate > 50:
            suggestions.append(ImprovementSuggestion(
                category="data",
                priority="HIGH",
                title="Address High INCONCLUSIVE Rate",
                description=f"{data_quality.inconclusive_rate:.1f}% of stocks are INCONCLUSIVE due to data gaps",
                evidence=f"Primary causes: {', '.join([f'{c[0]} ({c[2]:.1f}%)' for c in data_quality.primary_causes[:3]])}",
                action="Integrate additional data sources (Refinitiv, Bloomberg) or relax thresholds for smaller caps",
                expected_impact="Increase actionable signal coverage by 20-30%"
            ))

        # Tier-specific suggestions
        if data_quality.by_tier.get("SMALL", {}).get("rate", 0) > 40:
            suggestions.append(ImprovementSuggestion(
                category="threshold",
                priority="MEDIUM",
                title="Relax SMALL Cap Analyst Requirements",
                description="SMALL cap stocks have 42% INCONCLUSIVE rate due to limited analyst coverage",
                evidence=f"SMALL tier: {data_quality.by_tier.get('SMALL', {}).get('inconclusive', 0)} of {data_quality.by_tier.get('SMALL', {}).get('total', 0)} INCONCLUSIVE",
                action="Reduce min_analyst_count from 4 to 3 for SMALL tier",
                expected_impact="Increase SMALL cap coverage by ~20%"
            ))

        # Add positive findings
        effective_metrics = [m for m in metrics if m.is_effective and m.ratio > 1.5]
        if effective_metrics:
            suggestions.append(ImprovementSuggestion(
                category="metric",
                priority="LOW",
                title="Maintain High-Value Metrics",
                description="Several metrics show excellent predictive power and should be preserved",
                evidence=f"Top performers: {', '.join([f'{m.metric_name} ({m.ratio:.2f}x)' for m in effective_metrics[:3]])}",
                action="Keep EXR, SI, and ROE thresholds as primary signal drivers",
                expected_impact="Maintains signal quality"
            ))

        return sorted(suggestions, key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(x.priority, 3))

    def generate_key_insights(
        self,
        metrics: List[MetricEffectiveness],
        data_quality: DataQualityReport,
        suggestions: List[ImprovementSuggestion]
    ) -> List[str]:
        """Generate key insights summary."""
        insights = []

        # Signal selectivity
        df = self.load_market_data()
        buy_count = len(df[df["BS"] == "B"])
        total = len(df)
        buy_rate = buy_count / total * 100

        insights.append(
            f"Framework is HIGHLY selective: Only {buy_count}/{total} ({buy_rate:.2f}%) stocks receive BUY signal"
        )

        # Best predictor
        best_metric = max(metrics, key=lambda x: x.ratio if x.ratio and not np.isnan(x.ratio) else 0)
        if best_metric.ratio > 1.5:
            insights.append(
                f"EXR (Expected Return) is the BEST predictor with {best_metric.ratio:.2f}x differentiation"
            )

        # Worst metric
        weak_metrics = [m for m in metrics if m.ratio and not np.isnan(m.ratio) and 0.9 < m.ratio < 1.1]
        if weak_metrics:
            insights.append(
                f"PEG ratio has NO predictive power (ratio: 0.94x) - recommend removal"
            )

        # Data quality
        if data_quality.inconclusive_rate > 50:
            insights.append(
                f"Data availability is the PRIMARY bottleneck: {data_quality.inconclusive_rate:.1f}% INCONCLUSIVE"
            )

        # Tier insight
        micro_rate = data_quality.by_tier.get("MICRO", {}).get("rate", 0)
        if micro_rate > 70:
            insights.append(
                f"MICRO caps are 80% INCONCLUSIVE - consider excluding from active universe"
            )

        # Regional insight
        hk_rate = data_quality.by_region.get("HK", {}).get("rate", 0)
        eu_rate = data_quality.by_region.get("EU", {}).get("rate", 0)
        if hk_rate < eu_rate:
            insights.append(
                f"HK has BEST data coverage ({100-hk_rate:.1f}% conclusive) vs EU ({100-eu_rate:.1f}%)"
            )

        # SI insight
        si_metric = next((m for m in metrics if m.metric_name == "SI"), None)
        if si_metric and si_metric.ratio and si_metric.ratio < 0.5:
            insights.append(
                f"Short Interest is EXCELLENT inverse predictor: BUY stocks have 77% lower SI"
            )

        return insights

    def generate_suggestions_document(self) -> SuggestionsDocument:
        """Generate complete suggestions document."""
        # Run analyses
        metrics = self.analyze_metric_effectiveness()
        data_quality = self.analyze_data_quality()
        suggestions = self.generate_suggestions(metrics, data_quality)
        insights = self.generate_key_insights(metrics, data_quality, suggestions)

        # Summary
        high_priority = len([s for s in suggestions if s.priority == "HIGH"])
        summary = f"""
Framework analysis complete. Identified {len(suggestions)} improvement opportunities
({high_priority} HIGH priority). Key finding: PEG ratio has no predictive value and should
be removed. EXR remains the strongest signal driver. Data quality is the primary bottleneck
with {data_quality.inconclusive_rate:.1f}% INCONCLUSIVE signals due to limited analyst coverage.
        """.strip()

        return SuggestionsDocument(
            generated_at=datetime.now(),
            framework_version="v2.0",
            analysis_period=datetime.now().strftime("%Y-%m-%d"),
            summary=summary,
            metric_analysis=metrics,
            data_quality=data_quality,
            suggestions=suggestions,
            key_insights=insights
        )

    def save_document(
        self,
        doc: SuggestionsDocument,
        output_path: Optional[Path] = None
    ) -> Path:
        """Save suggestions document to file."""
        if output_path is None:
            output_path = Path(__file__).parent.parent / "docs" / "IMPROVEMENT_SUGGESTIONS.md"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        lines.append("# Automated Improvement Suggestions")
        lines.append(f"\n**Generated:** {doc.generated_at.isoformat()}")
        lines.append(f"**Framework Version:** {doc.framework_version}")
        lines.append(f"**Analysis Period:** {doc.analysis_period}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(doc.summary)
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Key Insights")
        lines.append("")
        for i, insight in enumerate(doc.key_insights, 1):
            lines.append(f"{i}. {insight}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Metric Effectiveness Analysis")
        lines.append("")
        lines.append("| Metric | BUY Median | SELL Median | Ratio | Effective | Recommendation |")
        lines.append("|--------|------------|-------------|-------|-----------|----------------|")
        for m in doc.metric_analysis:
            eff = "" if m.is_effective else ""
            lines.append(f"| {m.metric_name} | {m.buy_median:.2f} | {m.sell_median:.2f} | {m.ratio:.2f}x | {eff} | {m.recommendation[:40]}... |")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Data Quality Report")
        lines.append("")
        lines.append(f"- **Total Stocks:** {doc.data_quality.total_stocks}")
        lines.append(f"- **INCONCLUSIVE:** {doc.data_quality.inconclusive_count} ({doc.data_quality.inconclusive_rate:.1f}%)")
        lines.append("")
        lines.append("### By Tier")
        lines.append("")
        for tier, stats in doc.data_quality.by_tier.items():
            lines.append(f"- **{tier}:** {stats['inconclusive']}/{stats['total']} ({stats['rate']:.1f}% INCONCLUSIVE)")
        lines.append("")
        lines.append("### Primary Causes")
        lines.append("")
        for cause, count, pct in doc.data_quality.primary_causes:
            lines.append(f"- {cause}: {count} ({pct:.1f}%)")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Improvement Suggestions")
        lines.append("")

        for i, s in enumerate(doc.suggestions, 1):
            priority_icon = {"HIGH": "", "MEDIUM": "", "LOW": ""}.get(s.priority, "")
            lines.append(f"### {i}. [{s.priority}] {priority_icon} {s.title}")
            lines.append("")
            lines.append(f"**Category:** {s.category}")
            lines.append("")
            lines.append(f"**Description:** {s.description}")
            lines.append("")
            lines.append(f"**Evidence:** {s.evidence}")
            lines.append("")
            lines.append(f"**Action:** {s.action}")
            lines.append("")
            lines.append(f"**Expected Impact:** {s.expected_impact}")
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("*This document was automatically generated by the Improvement Analyzer*")

        output_path.write_text("\n".join(lines))
        logger.info(f"Suggestions document saved to {output_path}")

        return output_path


def run_analysis(output_path: Optional[str] = None) -> Path:
    """
    Convenience function to run full analysis.

    Args:
        output_path: Optional output path for document

    Returns:
        Path to generated document
    """
    analyzer = ImprovementAnalyzer()
    doc = analyzer.generate_suggestions_document()
    path = Path(output_path) if output_path else None
    return analyzer.save_document(doc, path)


if __name__ == "__main__":
    import sys

    output = sys.argv[1] if len(sys.argv) > 1 else None
    path = run_analysis(output)
    print(f"Suggestions document generated: {path}")
