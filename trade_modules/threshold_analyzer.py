"""
Threshold Analyzer Module

Provides tools for analyzing and potentially optimizing trading thresholds.
Since historical backtesting is not feasible (no historical analyst targets),
this module enables:

1. Sensitivity analysis - how signal counts change with threshold variations
2. Distribution analysis - understand current signal distribution
3. Threshold recommendations based on statistical properties

This addresses the "arbitrary threshold" concern from the hedge fund review.
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ThresholdAnalysis:
    """Results of threshold sensitivity analysis."""
    current_value: float
    test_values: List[float]
    signal_counts: Dict[float, Dict[str, int]]
    recommendation: Optional[float]
    reasoning: str


def load_market_data(market_path: Optional[Path] = None) -> pd.DataFrame:
    """Load market data from CSV."""
    if market_path is None:
        market_path = Path(__file__).parent.parent / "yahoofinance" / "output" / "market.csv"

    if not market_path.exists():
        raise FileNotFoundError(f"Market data not found: {market_path}")

    return pd.read_csv(market_path)


def parse_percentage(val) -> float:
    """Parse percentage value from string or number."""
    if pd.isna(val) or val == '--':
        return np.nan
    try:
        return float(str(val).rstrip('%'))
    except (ValueError, TypeError):
        return np.nan


def analyze_upside_distribution(df: pd.DataFrame) -> Dict:
    """Analyze upside distribution to inform threshold selection."""
    df = df.copy()
    df['upside_num'] = df['UP%'].apply(parse_percentage)

    valid_upside = df['upside_num'].dropna()

    return {
        'count': len(valid_upside),
        'mean': valid_upside.mean(),
        'median': valid_upside.median(),
        'std': valid_upside.std(),
        'percentile_10': valid_upside.quantile(0.10),
        'percentile_25': valid_upside.quantile(0.25),
        'percentile_50': valid_upside.quantile(0.50),
        'percentile_75': valid_upside.quantile(0.75),
        'percentile_90': valid_upside.quantile(0.90),
        'min': valid_upside.min(),
        'max': valid_upside.max(),
    }


def analyze_buy_percentage_distribution(df: pd.DataFrame) -> Dict:
    """Analyze buy percentage distribution to inform threshold selection."""
    df = df.copy()
    df['buy_pct_num'] = df['%B'].apply(parse_percentage)

    valid_pct = df['buy_pct_num'].dropna()

    return {
        'count': len(valid_pct),
        'mean': valid_pct.mean(),
        'median': valid_pct.median(),
        'std': valid_pct.std(),
        'percentile_10': valid_pct.quantile(0.10),
        'percentile_25': valid_pct.quantile(0.25),
        'percentile_50': valid_pct.quantile(0.50),
        'percentile_75': valid_pct.quantile(0.75),
        'percentile_90': valid_pct.quantile(0.90),
    }


def threshold_sensitivity_analysis(
    df: pd.DataFrame,
    metric_col: str,
    threshold_range: Tuple[float, float],
    step: float,
    direction: str = 'above'
) -> ThresholdAnalysis:
    """
    Analyze how signal counts change with threshold variations.

    Args:
        df: Market data DataFrame
        metric_col: Column to analyze (e.g., 'UP%', '%B')
        threshold_range: (min, max) values to test
        step: Step size for threshold testing
        direction: 'above' for min thresholds, 'below' for max thresholds

    Returns:
        ThresholdAnalysis with results
    """
    df = df.copy()
    df['metric_num'] = df[metric_col].apply(parse_percentage)

    test_values = np.arange(threshold_range[0], threshold_range[1] + step, step)
    signal_counts = {}

    for threshold in test_values:
        if direction == 'above':
            passing = (df['metric_num'] >= threshold).sum()
            failing = (df['metric_num'] < threshold).sum()
        else:
            passing = (df['metric_num'] <= threshold).sum()
            failing = (df['metric_num'] > threshold).sum()

        signal_counts[threshold] = {
            'passing': int(passing),
            'failing': int(failing),
            'pass_rate': passing / len(df) * 100 if len(df) > 0 else 0,
        }

    # Find threshold that gives ~10% pass rate (selective BUY)
    target_rate = 10.0
    recommendation = None
    min_diff = float('inf')

    for threshold, counts in signal_counts.items():
        diff = abs(counts['pass_rate'] - target_rate)
        if diff < min_diff:
            min_diff = diff
            recommendation = threshold

    return ThresholdAnalysis(
        current_value=threshold_range[0],
        test_values=list(test_values),
        signal_counts=signal_counts,
        recommendation=recommendation,
        reasoning=f"Threshold {recommendation} gives ~{signal_counts[recommendation]['pass_rate']:.1f}% pass rate (target: ~10% for selective signals)"
    )


def generate_threshold_report(market_path: Optional[Path] = None) -> str:
    """Generate a comprehensive threshold analysis report."""
    df = load_market_data(market_path)

    report = []
    report.append("=" * 70)
    report.append("THRESHOLD ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")

    # Upside distribution
    upside_stats = analyze_upside_distribution(df)
    report.append("UPSIDE (UP%) Distribution:")
    report.append(f"  Count: {upside_stats['count']}")
    report.append(f"  Mean: {upside_stats['mean']:.1f}%")
    report.append(f"  Median: {upside_stats['median']:.1f}%")
    report.append(f"  Std Dev: {upside_stats['std']:.1f}%")
    report.append(f"  10th percentile: {upside_stats['percentile_10']:.1f}%")
    report.append(f"  25th percentile: {upside_stats['percentile_25']:.1f}%")
    report.append(f"  75th percentile: {upside_stats['percentile_75']:.1f}%")
    report.append(f"  90th percentile: {upside_stats['percentile_90']:.1f}%")
    report.append("")

    # Buy percentage distribution
    buy_pct_stats = analyze_buy_percentage_distribution(df)
    report.append("BUY PERCENTAGE (%B) Distribution:")
    report.append(f"  Count: {buy_pct_stats['count']}")
    report.append(f"  Mean: {buy_pct_stats['mean']:.1f}%")
    report.append(f"  Median: {buy_pct_stats['median']:.1f}%")
    report.append(f"  75th percentile: {buy_pct_stats['percentile_75']:.1f}%")
    report.append(f"  90th percentile: {buy_pct_stats['percentile_90']:.1f}%")
    report.append("")

    # Threshold recommendations
    report.append("THRESHOLD RECOMMENDATIONS:")
    report.append("(Based on achieving selective ~10% pass rate for BUY signals)")
    report.append("")

    # Analyze upside threshold
    upside_analysis = threshold_sensitivity_analysis(
        df, 'UP%', (5, 30), 5, 'above'
    )
    report.append(f"  min_upside recommendation: {upside_analysis.recommendation}%")
    report.append(f"    {upside_analysis.reasoning}")
    report.append("")

    # Analyze buy percentage threshold
    buy_pct_analysis = threshold_sensitivity_analysis(
        df, '%B', (60, 95), 5, 'above'
    )
    report.append(f"  min_buy_percentage recommendation: {buy_pct_analysis.recommendation}%")
    report.append(f"    {buy_pct_analysis.reasoning}")
    report.append("")

    # Current signal distribution
    report.append("CURRENT SIGNAL DISTRIBUTION:")
    signal_counts = df['BS'].value_counts()
    total = len(df)
    for signal, count in signal_counts.items():
        report.append(f"  {signal}: {count} ({count/total*100:.1f}%)")
    report.append("")

    # Selectivity assessment
    buy_rate = signal_counts.get('B', 0) / total * 100
    report.append("SELECTIVITY ASSESSMENT:")
    if buy_rate < 5:
        report.append(f"  Current BUY rate: {buy_rate:.1f}% (VERY SELECTIVE - may be too strict)")
    elif buy_rate < 15:
        report.append(f"  Current BUY rate: {buy_rate:.1f}% (SELECTIVE - good range)")
    elif buy_rate < 30:
        report.append(f"  Current BUY rate: {buy_rate:.1f}% (MODERATE - consider tightening)")
    else:
        report.append(f"  Current BUY rate: {buy_rate:.1f}% (TOO LOOSE - signals may lack conviction)")

    report.append("")
    report.append("=" * 70)

    return "\n".join(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(generate_threshold_report())
