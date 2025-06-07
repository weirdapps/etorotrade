"""
Price target validation and robustness utilities.

This module provides functions for validating price target quality and
identifying stocks with unreliable analyst coverage.
"""

from typing import Any, Dict, Optional, Tuple
import math

from ...core.logging import get_logger

logger = get_logger(__name__)


def calculate_price_target_robustness(
    mean: Optional[float],
    median: Optional[float],
    high: Optional[float],
    low: Optional[float],
    current_price: Optional[float],
    analyst_count: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Calculate price target robustness metrics.

    Args:
        mean: Mean price target
        median: Median price target
        high: Highest price target
        low: Lowest price target
        current_price: Current stock price
        analyst_count: Number of analysts providing price targets

    Returns:
        Dict containing robustness metrics and flags
    """
    result = {
        "is_robust": False,
        "robustness_score": 0.0,  # 0-100 scale
        "spread_percent": None,
        "mean_median_diff_percent": None,
        "outlier_ratio": None,
        "warning_flags": [],
        "quality_grade": "F",  # A, B, C, D, F
    }

    # Early return if we don't have basic data
    if not all([mean, median, high, low, current_price]) or current_price <= 0:
        result["warning_flags"].append("Insufficient price target data")
        return result

    try:
        # 1. Calculate spread as percentage of median price
        spread = high - low
        spread_percent = (spread / median) * 100
        result["spread_percent"] = spread_percent

        # 2. Calculate mean vs median difference (indicates skewness)
        mean_median_diff = abs(mean - median)
        mean_median_diff_percent = (mean_median_diff / median) * 100
        result["mean_median_diff_percent"] = mean_median_diff_percent

        # 3. Calculate outlier ratio (how far high/low are from mean)
        high_deviation = abs(high - mean) / mean * 100
        low_deviation = abs(low - mean) / mean * 100
        max_deviation = max(high_deviation, low_deviation)
        result["outlier_ratio"] = max_deviation

        # 4. Scoring criteria
        score = 100.0

        # Penalize excessive spread with stricter thresholds
        # Wide disagreement among analysts indicates uncertainty
        if spread_percent > 100:
            score -= 50  # Increased from 40
            result["warning_flags"].append(f"Extreme price target spread ({spread_percent:.1f}%)")
        elif spread_percent > 75:
            score -= 35  # New threshold for 75-100% spread
            result["warning_flags"].append(f"Very high price target spread ({spread_percent:.1f}%)")
        elif spread_percent > 50:
            score -= 25  # Increased from 20
            result["warning_flags"].append(f"High price target spread ({spread_percent:.1f}%)")
        elif spread_percent > 30:
            score -= 15  # Increased from 10, lowered threshold from 25%
            result["warning_flags"].append(f"Moderate price target spread ({spread_percent:.1f}%)")

        # Penalize mean-median skewness (>10% indicates outliers)
        if mean_median_diff_percent > 20:
            score -= 25
            result["warning_flags"].append(
                f"High mean-median difference ({mean_median_diff_percent:.1f}%)"
            )
        elif mean_median_diff_percent > 10:
            score -= 15
        elif mean_median_diff_percent > 5:
            score -= 5

        # Penalize extreme outliers (>100% deviation from mean)
        if max_deviation > 200:
            score -= 25
            result["warning_flags"].append(
                f"Extreme outlier price targets ({max_deviation:.1f}% deviation)"
            )
        elif max_deviation > 100:
            score -= 15
        elif max_deviation > 50:
            score -= 5

        # Bonus for sufficient analyst coverage
        if analyst_count and analyst_count >= 10:
            score += 5
        elif analyst_count and analyst_count >= 5:
            score += 2
        elif analyst_count and analyst_count < 3:
            score -= 15
            result["warning_flags"].append(f"Low analyst coverage ({analyst_count} analysts)")

        # Check for extreme price targets vs current price
        median_vs_current = abs(median - current_price) / current_price * 100
        if median_vs_current > 300:  # >300% upside/downside
            score -= 30
            result["warning_flags"].append(
                f"Extreme median price target vs current price ({median_vs_current:.1f}%)"
            )
        elif median_vs_current > 150:  # >150% upside/downside
            score -= 15

        # Ensure score is in valid range
        score = max(0, min(100, score))
        result["robustness_score"] = score

        # 5. Quality grading with stricter thresholds
        if score >= 85:  # Raised from 80
            result["quality_grade"] = "A"
            result["is_robust"] = True
        elif score >= 70:  # Raised from 65
            result["quality_grade"] = "B"
            result["is_robust"] = True
        elif score >= 55:  # Raised from 50
            result["quality_grade"] = "C"
        elif score >= 40:  # Raised from 35
            result["quality_grade"] = "D"
        else:
            result["quality_grade"] = "F"

        # 6. Additional robustness criteria
        # Consider robust if spread < 40% AND mean-median diff < 15%
        if spread_percent < 40 and mean_median_diff_percent < 15 and score >= 50:
            result["is_robust"] = True

        logger.debug(
            f"Price target robustness: score={score:.1f}, grade={result['quality_grade']}, robust={result['is_robust']}"
        )

    except (ValueError, TypeError, ZeroDivisionError) as e:
        logger.warning(f"Error calculating price target robustness: {e}")
        result["warning_flags"].append("Error calculating robustness metrics")

    return result


def validate_price_target_data(ticker_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate price target data quality for a ticker.

    Args:
        ticker_data: Dict containing ticker information with price target fields

    Returns:
        Tuple of (is_valid, validation_info)
    """
    validation_info = {
        "has_price_targets": False,
        "has_robust_targets": False,
        "robustness_metrics": None,
        "recommended_action": "use_median",  # use_median, use_mean, exclude, manual_review
        "confidence_level": "low",  # low, medium, high
    }

    # Extract price target fields
    mean = ticker_data.get("target_price_mean")
    median = ticker_data.get("target_price_median")
    high = ticker_data.get("target_price_high")
    low = ticker_data.get("target_price_low")
    current_price = ticker_data.get("price")
    analyst_count = ticker_data.get("price_target_analyst_count")

    # Check if we have any price target data
    if not any([mean, median, high, low]):
        validation_info["recommended_action"] = "exclude"
        return False, validation_info

    validation_info["has_price_targets"] = True

    # Calculate robustness metrics
    robustness = calculate_price_target_robustness(
        mean, median, high, low, current_price, analyst_count
    )
    validation_info["robustness_metrics"] = robustness

    # Determine recommended action based on robustness
    if robustness["is_robust"]:
        validation_info["has_robust_targets"] = True
        validation_info["confidence_level"] = "high"
        validation_info["recommended_action"] = "use_median"
    elif robustness["robustness_score"] >= 35:
        validation_info["confidence_level"] = "medium"
        validation_info["recommended_action"] = (
            "use_median"  # Still prefer median for outlier resistance
        )
    elif robustness["robustness_score"] >= 20:
        validation_info["confidence_level"] = "low"
        validation_info["recommended_action"] = "manual_review"
    else:
        validation_info["confidence_level"] = "low"
        validation_info["recommended_action"] = "exclude"

    # Special cases - only exclude for very poor quality
    # Allow stocks with high spread (like TSLA) but exclude truly extreme cases
    if len(robustness["warning_flags"]) >= 4:
        validation_info["recommended_action"] = "exclude"
    elif any(
        "Extreme median price target vs current price" in flag
        for flag in robustness["warning_flags"]
    ):
        validation_info["recommended_action"] = "exclude"
    elif robustness["robustness_score"] < 20:
        validation_info["recommended_action"] = "exclude"

    is_valid = validation_info["recommended_action"] in ["use_median", "use_mean", "manual_review"]

    return is_valid, validation_info


def get_preferred_price_target(ticker_data: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """
    Get the preferred price target based on data quality.

    Args:
        ticker_data: Dict containing ticker information

    Returns:
        Tuple of (preferred_target_price, source_description)
    """
    is_valid, validation_info = validate_price_target_data(ticker_data)

    if not is_valid or validation_info["recommended_action"] == "exclude":
        return None, "excluded_due_to_poor_quality"

    # Get the values
    median = ticker_data.get("target_price_median")
    mean = ticker_data.get("target_price_mean")

    # Prefer median for robustness against outliers
    if median is not None:
        return median, f"median_robust_{validation_info['confidence_level']}_confidence"
    elif mean is not None:
        return mean, f"mean_fallback_{validation_info['confidence_level']}_confidence"
    else:
        return None, "no_valid_price_targets"
