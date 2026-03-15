"""
Signal Scorecard Integration

Integrates signal scorecard warnings into daily trading output.
Reads cached scorecard results and surfaces calibration alerts.

CIO Review Finding #5: Integrate signal scorecard warnings.
"""

import json
import logging
from pathlib import Path
from typing import List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Default scorecard path
SCORECARD_CACHE_PATH = Path.home() / ".weirdapps-trading" / "signals" / "signal_scorecard.json"
SCORECARD_MAX_AGE_DAYS = 7  # Alert if scorecard is older than 7 days


def get_scorecard_warnings() -> List[str]:
    """
    Load cached signal scorecard and extract calibration alerts.

    Reads the most recent scorecard from cache. If the scorecard is stale
    (>7 days old) or missing, returns empty list. Does NOT run the scorecard
    (too expensive for daily runs - scorecard should be run separately).

    Returns:
        List of warning strings for underperforming tier-region combinations.
        Empty list if scorecard unavailable or stale.
    """
    try:
        # Check if scorecard file exists
        if not SCORECARD_CACHE_PATH.exists():
            logger.debug(f"Scorecard not found at {SCORECARD_CACHE_PATH}")
            return []

        # Load scorecard
        with open(SCORECARD_CACHE_PATH, 'r') as f:
            scorecard = json.load(f)

        # Check freshness
        generated_at_str = scorecard.get('generated_at')
        if not generated_at_str:
            logger.warning("Scorecard missing 'generated_at' field")
            return []

        # Parse timestamp
        try:
            generated_at = datetime.fromisoformat(generated_at_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            logger.warning(f"Invalid scorecard timestamp: {generated_at_str}")
            return []

        # Check age
        age_days = (datetime.now().replace(tzinfo=None) - generated_at.replace(tzinfo=None)).days
        if age_days > SCORECARD_MAX_AGE_DAYS:
            logger.info(
                f"Scorecard is {age_days} days old (max: {SCORECARD_MAX_AGE_DAYS}), "
                "skipping warnings"
            )
            return []

        # Extract calibration alerts
        alerts = scorecard.get('calibration_alerts', [])
        if alerts:
            logger.info(f"Found {len(alerts)} calibration alerts from scorecard")
        else:
            logger.debug("No calibration alerts in scorecard")

        return alerts

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse scorecard JSON: {e}")
        return []
    except Exception as e:
        logger.warning(f"Failed to load scorecard: {e}")
        return []


def format_scorecard_warnings_for_console(warnings: List[str]) -> str:
    """
    Format scorecard warnings for console output.

    Args:
        warnings: List of warning strings from get_scorecard_warnings()

    Returns:
        Formatted string for console display, or empty string if no warnings
    """
    if not warnings:
        return ""

    output = "\n" + "=" * 70 + "\n"
    output += "  SIGNAL CALIBRATION WARNINGS\n"
    output += "=" * 70 + "\n"
    output += "The following tier-region combinations are underperforming:\n\n"

    for warning in warnings:
        output += f"  ! {warning}\n"

    output += "\nConsider reviewing thresholds in config.yaml for these segments.\n"
    output += "=" * 70 + "\n"

    return output


def get_scorecard_summary() -> dict:
    """
    Get scorecard summary for inclusion in reports.

    Returns:
        Dictionary with:
        - warnings: List of calibration alerts
        - scorecard_age_days: Age of scorecard in days
        - scorecard_date: ISO timestamp of scorecard generation
        - is_fresh: Boolean indicating if scorecard is recent
    """
    try:
        if not SCORECARD_CACHE_PATH.exists():
            return {
                'warnings': [],
                'scorecard_age_days': None,
                'scorecard_date': None,
                'is_fresh': False,
            }

        with open(SCORECARD_CACHE_PATH, 'r') as f:
            scorecard = json.load(f)

        generated_at_str = scorecard.get('generated_at')
        if not generated_at_str:
            return {
                'warnings': [],
                'scorecard_age_days': None,
                'scorecard_date': None,
                'is_fresh': False,
            }

        try:
            generated_at = datetime.fromisoformat(generated_at_str.replace('Z', '+00:00'))
            age_days = (datetime.now().replace(tzinfo=None) - generated_at.replace(tzinfo=None)).days
            is_fresh = age_days <= SCORECARD_MAX_AGE_DAYS
        except (ValueError, AttributeError):
            return {
                'warnings': [],
                'scorecard_age_days': None,
                'scorecard_date': None,
                'is_fresh': False,
            }

        return {
            'warnings': scorecard.get('calibration_alerts', []),
            'scorecard_age_days': age_days,
            'scorecard_date': generated_at_str,
            'is_fresh': is_fresh,
        }

    except Exception as e:
        logger.warning(f"Failed to get scorecard summary: {e}")
        return {
            'warnings': [],
            'scorecard_age_days': None,
            'scorecard_date': None,
            'is_fresh': False,
        }
