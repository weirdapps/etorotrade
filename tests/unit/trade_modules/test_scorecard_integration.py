"""
Tests for Signal Scorecard Integration
"""

import pytest
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, mock_open

from trade_modules.scorecard_integration import (
    get_scorecard_warnings,
    format_scorecard_warnings_for_console,
    get_scorecard_summary,
    SCORECARD_MAX_AGE_DAYS,
)


class TestScorecardIntegration:
    """Tests for scorecard integration."""

    def test_get_scorecard_warnings_fresh(self):
        """Test loading warnings from fresh scorecard."""
        scorecard_data = {
            'generated_at': datetime.now().isoformat(),
            'calibration_alerts': [
                'US LARGE BUY has only 45% hit rate at T+21 (n=20) - review thresholds',
                'EU MID SELL has only 40% hit rate at T+21 (n=15) - review thresholds',
            ]
        }

        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(scorecard_data))):
                warnings = get_scorecard_warnings()

        assert len(warnings) == 2
        assert 'US LARGE BUY' in warnings[0]
        assert 'EU MID SELL' in warnings[1]

    def test_get_scorecard_warnings_stale(self):
        """Test returns empty for stale scorecard."""
        old_date = datetime.now() - timedelta(days=SCORECARD_MAX_AGE_DAYS + 1)
        scorecard_data = {
            'generated_at': old_date.isoformat(),
            'calibration_alerts': [
                'US LARGE BUY has only 45% hit rate at T+21 (n=20) - review thresholds',
            ]
        }

        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(scorecard_data))):
                warnings = get_scorecard_warnings()

        assert warnings == []

    def test_get_scorecard_warnings_missing_file(self):
        """Test handles missing scorecard file."""
        with patch('pathlib.Path.exists', return_value=False):
            warnings = get_scorecard_warnings()

        assert warnings == []

    def test_get_scorecard_warnings_no_alerts(self):
        """Test handles scorecard with no alerts."""
        scorecard_data = {
            'generated_at': datetime.now().isoformat(),
            'calibration_alerts': []
        }

        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(scorecard_data))):
                warnings = get_scorecard_warnings()

        assert warnings == []

    def test_get_scorecard_warnings_invalid_json(self):
        """Test handles invalid JSON gracefully."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data='invalid json {')):
                warnings = get_scorecard_warnings()

        assert warnings == []

    def test_get_scorecard_warnings_missing_generated_at(self):
        """Test handles scorecard missing timestamp."""
        scorecard_data = {
            'calibration_alerts': ['Some warning']
        }

        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(scorecard_data))):
                warnings = get_scorecard_warnings()

        assert warnings == []

    def test_get_scorecard_warnings_invalid_timestamp(self):
        """Test handles invalid timestamp format."""
        scorecard_data = {
            'generated_at': 'not-a-timestamp',
            'calibration_alerts': ['Some warning']
        }

        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(scorecard_data))):
                warnings = get_scorecard_warnings()

        assert warnings == []

    def test_format_scorecard_warnings_for_console_with_warnings(self):
        """Test formatting warnings for console."""
        warnings = [
            'US LARGE BUY has only 45% hit rate at T+21 (n=20) - review thresholds',
            'EU MID SELL has only 40% hit rate at T+21 (n=15) - review thresholds',
        ]

        output = format_scorecard_warnings_for_console(warnings)

        assert 'SIGNAL CALIBRATION WARNINGS' in output
        assert 'US LARGE BUY' in output
        assert 'EU MID SELL' in output
        assert 'underperforming' in output

    def test_format_scorecard_warnings_for_console_empty(self):
        """Test formatting with no warnings."""
        output = format_scorecard_warnings_for_console([])

        assert output == ""

    def test_get_scorecard_summary_fresh(self):
        """Test getting scorecard summary with fresh data."""
        now = datetime.now()
        scorecard_data = {
            'generated_at': now.isoformat(),
            'calibration_alerts': [
                'US LARGE BUY has only 45% hit rate at T+21 (n=20) - review thresholds'
            ]
        }

        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(scorecard_data))):
                summary = get_scorecard_summary()

        assert len(summary['warnings']) == 1
        assert summary['scorecard_age_days'] == 0
        assert summary['scorecard_date'] == now.isoformat()
        assert summary['is_fresh'] is True

    def test_get_scorecard_summary_stale(self):
        """Test getting scorecard summary with stale data."""
        old_date = datetime.now() - timedelta(days=SCORECARD_MAX_AGE_DAYS + 1)
        scorecard_data = {
            'generated_at': old_date.isoformat(),
            'calibration_alerts': ['Some warning']
        }

        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(scorecard_data))):
                summary = get_scorecard_summary()

        assert summary['scorecard_age_days'] == SCORECARD_MAX_AGE_DAYS + 1
        assert summary['is_fresh'] is False

    def test_get_scorecard_summary_missing_file(self):
        """Test getting summary when file is missing."""
        with patch('pathlib.Path.exists', return_value=False):
            summary = get_scorecard_summary()

        assert summary['warnings'] == []
        assert summary['scorecard_age_days'] is None
        assert summary['scorecard_date'] is None
        assert summary['is_fresh'] is False

    def test_scorecard_max_age_constant(self):
        """Test the max age constant is reasonable."""
        assert SCORECARD_MAX_AGE_DAYS == 7

    def test_get_scorecard_warnings_with_timezone(self):
        """Test handles ISO timestamps with timezone info."""
        scorecard_data = {
            'generated_at': '2026-03-15T10:30:00Z',
            'calibration_alerts': ['Test warning']
        }

        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(scorecard_data))):
                with patch('trade_modules.scorecard_integration.datetime') as mock_dt:
                    # Mock current time to be within 7 days
                    mock_dt.now.return_value = datetime(2026, 3, 16, 12, 0)
                    mock_dt.fromisoformat = datetime.fromisoformat
                    warnings = get_scorecard_warnings()

        # Should load warnings since it's recent
        assert len(warnings) == 1
