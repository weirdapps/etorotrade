"""
M15: Hard-fail on broken agent JSON outputs — CIO v36 Empirical Refoundation.

Today's news.json and census.json are HTTP error blobs (OAuth 401), but the
committee silently feeds empty breaking_news=[] and divergences={} into
synthesis as if all-clear. M15 surfaces this with a hard error so a broken
agent run can't masquerade as a clean one.

Contract:
- validate_agent_report(report, agent_name) returns the report unchanged
  when valid; raises BrokenAgentReportError if the report is an HTTP error
  envelope (has "error" + "error_description" keys).
- A report can opt into graceful degradation by setting
  data_status="INSUFFICIENT_DATA"; the validator logs but does not raise.
"""

import json

import pytest


class TestValidateAgentReport:
    def test_oauth_error_envelope_raises(self):
        from trade_modules.committee_synthesis import (
            BrokenAgentReportError,
            validate_agent_report,
        )

        report = {
            "error": "invalid_grant",
            "error_description": "Token has expired",
        }
        with pytest.raises(BrokenAgentReportError) as exc:
            validate_agent_report(report, agent_name="news")
        assert "news" in str(exc.value)
        assert "invalid_grant" in str(exc.value) or "error" in str(exc.value).lower()

    def test_http_error_status_raises(self):
        from trade_modules.committee_synthesis import (
            BrokenAgentReportError,
            validate_agent_report,
        )

        report = {"error": "401", "error_description": "Unauthorized"}
        with pytest.raises(BrokenAgentReportError):
            validate_agent_report(report, agent_name="census")

    def test_explicit_insufficient_data_logs_but_does_not_raise(self, caplog):
        import logging

        from trade_modules.committee_synthesis import validate_agent_report

        caplog.set_level(logging.WARNING)

        report = {
            "data_status": "INSUFFICIENT_DATA",
            "stocks": {},
        }
        # Should NOT raise
        out = validate_agent_report(report, agent_name="fundamental")
        assert out is report
        assert any("INSUFFICIENT_DATA" in rec.message for rec in caplog.records)

    def test_realistic_report_passes_through(self):
        from trade_modules.committee_synthesis import validate_agent_report

        report = {
            "stocks": {"AAPL": {"score": 85}},
            "summary": "5 stocks scored",
        }
        out = validate_agent_report(report, agent_name="fundamental")
        assert out is report

    def test_non_dict_input_raises(self):
        from trade_modules.committee_synthesis import (
            BrokenAgentReportError,
            validate_agent_report,
        )

        for bad in (None, "string", 42, [1, 2, 3]):
            with pytest.raises(BrokenAgentReportError):
                validate_agent_report(bad, agent_name="x")


class TestLoadAgentReportFromFile:
    """load_agent_report path → validated report or raise."""

    def test_load_valid_report_returns_dict(self, tmp_path):
        from trade_modules.committee_synthesis import load_agent_report

        path = tmp_path / "fundamental.json"
        with open(path, "w") as f:
            json.dump({"stocks": {"AAPL": {"score": 80}}}, f)

        report = load_agent_report(path, agent_name="fundamental")
        assert report["stocks"]["AAPL"]["score"] == 80

    def test_load_oauth_error_raises(self, tmp_path):
        from trade_modules.committee_synthesis import (
            BrokenAgentReportError,
            load_agent_report,
        )

        path = tmp_path / "news.json"
        with open(path, "w") as f:
            json.dump({"error": "invalid_grant", "error_description": "expired"}, f)

        with pytest.raises(BrokenAgentReportError) as exc:
            load_agent_report(path, agent_name="news")
        assert "news" in str(exc.value)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        from trade_modules.committee_synthesis import load_agent_report

        path = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            load_agent_report(path, agent_name="x")

    def test_invalid_json_raises(self, tmp_path):
        from trade_modules.committee_synthesis import (
            BrokenAgentReportError,
            load_agent_report,
        )

        path = tmp_path / "broken.json"
        path.write_text("{this is not valid json")
        with pytest.raises(BrokenAgentReportError):
            load_agent_report(path, agent_name="news")
