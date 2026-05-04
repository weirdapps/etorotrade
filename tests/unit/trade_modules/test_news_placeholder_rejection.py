"""
M1: News placeholder/error rejection tests — CIO v36 Empirical Refoundation.

Audit found that ~/.weirdapps-trading/committee/scripts/fetch_news_events.py
has hardcoded placeholder breaking_news entries (line 142-181) — fictional
"Fed maintains rates" and "AI chip demand robust" headlines that ship to
production and bias regime momentum scoring (committee_synthesis.py:1100-1109).

Today's news.json is also an HTTP OAuth error blob — silently feeds empty
breaking_news=[] as if all-clear.

This module's contract: validate_news_report() returns a status indicating
whether the report is OK / PLACEHOLDER / ERROR / EMPTY, and the synthesis
pipeline must use this status to skip the news_catalyst modifier and the
breaking_news regime-momentum signal when status is not OK.
"""


class TestValidateNewsReport:
    """validate_news_report(report) -> str in {OK, PLACEHOLDER, ERROR, EMPTY}."""

    def test_oauth_error_envelope_returns_error(self):
        from trade_modules.committee_synthesis import validate_news_report

        report = {
            "error": "invalid_grant",
            "error_description": "Token has expired",
        }
        assert validate_news_report(report) == "ERROR"

    def test_http_error_envelope_returns_error(self):
        from trade_modules.committee_synthesis import validate_news_report

        report = {"error": "401", "error_description": "Unauthorized"}
        assert validate_news_report(report) == "ERROR"

    def test_hardcoded_fed_placeholder_returns_placeholder(self):
        from trade_modules.committee_synthesis import validate_news_report

        # Exact fingerprint from fetch_news_events.py:144
        report = {
            "breaking_news": [
                {
                    "headline": "Fed maintains rates, signals data-dependent approach for 2026",
                    "impact": "NEUTRAL",
                    "affected_sectors": ["Financials", "Technology"],
                    "affected_tickers": ["JPM", "BAC"],
                    "source": "Reuters",
                },
                {
                    "headline": "AI chip demand remains robust despite China export restrictions",
                    "impact": "LOW_POSITIVE",
                    "affected_sectors": ["Semiconductors"],
                    "affected_tickers": ["NVDA", "AMD", "TSM"],
                    "source": "Bloomberg",
                },
            ],
            "data_quality": "PARTIAL",
        }
        assert validate_news_report(report) == "PLACEHOLDER"

    def test_explicit_placeholder_data_status_returns_placeholder(self):
        from trade_modules.committee_synthesis import validate_news_report

        report = {
            "breaking_news": [{"headline": "x", "impact": "NEUTRAL"}],
            "data_status": "PLACEHOLDER",
        }
        assert validate_news_report(report) == "PLACEHOLDER"

    def test_empty_report_returns_empty(self):
        from trade_modules.committee_synthesis import validate_news_report

        assert validate_news_report({}) == "EMPTY"
        assert validate_news_report({"breaking_news": []}) == "EMPTY"

    def test_realistic_report_returns_ok(self):
        from trade_modules.committee_synthesis import validate_news_report

        report = {
            "breaking_news": [
                {
                    "headline": "Apple unveils M5 chip with 40% performance gain",
                    "impact": "LOW_POSITIVE",
                    "affected_tickers": ["AAPL"],
                    "source": "Reuters",
                },
                {
                    "headline": "European banks face higher capital requirements under Basel IV",
                    "impact": "LOW_NEGATIVE",
                    "affected_sectors": ["Financials"],
                    "source": "Bloomberg",
                },
            ],
            "data_quality": "GOOD",
        }
        assert validate_news_report(report) == "OK"


class TestNormalizeBreakingNewsRejectsPlaceholders:
    """_normalize_breaking_news must clear breaking_news when validate returns non-OK."""

    def test_oauth_error_clears_breaking_news(self):
        from trade_modules.committee_synthesis import _normalize_breaking_news

        report = {
            "error": "invalid_grant",
            "error_description": "expired",
            "breaking_news": [{"headline": "x", "impact": "NEUTRAL"}],
        }
        out = _normalize_breaking_news(report)
        assert out["breaking_news"] == []
        assert out.get("news_data_status") == "ERROR"

    def test_hardcoded_placeholder_clears_breaking_news(self):
        from trade_modules.committee_synthesis import _normalize_breaking_news

        report = {
            "breaking_news": [
                {
                    "headline": "Fed maintains rates, signals data-dependent approach for 2026",
                    "impact": "NEUTRAL",
                },
                {
                    "headline": "AI chip demand remains robust despite China export restrictions",
                    "impact": "LOW_POSITIVE",
                },
            ],
        }
        out = _normalize_breaking_news(report)
        assert out["breaking_news"] == []
        assert out.get("news_data_status") == "PLACEHOLDER"

    def test_realistic_report_preserves_breaking_news(self):
        from trade_modules.committee_synthesis import _normalize_breaking_news

        report = {
            "breaking_news": [
                {
                    "headline": "TSMC posts record Q1 revenue on AI chip demand",
                    "impact": "LOW_POSITIVE",
                    "affected_tickers": ["TSM"],
                },
            ],
        }
        out = _normalize_breaking_news(report)
        assert len(out["breaking_news"]) == 1
        assert out.get("news_data_status") == "OK"


class TestRegimeMomentumIgnoresPlaceholderNews:
    """compute_regime_momentum must NOT count placeholder breaking_news entries."""

    def test_placeholder_news_does_not_lift_regime_score(self):
        """The fake LOW_POSITIVE 'AI chip demand' headline must not push score +1."""
        from trade_modules.committee_synthesis import (
            _normalize_breaking_news,
            compute_regime_momentum,
        )

        # Build a baseline news_report with placeholder + neutral other signals
        placeholder = _normalize_breaking_news(
            {
                "breaking_news": [
                    {
                        "headline": "Fed maintains rates, signals data-dependent approach for 2026",
                        "impact": "NEUTRAL",
                    },
                    {
                        "headline": "AI chip demand remains robust despite China export restrictions",
                        "impact": "LOW_POSITIVE",
                    },
                ],
            }
        )

        # Minimal macro/tech/fund inputs — only varying signal is news
        macro = {"executive_summary": {"regime": "NEUTRAL"}, "indicators": {}}
        tech = {"stocks": {}}
        fund = {"stocks": {}}

        # placeholder breaking_news is now empty; regime should not show news lift
        result = compute_regime_momentum(macro, tech, fund, placeholder)
        # The result is a string label; a clean placeholder run should produce STABLE
        assert result == "STABLE", f"placeholder news leaked into regime momentum (got {result})"
