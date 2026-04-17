"""Tests for committee_qa.py — Stage 0/1/2 QA validation."""
from trade_modules.committee_qa import (
    normalize_agent_reports,
    validate_pre_html,
    validate_post_html,
    run_qa,
    format_qa_report,
    CRITICAL,
    WARNING,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _base_macro():
    return {
        "regime": "CAUTIOUS",
        "macro_score": 65,
        "rotation_phase": "LATE_CYCLE",
        "indicators": {"vix": 19.2, "us_10y_yield": 4.31, "eur_usd": 1.08,
                        "dxy": 98.7, "brent_crude": 96.4},
        "sector_rankings": {"Technology": "Overweight"},
    }

def _base_census():
    return {
        "stocks": {"NVDA": {"holders_pct": 45}},
        "cash_trends": {"mean_cash_pct": 12.5},
    }

def _base_news():
    return {
        "breaking_news": [{"headline": "Fed holds rates"}],
        "earnings_calendar": {"next_2_weeks": [{"ticker": "NVDA"}]},
    }

def _base_risk():
    return {
        "portfolio_risk": {"var_95": 2.0, "max_drawdown": 0.1,
                           "risk_score": 55, "portfolio_beta": 1.05},
        "correlation_clusters": [["NVDA", "AMD"]],
    }

def _base_opps():
    return {
        "top_opportunities": [
            {"ticker": "AVGO", "why_compelling": "Strong AI play",
             "opportunity_score": 85, "exret": 24.6, "buy_pct": 71,
             "pe_forward": 19.7, "sector": "Technology"},
        ],
    }

def _base_fund():
    return {"stocks": {"NVDA": {"fundamental_score": 80,
                                "key_metrics": {"pe": 37}}}}

def _base_tech():
    return {"stocks": {"NVDA": {"rsi": 62.5, "trend": "BULLISH",
                                "adx_trend": "STRONG_UP",
                                "timing_signal": "ENTER_NOW",
                                "momentum_score": 65,
                                "macd_signal": "BULLISH"}}}

def _base_synthesis():
    return {
        "concordance": [
            {"ticker": "NVDA", "action": "HOLD", "conviction": 72,
             "conviction_waterfall": {"base": 60, "tech_bonus": 5},
             "signal": "B", "pet": 37.5, "pef": 16.5},
        ],
        "regime": "CAUTIOUS",
        "risk_score": 55,
        "macro_score": 65,
        "rotation": "Late cycle positioning",
        "portfolio_beta": 1.05,
        "indicators": {"vix": 19.2, "us_10y_yield": 4.31,
                        "eur_usd": 1.08, "dxy": 98.7, "brent_crude": 96.4},
        "sector_rankings": {"Technology": "Overweight"},
        "breaking_news": [{"headline": "Fed holds rates"}],
        "earnings_calendar": {"next_2_weeks": [{"ticker": "NVDA"}]},
    }

# ---------------------------------------------------------------------------
# Stage 0: Normalize
# ---------------------------------------------------------------------------

class TestNormalize:

    def test_key_indicators_alias(self):
        macro = {"key_indicators": {"vix": 19.2}}
        fixes = normalize_agent_reports(macro, {}, {}, {}, {})
        assert "indicators" in macro
        assert macro["indicators"]["vix"] == 19.2
        assert any("key_indicators" in f for f in fixes)

    def test_key_indicators_not_overwritten(self):
        """If indicators already exists, key_indicators should not overwrite it."""
        macro = {"key_indicators": {"vix": 10}, "indicators": {"vix": 19.2}}
        normalize_agent_reports(macro, {}, {}, {}, {})
        assert macro["indicators"]["vix"] == 19.2

    def test_stock_macro_fit_string_to_portfolio_implications(self):
        macro = {"stock_macro_fit": {"NVDA": "FAVORABLE", "TSLA": "UNFAVORABLE"}}
        fixes = normalize_agent_reports(macro, {}, {}, {}, {})
        assert "portfolio_implications" in macro
        assert macro["portfolio_implications"]["NVDA"]["fit"] == "FAVORABLE"
        assert macro["portfolio_implications"]["TSLA"]["fit"] == "UNFAVORABLE"
        assert any("stock_macro_fit" in f for f in fixes)

    def test_stock_macro_fit_dict_to_portfolio_implications(self):
        macro = {"stock_macro_fit": {"NVDA": {"fit": "FAVORABLE", "rationale": "AI capex"}}}
        normalize_agent_reports(macro, {}, {}, {}, {})
        assert macro["portfolio_implications"]["NVDA"]["fit"] == "FAVORABLE"

    def test_stock_macro_fit_not_overwritten(self):
        """If portfolio_implications already exists, stock_macro_fit should not overwrite."""
        macro = {
            "portfolio_implications": {"NVDA": {"fit": "NEUTRAL"}},
            "stock_macro_fit": {"NVDA": "FAVORABLE"},
        }
        normalize_agent_reports(macro, {}, {}, {}, {})
        assert macro["portfolio_implications"]["NVDA"]["fit"] == "NEUTRAL"

    def test_flatten_yield_curve(self):
        macro = {"indicators": {"yield_curve": {"10y": 4.31, "spread_2_10": 0.25}}}
        fixes = normalize_agent_reports(macro, {}, {}, {}, {})
        ind = macro["indicators"]
        assert ind["us_10y_yield"] == 4.31
        assert ind["yield_curve_10y_2y"] == 0.25
        assert any("yield_curve" in f for f in fixes)

    def test_flatten_currency(self):
        macro = {"indicators": {"currency": {"dxy": 98.7, "eur_usd": 1.08}}}
        fixes = normalize_agent_reports(macro, {}, {}, {}, {})
        ind = macro["indicators"]
        assert ind["dxy"] == 98.7
        assert ind["eur_usd"] == 1.08
        assert any("currency" in f for f in fixes)

    def test_oil_brent_alias(self):
        macro = {"indicators": {"oil_brent": 96.4}}
        fixes = normalize_agent_reports(macro, {}, {}, {}, {})
        assert macro["indicators"]["brent_crude"] == 96.4
        assert any("oil_brent" in f for f in fixes)

    def test_brent_crude_not_overwritten(self):
        macro = {"indicators": {"oil_brent": 96.4, "brent_crude": 100.0}}
        normalize_agent_reports(macro, {}, {}, {}, {})
        assert macro["indicators"]["brent_crude"] == 100.0

    def test_indicators_written_to_both_keys(self):
        macro = {"indicators": {"vix": 19.2}}
        normalize_agent_reports(macro, {}, {}, {}, {})
        assert macro["indicators"] == macro["macro_indicators"]

    def test_regime_string_to_dict(self):
        macro = {"regime": "RISK_ON"}
        fixes = normalize_agent_reports(macro, {}, {}, {}, {})
        assert macro["regime"] == {"classification": "RISK_ON"}
        assert any("regime" in f for f in fixes)

    def test_regime_dict_unchanged(self):
        macro = {"regime": {"classification": "CAUTIOUS", "confidence": 80}}
        normalize_agent_reports(macro, {}, {}, {}, {})
        assert macro["regime"]["classification"] == "CAUTIOUS"

    def test_missing_popular_list_to_dict(self):
        census = {"missing_popular": ["TSLA", "AAPL"]}
        fixes = normalize_agent_reports({}, census, {}, {}, {})
        assert census["missing_popular"]["stocks_not_in_portfolio_but_popular"] == ["TSLA", "AAPL"]
        assert any("missing_popular" in f for f in fixes)

    def test_per_stock_to_stocks_alias(self):
        census = {"per_stock": {"NVDA": {"holders_pct": 45}}}
        fixes = normalize_agent_reports({}, census, {}, {}, {})
        assert census["stocks"] == census["per_stock"]
        assert any("per_stock" in f for f in fixes)

    def test_stocks_to_per_stock_alias(self):
        census = {"stocks": {"NVDA": {"holders_pct": 45}}}
        normalize_agent_reports({}, census, {}, {}, {})
        assert census["per_stock"] == census["stocks"]

    def test_market_news_to_breaking_news(self):
        news = {"market_news": [{"headline": "Rate cut"}]}
        fixes = normalize_agent_reports({}, {}, news, {}, {})
        assert news["breaking_news"] == [{"headline": "Rate cut"}]
        assert any("market_news" in f for f in fixes)

    def test_portfolio_news_list_to_dict(self):
        news = {"portfolio_news": [
            {"ticker": "NVDA", "headline": "Earnings beat"},
            {"ticker": "AAPL", "headline": "New product"},
        ]}
        fixes = normalize_agent_reports({}, {}, news, {}, {})
        assert isinstance(news["portfolio_news"], dict)
        assert news["portfolio_news"]["NVDA"]["headline"] == "Earnings beat"
        assert any("portfolio_news" in f for f in fixes)

    def test_consensus_warnings_dict_to_list(self):
        risk = {"consensus_warnings": {"NVDA": {"warning": "high beta"}}}
        fixes = normalize_agent_reports({}, {}, {}, risk, {})
        assert isinstance(risk["consensus_warnings"], list)
        assert risk["consensus_warnings"][0]["ticker"] == "NVDA"
        assert any("consensus_warnings" in f for f in fixes)

    def test_consensus_warnings_string_values(self):
        risk = {"consensus_warnings": {"NVDA": "high beta"}}
        normalize_agent_reports({}, {}, {}, risk, {})
        assert risk["consensus_warnings"][0]["warning"] == "high beta"

    def test_top_opportunities_alternatives(self):
        for alt_key in ("opportunities", "screened_stocks", "results"):
            opps = {alt_key: [{"ticker": "AVGO"}]}
            fixes = normalize_agent_reports({}, {}, {}, {}, opps)
            assert opps["top_opportunities"] == [{"ticker": "AVGO"}]
            assert any(alt_key in f for f in fixes)

    def test_no_fixes_returns_empty(self):
        fixes = normalize_agent_reports({}, {}, {}, {}, {})
        assert fixes == []

    def test_risk_warnings_by_stock_normalized(self):
        """v33.0: risk_warnings_by_stock ingested into consensus_warnings."""
        risk = {"risk_warnings_by_stock": {"NVDA": {"severity": "HIGH", "warning": "Concentration"}}}
        fixes = normalize_agent_reports({}, {}, {}, risk, {})
        assert any("risk_warnings_by_stock" in f for f in fixes)
        assert any(w.get("ticker") == "NVDA" for w in risk.get("consensus_warnings", [])
                   if isinstance(w, dict))

    def test_risk_warnings_by_stock_string_values(self):
        """v33.0: String value in risk_warnings_by_stock."""
        risk = {"risk_warnings_by_stock": {"AAPL": "high beta exposure"}}
        normalize_agent_reports({}, {}, {}, risk, {})
        cw = risk.get("consensus_warnings", [])
        assert any(w.get("ticker") == "AAPL" and "high beta" in w.get("reason", "")
                   for w in cw if isinstance(w, dict))

    def test_risk_warnings_by_stock_no_duplicates(self):
        """v33.0: Ticker already in consensus_warnings is not doubled."""
        risk = {
            "consensus_warnings": [{"ticker": "NVDA", "severity": "HIGH"}],
            "risk_warnings_by_stock": {"NVDA": {"severity": "MODERATE"}},
        }
        normalize_agent_reports({}, {}, {}, risk, {})
        nvda_count = sum(1 for w in risk["consensus_warnings"]
                         if isinstance(w, dict) and w.get("ticker") == "NVDA")
        assert nvda_count == 1

# ---------------------------------------------------------------------------
# Stage 1: Pre-HTML Validate
# ---------------------------------------------------------------------------

class TestPreHTML:

    def test_clean_data_passes(self):
        synth = _base_synthesis()
        gaps = validate_pre_html(synth, _base_fund(), _base_tech(),
                                 _base_macro(), _base_census(), _base_news(),
                                 _base_opps(), _base_risk())
        criticals = [g for g in gaps if g["severity"] == CRITICAL]
        assert len(criticals) == 0

    def test_empty_concordance_is_critical(self):
        synth = _base_synthesis()
        synth["concordance"] = []
        gaps = validate_pre_html(synth, _base_fund(), _base_tech(),
                                 _base_macro(), _base_census(), _base_news(),
                                 _base_opps(), _base_risk())
        crit = [g for g in gaps if g["severity"] == CRITICAL]
        assert any("concordance" in g["field"] for g in crit)

    def test_missing_regime_warned(self):
        synth = _base_synthesis()
        synth["regime"] = ""
        gaps = validate_pre_html(synth, _base_fund(), _base_tech(),
                                 _base_macro(), _base_census(), _base_news(),
                                 _base_opps(), _base_risk())
        warns = [g for g in gaps if g["severity"] == WARNING]
        assert any("regime" in g["field"] for g in warns)

    def test_missing_vix_warned(self):
        synth = _base_synthesis()
        synth["indicators"]["vix"] = 0
        gaps = validate_pre_html(synth, _base_fund(), _base_tech(),
                                 _base_macro(), _base_census(), _base_news(),
                                 _base_opps(), _base_risk())
        warns = [g for g in gaps if g["severity"] == WARNING]
        assert any("vix" in g["field"] for g in warns)

    def test_missing_10y_yield_warned(self):
        synth = _base_synthesis()
        synth["indicators"]["us_10y_yield"] = None
        gaps = validate_pre_html(synth, _base_fund(), _base_tech(),
                                 _base_macro(), _base_census(), _base_news(),
                                 _base_opps(), _base_risk())
        warns = [g for g in gaps if g["severity"] == WARNING]
        assert any("us_10y_yield" in g["field"] for g in warns)

    def test_missing_sector_rankings_warned(self):
        synth = _base_synthesis()
        synth.pop("sector_rankings", None)
        gaps = validate_pre_html(synth, _base_fund(), _base_tech(),
                                 _base_macro(), _base_census(), _base_news(),
                                 _base_opps(), _base_risk())
        warns = [g for g in gaps if g["severity"] == WARNING]
        assert any("sector_rankings" in g["field"] for g in warns)

    def test_no_fund_stocks_critical(self):
        gaps = validate_pre_html(_base_synthesis(), {"stocks": {}}, _base_tech(),
                                 _base_macro(), _base_census(), _base_news(),
                                 _base_opps(), _base_risk())
        crit = [g for g in gaps if g["severity"] == CRITICAL]
        assert any("stocks" in g["field"] for g in crit)

    def test_no_tech_stocks_critical(self):
        gaps = validate_pre_html(_base_synthesis(), _base_fund(), {"stocks": {}},
                                 _base_macro(), _base_census(), _base_news(),
                                 _base_opps(), _base_risk())
        crit = [g for g in gaps if g["severity"] == CRITICAL]
        assert any("stocks" in g["field"] for g in crit)

    def test_missing_why_compelling_warned(self):
        opps = {"top_opportunities": [{"ticker": "AVGO"}]}
        gaps = validate_pre_html(_base_synthesis(), _base_fund(), _base_tech(),
                                 _base_macro(), _base_census(), _base_news(),
                                 opps, _base_risk())
        warns = [g for g in gaps if g["severity"] == WARNING]
        assert any("why_compelling" in g["field"] for g in warns)

    def test_no_action_in_concordance_is_critical(self):
        synth = _base_synthesis()
        synth["concordance"] = [{"ticker": "NVDA", "conviction": 72, "signal": "B"}]
        gaps = validate_pre_html(synth, _base_fund(), _base_tech(),
                                 _base_macro(), _base_census(), _base_news(),
                                 _base_opps(), _base_risk())
        crit = [g for g in gaps if g["severity"] == CRITICAL]
        assert any("action" in g["field"] for g in crit)

    def test_default_portfolio_beta_warned(self):
        synth = _base_synthesis()
        synth["portfolio_beta"] = 1.0
        gaps = validate_pre_html(synth, _base_fund(), _base_tech(),
                                 _base_macro(), _base_census(), _base_news(),
                                 _base_opps(), _base_risk())
        warns = [g for g in gaps if g["severity"] == WARNING]
        assert any("portfolio_beta" in g["field"] for g in warns)

    def test_no_breaking_news_warned(self):
        synth = _base_synthesis()
        synth["breaking_news"] = []
        gaps = validate_pre_html(synth, _base_fund(), _base_tech(),
                                 _base_macro(), _base_census(), _base_news(),
                                 _base_opps(), _base_risk())
        warns = [g for g in gaps if g["severity"] == WARNING]
        assert any("breaking_news" in g["field"] for g in warns)

# ---------------------------------------------------------------------------
# Stage 2: Post-HTML Validate
# ---------------------------------------------------------------------------

class TestPostHTML:

    def _full_html(self, extra=""):
        html = "<h2>Executive Summary</h2>" + "x" * 200
        html += "<h2>Macro &amp; Market Context</h2>" + "x" * 200
        html += "<h2>News &amp; Events</h2>" + "x" * 600
        html += "<h2>Technical Analysis</h2>" + "x" * 900
        html += "<h2>Fundamental Deep Dive</h2>" + "x" * 1100
        html += "<h2>Sentiment &amp; Census</h2>" + "x" * 200
        return extra + html

    def test_clean_html_passes(self):
        gaps = validate_post_html(self._full_html())
        crit = [g for g in gaps if g["severity"] == CRITICAL]
        assert len(crit) == 0

    def test_many_na_cells_warned(self):
        gaps = validate_post_html(self._full_html(">N/A</span>" * 25))
        warns = [g for g in gaps if g["severity"] == WARNING]
        assert any("N/A" in g["field"] for g in warns)

    def test_few_na_cells_ok(self):
        gaps = validate_post_html(self._full_html(">N/A</span>" * 10))
        na_warns = [g for g in gaps if "N/A" in g.get("field", "")]
        assert len(na_warns) == 0

    def test_empty_spans_warned(self):
        gaps = validate_post_html(
            self._full_html('color:#64748b;"> </span></td>' * 6)
        )
        warns = [g for g in gaps if g["severity"] == WARNING]
        assert any("empty" in g["field"].lower() for g in warns)

    def test_missing_section_critical(self):
        html = "<h2>Executive Summary</h2>"
        gaps = validate_post_html(html)
        crit = [g for g in gaps if g["severity"] == CRITICAL]
        assert len(crit) >= 1

    def test_short_section_warned(self):
        html = "<h2>Executive Summary</h2>" + "x" * 200
        html += "<h2>Macro &amp; Market Context</h2>" + "x" * 200
        html += "<h2>News &amp; Events</h2>" + "x" * 50  # too short
        html += "<h2>Technical Analysis</h2>" + "x" * 900
        html += "<h2>Fundamental Deep Dive</h2>" + "x" * 1100
        html += "<h2>Sentiment &amp; Census</h2>"
        gaps = validate_post_html(html)
        warns = [g for g in gaps if g["severity"] == WARNING]
        assert any("News" in g.get("field", "") for g in warns)

# ---------------------------------------------------------------------------
# Integration: run_qa
# ---------------------------------------------------------------------------

class TestRunQA:

    def test_clean_run_passes(self):
        synth = _base_synthesis()
        passed, gaps = run_qa(synth, _base_fund(), _base_tech(),
                              _base_macro(), _base_census(), _base_news(),
                              _base_opps(), _base_risk(), normalize=False)
        assert passed is True

    def test_normalize_fixes_key_indicators(self):
        macro = {"key_indicators": {"vix": 19.2, "us_10y_yield": 4.31,
                                    "eur_usd": 1.08, "dxy": 98.7,
                                    "brent_crude": 96.4},
                 "regime": "CAUTIOUS", "macro_score": 65,
                 "sector_rankings": {}}
        synth = _base_synthesis()
        passed, gaps = run_qa(synth, _base_fund(), _base_tech(),
                              macro, _base_census(), _base_news(),
                              _base_opps(), _base_risk(), normalize=True)
        assert "indicators" in macro
        info = [g for g in gaps if g["section"] == "Normalize"]
        assert any("key_indicators" in g["message"] for g in info)

    def test_normalize_off_skips_fixes(self):
        macro = {"key_indicators": {"vix": 19.2}}
        synth = _base_synthesis()
        passed, gaps = run_qa(synth, _base_fund(), _base_tech(),
                              macro, _base_census(), _base_news(),
                              _base_opps(), _base_risk(), normalize=False)
        assert "key_indicators" in macro  # not renamed
        info = [g for g in gaps if g["section"] == "Normalize"]
        assert len(info) == 0

    def test_format_qa_report_no_gaps(self):
        report = format_qa_report([])
        assert "PASSED" in report

    def test_format_qa_report_with_criticals(self):
        gaps = [{"severity": CRITICAL, "section": "Exec", "field": "regime",
                 "message": "UNKNOWN"}]
        report = format_qa_report(gaps)
        assert "FAILED" in report

    def test_format_qa_report_with_warnings(self):
        gaps = [{"severity": WARNING, "section": "Macro", "field": "vix",
                 "message": "VIX is zero"}]
        report = format_qa_report(gaps)
        assert "PASSED with warnings" in report


# ---------------------------------------------------------------------------
# v33.0: Signal channel uniformity checks
# ---------------------------------------------------------------------------

class TestSignalChannelQA:
    """v33.0: QA catches broken signal channels."""

    def test_all_neutral_news_impact_is_critical(self):
        synth = _base_synthesis()
        synth["concordance"] = [
            {"ticker": f"T{i}", "action": "HOLD", "conviction": 60,
             "signal": "H", "news_impact": "NEUTRAL"}
            for i in range(15)
        ]
        from trade_modules.committee_qa import validate_synthesis_completeness
        gaps = validate_synthesis_completeness(synth)
        crit = [g for g in gaps if g["severity"] == CRITICAL and "news_impact" in g["field"]]
        assert len(crit) >= 1

    def test_mixed_news_impact_passes(self):
        synth = _base_synthesis()
        synth["concordance"] = [
            {"ticker": "NVDA", "action": "HOLD", "conviction": 60,
             "signal": "H", "news_impact": "LOW_POSITIVE"},
            {"ticker": "AAPL", "action": "ADD", "conviction": 70,
             "signal": "B", "news_impact": "NEUTRAL"},
        ]
        from trade_modules.committee_qa import validate_synthesis_completeness
        gaps = validate_synthesis_completeness(synth)
        news_crit = [g for g in gaps if "news_impact" in g.get("field", "")]
        assert len(news_crit) == 0

    def test_all_neutral_census_is_critical(self):
        synth = _base_synthesis()
        synth["concordance"] = [
            {"ticker": f"T{i}", "action": "HOLD", "conviction": 60,
             "signal": "H", "census": "NEUTRAL"}
            for i in range(15)
        ]
        from trade_modules.committee_qa import validate_synthesis_completeness
        gaps = validate_synthesis_completeness(synth)
        crit = [g for g in gaps if g["severity"] == CRITICAL and "census" in g["field"]]
        assert len(crit) >= 1

    def test_all_false_risk_warning_is_warning(self):
        synth = _base_synthesis()
        synth["concordance"] = [
            {"ticker": f"T{i}", "action": "HOLD", "conviction": 60,
             "signal": "H", "risk_warning": False}
            for i in range(20)
        ]
        from trade_modules.committee_qa import validate_synthesis_completeness
        gaps = validate_synthesis_completeness(synth)
        warns = [g for g in gaps if g["severity"] == WARNING and "risk_warning" in g["field"]]
        assert len(warns) >= 1

    def test_some_risk_warnings_no_alert(self):
        synth = _base_synthesis()
        synth["concordance"] = [
            {"ticker": f"T{i}", "action": "HOLD", "conviction": 60,
             "signal": "H", "risk_warning": i < 3}
            for i in range(20)
        ]
        from trade_modules.committee_qa import validate_synthesis_completeness
        gaps = validate_synthesis_completeness(synth)
        rw = [g for g in gaps if "risk_warning" in g.get("field", "")]
        assert len(rw) == 0
