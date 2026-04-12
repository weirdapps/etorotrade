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
                                "adx_trend": "STRONG_UP"}}}

def _base_synthesis():
    return {
        "concordance": [
            {"ticker": "NVDA", "action": "HOLD", "conviction": 72,
             "conviction_waterfall": {"base": 60, "tech_bonus": 5},
             "signal": "B", "pet": 37.5, "pef": 16.5},
        ],
        "regime": "CAUTIOUS",
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

    def test_flatten_nested_indicator_dicts(self):
        macro = {"indicators": {
            "vix": {"value": 19.2, "trend": "STABLE"},
            "us_10y": {"value": 4.31, "level": "elevated", "next_move": "hold"},
            "dxy": 98.7,  # already flat — should be preserved
        }}
        fixes = normalize_agent_reports(macro, {}, {}, {}, {})
        ind = macro["indicators"]
        assert ind["vix"] == 19.2
        assert ind["vix_trend"] == "STABLE"
        assert ind["us_10y"] == 4.31
        assert ind["us_10y_level"] == "elevated"
        assert ind["us_10y_next_move"] == "hold"
        assert ind["dxy"] == 98.7  # unchanged
        assert any("flattened nested" in f for f in fixes)

    def test_regime_confidence_fallback(self):
        macro = {"regime_confidence": 75}
        normalize_agent_reports(macro, {}, {}, {}, {})
        assert macro["macro_score"] == 75

    def test_fear_greed_integer_to_dict(self):
        census = {"fear_greed": 60}
        fixes = normalize_agent_reports({}, census, {}, {}, {})
        assert census["fear_greed"] == {"current": 60}
        assert any("fear_greed integer" in f for f in fixes)

    def test_earnings_calendar_list_to_dict(self):
        news = {"earnings_calendar": [
            {"ticker": "NVDA", "date": "2026-04-15"},
        ]}
        fixes = normalize_agent_reports({}, {}, news, {}, {})
        assert "next_2_weeks" in news["earnings_calendar"]
        assert len(news["earnings_calendar"]["next_2_weeks"]) == 1

    def test_rationale_to_why_compelling(self):
        opps = {"top_opportunities": [
            {"ticker": "AVGO", "rationale": "Strong AI play"},
            {"ticker": "V", "rationale": "Payments leader"},
            {"ticker": "MSFT", "why_compelling": "Already set"},  # no override
        ]}
        fixes = normalize_agent_reports({}, {}, {}, {}, opps)
        assert opps["top_opportunities"][0]["why_compelling"] == "Strong AI play"
        assert opps["top_opportunities"][1]["why_compelling"] == "Payments leader"
        assert opps["top_opportunities"][2]["why_compelling"] == "Already set"
        assert any("rationale → why_compelling for 2" in f for f in fixes)

    def test_missing_popular_list_to_dict(self):
        census = {"missing_popular": ["TSLA", "AAPL"]}
        normalize_agent_reports({}, census, {}, {}, {})
        assert census["missing_popular"]["stocks_not_in_portfolio_but_popular"] == ["TSLA", "AAPL"]

    def test_oil_brent_alias(self):
        macro = {"indicators": {"oil_brent": 96.4}}
        normalize_agent_reports(macro, {}, {}, {}, {})
        assert macro["indicators"]["brent_crude"] == 96.4

    def test_regime_string_to_dict(self):
        macro = {"regime": "RISK_ON"}
        normalize_agent_reports(macro, {}, {}, {}, {})
        assert macro["regime"] == {"classification": "RISK_ON"}

    def test_consensus_warnings_dict_to_list(self):
        risk = {"consensus_warnings": {"NVDA": {"warning": "high beta"}}}
        normalize_agent_reports({}, {}, {}, risk, {})
        assert isinstance(risk["consensus_warnings"], list)
        assert risk["consensus_warnings"][0]["ticker"] == "NVDA"

    # ── Grid data normalizations ──

    def test_macro_stocks_to_portfolio_implications(self):
        """Macro agent writes stocks[tkr].macro_fit; synthesis needs portfolio_implications."""
        macro = {"stocks": {
            "NVDA": {"macro_fit": "FAVORABLE", "detail": "AI capex"},
            "AAPL": {"macro_fit": "NEUTRAL"},
        }}
        fixes = normalize_agent_reports(macro, {}, {}, {}, {})
        assert "portfolio_implications" in macro
        assert macro["portfolio_implications"]["NVDA"]["macro_fit"] == "FAVORABLE"
        assert macro["portfolio_implications"]["NVDA"]["fit"] == "FAVORABLE"
        assert macro["portfolio_implications"]["AAPL"]["fit"] == "NEUTRAL"
        assert any("stocks → portfolio_implications" in f for f in fixes)

    def test_macro_stock_macro_fit_string_to_portfolio_implications(self):
        """Macro agent writes stock_macro_fit[tkr] = 'FAVORABLE' string."""
        macro = {"stock_macro_fit": {
            "NVDA": "FAVORABLE",
            "TSLA": "UNFAVORABLE",
        }}
        fixes = normalize_agent_reports(macro, {}, {}, {}, {})
        assert macro["portfolio_implications"]["NVDA"]["macro_fit"] == "FAVORABLE"
        assert macro["portfolio_implications"]["TSLA"]["fit"] == "UNFAVORABLE"

    def test_census_divergences_from_per_stock_sentiment(self):
        """Census agent writes per-stock sentiment; synthesis needs structured divergences."""
        census = {"stocks": {
            "NVDA": {"sentiment": "ALIGNED", "div_score": 30},
            "TSLA": {"sentiment": "CONTRARIAN", "div_score": 85},
            "AAPL": {"sentiment": "BULLISH", "divergence_score": 20},
            "META": {"sentiment": "DIVERGENT", "div_score": 70},
        }}
        fixes = normalize_agent_reports({}, census, {}, {}, {})
        divs = census["divergences"]
        assert isinstance(divs, dict)
        assert len(divs["signal_divergences"]) == 2  # TSLA, META
        assert len(divs["consensus_aligned"]) == 2   # NVDA, AAPL
        tickers_div = {d["ticker"] for d in divs["signal_divergences"]}
        assert "TSLA" in tickers_div
        assert "META" in tickers_div

    def test_news_portfolio_news_from_earnings_calendar(self):
        """News agent has earnings_calendar but no portfolio_news; should build it."""
        news = {"earnings_calendar": {"next_2_weeks": [
            {"ticker": "NVDA", "date": "2026-04-15"},
            {"ticker": "AAPL", "date": "2026-04-25"},
        ]}}
        fixes = normalize_agent_reports({}, {}, news, {}, {})
        pn = news["portfolio_news"]
        assert "NVDA" in pn
        assert pn["NVDA"][0]["type"] == "earnings"
        assert any("built portfolio_news" in f for f in fixes)

    def test_tech_field_name_normalization(self):
        """Tech agent writes tech_signal/trend_strength; synthesis needs timing_signal/trend."""
        tech = {"stocks": {
            "NVDA": {"tech_signal": "BUY", "trend_strength": "STRONG_UP",
                     "tech_momentum": 72},
            "AAPL": {"tech_signal": "HOLD", "trend_strength": "NEUTRAL"},
            "TSLA": {"tech_signal": "AVOID", "trend_strength": "DOWN"},
        }}
        fixes = normalize_agent_reports({}, {}, {}, {}, {}, tech=tech)
        assert tech["stocks"]["NVDA"]["timing_signal"] == "ENTER_NOW"
        assert tech["stocks"]["NVDA"]["trend"] == "STRONG_UPTREND"
        assert tech["stocks"]["NVDA"]["momentum_score"] == 22.0  # 72 - 50
        assert tech["stocks"]["AAPL"]["timing_signal"] == "HOLD"
        assert tech["stocks"]["AAPL"]["trend"] == "CONSOLIDATION"
        assert tech["stocks"]["TSLA"]["timing_signal"] == "AVOID"
        assert tech["stocks"]["TSLA"]["trend"] == "WEAK_DOWNTREND"

    def test_run_qa_passes_tech_fund_to_normalize(self):
        """run_qa should pass tech and fund through to normalize_agent_reports."""
        tech = {"stocks": {
            "NVDA": {"tech_signal": "BUY", "trend_strength": "STRONG_UP"},
        }}
        synth = _base_synthesis()
        passed, gaps = run_qa(synth, _base_fund(), tech,
                              _base_macro(), _base_census(), _base_news(),
                              _base_opps(), _base_risk(), normalize=True)
        # Tech normalization should have been applied
        assert tech["stocks"]["NVDA"]["timing_signal"] == "ENTER_NOW"
        # And logged
        info = [g for g in gaps if g["section"] == "Normalize"]
        assert any("tech:" in g["message"] for g in info)

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

    def test_unknown_regime_is_critical(self):
        synth = _base_synthesis()
        synth["regime"] = "UNKNOWN"
        gaps = validate_pre_html(synth, _base_fund(), _base_tech(),
                                 _base_macro(), _base_census(), _base_news(),
                                 _base_opps(), _base_risk())
        crit = [g for g in gaps if g["severity"] == CRITICAL]
        assert any("regime" in g["field"] for g in crit)

    def test_zero_macro_score_warned(self):
        synth = _base_synthesis()
        synth["macro_score"] = 0
        gaps = validate_pre_html(synth, _base_fund(), _base_tech(),
                                 _base_macro(), _base_census(), _base_news(),
                                 _base_opps(), _base_risk())
        warns = [g for g in gaps if g["severity"] == WARNING]
        assert any("macro_score" in g["field"] for g in warns)

    def test_unknown_rotation_warned(self):
        synth = _base_synthesis()
        synth["rotation"] = "Unknown"
        gaps = validate_pre_html(synth, _base_fund(), _base_tech(),
                                 _base_macro(), _base_census(), _base_news(),
                                 _base_opps(), _base_risk())
        warns = [g for g in gaps if g["severity"] == WARNING]
        assert any("rotation" in g["field"] for g in warns)

    def test_missing_vix_warned(self):
        synth = _base_synthesis()
        synth["indicators"]["vix"] = 0
        gaps = validate_pre_html(synth, _base_fund(), _base_tech(),
                                 _base_macro(), _base_census(), _base_news(),
                                 _base_opps(), _base_risk())
        warns = [g for g in gaps if g["severity"] == WARNING]
        assert any("vix" in g["field"] for g in warns)

    def test_nested_indicators_warned(self):
        synth = _base_synthesis()
        synth["indicators"]["gold"] = {"value": 2400, "trend": "UP"}
        gaps = validate_pre_html(synth, _base_fund(), _base_tech(),
                                 _base_macro(), _base_census(), _base_news(),
                                 _base_opps(), _base_risk())
        warns = [g for g in gaps if g["severity"] == WARNING]
        assert any("nested" in g["field"] for g in warns)

    def test_missing_pe_in_concordance_warned(self):
        synth = _base_synthesis()
        # Make >30% of equities missing PE
        synth["concordance"] = [
            {"ticker": f"T{i}", "action": "HOLD", "conviction": 60,
             "signal": "B", "conviction_waterfall": {"base": 60}}
            for i in range(5)
        ]
        gaps = validate_pre_html(synth, _base_fund(), _base_tech(),
                                 _base_macro(), _base_census(), _base_news(),
                                 _base_opps(), _base_risk())
        warns = [g for g in gaps if g["severity"] == WARNING]
        assert any("pe_data" in g["field"] for g in warns)

    def test_missing_why_compelling_is_critical(self):
        opps = {"top_opportunities": [
            {"ticker": "AVGO"},  # no why_compelling or rationale
        ]}
        synth = _base_synthesis()
        gaps = validate_pre_html(synth, _base_fund(), _base_tech(),
                                 _base_macro(), _base_census(), _base_news(),
                                 opps, _base_risk())
        crit = [g for g in gaps if g["severity"] == CRITICAL]
        assert any("why_compelling" in g["field"] for g in crit)

    def test_no_action_in_concordance_is_critical(self):
        synth = _base_synthesis()
        synth["concordance"] = [{"ticker": "NVDA", "conviction": 72, "signal": "B"}]
        gaps = validate_pre_html(synth, _base_fund(), _base_tech(),
                                 _base_macro(), _base_census(), _base_news(),
                                 _base_opps(), _base_risk())
        crit = [g for g in gaps if g["severity"] == CRITICAL]
        assert any("action" in g["field"] for g in crit)

# ---------------------------------------------------------------------------
# Stage 2: Post-HTML Validate
# ---------------------------------------------------------------------------

class TestPostHTML:

    def test_clean_html_passes(self):
        html = "<h2>Executive Summary</h2><h2>Macro &amp; Market Context</h2>"
        html += "<h2>News &amp; Events</h2>" + "x" * 600
        html += "<h2>Technical Analysis</h2>" + "x" * 900
        html += "<h2>Fundamental Deep Dive</h2>" + "x" * 1100
        html += "<h2>Sentiment &amp; Census</h2>"
        gaps = validate_post_html(html)
        crit = [g for g in gaps if g["severity"] == CRITICAL]
        assert len(crit) == 0

    def test_many_na_cells_warned(self):
        html = ">N/A</span>" * 8
        # Add required sections to avoid CRITICAL
        for s in ["Executive Summary", "Macro &amp; Market Context",
                   "News &amp; Events", "Technical Analysis",
                   "Fundamental Deep Dive", "Sentiment &amp; Census"]:
            html += f"<h2>{s}</h2>" + "x" * 1200
        gaps = validate_post_html(html)
        warns = [g for g in gaps if g["severity"] == WARNING]
        assert any("N/A" in g["field"] for g in warns)

    def test_unknown_values_critical(self):
        html = ">Unknown</span>" * 2
        for s in ["Executive Summary", "Macro &amp; Market Context",
                   "News &amp; Events", "Technical Analysis",
                   "Fundamental Deep Dive", "Sentiment &amp; Census"]:
            html += f"<h2>{s}</h2>" + "x" * 1200
        gaps = validate_post_html(html)
        crit = [g for g in gaps if g["severity"] == CRITICAL]
        assert any("Unknown" in g["field"] for g in crit)

    def test_empty_spans_warned(self):
        html = 'color:#64748b;"></span></td>' * 6
        for s in ["Executive Summary", "Macro &amp; Market Context",
                   "News &amp; Events", "Technical Analysis",
                   "Fundamental Deep Dive", "Sentiment &amp; Census"]:
            html += f"<h2>{s}</h2>" + "x" * 1200
        gaps = validate_post_html(html)
        warns = [g for g in gaps if g["severity"] == WARNING]
        assert any("empty" in g["field"].lower() for g in warns)

    def test_missing_section_critical(self):
        html = "<h2>Executive Summary</h2>"  # missing other sections
        gaps = validate_post_html(html)
        crit = [g for g in gaps if g["severity"] == CRITICAL]
        assert len(crit) >= 1  # at least one missing section

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

    def test_normalize_fixes_rationale(self):
        """Normalization should fix rationale→why_compelling so it's not flagged."""
        opps = {"top_opportunities": [
            {"ticker": "AVGO", "rationale": "Strong AI play",
             "opportunity_score": 85, "exret": 24.6, "buy_pct": 71},
        ]}
        synth = _base_synthesis()
        passed, gaps = run_qa(synth, _base_fund(), _base_tech(),
                              _base_macro(), _base_census(), _base_news(),
                              opps, _base_risk(), normalize=True)
        # Should be fixed by normalization, not flagged as critical
        crit = [g for g in gaps if g["severity"] == CRITICAL
                and "why_compelling" in g.get("field", "")]
        assert len(crit) == 0
        # And the fix should be logged
        info = [g for g in gaps if g["section"] == "Normalize"]
        assert any("rationale" in g["message"] for g in info)

    def test_normalize_fixes_key_indicators(self):
        """Normalization should fix key_indicators→indicators."""
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
