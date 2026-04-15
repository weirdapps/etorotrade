"""
Tests for Committee HTML Report Generator (CIO v10.0).

Validates that the HTML generator produces correct output from
synthesis data without requiring real agent report files.
"""

import pytest

from trade_modules.committee_html import (
    abbr,
    action_color,
    conv_color,
    conv_display,
    e,
    generate_report_html,
    sentiment_color,
    sf,
)


def _minimal_synth():
    """Minimal synthesis dict for testing."""
    return {
        "concordance": [
            {
                "ticker": "AAPL", "signal": "B", "action": "ADD",
                "conviction": 72, "fund_score": 80, "fund_view": "BUY",
                "tech_signal": "ENTER_NOW", "rsi": 55, "macro_fit": "FAVORABLE",
                "census": "ALIGNED", "news_impact": "NEUTRAL",
                "risk_warning": False, "exret": 12.5, "sector": "Technology",
                "bull_pct": 75, "bull_weight": 4.2, "bear_weight": 1.8,
                "beta": 1.1, "max_pct": 5.0, "tech_momentum": 25,
                "is_opportunity": False, "fund_synthetic": False,
            },
            {
                "ticker": "XOM", "signal": "S", "action": "SELL",
                "conviction": 35, "fund_score": 30, "fund_view": "SELL",
                "tech_signal": "AVOID", "rsi": 72, "macro_fit": "UNFAVORABLE",
                "census": "NEUTRAL", "news_impact": "HIGH_NEGATIVE",
                "risk_warning": True, "exret": -5.0, "sector": "Energy",
                "bull_pct": 25, "bull_weight": 1.0, "bear_weight": 5.0,
                "beta": 0.9, "max_pct": 3.0, "tech_momentum": -15,
                "is_opportunity": False, "fund_synthetic": False,
            },
        ],
        "regime": "CAUTIOUS",
        "macro_score": 45,
        "rotation_phase": "MID_CYCLE",
        "risk_score": 55,
        "var_95": 2.1,
        "max_drawdown": -8.5,
        "portfolio_beta": 1.05,
        "fg_top100": 52,
        "fg_broad": 48,
        "changes": [],
        "sector_gaps": [],
        "top_opportunities": [],
        "correlation_clusters": [],
        "concentration": {},
        "stress_scenarios": {},
        "indicators": {"vix": 18.5, "yield_10y": 4.25, "yield_curve_spread": 50},
        "sector_rankings": {"XLK": {"return_1m": 3.2}, "XLE": {"return_1m": -1.5}},
    }


def _minimal_reports():
    """Minimal agent report dicts for testing."""
    fund = {
        "stocks": {
            "AAPL": {"fundamental_score": 80, "pe_trajectory": "IMPROVING",
                      "exret": 12.5, "insider_sentiment": "NET_BUYING", "notes": "Strong"},
        },
        "quality_traps": [],
    }
    tech = {
        "stocks": {
            "AAPL": {"rsi": 55, "macd_signal": "BULLISH", "bb_position": 0.6,
                      "trend": "UPTREND", "momentum_score": 25, "timing_signal": "ENTER_NOW"},
        },
    }
    macro = {}
    census = {"sentiment": {"cash_top100": 10.5}}
    news = {"breaking_news": [], "portfolio_news": {}}
    opps = {"top_opportunities": []}
    risk = {"portfolio_risk": {"sortino_ratio": 1.2}}
    return fund, tech, macro, census, news, opps, risk


class TestHelperFunctions:
    def test_sf_parses_float(self):
        assert sf("12.5%") == pytest.approx(12.5)

    def test_sf_default_on_bad_input(self):
        assert sf("--", 0) == 0

    def test_action_color_known(self):
        assert action_color("SELL") == "#c0392b"
        assert action_color("BUY") == "#2d6a4f"

    def test_action_color_unknown(self):
        assert action_color("UNKNOWN") == "#888888"

    def test_conv_color_thresholds(self):
        assert conv_color(80) == "#2d6a4f"
        assert conv_color(55) == "#b7791f"
        assert conv_color(30) == "#999999"

    def test_sentiment_color_bullish(self):
        assert sentiment_color("ENTER_NOW") == "#2d6a4f"

    def test_sentiment_color_bearish(self):
        assert sentiment_color("AVOID") == "#c0392b"

    def test_html_escape(self):
        assert e("<script>") == "&lt;script&gt;"
        assert e('a"b') == "a&quot;b"

    def test_abbr_known_terms(self):
        assert abbr("ENTER_NOW") == "ENTER"
        assert abbr("UNFAVORABLE") == "UNFAV"
        assert abbr("ALIGNED") == "ALIGN"
        assert abbr("HIGH_NEGATIVE") == "H.NEG"

    def test_abbr_unknown_passthrough(self):
        assert abbr("BUY") == "BUY"
        assert abbr("SELL") == "SELL"
        assert abbr("AVOID") == "AVOID"

    def test_conv_display_with_positive_delta(self):
        html = conv_display(72, delta=5)
        assert "72" in html
        assert "&#9650;" in html
        assert "5" in html

    def test_conv_display_with_negative_delta(self):
        html = conv_display(45, delta=-8)
        assert "45" in html
        assert "&#9660;" in html
        assert "8" in html

    def test_conv_display_no_delta(self):
        html = conv_display(60)
        assert "60" in html
        assert "&#9650;" not in html
        assert "&#9660;" not in html


class TestGenerateReportHtml:
    def test_returns_html_string(self):
        synth = _minimal_synth()
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        assert isinstance(html, str)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_all_sections(self):
        synth = _minimal_synth()
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        assert "Executive Summary" in html
        assert "Macro" in html  # v9.0: "Macro & Market Context"
        assert "Stock Analysis Grid" in html
        assert "Where We Disagreed" in html
        assert "Census" in html
        assert "News" in html
        assert "Portfolio Risk" in html
        assert "Action Items" in html

    def test_tickers_appear_in_output(self):
        synth = _minimal_synth()
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        assert "AAPL" in html
        assert "XOM" in html

    def test_regime_colors(self):
        synth = _minimal_synth()
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        # CAUTIOUS regime
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        assert "CAUTIOUS" in html

        # RISK_ON regime
        synth["regime"] = "RISK_ON"
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        assert "RISK-ON" in html

        # RISK_OFF regime
        synth["regime"] = "RISK_OFF"
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        assert "DEFENSIVE" in html

    def test_custom_date(self):
        synth = _minimal_synth()
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(
            synth, fund, tech, macro, census, news, opps, risk,
            date_str="2026-03-17",
        )
        assert "March 17, 2026" in html

    def test_empty_concordance(self):
        synth = _minimal_synth()
        synth["concordance"] = []
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        assert "<!DOCTYPE html>" in html

    def test_version_string(self):
        synth = _minimal_synth()
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        assert "v33.0" in html

    def test_disclaimer_present(self):
        synth = _minimal_synth()
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        assert "Not financial advice" in html

    def test_action_items_split(self):
        synth = _minimal_synth()
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        assert "Action Items" in html  # v9.0: tiered action format

    def test_exret_in_grid(self):
        synth = _minimal_synth()
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        assert "EXR" in html  # v9.0: shortened column header
        assert "12%" in html  # v9.0: integer format

    def test_exr_column_in_grid(self):
        synth = _minimal_synth()
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        assert "EXR" in html  # v10.0: expected return column

    def test_sector_exposure_shown(self):
        synth = _minimal_synth()
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        assert "Portfolio Risk" in html  # v9.0: sector exposure inside Risk Dashboard
        assert "Technology" in html

    def test_designed_abbreviations_in_grid(self):
        synth = _minimal_synth()
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        # ENTER_NOW should become ENTER, not ENTER_
        assert "ENTER(55)" in html
        # UNFAVORABLE should become UNFAV, not UNFAV
        assert "UNFAV" in html

    def test_dynamic_dates(self):
        synth = _minimal_synth()
        synth["signal_date"] = "2026-03-15"
        synth["census_date"] = "2026-03-14"
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        assert "Signals: 2026-03-15" in html
        assert "Census: 2026-03-14" in html

    def test_conviction_delta_shown(self):
        synth = _minimal_synth()
        synth["changes"] = [
            {"ticker": "AAPL", "delta": 5, "prev_action": "HOLD",
             "curr_action": "ADD", "prev_conviction": 67, "curr_conviction": 72},
        ]
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        assert "&#9650;" in html  # up arrow for AAPL

    def test_hold_stocks_in_monitor_section(self):
        """v9.0: HOLD stocks appear in compact grid, not in removed deep dive section."""
        synth = _minimal_synth()
        synth["concordance"].append({
            "ticker": "MSFT", "signal": "H", "action": "HOLD",
            "conviction": 60, "fund_score": 90, "fund_view": "BUY",
            "tech_signal": "HOLD", "rsi": 50, "macro_fit": "FAVORABLE",
            "census": "ALIGNED", "news_impact": "NEUTRAL",
            "risk_warning": False, "exret": 8.0, "sector": "Technology",
            "bull_pct": 65, "bull_weight": 3.5, "bear_weight": 2.0,
            "beta": 1.0, "max_pct": 5.0, "tech_momentum": 10,
            "is_opportunity": False, "fund_synthetic": False,
        })
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        # MSFT (HOLD) should appear in HOLD section
        assert "HOLD (" in html
        assert "MSFT" in html


class TestV12P2CapitalEfficiency:
    """CIO v12.0 P2: Capital efficiency shown in BUY/ADD action cards."""

    def test_ce_displayed_for_add_items(self):
        """Capital efficiency should appear in ADD action cards."""
        synth = _minimal_synth()
        synth["concordance"][0]["capital_efficiency"] = 15.3
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        assert "CE 15.3" in html

    def test_ce_not_displayed_for_sell(self):
        """Capital efficiency should NOT appear in SELL action cards."""
        synth = _minimal_synth()
        synth["concordance"][1]["capital_efficiency"] = 5.0
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        # SELL card should not have CE
        # XOM is SELL — CE should not appear next to it
        assert "CE 5.0" not in html

    def test_buy_add_sorted_by_ce_tiebreak(self):
        """Among ADD items at same conviction, higher CE should come first."""
        synth = _minimal_synth()
        synth["concordance"] = [
            {
                "ticker": "LOW_CE", "signal": "B", "action": "ADD",
                "conviction": 70, "fund_score": 70, "fund_view": "BUY",
                "tech_signal": "HOLD", "rsi": 50, "macro_fit": "NEUTRAL",
                "census": "NEUTRAL", "news_impact": "NEUTRAL",
                "risk_warning": False, "exret": 5.0, "sector": "Tech",
                "bull_pct": 60, "bull_weight": 3.0, "bear_weight": 2.0,
                "beta": 1.0, "max_pct": 5.0, "tech_momentum": 0,
                "is_opportunity": False, "fund_synthetic": False,
                "capital_efficiency": 3.0,
            },
            {
                "ticker": "HIGH_CE", "signal": "B", "action": "ADD",
                "conviction": 70, "fund_score": 70, "fund_view": "BUY",
                "tech_signal": "HOLD", "rsi": 50, "macro_fit": "NEUTRAL",
                "census": "NEUTRAL", "news_impact": "NEUTRAL",
                "risk_warning": False, "exret": 15.0, "sector": "Health",
                "bull_pct": 60, "bull_weight": 3.0, "bear_weight": 2.0,
                "beta": 1.0, "max_pct": 5.0, "tech_momentum": 0,
                "is_opportunity": False, "fund_synthetic": False,
                "capital_efficiency": 20.0,
            },
        ]
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        # In the Action Items section, HIGH_CE should appear before LOW_CE
        action_section = html[html.index("Action Items"):]
        pos_high = action_section.index("HIGH_CE")
        pos_low = action_section.index("LOW_CE")
        assert pos_high < pos_low


# ============================================================
# CIO v17.0: Track Record section
# ============================================================


class TestTrackRecordSection:
    """Tests for the performance feedback Track Record section."""

    def test_track_record_rendered_when_performance_data_present(self):
        synth = _minimal_synth()
        synth["performance"] = {
            "status": "complete",
            "prev_committee_date": "2026-03-20",
            "total_evaluated": 8,
            "actions": {
                "ADD": {
                    "count": 5, "hit_rate": 60.0, "avg_return": 2.5,
                    "best": {"ticker": "NVDA", "return_pct": 8.3, "conviction": 72, "action": "ADD"},
                    "worst": {"ticker": "MSFT", "return_pct": -2.1, "conviction": 55, "action": "ADD"},
                },
                "HOLD": {"count": 3, "hit_rate": 66.7, "avg_return": 0.3},
            },
        }
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        assert "Track Record" in html
        assert "60%" in html  # hit rate for ADD
        assert "NVDA" in html  # best performer
        assert "MSFT" in html  # worst performer

    def test_track_record_not_rendered_when_no_performance_data(self):
        synth = _minimal_synth()
        # No performance key at all
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        assert "Track Record" not in html

    def test_track_record_not_rendered_when_zero_evaluated(self):
        synth = _minimal_synth()
        synth["performance"] = {"status": "complete", "total_evaluated": 0, "actions": {}}
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        assert "Track Record" not in html

    def test_track_record_not_rendered_when_status_not_complete(self):
        synth = _minimal_synth()
        synth["performance"] = {"status": "no_history"}
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        assert "Track Record" not in html

    def test_track_record_shows_multiple_action_types(self):
        synth = _minimal_synth()
        synth["performance"] = {
            "status": "complete",
            "prev_committee_date": "2026-03-19",
            "total_evaluated": 12,
            "actions": {
                "ADD": {"count": 5, "hit_rate": 80.0, "avg_return": 3.1},
                "HOLD": {"count": 4, "hit_rate": 50.0, "avg_return": 0.5},
                "SELL": {"count": 2, "hit_rate": 100.0, "avg_return": -5.2},
                "TRIM": {"count": 1, "hit_rate": 100.0, "avg_return": -3.0},
            },
        }
        fund, tech, macro, census, news, opps, risk = _minimal_reports()
        html = generate_report_html(synth, fund, tech, macro, census, news, opps, risk)
        assert "ADD (n=5)" in html
        assert "SELL (n=2)" in html
        assert "80%" in html
