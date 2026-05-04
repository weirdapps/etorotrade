"""
Characterization tests for committee_qa.normalize_agent_reports (cog-complexity 244).

These tests capture the CURRENT behavior of the most complex function in the codebase
(line 31, cognitive complexity 244) to enable safe refactoring. They document what
the code DOES, not what it SHOULD do — the goal is regression detection.

Each test exercises a specific code path or combination of conditions and asserts
the EXACT output observed from the current implementation.
"""

import pytest

from trade_modules.committee_qa import normalize_agent_reports


@pytest.mark.characterization
class TestNormalizeAgentReportsCharacterization:
    """Characterization tests for the cog-244 normalize_agent_reports function."""

    def test_macro_stocks_extracts_macro_fit_when_portfolio_implications_missing(self):
        """
        Characterization: BUG 2 fix (lines 63-82). When macro["stocks"][ticker]
        contains macro_fit but portfolio_implications is empty, extracts fit from
        stocks dict and populates portfolio_implications.
        """
        macro = {
            "stocks": {
                "AAA": {"macro_fit": "FAVORABLE", "notes": "AI growth"},
                "BBB": {"fit": "UNFAVORABLE", "rationale": "Cyclical risk"},
                "CCC": {"macro_fit": "NEUTRAL"},  # No notes/rationale
            }
        }
        fixes = normalize_agent_reports(macro, {}, {}, {}, {})

        assert "portfolio_implications" in macro
        assert macro["portfolio_implications"]["AAA"]["fit"] == "FAVORABLE"
        assert macro["portfolio_implications"]["AAA"]["rationale"] == "AI growth"
        assert macro["portfolio_implications"]["BBB"]["fit"] == "UNFAVORABLE"
        assert macro["portfolio_implications"]["BBB"]["rationale"] == "Cyclical risk"
        assert macro["portfolio_implications"]["CCC"]["fit"] == "NEUTRAL"
        assert macro["portfolio_implications"]["CCC"]["rationale"] == ""
        assert any("extracted macro_fit from stocks" in f for f in fixes)

    def test_macro_stocks_skips_tickers_already_in_portfolio_implications(self):
        """
        Characterization: Lines 71-72. When portfolio_implications already has an
        entry for a ticker, it is NOT overwritten by stocks[ticker].macro_fit.
        """
        macro = {
            "portfolio_implications": {"AAA": {"fit": "NEUTRAL", "rationale": "Existing"}},
            "stocks": {"AAA": {"macro_fit": "FAVORABLE", "notes": "New data"}},
        }
        normalize_agent_reports(macro, {}, {}, {}, {})

        # Existing entry is preserved
        assert macro["portfolio_implications"]["AAA"]["fit"] == "NEUTRAL"
        assert macro["portfolio_implications"]["AAA"]["rationale"] == "Existing"

    def test_macro_indicators_nested_yield_curve_flattening(self):
        """
        Characterization: Lines 91-97. When indicators.yield_curve exists as dict,
        flattens 10y and spread_2_10 into top-level us_10y_yield and yield_curve_10y_2y.
        Uses setdefault so existing values are NOT overwritten.
        """
        macro = {
            "indicators": {
                "yield_curve": {"10y": 4.31, "spread_2_10": 0.25},
                "us_10y_yield": 4.50,  # pre-existing, should NOT be overwritten
            }
        }
        normalize_agent_reports(macro, {}, {}, {}, {})

        assert macro["indicators"]["us_10y_yield"] == pytest.approx(4.50)  # unchanged
        assert macro["indicators"]["yield_curve_10y_2y"] == pytest.approx(0.25)  # added

    def test_macro_indicators_nested_currency_flattening(self):
        """
        Characterization: Lines 98-102. When indicators.currency exists as dict,
        flattens dxy and eur_usd into top-level fields.
        """
        macro = {"indicators": {"currency": {"dxy": 98.7, "eur_usd": 1.08}}}
        normalize_agent_reports(macro, {}, {}, {}, {})

        assert macro["indicators"]["dxy"] == pytest.approx(98.7)
        assert macro["indicators"]["eur_usd"] == pytest.approx(1.08)

    def test_news_synthesize_breaking_from_key_themes_string_list(self):
        """
        Characterization: Lines 182-202. When breaking_news is missing and key_themes
        is a list of strings, synthesizes breaking_news items with defaults.
        """
        news = {"key_themes": ["Fed holds rates", "Tech rally continues"]}
        fixes = normalize_agent_reports({}, {}, news, {}, {})

        assert "breaking_news" in news
        assert len(news["breaking_news"]) == 2
        assert news["breaking_news"][0]["headline"] == "Fed holds rates"
        assert news["breaking_news"][0]["impact"] == "NEUTRAL"
        assert news["breaking_news"][0]["affected_tickers"] == []
        assert any("synthesised breaking_news" in f for f in fixes)

    def test_news_synthesize_breaking_from_key_themes_dict_list(self):
        """
        Characterization: Lines 191-199. When key_themes contains dicts with
        theme/title/headline fields, extracts headline and maps sentiment to impact.
        """
        news = {
            "key_themes": [
                {"theme": "Fed pivot", "sentiment": "POSITIVE", "tickers": ["AAA"]},
                {"title": "Earnings beat", "impact": "HIGH_POSITIVE"},
                {"headline": "Guidance cut"},  # No sentiment, defaults to NEUTRAL
            ]
        }
        normalize_agent_reports({}, {}, news, {}, {})

        assert news["breaking_news"][0]["headline"] == "Fed pivot"
        assert news["breaking_news"][0]["impact"] == "POSITIVE"
        assert news["breaking_news"][0]["affected_tickers"] == ["AAA"]

        assert news["breaking_news"][1]["headline"] == "Earnings beat"
        assert news["breaking_news"][1]["impact"] == "HIGH_POSITIVE"

        assert news["breaking_news"][2]["headline"] == "Guidance cut"
        assert news["breaking_news"][2]["impact"] == "NEUTRAL"

    def test_macro_extract_eur_usd_from_narrative_context(self):
        """
        Characterization: Lines 229-236. When indicators.eur_usd is missing,
        attempts regex extraction from fx_impact or risk_indicators.usd_context.
        Handles "EUR/USD at 1.08" pattern.
        """
        macro = {
            "indicators": {},
            "fx_impact": "EUR/USD at 1.0845 amid dollar weakness",
        }
        fixes = normalize_agent_reports(macro, {}, {}, {}, {})

        assert macro["indicators"]["eur_usd"] == pytest.approx(1.0845)
        assert any("extracted eur_usd" in f for f in fixes)

    def test_macro_extract_10y_yield_from_narrative_context(self):
        """
        Characterization: Lines 237-243. When us_10y_yield is missing, extracts
        from risk_indicators.yield_curve_context using regex for "10-year Treasury near 4.31%".
        """
        macro = {
            "indicators": {},
            "risk_indicators": {"yield_curve_context": "10-year Treasury near 4.31%"},
        }
        fixes = normalize_agent_reports(macro, {}, {}, {}, {})

        assert macro["indicators"]["us_10y_yield"] == pytest.approx(4.31)
        assert any("extracted us_10y_yield" in f for f in fixes)

    def test_risk_derive_stress_scenarios_from_tail_risks(self):
        """
        Characterization: Lines 248-265. When stress_scenarios is missing but
        tail_risks exists as list of dicts, builds scenarios with impact mapped
        from CATASTROPHIC/HIGH/MEDIUM/LOW to -25/-15/-10/-5.
        Takes first 3 tail risks only.
        """
        risk = {
            "tail_risks": [
                {"risk": "Trade war escalation", "impact": "CATASTROPHIC", "probability": "LOW"},
                {"name": "Banking crisis", "impact": "HIGH", "probability": "VERY_LOW"},
                {"risk": "Supply shock", "impact": "MEDIUM"},
                {"risk": "Ignored fourth risk"},  # Should not appear
            ]
        }
        fixes = normalize_agent_reports({}, {}, {}, risk, {})

        assert len(risk["stress_scenarios"]) == 3
        assert risk["stress_scenarios"][0]["name"] == "Trade war escalation"
        assert risk["stress_scenarios"][0]["portfolio_impact_pct"] == -25
        assert risk["stress_scenarios"][0]["probability"] == "LOW"

        assert risk["stress_scenarios"][1]["name"] == "Banking crisis"
        assert risk["stress_scenarios"][1]["portfolio_impact_pct"] == -15

        assert risk["stress_scenarios"][2]["name"] == "Supply shock"
        assert risk["stress_scenarios"][2]["portfolio_impact_pct"] == -10
        assert risk["stress_scenarios"][2]["probability"] == "MEDIUM"  # default

        assert any("derived stress_scenarios" in f for f in fixes)

    def test_risk_derive_stress_scenarios_from_tail_risks_string_list(self):
        """
        Characterization: Lines 261-262. When tail_risks contains strings instead
        of dicts, creates scenarios with default -10% impact and MEDIUM probability.
        """
        risk = {"tail_risks": ["Recession", "Geopolitical shock"]}
        normalize_agent_reports({}, {}, {}, risk, {})

        assert risk["stress_scenarios"][0]["name"] == "Recession"
        assert risk["stress_scenarios"][0]["portfolio_impact_pct"] == -10
        assert risk["stress_scenarios"][0]["probability"] == "MEDIUM"

    def test_macro_derive_portfolio_implications_from_sector_rankings(self):
        """
        Characterization: Lines 270-294. When portfolio_implications is missing
        but sector_rankings exists, builds per-sector fit by mapping outlook to
        fit (OVERWEIGHT→FAVORABLE, UNDERWEIGHT→UNFAVORABLE, NEUTRAL→NEUTRAL).
        Also enriches sector_rankings with fit field and rank based on outlook order.
        """
        macro = {
            "sector_rankings": {
                "Technology": {"recommendation": "OVERWEIGHT", "notes": "AI growth"},
                "Energy": {"outlook": "UNDERWEIGHT", "thesis": "Transition risk"},
                "Healthcare": {"outlook": "NEUTRAL"},
            }
        }
        fixes = normalize_agent_reports(macro, {}, {}, {}, {})

        # Check enrichment (fit field added to each sector)
        assert macro["sector_rankings"]["Technology"]["fit"] == "FAVORABLE"
        assert macro["sector_rankings"]["Energy"]["fit"] == "UNFAVORABLE"
        assert macro["sector_rankings"]["Healthcare"]["fit"] == "NEUTRAL"

        # Check ranking (OVERWEIGHT=1, NEUTRAL=2, UNDERWEIGHT=3)
        assert macro["sector_rankings"]["Technology"]["rank"] == 1
        assert macro["sector_rankings"]["Healthcare"]["rank"] == 2
        assert macro["sector_rankings"]["Energy"]["rank"] == 3

        assert any("enriched sector_rankings" in f for f in fixes)

    def test_census_derive_alignment_from_trend_accumulating(self):
        """
        Characterization: Lines 308-311. When census.stocks[ticker] has trend=
        ACCUMULATING/STRONG_ACCUMULATION but no alignment field, sets alignment
        to "ACCUMULATING".
        """
        census = {
            "stocks": {
                "AAA": {"trend": "ACCUMULATING"},
                "BBB": {"trend": "STRONG_ACCUMULATION"},
            }
        }
        fixes = normalize_agent_reports({}, census, {}, {}, {})

        assert census["stocks"]["AAA"]["alignment"] == "ACCUMULATING"
        assert census["stocks"]["BBB"]["alignment"] == "ACCUMULATING"
        assert any("derived alignment from trend" in f for f in fixes)

    def test_census_derive_alignment_from_trend_distributing(self):
        """
        Characterization: Lines 312-314. When trend=DISTRIBUTING/DISTRIBUTION,
        sets alignment to "DISTRIBUTING".
        """
        census = {
            "stocks": {
                "AAA": {"trend": "DISTRIBUTING"},
                "BBB": {"trend": "STRONG_DISTRIBUTION"},
            }
        }
        normalize_agent_reports({}, census, {}, {}, {})

        assert census["stocks"]["AAA"]["alignment"] == "DISTRIBUTING"
        assert census["stocks"]["BBB"]["alignment"] == "DISTRIBUTING"

    def test_census_derive_alignment_from_interest_level_high_stable(self):
        """
        Characterization: Lines 315-317. When trend=STABLE and interest_level=
        HIGH or MODERATE, sets alignment to "CONSENSUS_ALIGNED".
        """
        census = {
            "stocks": {
                "AAA": {"trend": "STABLE", "interest_level": "HIGH"},
                "BBB": {"trend": "STABLE", "interest_level": "MODERATE"},
            }
        }
        normalize_agent_reports({}, census, {}, {}, {})

        assert census["stocks"]["AAA"]["alignment"] == "CONSENSUS_ALIGNED"
        assert census["stocks"]["BBB"]["alignment"] == "CONSENSUS_ALIGNED"

    def test_news_derive_portfolio_news_from_stock_news(self):
        """
        Characterization: Lines 324-347. When portfolio_news is missing but
        stock_news exists, derives portfolio_news by mapping news_sentiment
        and impact_magnitude to impact codes. POSITIVE+HIGH→HIGH_POSITIVE,
        NEGATIVE+HIGH→HIGH_NEGATIVE, MIXED+HIGH→LOW_POSITIVE, else NEUTRAL.
        """
        news = {
            "stock_news": {
                "AAA": {
                    "news_sentiment": "VERY_POSITIVE",
                    "impact_magnitude": "HIGH",
                    "recent_headlines": ["Earnings beat", "Guidance raise"],
                },
                "BBB": {
                    "news_sentiment": "NEGATIVE",
                    "impact_magnitude": "LOW",
                    "recent_headlines": ["Miss"],
                },
                "CCC": {
                    "news_sentiment": "MIXED",
                    "impact_magnitude": "HIGH",
                },
            }
        }
        fixes = normalize_agent_reports({}, {}, news, {}, {})

        assert news["portfolio_news"]["AAA"][0]["impact"] == "HIGH_POSITIVE"
        assert "Earnings beat; Guidance raise" in news["portfolio_news"]["AAA"][0]["headline"]

        assert news["portfolio_news"]["BBB"][0]["impact"] == "LOW_NEGATIVE"

        assert news["portfolio_news"]["CCC"][0]["impact"] == "LOW_POSITIVE"

        assert any("derived portfolio_news from stock_news" in f for f in fixes)

    def test_risk_derive_risk_warning_from_risk_level(self):
        """
        Characterization: Lines 352-363. When risk.stocks[ticker] has no
        risk_warning field but has risk_level=HIGH or EXTREME, sets
        risk_warning=True. Otherwise False.
        """
        risk = {
            "stocks": {
                "AAA": {"risk_level": "HIGH"},
                "BBB": {"risk_level": "EXTREME"},
                "CCC": {"risk_level": "MODERATE"},
            }
        }
        fixes = normalize_agent_reports({}, {}, {}, risk, {})

        assert risk["stocks"]["AAA"]["risk_warning"] is True
        assert risk["stocks"]["BBB"]["risk_warning"] is True
        assert risk["stocks"]["CCC"]["risk_warning"] is False
        assert any("derived risk_warning" in f for f in fixes)

    def test_risk_consensus_warnings_string_reset_to_empty_list(self):
        """
        Characterization: Lines 366-369. If consensus_warnings or risk_warnings_by_stock
        is a string (malformed agent output), resets to empty list.
        """
        risk = {
            "consensus_warnings": "High volatility across portfolio",
            "risk_warnings_by_stock": "See details in attachment",
        }
        fixes = normalize_agent_reports({}, {}, {}, risk, {})

        assert risk["consensus_warnings"] == []
        assert risk["risk_warnings_by_stock"] == []
        assert any("consensus_warnings was string" in f for f in fixes)
        assert any("risk_warnings_by_stock was string" in f for f in fixes)

    def test_complex_multi_agent_normalization_cascade(self):
        """
        Characterization: Tests multiple normalizations firing in sequence,
        ensuring they compose correctly without interfering with each other.
        Observed behavior: sector_rankings enrichment does NOT run when
        portfolio_implications already exists (from stocks extraction).
        """
        macro = {
            "key_indicators": {"vix": 19.2},
            "regime": "CAUTIOUS",
            "stocks": {"AAA": {"macro_fit": "FAVORABLE", "notes": "Growth"}},
            "sector_rankings": {"Tech": {"recommendation": "OVERWEIGHT"}},
        }
        census = {
            "missing_popular": ["BBB", "CCC"],
            "per_stock": {"AAA": {"trend": "ACCUMULATING"}},
        }
        news = {
            "market_news": [{"headline": "Rate hold"}],
            "key_themes": ["Fed pause"],
        }
        risk = {
            "consensus_warnings": {"AAA": "concentration"},
            "tail_risks": [{"risk": "Recession", "impact": "HIGH"}],
            "stocks": {"AAA": {"risk_level": "EXTREME"}},
        }
        opps = {"opportunities": [{"ticker": "DDD"}]}

        fixes = normalize_agent_reports(macro, census, news, risk, opps)

        # Macro normalizations
        assert "indicators" in macro
        assert macro["indicators"]["vix"] == pytest.approx(19.2)
        assert macro["regime"]["classification"] == "CAUTIOUS"
        assert "portfolio_implications" in macro
        assert macro["portfolio_implications"]["AAA"]["fit"] == "FAVORABLE"
        # sector_rankings NOT enriched because portfolio_implications exists
        assert "fit" not in macro["sector_rankings"]["Tech"]

        # Census normalizations
        assert census["missing_popular"]["stocks_not_in_portfolio_but_popular"] == ["BBB", "CCC"]
        assert census["stocks"] == census["per_stock"]
        assert census["stocks"]["AAA"]["alignment"] == "ACCUMULATING"

        # News normalizations
        assert news["breaking_news"][0]["headline"] == "Rate hold"

        # Risk normalizations
        assert isinstance(risk["consensus_warnings"], list)
        assert risk["consensus_warnings"][0]["ticker"] == "AAA"
        assert len(risk["stress_scenarios"]) == 1
        assert risk["stocks"]["AAA"]["risk_warning"] is True

        # Opps normalization
        assert opps["top_opportunities"][0]["ticker"] == "DDD"

        # Multiple fixes reported
        assert len(fixes) >= 8

    def test_empty_inputs_returns_empty_fixes(self):
        """
        Characterization: When all agent reports are empty dicts, no normalization
        is needed and returns empty fixes list.
        """
        fixes = normalize_agent_reports({}, {}, {}, {}, {})
        assert fixes == []

    def test_macro_risk_indicators_merge_into_indicators(self):
        """
        Characterization: Lines 212-222. When both risk_indicators and indicators
        exist, merges fields from risk_indicators into indicators without overwriting.
        """
        macro = {
            "indicators": {"vix": 19.2, "dxy": 98.7},
            "risk_indicators": {"vix": 25.0, "credit_spreads": 150},
        }
        fixes = normalize_agent_reports(macro, {}, {}, {}, {})

        assert macro["indicators"]["vix"] == pytest.approx(19.2)  # not overwritten
        assert macro["indicators"]["credit_spreads"] == 150  # merged
        assert any("merged" in f and "risk_indicators" in f for f in fixes)

    def test_risk_warnings_by_stock_list_values(self):
        """
        Characterization: Lines 174-176. When risk_warnings_by_stock[ticker]
        is a list, joins items with "; " and truncates to 200 chars.
        """
        risk = {
            "risk_warnings_by_stock": {"AAA": ["High beta", "Concentration", "Valuation stretched"]}
        }
        normalize_agent_reports({}, {}, {}, risk, {})

        cw = risk["consensus_warnings"]
        assert cw[0]["ticker"] == "AAA"
        assert cw[0]["reason"] == "High beta; Concentration; Valuation stretched"
        assert cw[0]["severity"] == "MODERATE"

    def test_news_stock_news_catalyst_type_preserved(self):
        """
        Characterization: Lines 344. When stock_news[ticker] has catalyst_type,
        it is preserved in the derived portfolio_news.
        """
        news = {
            "stock_news": {
                "AAA": {
                    "news_sentiment": "POSITIVE",
                    "impact_magnitude": "LOW",
                    "catalyst_type": "EARNINGS",
                    "recent_headlines": [],
                }
            }
        }
        normalize_agent_reports({}, {}, news, {}, {})

        assert news["portfolio_news"]["AAA"][0]["catalyst"] == "EARNINGS"

    def test_macro_indicators_written_to_both_keys_after_normalization(self):
        """
        Characterization: Lines 107-109, 244-245. After ALL indicator normalizations,
        the final indicators dict is written to both "indicators" and "macro_indicators".
        """
        macro = {
            "key_indicators": {"vix": 19.2},
            "risk_indicators": {"credit_spreads": 150},
        }
        normalize_agent_reports(macro, {}, {}, {}, {})

        assert macro["indicators"] == macro["macro_indicators"]
        assert macro["indicators"]["vix"] == pytest.approx(19.2)
        assert macro["indicators"]["credit_spreads"] == 150

    def test_census_stocks_non_dict_values_skipped(self):
        """
        Characterization: Lines 303-304. When census.stocks[ticker] is not a dict
        (e.g., None, string, list), the alignment derivation skips it without error.
        """
        census = {
            "stocks": {
                "AAA": None,
                "BBB": "invalid",
                "CCC": {"trend": "ACCUMULATING"},
            }
        }
        normalize_agent_reports({}, census, {}, {}, {})

        # Only CCC gets alignment
        assert "alignment" in census["stocks"]["CCC"]
        assert "alignment" not in (census["stocks"]["AAA"] or {})

    def test_news_stock_news_non_dict_values_skipped(self):
        """
        Characterization: Lines 329-330. When stock_news[ticker] is not a dict,
        portfolio_news derivation skips it.
        """
        news = {
            "stock_news": {
                "AAA": None,
                "BBB": {"news_sentiment": "POSITIVE", "impact_magnitude": "HIGH"},
            }
        }
        normalize_agent_reports({}, {}, news, {}, {})

        assert "AAA" not in news["portfolio_news"]
        assert "BBB" in news["portfolio_news"]

    def test_risk_stocks_non_dict_values_skipped(self):
        """
        Characterization: Lines 355-356. When risk.stocks[ticker] is not a dict,
        risk_warning derivation skips it.
        """
        risk = {
            "stocks": {
                "AAA": None,
                "BBB": {"risk_level": "HIGH"},
            }
        }
        normalize_agent_reports({}, {}, {}, risk, {})

        assert "risk_warning" not in (risk["stocks"]["AAA"] or {})
        assert risk["stocks"]["BBB"]["risk_warning"] is True
