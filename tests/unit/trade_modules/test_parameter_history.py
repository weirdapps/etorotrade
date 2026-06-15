"""Tests for trade_modules/parameter_history.py

Covers: _num(), _write_record(), every _write_* source writer,
save_parameter_history() end-to-end, and get_parameter_history() filtering.
"""

import json

import pytest

import trade_modules.parameter_history as ph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _redirect_history(tmp_path, monkeypatch):
    """Route HISTORY_PATH to a temp dir so tests never touch the real file."""
    monkeypatch.setattr(ph, "HISTORY_PATH", tmp_path / "parameter_history.jsonl")


@pytest.fixture()
def history_file(tmp_path):
    return tmp_path / "parameter_history.jsonl"


# ---------------------------------------------------------------------------
# _num()
# ---------------------------------------------------------------------------


class TestNum:
    def test_none(self):
        assert ph._num(None) is None

    def test_int(self):
        assert ph._num(42) == 42

    def test_float(self):
        assert ph._num(3.14) == pytest.approx(3.14)

    def test_negative_float(self):
        assert ph._num(-1.5) == pytest.approx(-1.5)

    def test_zero(self):
        assert ph._num(0) == 0

    def test_str_numeric(self):
        assert ph._num("12.5") == pytest.approx(12.5)

    def test_str_with_whitespace(self):
        assert ph._num("  7  ") == pytest.approx(7.0)

    def test_str_negative(self):
        assert ph._num("-3.2") == pytest.approx(-3.2)

    def test_str_non_numeric(self):
        assert ph._num("abc") is None

    def test_str_empty(self):
        assert ph._num("") is None

    def test_dict_with_value(self):
        assert ph._num({"value": 55}) == 55

    def test_dict_with_score(self):
        assert ph._num({"score": 88}) == 88

    def test_dict_with_value_and_score(self):
        """'value' takes precedence over 'score'."""
        assert ph._num({"value": 10, "score": 20}) == 10

    def test_dict_empty(self):
        assert ph._num({}) is None

    def test_dict_no_value_or_score(self):
        assert ph._num({"other": 5}) is None

    def test_bool_true(self):
        # bool is subclass of int in Python
        assert ph._num(True) == 1

    def test_bool_false(self):
        assert ph._num(False) == 0


# ---------------------------------------------------------------------------
# _write_record()
# ---------------------------------------------------------------------------


class TestWriteRecord:
    def test_writes_json_line(self, history_file):
        record = {"date": "2026-01-01", "ticker": "AAPL", "source": "test"}
        with open(history_file, "a") as f:
            result = ph._write_record(f, record)
        assert result == 1
        line = history_file.read_text().strip()
        parsed = json.loads(line)
        assert parsed == record

    def test_appends_newline(self, history_file):
        with open(history_file, "a") as f:
            ph._write_record(f, {"a": 1})
            ph._write_record(f, {"b": 2})
        lines = history_file.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_non_serializable_uses_default_str(self, history_file):
        from datetime import datetime as dt

        record = {"ts": dt(2026, 1, 1, 12, 0)}
        with open(history_file, "a") as f:
            ph._write_record(f, record)
        parsed = json.loads(history_file.read_text().strip())
        assert isinstance(parsed["ts"], str)


# ---------------------------------------------------------------------------
# _write_signals()
# ---------------------------------------------------------------------------


class TestWriteSignals:
    def test_full_data(self, history_file):
        signals = {
            "AAPL": {
                "price": 180.0,
                "upside": 12.5,
                "buy_pct": 85.0,
                "exret": 0.03,
                "beta": 1.2,
                "pet": 15.0,
                "pef": 14.0,
                "pp": 0.9,
                "52w": 0.75,
                "am": 3,
                "analyst_type": "STRONG",
                "signal": "BUY",
                "short_interest": 1.5,
                "roe": 0.45,
                "de": 1.8,
                "fcf": 2.1,
                "num_targets": 30,
            }
        }
        with open(history_file, "a") as f:
            result = ph._write_signals(f, "2026-01-01", "AAPL", signals)
        assert result == 1
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["source"] == "signals"
        assert parsed["ticker"] == "AAPL"
        assert parsed["price"] == 180.0
        assert parsed["signal"] == "BUY"
        assert parsed["num_targets"] == 30

    def test_missing_ticker(self, history_file):
        with open(history_file, "a") as f:
            result = ph._write_signals(f, "2026-01-01", "MSFT", {"AAPL": {}})
        assert result == 0
        assert not history_file.exists() or history_file.read_text() == ""

    def test_empty_signals(self, history_file):
        with open(history_file, "a") as f:
            result = ph._write_signals(f, "2026-01-01", "AAPL", {})
        assert result == 0

    def test_none_values_excluded(self, history_file):
        signals = {"AAPL": {"price": 100.0, "upside": None, "beta": None}}
        with open(history_file, "a") as f:
            ph._write_signals(f, "2026-01-01", "AAPL", signals)
        parsed = json.loads(history_file.read_text().strip())
        assert "price" in parsed
        assert "upside" not in parsed
        assert "beta" not in parsed

    def test_partial_data(self, history_file):
        signals = {"TSLA": {"price": 250.0, "signal": "HOLD"}}
        with open(history_file, "a") as f:
            result = ph._write_signals(f, "2026-01-01", "TSLA", signals)
        assert result == 1
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["price"] == 250.0
        assert "upside" not in parsed

    def test_empty_dict_for_ticker_returns_zero(self, history_file):
        """An empty dict is falsy, so `not sig` returns True."""
        signals = {"AAPL": {}}
        with open(history_file, "a") as f:
            result = ph._write_signals(f, "2026-01-01", "AAPL", signals)
        assert result == 0


# ---------------------------------------------------------------------------
# _write_fundamental()
# ---------------------------------------------------------------------------


class TestWriteFundamental:
    def test_full_data(self, history_file):
        fund = {
            "stocks": {
                "AAPL": {
                    "fundamental_score": 82,
                    "outlook": "positive",
                    "pe_trajectory": "declining",
                    "quality_trap": False,
                    "piotroski_score": 7,
                    "revenue_growth_class": "high",
                    "eps_revisions": 0.05,
                    "insider_sentiment": "bullish",
                    "earnings_quality": "high",
                }
            }
        }
        with open(history_file, "a") as f:
            result = ph._write_fundamental(f, "2026-01-01", "AAPL", fund)
        assert result == 1
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["source"] == "fundamental"
        assert parsed["fundamental_score"] == 82

    def test_missing_stocks_key(self, history_file):
        with open(history_file, "a") as f:
            result = ph._write_fundamental(f, "2026-01-01", "AAPL", {})
        assert result == 0

    def test_missing_ticker(self, history_file):
        fund = {"stocks": {"MSFT": {"fundamental_score": 50}}}
        with open(history_file, "a") as f:
            result = ph._write_fundamental(f, "2026-01-01", "AAPL", fund)
        assert result == 0

    def test_stock_value_not_dict(self, history_file):
        """If stock entry is a non-dict (e.g. a string), should return 0."""
        fund = {"stocks": {"AAPL": "invalid"}}
        with open(history_file, "a") as f:
            result = ph._write_fundamental(f, "2026-01-01", "AAPL", fund)
        assert result == 0

    def test_stock_value_none(self, history_file):
        fund = {"stocks": {"AAPL": None}}
        with open(history_file, "a") as f:
            result = ph._write_fundamental(f, "2026-01-01", "AAPL", fund)
        assert result == 0

    def test_none_values_excluded(self, history_file):
        fund = {"stocks": {"AAPL": {"fundamental_score": 70, "outlook": None}}}
        with open(history_file, "a") as f:
            ph._write_fundamental(f, "2026-01-01", "AAPL", fund)
        parsed = json.loads(history_file.read_text().strip())
        assert "fundamental_score" in parsed
        assert "outlook" not in parsed


# ---------------------------------------------------------------------------
# _write_technical()
# ---------------------------------------------------------------------------


class TestWriteTechnical:
    def test_full_data(self, history_file):
        tech = {
            "stocks": {
                "AAPL": {
                    "price": 180.0,
                    "rsi": 55,
                    "macd_signal": "bullish",
                    "macd_histogram": 0.5,
                    "bb_position": 0.6,
                    "above_sma50": True,
                    "above_sma200": True,
                    "golden_cross": True,
                    "vol_ratio": 1.2,
                    "support": 170.0,
                    "resistance": 190.0,
                    "momentum_score": 75,
                    "trend": "up",
                    "timing_signal": "BUY",
                    "relative_strength_vs_spy": 1.05,
                    "atr_pct": 2.1,
                    "adx": 28,
                    "adx_trend": "strengthening",
                }
            }
        }
        with open(history_file, "a") as f:
            result = ph._write_technical(f, "2026-01-01", "AAPL", tech)
        assert result == 1
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["source"] == "technical"
        assert parsed["rsi"] == 55

    def test_missing_ticker(self, history_file):
        tech = {"stocks": {}}
        with open(history_file, "a") as f:
            result = ph._write_technical(f, "2026-01-01", "AAPL", tech)
        assert result == 0

    def test_stock_not_dict(self, history_file):
        tech = {"stocks": {"AAPL": 42}}
        with open(history_file, "a") as f:
            result = ph._write_technical(f, "2026-01-01", "AAPL", tech)
        assert result == 0


# ---------------------------------------------------------------------------
# _write_macro()
# ---------------------------------------------------------------------------


class TestWriteMacro:
    def test_full_data(self, history_file):
        macro = {
            "regime": "expansion",
            "macro_score": 72,
            "rotation_phase": "early_cycle",
            "portfolio_implications": {
                "AAPL": {
                    "macro_fit": "high",
                    "rate_sensitive": False,
                    "dollar_impact": "neutral",
                }
            },
        }
        with open(history_file, "a") as f:
            result = ph._write_macro(f, "2026-01-01", "AAPL", macro)
        assert result == 1
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["source"] == "macro"
        assert parsed["regime"] == "expansion"
        assert parsed["macro_score"] == 72
        assert parsed["macro_fit"] == "high"

    def test_missing_portfolio_implications(self, history_file):
        macro = {"regime": "contraction"}
        with open(history_file, "a") as f:
            result = ph._write_macro(f, "2026-01-01", "AAPL", macro)
        assert result == 0

    def test_missing_ticker_in_implications(self, history_file):
        macro = {"portfolio_implications": {"MSFT": {"macro_fit": "low"}}}
        with open(history_file, "a") as f:
            result = ph._write_macro(f, "2026-01-01", "AAPL", macro)
        assert result == 0

    def test_implications_not_dict(self, history_file):
        macro = {"portfolio_implications": {"AAPL": "invalid"}}
        with open(history_file, "a") as f:
            result = ph._write_macro(f, "2026-01-01", "AAPL", macro)
        assert result == 0

    def test_partial_implication_keys(self, history_file):
        macro = {
            "regime": None,
            "macro_score": None,
            "rotation_phase": None,
            "portfolio_implications": {"AAPL": {"macro_fit": "medium"}},
        }
        with open(history_file, "a") as f:
            ph._write_macro(f, "2026-01-01", "AAPL", macro)
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["macro_fit"] == "medium"
        # None regime is still written (no None-exclusion logic for top-level macro keys)
        assert parsed["regime"] is None


# ---------------------------------------------------------------------------
# _write_census()
# ---------------------------------------------------------------------------


class TestWriteCensus:
    def test_full_sentiment(self, history_file):
        census = {
            "sentiment": {
                "fg_top100": 65,
                "fg_broad": 58,
                "cash_top100": 12.5,
                "cash_broad": 15.0,
            }
        }
        with open(history_file, "a") as f:
            result = ph._write_census(f, "2026-01-01", "AAPL", census)
        assert result == 1
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["source"] == "census"
        assert parsed["fg_top100"] == 65

    def test_no_sentiment_key(self, history_file):
        """Census always writes a record, even without sentiment."""
        with open(history_file, "a") as f:
            result = ph._write_census(f, "2026-01-01", "AAPL", {})
        assert result == 1
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["fg_top100"] is None
        assert parsed["fg_broad"] is None

    def test_sentiment_values_as_dicts(self, history_file):
        """_num should extract 'value' from dict-wrapped sentiment."""
        census = {
            "sentiment": {
                "fg_top100": {"value": 70},
                "fg_broad": {"score": 55},
                "cash_top100": None,
                "cash_broad": "10.5",
            }
        }
        with open(history_file, "a") as f:
            ph._write_census(f, "2026-01-01", "AAPL", census)
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["fg_top100"] == 70
        assert parsed["fg_broad"] == 55
        assert parsed["cash_top100"] is None
        assert parsed["cash_broad"] == pytest.approx(10.5)


# ---------------------------------------------------------------------------
# _write_news()
# ---------------------------------------------------------------------------


class TestWriteNews:
    def test_neutral_default(self, history_file):
        news = {}
        with open(history_file, "a") as f:
            result = ph._write_news(f, "2026-01-01", "AAPL", news)
        assert result == 1
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["news_impact"] == "NEUTRAL"
        assert parsed["earnings_days_away"] is None
        assert parsed["news_count"] == 0

    def test_negative_impact(self, history_file):
        news = {
            "portfolio_news": {
                "AAPL": [
                    {"impact": "POSITIVE", "headline": "good"},
                    {"impact": "NEGATIVE", "headline": "bad"},
                ]
            }
        }
        with open(history_file, "a") as f:
            ph._write_news(f, "2026-01-01", "AAPL", news)
        parsed = json.loads(history_file.read_text().strip())
        # NEGATIVE overrides POSITIVE
        assert parsed["news_impact"] == "NEGATIVE"
        assert parsed["news_count"] == 2

    def test_positive_impact_only(self, history_file):
        news = {"portfolio_news": {"AAPL": [{"impact": "POSITIVE", "headline": "up"}]}}
        with open(history_file, "a") as f:
            ph._write_news(f, "2026-01-01", "AAPL", news)
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["news_impact"] == "POSITIVE"

    def test_neutral_only_articles(self, history_file):
        news = {"portfolio_news": {"AAPL": [{"impact": "NEUTRAL"}]}}
        with open(history_file, "a") as f:
            ph._write_news(f, "2026-01-01", "AAPL", news)
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["news_impact"] == "NEUTRAL"
        assert parsed["news_count"] == 1

    def test_earnings_days_away(self, history_file):
        news = {
            "earnings_calendar": [
                {"ticker": "MSFT", "days_away": 10},
                {"ticker": "AAPL", "days_away": 5},
            ]
        }
        with open(history_file, "a") as f:
            ph._write_news(f, "2026-01-01", "AAPL", news)
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["earnings_days_away"] == 5

    def test_earnings_days_away_invalid(self, history_file):
        news = {
            "earnings_calendar": [
                {"ticker": "AAPL", "days_away": "N/A"},
            ]
        }
        with open(history_file, "a") as f:
            ph._write_news(f, "2026-01-01", "AAPL", news)
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["earnings_days_away"] is None

    def test_earnings_days_away_none(self, history_file):
        news = {
            "earnings_calendar": [
                {"ticker": "AAPL", "days_away": None},
            ]
        }
        with open(history_file, "a") as f:
            ph._write_news(f, "2026-01-01", "AAPL", news)
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["earnings_days_away"] is None

    def test_no_matching_ticker_in_earnings(self, history_file):
        news = {"earnings_calendar": [{"ticker": "MSFT", "days_away": 3}]}
        with open(history_file, "a") as f:
            ph._write_news(f, "2026-01-01", "AAPL", news)
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["earnings_days_away"] is None

    def test_portfolio_news_not_a_list(self, history_file):
        """If portfolio_news[ticker] is not a list, news_count should be 0."""
        news = {"portfolio_news": {"AAPL": "invalid"}}
        with open(history_file, "a") as f:
            ph._write_news(f, "2026-01-01", "AAPL", news)
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["news_impact"] == "NEUTRAL"
        assert parsed["news_count"] == 0

    def test_empty_portfolio_news_list(self, history_file):
        news = {"portfolio_news": {"AAPL": []}}
        with open(history_file, "a") as f:
            ph._write_news(f, "2026-01-01", "AAPL", news)
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["news_impact"] == "NEUTRAL"
        assert parsed["news_count"] == 0

    def test_earnings_entry_not_dict(self, history_file):
        """Non-dict entries in earnings_calendar should be skipped."""
        news = {"earnings_calendar": ["not-a-dict", 42]}
        with open(history_file, "a") as f:
            ph._write_news(f, "2026-01-01", "AAPL", news)
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["earnings_days_away"] is None

    def test_article_without_impact_key(self, history_file):
        """Articles missing 'impact' default to 'NEUTRAL'."""
        news = {"portfolio_news": {"AAPL": [{"headline": "some news"}]}}
        with open(history_file, "a") as f:
            ph._write_news(f, "2026-01-01", "AAPL", news)
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["news_impact"] == "NEUTRAL"

    def test_negative_dominates_positive(self, history_file):
        """NEGATIVE checked before POSITIVE, so mixed = NEGATIVE."""
        news = {
            "portfolio_news": {
                "AAPL": [
                    {"impact": "SLIGHTLY_NEGATIVE"},  # contains "NEGATIVE"
                    {"impact": "STRONGLY_POSITIVE"},
                ]
            }
        }
        with open(history_file, "a") as f:
            ph._write_news(f, "2026-01-01", "AAPL", news)
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["news_impact"] == "NEGATIVE"


# ---------------------------------------------------------------------------
# _write_risk()
# ---------------------------------------------------------------------------


class TestWriteRisk:
    def test_full_data(self, history_file):
        risk = {
            "position_limits": {"AAPL": {"max_pct": 8.0}},
            "portfolio_risk": {"risk_score": 65, "portfolio_beta": 1.1},
        }
        with open(history_file, "a") as f:
            result = ph._write_risk(f, "2026-01-01", "AAPL", risk)
        assert result == 1
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["source"] == "risk"
        assert parsed["position_limit_pct"] == 8.0
        assert parsed["portfolio_risk_score"] == 65
        assert parsed["portfolio_beta"] == pytest.approx(1.1)

    def test_no_position_limits(self, history_file):
        risk = {"portfolio_risk": {"risk_score": 50}}
        with open(history_file, "a") as f:
            ph._write_risk(f, "2026-01-01", "AAPL", risk)
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["position_limit_pct"] is None

    def test_limits_not_dict(self, history_file):
        risk = {"position_limits": {"AAPL": "invalid"}}
        with open(history_file, "a") as f:
            ph._write_risk(f, "2026-01-01", "AAPL", risk)
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["position_limit_pct"] is None

    def test_empty_risk(self, history_file):
        with open(history_file, "a") as f:
            result = ph._write_risk(f, "2026-01-01", "AAPL", {})
        assert result == 1  # Always writes a record
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["position_limit_pct"] is None
        assert parsed["portfolio_risk_score"] is None
        assert parsed["portfolio_beta"] is None


# ---------------------------------------------------------------------------
# _write_synthesis()
# ---------------------------------------------------------------------------


class TestWriteSynthesis:
    def test_full_concordance_entry(self, history_file):
        concordance = [
            {
                "ticker": "AAPL",
                "conviction": 78,
                "action": "BUY",
                "base": 60,
                "bonuses": ["+5 momentum"],
                "penalties": ["-3 valuation"],
                "bull_weight": 0.6,
                "bear_weight": 0.4,
                "bull_pct": 60,
                "signal_velocity": 0.02,
                "earnings_surprise": 5.3,
                "days_held": 45,
                "holding_review_flag": False,
                "position_size_pct": 4.5,
                "entry_timing": "GOOD",
                "is_opportunity": True,
                "sector": "Technology",
                "directional_confidence": 0.85,
                "hold_tier": "A",
                "max_pct": 8.0,
                "fund_score": 82,
                "tech_signal": "BUY",
                "macro_fit": "high",
                "census": "bullish",
                "news_impact": "POSITIVE",
                "rsi": 55,
                "exret": 0.03,
                "buy_pct": 85.0,
                "signal": "BUY",
                "price": 180.0,
                "conviction_waterfall": {"base": 60, "momentum": 5},
            }
        ]
        with open(history_file, "a") as f:
            result = ph._write_synthesis(f, "2026-01-01", "AAPL", concordance)
        assert result == 1
        parsed = json.loads(history_file.read_text().strip())
        assert parsed["source"] == "synthesis"
        assert parsed["conviction"] == 78
        assert parsed["action"] == "BUY"
        assert parsed["waterfall"] == {"base": 60, "momentum": 5}

    def test_ticker_not_in_concordance(self, history_file):
        concordance = [{"ticker": "MSFT", "conviction": 50}]
        with open(history_file, "a") as f:
            result = ph._write_synthesis(f, "2026-01-01", "AAPL", concordance)
        assert result == 0

    def test_empty_concordance(self, history_file):
        with open(history_file, "a") as f:
            result = ph._write_synthesis(f, "2026-01-01", "AAPL", [])
        assert result == 0

    def test_no_waterfall(self, history_file):
        concordance = [{"ticker": "AAPL", "conviction": 60}]
        with open(history_file, "a") as f:
            ph._write_synthesis(f, "2026-01-01", "AAPL", concordance)
        parsed = json.loads(history_file.read_text().strip())
        assert "waterfall" not in parsed

    def test_empty_waterfall_excluded(self, history_file):
        concordance = [{"ticker": "AAPL", "conviction": 60, "conviction_waterfall": {}}]
        with open(history_file, "a") as f:
            ph._write_synthesis(f, "2026-01-01", "AAPL", concordance)
        parsed = json.loads(history_file.read_text().strip())
        assert "waterfall" not in parsed

    def test_none_values_excluded(self, history_file):
        concordance = [{"ticker": "AAPL", "conviction": 70, "action": None}]
        with open(history_file, "a") as f:
            ph._write_synthesis(f, "2026-01-01", "AAPL", concordance)
        parsed = json.loads(history_file.read_text().strip())
        assert "conviction" in parsed
        assert "action" not in parsed


# ---------------------------------------------------------------------------
# save_parameter_history() — end-to-end
# ---------------------------------------------------------------------------


class TestSaveParameterHistory:
    def _minimal_args(self):
        """Return minimal valid arguments for save_parameter_history."""
        return {
            "date": "2026-06-01",
            "concordance": [{"ticker": "AAPL", "conviction": 75, "action": "HOLD"}],
            "portfolio_signals": {
                "AAPL": {"price": 180.0, "signal": "BUY"},
            },
            "fund_report": {"stocks": {"AAPL": {"fundamental_score": 80}}},
            "tech_report": {"stocks": {"AAPL": {"rsi": 55}}},
            "macro_report": {
                "regime": "expansion",
                "macro_score": 70,
                "rotation_phase": "mid_cycle",
                "portfolio_implications": {"AAPL": {"macro_fit": "high"}},
            },
            "census_report": {"sentiment": {"fg_top100": 60}},
            "news_report": {},
            "risk_report": {},
        }

    def test_writes_all_sources(self, history_file):
        lines = ph.save_parameter_history(**self._minimal_args())
        # 8 sources per ticker: signals, fundamental, technical, macro, census, news, risk, synthesis
        assert lines == 8
        records = [json.loads(l) for l in history_file.read_text().strip().split("\n")]
        sources = {r["source"] for r in records}
        assert sources == {
            "signals",
            "fundamental",
            "technical",
            "macro",
            "census",
            "news",
            "risk",
            "synthesis",
        }

    def test_multiple_tickers(self, history_file):
        args = self._minimal_args()
        args["concordance"].append({"ticker": "MSFT", "conviction": 60})
        args["portfolio_signals"]["MSFT"] = {"price": 400.0}
        lines = ph.save_parameter_history(**args)
        records = [json.loads(l) for l in history_file.read_text().strip().split("\n")]
        tickers = {r["ticker"] for r in records}
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_ticker_from_concordance_only(self, history_file):
        """A ticker in concordance but not in portfolio_signals still gets processed."""
        args = self._minimal_args()
        args["concordance"] = [{"ticker": "GOOG", "conviction": 55}]
        args["portfolio_signals"] = {}
        lines = ph.save_parameter_history(**args)
        records = [json.loads(l) for l in history_file.read_text().strip().split("\n")]
        # signals will return 0 (no GOOG in portfolio_signals), but census/news/risk always write
        assert any(r["ticker"] == "GOOG" for r in records)

    def test_ticker_from_signals_only(self, history_file):
        """A ticker in portfolio_signals but not in concordance still gets processed."""
        args = self._minimal_args()
        args["concordance"] = []
        args["portfolio_signals"] = {"NVDA": {"price": 900.0, "signal": "BUY"}}
        lines = ph.save_parameter_history(**args)
        records = [json.loads(l) for l in history_file.read_text().strip().split("\n")]
        assert any(r["ticker"] == "NVDA" for r in records)

    def test_empty_ticker_discarded(self, history_file):
        """Concordance entries with empty ticker string are discarded."""
        args = self._minimal_args()
        args["concordance"] = [{"ticker": "", "conviction": 0}]
        args["portfolio_signals"] = {}
        lines = ph.save_parameter_history(**args)
        assert lines == 0

    def test_creates_parent_directory(self, tmp_path, monkeypatch):
        nested = tmp_path / "deep" / "nested" / "history.jsonl"
        monkeypatch.setattr(ph, "HISTORY_PATH", nested)
        args = self._minimal_args()
        ph.save_parameter_history(**args)
        assert nested.exists()

    def test_appends_to_existing(self, history_file):
        # Write a first run
        args = self._minimal_args()
        first_lines = ph.save_parameter_history(**args)
        # Write a second run
        args["date"] = "2026-06-02"
        second_lines = ph.save_parameter_history(**args)
        total = len(history_file.read_text().strip().split("\n"))
        assert total == first_lines + second_lines

    def test_returns_zero_for_no_tickers(self, history_file):
        args = self._minimal_args()
        args["concordance"] = []
        args["portfolio_signals"] = {}
        lines = ph.save_parameter_history(**args)
        assert lines == 0

    def test_tickers_sorted(self, history_file):
        args = self._minimal_args()
        args["concordance"] = [
            {"ticker": "TSLA", "conviction": 50},
            {"ticker": "AAPL", "conviction": 60},
            {"ticker": "MSFT", "conviction": 70},
        ]
        args["portfolio_signals"] = {}
        ph.save_parameter_history(**args)
        records = [json.loads(l) for l in history_file.read_text().strip().split("\n")]
        # Extract ticker order for first occurrence of each
        seen = []
        for r in records:
            if r["ticker"] not in seen:
                seen.append(r["ticker"])
        assert seen == ["AAPL", "MSFT", "TSLA"]

    def test_missing_data_sources_produce_partial_records(self, history_file):
        """Tickers with no data in some sources still get census/news/risk records."""
        args = self._minimal_args()
        args["fund_report"] = {}  # No fundamental data
        args["tech_report"] = {}  # No technical data
        args["macro_report"] = {}  # No macro data
        lines = ph.save_parameter_history(**args)
        records = [json.loads(l) for l in history_file.read_text().strip().split("\n")]
        sources = {r["source"] for r in records}
        # fundamental, technical, macro will return 0 lines
        # signals, census, news, risk, synthesis will write
        assert "fundamental" not in sources
        assert "technical" not in sources
        assert "macro" not in sources
        assert "signals" in sources
        assert "census" in sources
        assert "news" in sources
        assert "risk" in sources


# ---------------------------------------------------------------------------
# get_parameter_history()
# ---------------------------------------------------------------------------


class TestGetParameterHistory:
    def _seed_records(self, history_file, records):
        """Write pre-built records to the JSONL file."""
        with open(history_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_empty_file(self, history_file):
        assert ph.get_parameter_history() == []

    def test_file_does_not_exist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ph, "HISTORY_PATH", tmp_path / "nonexistent.jsonl")
        assert ph.get_parameter_history() == []

    def test_reads_all_records(self, history_file):
        records = [
            {"date": "2026-06-01", "ticker": "AAPL", "source": "signals"},
            {"date": "2026-06-01", "ticker": "AAPL", "source": "fundamental"},
        ]
        self._seed_records(history_file, records)
        result = ph.get_parameter_history()
        assert len(result) == 2

    def test_filter_by_ticker(self, history_file):
        records = [
            {"date": "2026-06-01", "ticker": "AAPL", "source": "signals"},
            {"date": "2026-06-01", "ticker": "MSFT", "source": "signals"},
        ]
        self._seed_records(history_file, records)
        result = ph.get_parameter_history(ticker="AAPL")
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"

    def test_filter_by_source(self, history_file):
        records = [
            {"date": "2026-06-01", "ticker": "AAPL", "source": "signals"},
            {"date": "2026-06-01", "ticker": "AAPL", "source": "fundamental"},
        ]
        self._seed_records(history_file, records)
        result = ph.get_parameter_history(source="fundamental")
        assert len(result) == 1
        assert result[0]["source"] == "fundamental"

    def test_filter_by_ticker_and_source(self, history_file):
        records = [
            {"date": "2026-06-01", "ticker": "AAPL", "source": "signals"},
            {"date": "2026-06-01", "ticker": "AAPL", "source": "fundamental"},
            {"date": "2026-06-01", "ticker": "MSFT", "source": "signals"},
        ]
        self._seed_records(history_file, records)
        result = ph.get_parameter_history(ticker="AAPL", source="signals")
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"
        assert result[0]["source"] == "signals"

    def test_date_cutoff(self, history_file):
        records = [
            {"date": "2020-01-01", "ticker": "AAPL", "source": "signals"},  # very old
            {"date": "2026-06-01", "ticker": "AAPL", "source": "signals"},  # recent
        ]
        self._seed_records(history_file, records)
        result = ph.get_parameter_history(days=90)
        assert len(result) == 1
        assert result[0]["date"] == "2026-06-01"

    def test_large_days_returns_all(self, history_file):
        records = [
            {"date": "2023-01-01", "ticker": "AAPL", "source": "signals"},
            {"date": "2026-06-01", "ticker": "AAPL", "source": "signals"},
        ]
        self._seed_records(history_file, records)
        result = ph.get_parameter_history(days=9999)
        assert len(result) == 2

    def test_skips_blank_lines(self, history_file):
        content = (
            '{"date": "2026-06-01", "ticker": "AAPL", "source": "signals"}\n'
            "\n"
            '{"date": "2026-06-01", "ticker": "MSFT", "source": "signals"}\n'
            "   \n"
        )
        history_file.write_text(content)
        result = ph.get_parameter_history()
        assert len(result) == 2

    def test_skips_invalid_json(self, history_file):
        content = (
            '{"date": "2026-06-01", "ticker": "AAPL", "source": "signals"}\n'
            "this is not json\n"
            '{"date": "2026-06-01", "ticker": "MSFT", "source": "signals"}\n'
        )
        history_file.write_text(content)
        result = ph.get_parameter_history()
        assert len(result) == 2

    def test_no_filters_returns_all_recent(self, history_file):
        records = [
            {"date": "2026-06-01", "ticker": "AAPL", "source": "signals", "price": 180},
            {"date": "2026-06-01", "ticker": "MSFT", "source": "risk", "score": 50},
        ]
        self._seed_records(history_file, records)
        result = ph.get_parameter_history()
        assert len(result) == 2

    def test_missing_date_field_excluded(self, history_file):
        """Records without a date field default to '' which is < any cutoff date."""
        records = [
            {"ticker": "AAPL", "source": "signals"},  # no date
            {"date": "2026-06-01", "ticker": "MSFT", "source": "signals"},
        ]
        self._seed_records(history_file, records)
        result = ph.get_parameter_history(days=90)
        # "" < cutoff date, so the dateless record is excluded
        assert len(result) == 1
        assert result[0]["ticker"] == "MSFT"

    def test_ticker_none_returns_all_tickers(self, history_file):
        records = [
            {"date": "2026-06-01", "ticker": "AAPL", "source": "signals"},
            {"date": "2026-06-01", "ticker": "MSFT", "source": "signals"},
        ]
        self._seed_records(history_file, records)
        result = ph.get_parameter_history(ticker=None)
        assert len(result) == 2

    def test_source_none_returns_all_sources(self, history_file):
        records = [
            {"date": "2026-06-01", "ticker": "AAPL", "source": "signals"},
            {"date": "2026-06-01", "ticker": "AAPL", "source": "risk"},
        ]
        self._seed_records(history_file, records)
        result = ph.get_parameter_history(source=None)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Integration: round-trip save + read
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_save_then_read(self, history_file):
        """Save data and read it back with filters."""
        concordance = [
            {"ticker": "AAPL", "conviction": 75, "action": "BUY"},
            {"ticker": "MSFT", "conviction": 60, "action": "HOLD"},
        ]
        portfolio_signals = {
            "AAPL": {"price": 180.0, "signal": "BUY"},
            "MSFT": {"price": 400.0, "signal": "HOLD"},
        }
        ph.save_parameter_history(
            date="2026-06-15",
            concordance=concordance,
            portfolio_signals=portfolio_signals,
            fund_report={"stocks": {"AAPL": {"fundamental_score": 80}}},
            tech_report={},
            macro_report={},
            census_report={},
            news_report={},
            risk_report={},
        )

        # Read AAPL signals only
        results = ph.get_parameter_history(ticker="AAPL", source="signals")
        assert len(results) == 1
        assert results[0]["price"] == 180.0

        # Read all MSFT records
        results = ph.get_parameter_history(ticker="MSFT")
        msft_sources = {r["source"] for r in results}
        assert "census" in msft_sources  # census always writes
        assert "news" in msft_sources  # news always writes

        # Read synthesis only
        results = ph.get_parameter_history(source="synthesis")
        assert len(results) == 2
        tickers = {r["ticker"] for r in results}
        assert tickers == {"AAPL", "MSFT"}
