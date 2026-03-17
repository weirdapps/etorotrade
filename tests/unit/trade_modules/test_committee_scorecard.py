"""
Tests for Committee Scorecard — CIO Review v4 Findings F5 and F10.

F5: Persistent Opportunity Tracking (track_opportunities, save_opportunity_history)
F10: Automated Performance Check (check_previous_recommendations, get_track_record_summary)

All tests use temp files and mocks — no network or filesystem side-effects.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trade_modules.committee_scorecard import (
    BONUS_PER_APPEARANCE,
    MAX_CONSECUTIVE_BONUS,
    OPPORTUNITY_HISTORY_PATH,
    _empty_performance_check,
    _load_all_actions,
    _load_opportunity_history,
    check_previous_recommendations,
    get_track_record_summary,
    save_opportunity_history,
    track_opportunities,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def tmp_committee_dir(tmp_path):
    """Create a temp committee directory structure."""
    committee_dir = tmp_path / "committee"
    committee_dir.mkdir()
    return committee_dir


@pytest.fixture
def history_path(tmp_committee_dir):
    """Path for opportunity history JSON."""
    return tmp_committee_dir / "opportunity_history.json"


@pytest.fixture
def action_log_path(tmp_committee_dir):
    """Path for action log JSONL."""
    return tmp_committee_dir / "action_log.jsonl"


@pytest.fixture
def scorecard_path(tmp_committee_dir):
    """Path for committee scorecard JSON."""
    return tmp_committee_dir / "committee_scorecard.json"


def _write_action_log(path: Path, actions: list) -> None:
    """Write actions to JSONL file."""
    with open(path, "w") as f:
        for action in actions:
            f.write(json.dumps(action) + "\n")


# ============================================================
# F5: track_opportunities
# ============================================================


class TestTrackOpportunities:
    """Tests for F5 — Persistent Opportunity Tracking."""

    def test_first_run_no_history(self, history_path):
        """First run with no history: all tickers get streak=1, bonus=0."""
        result = track_opportunities(
            ["AAPL", "MSFT", "GOOGL"],
            "2026-03-16",
            history_path=history_path,
        )

        assert len(result) == 3
        for ticker in ("AAPL", "MSFT", "GOOGL"):
            assert result[ticker]["consecutive_appearances"] == 1
            assert result[ticker]["conviction_bonus"] == 0
            assert result[ticker]["first_seen"] == "2026-03-16"

    def test_second_consecutive_run(self, history_path):
        """Second consecutive run: returning tickers get streak=2, bonus=3."""
        # First run
        track_opportunities(["AAPL", "MSFT"], "2026-03-09", history_path=history_path)

        # Second run — AAPL returns, NVDA is new
        result = track_opportunities(
            ["AAPL", "NVDA"], "2026-03-16", history_path=history_path
        )

        assert result["AAPL"]["consecutive_appearances"] == 2
        assert result["AAPL"]["conviction_bonus"] == BONUS_PER_APPEARANCE  # +3
        assert result["AAPL"]["first_seen"] == "2026-03-09"

        assert result["NVDA"]["consecutive_appearances"] == 1
        assert result["NVDA"]["conviction_bonus"] == 0

    def test_three_consecutive_runs(self, history_path):
        """Three runs: streak=3, bonus=6."""
        track_opportunities(["AAPL"], "2026-03-02", history_path=history_path)
        track_opportunities(["AAPL"], "2026-03-09", history_path=history_path)
        result = track_opportunities(
            ["AAPL"], "2026-03-16", history_path=history_path
        )

        assert result["AAPL"]["consecutive_appearances"] == 3
        assert result["AAPL"]["conviction_bonus"] == 6
        assert result["AAPL"]["first_seen"] == "2026-03-02"

    def test_bonus_capped_at_max(self, history_path):
        """Conviction bonus is capped at MAX_CONSECUTIVE_BONUS (9)."""
        # Run 5 consecutive times to exceed the cap
        for i in range(5):
            date = f"2026-03-{(i + 1):02d}"
            result = track_opportunities(
                ["AAPL"], date, history_path=history_path
            )

        # streak=5, raw bonus = (5-1)*3 = 12, capped at 9
        assert result["AAPL"]["consecutive_appearances"] == 5
        assert result["AAPL"]["conviction_bonus"] == MAX_CONSECUTIVE_BONUS

    def test_gap_resets_streak(self, history_path):
        """Ticker missing from a run resets its streak."""
        track_opportunities(["AAPL", "MSFT"], "2026-03-02", history_path=history_path)

        # AAPL drops out
        track_opportunities(["MSFT"], "2026-03-09", history_path=history_path)

        # AAPL returns — streak resets to 1
        result = track_opportunities(
            ["AAPL", "MSFT"], "2026-03-16", history_path=history_path
        )

        assert result["AAPL"]["consecutive_appearances"] == 1
        assert result["AAPL"]["conviction_bonus"] == 0
        # AAPL retains its first_seen from the original appearance
        assert result["AAPL"]["first_seen"] == "2026-03-02"

        assert result["MSFT"]["consecutive_appearances"] == 3
        assert result["MSFT"]["conviction_bonus"] == 6

    def test_empty_ticker_list(self, history_path):
        """Empty ticker list returns empty dict."""
        result = track_opportunities([], "2026-03-16", history_path=history_path)
        assert result == {}

    def test_history_persisted(self, history_path):
        """History is saved to disk after tracking."""
        track_opportunities(["AAPL"], "2026-03-16", history_path=history_path)

        assert history_path.exists()
        with open(history_path) as f:
            data = json.load(f)

        assert data["last_committee_date"] == "2026-03-16"
        assert "AAPL" in data["active_tickers"]
        assert "AAPL" in data["ticker_records"]

    def test_single_appearance_no_bonus(self, history_path):
        """Single appearance gives zero bonus (bonus starts at appearance 2)."""
        result = track_opportunities(
            ["TSLA"], "2026-03-16", history_path=history_path
        )
        assert result["TSLA"]["conviction_bonus"] == 0


# ============================================================
# F5: save_opportunity_history / _load_opportunity_history
# ============================================================


class TestOpportunityHistoryIO:
    """Tests for opportunity history save/load."""

    def test_save_creates_file(self, history_path):
        """save_opportunity_history creates the file."""
        history = {
            "last_committee_date": "2026-03-16",
            "active_tickers": ["AAPL"],
            "ticker_records": {
                "AAPL": {
                    "consecutive_appearances": 1,
                    "first_seen": "2026-03-16",
                    "last_seen": "2026-03-16",
                }
            },
        }
        save_opportunity_history(history, history_path)

        assert history_path.exists()
        with open(history_path) as f:
            loaded = json.load(f)
        assert loaded == history

    def test_load_missing_file(self, tmp_path):
        """Loading a missing file returns empty structure."""
        path = tmp_path / "nonexistent.json"
        result = _load_opportunity_history(path)

        assert result["last_committee_date"] is None
        assert result["active_tickers"] == []
        assert result["ticker_records"] == {}

    def test_load_corrupt_json(self, history_path):
        """Loading corrupt JSON returns empty structure."""
        with open(history_path, "w") as f:
            f.write("not valid json {{{")

        result = _load_opportunity_history(history_path)
        assert result["last_committee_date"] is None

    def test_save_creates_parent_dirs(self, tmp_path):
        """save_opportunity_history creates parent directories."""
        deep_path = tmp_path / "a" / "b" / "c" / "history.json"
        save_opportunity_history(
            {"last_committee_date": None, "active_tickers": [], "ticker_records": {}},
            deep_path,
        )
        assert deep_path.exists()

    def test_default_path_constant(self):
        """OPPORTUNITY_HISTORY_PATH is under ~/.weirdapps-trading/committee/."""
        assert "committee" in str(OPPORTUNITY_HISTORY_PATH)
        assert "opportunity_history.json" in str(OPPORTUNITY_HISTORY_PATH)


# ============================================================
# F10: check_previous_recommendations
# ============================================================


class TestCheckPreviousRecommendations:
    """Tests for F10 — Automated Performance Check."""

    def test_no_log_file(self, tmp_path):
        """Returns empty result when log file does not exist."""
        result = check_previous_recommendations(
            log_path=tmp_path / "nonexistent.jsonl"
        )
        assert result == _empty_performance_check()

    def test_empty_log_file(self, action_log_path):
        """Returns empty result when log file is empty."""
        action_log_path.touch()
        result = check_previous_recommendations(log_path=action_log_path)
        assert result["total_recommendations"] == 0

    def test_no_buy_actions(self, action_log_path):
        """Returns empty when only SELL actions exist."""
        _write_action_log(action_log_path, [
            {
                "committee_date": "2026-03-10",
                "ticker": "BADCO",
                "action": "SELL",
                "price_at_recommendation": 50.0,
            }
        ])
        result = check_previous_recommendations(log_path=action_log_path)
        assert result["total_recommendations"] == 0

    def test_yfinance_import_failure(self, action_log_path):
        """Returns empty when yfinance is not available."""
        _write_action_log(action_log_path, [
            {
                "committee_date": "2026-03-10",
                "ticker": "AAPL",
                "action": "BUY",
                "price_at_recommendation": 150.0,
            }
        ])

        with patch.dict("sys.modules", {"yfinance": None}):
            with patch(
                "trade_modules.committee_scorecard.check_previous_recommendations"
            ) as mock_check:
                # When yfinance import fails in the function, it returns empty
                mock_check.return_value = _empty_performance_check()
                result = mock_check(log_path=action_log_path)

        assert result["total_recommendations"] == 0

    def test_successful_check_with_mock_yfinance(self, action_log_path):
        """Full happy-path with mocked yfinance."""
        _write_action_log(action_log_path, [
            {
                "committee_date": "2026-03-10",
                "ticker": "AAPL",
                "action": "BUY",
                "conviction": 75,
                "price_at_recommendation": 150.0,
            },
            {
                "committee_date": "2026-03-10",
                "ticker": "MSFT",
                "action": "ADD",
                "conviction": 65,
                "price_at_recommendation": 400.0,
            },
        ])

        mock_yf = MagicMock()

        def make_mock_ticker(ticker):
            mock_stock = MagicMock()
            prices = {"AAPL": 165.0, "MSFT": 380.0}
            mock_stock.fast_info = {"lastPrice": prices.get(ticker, 0)}
            return mock_stock

        mock_yf.Ticker.side_effect = make_mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = check_previous_recommendations(log_path=action_log_path)

        assert result["total_recommendations"] == 2
        assert result["active_buys"] == 2
        assert result["committee_date"] == "2026-03-10"

        # AAPL: (165-150)/150*100 = +10.0%
        # MSFT: (380-400)/400*100 = -5.0%
        assert result["best_performer"]["ticker"] == "AAPL"
        assert result["best_performer"]["return_pct"] == 10.0
        assert result["worst_performer"]["ticker"] == "MSFT"
        assert result["worst_performer"]["return_pct"] == -5.0

        # Average: (10.0 + -5.0) / 2 = 2.5
        assert result["avg_return_to_date"] == 2.5

        # Placeholder
        assert result["triggered_kill_theses"] == []

    def test_uses_most_recent_date(self, action_log_path):
        """Only checks recommendations from the most recent committee date."""
        _write_action_log(action_log_path, [
            {
                "committee_date": "2026-03-03",
                "ticker": "OLD",
                "action": "BUY",
                "price_at_recommendation": 100.0,
            },
            {
                "committee_date": "2026-03-10",
                "ticker": "AAPL",
                "action": "BUY",
                "price_at_recommendation": 150.0,
            },
        ])

        mock_yf = MagicMock()
        mock_stock = MagicMock()
        mock_stock.fast_info = {"lastPrice": 160.0}
        mock_yf.Ticker.return_value = mock_stock

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = check_previous_recommendations(log_path=action_log_path)

        assert result["committee_date"] == "2026-03-10"
        assert result["total_recommendations"] == 1
        # Only AAPL from 2026-03-10, not OLD from 2026-03-03
        assert result["recommendations"][0]["ticker"] == "AAPL"

    def test_skips_invalid_prices(self, action_log_path):
        """Skips actions with missing or invalid entry prices."""
        _write_action_log(action_log_path, [
            {
                "committee_date": "2026-03-10",
                "ticker": "NOPR",
                "action": "BUY",
                # Missing price_at_recommendation
            },
            {
                "committee_date": "2026-03-10",
                "ticker": "BADPR",
                "action": "BUY",
                "price_at_recommendation": "not_a_number",
            },
            {
                "committee_date": "2026-03-10",
                "ticker": "AAPL",
                "action": "BUY",
                "price_at_recommendation": 150.0,
            },
        ])

        mock_yf = MagicMock()
        mock_stock = MagicMock()
        mock_stock.fast_info = {"lastPrice": 160.0}
        mock_yf.Ticker.return_value = mock_stock

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = check_previous_recommendations(log_path=action_log_path)

        # Only AAPL should be in results (3 BUY total, 1 valid)
        assert result["total_recommendations"] == 3
        assert result["active_buys"] == 1
        assert result["recommendations"][0]["ticker"] == "AAPL"

    def test_fallback_to_history_when_fast_info_fails(self, action_log_path):
        """Falls back to stock.history when fast_info gives zero price."""
        import pandas as pd

        _write_action_log(action_log_path, [
            {
                "committee_date": "2026-03-10",
                "ticker": "AAPL",
                "action": "BUY",
                "price_at_recommendation": 150.0,
            },
        ])

        mock_yf = MagicMock()
        mock_stock = MagicMock()
        mock_stock.fast_info = {"lastPrice": 0}
        mock_stock.history.return_value = pd.DataFrame({"Close": [165.0]})
        mock_yf.Ticker.return_value = mock_stock

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = check_previous_recommendations(log_path=action_log_path)

        assert result["active_buys"] == 1
        assert result["recommendations"][0]["return_pct"] == 10.0

    def test_all_prices_fail_returns_empty(self, action_log_path):
        """Returns empty when all price fetches fail."""
        _write_action_log(action_log_path, [
            {
                "committee_date": "2026-03-10",
                "ticker": "FAIL",
                "action": "BUY",
                "price_at_recommendation": 100.0,
            },
        ])

        mock_yf = MagicMock()
        mock_stock = MagicMock()
        mock_stock.fast_info = {"lastPrice": 0}
        import pandas as pd
        mock_stock.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_stock

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = check_previous_recommendations(log_path=action_log_path)

        assert result == _empty_performance_check()


# ============================================================
# F10: get_track_record_summary
# ============================================================


class TestGetTrackRecordSummary:
    """Tests for the one-line track record summary."""

    def test_no_scorecard_file(self):
        """Returns 'insufficient data' when no scorecard exists."""
        with patch("trade_modules.committee_scorecard.SCORECARD_OUTPUT_PATH") as mock_path:
            mock_path.exists.return_value = False
            result = get_track_record_summary()

        assert "insufficient data" in result
        assert "no scorecard" in result

    def test_corrupt_scorecard(self, scorecard_path):
        """Returns 'insufficient data' for corrupt scorecard."""
        with open(scorecard_path, "w") as f:
            f.write("not json")

        with patch(
            "trade_modules.committee_scorecard.SCORECARD_OUTPUT_PATH",
            scorecard_path,
        ):
            result = get_track_record_summary()

        assert "insufficient data" in result
        assert "unreadable" in result

    def test_zero_total(self, scorecard_path):
        """Returns 'n=0' when scorecard has zero recommendations."""
        with open(scorecard_path, "w") as f:
            json.dump({"buy_recommendations": {"total": 0}}, f)

        with patch(
            "trade_modules.committee_scorecard.SCORECARD_OUTPUT_PATH",
            scorecard_path,
        ):
            result = get_track_record_summary()

        assert "n=0" in result

    def test_with_30d_hit_rate(self, scorecard_path):
        """Returns T+30 hit rate when available."""
        with open(scorecard_path, "w") as f:
            json.dump(
                {
                    "buy_recommendations": {
                        "total": 15,
                        "hit_rate_30d": 73.3,
                        "hit_rate_7d": 60.0,
                    }
                },
                f,
            )

        with patch(
            "trade_modules.committee_scorecard.SCORECARD_OUTPUT_PATH",
            scorecard_path,
        ):
            result = get_track_record_summary()

        assert "73%" in result
        assert "T+30" in result
        assert "n=15" in result

    def test_fallback_to_7d_hit_rate(self, scorecard_path):
        """Falls back to T+7 when T+30 is not available."""
        with open(scorecard_path, "w") as f:
            json.dump(
                {
                    "buy_recommendations": {
                        "total": 8,
                        "hit_rate_7d": 62.5,
                    }
                },
                f,
            )

        with patch(
            "trade_modules.committee_scorecard.SCORECARD_OUTPUT_PATH",
            scorecard_path,
        ):
            result = get_track_record_summary()

        assert "62%" in result
        assert "T+7" in result
        assert "n=8" in result

    def test_no_hit_rate_data(self, scorecard_path):
        """Returns 'awaiting' when total > 0 but no hit rate yet."""
        with open(scorecard_path, "w") as f:
            json.dump({"buy_recommendations": {"total": 5}}, f)

        with patch(
            "trade_modules.committee_scorecard.SCORECARD_OUTPUT_PATH",
            scorecard_path,
        ):
            result = get_track_record_summary()

        assert "5 BUY" in result
        assert "awaiting" in result


# ============================================================
# _load_all_actions
# ============================================================


class TestLoadAllActions:
    """Tests for _load_all_actions helper."""

    def test_load_valid_actions(self, action_log_path):
        """Loads valid JSONL entries."""
        _write_action_log(action_log_path, [
            {"committee_date": "2026-03-10", "ticker": "AAPL", "action": "BUY"},
            {"committee_date": "2026-03-10", "ticker": "MSFT", "action": "SELL"},
        ])
        result = _load_all_actions(action_log_path)
        assert len(result) == 2

    def test_skips_invalid_lines(self, action_log_path):
        """Skips malformed JSON lines."""
        with open(action_log_path, "w") as f:
            f.write('{"ticker": "AAPL"}\n')
            f.write("bad json\n")
            f.write('{"ticker": "MSFT"}\n')
            f.write("\n")  # empty line

        result = _load_all_actions(action_log_path)
        assert len(result) == 2

    def test_nonexistent_file(self, tmp_path):
        """Returns empty list for missing file."""
        result = _load_all_actions(tmp_path / "missing.jsonl")
        assert result == []


# ============================================================
# _empty_performance_check
# ============================================================


class TestEmptyPerformanceCheck:
    """Tests for the empty result structure."""

    def test_structure(self):
        """Verify empty structure has all required keys."""
        result = _empty_performance_check()
        assert result["committee_date"] is None
        assert result["total_recommendations"] == 0
        assert result["active_buys"] == 0
        assert result["avg_return_to_date"] == 0.0
        assert result["best_performer"] is None
        assert result["worst_performer"] is None
        assert result["triggered_kill_theses"] == []
        assert result["recommendations"] == []


# ============================================================
# Constants
# ============================================================


class TestConstants:
    """Tests for F5 constants."""

    def test_bonus_per_appearance(self):
        """BONUS_PER_APPEARANCE is 3."""
        assert BONUS_PER_APPEARANCE == 3

    def test_max_consecutive_bonus(self):
        """MAX_CONSECUTIVE_BONUS is 9."""
        assert MAX_CONSECUTIVE_BONUS == 9

    def test_max_bonus_is_multiple_of_per_appearance(self):
        """Max bonus should be a clean multiple of per-appearance bonus."""
        assert MAX_CONSECUTIVE_BONUS % BONUS_PER_APPEARANCE == 0


# ============================================================
# CIO Legacy Review: D2 — Custom Kill Theses
# ============================================================


class TestCustomKillTheses:
    """Tests for CIO Legacy D2: machine-checkable kill thesis conditions."""

    def test_log_kill_thesis_with_conditions(self, tmp_path):
        """log_kill_theses should store conditions field."""
        from trade_modules.committee_scorecard import log_kill_theses
        log_file = tmp_path / "kill_theses.json"
        theses = [{
            "ticker": "AAPL",
            "kill_thesis": "Revenue growth stalls",
            "conditions": [
                {"metric": "EG", "operator": "lt", "threshold": 5.0},
                {"metric": "UP%", "operator": "lt", "threshold": 0},
            ],
        }]
        log_kill_theses("2026-03-17", theses, log_path=log_file)

        data = json.loads(log_file.read_text())
        assert len(data) == 1
        assert len(data[0]["conditions"]) == 2
        assert data[0]["conditions"][0]["metric"] == "EG"

    def test_check_custom_condition_lt_triggers(self, tmp_path):
        """Custom condition with operator 'lt' should trigger."""
        from trade_modules.committee_scorecard import (
            check_kill_theses,
            log_kill_theses,
        )
        # Setup: log a thesis with condition EG < 5
        log_file = tmp_path / "kill_theses.json"
        theses = [{
            "ticker": "AAPL",
            "kill_thesis": "Growth collapse",
            "conditions": [
                {"metric": "EG", "operator": "lt", "threshold": 5.0},
            ],
        }]
        log_kill_theses("2026-03-01", theses, log_path=log_file)

        # Create mock portfolio CSV with EG=3 (below threshold)
        portfolio_csv = tmp_path / "portfolio.csv"
        portfolio_csv.write_text(
            "TKR,NAME,CAP,PRC,TGT,UP%,#T,%%B,#A,AM,A,EXR,B,52W,2H,PET,PEF,"
            "P/S,PEG,DV,SI,EG,PP,ROE,DE,FCF,ERN,SZ,BS\n"
            "AAPL,Apple,MEGA,150,180,20,15,60,20,2.0,A,12,1.1,75,50,25,22,"
            "8,2.5,0.6,1.5,3.0,80,120,1.5,5.0,2026-04,3.0,B\n"
        )

        result = check_kill_theses(
            portfolio_signals_path=portfolio_csv,
            log_path=log_file,
        )
        assert len(result["triggered_theses"]) == 1
        triggers = result["triggered_theses"][0]["triggers"]
        assert any("custom:EG lt" in t for t in triggers)

    def test_check_custom_condition_gt_triggers(self, tmp_path):
        """Custom condition with operator 'gt' should trigger."""
        from trade_modules.committee_scorecard import (
            check_kill_theses,
            log_kill_theses,
        )
        log_file = tmp_path / "kill_theses.json"
        theses = [{
            "ticker": "AAPL",
            "kill_thesis": "Beta too high",
            "conditions": [
                {"metric": "SI", "operator": "gt", "threshold": 10.0},
            ],
        }]
        log_kill_theses("2026-03-01", theses, log_path=log_file)

        portfolio_csv = tmp_path / "portfolio.csv"
        portfolio_csv.write_text(
            "TKR,NAME,CAP,PRC,TGT,UP%,#T,%%B,#A,AM,A,EXR,B,52W,2H,PET,PEF,"
            "P/S,PEG,DV,SI,EG,PP,ROE,DE,FCF,ERN,SZ,BS\n"
            "AAPL,Apple,MEGA,150,180,20,15,60,20,2.0,A,12,1.1,75,50,25,22,"
            "8,2.5,0.6,15.0,8.0,80,120,1.5,5.0,2026-04,3.0,B\n"
        )

        result = check_kill_theses(
            portfolio_signals_path=portfolio_csv,
            log_path=log_file,
        )
        assert len(result["triggered_theses"]) == 1
        triggers = result["triggered_theses"][0]["triggers"]
        assert any("custom:SI gt" in t for t in triggers)

    def test_custom_condition_not_triggered(self, tmp_path):
        """Custom condition should NOT trigger if metric is within threshold."""
        from trade_modules.committee_scorecard import (
            check_kill_theses,
            log_kill_theses,
        )
        log_file = tmp_path / "kill_theses.json"
        theses = [{
            "ticker": "AAPL",
            "kill_thesis": "Revenue collapse",
            "conditions": [
                {"metric": "EG", "operator": "lt", "threshold": 5.0},
            ],
        }]
        log_kill_theses("2026-03-01", theses, log_path=log_file)

        portfolio_csv = tmp_path / "portfolio.csv"
        portfolio_csv.write_text(
            "TKR,NAME,CAP,PRC,TGT,UP%,#T,%%B,#A,AM,A,EXR,B,52W,2H,PET,PEF,"
            "P/S,PEG,DV,SI,EG,PP,ROE,DE,FCF,ERN,SZ,BS\n"
            "AAPL,Apple,MEGA,150,180,20,15,60,20,2.0,A,12,1.1,75,50,25,22,"
            "8,2.5,0.6,1.5,10.0,80,120,1.5,5.0,2026-04,3.0,B\n"
        )

        result = check_kill_theses(
            portfolio_signals_path=portfolio_csv,
            log_path=log_file,
        )
        # EG=10.0 >= 5.0, so custom condition should NOT trigger
        assert len(result["triggered_theses"]) == 0
        assert len(result["active_theses"]) == 1

    def test_no_conditions_field_backwards_compatible(self, tmp_path):
        """Thesis without conditions field should still work."""
        from trade_modules.committee_scorecard import (
            check_kill_theses,
            log_kill_theses,
        )
        log_file = tmp_path / "kill_theses.json"
        theses = [{
            "ticker": "AAPL",
            "kill_thesis": "Revenue collapse",
            # No conditions field
        }]
        log_kill_theses("2026-03-01", theses, log_path=log_file)

        portfolio_csv = tmp_path / "portfolio.csv"
        portfolio_csv.write_text(
            "TKR,NAME,CAP,PRC,TGT,UP%,#T,%%B,#A,AM,A,EXR,B,52W,2H,PET,PEF,"
            "P/S,PEG,DV,SI,EG,PP,ROE,DE,FCF,ERN,SZ,BS\n"
            "AAPL,Apple,MEGA,150,180,20,15,60,20,2.0,A,12,1.1,75,50,25,22,"
            "8,2.5,0.6,1.5,10.0,80,120,1.5,5.0,2026-04,3.0,B\n"
        )

        result = check_kill_theses(
            portfolio_signals_path=portfolio_csv,
            log_path=log_file,
        )
        # No conditions = no custom triggers, no heuristic triggers either
        assert len(result["active_theses"]) == 1
