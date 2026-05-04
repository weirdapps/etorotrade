"""
Tests for Kill Thesis Monitoring — CIO Review v4 Finding F11.

F11: Structured storage and automated heuristic checking of kill theses
for BUY/ADD positions in the committee system.

All tests use tmp_path for file isolation — no filesystem side-effects.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from trade_modules.committee_scorecard import (
    KILL_THESIS_DEFAULT_EXPIRY_DAYS,
    KILL_THESIS_LOG_PATH,
    _load_kill_theses,
    _load_portfolio_signals,
    _safe_float,
    _save_kill_theses,
    check_kill_theses,
    expire_old_theses,
    log_kill_theses,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def kill_thesis_path(tmp_path):
    """Path for kill thesis log JSON."""
    return tmp_path / "committee" / "kill_thesis_log.json"


@pytest.fixture
def portfolio_csv_path(tmp_path):
    """Path for mock portfolio CSV."""
    return tmp_path / "portfolio.csv"


def _write_portfolio_csv(path: Path, rows: list) -> None:
    """Write a mock portfolio CSV with headers and rows."""
    headers = "TKR,NAME,CAP,PRC,TGT,UP%,#T,%B,#A,AM,A,EXR,B,52W,2H,PET,PEF,P/S,PEG,DV,SI,EG,PP,ROE,DE,FCF,ERN,SZ,BS"
    lines = [headers]
    for row in rows:
        lines.append(row)
    path.write_text("\n".join(lines) + "\n")


def _make_thesis(
    ticker: str,
    kill_thesis: str = "Test thesis",
    status: str = "active",
    committee_date: str = "2026-03-16",
    expiry_date=None,
) -> dict:
    """Helper to build a thesis dict."""
    return {
        "ticker": ticker,
        "kill_thesis": kill_thesis,
        "committee_date": committee_date,
        "status": status,
        "expiry_date": expiry_date,
    }


# ============================================================
# Constants
# ============================================================


class TestConstants:
    """Tests for F11 constants."""

    def test_kill_thesis_log_path(self):
        """KILL_THESIS_LOG_PATH is under ~/.weirdapps-trading/committee/."""
        assert "committee" in str(KILL_THESIS_LOG_PATH)
        assert "kill_thesis_log.json" in str(KILL_THESIS_LOG_PATH)

    def test_default_expiry_days(self):
        """Default expiry is 90 days."""
        assert KILL_THESIS_DEFAULT_EXPIRY_DAYS == 90


# ============================================================
# _safe_float
# ============================================================


class TestSafeFloat:
    """Tests for the _safe_float helper."""

    def test_normal_float(self):
        assert _safe_float("42.5") == pytest.approx(42.5)

    def test_integer_string(self):
        assert _safe_float("10") == pytest.approx(10.0)

    def test_percentage_strip(self):
        assert _safe_float("88%") == pytest.approx(88.0)

    def test_none_returns_default(self):
        assert _safe_float(None) == pytest.approx(0.0)
        assert _safe_float(None, default=99.0) == pytest.approx(99.0)

    def test_dash_returns_default(self):
        assert _safe_float("--") == pytest.approx(0.0)

    def test_na_returns_default(self):
        assert _safe_float("N/A") == pytest.approx(0.0)

    def test_empty_string_returns_default(self):
        assert _safe_float("") == pytest.approx(0.0)

    def test_nan_string_returns_default(self):
        assert _safe_float("nan") == pytest.approx(0.0)

    def test_invalid_string_returns_default(self):
        assert _safe_float("abc", default=-1.0) == pytest.approx(-1.0)

    def test_negative_float(self):
        assert _safe_float("-3.5") == pytest.approx(-3.5)

    def test_actual_float_value(self):
        assert _safe_float(3.14) == pytest.approx(3.14)


# ============================================================
# log_kill_theses
# ============================================================


class TestLogKillTheses:
    """Tests for F11 — Logging kill theses."""

    def test_log_new_theses(self, kill_thesis_path):
        """Logs new theses to an empty file."""
        theses = [
            _make_thesis("NVDA", "Fails if Q2 revenue growth <40%"),
            _make_thesis("AAPL", "Fails if iPhone sales decline >10%"),
        ]

        log_kill_theses("2026-03-16", theses, log_path=kill_thesis_path)

        loaded = _load_kill_theses(kill_thesis_path)
        assert len(loaded) == 2
        assert loaded[0]["ticker"] == "NVDA"
        assert loaded[0]["kill_thesis"] == "Fails if Q2 revenue growth <40%"
        assert loaded[0]["committee_date"] == "2026-03-16"
        assert loaded[0]["status"] == "active"
        assert loaded[1]["ticker"] == "AAPL"

    def test_creates_parent_dirs(self, tmp_path):
        """Creates parent directories if they don't exist."""
        deep_path = tmp_path / "a" / "b" / "c" / "thesis.json"
        theses = [_make_thesis("TSLA")]

        log_kill_theses("2026-03-16", theses, log_path=deep_path)

        assert deep_path.exists()
        loaded = _load_kill_theses(deep_path)
        assert len(loaded) == 1

    def test_appends_to_existing(self, kill_thesis_path):
        """Appends new theses to existing ones."""
        log_kill_theses(
            "2026-03-09",
            [_make_thesis("NVDA", committee_date="2026-03-09")],
            log_path=kill_thesis_path,
        )
        log_kill_theses(
            "2026-03-16",
            [_make_thesis("AAPL")],
            log_path=kill_thesis_path,
        )

        loaded = _load_kill_theses(kill_thesis_path)
        assert len(loaded) == 2
        tickers = [t["ticker"] for t in loaded]
        assert "NVDA" in tickers
        assert "AAPL" in tickers

    def test_duplicate_prevention_same_ticker_same_date(self, kill_thesis_path):
        """Does not add duplicate for same ticker+date."""
        theses = [_make_thesis("NVDA")]

        log_kill_theses("2026-03-16", theses, log_path=kill_thesis_path)
        log_kill_theses("2026-03-16", theses, log_path=kill_thesis_path)

        loaded = _load_kill_theses(kill_thesis_path)
        assert len(loaded) == 1

    def test_same_ticker_different_date_allowed(self, kill_thesis_path):
        """Same ticker on different dates is not a duplicate."""
        log_kill_theses(
            "2026-03-09",
            [_make_thesis("NVDA", committee_date="2026-03-09")],
            log_path=kill_thesis_path,
        )
        log_kill_theses(
            "2026-03-16",
            [_make_thesis("NVDA")],
            log_path=kill_thesis_path,
        )

        loaded = _load_kill_theses(kill_thesis_path)
        assert len(loaded) == 2

    def test_skips_thesis_without_ticker(self, kill_thesis_path):
        """Skips theses that have no ticker."""
        theses = [
            {"kill_thesis": "No ticker here"},
            _make_thesis("AAPL"),
        ]

        log_kill_theses("2026-03-16", theses, log_path=kill_thesis_path)

        loaded = _load_kill_theses(kill_thesis_path)
        assert len(loaded) == 1
        assert loaded[0]["ticker"] == "AAPL"

    def test_empty_theses_list(self, kill_thesis_path):
        """Empty theses list creates an empty log."""
        log_kill_theses("2026-03-16", [], log_path=kill_thesis_path)

        loaded = _load_kill_theses(kill_thesis_path)
        assert loaded == []

    def test_default_status_is_active(self, kill_thesis_path):
        """Thesis status defaults to 'active' when not specified."""
        theses = [{"ticker": "MSFT", "kill_thesis": "Test"}]

        log_kill_theses("2026-03-16", theses, log_path=kill_thesis_path)

        loaded = _load_kill_theses(kill_thesis_path)
        assert loaded[0]["status"] == "active"

    def test_custom_expiry_date(self, kill_thesis_path):
        """Thesis with explicit expiry_date is stored."""
        theses = [
            _make_thesis("NVDA", expiry_date="2026-06-16"),
        ]

        log_kill_theses("2026-03-16", theses, log_path=kill_thesis_path)

        loaded = _load_kill_theses(kill_thesis_path)
        assert loaded[0]["expiry_date"] == "2026-06-16"


# ============================================================
# _load_kill_theses / _save_kill_theses
# ============================================================


class TestKillThesisIO:
    """Tests for kill thesis load/save helpers."""

    def test_load_missing_file(self, tmp_path):
        """Loading a missing file returns empty list."""
        result = _load_kill_theses(tmp_path / "nonexistent.json")
        assert result == []

    def test_load_corrupt_json(self, kill_thesis_path):
        """Loading corrupt JSON returns empty list."""
        kill_thesis_path.parent.mkdir(parents=True, exist_ok=True)
        kill_thesis_path.write_text("not valid json {{{")

        result = _load_kill_theses(kill_thesis_path)
        assert result == []

    def test_load_non_list_json(self, kill_thesis_path):
        """Loading JSON that is not a list returns empty list."""
        kill_thesis_path.parent.mkdir(parents=True, exist_ok=True)
        kill_thesis_path.write_text('{"not": "a list"}')

        result = _load_kill_theses(kill_thesis_path)
        assert result == []

    def test_roundtrip(self, kill_thesis_path):
        """Save then load returns identical data."""
        theses = [
            _make_thesis("AAPL", "Thesis A"),
            _make_thesis("MSFT", "Thesis B"),
        ]

        _save_kill_theses(theses, kill_thesis_path)
        loaded = _load_kill_theses(kill_thesis_path)

        assert len(loaded) == 2
        assert loaded[0]["ticker"] == "AAPL"
        assert loaded[1]["ticker"] == "MSFT"


# ============================================================
# _load_portfolio_signals
# ============================================================


class TestLoadPortfolioSignals:
    """Tests for portfolio signal CSV loading."""

    def test_load_valid_csv(self, portfolio_csv_path):
        """Loads valid portfolio CSV into dict keyed by ticker."""
        _write_portfolio_csv(
            portfolio_csv_path,
            [
                "AAPL,Apple,2.5T,185.00,210.00,13.5%,30,88%,30,-2,A,12.0%,0.8,72,N,28.0,26.0,7.5,2.1,0.5%,1.2%,15.0,-3.0,150.0,120.0,5.2%,01/30,25k,B",
                "NVDA,NVIDIA,2.0T,450.00,500.00,11.1%,35,92%,35,3,A,10.2%,0.9,85,N,55.0,45.0,20.0,1.5,0.0%,1.5%,25.0,-1.0,90.0,50.0,3.1%,02/20,20k,S",
            ],
        )

        signals = _load_portfolio_signals(portfolio_csv_path)

        assert len(signals) == 2
        assert "AAPL" in signals
        assert signals["AAPL"]["BS"] == "B"
        assert signals["NVDA"]["BS"] == "S"
        assert signals["AAPL"]["AM"] == "-2"

    def test_missing_file(self, tmp_path):
        """Returns empty dict for missing file."""
        result = _load_portfolio_signals(tmp_path / "missing.csv")
        assert result == {}

    def test_empty_csv(self, portfolio_csv_path):
        """Returns empty dict for CSV with only headers."""
        portfolio_csv_path.write_text(
            "TKR,NAME,CAP,PRC,TGT,UP%,#T,%B,#A,AM,A,EXR,B,52W,2H,PET,PEF,P/S,PEG,DV,SI,EG,PP,ROE,DE,FCF,ERN,SZ,BS\n"
        )
        result = _load_portfolio_signals(portfolio_csv_path)
        assert result == {}


# ============================================================
# check_kill_theses
# ============================================================


class TestCheckKillTheses:
    """Tests for F11 — Heuristic kill thesis checking."""

    def test_no_theses_returns_empty(self, kill_thesis_path, portfolio_csv_path):
        """Returns empty lists when no theses exist."""
        result = check_kill_theses(
            portfolio_signals_path=portfolio_csv_path,
            log_path=kill_thesis_path,
        )
        assert result["active_theses"] == []
        assert result["triggered_theses"] == []
        assert result["expired_theses"] == []

    def test_signal_deterioration_trigger(self, kill_thesis_path, portfolio_csv_path):
        """Triggers when ticker signal is SELL."""
        log_kill_theses(
            "2026-03-16",
            [_make_thesis("NVDA", "Fails if VIX >35")],
            log_path=kill_thesis_path,
        )

        # NVDA has BS=S (SELL signal)
        _write_portfolio_csv(
            portfolio_csv_path,
            [
                "NVDA,NVIDIA,2.0T,450.00,500.00,11.1%,35,92%,35,0,A,10.2%,0.9,85,N,55.0,45.0,20.0,1.5,0.0%,1.5%,25.0,-1.0,90.0,50.0,3.1%,02/20,20k,S",
            ],
        )

        result = check_kill_theses(
            portfolio_signals_path=portfolio_csv_path,
            log_path=kill_thesis_path,
        )

        assert len(result["triggered_theses"]) == 1
        assert result["triggered_theses"][0]["ticker"] == "NVDA"
        assert "signal_deteriorated" in result["triggered_theses"][0]["triggers"]

    def test_price_collapsed_trigger(self, kill_thesis_path, portfolio_csv_path):
        """Triggers when 52W performance drops below 40."""
        log_kill_theses(
            "2026-03-16",
            [_make_thesis("BADCO", "Fails if revenue declines")],
            log_path=kill_thesis_path,
        )

        # BADCO has 52W=25 (below 40 threshold)
        _write_portfolio_csv(
            portfolio_csv_path,
            [
                "BADCO,Bad Company,1.0B,10.00,15.00,50.0%,5,60%,5,0,A,30.0%,0.5,25,N,10.0,8.0,1.0,0.5,0.0%,5.0%,10.0,0.0,5.0,80.0,1.0%,03/15,5k,H",
            ],
        )

        result = check_kill_theses(
            portfolio_signals_path=portfolio_csv_path,
            log_path=kill_thesis_path,
        )

        assert len(result["triggered_theses"]) == 1
        assert "price_collapsed" in result["triggered_theses"][0]["triggers"]

    def test_analyst_downgrade_trigger(self, kill_thesis_path, portfolio_csv_path):
        """Triggers when AM (analyst momentum) is strongly negative (<-5)."""
        log_kill_theses(
            "2026-03-16",
            [_make_thesis("WARN", "Fails if analyst consensus deteriorates")],
            log_path=kill_thesis_path,
        )

        # WARN has AM=-8 (strongly negative)
        _write_portfolio_csv(
            portfolio_csv_path,
            [
                "WARN,Warning Co,5.0B,30.00,35.00,16.7%,10,55%,10,-8,A,9.2%,0.6,60,N,15.0,14.0,2.0,1.0,1.0%,3.0%,8.0,-2.0,12.0,40.0,2.5%,04/20,8k,H",
            ],
        )

        result = check_kill_theses(
            portfolio_signals_path=portfolio_csv_path,
            log_path=kill_thesis_path,
        )

        assert len(result["triggered_theses"]) == 1
        assert "analyst_downgrade" in result["triggered_theses"][0]["triggers"]

    def test_multiple_triggers_on_same_thesis(self, kill_thesis_path, portfolio_csv_path):
        """Multiple triggers fire on the same thesis."""
        log_kill_theses(
            "2026-03-16",
            [_make_thesis("DOOM", "Complete failure")],
            log_path=kill_thesis_path,
        )

        # DOOM has BS=S, 52W=20, AM=-10 (all three triggers)
        _write_portfolio_csv(
            portfolio_csv_path,
            [
                "DOOM,Doomed Inc,500M,5.00,10.00,100.0%,3,30%,3,-10,C,30.0%,0.3,20,N,--,--,0.5,--,0.0%,15.0%,5.0,0.0,--,--,--,--,2k,S",
            ],
        )

        result = check_kill_theses(
            portfolio_signals_path=portfolio_csv_path,
            log_path=kill_thesis_path,
        )

        assert len(result["triggered_theses"]) == 1
        triggers = result["triggered_theses"][0]["triggers"]
        assert "signal_deteriorated" in triggers
        assert "price_collapsed" in triggers
        assert "analyst_downgrade" in triggers

    def test_no_trigger_for_healthy_stock(self, kill_thesis_path, portfolio_csv_path):
        """No triggers fire for a healthy stock."""
        log_kill_theses(
            "2026-03-16",
            [_make_thesis("AAPL", "Fails if iPhone sales collapse")],
            log_path=kill_thesis_path,
        )

        # AAPL is healthy: BS=B, 52W=72, AM=2
        _write_portfolio_csv(
            portfolio_csv_path,
            [
                "AAPL,Apple,2.5T,185.00,210.00,13.5%,30,88%,30,2,A,12.0%,0.8,72,N,28.0,26.0,7.5,2.1,0.5%,1.2%,15.0,-3.0,150.0,120.0,5.2%,01/30,25k,B",
            ],
        )

        result = check_kill_theses(
            portfolio_signals_path=portfolio_csv_path,
            log_path=kill_thesis_path,
        )

        assert len(result["active_theses"]) == 1
        assert len(result["triggered_theses"]) == 0

    def test_thesis_not_in_portfolio_stays_active(self, kill_thesis_path, portfolio_csv_path):
        """Thesis for ticker not in portfolio CSV stays active (no data)."""
        log_kill_theses(
            "2026-03-16",
            [_make_thesis("ABSENT", "No data available")],
            log_path=kill_thesis_path,
        )

        # Portfolio has other tickers, not ABSENT
        _write_portfolio_csv(
            portfolio_csv_path,
            [
                "AAPL,Apple,2.5T,185.00,210.00,13.5%,30,88%,30,2,A,12.0%,0.8,72,N,28.0,26.0,7.5,2.1,0.5%,1.2%,15.0,-3.0,150.0,120.0,5.2%,01/30,25k,B",
            ],
        )

        result = check_kill_theses(
            portfolio_signals_path=portfolio_csv_path,
            log_path=kill_thesis_path,
        )

        assert len(result["active_theses"]) == 1
        assert result["active_theses"][0]["ticker"] == "ABSENT"

    def test_expired_theses_by_age(self, kill_thesis_path, portfolio_csv_path):
        """Theses older than 90 days are expired."""
        old_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")

        log_kill_theses(
            old_date,
            [_make_thesis("OLD", committee_date=old_date)],
            log_path=kill_thesis_path,
        )

        result = check_kill_theses(
            portfolio_signals_path=portfolio_csv_path,
            log_path=kill_thesis_path,
        )

        assert len(result["expired_theses"]) == 1
        assert result["expired_theses"][0]["ticker"] == "OLD"
        assert result["expired_theses"][0]["status"] == "expired"

    def test_explicit_expiry_date(self, kill_thesis_path, portfolio_csv_path):
        """Thesis with explicit expiry_date in the past is expired."""
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        log_kill_theses(
            "2026-03-16",
            [_make_thesis("EXPR", expiry_date=yesterday)],
            log_path=kill_thesis_path,
        )

        result = check_kill_theses(
            portfolio_signals_path=portfolio_csv_path,
            log_path=kill_thesis_path,
        )

        assert len(result["expired_theses"]) == 1
        assert result["expired_theses"][0]["ticker"] == "EXPR"

    def test_already_triggered_stays_triggered(self, kill_thesis_path, portfolio_csv_path):
        """Previously triggered theses are returned in triggered list."""
        theses = [_make_thesis("PREV", status="triggered")]
        _save_kill_theses(theses, kill_thesis_path)

        result = check_kill_theses(
            portfolio_signals_path=portfolio_csv_path,
            log_path=kill_thesis_path,
        )

        assert len(result["triggered_theses"]) == 1
        assert result["triggered_theses"][0]["ticker"] == "PREV"

    def test_already_expired_stays_expired(self, kill_thesis_path, portfolio_csv_path):
        """Previously expired theses are returned in expired list."""
        theses = [_make_thesis("OLDEXP", status="expired")]
        _save_kill_theses(theses, kill_thesis_path)

        result = check_kill_theses(
            portfolio_signals_path=portfolio_csv_path,
            log_path=kill_thesis_path,
        )

        assert len(result["expired_theses"]) == 1
        assert result["expired_theses"][0]["ticker"] == "OLDEXP"

    def test_mixed_theses(self, kill_thesis_path, portfolio_csv_path):
        """Mix of active, triggered, and expired theses."""
        old_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")

        theses = [
            _make_thesis("HEALTHY", committee_date="2026-03-16"),
            _make_thesis("SELLING", committee_date="2026-03-16"),
            _make_thesis("ANCIENT", committee_date=old_date),
        ]
        _save_kill_theses(theses, kill_thesis_path)

        _write_portfolio_csv(
            portfolio_csv_path,
            [
                "HEALTHY,Healthy Inc,10B,100.00,120.00,20.0%,15,80%,15,2,A,16.0%,0.7,70,N,20.0,18.0,5.0,1.5,1.0%,2.0%,12.0,-1.0,25.0,30.0,4.0%,04/15,12k,B",
                "SELLING,Sell Corp,3B,25.00,20.00,-20.0%,8,30%,8,-3,C,-6.0%,0.4,45,N,12.0,15.0,1.5,--,0.0%,8.0%,5.0,0.0,8.0,60.0,1.5%,05/01,6k,S",
            ],
        )

        result = check_kill_theses(
            portfolio_signals_path=portfolio_csv_path,
            log_path=kill_thesis_path,
        )

        assert len(result["active_theses"]) == 1
        assert result["active_theses"][0]["ticker"] == "HEALTHY"

        assert len(result["triggered_theses"]) == 1
        assert result["triggered_theses"][0]["ticker"] == "SELLING"

        assert len(result["expired_theses"]) == 1
        assert result["expired_theses"][0]["ticker"] == "ANCIENT"

    def test_missing_portfolio_csv(self, kill_thesis_path, tmp_path):
        """Theses stay active when portfolio CSV does not exist."""
        log_kill_theses(
            "2026-03-16",
            [_make_thesis("AAPL")],
            log_path=kill_thesis_path,
        )

        result = check_kill_theses(
            portfolio_signals_path=tmp_path / "nonexistent.csv",
            log_path=kill_thesis_path,
        )

        # No signal data => no triggers, thesis remains active
        assert len(result["active_theses"]) == 1

    def test_statuses_persisted_after_check(self, kill_thesis_path, portfolio_csv_path):
        """Status changes are saved to disk after checking."""
        log_kill_theses(
            "2026-03-16",
            [_make_thesis("NVDA", "Fails if VIX >35")],
            log_path=kill_thesis_path,
        )

        _write_portfolio_csv(
            portfolio_csv_path,
            [
                "NVDA,NVIDIA,2.0T,450.00,500.00,11.1%,35,92%,35,0,A,10.2%,0.9,85,N,55.0,45.0,20.0,1.5,0.0%,1.5%,25.0,-1.0,90.0,50.0,3.1%,02/20,20k,S",
            ],
        )

        check_kill_theses(
            portfolio_signals_path=portfolio_csv_path,
            log_path=kill_thesis_path,
        )

        # Reload from disk — status should be updated
        reloaded = _load_kill_theses(kill_thesis_path)
        assert reloaded[0]["status"] == "triggered"


# ============================================================
# expire_old_theses
# ============================================================


class TestExpireOldTheses:
    """Tests for F11 — Expiring old kill theses."""

    def test_expire_old_theses(self, kill_thesis_path):
        """Expires theses older than the specified days."""
        old_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")
        recent_date = datetime.now().strftime("%Y-%m-%d")

        theses = [
            _make_thesis("OLD", committee_date=old_date),
            _make_thesis("NEW", committee_date=recent_date),
        ]
        _save_kill_theses(theses, kill_thesis_path)

        expired_count = expire_old_theses(days=90, log_path=kill_thesis_path)

        assert expired_count == 1

        reloaded = _load_kill_theses(kill_thesis_path)
        old_thesis = next(t for t in reloaded if t["ticker"] == "OLD")
        new_thesis = next(t for t in reloaded if t["ticker"] == "NEW")
        assert old_thesis["status"] == "expired"
        assert new_thesis["status"] == "active"

    def test_expire_with_custom_days(self, kill_thesis_path):
        """Expires theses with a custom day threshold."""
        date_15_days_ago = (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d")

        theses = [
            _make_thesis("RECENT", committee_date=date_15_days_ago),
        ]
        _save_kill_theses(theses, kill_thesis_path)

        # 30-day threshold: 15-day-old thesis is NOT expired
        expired_count = expire_old_theses(days=30, log_path=kill_thesis_path)
        assert expired_count == 0

        # 10-day threshold: 15-day-old thesis IS expired
        expired_count = expire_old_theses(days=10, log_path=kill_thesis_path)
        assert expired_count == 1

    def test_no_theses(self, kill_thesis_path):
        """Returns 0 when no theses exist."""
        expired_count = expire_old_theses(log_path=kill_thesis_path)
        assert expired_count == 0

    def test_already_expired_not_counted(self, kill_thesis_path):
        """Already expired theses are not counted again."""
        old_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")

        theses = [
            _make_thesis("OLD", committee_date=old_date, status="expired"),
        ]
        _save_kill_theses(theses, kill_thesis_path)

        expired_count = expire_old_theses(days=90, log_path=kill_thesis_path)
        assert expired_count == 0

    def test_triggered_theses_not_expired(self, kill_thesis_path):
        """Triggered theses are not expired (they have a different status)."""
        old_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")

        theses = [
            _make_thesis("TRIG", committee_date=old_date, status="triggered"),
        ]
        _save_kill_theses(theses, kill_thesis_path)

        expired_count = expire_old_theses(days=90, log_path=kill_thesis_path)
        assert expired_count == 0

    def test_invalid_date_skipped(self, kill_thesis_path):
        """Theses with invalid committee_date are skipped (not expired)."""
        theses = [
            _make_thesis("BAD", committee_date="not-a-date"),
        ]
        _save_kill_theses(theses, kill_thesis_path)

        expired_count = expire_old_theses(days=90, log_path=kill_thesis_path)
        assert expired_count == 0

        reloaded = _load_kill_theses(kill_thesis_path)
        assert reloaded[0]["status"] == "active"

    def test_expiry_persisted(self, kill_thesis_path):
        """Expiry changes are persisted to disk."""
        old_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")

        theses = [_make_thesis("OLD", committee_date=old_date)]
        _save_kill_theses(theses, kill_thesis_path)

        expire_old_theses(days=90, log_path=kill_thesis_path)

        reloaded = _load_kill_theses(kill_thesis_path)
        assert reloaded[0]["status"] == "expired"
