"""Tests for scripts/market_snapshot.py.

Regression context (2026-09-23/24): the midday brief quoted moves that were not the day's move,
and validate_brief.py passed them because it checks the post against this same snapshot.

- Yahoo's rolling futures series switched contract mid-morning, so "Brent -3.86%" and
  "WTI -4.93%" on 23 Sep were the gap between two contracts.
- Yahoo's completed FX daily bars carry the day's opening tick as their close, so EUR/USD and
  USD/JPY were measured from the previous day's OPEN: -0.62% and +0.83% on 24 Sep, for a day
  that was -0.08% and +0.31% against the New York close.
- Yahoo's daily history has no 22 Sep row for any cash index or for ^TNX, so the 23 Sep index
  moves were two-day moves and the 10-year was "up from 4.96%" (21 Sep) instead of 4.97%.
- ^TNX has no bar before the US session, so at 14:00 Athens it was reported as a holiday.
"""

import importlib.util
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).parent.parent.parent.parent / "scripts" / "market_snapshot.py"

# 14:03 Athens on Thursday 24 Sep 2026, when the daily pipeline ran.
MIDDAY = datetime(2026, 9, 24, 11, 3, tzinfo=timezone.utc)


def _load():
    spec = importlib.util.spec_from_file_location("market_snapshot", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ms = _load()


def daily(tz, rows):
    """A yfinance daily frame from (date, open, close) rows."""
    return pd.DataFrame(
        {"Open": [o for _, o, _ in rows], "Close": [c for _, _, c in rows]},
        index=pd.DatetimeIndex([pd.Timestamp(d, tz=tz) for d, _, _ in rows]),
    )


def intraday(tz, rows):
    """A yfinance intraday frame from (local timestamp, close) rows."""
    return pd.DataFrame(
        {"Open": [c for _, c in rows], "Close": [c for _, c in rows]},
        index=pd.DatetimeIndex([pd.Timestamp(t, tz=tz) for t, _ in rows]),
    )


@pytest.fixture
def market(monkeypatch):
    """Serve canned frames keyed by (symbol, interval); anything else comes back empty,
    as yfinance returns for an unknown or expired symbol."""
    frames = {}
    calls = []

    def fake_history(symbol, period="5d", interval="1d"):
        calls.append((symbol, interval))
        return frames.get((symbol, interval), pd.DataFrame())

    monkeypatch.setattr(ms, "_history", fake_history)

    def snapshot(instruments, now=MIDDAY):
        monkeypatch.setattr(ms, "INSTRUMENTS", instruments)
        return ms.fetch_snapshot(now=now)

    return frames, calls, snapshot


# --- rolling futures -------------------------------------------------------------------------


def test_dated_contracts_cover_the_front_months_of_each_root():
    on = date(2026, 9, 24)
    assert ms._dated_contracts("BZ=F", on) == [
        "BZU26.NYM",
        "BZV26.NYM",
        "BZX26.NYM",
        "BZZ26.NYM",
        "BZF27.NYM",
    ]
    assert ms._dated_contracts("ES=F", on) == ["ESU26.CME", "ESZ26.CME"]
    assert ms._dated_contracts("GC=F", on) == ["GCV26.CMX", "GCZ26.CMX"]
    assert ms._dated_contracts("CL=F", date(2026, 11, 2))[-3:] == [
        "CLF27.NYM",
        "CLG27.NYM",
        "CLH27.NYM",
    ]


def test_a_contract_switch_is_not_reported_as_a_move(market):
    """23 Sep: the rolling series closed Tuesday on November and quoted December by midday."""
    frames, _, snapshot = market
    ny = "America/New_York"
    frames["BZ=F", "1d"] = daily(ny, [("2026-09-22", 99.88, 99.25), ("2026-09-23", 94.78, 95.42)])
    frames["BZX26.NYM", "1d"] = daily(
        ny, [("2026-09-22", 99.88, 99.25), ("2026-09-23", 99.05, 100.21)]
    )
    frames["BZZ26.NYM", "1d"] = daily(
        ny, [("2026-09-22", 95.87, 95.41), ("2026-09-23", 94.78, 95.42)]
    )

    brent = snapshot({"BZ=F": "Brent crude"})["instruments"]["BZ=F"]

    assert brent["contract"] == "BZZ26.NYM"
    assert (brent["price"], brent["prev_close"], brent["change_pct"]) == (95.42, 95.41, 0.01)


def test_a_series_that_flips_back_is_measured_on_the_contract_it_quotes(market):
    """24 Sep: the rolling series' previous close was December, its last price November."""
    frames, _, snapshot = market
    ny = "America/New_York"
    frames["BZ=F", "1d"] = daily(ny, [("2026-09-23", 94.78, 98.12), ("2026-09-24", 98.23, 104.45)])
    frames["BZX26.NYM", "1d"] = daily(
        ny, [("2026-09-23", 99.05, 103.08), ("2026-09-24", 103.39, 104.45)]
    )
    frames["BZZ26.NYM", "1d"] = daily(
        ny, [("2026-09-23", 94.78, 98.12), ("2026-09-24", 98.23, 99.40)]
    )

    brent = snapshot({"BZ=F": "Brent crude"})["instruments"]["BZ=F"]

    assert brent["contract"] == "BZX26.NYM"
    assert (brent["prev_close"], brent["change_pct"]) == (103.08, 1.33)


def test_an_expired_contract_never_matches_even_at_the_same_price(market):
    frames, _, snapshot = market
    ny = "America/New_York"
    frames["CL=F", "1d"] = daily(ny, [("2026-09-23", 89.89, 92.16), ("2026-09-24", 92.72, 93.01)])
    # October expired on 22 Sep; its stale last close happens to equal today's quote.
    frames["CLV26.NYM", "1d"] = daily(ny, [("2026-09-21", 96.0, 95.0), ("2026-09-22", 95.0, 93.01)])
    frames["CLX26.NYM", "1d"] = daily(
        ny, [("2026-09-23", 89.89, 92.16), ("2026-09-24", 92.72, 93.02)]
    )

    wti = snapshot({"CL=F": "WTI crude"})["instruments"]["CL=F"]

    assert wti["contract"] == "CLX26.NYM"
    assert wti["prev_close"] == 92.16


def test_an_unmatched_rolling_quote_is_left_out_not_guessed(market):
    frames, _, snapshot = market
    ny = "America/New_York"
    frames["BZ=F", "1d"] = daily(ny, [("2026-09-23", 94.78, 98.12), ("2026-09-24", 98.23, 104.45)])
    frames["BZX26.NYM", "1d"] = daily(
        ny, [("2026-09-23", 99.05, 108.00), ("2026-09-24", 103.39, 110.00)]
    )

    snap = snapshot({"BZ=F": "Brent crude"})

    assert "BZ=F" not in snap["instruments"]
    assert any(e.startswith("BZ=F (Brent crude)") for e in snap["errors"])


def test_a_candidate_that_raises_is_skipped_not_fatal(market, monkeypatch):
    frames, _, snapshot = market
    ny = "America/New_York"
    frames["ES=F", "1d"] = daily(
        ny, [("2026-09-23", 7830.0, 7772.5), ("2026-09-24", 7774.75, 7732.75)]
    )
    frames["ESZ26.CME", "1d"] = daily(
        ny, [("2026-09-23", 7830.0, 7772.5), ("2026-09-24", 7774.75, 7732.75)]
    )
    served = ms._history

    def history(symbol, period="5d", interval="1d"):
        if symbol == "ESU26.CME":
            raise RuntimeError("Quote not found for symbol: ESU26.CME")
        return served(symbol, period=period, interval=interval)

    monkeypatch.setattr(ms, "_history", history)

    es = snapshot({"ES=F": "S&P 500 futures"})["instruments"]["ES=F"]

    assert (es["contract"], es["change_pct"]) == ("ESZ26.CME", -0.51)


# --- FX --------------------------------------------------------------------------------------


def test_fx_moves_are_measured_from_the_previous_close_not_the_previous_open(market):
    frames, _, snapshot = market
    ldn = "Europe/London"
    # Yahoo's 23 Sep bar: opened 1.14471, "closed" 1.14479, although the pair traded down to
    # 1.1387 by the New York close and opened 24 Sep there.
    frames["EURUSD=X", "1d"] = daily(
        ldn, [("2026-09-23", 1.14471, 1.14479), ("2026-09-24", 1.13869, 1.13779)]
    )
    frames["JPY=X", "1d"] = daily(
        ldn, [("2026-09-23", 157.474, 157.464), ("2026-09-24", 158.297, 158.776)]
    )

    snap = snapshot({"EURUSD=X": "EUR/USD", "JPY=X": "USD/JPY"})["instruments"]

    assert snap["EURUSD=X"]["change_pct"] == -0.08
    assert snap["JPY=X"]["change_pct"] == 0.3


def test_fx_levels_keep_enough_decimals_to_show_the_move(market):
    frames, _, snapshot = market
    frames["EURUSD=X", "1d"] = daily(
        "Europe/London", [("2026-09-23", 1.14471, 1.14479), ("2026-09-24", 1.13869, 1.13779)]
    )

    eur = snapshot({"EURUSD=X": "EUR/USD"})["instruments"]["EURUSD=X"]

    assert (eur["price"], eur["prev_close"]) == (1.1378, 1.1387)


# --- sessions the daily history skipped ------------------------------------------------------


def test_a_session_missing_from_the_daily_history_is_read_from_intraday_bars(market):
    frames, _, snapshot = market
    chi = "America/Chicago"
    frames["^TNX", "1d"] = daily(
        chi,
        [("2026-09-18", 4.980, 4.998), ("2026-09-21", 4.953, 4.963), ("2026-09-23", 4.99, 5.114)],
    )
    frames["^TNX", "30m"] = intraday(
        chi,
        [
            ("2026-09-21 13:30", 4.963),
            ("2026-09-22 13:00", 4.968),
            ("2026-09-22 13:30", 4.968),
            ("2026-09-23 13:30", 5.114),
        ],
    )

    tnx = snapshot({"^TNX": "10Y UST yield"})["instruments"]["^TNX"]

    assert (tnx["price"], tnx["prev_close"], tnx["change_pct"]) == (5.11, 4.97, 2.94)


def test_a_real_holiday_keeps_the_last_close_before_it(market):
    """Japan was shut 21-23 Sep: no intraday bars either, so Friday's close is the reference."""
    frames, _, snapshot = market
    tokyo = "Asia/Tokyo"
    frames["^N225", "1d"] = daily(
        tokyo, [("2026-09-18", 64681.55, 65018.95), ("2026-09-24", 65476.44, 65513.99)]
    )
    frames["^N225", "30m"] = intraday(
        tokyo, [("2026-09-18 14:30", 65018.95), ("2026-09-24 14:30", 65513.99)]
    )

    nikkei = snapshot({"^N225": "Nikkei 225"})["instruments"]["^N225"]

    assert (nikkei["prev_close"], nikkei["change_pct"]) == (65018.95, 0.76)


def test_consecutive_sessions_need_no_intraday_lookup(market):
    frames, calls, snapshot = market
    frames["^GDAXI", "1d"] = daily(
        "Europe/Berlin", [("2026-09-23", 25724.13, 25410.63), ("2026-09-24", 25263.11, 25352.47)]
    )

    dax = snapshot({"^GDAXI": "DAX 40"})["instruments"]["^GDAXI"]

    assert (dax["prev_close"], dax["change_pct"]) == (25410.63, -0.23)
    assert ("^GDAXI", "30m") not in calls


# --- status ----------------------------------------------------------------------------------


def _tnx_after_close(frames):
    frames["^TNX", "1d"] = daily(
        "America/Chicago", [("2026-09-22", 4.95, 4.968), ("2026-09-23", 4.99, 5.114)]
    )


def test_the_us_yield_before_the_us_session_is_pre_open_not_a_holiday(market):
    frames, _, snapshot = market
    _tnx_after_close(frames)

    snap = snapshot({"^TNX": "10Y UST yield"})

    assert snap["instruments"]["^TNX"]["status"] == "pre-open"
    assert snap["holidays_detected"] == []


def test_no_bar_after_the_session_opened_is_still_a_holiday(market):
    frames, _, snapshot = market
    _tnx_after_close(frames)

    snap = snapshot(
        {"^TNX": "10Y UST yield"}, now=datetime(2026, 9, 24, 15, 0, tzinfo=timezone.utc)
    )

    assert snap["instruments"]["^TNX"]["status"] == "closed"
    assert snap["holidays_detected"] == ["^TNX"]


def test_an_exchange_closed_for_the_day_is_a_holiday(market):
    frames, _, snapshot = market
    frames["^KS11", "1d"] = daily(
        "Asia/Seoul", [("2026-09-22", 6950.0, 7007.72), ("2026-09-23", 7010.0, 7080.92)]
    )

    snap = snapshot({"^KS11": "KOSPI"})

    assert snap["instruments"]["^KS11"]["status"] == "closed"
    assert snap["holidays_detected"] == ["^KS11"]
    assert snap["today_athens"] == "2026-09-24"
