"""Tests for trade_modules.validation.regime_join (S0-T2)."""

import numpy as np

from trade_modules.validation.regime_join import attach_regime, label_dates

# ---------------------------------------------------------------------------
# Synthetic arrays used across label_dates tests
# ---------------------------------------------------------------------------
N = 600
_spy = np.linspace(100, 200, N)  # rising SPY
_vix = np.full(N, 12.0)  # flat low VIX → risk_on
_vix3m = np.full(N, 14.0)  # flat VIX3M (contango, risk_on)
_dates = [f"2020-{(i // 21 + 1):02d}-{(i % 21 + 1):02d}" for i in range(N)]

VALID_REGIMES = {"risk_on", "neutral", "risk_off", "crisis"}


# ---------------------------------------------------------------------------
# attach_regime tests
# ---------------------------------------------------------------------------


def test_attach_regime_maps_labels_correctly():
    rows = [
        {"signal_date": "2024-01-01", "ticker": "AAPL"},
        {"signal_date": "2024-01-02", "ticker": "MSFT"},
    ]
    date_to_regime = {"2024-01-01": "risk_on", "2024-01-02": "risk_off"}

    result = attach_regime(rows, date_to_regime)

    assert result[0]["regime"] == "risk_on"
    assert result[1]["regime"] == "risk_off"


def test_attach_regime_missing_date_defaults_neutral():
    rows = [{"signal_date": "2099-12-31", "ticker": "XYZ"}]
    result = attach_regime(rows, {})
    assert result[0]["regime"] == "neutral"


def test_attach_regime_does_not_mutate_input():
    rows = [{"signal_date": "2024-01-01", "ticker": "AAPL"}]
    original_row = dict(rows[0])

    attach_regime(rows, {"2024-01-01": "risk_on"})

    # Input list unchanged
    assert len(rows) == 1
    # Input dict unchanged (no 'regime' key added to original)
    assert rows[0] == original_row


# ---------------------------------------------------------------------------
# label_dates tests
# ---------------------------------------------------------------------------


def test_label_dates_returns_valid_labels():
    # Positions ≥ 30 — expect valid regime labels
    target_dates = [_dates[50], _dates[100], _dates[300], _dates[500]]
    result = label_dates(target_dates, _vix, _vix3m, _spy, _dates)

    assert set(result.keys()) == set(target_dates)
    for date, label in result.items():
        assert label in VALID_REGIMES, f"Invalid label '{label}' for date {date}"


def test_label_dates_short_history_returns_neutral():
    # Position < 30 → 'neutral'
    early_date = _dates[10]  # position 10 < 30
    result = label_dates([early_date], _vix, _vix3m, _spy, _dates)
    assert result[early_date] == "neutral"


def test_label_dates_not_found_returns_neutral():
    missing_date = "1900-01-01"
    result = label_dates([missing_date], _vix, _vix3m, _spy, _dates)
    assert result[missing_date] == "neutral"


def test_label_dates_returns_label_for_every_requested_date():
    requested = [_dates[30], _dates[31], "9999-99-99", _dates[5]]
    result = label_dates(requested, _vix, _vix3m, _spy, _dates)
    # Every requested date must appear as a key
    for d in requested:
        assert d in result
