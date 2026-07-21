"""Tests for riskfirst.fx — currency inference, USD-bloc exposure, binding cap,
hedge advisory. Policy: UNHEDGED by default; cap the net USD-bloc concentration
and only ADVISE a hedge when over-exposed AND the carry is favorable.
HKD is USD-pegged, so it counts in the USD bloc.
"""

import numpy as np
import pytest

from trade_modules.riskfirst.fx import (
    USD_BLOC,
    bloc_exposure,
    cap_bloc,
    currency_of,
    hedge_advisory,
)


def test_currency_inference():
    assert currency_of("AAPL") == "USD"
    assert currency_of("BARC.L") == "GBP"
    assert currency_of("SAP.DE") == "EUR"
    assert currency_of("0700.HK") == "HKD"
    assert currency_of("7012.T") == "JPY"
    # Nordic listings float against the EUR -> resolve to their OWN currency so the
    # USD cap/ADV normalization applies the right rate (Investor AB is ~$115B, not $1.3T).
    assert currency_of("INVE-A.ST") == "SEK"
    assert currency_of("EQNR.OL") == "NOK"
    assert currency_of("NOVO-B.CO") == "DKK"
    # genuine eurozone listings stay EUR
    assert currency_of("ASML.AS") == "EUR"
    assert currency_of("NESTE.HE") == "EUR"
    # all of them remain OUTSIDE the USD bloc (bloc classification unchanged)
    assert not any(currency_of(t) in USD_BLOC for t in ("INVE-A.ST", "EQNR.OL", "NOVO-B.CO"))


def test_bloc_exposure_counts_usd_and_hkd():
    weights = {"AAPL": 0.3, "0700.HK": 0.2, "SAP.DE": 0.5}
    assert bloc_exposure(weights) == pytest.approx(0.5)  # USD 0.3 + HKD 0.2


def test_cap_bloc_scales_down_and_redistributes():
    w = cap_bloc(np.array([0.3, 0.2, 0.5]), np.array([True, True, False]), cap=0.4)
    assert w == pytest.approx([0.24, 0.16, 0.6])
    assert w.sum() == pytest.approx(1.0)


def test_cap_bloc_noop_when_under_cap():
    w = cap_bloc(np.array([0.2, 0.1, 0.7]), np.array([True, True, False]), cap=0.5)
    assert w == pytest.approx([0.2, 0.1, 0.7])


def test_hedge_advisory_holds_unhedged_when_carry_costly():
    # over the cap, but USD rates > EUR (positive diff) => hedging costs carry => stay unhedged
    a = hedge_advisory(bloc_exposure_pct=0.55, rate_diff_pct=1.5, cap=0.4)
    assert a["over_cap"] is True
    assert a["hedge_recommended"] is False


def test_hedge_advisory_recommends_when_over_and_carry_favorable():
    a = hedge_advisory(bloc_exposure_pct=0.55, rate_diff_pct=-0.5, cap=0.4)
    assert a["hedge_recommended"] is True
