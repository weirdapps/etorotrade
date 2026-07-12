"""TDD tests for trade_modules.v3.features.enrich_features.

All tests inject FAKE info_fetch + price_fetch and a tmp CSV — no network.
"""

import numpy as np
import pandas as pd

from trade_modules.v3.features import enrich_features, trade_levels

# The 31-column etoro/portfolio/buy schema.
FULL_COLS = [
    "TKR",
    "NAME",
    "CAP",
    "PRC",
    "TGT",
    "UP%",
    "#T",
    "%B",
    "#A",
    "AM",
    "A",
    "EXR",
    "B",
    "52W",
    "2H",
    "PET",
    "PEF",
    "P/S",
    "PEG",
    "DV",
    "SI",
    "EG",
    "PP",
    "ROE",
    "DE",
    "FCF",
    "ERN",
    "SZ",
    "BS",
    "SIGNAL_TRACK",
    "SIGNAL_HORIZON",
]


def _write_csv(tmp_path):
    rows = [
        {
            "TKR": "AAPL",
            "NAME": "APPLE INC",
            "CAP": "3.5T",
            "PRC": "200.00",
            "TGT": "240.00",
            "UP%": "20.0%",
            "#T": "30",
            "%B": "78%",
            "#A": "30",
            "AM": "5",
            "A": "A",
            "EXR": "18.8%",
            "B": "1.2",
            "52W": "95",
            "2H": "Y",
            "PET": "28.5",
            "PEF": "25.0",
            "P/S": "7.0",
            "PEG": "2.1",
            "DV": "0.5%",
            "SI": "1.1",
            "EG": "12.0",
            "PP": "15.3",
            "ROE": "45.0",
            "DE": "120.0",
            "FCF": "3.9%",
            "ERN": "07/31",
            "SZ": "50M",
            "BS": "B",
            "SIGNAL_TRACK": "momentum",
            "SIGNAL_HORIZON": "30",
        },
        {
            "TKR": "MSFT",
            "NAME": "MICROSOFT",
            "CAP": "3.0T",
            "PRC": "400.00",
            "TGT": "460.00",
            "UP%": "15.0%",
            "#T": "32",
            "%B": "80%",
            "#A": "32",
            "AM": "3",
            "A": "A",
            "EXR": "12.0%",
            "B": "0.9",
            "52W": "88",
            "2H": "N",
            "PET": "35.0",
            "PEF": "30.0",
            "P/S": "12.0",
            "PEG": "2.77",
            "DV": "2.77%",
            "SI": "0.8",
            "EG": "14.0",
            "PP": "22.0",
            "ROE": "38.0",
            "DE": "50.0",
            "FCF": "2.5%",
            "ERN": "07/25",
            "SZ": "40M",
            "BS": "B",
            "SIGNAL_TRACK": "value",
            "SIGNAL_HORIZON": "90",
        },
        {
            # ZZZ intentionally absent from the fake .info dict.
            "TKR": "ZZZ",
            "NAME": "ZED CORP",
            "CAP": "5B",
            "PRC": "10.00",
            "TGT": "12.00",
            "UP%": "20.0%",
            "#T": "5",
            "%B": "60%",
            "#A": "5",
            "AM": "-2",
            "A": "C",
            "EXR": "--",
            "B": "1.5",
            "52W": "40",
            "2H": "N",
            "PET": "8.0",
            "PEF": "7.0",
            "P/S": "0.5",
            "PEG": "0.4",
            "DV": "--",
            "SI": "5.0",
            "EG": "--",
            "PP": "-3.0",
            "ROE": "6.0",
            "DE": "200.0",
            "FCF": "--",
            "ERN": "08/01",
            "SZ": "1M",
            "BS": "H",
            "SIGNAL_TRACK": "value",
            "SIGNAL_HORIZON": "90",
        },
    ]
    df = pd.DataFrame(rows)[FULL_COLS]
    p = tmp_path / "etoro.csv"
    df.to_csv(p, index=False)
    return str(p)


def _fake_info(tickers):
    data = {
        "AAPL": {
            "priceToBook": 50.0,
            "enterpriseToEbitda": 22.0,
            "returnOnAssets": 0.28,
            "grossMargins": 0.44,
            "operatingMargins": 0.30,
            "currentRatio": 1.05,
            "targetHighPrice": 250.0,
            "targetLowPrice": 150.0,
            "averageVolume": 1_000_000,
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "quoteType": "EQUITY",
        },
        "MSFT": {
            "priceToBook": 12.0,
            "enterpriseToEbitda": 25.0,
            "returnOnAssets": 0.19,
            "grossMargins": 0.69,
            "operatingMargins": 0.42,
            "currentRatio": 1.3,
            "targetHighPrice": 500.0,
            "targetLowPrice": 420.0,
            "averageVolume": 800_000,
            "sector": "Technology",
            "industry": "Software",
            "quoteType": "EQUITY",
        },
    }
    return {t: data[t] for t in tickers if t in data}


def _fake_prices(tickers, period="2y", **_kw):
    """AAPL gets 300 bars (enough for 12-1 mom + vol); others get 100 (too short)."""
    idx_long = pd.date_range("2023-01-02", periods=300, freq="B")
    idx_short = idx_long[-100:]
    rng = np.random.default_rng(7)
    cols = {}
    for t in tickers:
        if t == "AAPL":
            cols[t] = pd.Series(100 + np.cumsum(rng.normal(0.1, 1.0, 300)), index=idx_long)
        else:
            cols[t] = pd.Series(50 + np.cumsum(rng.normal(0.05, 0.8, 100)), index=idx_short)
    return pd.DataFrame(cols)


def _run(tmp_path):
    return enrich_features(
        ["AAPL", "MSFT", "ZZZ"],
        _write_csv(tmp_path),
        info_fetch=_fake_info,
        price_fetch=_fake_prices,
    )


def test_frame_indexed_by_ticker_with_all_tickers(tmp_path):
    feats = _run(tmp_path)
    assert feats.index.name == "ticker"
    assert set(feats.index) == {"AAPL", "MSFT", "ZZZ"}


def test_expected_columns_present(tmp_path):
    feats = _run(tmp_path)
    expected = {
        # native
        "name",
        "cap",
        "price",
        "pe_trailing",
        "pe_forward",
        "ps_sector",
        "peg",
        "roe",
        "de",
        "fcf",
        "beta",
        "pct_52w_high",
        "price_perf",
        "short_interest",
        "analyst_mom",
        "earn_growth",
        "div_yield",
        "upside",
        "buy_pct",
        # added
        "pb",
        "ev_ebitda",
        "roa",
        "gross_margin",
        "op_margin",
        "current_ratio",
        "target_high",
        "target_low",
        "avg_volume",
        "sector",
        "industry",
        # derived
        "target_dispersion",
        "adv_usd",
        "mom_12_1",
        "realized_vol",
    }
    assert expected.issubset(set(feats.columns))


def test_percent_and_numeric_coercion(tmp_path):
    feats = _run(tmp_path)
    # "2.77%" -> 2.77
    assert feats.loc["MSFT", "div_yield"] == 2.77
    # "20.0%" -> 20.0 ; "78%" -> 78.0 ; "3.9%" -> 3.9
    assert feats.loc["AAPL", "upside"] == 20.0
    assert feats.loc["AAPL", "buy_pct"] == 78.0
    assert feats.loc["AAPL", "fcf"] == 3.9
    # cap suffix parse
    assert feats.loc["AAPL", "cap"] == 3.5e12
    # plain numerics coerced to float
    assert feats.loc["AAPL", "pe_trailing"] == 28.5
    assert pd.api.types.is_float_dtype(feats["pe_trailing"])
    # "--" native -> NaN
    assert pd.isna(feats.loc["ZZZ", "div_yield"])


def test_target_dispersion_math(tmp_path):
    feats = _run(tmp_path)
    # (250 - 150) / 200 = 0.5
    assert feats.loc["AAPL", "target_dispersion"] == 0.5
    # (500 - 420) / 400 = 0.2
    assert abs(feats.loc["MSFT", "target_dispersion"] - 0.2) < 1e-9


def test_adv_usd_math(tmp_path):
    feats = _run(tmp_path)
    # 1_000_000 * 200 = 2e8
    assert feats.loc["AAPL", "adv_usd"] == 2e8
    assert feats.loc["MSFT", "adv_usd"] == 800_000 * 400.0


def test_ticker_missing_from_info_still_appears(tmp_path):
    feats = _run(tmp_path)
    # ZZZ is not in .info: native fields present, added fields NaN.
    assert feats.loc["ZZZ", "pe_trailing"] == 8.0  # native still there
    assert feats.loc["ZZZ", "name"] == "ZED CORP"
    assert pd.isna(feats.loc["ZZZ", "pb"])  # added -> NaN
    assert pd.isna(feats.loc["ZZZ", "roa"])
    assert pd.isna(feats.loc["ZZZ", "sector"])
    # derived that depend on .info also NaN
    assert pd.isna(feats.loc["ZZZ", "target_dispersion"])
    assert pd.isna(feats.loc["ZZZ", "adv_usd"])


def test_price_factors_last_bar(tmp_path):
    feats = _run(tmp_path)
    # AAPL has 300 bars -> mom_12_1 and realized_vol computed.
    assert np.isfinite(feats.loc["AAPL", "mom_12_1"])
    assert np.isfinite(feats.loc["AAPL", "realized_vol"])
    assert feats.loc["AAPL", "realized_vol"] > 0
    # MSFT/ZZZ have only 100 bars -> too short -> NaN.
    assert pd.isna(feats.loc["MSFT", "mom_12_1"])
    assert pd.isna(feats.loc["ZZZ", "realized_vol"])


def test_quote_type_from_info(tmp_path):
    feats = _run(tmp_path)
    assert feats.loc["AAPL", "quote_type"] == "EQUITY"
    assert feats.loc["MSFT", "quote_type"] == "EQUITY"
    # ZZZ is absent from the fake .info -> quote_type NaN.
    assert pd.isna(feats.loc["ZZZ", "quote_type"])


def test_trade_levels_known_values():
    # realized_vol chosen so sigma_m = realized_vol/sqrt(12) = 0.10 exactly.
    lv = trade_levels(100.0, (12**0.5) * 0.1)
    assert abs(lv["sigma_m"] - 0.10) < 1e-9
    assert abs(lv["entry"] - 100.0) < 1e-9
    assert abs(lv["stop_loss"] - 80.0) < 1e-9  # 100 * (1 - 2*0.10)
    assert abs(lv["take_profit"] - 130.0) < 1e-9  # 100 * (1 + 3*0.10)
    assert abs(lv["rr"] - 1.5) < 1e-9  # (130-100)/(100-80)


def test_trade_levels_degenerate():
    # Missing vol -> entry still echoes price, but stop/target/rr NaN.
    miss = trade_levels(100.0, float("nan"))
    assert miss["entry"] == 100.0
    for k in ("sigma_m", "stop_loss", "take_profit", "rr"):
        assert pd.isna(miss[k])
    # Zero vol is degenerate too.
    assert pd.isna(trade_levels(100.0, 0.0)["stop_loss"])
    # Missing / non-positive price -> everything NaN (no valid entry).
    for bad in (float("nan"), 0.0, -5.0):
        lv = trade_levels(bad, 0.30)
        assert all(pd.isna(lv[k]) for k in ("entry", "stop_loss", "take_profit", "rr"))


def test_enrich_adds_trade_level_columns(tmp_path):
    feats = _run(tmp_path)
    for col in ("entry", "sigma_m", "stop_loss", "take_profit", "rr"):
        assert col in feats.columns
    # AAPL has a realized_vol -> full levels, entry == price, rr ~ 1.5.
    assert feats.loc["AAPL", "entry"] == feats.loc["AAPL", "price"]
    assert np.isfinite(feats.loc["AAPL", "stop_loss"])
    assert np.isfinite(feats.loc["AAPL", "take_profit"])
    assert feats.loc["AAPL", "stop_loss"] < feats.loc["AAPL", "price"]
    assert feats.loc["AAPL", "take_profit"] > feats.loc["AAPL", "price"]
    assert abs(feats.loc["AAPL", "rr"] - 1.5) < 1e-9
    # MSFT has NaN realized_vol -> entry echoes price, but stop/target NaN.
    assert feats.loc["MSFT", "entry"] == feats.loc["MSFT", "price"]
    assert pd.isna(feats.loc["MSFT", "stop_loss"])
    assert pd.isna(feats.loc["MSFT", "rr"])
