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


_AAPL_SUMMARY = (
    "Apple Inc. designs, manufactures, and markets smartphones, personal computers, "
    "tablets, wearables, and accessories worldwide. The company operates through "
    "several product and service segments including iPhone, Mac, iPad, and Services."
)


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
            "country": "United States",
            "quoteType": "EQUITY",
            "longBusinessSummary": _AAPL_SUMMARY,
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
            "country": "United States",
            "quoteType": "EQUITY",
            "longBusinessSummary": "Microsoft develops software and cloud services.",
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
        accruals_fetch=lambda tickers: {},  # no-op: no network in unit tests
    )


def test_enrich_honors_injected_sector_map_for_uncovered_names(tmp_path):
    """BUILD ③: an injected offline map fills sectors yfinance did not provide."""
    feats = enrich_features(
        ["AAPL", "MSFT", "ZZZ"],
        _write_csv(tmp_path),
        info_fetch=_fake_info,
        price_fetch=_fake_prices,
        accruals_fetch=lambda tickers: {},
        sector_map={"ZZZ": "Energy"},  # ZZZ has no live sector
    )
    assert feats.loc["ZZZ", "sector"] == "Energy"
    assert feats.loc["AAPL", "sector"] == "Technology"  # not in map -> live fallback


def test_enrich_offline_map_overrides_live_sector(tmp_path):
    """BUILD ③: static/offline sector outranks live yfinance (higher trust)."""
    feats = enrich_features(
        ["AAPL", "MSFT", "ZZZ"],
        _write_csv(tmp_path),
        info_fetch=_fake_info,
        price_fetch=_fake_prices,
        accruals_fetch=lambda tickers: {},
        sector_map={"AAPL": "Consumer Cyclical"},
    )
    assert feats.loc["AAPL", "sector"] == "Consumer Cyclical"


def test_enrich_without_sector_map_is_unchanged(tmp_path):
    """BUILD ③ back-compat: no map -> exactly today's live-only behavior."""
    feats = enrich_features(
        ["AAPL", "MSFT", "ZZZ"],
        _write_csv(tmp_path),
        info_fetch=_fake_info,
        price_fetch=_fake_prices,
        accruals_fetch=lambda tickers: {},
    )
    assert feats.loc["AAPL", "sector"] == "Technology"
    assert pd.isna(feats.loc["ZZZ", "sector"])


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
        "country",
        # derived
        "target_dispersion",
        "earn_trajectory",
        "adv_usd",
        "mom_12_1",
        "realized_vol",
        # earnings quality
        "accruals",
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


def test_earn_trajectory_math(tmp_path):
    """earn_trajectory = trailing/forward P/E (>1 = forward cheaper = earnings rising)."""
    feats = _run(tmp_path)
    assert abs(feats.loc["AAPL", "earn_trajectory"] - 28.5 / 25.0) < 1e-9
    assert abs(feats.loc["MSFT", "earn_trajectory"] - 35.0 / 30.0) < 1e-9


def test_target_dispersion_math(tmp_path):
    feats = _run(tmp_path)
    # (250 - 150) / 200 = 0.5
    assert feats.loc["AAPL", "target_dispersion"] == 0.5
    # (500 - 420) / 400 = 0.2
    assert abs(feats.loc["MSFT", "target_dispersion"] - 0.2) < 1e-9


def test_adv_usd_math(tmp_path):
    feats = _run(tmp_path)
    # 1_000_000 * 200 = 2e8 (AAPL is a US listing -> FX rate 1.0, no conversion)
    assert feats.loc["AAPL", "adv_usd"] == 2e8
    assert feats.loc["MSFT", "adv_usd"] == 800_000 * 400.0


def test_usd_rate_normalizes_local_currency_cap_and_adv():
    """cap + adv_usd must convert local currency to USD via _usd_rate_for."""
    from trade_modules.v3.features import _usd_rate_for

    assert _usd_rate_for("AAPL") == 1.0  # US listing -> no conversion
    assert _usd_rate_for("UNKNOWN.ZZ") == 1.0  # unmapped suffix -> USD (no-op)
    assert _usd_rate_for("7203.T") < 0.02  # yen cap must shrink hard to USD
    assert _usd_rate_for("SAP.DE") > 1.0  # euro is worth more than a dollar


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


def test_country_from_info(tmp_path):
    feats = _run(tmp_path)
    # country is enriched from yfinance .info "country" (needed for dual-listing dedup).
    assert feats.loc["AAPL", "country"] == "United States"
    assert feats.loc["MSFT", "country"] == "United States"
    # ZZZ is absent from the fake .info -> country NaN.
    assert pd.isna(feats.loc["ZZZ", "country"])


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


# ---------------------------------------------------------------------------
# Sloan accruals (earnings-quality)
# ---------------------------------------------------------------------------


def test_accruals_formula_math(tmp_path):
    """inject a fake accruals_fetch with known (ni-cfo)/avg_assets values."""
    # ni=100, cfo=60, avg_assets=500  → (100-60)/500 = 0.08  (AAPL)
    # ni=50,  cfo=80, avg_assets=1000 → (50-80)/1000 = -0.03 (MSFT)
    expected = {"AAPL": 0.08, "MSFT": -0.03}
    feats = enrich_features(
        ["AAPL", "MSFT", "ZZZ"],
        _write_csv(tmp_path),
        info_fetch=_fake_info,
        price_fetch=_fake_prices,
        accruals_fetch=lambda tickers: {t: expected[t] for t in tickers if t in expected},
    )
    assert "accruals" in feats.columns
    assert abs(feats.loc["AAPL", "accruals"] - 0.08) < 1e-12
    assert abs(feats.loc["MSFT", "accruals"] - (-0.03)) < 1e-12
    # ZZZ absent from fake fetcher dict -> NaN
    assert pd.isna(feats.loc["ZZZ", "accruals"])


def test_accruals_noop_fetcher_yields_all_nan(tmp_path):
    """A no-op accruals_fetch (empty dict) produces NaN accruals for every ticker."""
    feats = enrich_features(
        ["AAPL", "MSFT"],
        _write_csv(tmp_path),
        info_fetch=_fake_info,
        price_fetch=_fake_prices,
        accruals_fetch=lambda tickers: {},
    )
    assert "accruals" in feats.columns
    assert feats["accruals"].isna().all()


def test_accruals_partial_coverage(tmp_path):
    """Tickers present in the fetcher dict get values; absent ones get NaN."""
    feats = enrich_features(
        ["AAPL", "MSFT", "ZZZ"],
        _write_csv(tmp_path),
        info_fetch=_fake_info,
        price_fetch=_fake_prices,
        accruals_fetch=lambda tickers: {"AAPL": -0.05},
    )
    assert np.isfinite(feats.loc["AAPL", "accruals"])
    assert pd.isna(feats.loc["MSFT", "accruals"])
    assert pd.isna(feats.loc["ZZZ", "accruals"])


# ---------------------------------------------------------------------------
# Business description (longBusinessSummary → description)
# ---------------------------------------------------------------------------


def test_description_column_present(tmp_path):
    """enrich_features always produces a 'description' column."""
    feats = _run(tmp_path)
    assert "description" in feats.columns


def test_description_extracted_from_info(tmp_path):
    """Tickers with longBusinessSummary get a non-empty description ≤ 220 chars."""
    feats = _run(tmp_path)
    assert feats.loc["AAPL", "description"] != ""
    assert len(feats.loc["AAPL", "description"]) <= 220
    assert feats.loc["MSFT", "description"] != ""


def test_description_empty_when_info_missing(tmp_path):
    """Tickers absent from .info (ZZZ) get an empty-string description, not NaN."""
    feats = _run(tmp_path)
    assert feats.loc["ZZZ", "description"] == ""
    assert not pd.isna(feats.loc["ZZZ", "description"])


def test_description_truncation_at_sentence_boundary(tmp_path):
    """A long summary is cut at the last sentence boundary within 220 chars."""
    first_sentence = "X" * 98 + "."  # 99 chars, ends with period at index 98
    rest = " " + "Y" * 200 + "."  # pushes total well past 220
    long_summary = first_sentence + rest  # ~300 chars total

    def _info_long(tickers):
        return {"AAPL": {"longBusinessSummary": long_summary}}

    feats = enrich_features(
        ["AAPL"],
        _write_csv(tmp_path),
        info_fetch=_info_long,
        price_fetch=_fake_prices,
        accruals_fetch=lambda tickers: {},
    )
    desc = feats.loc["AAPL", "description"]
    assert desc.endswith(".")
    assert len(desc) <= 220
    # The cut should happen at the first-sentence boundary (~99 chars), not in the middle
    assert len(desc) < 150


def test_description_word_boundary_fallback(tmp_path):
    """When no sentence boundary within 220 chars, cut at a word boundary."""
    no_period = "word " * 60  # 300 chars, no period

    def _info_no_period(tickers):
        return {"AAPL": {"longBusinessSummary": no_period}}

    feats = enrich_features(
        ["AAPL"],
        _write_csv(tmp_path),
        info_fetch=_info_no_period,
        price_fetch=_fake_prices,
        accruals_fetch=lambda tickers: {},
    )
    desc = feats.loc["AAPL", "description"]
    assert len(desc) <= 222  # 220 chars + possible "…" (1 char)
    # Must not end with a raw space (no trailing whitespace after trim)
    assert not desc.rstrip("…").endswith(" ")
