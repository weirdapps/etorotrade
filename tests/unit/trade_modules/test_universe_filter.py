"""Unit tests for trade_modules.universe.filter (S1 — Universe & Quality Filter).

All tests are pure: no I/O, no network, no monkey-patching of external systems.
"""

import pandas as pd

from trade_modules.universe.filter import filter_universe, route_asset

# ---------------------------------------------------------------------------
# Fixtures — canonical rows
# ---------------------------------------------------------------------------


def _equity_row(**overrides):
    """A passing fundamental equity row (AAPL-like)."""
    base = {
        "TKR": "AAPL",
        "NAME": "Apple Inc",
        "CAP": "2.9T",
        "PRC": "195.5",
        "#A": "42",
        "PET": "28.0",
        "PEF": "25.0",
        "FCF": "5.2%",
        "52W": "72",
        "BS": "B",
    }
    base.update(overrides)
    return base


def _row(**overrides):
    return _equity_row(**overrides)


# ---------------------------------------------------------------------------
# route_asset — crypto
# ---------------------------------------------------------------------------


class TestRouteAssetCrypto:
    def test_btc_usd_price_only(self):
        row = _row(TKR="BTC-USD", NAME="Bitcoin", PET="--", PEF="--", FCF="--", **{"#A": "--"})
        assert route_asset(row) == "price_only"

    def test_eth_usd_price_only(self):
        row = _row(TKR="ETH-USD", NAME="Ethereum", PET="--", PEF="--", FCF="--", **{"#A": "--"})
        assert route_asset(row) == "price_only"

    def test_arbitrary_usd_suffix(self):
        row = _row(TKR="SOL-USD", NAME="Solana", PET="", PEF="", FCF="", **{"#A": ""})
        assert route_asset(row) == "price_only"


# ---------------------------------------------------------------------------
# route_asset — leveraged/inverse by name
# ---------------------------------------------------------------------------


class TestRouteAssetLeveraged:
    def test_ultra_in_name(self):
        row = _row(TKR="UPRO", NAME="ProShares UltraPro S&P500", **{"#A": "--"})
        assert route_asset(row) == "price_only"

    def test_inverse_in_name(self):
        row = _row(TKR="SH", NAME="ProShares Inverse S&P500", **{"#A": "--"})
        assert route_asset(row) == "price_only"

    def test_2x_in_name(self):
        row = _row(TKR="SSO", NAME="ProShares 2X S&P500", **{"#A": "--"})
        assert route_asset(row) == "price_only"

    def test_3x_in_name(self):
        row = _row(TKR="TQQQ", NAME="ProShares 3X QQQ", **{"#A": "--"})
        assert route_asset(row) == "price_only"

    def test_leveraged_in_name(self):
        row = _row(TKR="FNGU", NAME="MicroSectors Leveraged FANG ETN", **{"#A": "--"})
        assert route_asset(row) == "price_only"

    def test_case_insensitive_ultra(self):
        row = _row(TKR="X", NAME="ultra something fund", **{"#A": "--"})
        assert route_asset(row) == "price_only"


# ---------------------------------------------------------------------------
# route_asset — known price-only tickers
# ---------------------------------------------------------------------------


class TestRouteAssetKnownPriceOnly:
    def test_uvxy(self):
        row = _row(TKR="UVXY", NAME="ProShares Ultra VIX Short-Term Futures ETF", **{"#A": "--"})
        assert route_asset(row) == "price_only"

    def test_gld(self):
        row = _row(TKR="GLD", NAME="SPDR Gold Shares", **{"#A": "--"})
        assert route_asset(row) == "price_only"

    def test_lyxgre(self):
        row = _row(TKR="LYXGRE.DE", NAME="Lyxor Greece ETF", **{"#A": "--"})
        assert route_asset(row) == "price_only"


# ---------------------------------------------------------------------------
# route_asset — plain ETF (no fundamentals)
# ---------------------------------------------------------------------------


class TestRouteAssetPlainETF:
    def test_etf_no_fundamentals_price_only(self):
        """Row with price but ALL fundamentals blank → price_only."""
        row = {
            "TKR": "XLK",
            "NAME": "Technology Select Sector SPDR",
            "PRC": "180.0",
            "CAP": "50B",
            "#A": "--",
            "PET": "--",
            "PEF": "",
            "FCF": None,
            "52W": "65",
        }
        assert route_asset(row) == "price_only"

    def test_equity_with_analyst_count_is_fundamental(self):
        """Row with #A present → fundamental even if PE missing."""
        row = _row(PET="--", PEF="--", FCF="--")
        assert route_asset(row) == "fundamental"


# ---------------------------------------------------------------------------
# route_asset — excluded (no usable price)
# ---------------------------------------------------------------------------


class TestRouteAssetExcluded:
    def test_prc_missing(self):
        row = _row(PRC=None)
        assert route_asset(row) == "excluded"

    def test_prc_blank(self):
        row = _row(PRC="")
        assert route_asset(row) == "excluded"

    def test_prc_dash(self):
        row = _row(PRC="--")
        assert route_asset(row) == "excluded"

    def test_prc_zero(self):
        row = _row(PRC="0")
        assert route_asset(row) == "excluded"

    def test_prc_negative(self):
        row = _row(PRC="-5.0")
        assert route_asset(row) == "excluded"

    def test_prc_non_numeric(self):
        row = _row(PRC="N/A")
        assert route_asset(row) == "excluded"


# ---------------------------------------------------------------------------
# route_asset — fundamental equity
# ---------------------------------------------------------------------------


class TestRouteAssetFundamental:
    def test_aapl_like_row(self):
        row = _row()
        assert route_asset(row) == "fundamental"

    def test_equity_with_only_pet(self):
        row = _row(PEF="--", FCF="--")
        assert route_asset(row) == "fundamental"

    def test_european_ticker(self):
        row = _row(TKR="ASML.AS", NAME="ASML Holding")
        assert route_asset(row) == "fundamental"


# ---------------------------------------------------------------------------
# filter_universe — helpers
# ---------------------------------------------------------------------------


def _make_df(*rows):
    return pd.DataFrame(list(rows))


def _passing_row(**overrides):
    """Row that passes all default gates."""
    base = {
        "TKR": "AAPL",
        "NAME": "Apple Inc",
        "CAP": "2.9T",
        "PRC": "195.5",
        "#A": "42",
        "PET": "28.0",
        "PEF": "25.0",
        "FCF": "5.2%",
        "52W": "72",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# filter_universe — eligible / price_only / excluded routing
# ---------------------------------------------------------------------------


class TestFilterUniverseRouting:
    def test_passing_equity_in_eligible(self):
        df = _make_df(_passing_row())
        result = filter_universe(df)
        assert "AAPL" in result["eligible"]["TKR"].values

    def test_crypto_goes_to_price_only(self):
        df = _make_df(
            _passing_row(),
            {
                "TKR": "BTC-USD",
                "NAME": "Bitcoin",
                "PRC": "65000",
                "CAP": "1.2T",
                "#A": "--",
                "PET": "--",
                "PEF": "--",
                "FCF": "--",
                "52W": "60",
            },
        )
        result = filter_universe(df)
        assert "BTC-USD" in result["price_only"]
        assert "BTC-USD" not in result["eligible"]["TKR"].values

    def test_no_price_goes_to_excluded(self):
        df = _make_df(_passing_row(TKR="NOPRC", PRC=None))
        result = filter_universe(df)
        assert "NOPRC" in result["excluded"]

    def test_summary_counts_consistent(self):
        df = _make_df(
            _passing_row(TKR="E1"),
            _passing_row(TKR="E2", PRC=None),
            {
                "TKR": "BTC-USD",
                "NAME": "Bitcoin",
                "PRC": "1000",
                "CAP": "1T",
                "#A": "",
                "PET": "",
                "PEF": "",
                "FCF": "",
                "52W": "",
            },
        )
        r = filter_universe(df)
        assert r["summary"]["total"] == 3
        assert r["summary"]["excluded"] == 1
        assert r["summary"]["price_only"] == 1


# ---------------------------------------------------------------------------
# filter_universe — min_cap gate
# ---------------------------------------------------------------------------


class TestFilterUniverseMinCap:
    def test_below_min_cap_fails(self):
        df = _make_df(_passing_row(TKR="TINY", CAP="400M"))
        result = filter_universe(df)
        assert "TINY" not in result["eligible"]["TKR"].values
        reasons = result["reasons"]["TINY"]
        assert any("cap" in r.lower() for r in reasons)

    def test_above_min_cap_passes(self):
        df = _make_df(_passing_row(TKR="BIG", CAP="5B"))
        result = filter_universe(df)
        assert "BIG" in result["eligible"]["TKR"].values

    def test_config_override_min_cap(self):
        """Raising the floor should drop a name that was previously eligible."""
        df = _make_df(_passing_row(TKR="MID", CAP="3B"))
        r_default = filter_universe(df)
        r_strict = filter_universe(df, config={"min_cap": 5e9})
        assert "MID" in r_default["eligible"]["TKR"].values
        assert "MID" not in r_strict["eligible"]["TKR"].values

    def test_blank_cap_fails(self):
        df = _make_df(_passing_row(TKR="NOCAP", CAP="--"))
        result = filter_universe(df)
        assert "NOCAP" not in result["eligible"]["TKR"].values


# ---------------------------------------------------------------------------
# filter_universe — min_analysts gate (fail-closed)
# ---------------------------------------------------------------------------


class TestFilterUniverseMinAnalysts:
    def test_too_few_analysts_fails(self):
        df = _make_df(_passing_row(TKR="FEW", **{"#A": "3"}))
        result = filter_universe(df)
        assert "FEW" not in result["eligible"]["TKR"].values
        reasons = result["reasons"]["FEW"]
        assert any("analysts 3 < 5" in r for r in reasons)

    def test_exact_threshold_passes(self):
        df = _make_df(_passing_row(TKR="EXACT", **{"#A": "5"}))
        result = filter_universe(df)
        assert "EXACT" in result["eligible"]["TKR"].values

    def test_missing_analysts_fail_closed(self):
        df = _make_df(_passing_row(TKR="NOANA", **{"#A": None}))
        result = filter_universe(df)
        assert "NOANA" not in result["eligible"]["TKR"].values
        reasons = result["reasons"]["NOANA"]
        assert any("unknown" in r.lower() or "missing" in r.lower() for r in reasons)

    def test_dash_analysts_fail_closed(self):
        df = _make_df(_passing_row(TKR="DASHANA", **{"#A": "--"}))
        result = filter_universe(df)
        assert "DASHANA" not in result["eligible"]["TKR"].values

    def test_config_override_min_analysts(self):
        df = _make_df(_passing_row(TKR="FEW2", **{"#A": "3"}))
        r_strict = filter_universe(df)
        r_relaxed = filter_universe(df, config={"min_analysts": 3})
        assert "FEW2" not in r_strict["eligible"]["TKR"].values
        assert "FEW2" in r_relaxed["eligible"]["TKR"].values


# ---------------------------------------------------------------------------
# filter_universe — positive_earnings gate (fail-closed when all missing)
# ---------------------------------------------------------------------------


class TestFilterUniversePositiveEarnings:
    def test_all_negative_earnings_fails(self):
        df = _make_df(_passing_row(TKR="LOSS", PET="-5.0", PEF="-3.0", FCF="-1.0%"))
        result = filter_universe(df)
        assert "LOSS" not in result["eligible"]["TKR"].values
        reasons = result["reasons"]["LOSS"]
        assert any("negative earnings" in r.lower() for r in reasons)

    def test_positive_pet_passes(self):
        df = _make_df(_passing_row(TKR="PETOK", PET="20.0", PEF="-5.0", FCF="-2.0%"))
        result = filter_universe(df)
        assert "PETOK" in result["eligible"]["TKR"].values

    def test_positive_fcf_passes(self):
        df = _make_df(_passing_row(TKR="FCFOK", PET="-5.0", PEF="-5.0", FCF="3.0%"))
        result = filter_universe(df)
        assert "FCFOK" in result["eligible"]["TKR"].values

    def test_all_earnings_missing_fail_closed(self):
        df = _make_df(_passing_row(TKR="NOMISSING", PET="--", PEF=None, FCF=""))
        result = filter_universe(df)
        assert "NOMISSING" not in result["eligible"]["TKR"].values
        reasons = result["reasons"]["NOMISSING"]
        assert any("missing" in r.lower() for r in reasons)

    def test_require_positive_earnings_false_skips_gate(self):
        df = _make_df(_passing_row(TKR="NOGATE", PET="-5.0", PEF="-3.0", FCF="-1.0%"))
        result = filter_universe(df, config={"require_positive_earnings": False})
        assert "NOGATE" in result["eligible"]["TKR"].values


# ---------------------------------------------------------------------------
# filter_universe — trend / 52W gate
# ---------------------------------------------------------------------------


class TestFilterUniverseTrend:
    def test_melting_below_floor_fails(self):
        df = _make_df(_passing_row(TKR="MELT", **{"52W": "18"}))
        result = filter_universe(df)
        assert "MELT" not in result["eligible"]["TKR"].values
        reasons = result["reasons"]["MELT"]
        assert any("melting" in r.lower() and "18" in r for r in reasons)

    def test_exactly_at_floor_passes(self):
        df = _make_df(_passing_row(TKR="ATFLOOR", **{"52W": "30"}))
        result = filter_universe(df)
        assert "ATFLOOR" in result["eligible"]["TKR"].values

    def test_missing_52w_does_not_fail(self):
        """Missing 52W → gate is left open (we don't penalise absence of trend data)."""
        df = _make_df(_passing_row(TKR="NO52W", **{"52W": "--"}))
        result = filter_universe(df)
        assert "NO52W" in result["eligible"]["TKR"].values

    def test_config_override_min_52w(self):
        df = _make_df(_passing_row(TKR="LOW52", **{"52W": "35"}))
        r_strict = filter_universe(df, config={"min_52w": 50})
        r_relaxed = filter_universe(df, config={"min_52w": 30})
        assert "LOW52" not in r_strict["eligible"]["TKR"].values
        assert "LOW52" in r_relaxed["eligible"]["TKR"].values


# ---------------------------------------------------------------------------
# filter_universe — reasons / human-readable messages
# ---------------------------------------------------------------------------


class TestFilterUniverseReasons:
    def test_passing_row_has_empty_reasons(self):
        df = _make_df(_passing_row())
        result = filter_universe(df)
        assert result["reasons"]["AAPL"] == []

    def test_multiple_failures_all_listed(self):
        """A row that fails both min_cap and min_analysts should list both reasons."""
        df = _make_df(_passing_row(TKR="MULTI", CAP="400M", **{"#A": "2"}))
        result = filter_universe(df)
        reasons = result["reasons"]["MULTI"]
        assert any("cap" in r.lower() for r in reasons)
        assert any("analysts 2 < 5" in r for r in reasons)


# ---------------------------------------------------------------------------
# filter_universe — input not mutated
# ---------------------------------------------------------------------------


class TestFilterUniverseImmutability:
    def test_input_df_not_mutated(self):
        df = _make_df(_passing_row())
        original_cols = list(df.columns)
        original_values = df.copy()
        filter_universe(df)
        assert list(df.columns) == original_cols
        pd.testing.assert_frame_equal(df, original_values)


# ---------------------------------------------------------------------------
# filter_universe — dash / blank / None robustness
# ---------------------------------------------------------------------------


class TestFilterUniverseEdgeCases:
    def test_all_dash_fields_fundamental_with_analyst(self):
        """Row with #A present but PET/PEF/FCF all dashes — routed fundamental,
        fails positive_earnings gate (all missing → fail-closed)."""
        row = {
            "TKR": "DASHROW",
            "NAME": "Dash Company",
            "CAP": "10B",
            "PRC": "50.0",
            "#A": "8",
            "PET": "--",
            "PEF": "--",
            "FCF": "--",
            "52W": "55",
        }
        df = _make_df(row)
        result = filter_universe(df)
        # routed to fundamental (has #A), fails earnings gate
        assert "DASHROW" in result["reasons"]
        assert "DASHROW" not in result["eligible"]["TKR"].values

    def test_empty_dataframe_safe(self):
        df = pd.DataFrame(columns=["TKR", "NAME", "CAP", "PRC", "#A", "PET", "PEF", "FCF", "52W"])
        result = filter_universe(df)
        assert len(result["eligible"]) == 0
        assert result["price_only"] == []
        assert result["excluded"] == {}

    def test_garbage_cap_fails_gracefully(self):
        df = _make_df(_passing_row(TKR="GCAP", CAP="xyz"))
        result = filter_universe(df)
        assert "GCAP" not in result["eligible"]["TKR"].values

    def test_summary_gate_failures_counted(self):
        rows = [_passing_row(TKR=f"F{i}", CAP="400M") for i in range(3)]
        df = _make_df(*rows)
        result = filter_universe(df)
        assert result["summary"]["gate_failures"]["min_cap"] == 3
