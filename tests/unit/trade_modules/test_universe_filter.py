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
    """True leveraged/inverse ETPs have no analyst coverage AND no earnings data."""

    def test_ultra_in_name(self):
        row = _row(
            TKR="UPRO",
            NAME="ProShares UltraPro S&P500",
            **{"#A": "--", "PET": "--", "PEF": "--", "FCF": "--"},
        )
        assert route_asset(row) == "price_only"

    def test_inverse_in_name(self):
        row = _row(
            TKR="SH",
            NAME="ProShares Inverse S&P500",
            **{"#A": "--", "PET": "--", "PEF": "--", "FCF": "--"},
        )
        assert route_asset(row) == "price_only"

    def test_2x_in_name(self):
        row = _row(
            TKR="SSO",
            NAME="ProShares 2X S&P500",
            **{"#A": "--", "PET": "--", "PEF": "--", "FCF": "--"},
        )
        assert route_asset(row) == "price_only"

    def test_3x_in_name(self):
        row = _row(
            TKR="TQQQ",
            NAME="ProShares 3X QQQ",
            **{"#A": "--", "PET": "--", "PEF": "--", "FCF": "--"},
        )
        assert route_asset(row) == "price_only"

    def test_leveraged_in_name(self):
        row = _row(
            TKR="FNGU",
            NAME="MicroSectors Leveraged FANG ETN",
            **{"#A": "--", "PET": "--", "PEF": "--", "FCF": "--"},
        )
        assert route_asset(row) == "price_only"

    def test_case_insensitive_ultra(self):
        row = _row(
            TKR="X",
            NAME="ultra something fund",
            **{"#A": "--", "PET": "--", "PEF": "--", "FCF": "--"},
        )
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

    def test_positive_pet_passes_when_not_clear_loss_maker(self):
        """PET positive, PEF negative but FCF positive → not a clear loss-maker → passes."""
        df = _make_df(_passing_row(TKR="PETOK", PET="20.0", PEF="-5.0", FCF="2.0%"))
        result = filter_universe(df)
        assert "PETOK" in result["eligible"]["TKR"].values

    def test_positive_pet_both_pef_and_fcf_negative_is_clear_loss_maker(self):
        """PET positive, PEF<0 AND FCF<0 → clear loss-maker → fails gate (Fix 3)."""
        df = _make_df(_passing_row(TKR="LOSERPET", PET="20.0", PEF="-5.0", FCF="-2.0%"))
        result = filter_universe(df)
        assert "LOSERPET" not in result["eligible"]["TKR"].values

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


# ---------------------------------------------------------------------------
# FIX 1 REGRESSION — name-heuristic must not misroute real equities
# ---------------------------------------------------------------------------


class TestRouteAssetLeveragedNameHeuristicGuard:
    """Real operating companies with 'ULTRA'/'2X'/'3X' in their name must NOT
    be routed to price_only when they have analyst coverage or earnings data."""

    def test_vvx_ultra_name_with_analysts_is_fundamental(self):
        """VVX — 'V2X, INC.' has 10 analysts and PET 26.6 → fundamental."""
        row = {
            "TKR": "VVX",
            "NAME": "V2X, INC.",
            "PRC": "30.0",
            "CAP": "3B",
            "#A": "10",
            "PET": "26.6",
            "PEF": "--",
            "FCF": "--",
            "52W": "55",
        }
        assert route_asset(row) == "fundamental"

    def test_uctt_ultra_name_with_analysts_is_fundamental(self):
        """UCTT — 'Ultra Clean Holdings' has 5 analysts → fundamental."""
        row = {
            "TKR": "UCTT",
            "NAME": "Ultra Clean Holdings Inc",
            "PRC": "25.0",
            "CAP": "2.5B",
            "#A": "5",
            "PET": "--",
            "PEF": "18.0",
            "FCF": "3.5%",
            "52W": "40",
        }
        assert route_asset(row) == "fundamental"

    def test_rare_ultra_name_with_analysts_is_fundamental(self):
        """RARE — 'Ultragenyx Pharmaceutical' has 20 analysts → fundamental."""
        row = {
            "TKR": "RARE",
            "NAME": "Ultragenyx Pharmaceutical Inc",
            "PRC": "50.0",
            "CAP": "3.5B",
            "#A": "20",
            "PET": "--",
            "PEF": "--",
            "FCF": "1.2%",
            "52W": "60",
        }
        assert route_asset(row) == "fundamental"

    def test_ugp_ultra_name_with_analysts_is_fundamental(self):
        """UGP — 'Ultrapar Participacoes' has 4 analysts → fundamental."""
        row = {
            "TKR": "UGP",
            "NAME": "Ultrapar Participacoes S.A.",
            "PRC": "5.0",
            "CAP": "2.2B",
            "#A": "4",
            "PET": "12.0",
            "PEF": "--",
            "FCF": "--",
            "52W": "48",
        }
        assert route_asset(row) == "fundamental"

    def test_true_leveraged_etp_no_fundamentals_is_price_only(self):
        """A genuine leveraged ETP with no analysts/earnings stays price_only."""
        row = {
            "TKR": "UPRO",
            "NAME": "ProShares UltraPro 3X S&P500",
            "PRC": "55.0",
            "CAP": "5B",
            "#A": "--",
            "PET": "--",
            "PEF": "--",
            "FCF": "--",
            "52W": "65",
        }
        assert route_asset(row) == "price_only"

    def test_2x_name_with_earnings_is_fundamental(self):
        """A real company whose name contains '2X' but has earnings → fundamental."""
        row = {
            "TKR": "VVX",
            "NAME": "V2X, INC.",
            "PRC": "30.0",
            "CAP": "3B",
            "#A": "--",
            "PET": "26.6",
            "PEF": "--",
            "FCF": "--",
            "52W": "55",
        }
        assert route_asset(row) == "fundamental"

    def test_2x_name_no_fundamentals_is_price_only(self):
        """'2X' in name with no analysts AND no earnings → price_only (true ETP)."""
        row = {
            "TKR": "SSO",
            "NAME": "ProShares 2X S&P500",
            "PRC": "80.0",
            "CAP": "4B",
            "#A": "--",
            "PET": "--",
            "PEF": "--",
            "FCF": "--",
            "52W": "70",
        }
        assert route_asset(row) == "price_only"


# ---------------------------------------------------------------------------
# FIX 2 REGRESSION — 52W domain: values >100 must be fail-open
# ---------------------------------------------------------------------------


class TestFilterUniverseTrendDomainGuard:
    """52W values outside [0, 100] are meaningless percentiles → fail-open."""

    def test_52w_above_100_is_not_excluded_even_with_high_floor(self):
        """52W = 10000 (raw data artefact, out-of-domain) → gate must NOT exclude,
        even when min_52w is set high enough that it would catch a real value."""
        # With min_52w=9999, a real value of 10000 would pass (10000 >= 9999),
        # but the spec says out-of-domain (>100) must be fail-open regardless.
        # Use default config (min_52w=30): 10000 > 100, so it must be treated as
        # UNKNOWN and the gate must be fail-open (not excluded).
        df = _make_df(_passing_row(TKR="BIGW", **{"52W": "10000"}))
        result = filter_universe(df)
        assert "BIGW" in result["eligible"]["TKR"].values
        reasons = result["reasons"]["BIGW"]
        assert not any("melting" in r for r in reasons)

    def test_52w_above_100_treated_as_unknown_fail_open(self):
        """52W = 11333 (real data point observed) → UNKNOWN → fail-open.
        Critically: even with min_52w=200 (absurd but shows the gate treats it as unknown)
        the row must NOT be excluded."""
        df = _make_df(_passing_row(TKR="OOD", **{"52W": "11333"}))
        result = filter_universe(df, config={"min_52w": 200})
        # If current code applies the gate naively: 11333 < 200 is False → passes anyway.
        # After fix: 11333 > 100 → UNKNOWN → fail-open → passes.
        # Both arrive at same result here; the distinguishing test is with min_52w < value.
        # Real distinction: 50 < 10000 but 50 is a valid floor. Use min_52w=99.
        df2 = _make_df(_passing_row(TKR="OOD2", **{"52W": "150"}))
        result2 = filter_universe(df2, config={"min_52w": 99})
        # 150 > 100 → out of domain → UNKNOWN → fail-open → must NOT be excluded
        assert "OOD2" in result2["eligible"]["TKR"].values
        reasons2 = result2["reasons"]["OOD2"]
        assert not any("melting" in r for r in reasons2)

    def test_52w_below_floor_within_domain_is_excluded(self):
        """52W = 15 (valid percentile 0-100, below 30 floor) → excluded by trend gate."""
        df = _make_df(_passing_row(TKR="LOWW", **{"52W": "15"}))
        result = filter_universe(df)
        assert "LOWW" not in result["eligible"]["TKR"].values
        reasons = result["reasons"]["LOWW"]
        assert any("melting" in r for r in reasons)


# ---------------------------------------------------------------------------
# FIX 3 REGRESSION — positive-earnings gate: (PEF<0 AND FCF<0) → fail
# ---------------------------------------------------------------------------


class TestFilterUniverseEarningsClearLossMaker:
    """Stale positive PET must not rescue a clear loss-maker (PEF<0 AND FCF<0)."""

    def test_positive_pet_negative_pef_and_fcf_fails(self):
        """{PET:20, PEF:-5, FCF:-3%} → excluded (clear loss-maker)."""
        df = _make_df(_passing_row(TKR="LOSER", PET="20.0", PEF="-5.0", FCF="-3.0%"))
        result = filter_universe(df)
        assert "LOSER" not in result["eligible"]["TKR"].values
        reasons = result["reasons"]["LOSER"]
        assert any("negative earnings" in r.lower() or "loss" in r.lower() for r in reasons)

    def test_positive_pet_negative_pef_positive_fcf_passes(self):
        """{PET:20, PEF:-5, FCF:+1%} → eligible (FCF positive saves it)."""
        df = _make_df(_passing_row(TKR="FCFOK2", PET="20.0", PEF="-5.0", FCF="1.0%"))
        result = filter_universe(df)
        assert "FCFOK2" in result["eligible"]["TKR"].values

    def test_positive_pet_and_pef_negative_fcf_passes(self):
        """{PET:20, PEF:15, FCF:-1%} → eligible (PEF positive saves it)."""
        df = _make_df(_passing_row(TKR="PEFOK", PET="20.0", PEF="15.0", FCF="-1.0%"))
        result = filter_universe(df)
        assert "PEFOK" in result["eligible"]["TKR"].values
