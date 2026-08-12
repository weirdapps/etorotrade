"""Per-market short-interest thresholds, and the removal of the sell trigger.

ONE NUMBER WAS NEVER ONE GATE. Four regulators measure short interest four different ways, so
`max_short_interest = 2.0` meant four different strictnesses. Measured on the live pools:

    source                              n      percentile of 2.0   median
    US live (yfinance, % of float)      2,333  7th                 7.00
    US FINRA (% of shares outstanding)  95     42nd                2.28
    HK SFC                              107    25th                3.80
    JP JPX                              17     76th                1.17

Rows 1 and 2 are the clean natural experiment: same market, same days, 3x apart, purely because
one is percent-of-float and the other a percent-of-outstanding lower bound. If the basis moves
the number 3x inside one market, comparing HK's 3.80 to JP's 1.17 says nothing about which
market is more shorted.

The anchor is the owner's: US 8.0 (p67, just above the 7.00 median), mapped onto each market's
OWN distribution so every market gets the same STRICTNESS rather than the same percentage —
23% / 29% / 23% rejected, against 65% / 70% / 5% under the single 2.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_modules.analysis.signals import (
    resolve_max_short_interest,
    si_market_ceilings,
)

# The ceilings as config.yaml's regional_adjustments declares them.
CEILINGS = {".HK": 4.68, ".T": 0.0}


class TestThresholdResolution:
    """`.T` -> JP, `.HK` -> HK, everything else -> the default."""

    @pytest.fixture
    def criteria(self):
        return {"max_short_interest": 8.0}

    @pytest.mark.parametrize(
        ("ticker", "expected"),
        [
            ("7203.T", 0.0),  # Tokyo
            ("7741.T", 0.0),
            ("0700.HK", 4.68),  # Hong Kong
            ("1308.HK", 4.68),
            ("AAPL", 8.0),  # US, no suffix
            ("SAP.DE", 8.0),  # Europe falls through to the default
            ("RRL.AX", 8.0),
            ("BP.L", 8.0),
        ],
    )
    def test_suffix_selects_the_market(self, criteria, ticker, expected):
        assert resolve_max_short_interest(ticker, criteria, CEILINGS) == expected

    def test_absent_override_map_falls_back_to_the_scalar(self):
        """The scalar keeps its old meaning, so a config without the map is unchanged, and a
        missing/unreadable YAML makes the per-market layer inert rather than lethal."""
        assert resolve_max_short_interest("7203.T", {"max_short_interest": 2.0}) == 2.0

    def test_absent_scalar_yields_none_so_the_gate_is_skipped(self):
        assert resolve_max_short_interest("AAPL", {}, CEILINGS) is None

    def test_case_and_whitespace_do_not_change_the_market(self):
        c = {"max_short_interest": 8.0}
        assert resolve_max_short_interest(" 7203.t ", c, CEILINGS) == 0.0
        assert resolve_max_short_interest("0700.hk", c, CEILINGS) == 4.68


class TestJapanIsZeroOnPurpose:
    """READ B.4 BEFORE CHANGING THIS. 0.0 will look wrong and is not.

    Japan discloses only positions of 0.5% or more, PER SELLER, so the distribution has no mass
    between 0 and 0.5: 78% of TSE names are exactly 0.00 and 22% are >= 0.5%. Every threshold in
    the open interval (0, 0.5) vetoes the identical 50 names, so 0.0 is the only distinct choice
    in that range — a tuning knob with one position.

    The operator is strict (`row_si > threshold`), so 0.0 asks a CATEGORICAL question: does a
    reportable short position exist? In the US the median name is 7% shorted and existence
    carries no information, so a level is needed. In Japan only 22% have any, so existence IS
    the signal. Same factor, different operator, because the regulators differ.
    """

    def test_zero_passes_a_name_with_no_reportable_position(self):
        assert not _vetoed(si=0.0, threshold=0.0)

    def test_zero_vetoes_a_name_that_has_one(self):
        assert _vetoed(si=0.6, threshold=0.0)

    def test_every_threshold_below_the_disclosure_floor_is_the_same_gate(self):
        """(0, 0.5) is empty in the data, so 0.1 / 0.25 / 0.49 cannot differ from each other."""
        japan_like = [0.0] * 78 + [0.5, 0.8, 1.17, 2.2] * 5  # 78% zeros, rest >= 0.5
        counts = {t: sum(1 for si in japan_like if si > t) for t in (0.1, 0.25, 0.49)}
        assert len(set(counts.values())) == 1


def _vetoed(si: float, threshold: float) -> bool:
    """The operator as written at the enforcement site: strict `>`."""
    return bool(si > threshold)


class TestBuyVetoEndToEnd:
    """Through `calculate_action_vectorized`, the path production actually runs."""

    def _frame(self, ticker: str, si: float | None) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "upside": 30.0,
                    "buy_percentage": 90.0,
                    "analyst_count": 20,
                    "total_ratings": 20,
                    "beta": 1.0,
                    "pe_trailing": 20.0,
                    "pe_forward": 15.0,
                    "peg_ratio": 1.0,
                    "short_percent": si,
                    "market_cap": 5e10,
                    "price": 100.0,
                    "target_price": 130.0,
                    "EXRET": 20.0,
                }
            ]
        ).set_index("ticker", drop=False)

    @pytest.mark.parametrize(
        ("ticker", "below", "above"),
        [
            # Japan: 0.00 (no reportable position) vs 0.6 (one exists). NOT 0.4 — the handoff's
            # B.7 asked for "0.4 passes", which contradicts its own B.4: the operator is strict, so
            # 0.4 > 0.0 vetoes. The contradiction is harmless only because the value cannot occur —
            # the disclosure floor is 0.5%, so (0, 0.5) is empty in the data.
            ("7203.T", 0.0, 0.6),
            ("0700.HK", 4.0, 5.0),  # either side of HK's 4.68
            ("AAPL", 7.0, 9.0),  # either side of the US 8.0
            ("SAP.DE", 7.0, 9.0),  # unknown suffix -> the default applies
        ],
    )
    def test_the_gate_bites_only_above_the_market_ceiling(self, ticker, below, above):
        """Self-validating: the NaN row is the control. If it is not a buy, the fixture cannot
        exercise the gate at all and the test says so instead of passing vacuously."""
        from trade_modules.analysis.signals import calculate_action_vectorized

        control, _, _ = calculate_action_vectorized(self._frame(ticker, np.nan), "market")
        if control.iloc[0] != "B":
            pytest.skip(
                f"{ticker} is not a buy with SI absent ({control.iloc[0]}) — "
                "the fixture fails some other criterion, so the SI gate is untestable here"
            )
        lo, _, _ = calculate_action_vectorized(self._frame(ticker, below), "market")
        hi, _, _ = calculate_action_vectorized(self._frame(ticker, above), "market")
        assert lo.iloc[0] == "B", f"{ticker} at SI={below} should pass its ceiling"
        assert hi.iloc[0] != "B", f"{ticker} at SI={above} should be vetoed"

    @pytest.mark.parametrize("ticker", ["7203.T", "0700.HK", "AAPL", "SAP.DE"])
    def test_nan_is_a_free_pass_in_every_market(self, ticker):
        """A percentile of an unknown is unknown. An absent value must never become a veto —
        the US gate is already substantially a filter on data COVERAGE (only 30% of survivors
        carry SI at all, and 65% of those are killed), and this keeps it from getting worse.

        Asserted as an EQUIVALENCE, not an absolute action: this fixture reaches the live
        provider, so the action it lands on depends on data that moves. The property that
        matters is that the SI gate contributes nothing when the value is absent, i.e. NaN
        behaves exactly like a value under the ceiling. 0.0 is under every market's ceiling
        (0.0 > 0.0 is False), so it is the right comparand everywhere.
        """
        from trade_modules.analysis.signals import calculate_action_vectorized

        absent, _, _ = calculate_action_vectorized(self._frame(ticker, np.nan), "market")
        passing, _, _ = calculate_action_vectorized(self._frame(ticker, 0.0), "market")
        assert absent.iloc[0] == passing.iloc[0], (
            f"{ticker}: absent SI gave {absent.iloc[0]}, a passing SI gave {passing.iloc[0]} — "
            "the gate is treating an unknown differently from a known-good value"
        )


class TestSellTriggerIsGone:
    """`min_short_interest: 3.0` fired a sell UNCONDITIONALLY — `signals.py` marks a sell on
    `if any(sell_conditions)`, so high short interest alone forced one.

    Deleted rather than re-calibrated, on this codebase's own evidence:
      * the CONTINUOUS form works — `empirical_factor.py` rho = -0.111 over n=32,589, the
        largest |rho| of the four factors;
      * the DISCRETE form was already tried and killed — `synthesis.py` V37 deprecations,
        "short_interest_weakness  # n=32  penalized winners", removed 2026-05-16;
      * the factor is NOT monotone — high SI is both the bear case and the squeeze setup, which
        is why `synthesis.py` already makes it conditional on tech_signal and fund_score. An
        absolute threshold can only express a monotone rule, so it must get one regime wrong.

    Asymmetry: a buy veto forgoes one opportunity among thousands; a sell trigger forces a
    realised transaction — spread, tax, and the destruction of a researched position.
    """

    def test_the_key_is_gone_from_the_live_config_and_the_fallback(self):
        """config.yaml is the live path — trade_config.py is only the fallback when the YAML is
        missing, which is why the original handoff's file/line table missed 30 of the 38 sites."""
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[3]
        for rel in ("config.yaml", "trade_modules/trade_config.py"):
            text = (root / rel).read_text()
            assert "min_short_interest:" not in text and '"min_short_interest"' not in text, rel

    def test_no_source_file_still_reads_it(self):
        """The branch must go, not just the config value — a live branch reading an absent key
        is a gate that silently stops existing rather than one that was removed."""
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[3]
        # A live reference is a dict access or a key literal. The deletion note that explains
        # WHY it went is prose and must survive — a rule removed without its reasoning gets
        # reinstated by the next reader.
        live = re.compile(
            r"""(get\(\s*["']min_short_interest|\[["']min_short_interest|["']min_short_interest["']\s*:)"""
        )
        hits = [
            str(p.relative_to(root))
            for p in (root / "trade_modules").rglob("*.py")
            if live.search(p.read_text())
        ]
        assert hits == [], f"min_short_interest is still READ in {hits}"

    def test_high_short_interest_alone_no_longer_forces_a_sell(self):
        from trade_modules.analysis.signals import calculate_action_vectorized

        df = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "upside": 25.0,
                    "buy_percentage": 85.0,
                    "analyst_count": 20,
                    "total_ratings": 20,
                    "beta": 1.0,
                    "pe_trailing": 20.0,
                    "pe_forward": 15.0,
                    "peg_ratio": 1.0,
                    "short_percent": 12.0,
                    "market_cap": 5e10,
                    "price": 100.0,
                    "target_price": 125.0,
                    "EXRET": 20.0,
                }
            ]
        ).set_index("ticker", drop=False)
        out, _, _ = calculate_action_vectorized(df, "market")
        assert out.iloc[0] != "S", "short interest alone still forces a sell"


class TestCeilingsComeFromTheConfigNotACopy:
    """The suffix -> market table is declared once, in config.yaml's regional_adjustments.
    A private copy in signals.py would be the sixth copy of one mapping in this estate."""

    def test_built_from_regional_adjustments(self):
        cfg = {
            "regional_adjustments": {
                "hong_kong": {"suffixes": [".HK"], "max_short_interest": 4.68},
                "japan": {"suffixes": [".T"], "max_short_interest": 0.0},
                "europe": {"suffixes": [".DE", ".L"]},  # no ceiling -> contributes nothing
                "us": {"adjustments": "none"},
            }
        }
        assert si_market_ceilings(cfg) == {".HK": 4.68, ".T": 0.0}

    def test_missing_config_is_inert_not_lethal(self):
        assert si_market_ceilings(None) == {}
        assert si_market_ceilings({}) == {}

    def test_the_live_yaml_actually_carries_them(self):
        """Guards the wiring, not just the function: if someone removes the ceiling from
        regional_adjustments, the per-market layer silently stops existing."""
        import pathlib

        import yaml

        root = pathlib.Path(__file__).resolve().parents[3]
        cfg = yaml.safe_load((root / "config.yaml").read_text())
        assert si_market_ceilings(cfg) == {".HK": 4.68, ".T": 0.0}

    def test_no_private_suffix_table_was_reintroduced(self):
        import inspect

        from trade_modules.analysis import signals

        src = inspect.getsource(signals.si_market_ceilings)
        assert "regional_adjustments" in src
        assert '"T": "JP"' not in inspect.getsource(signals)
