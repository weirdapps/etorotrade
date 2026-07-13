"""Tests for trade_modules/validation/xsection_ic.py (Phase 2B).

TDD: written FIRST — should fail on import until xsection_ic.py exists.

Covers:
  - cross_sectional_ic : per-date Spearman rank IC of a signal vs forward return
  - incremental_ic     : incumbent-partialled marginal cross-sectional rank IC
  - harness.evaluate wiring (optional signal_score_col block)

All assertions check real numeric behaviour, not tautologies.
"""

import numpy as np
import pandas as pd

from trade_modules.validation.harness import evaluate
from trade_modules.validation.xsection_ic import cross_sectional_ic, incremental_ic

# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------


def _predictive_panel(n_dates: int, n_names: int, noise: float, seed: int) -> pd.DataFrame:
    """score = forward + small noise → strong positive per-date IC."""
    rng = np.random.default_rng(seed)
    recs = []
    for d in range(n_dates):
        date = f"2026-01-{d + 1:02d}"
        fwd = rng.normal(0.0, 1.0, n_names)
        score = fwd + rng.normal(0.0, noise, n_names)
        for i in range(n_names):
            recs.append(
                {"signal_date": date, "ticker": f"T{i:02d}", "score": score[i], "forward": fwd[i]}
            )
    return pd.DataFrame(recs)


def _noise_panel(n_dates: int, n_names: int, seed: int) -> pd.DataFrame:
    """score independent of forward → IC centred on zero."""
    rng = np.random.default_rng(seed)
    recs = []
    for d in range(n_dates):
        date = f"2026-02-{d + 1:02d}"
        fwd = rng.normal(0.0, 1.0, n_names)
        score = rng.normal(0.0, 1.0, n_names)
        for i in range(n_names):
            recs.append(
                {"signal_date": date, "ticker": f"T{i:02d}", "score": score[i], "forward": fwd[i]}
            )
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# cross_sectional_ic
# ---------------------------------------------------------------------------


class TestCrossSectionalIC:
    def test_keys_present(self):
        df = _predictive_panel(6, 20, 0.1, seed=1)
        res = cross_sectional_ic(df, "score", "forward")
        assert set(res.keys()) == {
            "mean_ic",
            "ic_std",
            "t_stat",
            "hit_rate",
            "n_dates",
            "ic_by_date",
        }

    def test_predictive_signal_strong_positive_ic(self):
        df = _predictive_panel(12, 25, 0.1, seed=2)
        res = cross_sectional_ic(df, "score", "forward")
        assert res["n_dates"] == 12
        assert res["mean_ic"] > 0.7
        assert res["t_stat"] > 5.0
        assert res["hit_rate"] > 0.9

    def test_pure_noise_signal_near_zero_ic(self):
        df = _noise_panel(30, 40, seed=7)
        res = cross_sectional_ic(df, "score", "forward")
        assert res["n_dates"] == 30
        assert abs(res["mean_ic"]) < 0.1
        assert abs(res["t_stat"]) < 2.0

    def test_hit_rate_matches_ic_by_date(self):
        df = _predictive_panel(8, 20, 0.1, seed=3)
        res = cross_sectional_ic(df, "score", "forward")
        pos = sum(1 for v in res["ic_by_date"].values() if v > 0)
        assert res["hit_rate"] == pos / res["n_dates"]

    def test_dates_with_fewer_than_three_names_skipped(self):
        df = _predictive_panel(5, 10, 0.1, seed=4)
        # add a thin date with only 2 names — must be skipped
        thin = pd.DataFrame(
            [
                {"signal_date": "2026-01-31", "ticker": "Z0", "score": 1.0, "forward": 1.0},
                {"signal_date": "2026-01-31", "ticker": "Z1", "score": 2.0, "forward": 2.0},
            ]
        )
        res = cross_sectional_ic(pd.concat([df, thin], ignore_index=True), "score", "forward")
        assert res["n_dates"] == 5
        assert "2026-01-31" not in res["ic_by_date"]

    def test_nan_pairs_dropped_but_date_kept(self):
        df = _predictive_panel(4, 12, 0.1, seed=5)
        # corrupt a few cells with NaN — pairwise-dropped, dates keep >=3 names
        df.loc[0, "score"] = np.nan
        df.loc[1, "forward"] = np.nan
        res = cross_sectional_ic(df, "score", "forward")
        assert res["n_dates"] == 4
        assert res["mean_ic"] > 0.5  # still strongly predictive after dropping

    def test_empty_panel_graceful(self):
        df = pd.DataFrame(columns=["signal_date", "score", "forward"])
        res = cross_sectional_ic(df, "score", "forward")
        assert res["n_dates"] == 0
        assert res["mean_ic"] is None
        assert res["t_stat"] is None
        assert res["ic_by_date"] == {}

    def test_custom_date_col(self):
        df = _predictive_panel(5, 15, 0.1, seed=6).rename(columns={"signal_date": "asof"})
        res = cross_sectional_ic(df, "score", "forward", date_col="asof")
        assert res["n_dates"] == 5
        assert res["mean_ic"] > 0.7


# ---------------------------------------------------------------------------
# incremental_ic
# ---------------------------------------------------------------------------


class TestIncrementalIC:
    def test_keys_present(self):
        df = _predictive_panel(6, 20, 0.1, seed=11)
        df["incumbent"] = df["score"]
        res = incremental_ic(df, "score", ["incumbent"], "forward")
        assert set(res.keys()) == {"raw_ic", "incremental_ic", "ratio", "t_stat", "n_dates"}

    def test_signal_equals_incumbent_zero_incremental(self):
        """When the new signal IS the incumbent, it adds no marginal rank info."""
        df = _predictive_panel(10, 25, 0.1, seed=12)
        df["incumbent"] = df["score"]
        res = incremental_ic(df, "score", ["incumbent"], "forward")
        assert res["raw_ic"] > 0.5  # raw signal is predictive
        assert abs(res["incremental_ic"]) < 0.02  # but nothing beyond incumbent
        assert abs(res["ratio"]) < 0.05

    def test_orthogonal_incumbent_preserves_incremental(self):
        """Incumbent uncorrelated with signal → incremental ≈ raw IC."""
        rng = np.random.default_rng(13)
        recs = []
        for d in range(12):
            date = f"2026-04-{d + 1:02d}"
            fwd = rng.normal(0.0, 1.0, 30)
            score = fwd + rng.normal(0.0, 0.1, 30)
            incumbent = rng.normal(0.0, 1.0, 30)  # independent of score & forward
            for i in range(30):
                recs.append(
                    {
                        "signal_date": date,
                        "ticker": f"T{i:02d}",
                        "score": score[i],
                        "incumbent": incumbent[i],
                        "forward": fwd[i],
                    }
                )
        df = pd.DataFrame(recs)
        res = incremental_ic(df, "score", ["incumbent"], "forward")
        assert res["incremental_ic"] > 0.5
        assert abs(res["incremental_ic"] - res["raw_ic"]) < 0.15
        assert res["ratio"] > 0.8
        assert res["t_stat"] is not None and res["t_stat"] > 3.0

    def test_multiple_incumbents(self):
        rng = np.random.default_rng(14)
        recs = []
        for d in range(10):
            date = f"2026-05-{d + 1:02d}"
            fwd = rng.normal(0.0, 1.0, 40)
            score = fwd + rng.normal(0.0, 0.1, 40)
            inc1 = rng.normal(0.0, 1.0, 40)
            inc2 = rng.normal(0.0, 1.0, 40)
            for i in range(40):
                recs.append(
                    {
                        "signal_date": date,
                        "ticker": f"T{i:02d}",
                        "score": score[i],
                        "inc1": inc1[i],
                        "inc2": inc2[i],
                        "forward": fwd[i],
                    }
                )
        df = pd.DataFrame(recs)
        res = incremental_ic(df, "score", ["inc1", "inc2"], "forward")
        assert res["n_dates"] == 10
        assert res["incremental_ic"] > 0.5  # orthogonal incumbents don't absorb the edge

    def test_empty_panel_graceful(self):
        df = pd.DataFrame(columns=["signal_date", "score", "incumbent", "forward"])
        res = incremental_ic(df, "score", ["incumbent"], "forward")
        assert res["n_dates"] == 0
        assert res["incremental_ic"] is None
        assert res["raw_ic"] is None
        assert res["ratio"] is None


# ---------------------------------------------------------------------------
# harness.evaluate wiring
# ---------------------------------------------------------------------------


def _rows_with_conviction(n_dates: int, n_names: int, seed: int, predictive: bool) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        date = f"2026-06-{d + 1:02d}"
        for i in range(n_names):
            conv = float(rng.normal())
            if predictive:
                alpha = conv * 0.02 + float(rng.normal(0.0, 0.002))
            else:
                alpha = float(rng.normal())
            rows.append(
                {
                    "signal": "TEST",
                    "signal_date": date,
                    "ticker": f"T{i:02d}",
                    "horizon": 30,
                    "alpha": alpha,
                    "net_alpha": alpha - 0.001,
                    "future_price": 100.0,
                    "conviction": conv,
                    "regime": "risk_on" if (d + i) % 2 == 0 else "risk_off",
                }
            )
    return rows


class TestHarnessWiring:
    def test_block_absent_column_skips_gracefully(self):
        rows = _rows_with_conviction(6, 15, seed=21, predictive=True)
        for r in rows:
            del r["conviction"]
        result = evaluate(rows)
        assert "cross_sectional_ic" in result
        assert result["cross_sectional_ic"]["computed"] is False
        # existing report structure intact
        assert "families" in result and "overall" in result

    def test_block_populates_when_conviction_predictive(self):
        rows = _rows_with_conviction(10, 15, seed=22, predictive=True)
        result = evaluate(rows)
        block = result["cross_sectional_ic"]
        assert block["computed"] is True
        assert block["signal_score_col"] == "conviction"
        assert block["mean_ic"] > 0.3
        assert block["n_dates"] == 10

    def test_custom_score_col(self):
        rows = _rows_with_conviction(8, 15, seed=23, predictive=True)
        for r in rows:
            r["edge_score"] = r.pop("conviction")
        result = evaluate(rows, signal_score_col="edge_score")
        assert result["cross_sectional_ic"]["computed"] is True
        assert result["cross_sectional_ic"]["signal_score_col"] == "edge_score"
