"""TDD tests for trade_modules.v3.combine.compute_scores."""

import numpy as np
import pandas as pd
import pytest

from trade_modules.v3.combine import CLUSTERS, compute_scores


def _base(index):
    """A features frame with all metric columns NaN, so each test isolates one."""
    cols = set()
    for members in CLUSTERS.values():
        cols.update(members)
    df = pd.DataFrame({c: [np.nan] * len(index) for c in cols}, index=index)
    df["sector"] = ["__NA__"] * len(index)
    return df


def test_value_metric_negation_cheap_ranks_high():
    """Low P/E (cheap) must produce a HIGH value_z (value metrics are negated)."""
    df = _base(["CHEAP", "MID", "RICH"])
    df["pe_trailing"] = [5.0, 15.0, 50.0]
    scores = compute_scores(df, sector_neutral=False)
    assert scores.loc["CHEAP", "value_z"] > scores.loc["MID", "value_z"]
    assert scores.loc["MID", "value_z"] > scores.loc["RICH", "value_z"]
    # cheap should be clearly positive, rich clearly negative
    assert scores.loc["CHEAP", "value_z"] > 0 > scores.loc["RICH", "value_z"]


def test_quality_metric_kept_high_is_good():
    """High ROE must produce a HIGH quality_z (quality metrics are kept)."""
    df = _base(["A", "B", "C"])
    df["roe"] = [10.0, 20.0, 40.0]
    scores = compute_scores(df, sector_neutral=False)
    assert scores.loc["C", "quality_z"] > scores.loc["A", "quality_z"]
    assert scores.loc["C", "quality_z"] > 0 > scores.loc["A", "quality_z"]


def test_lowvol_metric_negated_low_vol_is_good():
    """Low realized_vol must produce a HIGH lowvol_z (negated)."""
    df = _base(["CALM", "WILD"])
    df["realized_vol"] = [0.10, 0.60]
    scores = compute_scores(df, sector_neutral=False)
    assert scores.loc["CALM", "lowvol_z"] > scores.loc["WILD", "lowvol_z"]


def test_sector_neutral_flips_a_sector_leader_positive():
    """A globally below-average name that LEADS its sector flips positive under
    sector-neutral demeaning."""
    df = _base(["A", "B", "C", "D"])
    df["sector"] = ["Tech", "Tech", "Fin", "Fin"]
    # Tech ROE high in absolute terms; Fin low. C leads Fin but is globally low.
    df["roe"] = [50.0, 40.0, 20.0, 10.0]

    glob = compute_scores(df, sector_neutral=False)
    sn = compute_scores(df, sector_neutral=True)

    # Globally, C (roe=20, below the 30 mean) is negative on quality.
    assert glob.loc["C", "quality_z"] < 0
    # Sector-neutral: C leads Fin (20 > 10) -> positive.
    assert sn.loc["C", "quality_z"] > 0


def test_conviction_and_rank_produced():
    df = _base(["A", "B", "C", "D"])
    df["pe_trailing"] = [5.0, 12.0, 25.0, 60.0]
    df["roe"] = [40.0, 25.0, 15.0, 5.0]
    df["mom_12_1"] = [0.30, 0.10, -0.05, -0.20]
    scores = compute_scores(df, sector_neutral=False)

    assert "conviction" in scores.columns
    assert "rank" in scores.columns
    # A is best on every axis -> rank 1, highest conviction.
    assert scores.loc["A", "rank"] == 1
    assert scores["conviction"].idxmax() == "A"
    # ranks are a permutation of 1..N
    assert sorted(int(r) for r in scores["rank"]) == [1, 2, 3, 4]
    # re-z'd conviction is ~mean 0
    assert abs(float(scores["conviction"].mean())) < 1e-6


def test_nans_tolerated():
    """Rows/metrics with NaN don't crash; clusters use present metrics only."""
    df = _base(["A", "B", "C"])
    df["pe_trailing"] = [5.0, np.nan, 40.0]  # B missing value metric
    df["roe"] = [30.0, 20.0, np.nan]  # C missing quality metric
    scores = compute_scores(df, sector_neutral=False)

    # B has no value metric -> value_z NaN, but still gets conviction from quality.
    assert pd.isna(scores.loc["B", "value_z"])
    assert np.isfinite(scores.loc["B", "conviction"])
    # C has no quality metric -> quality_z NaN, still ranked.
    assert pd.isna(scores.loc["C", "quality_z"])
    assert scores["rank"].notna().all()


def test_value_quality_weight_cap_about_55pct():
    """Value + Quality jointly carry ~55% of conviction weight."""
    from trade_modules.v3.combine import CLUSTER_WEIGHTS

    total = sum(CLUSTER_WEIGHTS.values())
    vq = CLUSTER_WEIGHTS["value_z"] + CLUSTER_WEIGHTS["quality_z"]
    assert abs(vq / total - 0.55) < 0.01


def test_no_quote_type_column_means_all_eligible():
    """Abstract combiner frames (no quote_type) keep the full ranked cross-section."""
    df = _base(["A", "B", "C"])
    df["pe_trailing"] = [5.0, 15.0, 40.0]
    df["roe"] = [40.0, 20.0, 10.0]
    scores = compute_scores(df, sector_neutral=False)
    assert scores["eligible"].all()
    assert scores["rank"].notna().all()


# (removed 2026-07-18, D8) test_accruals_in_quality_* — accruals is no longer a
# scored quality metric (moved to shadow; non-PIT, needs a point-in-time balance
# sheet). Coverage is now via test_fixnow_quality_interim_is_roe_fcf.


def test_eligibility_excludes_non_equity_and_dataless():
    """With quote_type present, ETFs and dataless names drop out of the ranking."""
    idx = ["EQ1", "EQ2", "ETF1", "DATALESS"]
    df = _base(idx)
    df["quote_type"] = ["EQUITY", "EQUITY", "ETF", "EQUITY"]
    df["price"] = [100.0, 50.0, 200.0, 30.0]
    df["pe_trailing"] = [10.0, 20.0, 15.0, 12.0]
    df["roe"] = [30.0, 20.0, 25.0, 18.0]
    df["mom_12_1"] = [0.20, 0.10, 0.15, 0.05]
    df["realized_vol"] = [0.20, 0.25, 0.22, 0.30]
    # DATALESS: strip its core data (no price/mom/vol, and its cluster members go NaN).
    df.loc["DATALESS", ["price", "pe_trailing", "roe", "mom_12_1", "realized_vol"]] = np.nan

    scores = compute_scores(df, sector_neutral=False)

    # Two data-complete equities are eligible and ranked.
    assert bool(scores.loc["EQ1", "eligible"]) is True
    assert bool(scores.loc["EQ2", "eligible"]) is True
    assert scores.loc["EQ1", "rank"] in (1, 2)
    # The ETF is excluded despite having full data (non-equity).
    assert bool(scores.loc["ETF1", "eligible"]) is False
    assert pd.isna(scores.loc["ETF1", "conviction"])
    assert pd.isna(scores.loc["ETF1", "rank"])
    # The dataless equity is excluded (insufficient data).
    assert bool(scores.loc["DATALESS", "eligible"]) is False
    assert pd.isna(scores.loc["DATALESS", "rank"])
    # Ranking is dense over the two eligible names only.
    assert sorted(int(r) for r in scores["rank"].dropna()) == [1, 2]


# --------------------------------------------------------------------------- #
# PEG removal (honor the user's EXCLUDE decision)
# --------------------------------------------------------------------------- #


def test_peg_removed_from_direction_and_value_cluster():
    """PEG must not participate in scoring: gone from DIRECTION and value_z."""
    from trade_modules.v3.combine import CLUSTERS, DIRECTION

    assert "peg" not in DIRECTION
    assert "peg" not in CLUSTERS["value_z"]
    # No cluster references peg any more.
    for members in CLUSTERS.values():
        assert "peg" not in members


# --------------------------------------------------------------------------- #
# IC-weighting
# --------------------------------------------------------------------------- #


def test_derive_cluster_weights_none_uses_fixed_weights():
    from trade_modules.v3.combine import CLUSTER_WEIGHTS, derive_cluster_weights

    assert derive_cluster_weights(None) == CLUSTER_WEIGHTS


def test_derive_cluster_weights_proportional_to_positive_ic():
    """With no cap binding, cluster weights are ∝ max(IC, 0), renormalized."""
    from trade_modules.v3.combine import derive_cluster_weights

    ic = {
        "value_z": 0.1,
        "quality_z": 0.1,
        "momentum_z": 0.2,
        "lowvol_z": 0.3,
        "strength_z": 0.3,
    }
    w = derive_cluster_weights(ic)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    # ordering follows IC, and Value+Quality (0.2) is below the 0.55 cap so it is
    # left untouched (weights equal the normalized IC).
    assert w["strength_z"] > w["momentum_z"] > w["value_z"]
    assert abs(w["momentum_z"] - 0.2) < 1e-9
    assert abs((w["value_z"] + w["quality_z"]) - 0.2) < 1e-9


def test_derive_cluster_weights_preserves_value_quality_cap():
    """IC that would over-weight Value+Quality is capped at 0.55, excess to the rest."""
    from trade_modules.v3.combine import derive_cluster_weights

    ic = {
        "value_z": 0.5,
        "quality_z": 0.5,
        "momentum_z": 0.05,
        "lowvol_z": 0.05,
        "strength_z": 0.05,
    }
    w = derive_cluster_weights(ic)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    vq = w["value_z"] + w["quality_z"]
    assert abs(vq - 0.55) < 1e-9  # capped exactly at the joint cap
    # equal IC -> equal split inside the VQ bucket
    assert abs(w["value_z"] - w["quality_z"]) < 1e-9
    # the freed 0.45 is spread over the other three (equal IC -> equal split)
    assert abs(w["momentum_z"] - 0.15) < 1e-9


def test_derive_cluster_weights_clamps_negative_ic():
    from trade_modules.v3.combine import derive_cluster_weights

    ic = {
        "value_z": -0.5,  # negative IC -> zero weight
        "quality_z": 0.3,
        "momentum_z": 0.3,
        "lowvol_z": 0.2,
        "strength_z": 0.2,
    }
    w = derive_cluster_weights(ic)
    assert w["value_z"] == 0.0
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_derive_cluster_weights_all_nonpositive_falls_back_to_fixed():
    from trade_modules.v3.combine import CLUSTER_WEIGHTS, derive_cluster_weights

    ic = dict.fromkeys(CLUSTER_WEIGHTS, -1.0)
    assert derive_cluster_weights(ic) == CLUSTER_WEIGHTS


def test_derive_cluster_weights_equal_spread_when_rest_ic_nonpositive():
    """Freed weight spreads equally when non-VQ clusters all have IC ≤ 0 (else branch).

    Triggers the ``else`` branch at combine.py ~line 118: VQ IC is high enough
    to push sum above the 0.55 cap, and the three remaining clusters (momentum,
    lowvol, strength) have IC ≤ 0 so their raw weights are all zero.  The freed
    weight (1 - 0.55 = 0.45) must be spread equally across the three clusters,
    giving each 0.15, and the VQ pair is still exactly at the 0.55 cap.
    """
    from trade_modules.v3.combine import derive_cluster_weights

    ic = {
        "value_z": 0.6,
        "quality_z": 0.6,
        "momentum_z": -0.1,  # negative IC -> clamped to 0 (else branch triggered)
        "lowvol_z": 0.0,  # zero IC -> clamped to 0
        "strength_z": -0.2,  # negative IC -> clamped to 0
    }
    w = derive_cluster_weights(ic)

    assert abs(sum(w.values()) - 1.0) < 1e-9, f"weights sum {sum(w.values())} != 1.0"
    assert abs(w["value_z"] + w["quality_z"] - 0.55) < 1e-9, "VQ cap not exactly 0.55"

    expected_each = (1.0 - 0.55) / 3  # 0.45 / 3 = 0.15
    for c in ("momentum_z", "lowvol_z", "strength_z"):
        assert abs(w[c] - expected_each) < 1e-9, (
            f"{c}: got {w[c]:.6f}, expected {expected_each:.6f}"
        )


def test_ic_weighting_shifts_conviction_toward_high_ic_cluster():
    """A momentum-favoring IC vector flips the value-name / momentum-name ranking."""
    df = _base(["MOM_STAR", "VAL_STAR", "MID1", "MID2"])
    df["pe_trailing"] = [50.0, 5.0, 20.0, 25.0]  # VAL_STAR cheapest -> best value
    df["mom_12_1"] = [0.50, -0.20, 0.10, 0.05]  # MOM_STAR strongest momentum

    default = compute_scores(df, sector_neutral=False)
    # Fixed weights lean on Value (Value+Quality ~55%) -> the value name wins.
    assert default.loc["VAL_STAR", "rank"] < default.loc["MOM_STAR", "rank"]

    ic = compute_scores(
        df,
        sector_neutral=False,
        ic_weights={
            "momentum_z": 1.0,
            "value_z": 0.05,
            "quality_z": 0.05,
            "lowvol_z": 0.05,
            "strength_z": 0.05,
        },
    )
    # Momentum-heavy IC -> the momentum name now out-ranks the value name.
    assert ic.loc["MOM_STAR", "rank"] < ic.loc["VAL_STAR", "rank"]


# --------------------------------------------------------------------------- #
# Growth cluster (6th cluster; off by default, weight 0.0)
# --------------------------------------------------------------------------- #

# The three factor-weight sensitivity configs (short cluster names), mirrored in
# scripts/v3_sensitivity.py.  Kept inline so these tests do not import the runner.
_VALUE_HEAVY = {
    "value": 0.275,
    "quality": 0.275,
    "momentum": 0.20,
    "growth": 0.0,
    "lowvol": 0.15,
    "strength": 0.10,
}
_GROWTH_FORWARD = {
    "value": 0.10,
    "quality": 0.22,
    "momentum": 0.30,
    "growth": 0.20,
    "lowvol": 0.10,
    "strength": 0.08,
}


def test_growth_cluster_registered_direction_positive():
    """Growth is the 6th cluster: earnings + revenue growth, DIRECTION +1."""
    from trade_modules.v3.combine import CLUSTERS, DIRECTION

    assert "growth_z" in CLUSTERS
    assert set(CLUSTERS["growth_z"]) == {"earn_growth", "rev_growth"}
    assert DIRECTION["earn_growth"] == +1
    assert DIRECTION["rev_growth"] == +1
    assert list(CLUSTERS.keys()) == [
        "value_z",
        "quality_z",
        "momentum_z",
        "growth_z",
        "lowvol_z",
        "strength_z",
    ]


def test_growth_z_high_growth_ranks_above_low_growth():
    """A high-growth name gets a HIGHER growth_z than a low-growth peer (kept, +1)."""
    df = _base(["FAST", "MID", "SLOW"])
    df["earn_growth"] = [40.0, 12.0, 1.0]
    df["rev_growth"] = [0.35, 0.10, 0.01]
    scores = compute_scores(df, sector_neutral=False)
    assert scores.loc["FAST", "growth_z"] > scores.loc["MID", "growth_z"]
    assert scores.loc["MID", "growth_z"] > scores.loc["SLOW", "growth_z"]
    assert scores.loc["FAST", "growth_z"] > 0 > scores.loc["SLOW", "growth_z"]


def test_growth_z_nan_safe_single_metric():
    """growth_z tolerates a missing revenue-growth column (earnings only)."""
    df = _base(["A", "B"])
    df["earn_growth"] = [30.0, 5.0]  # rev_growth stays NaN in _base
    scores = compute_scores(df, sector_neutral=False)
    assert scores.loc["A", "growth_z"] > scores.loc["B", "growth_z"]


def test_default_growth_weight_is_zero():
    """Backward-compat contract: growth carries ZERO weight in the default model."""
    from trade_modules.v3.combine import CLUSTER_WEIGHTS

    assert CLUSTER_WEIGHTS["growth_z"] == 0.0
    # The five evidence clusters still sum to 1.0 (growth adds nothing by default).
    assert abs(sum(CLUSTER_WEIGHTS.values()) - 1.0) < 1e-12


# --------------------------------------------------------------------------- #
# explicit cluster_weights override
# --------------------------------------------------------------------------- #


def test_resolve_cluster_weights_normalizes_to_sum_one():
    from trade_modules.v3.combine import CLUSTERS, resolve_cluster_weights

    names = list(CLUSTERS.keys())
    # Unnormalized (sums to 2.0) -> normalized to sum 1, ratios preserved.
    w = resolve_cluster_weights({"value": 1.0, "quality": 1.0}, None, names)
    assert abs(sum(w.values()) - 1.0) < 1e-12
    assert abs(w["value_z"] - 0.5) < 1e-12
    assert abs(w["quality_z"] - 0.5) < 1e-12


def test_resolve_cluster_weights_missing_cluster_is_zero():
    from trade_modules.v3.combine import CLUSTERS, resolve_cluster_weights

    names = list(CLUSTERS.keys())
    w = resolve_cluster_weights({"momentum": 1.0}, None, names)
    assert abs(w["momentum_z"] - 1.0) < 1e-12
    for c in ("value_z", "quality_z", "growth_z", "lowvol_z", "strength_z"):
        assert w[c] == 0.0


def test_resolve_cluster_weights_accepts_long_and_short_names():
    from trade_modules.v3.combine import CLUSTERS, resolve_cluster_weights

    names = list(CLUSTERS.keys())
    short = resolve_cluster_weights({"value": 1.0, "growth": 1.0}, None, names)
    longn = resolve_cluster_weights({"value_z": 1.0, "growth_z": 1.0}, None, names)
    assert short == longn


def test_cluster_weights_precedence_over_ic_and_default():
    """Explicit cluster_weights override BOTH ic_weights and the default."""
    df = _base(["MOM_STAR", "VAL_STAR", "MID"])
    df["pe_trailing"] = [50.0, 5.0, 20.0]
    df["mom_12_1"] = [0.50, -0.20, 0.10]
    # cluster_weights momentum-only should beat an ic_weights that favors value.
    out = compute_scores(
        df,
        sector_neutral=False,
        ic_weights={"value_z": 1.0},
        cluster_weights={"momentum": 1.0},
    )
    assert out.loc["MOM_STAR", "rank"] < out.loc["VAL_STAR", "rank"]


def test_cluster_weights_growth_forward_shifts_ranking():
    """Growth-forward ranks a high-growth expensive name ABOVE a cheap no-growth name."""
    df = _base(["GROWTH_EXP", "VALUE_CHEAP", "MID"])
    df["pe_trailing"] = [60.0, 6.0, 20.0]  # GROWTH_EXP expensive, VALUE_CHEAP cheap
    df["earn_growth"] = [40.0, 1.0, 15.0]  # GROWTH_EXP fast grower, VALUE_CHEAP flat
    df["rev_growth"] = [0.35, 0.01, 0.12]
    df["mom_12_1"] = [0.40, -0.10, 0.10]  # momentum reinforces the growth name

    vh = compute_scores(df, sector_neutral=False, cluster_weights=_VALUE_HEAVY)
    gf = compute_scores(df, sector_neutral=False, cluster_weights=_GROWTH_FORWARD)

    # Value-heavy: the cheap no-growth name wins (value dominates, growth off).
    assert vh.loc["VALUE_CHEAP", "rank"] < vh.loc["GROWTH_EXP", "rank"]
    # Growth-forward: the expensive fast-grower overtakes the cheap flat name.
    assert gf.loc["GROWTH_EXP", "rank"] < gf.loc["VALUE_CHEAP", "rank"]


def test_backward_compat_none_equals_explicit_default_weights():
    """cluster_weights=None reproduces the default model EXACTLY (allclose)."""
    df = _base(["A", "B", "C", "D"])
    df["pe_trailing"] = [5.0, 12.0, 25.0, 60.0]
    df["roe"] = [40.0, 25.0, 15.0, 5.0]
    df["mom_12_1"] = [0.30, 0.10, -0.05, -0.20]

    default = compute_scores(df, sector_neutral=False)
    explicit = compute_scores(
        df,
        sector_neutral=False,
        cluster_weights={
            "value": 0.275,
            "quality": 0.275,
            "momentum": 0.20,
            "growth": 0.0,
            "lowvol": 0.15,
            "strength": 0.10,
        },
    )
    assert np.allclose(
        default["conviction"].to_numpy(dtype=float),
        explicit["conviction"].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_rank_normalization_bounds_outlier_magnitude():
    """A fat-tailed outlier must not dominate: growth_z depends on RANK, not size.

    Two identical universes except the top name's earnings growth is 5 vs 5000.
    Under rank-normalization the top name's growth_z is IDENTICAL in both (still
    just "rank 1 of 5") and bounded. The old winsor-z transform failed this: the
    5000x value survived the loose 1/99 clip and blew the z up, which is exactly
    what let small-base biotech growth artifacts dominate the book.
    """
    idx = ["A", "B", "C", "D", "E"]
    moderate = _base(idx)
    moderate["earn_growth"] = [1.0, 2.0, 3.0, 4.0, 5.0]
    extreme = _base(idx)
    extreme["earn_growth"] = [1.0, 2.0, 3.0, 4.0, 5000.0]
    zm = compute_scores(moderate, sector_neutral=False).loc["E", "growth_z"]
    ze = compute_scores(extreme, sector_neutral=False).loc["E", "growth_z"]
    assert zm == pytest.approx(ze)  # rank-invariant to the outlier's magnitude
    assert abs(ze) < 2.0  # bounded by rank, not blown up by the 5000x value


def test_backward_compat_default_ignores_growth_data():
    """With the default weights, adding growth data does NOT change conviction/rank."""
    df = _base(["A", "B", "C", "D"])
    df["pe_trailing"] = [5.0, 12.0, 25.0, 60.0]
    df["roe"] = [40.0, 25.0, 15.0, 5.0]
    df["mom_12_1"] = [0.30, 0.10, -0.05, -0.20]

    without = compute_scores(df, sector_neutral=False)
    withg = df.copy()
    withg["earn_growth"] = [2.0, 50.0, 5.0, 40.0]  # would reorder if growth were weighted
    withg["rev_growth"] = [0.01, 0.40, 0.03, 0.30]
    withg = compute_scores(withg, sector_neutral=False)

    # growth_z is computed, but weight 0 -> conviction identical to the no-growth run.
    assert withg["growth_z"].notna().all()
    assert np.allclose(
        without["conviction"].to_numpy(dtype=float),
        withg["conviction"].to_numpy(dtype=float),
        equal_nan=True,
    )
    assert list(without["rank"].astype("float")) == list(withg["rank"].astype("float"))


# --------------------------------------------------------------------------- #
# FIX-NOW (agreed setup 2026-07-18): rebuilt scored factor set
# --------------------------------------------------------------------------- #


def test_fixnow_strength_cluster_is_analyst_mom_only():
    """strength_z collapses to analyst_mom alone: upside, buy_pct, short_interest
    and target_dispersion are removed from ALL scored clusters (D2-D6)."""
    from trade_modules.v3.combine import CLUSTERS

    assert CLUSTERS["strength_z"] == ["analyst_mom"]
    for dead in ("upside", "buy_pct", "short_interest", "target_dispersion"):
        for members in CLUSTERS.values():
            assert dead not in members, f"{dead} must not be scored in any cluster"


def test_fixnow_value_drops_raw_pb_anchors_ev_ebitda():
    """Value = EV/EBITDA (anchor) + trailing P/E; raw P/B is dropped from scoring
    (intangible-heavy tech universe degrades it) (D7)."""
    from trade_modules.v3.combine import CLUSTERS

    assert CLUSTERS["value_z"] == ["pe_trailing", "ev_ebitda"]
    for members in CLUSTERS.values():
        assert "pb" not in members, "raw P/B must not be scored in any cluster"


def test_fixnow_quality_interim_is_roe_fcf():
    """Interim quality = ROE + FCF (panel-available). Per-sales margins + roa retired
    (GP/assets is the right anchor but unbuildable now → BUILD); current_ratio + de move
    to a distress filter; accruals is shadow-only (non-PIT) (D8)."""
    from trade_modules.v3.combine import CLUSTERS

    assert CLUSTERS["quality_z"] == ["roe", "fcf"]
    for dead in ("roa", "gross_margin", "op_margin", "current_ratio", "de", "accruals"):
        for members in CLUSTERS.values():
            assert dead not in members, f"{dead} must not be scored in quality (interim)"
