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
    """Low forward P/E (cheap) must produce a HIGH value_z (value metrics are negated)."""
    df = _base(["CHEAP", "MID", "RICH"])
    df["pe_forward"] = [5.0, 15.0, 50.0]
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
    df["pe_forward"] = [5.0, 12.0, 25.0, 60.0]
    df["roe"] = [40.0, 25.0, 15.0, 5.0]
    df["pct_52w_high"] = [0.30, 0.10, -0.05, -0.20]
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
    df["pe_forward"] = [5.0, np.nan, 40.0]  # B missing value metric
    df["roe"] = [30.0, 20.0, np.nan]  # C missing quality metric
    scores = compute_scores(df, sector_neutral=False)

    # B has no value metric -> value_z NaN, but still gets conviction from quality.
    assert pd.isna(scores.loc["B", "value_z"])
    assert np.isfinite(scores.loc["B", "conviction"])
    # C has no quality metric -> quality_z NaN, still ranked.
    assert pd.isna(scores.loc["C", "quality_z"])
    assert scores["rank"].notna().all()


# (removed 2026-07-18, D7/D14) test_value_quality_weight_cap_about_55pct — the
# Value+Quality 0.55 cap-as-floor is deleted; core clusters are now equal-risk.
# See test_fixnow_equal_weight_core_no_vq_floor.


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


def test_value_trap_gate_excludes_forward_pe_far_above_trailing():
    """Owner rule: forward P/E > 1.10x trailing (earn_trajectory = PEF/PET > 1.10) = value
    trap -> INELIGIBLE, so a trap never surfaces as a BUY and a held trap trips the overlay's
    weak-name SELL. Names at/below the threshold stay eligible."""
    idx = ["GOOD", "FLAT", "TRAP"]
    df = _base(idx)
    df["quote_type"] = ["EQUITY", "EQUITY", "EQUITY"]
    df["price"] = [100.0, 50.0, 80.0]
    df["pe_trailing"] = [10.0, 15.0, 3.2]
    df["roe"] = [30.0, 20.0, 25.0]
    df["mom_12_1"] = [0.20, 0.15, 0.10]
    df["realized_vol"] = [0.20, 0.22, 0.25]
    # earn_trajectory = PEF/PET: GOOD 0.90 (forward cheaper), FLAT 1.0 (=trailing, not a
    # trap), TRAP 46/3.2 = 14.4 (forward 14x trailing = earnings collapsing).
    df["earn_trajectory"] = [0.90, 1.0, 46.0 / 3.2]

    scores = compute_scores(df, sector_neutral=False)

    assert bool(scores.loc["GOOD", "eligible"]) is True
    assert bool(scores.loc["FLAT", "eligible"]) is True  # exactly flat is not a trap
    assert bool(scores.loc["TRAP", "eligible"]) is False
    assert pd.isna(scores.loc["TRAP", "conviction"])
    assert pd.isna(scores.loc["TRAP", "rank"])


def test_negative_pe_forward_not_scored_as_cheap():
    """Owner 2026-07-23: a NEGATIVE forward P/E (loss expected) must NOT rank as the
    'cheapest' value. Non-positive valuation multiples are cleaned to NaN before scoring,
    so a loss-maker is UNSCORED on the P/E axis, not treated as an infinitely-low multiple
    that its low-is-good direction would flip into the best score."""
    idx = ["LOSS", "CHEAP", "RICH"]
    df = _base(idx)
    df["sector"] = ["Industrials"] * 3  # Group A value recipe -> pe_forward is scored
    df["pe_forward"] = [-10.0, 5.0, 50.0]
    s = compute_scores(df, sector_neutral=False)
    assert pd.isna(s.loc["LOSS", "value_z"])  # negative fwd P/E -> unscored, NOT best
    assert s.loc["CHEAP", "value_z"] > s.loc["RICH", "value_z"]


def test_forward_loss_is_ineligible_value_trap():
    """Owner 2026-07-23: profitable trailing but NON-positive forward earnings (PET>0,
    PEF<=0) is the extreme value-trap the PEF/PET ratio can't express (ratio = NaN) ->
    INELIGIBLE. A both-positive forward-cheaper name stays eligible."""
    idx = ["GOOD", "FWDLOSS"]
    df = _base(idx)
    df["quote_type"] = ["EQUITY", "EQUITY"]
    df["price"] = [100.0, 100.0]
    df["roe"] = [20.0, 20.0]
    df["pct_52w_high"] = [0.8, 0.8]
    df["mom_12_1"] = [0.1, 0.1]
    df["realized_vol"] = [0.2, 0.2]
    df["pe_trailing"] = [20.0, 20.0]
    df["pe_forward"] = [15.0, -5.0]  # GOOD: forward cheaper; FWDLOSS: forward loss
    s = compute_scores(df, sector_neutral=False)
    assert bool(s.loc["GOOD", "eligible"]) is True
    assert bool(s.loc["FWDLOSS", "eligible"]) is False


def test_eligibility_excludes_non_equity_and_dataless():
    """With quote_type present, ETFs and dataless names drop out of the ranking."""
    idx = ["EQ1", "EQ2", "ETF1", "DATALESS"]
    df = _base(idx)
    df["quote_type"] = ["EQUITY", "EQUITY", "ETF", "EQUITY"]
    df["price"] = [100.0, 50.0, 200.0, 30.0]
    df["pe_forward"] = [10.0, 20.0, 15.0, 12.0]
    df["roe"] = [30.0, 20.0, 25.0, 18.0]
    df["pct_52w_high"] = [0.20, 0.10, 0.15, 0.05]
    df["realized_vol"] = [0.20, 0.25, 0.22, 0.30]
    # DATALESS: strip its core data (no price/mom/vol, and its cluster members go NaN).
    df.loc["DATALESS", ["price", "pe_forward", "roe", "pct_52w_high", "realized_vol"]] = np.nan

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
    """IC that would over-weight Value+Quality is capped at 0.60, excess to the rest."""
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
    assert abs(vq - 0.60) < 1e-9  # capped exactly at the joint cap (raised 0.55->0.60)
    # equal IC -> equal split inside the VQ bucket
    assert abs(w["value_z"] - w["quality_z"]) < 1e-9
    # the freed 0.40 is spread over the three positive-IC clusters (equal IC -> equal split)
    assert abs(w["momentum_z"] - 0.40 / 3) < 1e-9


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
    to push sum above the 0.55 cap, and the four remaining clusters (momentum,
    lowvol, strength, pead) have IC ≤ 0 so their raw weights are all zero.  The freed
    weight (1 - 0.55 = 0.45) must be spread equally across the four clusters,
    giving each 0.1125, and the VQ pair is still exactly at the 0.55 cap.
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
    assert abs(w["value_z"] + w["quality_z"] - 0.60) < 1e-9, "VQ cap not exactly 0.60"

    expected_each = (1.0 - 0.60) / 5  # rest = momentum, lowvol, strength, pead, trajectory
    for c in ("momentum_z", "lowvol_z", "strength_z", "pead_z", "trajectory_z"):
        assert abs(w[c] - expected_each) < 1e-9, (
            f"{c}: got {w[c]:.6f}, expected {expected_each:.6f}"
        )


def test_ic_weighting_shifts_conviction_toward_high_ic_cluster():
    """IC weighting shifts the ranking: a value-heavy IC ranks the value name first,
    a momentum-heavy IC flips it to the momentum name. (Default weights are now
    equal-risk, so the shift is shown via explicit IC vectors, not the default.)"""
    df = _base(["MOM_STAR", "VAL_STAR", "MID1", "MID2"])
    df["pe_forward"] = [50.0, 5.0, 20.0, 25.0]  # VAL_STAR cheapest -> best value
    df["pct_52w_high"] = [0.50, -0.20, 0.10, 0.05]  # MOM_STAR strongest momentum

    val = compute_scores(
        df,
        sector_neutral=False,
        ic_weights={
            "value_z": 1.0,
            "quality_z": 0.05,
            "momentum_z": 0.05,
            "lowvol_z": 0.05,
            "strength_z": 0.05,
        },
    )
    # Value-heavy IC -> the value name wins.
    assert val.loc["VAL_STAR", "rank"] < val.loc["MOM_STAR", "rank"]

    mom = compute_scores(
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
    assert mom.loc["MOM_STAR", "rank"] < mom.loc["VAL_STAR", "rank"]


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
    """Growth cluster: earnings growth + earnings stability (2026-07-21: rev_growth demoted
    to watch/margin-conditioning; earn_stability added as the QMJ 'safety' leg). DIRECTION +1."""
    from trade_modules.v3.combine import CLUSTERS, DIRECTION

    assert "growth_z" in CLUSTERS
    assert set(CLUSTERS["growth_z"]) == {"earn_growth", "earn_stability"}
    assert DIRECTION["earn_growth"] == +1
    assert DIRECTION["earn_stability"] == +1
    assert list(CLUSTERS.keys()) == [
        "value_z",
        "quality_z",
        "momentum_z",
        "growth_z",
        "lowvol_z",
        "strength_z",
        "pead_z",
        "trajectory_z",
    ]


def test_trajectory_cluster_registered_direction_negative():
    """Earnings-trajectory (PEF/PET, owner 2026-07-23): LOW ratio = forward cheaper =
    earnings expected to rise, DIRECTION -1 (smaller is better). Inverted from the old
    PET/PEF so "< 1 is good"; strongest survivorship-clean signal (beta-neutral t 2.17)."""
    from trade_modules.v3.combine import CLUSTERS, DIRECTION

    assert CLUSTERS["trajectory_z"] == ["earn_trajectory"]
    assert DIRECTION["earn_trajectory"] == -1


def test_trajectory_z_rising_earnings_ranks_above_falling():
    """Low PEF/PET (earnings rising, forward cheaper) ranks above a falling-earnings peer."""
    df = _base(["RISING", "FLAT", "FALLING"])
    df["earn_trajectory"] = [0.5, 1.0, 2.0]  # PEF/PET: <1 rising (fwd cheaper), >1 falling
    scores = compute_scores(df, sector_neutral=False)
    assert scores.loc["RISING", "trajectory_z"] > scores.loc["FLAT", "trajectory_z"]
    assert scores.loc["FLAT", "trajectory_z"] > scores.loc["FALLING", "trajectory_z"]


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


# (removed 2026-07-18, D11/D14) test_default_growth_weight_is_zero — growth is now a
# weighted satellite (not zero). See test_fixnow_equal_weight_core_no_vq_floor.


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
    df["pe_forward"] = [50.0, 5.0, 20.0]
    df["pct_52w_high"] = [0.50, -0.20, 0.10]
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
    df["pe_forward"] = [60.0, 6.0, 20.0]  # GROWTH_EXP expensive, VALUE_CHEAP cheap
    df["earn_growth"] = [40.0, 1.0, 15.0]  # GROWTH_EXP fast grower, VALUE_CHEAP flat
    df["earn_stability"] = [0.9, 0.2, 0.5]  # GROWTH_EXP steadier earnings too
    df["pct_52w_high"] = [0.40, -0.10, 0.10]  # momentum reinforces the growth name

    vh = compute_scores(df, sector_neutral=False, cluster_weights=_VALUE_HEAVY)
    gf = compute_scores(df, sector_neutral=False, cluster_weights=_GROWTH_FORWARD)

    # Value-heavy: the cheap no-growth name wins (value dominates, growth off).
    assert vh.loc["VALUE_CHEAP", "rank"] < vh.loc["GROWTH_EXP", "rank"]
    # Growth-forward: the expensive fast-grower overtakes the cheap flat name.
    assert gf.loc["GROWTH_EXP", "rank"] < gf.loc["VALUE_CHEAP", "rank"]


def test_backward_compat_none_equals_explicit_default_weights():
    """cluster_weights=None reproduces the default model EXACTLY (allclose)."""
    df = _base(["A", "B", "C", "D"])
    df["pe_forward"] = [5.0, 12.0, 25.0, 60.0]
    df["roe"] = [40.0, 25.0, 15.0, 5.0]
    df["pct_52w_high"] = [0.30, 0.10, -0.05, -0.20]

    from trade_modules.v3.combine import CLUSTER_WEIGHTS

    default = compute_scores(df, sector_neutral=False)
    explicit = compute_scores(
        df,
        sector_neutral=False,
        # the default expressed with short keys — reproduces None exactly (drift-proof).
        cluster_weights={c[:-2]: w for c, w in CLUSTER_WEIGHTS.items()},
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


# (removed 2026-07-18, D11/D14) test_backward_compat_default_ignores_growth_data —
# growth now carries a satellite weight, so growth data DOES affect conviction.


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


def test_value_cluster_is_ps_forward_pb():
    """2026-07-21 (owner): value = P/S (sector default) + forward P/E + P/B (financials/REITs
    recipe); pe_trailing + ev_ebitda dropped from scoring (redundant / dead this regime)."""
    from trade_modules.v3.combine import CLUSTERS

    assert CLUSTERS["value_z"] == ["ps_sector", "pe_forward", "pb"]
    for dead in ("pe_trailing", "ev_ebitda"):
        for members in CLUSTERS.values():
            assert dead not in members, f"{dead} must not be scored in any cluster"


def test_quality_is_roe_fcf_gp_assets():
    """Quality = ROE + FCF + GP/assets (Novy-Marx anchor added 2026-07-19 once the
    Sharadar SF1 PIT data was in). Per-sales margins + roa stay retired; current_ratio
    + de are a distress filter; accruals shadow-only (no clean alpha)."""
    from trade_modules.v3.combine import CLUSTERS, DIRECTION

    assert CLUSTERS["quality_z"] == ["roe", "fcf", "gp_assets", "net_issuance"]
    assert DIRECTION["gp_assets"] == +1
    assert DIRECTION["net_issuance"] == -1  # dilution bad / buyback good
    for dead in ("roa", "gross_margin", "op_margin", "current_ratio", "de", "accruals"):
        for members in CLUSTERS.values():
            assert dead not in members, f"{dead} must not be scored in quality"


def test_master_taxonomy_weights_2026_07_23():
    """Owner taxonomy (2026-07-23), DERIVED from METRIC_WEIGHTS: quality-led 0.25 (ROE the
    0.10 lead), SUE + PEF/PET the raised expectation signals (0.15 each), analyst_mom /
    momentum / growth 0.10, value 0.15, low-vol -> 0 (sizing). V+Q within the 0.60 cap; Σ=1."""
    from trade_modules.v3.combine import CLUSTER_WEIGHTS as w

    assert w["quality_z"] == max(w.values()), "quality is the evidence leader"
    assert w["quality_z"] == pytest.approx(0.25)
    assert w["pead_z"] == pytest.approx(0.15) and w["trajectory_z"] == pytest.approx(0.15)
    assert w["momentum_z"] == pytest.approx(0.10) and w["strength_z"] == pytest.approx(0.10)
    assert w["growth_z"] == pytest.approx(0.10) and w["value_z"] == pytest.approx(0.15)
    assert w["lowvol_z"] == 0.0, "low-vol -> 0 (moved to sizing, not alpha)"
    assert (w["value_z"] + w["quality_z"]) <= 0.60 + 1e-9, "value+quality within the 0.60 cap"
    assert sum(w.values()) == pytest.approx(1.0)
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_fixnow_equal_vol_standardize_gives_unit_std_columns():
    """Equal-VOL: each cluster column is scaled to ~unit cross-sectional std, so equal
    nominal weights are equal-RISK (a low-dispersion cluster no longer under-contributes).
    Contemporaneous / per-snapshot -> look-ahead-free (D14 refinement)."""
    from trade_modules.v3.combine import _equal_vol_standardize

    zmat = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]}  # b has 10x the spread
    )
    out = _equal_vol_standardize(zmat)
    assert abs(float(out["a"].std(ddof=0)) - 1.0) < 1e-9
    assert abs(float(out["b"].std(ddof=0)) - 1.0) < 1e-9


def test_fixnow_equal_vol_standardize_is_nan_and_zero_std_safe():
    """A NaN-bearing column keeps its NaNs; a constant (zero-std) column is left
    unchanged (no divide-by-zero)."""
    from trade_modules.v3.combine import _equal_vol_standardize

    zmat = pd.DataFrame(
        {
            "with_nan": [1.0, np.nan, 3.0, 5.0],
            "constant": [2.0, 2.0, 2.0, 2.0],  # zero std -> unchanged
        }
    )
    out = _equal_vol_standardize(zmat)
    assert pd.isna(out.loc[1, "with_nan"])
    assert abs(float(out["with_nan"].std(ddof=0)) - 1.0) < 1e-9
    assert list(out["constant"]) == [2.0, 2.0, 2.0, 2.0]
