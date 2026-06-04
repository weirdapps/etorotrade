"""
Modifier Governance Policy (CIO v44.0)

Codifies the rules for promoting, demoting, and auditing conviction
modifiers in the committee synthesis pipeline.

RULE: No modifier enters ACTIVE_MODIFIERS without ALL FOUR criteria:

1. SIGNIFICANCE — Benjamini-Hochberg significance at alpha=0.05 across
   all tested modifiers (not per-modifier p<0.05, which is uncorrected).
   Source: scripts/calibrate_modifiers_t30.py + backtest_stats.fdr_correction.

2. OUT-OF-SAMPLE — Evidence must come from data NOT used to discover
   the modifier. Forward-recorded signal_log observations accumulated
   AFTER the modifier was coded count as OOS. Same-sample discoveries
   are in-sample regardless of CI width.

3. NET-OF-COST — Alpha must survive round-trip transaction costs at
   the operating horizon (T+30 minimum). Use committee_backtester's
   _round_trip_cost_pct for tier-aware eToro spread estimation.

4. CORRECT SIGN — The modifier's code direction (bonus vs penalty) must
   match the calibrated Spearman rho sign. A modifier with rho < 0 that
   is applied as a positive bonus is SIGN-INVERTED and must be fixed or
   removed, regardless of significance.

DEFAULT STATE
    SHADOW — tracked in the conviction waterfall (prefixed with ~),
    does not affect the final conviction score. This is the starting
    state for every new modifier idea.

PROMOTION
    SHADOW -> ACTIVE when all four criteria are met in the most recent
    calibration run. The calibrator (scripts/calibrate_modifiers_t30.py)
    enforces criteria 1-2. The operator reviews criteria 3-4.

DEMOTION
    Automatic when evidence expires:
    - Calibration file older than 90 days -> all modifiers demoted to SHADOW
    - Modifier loses BH significance in a new calibration -> demoted
    - Sign changes between calibration runs -> demoted + flagged

AUDIT CADENCE
    Weekly via scripts/calibrate_modifiers_t30.py (triggered by
    weekly-backtest.yml GitHub Action). The calibration produces:
    - modifier_t30_calibration.json (standard, p<0.05)
    - modifier_t30_calibration_rigorous.json (Bonferroni-corrected)
    Only the rigorous file gates ACTIVE_MODIFIERS promotions.

REFERENCES
    - committee_synthesis.py: ACTIVE_MODIFIERS set (the live gate)
    - committee_synthesis.py: filter_waterfall() (the shadow mechanism)
    - backtest_stats.py: fdr_correction() (BH implementation)
    - Harvey, Liu, Zhu (2016) "...and the Cross-Section of Expected
      Returns" — the multiple-testing framework this policy implements
"""

GOVERNANCE_VERSION = "v44.0"

PROMOTION_CRITERIA = {
    "bh_significance": "BH-corrected p < 0.05 across all tested modifiers",
    "out_of_sample": "Evidence from data accumulated AFTER modifier was coded",
    "net_of_cost": "T+30 alpha survives eToro tier round-trip spread",
    "correct_sign": "Code direction matches calibrated Spearman rho sign",
}

DEFAULT_STATE = "SHADOW"
CALIBRATION_EXPIRY_DAYS = 90
