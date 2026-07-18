"""FIX-NOW 2026-07-18 (D23 / spec §7): frozen acceptance gate + trial register.

One auditable place that (a) freezes the DSR/PBO/IC thresholds so they cannot be
quietly relaxed to pass a favoured signal, and (b) counts EVERY candidate signal
under test so DSR deflation sees the true number of trials (undercounting inflates
DSR). A factor earns conviction only by clearing THESE numbers on forward,
net-of-cost, ≥2-regime data. Sources: Bailey-López de Prado (DSR/PBO), Harvey-Liu-Zhu
(t>3), spec 2026-07-12 §7. Pure stdlib; no I/O.
"""

from __future__ import annotations

from types import MappingProxyType

# Frozen acceptance gate (read-only mapping — mutation raises TypeError).
ACCEPTANCE_THRESHOLDS = MappingProxyType(
    {
        "dsr_min": 0.95,  # Deflated Sharpe hurdle
        "pbo_max": 0.05,  # Probability of Backtest Overfitting (CSCV)
        "ic_mean_min": 0.02,  # mean incumbent-partialled cross-sectional IC
        "ic_t_min": 3.0,  # Harvey-Liu-Zhu multiple-testing t-hurdle
        "hit_rate_min": 0.55,
        "bh_q": 0.10,  # Benjamini-Hochberg FDR level
        "min_regimes": 2,  # must include a bear/stress regime
        "net_ir_min": 0.30,  # net of 2x modelled EUR cost
        "ablation_delta_ir_min": 0.10,  # Δ net OOS IR with vs without the signal
    }
)

# Append-only register of every candidate signal/config under test. DSR deflates for
# THIS count; candidates are registered BEFORE they accrue forward IC (spec line 165)
# so the trial count is never undercounted after the fact.
CANDIDATE_TRIALS = (
    "reinstate_peg",
    "gp_assets",
    "operating_profitability_assets",
    "accruals_pit",
    "investment_cma",
    "pead_sue",
    "analyst_eps_revision",
    "upside_orthogonalized_residual",
    "core_lowdin_orthogonalization",
    "growth_both_signs",
    "lowvol_beta_only",
    "lowvol_vol_only",
    "ic_weighting_sizing_tier",
    "fat_tail_cvar",
)


def trial_count() -> int:
    """Number of registered candidate trials — the N used for DSR deflation."""
    return len(CANDIDATE_TRIALS)


def threshold(name: str) -> float:
    """Read a frozen acceptance threshold (raises KeyError on an unknown gate)."""
    return float(ACCEPTANCE_THRESHOLDS[name])


def is_relaxation(name: str, proposed: float) -> bool:
    """True if ``proposed`` would LOOSEN the frozen gate ``name``.

    Guards against quietly relaxing the gate: ``*_min`` / ``*_regimes`` gates loosen
    when LOWERED; ``*_max`` / ``*_q`` gates loosen when RAISED.
    """
    frozen = float(ACCEPTANCE_THRESHOLDS[name])
    looser_when_lower = name.endswith("_min") or name.endswith("_regimes")
    return proposed < frozen if looser_when_lower else proposed > frozen
