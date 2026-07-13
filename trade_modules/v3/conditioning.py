"""v3 regime + Polymarket conditioning dials.

Single source of truth for deployment fractions and the two conditioning
adjustments (regime overlay + Polymarket tilt).  Imported by the v3_portfolio
runner; Polymarket seam is inert by default (``max_tilt=0``).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Regime → base deployment fraction
# ---------------------------------------------------------------------------

# Fraction of capital deployed by market regime (band 85-95%).
# This is the SINGLE SOURCE OF TRUTH — imported back into v3_portfolio.py.
DEPLOYMENT_BY_REGIME: dict[str, float] = {
    "risk_off": 0.85,
    "neutral": 0.90,
    "risk_on": 0.95,
}

_UNKNOWN_DEPLOYMENT = DEPLOYMENT_BY_REGIME["neutral"]


def regime_deployment(regime: str) -> float:
    """Return the base deployment fraction for a regime string.

    Unknown / missing regime labels fall back to neutral (0.90).

    Args:
        regime: One of ``"risk_off"``, ``"neutral"``, ``"risk_on"``.

    Returns:
        Deployment fraction in [0, 1].
    """
    return DEPLOYMENT_BY_REGIME.get(regime, _UNKNOWN_DEPLOYMENT)


# ---------------------------------------------------------------------------
# Polymarket tilt (shadow / zero-conviction seam)
# ---------------------------------------------------------------------------


def polymarket_adjustment(signal: float | None, *, max_tilt: float = 0.0) -> float:
    """Return a bounded deployment tilt from a Polymarket signal.

    Polymarket is a shadow / zero-conviction cross-repo signal.  The seam is
    INERT by default: when ``max_tilt == 0`` (the default) this function
    ALWAYS returns 0.0.  A nonzero ``max_tilt`` may only be passed after the
    signal has been validated and explicitly enabled.

    Args:
        signal: Normalised Polymarket signal in [-1, 1] (or ``None`` when
            unavailable).  Values outside [-1, 1] are clipped.
        max_tilt: Maximum absolute deployment adjustment the signal may produce.
            MUST be 0.0 (the default) while the signal is in shadow phase.

    Returns:
        Deployment tilt in ``[-max_tilt, +max_tilt]``; 0.0 when signal is
        ``None`` or ``max_tilt == 0``.
    """
    if signal is None or max_tilt == 0.0:
        return 0.0
    clipped = max(-1.0, min(1.0, float(signal)))
    return clipped * max_tilt


# ---------------------------------------------------------------------------
# Composite resolver
# ---------------------------------------------------------------------------


def resolve_deployment(
    regime: str,
    polymarket_signal: float | None = None,
    *,
    max_pm_tilt: float = 0.0,
    band: tuple[float, float] = (0.85, 0.95),
) -> tuple[float, dict]:
    """Resolve the final deployment fraction and emit a diagnostic dict.

    Combines the regime base with the (inert-by-default) Polymarket tilt and
    clamps the result to ``band``.

    Args:
        regime: Market regime string (``"risk_off"`` / ``"neutral"`` /
            ``"risk_on"``; unknown → neutral).
        polymarket_signal: Normalised Polymarket signal in [-1, 1], or ``None``.
            Ignored (zero effect) while ``max_pm_tilt == 0``.
        max_pm_tilt: Maximum absolute Polymarket tilt.  MUST remain 0.0 (the
            default) while the Polymarket signal is in shadow phase.
        band: ``(lo, hi)`` clamping band for the final deployment fraction.

    Returns:
        ``(deployment, dial_diag)`` where:

        - ``deployment`` is the clamped final fraction in ``band``.
        - ``dial_diag`` is a diagnostic dict with keys
          ``{regime, base_deployment, polymarket_signal, polymarket_tilt,
          polymarket_active, final_deployment}``.
    """
    lo, hi = band
    base = regime_deployment(regime)
    tilt = polymarket_adjustment(polymarket_signal, max_tilt=max_pm_tilt)
    deployment = max(lo, min(hi, base + tilt))

    dial_diag: dict = {
        "regime": regime,
        "base_deployment": base,
        "polymarket_signal": polymarket_signal,
        "polymarket_tilt": tilt,
        "polymarket_active": max_pm_tilt > 0.0,
        "final_deployment": deployment,
    }
    return deployment, dial_diag
