"""Adaptive cluster-weight proposal — Bayesian shrinkage from the best-bet prior
toward measured forward IC as evidence accrues (proposes; does NOT auto-impose).

The best-bet prior (``combine.CLUSTER_WEIGHTS``) is where we start while forward IC
is thin. Each cycle this proposes new weights = a confidence-weighted blend of the
prior and the IC-implied weights, where confidence ``κ = n / (n + N0)`` grows with the
number of forward observations. At n=0 the proposal equals the prior; as data
accumulates it drifts toward what the data actually pays for.

Guardrails: (a) durable-core FLOORS so a regime dip can't zero a decades-premium;
(b) a per-factor CAP; (c) a DSR gate — a factor may only be raised ABOVE its prior
once it has cleared the deflated-Sharpe referee (``promoted`` set); un-promoted
factors can be trimmed but not over-promoted on a lucky-but-unproven IC.

This module only COMPUTES a proposal. Applying it is a human decision (governance):
the proposal is logged and shown; weights change only on approval.
"""

from __future__ import annotations

N0_DEFAULT = 42  # months of forward IC at which the data gets ~half the say
CAP_DEFAULT = 0.30
# Durable factors keep a floor even through a regime dip (don't chase-drop a premium).
FLOORS_DEFAULT = {"value_z": 0.10, "quality_z": 0.10, "momentum_z": 0.10}


def _kappa(n: float, n0: float) -> float:
    n = max(float(n or 0.0), 0.0)
    return n / (n + n0) if n > 0 else 0.0


def propose_weights(
    prior: dict[str, float],
    ic: dict[str, float],
    n: dict[str, float],
    *,
    n0: float = N0_DEFAULT,
    cap: float = CAP_DEFAULT,
    floors: dict[str, float] | None = None,
    promoted: set[str] | None = None,
) -> dict[str, float]:
    """Confidence-weighted blend of ``prior`` and the IC-implied weights.

    Args:
        prior: the best-bet weights (sum ~1).
        ic: measured forward IC per cluster (may be negative / missing).
        n: number of forward observations per cluster (drives confidence).
        n0: prior strength (months); larger = trust the prior longer.
        cap: max weight per cluster; floors: per-cluster minimums.
        promoted: clusters that cleared the DSR gate (only these may exceed their prior).

    Returns proposed weights, non-negative, floored/capped, renormalized to sum 1.
    At n=0 everywhere this returns the (renormalized) prior unchanged.
    """
    clusters = list(prior)
    floors = floors or FLOORS_DEFAULT
    promoted = promoted or set()

    ic_pos = {c: max(float(ic.get(c, 0.0) or 0.0), 0.0) for c in clusters}
    ic_total = sum(ic_pos.values())
    ic_w = {c: (ic_pos[c] / ic_total if ic_total > 0 else prior[c]) for c in clusters}

    blended: dict[str, float] = {}
    for c in clusters:
        k = _kappa(n.get(c, 0.0), n0)
        w = (1.0 - k) * prior[c] + k * ic_w[c]
        if c not in promoted:  # DSR gate: un-proven factors can trim, not over-promote
            w = min(w, prior[c])
        w = min(max(w, floors.get(c, 0.0)), cap)
        blended[c] = w

    total = sum(blended.values())
    return {c: blended[c] / total for c in clusters} if total > 0 else dict(prior)


def proposal_table(prior: dict[str, float], proposed: dict[str, float]) -> list[dict]:
    """Rows [{cluster, prior, proposed, delta}] sorted by |delta| desc — for the report."""
    rows = [
        {
            "cluster": c,
            "prior": prior[c],
            "proposed": proposed.get(c, 0.0),
            "delta": proposed.get(c, 0.0) - prior[c],
        }
        for c in prior
    ]
    return sorted(rows, key=lambda r: abs(r["delta"]), reverse=True)
