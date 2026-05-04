"""
M4: EmpiricalFactorScore — CIO v36 Empirical Refoundation.

Multi-factor BUY/SELL score using parameters that EMPIRICALLY predict T+30
alpha on this system's data (n=32,589 obs in signal_log, study output at
~/.weirdapps-trading/parameter_study_results.json):

| factor          | ρ       | sign | weight |
|-----------------|---------|------|--------|
| pct_52w_high    | +0.103  | +    | +1.00  | momentum
| short_interest  | -0.111  | -    | -0.80  | shorts know things
| roe             | +0.035  | +    | +0.30  | quality
| upside          | -0.087  | -    | -0.40  | analyst-darlings underperform

The score is a weighted sum of cross-sectional z-scores. Computed per
universe, so a stock's score is its rank position relative to the run's
peers. Score > 0 means above-average factor signature.

Used as:
1. A BUY gate: BUY signals with score ≤ 0 get demoted to HOLD.
2. A diagnostic field on every concordance row.

Coefficient signs are validated quarterly by re-running
scripts/signal_parameter_study.py. If signs flip, update FACTOR_WEIGHTS.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)


# Empirically-validated factor coefficients (signs from parameter study).
# Update from quarterly re-run of scripts/signal_parameter_study.py.
FACTOR_WEIGHTS = {
    "momentum": +1.00,  # pct_52w_high
    "short_interest": -0.80,  # high SI → demote
    "roe": +0.30,  # quality
    "upside": -0.40,  # analyst-darling premium fades
}

# Stock-data field names mapped to factor inputs (some agents use synonyms)
_FIELD_ALIASES = {
    "momentum": ("pct_52w_high", "52w", "pct_52w"),
    "short_interest": ("short_interest", "si"),
    "roe": ("roe", "return_on_equity"),
    "upside": ("upside", "up_pct"),
}


def _to_float(value: Any) -> float | None:
    """Coerce signal value to float; None on failure."""
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.strip().rstrip("%").replace(",", "")
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract(stock: dict[str, Any], factor: str) -> float | None:
    """Pull the factor input from a stock dict, honoring field aliases."""
    for key in _FIELD_ALIASES.get(factor, ()):
        if key in stock:
            v = _to_float(stock.get(key))
            if v is not None:
                return v
    return None


def _zscore(values: list[float], target: float | None) -> float:
    """z-score of target against the universe mean/std. 0 if target is None."""
    if target is None:
        return 0.0
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if len(clean) < 2:
        return 0.0
    mean = sum(clean) / len(clean)
    var = sum((v - mean) ** 2 for v in clean) / len(clean)
    if var <= 0:
        return 0.0
    sigma = math.sqrt(var)
    z = (target - mean) / sigma
    # Clamp to ±3 for stability
    return max(-3.0, min(3.0, z))


def compute_factor_score(
    stock: dict[str, Any],
    universe: Iterable[dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute z-weighted factor score for `stock` within `universe`.

    Returns:
        {"score": float, "components": {factor: z}, "verdict": "ABOVE"|"BELOW"|"NEUTRAL"}
    """
    weights = weights or FACTOR_WEIGHTS
    universe_list = list(universe)
    components: dict[str, float] = {}
    score = 0.0
    for factor, weight in weights.items():
        target = _extract(stock, factor)
        all_values = [_extract(s, factor) for s in universe_list]
        z = _zscore([v for v in all_values if v is not None], target)
        components[factor] = round(z, 3)
        score += weight * z
    score = round(score, 3)
    verdict = "ABOVE" if score > 0.05 else ("BELOW" if score < -0.05 else "NEUTRAL")
    return {"score": score, "components": components, "verdict": verdict}


def passes_buy_gate(
    factor_result: dict[str, Any],
    threshold: float = 0.0,
) -> bool:
    """True if factor_score is strictly greater than threshold."""
    score = factor_result.get("score")
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return False
    return float(score) > threshold


def factor_correlation_matrix(
    universe: Iterable[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Pairwise Pearson correlation matrix among the M4 factor inputs.

    Used as a multicollinearity diagnostic. If two factors correlate too
    highly, the weighted-sum double-counts the same underlying signal.
    Returns dict {factor_name: {factor_name: ρ}} with diagonals = 1.0.
    """
    universe_list = list(universe)
    factors = list(FACTOR_WEIGHTS.keys())

    # Extract per-factor value vectors
    vectors: dict[str, list[float]] = {}
    for f in factors:
        col = []
        for s in universe_list:
            v = _extract(s, f)
            col.append(v if v is not None else float("nan"))
        vectors[f] = col

    matrix: dict[str, dict[str, float]] = {}
    for a in factors:
        matrix[a] = {}
        for b in factors:
            matrix[a][b] = _pearson(vectors[a], vectors[b])
    return matrix


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation, treating NaNs by pairwise deletion."""
    pairs = [(x, y) for x, y in zip(xs, ys, strict=False) if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 3:
        return 0.0
    cx = [p[0] for p in pairs]
    cy = [p[1] for p in pairs]
    n = len(pairs)
    mx, my = sum(cx) / n, sum(cy) / n
    cov = sum((cx[i] - mx) * (cy[i] - my) for i in range(n)) / n
    vx = sum((v - mx) ** 2 for v in cx) / n
    vy = sum((v - my) ** 2 for v in cy) / n
    if vx <= 0 or vy <= 0:
        return 0.0
    return round(cov / math.sqrt(vx * vy), 4)


def flag_redundant_factor_pairs(
    universe: Iterable[dict[str, Any]],
    threshold: float = 0.7,
) -> list[dict[str, Any]]:
    """Return list of factor pairs whose |correlation| exceeds threshold.

    Output: [{factor_a, factor_b, correlation, recommend}, ...]
    Operator review: if two factors correlate >0.7 in your data, consider
    dropping one or replacing with a residual / orthogonalized version.
    """
    matrix = factor_correlation_matrix(universe)
    factors = list(matrix.keys())
    flagged = []
    for i, a in enumerate(factors):
        for b in factors[i + 1 :]:
            corr = matrix[a][b]
            if abs(corr) >= threshold:
                flagged.append(
                    {
                        "factor_a": a,
                        "factor_b": b,
                        "correlation": corr,
                        "recommend": (
                            "Drop one or orthogonalize via Gram-Schmidt; "
                            "current weighted-sum is double-counting this signal"
                        ),
                    }
                )
    return flagged


def apply_empirical_gate(
    concordance: list[dict[str, Any]],
    threshold: float = 0.0,
) -> int:
    """Apply the empirical-factor BUY gate to a concordance list.

    Mutates each row to add `empirical_score`, `empirical_components`,
    `empirical_verdict`. For rows whose action is BUY/ADD and score ≤
    threshold, demotes the action to HOLD and sets
    `empirical_gate_demoted: True`.

    Returns the number of demotions applied so callers can audit.
    """
    if not concordance:
        return 0

    universe = [row for row in concordance if isinstance(row, dict)]
    demotions = 0
    for entry in concordance:
        if not isinstance(entry, dict):
            continue
        result = compute_factor_score(entry, universe)
        entry["empirical_score"] = result["score"]
        entry["empirical_components"] = result["components"]
        entry["empirical_verdict"] = result["verdict"]

        if entry.get("action") not in ("BUY", "ADD"):
            continue
        if not passes_buy_gate(result, threshold=threshold):
            entry["empirical_gate_demoted"] = True
            entry["original_action"] = entry.get("action")
            entry["action"] = "HOLD"
            demotions += 1

    if demotions:
        logger.info(
            "Empirical-factor gate demoted %d BUY/ADD → HOLD (threshold=%.2f)",
            demotions,
            threshold,
        )
    return demotions
