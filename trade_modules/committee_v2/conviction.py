"""conviction.py — evidence-based conviction scoring (0-100).

score_conviction(row, weights=None) -> float

Component mapping (each normalized to 0-100 before weighting):
  composite   composite_pct [0.0, 1.0]  → linear * 100
              missing/None              → 50 (neutral)
  upside      UP% string e.g. "30.7%"  → clamp(v, 0, 100)  (0% → 0, ≥100% → 100)
              missing/None              → 50 (neutral)
  buy_cons    %B string e.g. "69%"      → clamp(v, 0, 100)
              missing/None              → 50 (neutral)
  quality     ROE>0 AND FCF>0           → 100; one positive → 75; both <=0 → 0; missing → 50
              (binary quality read — monotonic: more positive = higher)

Default weights (must sum to 1.0):
  composite   0.40
  upside      0.25
  buy_cons    0.20
  quality     0.15
"""

from __future__ import annotations

_DEFAULT_WEIGHTS = {
    "composite": 0.40,
    "upside": 0.25,
    "buy_consensus": 0.20,
    "quality": 0.15,
}


def _parse_pct(value: object) -> float | None:
    """Parse a percentage string like '30.7%' or '-5%' → float value.
    Returns None if unparseable or None/empty input.
    """
    if value is None:
        return None
    s = str(value).strip().rstrip("%").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _composite_component(row: dict) -> float:
    """composite_pct [0, 1] → normalized [0, 100]. None → 50."""
    val = row.get("composite_pct")
    if val is None:
        return 50.0
    try:
        v = float(val)
    except (TypeError, ValueError):
        return 50.0
    return max(0.0, min(100.0, v * 100.0))


def _upside_component(row: dict) -> float:
    """UP% string → normalized [0, 100]. None/missing → 50."""
    raw = row.get("UP%")
    v = _parse_pct(raw)
    if v is None:
        return 50.0
    return max(0.0, min(100.0, v))


def _buy_consensus_component(row: dict) -> float:
    """%B string → normalized [0, 100]. None/missing → 50."""
    raw = row.get("%B")
    v = _parse_pct(raw)
    if v is None:
        return 50.0
    return max(0.0, min(100.0, v))


def _quality_component(row: dict) -> float:
    """Quality gate: ROE>0 AND FCF>0 → 100; one positive → 75; both <=0 → 25; missing → 50.

    Monotonic: more positive quality indicators = higher score.
    Scale: both positive=100, one positive=75, missing=50, both non-positive=0.
    """
    roe_raw = row.get("ROE")
    fcf_raw = row.get("FCF")
    roe_v = _parse_pct(roe_raw)
    fcf_v = _parse_pct(fcf_raw)

    if roe_v is None and fcf_v is None:
        return 50.0

    roe_pos = roe_v is not None and roe_v > 0
    fcf_pos = fcf_v is not None and fcf_v > 0

    if roe_pos and fcf_pos:
        return 100.0
    if roe_pos or fcf_pos:
        return 75.0
    # Both present but non-positive → minimum (0 allows the overall score to hit 0)
    return 0.0


def score_conviction(row: dict, weights: dict | None = None) -> float:
    """Compute conviction score for a candidate row.

    Args:
        row:     Dict with signal fields (any missing → neutral 50 for that component).
        weights: Optional override dict with keys composite, upside, buy_consensus, quality.
                 Values must be non-negative and sum to 1.0.

    Returns:
        float in [0.0, 100.0].
    """
    w = {**_DEFAULT_WEIGHTS, **(weights or {})}

    composite_score = _composite_component(row)
    upside_score = _upside_component(row)
    buy_score = _buy_consensus_component(row)
    quality_score = _quality_component(row)

    weighted = (
        w["composite"] * composite_score
        + w["upside"] * upside_score
        + w["buy_consensus"] * buy_score
        + w["quality"] * quality_score
    )

    return max(0.0, min(100.0, weighted))
