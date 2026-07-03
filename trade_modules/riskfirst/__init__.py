"""riskfirst — a risk-first, factor-based portfolio engine (SHADOW).

A clean rebuild of the selection/sizing stack per the 2026-07 senior-PM review:
- 5 theory-justified factors (Value, Quality, Momentum[vol-scaled], LowVol, Size),
  each a winsorized cross-sectional z-score (higher = more attractive);
- vol-targeted Equal-Risk-Contribution (ERC) construction;
- binding constraints: single-name / sector / cluster / net-USD caps, min names, cash floor;
- an edge-gate (walk-forward + Deflated Sharpe + PBO) that must clear before any
  un-clamping of sizing.

Runs in SHADOW mode: it emits recommendations and an edge verdict; it does NOT
drive autonomous trades. Promotion to live sizing is gated on the edge test.
"""
