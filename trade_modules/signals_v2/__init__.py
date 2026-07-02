"""signals_v2 — pure signal-composite module for the S2 signal rebuild.

Builds on the already-tested riskfirst factor primitives; maps composite
scores to B/H/S signals by explicit percentile thresholds (to be calibrated
by S0 referee evidence).

Public API:
    from trade_modules.signals_v2.composite import (
        factor_composite,
        map_to_signal,
        price_sleeve_signal,
    )
"""
