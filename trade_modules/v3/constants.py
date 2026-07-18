"""Trading Model v3 — declared model constants (single source of truth).

The signal horizon is the ONE number the whole validation stack must agree on:
the IC logger (``scripts/v3_ic_logger.py``), the price-spine gate
(``scripts/v3_spine_gate.py``), and the referee (``trade_modules/validation/
harness.py`` via ``trade_modules/v3/validate_spine.py``) all grade the model at
``V3_SIGNAL_HORIZON`` so every DSR / IC statistic measures the SAME target.

Owner decision 2026-07-18: primary = 21 trading days (~1 month). The equal-risk
blend is ~74% slow signals (value/quality/low-vol/growth) + ~26% fast (momentum,
analyst strength), but the no-trade-band construction implies a ~monthly holding
period and the fastest-decaying sleeves need a ~1mo read; value/quality are still
co-reported at 63. Sources: Grinold-Kahn (grade IC at the rebalance horizon);
Jegadeesh-Titman (momentum 1-12mo).
"""

from __future__ import annotations

# Primary signal horizon in trading-day steps (== IC-logger log-date steps on a
# daily schedule). Everything that grades the model grades at THIS horizon.
V3_SIGNAL_HORIZON = 21

# Unified IC / gate measurement grid: fast diagnostic (5), primary (21), slow
# value/quality secondary (63). V3_SIGNAL_HORIZON MUST be a member of this grid.
V3_IC_HORIZONS = (5, 21, 63)
