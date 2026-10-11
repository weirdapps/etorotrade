"""Restate the AM analyst-count change as a buy-share change in percentage points.

``AM`` is a signed COUNT since 2026-07-30: the net change in buy-rated analysts
(strongBuy+buy) over ~3 months (``recommendations_parser.calculate_analyst_momentum``).
Every AM threshold in this repo (``config.yaml`` ``max_analyst_momentum`` /
``min_analyst_momentum``, the score interpolation knots, the ``-5`` guards) was written for
the metric it replaced: the 3-month change in buy-% in PERCENTAGE POINTS. Compared against a
count, the ``-20`` sell trigger could never fire (live minimum -19) and the ``-5`` buy filter
meant five analysts whatever the coverage.

``am_share_pp`` divides the count by coverage, the analyst-opinion count (``#T``, internal
``analyst_count``), and clips to the old metric's [-100, +100] range. Measured on the
2026-10-11 etoro.csv against the last pp-scale file (2026-07-30 12:01 UTC, after the
period-label fixes), the share of rows below each threshold matches: -20pp 4.3% vs 3.9%,
-10pp 9.8% vs 8.8%, -5pp 15.3% vs 14.9%. The display column stays a count.

Unknown is not zero: a missing count or a missing or non-positive coverage yields NaN, which
every caller already treats as "no AM signal".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AM_SHARE_CLIP_PP = 100.0


def am_share_pp(am_count, coverage) -> float:
    """Net buy-rated analyst change as percentage points of coverage, NaN when unknown."""
    try:
        am = float(am_count)
        cov = float(coverage)
    except (TypeError, ValueError):
        return float("nan")
    if np.isnan(am) or np.isnan(cov) or cov <= 0:
        return float("nan")
    return float(np.clip(100.0 * am / cov, -AM_SHARE_CLIP_PP, AM_SHARE_CLIP_PP))


def am_share_pp_series(am_count: pd.Series, coverage: pd.Series) -> pd.Series:
    """Vectorised ``am_share_pp`` over aligned Series."""
    am = pd.to_numeric(am_count, errors="coerce")
    cov = pd.to_numeric(coverage, errors="coerce")
    cov = cov.where(cov > 0)
    return (100.0 * am / cov).clip(-AM_SHARE_CLIP_PP, AM_SHARE_CLIP_PP)
