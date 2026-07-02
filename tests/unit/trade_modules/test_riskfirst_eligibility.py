"""Tests for riskfirst.engine.eligible_universe — the investability gate.

Without this, the size factor pulls the book into illiquid micro-caps with no
data. We require a minimum market cap AND a minimum number of present fundamentals
before a name can be scored.
"""

import numpy as np
import pandas as pd

from trade_modules.riskfirst.engine import eligible_universe


def _df():
    return pd.DataFrame(
        {
            "CAP": ["500B", "50M", "10B"],
            "ROE": [20.0, np.nan, np.nan],
            "PET": [15.0, np.nan, np.nan],
            "PEF": [14.0, np.nan, np.nan],
            "FCF": [5.0, np.nan, np.nan],
        },
        index=["BIG", "MICRO", "MIDNODATA"],
    )


def test_filters_out_small_cap_and_dataless():
    out = eligible_universe(_df(), min_cap=2e9, min_factors=3)
    assert list(out.index) == ["BIG"]


def test_keeps_large_cap_with_enough_data():
    out = eligible_universe(_df(), min_cap=2e9, min_factors=3)
    assert "BIG" in out.index
    assert "MICRO" not in out.index  # too small
    assert "MIDNODATA" not in out.index  # big enough but no fundamentals
