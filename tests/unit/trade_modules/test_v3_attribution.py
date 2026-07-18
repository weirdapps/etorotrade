"""TDD tests for trade_modules.v3.attribution — FIX-NOW (D21): measure the owner
core sleeve (de-launder it) + the 3-layer attribution taxonomy classifier."""

import numpy as np
import pandas as pd
import pytest

from trade_modules.v3.attribution import attribution_layer, core_attribution


def test_core_attribution_variance_share_exceeds_weight_for_hot_core():
    """The core's share of BOOK VARIANCE can far exceed its weight — exactly the
    'the AI core dominates book risk' fact the self-review demanded be measured."""
    w = pd.Series({"CORE": 0.5, "OTHER": 0.5})
    cov = np.array([[0.16, 0.0], [0.0, 0.04]])  # core carries 4x the variance
    out = core_attribution(w, cov, ["CORE"], betas=[1.4, 0.8])
    assert out["core_weight"] == pytest.approx(0.5)
    assert out["core_variance_share"] == pytest.approx(0.8)  # 50% of weight, 80% of risk
    assert out["n_core_held"] == 1
    assert out["book_net_beta"] == pytest.approx(0.5 * 1.4 + 0.5 * 0.8)
    assert out["core_net_beta_contribution"] == pytest.approx(0.5 * 1.4)


def test_core_attribution_all_cash_is_zero_safe():
    """An empty / all-cash book returns zeros, never divides by zero."""
    out = core_attribution(pd.Series(dtype=float), np.zeros((0, 0)), ["CORE"])
    assert out["core_weight"] == 0.0
    assert out["core_variance_share"] == 0.0
    assert out["n_core_held"] == 0


def test_attribution_layer_classifies_saa_forecast_and_override():
    """3-layer taxonomy at the ticker level: managed sleeves = SAA (owned policy);
    scored equities = forecast (IC-gated); protect_core/core_floor names carry an
    owner_override flag (measured discretion, not process)."""
    managed = ["GLD", "UVXY", "LYXGRE.DE"]
    core = ["NVDA", "MSFT"]
    saa = attribution_layer("GLD", managed_sleeves=managed, core_list=core)
    assert saa["layer"] == "SAA" and saa["owner_override"] is False
    fc = attribution_layer("AAPL", managed_sleeves=managed, core_list=core)
    assert fc["layer"] == "forecast" and fc["owner_override"] is False
    ovr = attribution_layer("NVDA", managed_sleeves=managed, core_list=core)
    assert ovr["layer"] == "forecast" and ovr["owner_override"] is True
