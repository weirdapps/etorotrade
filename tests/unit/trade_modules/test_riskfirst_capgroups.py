"""Tests for riskfirst.construct.cap_groups — a generic aggregate-group cap.

Generalises the FX-bloc cap to ANY grouping (sector, cluster, region). Sector
concentration control plugs in here the moment a sector label is available.
"""

import numpy as np
import pytest

from trade_modules.riskfirst.construct import cap_groups


def test_caps_over_limit_group_and_redistributes():
    w = cap_groups(np.array([0.3, 0.3, 0.2, 0.2]), np.array(["A", "A", "B", "B"]), cap=0.5)
    assert w == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert w.sum() == pytest.approx(1.0)


def test_noop_when_all_groups_within_cap():
    w = cap_groups(np.array([0.2, 0.3, 0.5]), np.array(["A", "B", "C"]), cap=0.6)
    assert w == pytest.approx([0.2, 0.3, 0.5])


def test_single_group_over_cap_scales_to_cash_when_no_receiver():
    # everything in one sector, cap 0.6, no other group to receive -> holds cash
    w = cap_groups(np.array([0.5, 0.5]), np.array(["A", "A"]), cap=0.6)
    assert w.sum() == pytest.approx(0.6)
    assert w == pytest.approx([0.3, 0.3])


def test_three_sectors_one_hot():
    w = cap_groups(
        np.array([0.4, 0.2, 0.2, 0.2]),
        np.array(["tech", "tech", "energy", "health"]),
        cap=0.5,
    )
    tech = w[0] + w[1]
    assert tech <= 0.5 + 1e-9
    assert w.sum() == pytest.approx(1.0)
