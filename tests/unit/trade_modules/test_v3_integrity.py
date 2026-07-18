"""TDD — BUILD ② (2026-07-18): fail-loud integrity gate for the etoro.csv panel.

The v3 loaders previously did a bare ``pd.read_csv`` with no schema/dtype check,
so a silently dropped column or a same-field-count column-shift degraded every
downstream factor to NaN with no signal. This gate ASSERTS the panel's shape.

Scoping the gate deliberately guards CORRUPTION, not COVERAGE:
  - missing required column / non-numeric data in a numeric factor column -> RAISE
  - a legitimately non-numeric suffixed column (CAP='39.2T') -> never checked
  - a sparse or entirely-empty column (coverage) -> tolerated (that is ③/④'s job)
The panel is clean today; this is a forward guard.
"""

from __future__ import annotations

import pandas as pd
import pytest

from trade_modules.v3.integrity import PanelIntegrityError, validate_panel


def _good_panel(n: int = 6) -> pd.DataFrame:
    """A well-formed panel: every required column present + an extra column."""
    return pd.DataFrame(
        {
            "TKR": [f"T{i}" for i in range(n)],
            "NAME": [f"Name {i}" for i in range(n)],
            "CAP": ["39.2T", "1.2B", "500M", "2.3B", "10B", "900M"][:n],  # suffixed
            "PRC": [10.0, 20.5, 3.1, 100.0, 55.5, 7.7][:n],
            "PET": [16.4, 9.8, 12.0, 22.0, 8.0, 30.0][:n],
            "ROE": [9.4, 10.2, 5.0, 15.0, 20.0, 1.0][:n],
            "FCF": ["-3.5%", "2.0%", "5.0%", "1.0%", "-1.0%", "3.0%"][:n],  # % cleans
            "52W": [93.0, 72.0, 50.0, 88.0, 99.0, 20.0][:n],
            "B": [0.3, 1.1, 0.9, 1.5, 0.7, 1.2][:n],
            "AM": [-7.0, -3.0, 2.0, 5.0, 0.0, 1.0][:n],
            "EG": [12.4, -2.9, 3.0, 8.0, -1.0, 4.0][:n],
            "EXTRA_UNKNOWN": list(range(n)),  # extra column must be tolerated
        }
    )


def test_clean_panel_passes_and_returns_same_object():
    df = _good_panel()
    assert validate_panel(df, source="test") is df  # pass-through, no copy


def test_extra_columns_tolerated():
    validate_panel(_good_panel())  # EXTRA_UNKNOWN present -> no raise


def test_missing_required_column_raises():
    df = _good_panel().drop(columns=["PRC"])
    with pytest.raises(PanelIntegrityError, match="PRC"):
        validate_panel(df)


def test_column_shift_injecting_strings_into_numeric_raises():
    df = _good_panel()
    # Simulate a shift: company names land in the numeric PRC column.
    df["PRC"] = ["Toyota", "Mitsubishi", "Apple", "Sony", "Nintendo", "Honda"][: len(df)]
    with pytest.raises(PanelIntegrityError, match="PRC"):
        validate_panel(df)


def test_suffixed_cap_does_not_trigger_a_numeric_failure():
    # CAP='39.2T' is legitimately non-numeric and must NEVER be numeric-checked.
    validate_panel(_good_panel())  # would raise if CAP were numeric-gated


def test_sparse_but_clean_numeric_column_tolerated():
    df = _good_panel()
    df["AM"] = [None, None, None, None, 5.0, -2.0][: len(df)]  # mostly NA, rest numeric
    validate_panel(df)


def test_all_null_numeric_column_is_coverage_not_corruption():
    df = _good_panel()
    df["EG"] = [None] * len(df)  # entirely empty -> tolerated (coverage)
    validate_panel(df)


def test_empty_panel_raises():
    with pytest.raises(PanelIntegrityError, match="empty"):
        validate_panel(_good_panel().iloc[0:0])


def test_mostly_null_ticker_raises():
    df = _good_panel()
    df["TKR"] = ["", "", "", "", "", "T5"][: len(df)]  # 5/6 blank -> catastrophic
    with pytest.raises(PanelIntegrityError, match="TKR"):
        validate_panel(df)


def test_minimal_universe_requirement_set_passes():
    """load_universe only needs TKR/PRC/CAP — the gate is parameterizable."""
    df = pd.DataFrame({"TKR": ["AAPL", "MSFT"], "PRC": [200.0, 400.0], "CAP": ["3.5T", "2.5T"]})
    validate_panel(
        df, source="universe", required_columns=("TKR", "PRC", "CAP"), required_numeric=("PRC",)
    )
