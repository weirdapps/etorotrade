"""TDD — overlay holding-key canonicalization + managed-sleeve gross reporting.

eToro carries US listings with a ``.US`` suffix in the account + portfolio.csv (T.US),
while etoro.csv + the price store use the bare Yahoo ticker (T). The overlay normalizes
held/candidate keys to the etoro.csv-canonical form so the book, universe, price store
and display all agree (owner 2026-07-24).

Task 4: overlay_portfolio_view gains a ``managed_weight`` param so the reported
``gross`` includes held-out managed sleeves (GLD/UVXY/LYXGRE) and ``cash`` reflects
the true un-invested portion (owner 2026-07-24).
"""

from __future__ import annotations

import pytest

from scripts.v3_overlay_report import _canonicalize_keys


def test_us_suffix_maps_to_bare_csv_key():
    csv = ["T", "SBMO.NV", "AAPL", "GILD"]
    out = _canonicalize_keys(["T.US", "AAPL.US", "SBMO.NV", "GILD"], csv)
    assert out == ["T", "AAPL", "SBMO.NV", "GILD"]  # .US stripped to the bare CSV key


def test_unresolvable_key_is_left_unchanged():
    # A held name not in etoro.csv (e.g. a managed sleeve) is kept as-is.
    csv = ["T", "AAPL"]
    out = _canonicalize_keys(["GLD", "UVXY", "LYXGRE.DE"], csv)
    assert out == ["GLD", "UVXY", "LYXGRE.DE"]


def test_exact_csv_key_wins():
    # If the exact key is already in the CSV, keep it (don't remap by root).
    out = _canonicalize_keys(["T.US"], ["T.US", "T"])
    assert out == ["T.US"]


def test_ambiguous_root_left_unresolved():
    # Two CSV keys share the root "T" -> ambiguous -> a held "T.EUR" is not remapped.
    out = _canonicalize_keys(["T.EUR"], ["T", "T.L"])
    assert out == ["T.EUR"]


# ---------------------------------------------------------------------------
# Task 4: overlay_portfolio_view gross includes managed sleeves
# ---------------------------------------------------------------------------


def test_overlay_portfolio_view_gross_includes_managed_sleeves():
    import pandas as pd

    from scripts.v3_overlay_report import overlay_portfolio_view

    overlay = {"weights": pd.Series({"AAA": 0.40, "BBB": 0.35}), "diagnostics": {}}
    view = overlay_portfolio_view(overlay, None, managed_weight=0.14)
    assert view["model_gross"] == pytest.approx(0.75)
    assert view["gross"] == pytest.approx(0.89)  # 0.75 model + 0.14 managed
    assert view["cash"] == pytest.approx(0.11)
    assert view["managed_weight"] == pytest.approx(0.14)


def test_overlay_portfolio_view_defaults_managed_zero():
    import pandas as pd

    from scripts.v3_overlay_report import overlay_portfolio_view

    overlay = {"weights": pd.Series({"AAA": 0.40}), "diagnostics": {}}
    view = overlay_portfolio_view(overlay, None)  # default managed_weight=0.0
    assert view["gross"] == pytest.approx(0.40)
    assert view["managed_weight"] == pytest.approx(0.0)
