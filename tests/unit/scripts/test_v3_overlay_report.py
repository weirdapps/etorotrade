"""TDD — overlay holding-key canonicalization.

eToro carries US listings with a ``.US`` suffix in the account + portfolio.csv (T.US),
while etoro.csv + the price store use the bare Yahoo ticker (T). The overlay normalizes
held/candidate keys to the etoro.csv-canonical form so the book, universe, price store
and display all agree (owner 2026-07-24).
"""

from __future__ import annotations

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
