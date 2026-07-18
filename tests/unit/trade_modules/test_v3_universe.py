import pandas as pd

from trade_modules.v3.universe import load_universe, parse_cap


def test_parse_cap_suffixes():
    assert parse_cap("3.5T") == 3.5e12
    assert parse_cap("800B") == 800e9
    assert parse_cap("1.2M") == 1.2e6
    assert pd.isna(parse_cap("--"))


def test_load_universe_filters(tmp_path):
    csv = tmp_path / "etoro.csv"
    pd.DataFrame(
        {
            "TKR": ["AAPL", "PENNY", "SMALL", "7203.T", "MSFT"],
            "PRC": [200.0, 0.5, 50.0, 3000.0, 400.0],
            "CAP": ["3.5T", "10B", "100M", "40T", "2.5T"],
        }
    ).to_csv(csv, index=False)
    u = load_universe(str(csv), min_price=1.0, min_cap_usd=5e8)
    assert u == ["AAPL", "MSFT"]  # PENNY (price), SMALL (cap), 7203.T (non-USD) excluded


# ---------------------------------------------------------------------------
# BUILD ④ — coverage-gate universe (replaces the analyst AND-gate)
# ---------------------------------------------------------------------------

_FULL = {"PET": "10", "ROE": "9", "FCF": "1", "52W": "80", "B": "1.0", "AM": "2", "EG": "5"}


def _write_panel(tmp_path, rows):
    """Write a panel CSV with TKR/PRC/CAP + the 7 coverage factor columns."""
    cols = ["TKR", "PRC", "CAP", "PET", "ROE", "FCF", "52W", "B", "AM", "EG"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    p = tmp_path / "etoro.csv"
    df[cols].to_csv(p, index=False)
    return str(p)


def test_coverage_constants_declared():
    from trade_modules.v3.constants import COVERAGE_FACTORS, MIN_FACTOR_COVERAGE

    assert MIN_FACTOR_COVERAGE == 6
    assert set(COVERAGE_FACTORS) == {"PET", "ROE", "FCF", "52W", "B", "AM", "EG"}


def test_load_universe_coverage_gate_keeps_only_well_covered(tmp_path):
    from trade_modules.v3.universe import load_universe

    rows = [
        {"TKR": "AAA", "PRC": "10", "CAP": "1B", **_FULL},  # 7/7
        {"TKR": "BBB", "PRC": "10", "CAP": "1B", **{**_FULL, "AM": "--"}},  # 6/7
        {"TKR": "CCC", "PRC": "10", "CAP": "1B", **{**_FULL, "AM": "--", "EG": "--"}},  # 5/7
    ]
    u = load_universe(_write_panel(tmp_path, rows), min_factor_coverage=6)
    assert u == ["AAA", "BBB"]  # CCC (5/7) excluded


def test_load_universe_default_has_no_coverage_gate(tmp_path):
    from trade_modules.v3.universe import load_universe

    rows = [
        {"TKR": "AAA", "PRC": "10", "CAP": "1B", **_FULL},
        {"TKR": "CCC", "PRC": "10", "CAP": "1B", **{**_FULL, "AM": "--", "EG": "--", "PET": "--"}},
    ]
    u = load_universe(_write_panel(tmp_path, rows))  # default coverage=0 -> gate off
    assert u == ["AAA", "CCC"]


def test_coverage_gate_still_applies_cap_and_price_floors(tmp_path):
    from trade_modules.v3.universe import load_universe

    rows = [
        {"TKR": "AAA", "PRC": "10", "CAP": "1B", **_FULL},  # ok
        {"TKR": "SMALL", "PRC": "10", "CAP": "100M", **_FULL},  # 7/7 but sub-cap
        {"TKR": "PENNY", "PRC": "0.5", "CAP": "1B", **_FULL},  # 7/7 but penny
    ]
    u = load_universe(_write_panel(tmp_path, rows), min_factor_coverage=6)
    assert u == ["AAA"]  # coverage never overrides the cap/price/US base filters


def test_coverage_gate_missing_factor_column_counts_as_absent(tmp_path):
    from trade_modules.v3.universe import load_universe

    # Panel with no AM column at all -> AM absent for everyone -> max coverage 6/7.
    rows = [
        {"TKR": "AAA", "PRC": "10", "CAP": "1B", **{k: v for k, v in _FULL.items() if k != "AM"}}
    ]
    csv = _write_panel(tmp_path, rows)
    # AAA has 6 of the other factors -> passes >=6, but would fail >=7.
    assert load_universe(csv, min_factor_coverage=6) == ["AAA"]
    assert load_universe(csv, min_factor_coverage=7) == []


def test_assemble_scored_universe_unions_and_always_includes_holdings(tmp_path):
    from trade_modules.v3.universe import assemble_scored_universe

    rows = [
        {"TKR": "COVA", "PRC": "10", "CAP": "1B", **_FULL},  # covered 7/7
        {"TKR": "COVB", "PRC": "10", "CAP": "1B", **_FULL},  # covered 7/7
        {
            "TKR": "HELD",
            "PRC": "10",
            "CAP": "1B",
            **{**_FULL, "AM": "--", "EG": "--", "PET": "--"},
        },  # 4/7
    ]
    csv = _write_panel(tmp_path, rows)
    u = assemble_scored_universe(csv, holdings=["HELD"], candidates=["CAND"], min_factor_coverage=6)
    assert "HELD" in u  # low-coverage holding is never dropped
    assert "CAND" in u  # analyst candidate still included
    assert "COVA" in u and "COVB" in u  # coverage-gated names included
    assert u.index("HELD") < u.index("COVA")  # holdings first, then covered
    assert len(u) == len(set(u))  # de-duped
