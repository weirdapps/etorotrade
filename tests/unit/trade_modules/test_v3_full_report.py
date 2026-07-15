"""TDD tests for scripts.v3_full_report pure helpers + synthetic preview.

Covers:
  - load_account_json: parses the {"nav", "weights"} schema; nav null → None;
    missing file → (empty, None, present=False); env-var resolution.
  - resolve_current_weights: live account passthrough; equal-split fallback
    when the account file is absent (approx flag set).
  - build_synthetic_preview_html: network-free full render containing the exec
    panel, every action group, factor cards, no literal None, zero em-dashes.
"""

import json

import pandas as pd

from scripts.v3_full_report import (
    build_synthetic_preview_html,
    load_account_json,
    resolve_current_weights,
)

# ---------------------------------------------------------------------------
# load_account_json
# ---------------------------------------------------------------------------


def test_load_account_json_parses_schema(tmp_path):
    p = tmp_path / "acct.json"
    p.write_text(json.dumps({"nav": 123456.7, "weights": {"AAPL": 0.25, "MSFT": 0.1}}))
    weights, nav, present = load_account_json(str(p))
    assert present is True
    assert nav == 123456.7
    assert set(weights.index) == {"AAPL", "MSFT"}
    assert weights["AAPL"] == 0.25
    assert weights["MSFT"] == 0.1


def test_load_account_json_null_nav(tmp_path):
    p = tmp_path / "acct.json"
    p.write_text(json.dumps({"nav": None, "weights": {"AAPL": 1.0}}))
    weights, nav, present = load_account_json(str(p))
    assert present is True
    assert nav is None
    assert weights["AAPL"] == 1.0


def test_load_account_json_missing_file():
    weights, nav, present = load_account_json("/nonexistent/path/does-not-exist.json")
    assert present is False
    assert nav is None
    assert weights.empty


def test_load_account_json_env_var(tmp_path, monkeypatch):
    p = tmp_path / "env_acct.json"
    p.write_text(json.dumps({"nav": 5000.0, "weights": {"NVDA": 0.5}}))
    monkeypatch.setenv("V3_ACCOUNT_JSON", str(p))
    weights, nav, present = load_account_json()  # no explicit path → env resolution
    assert present is True
    assert nav == 5000.0
    assert weights["NVDA"] == 0.5


# ---------------------------------------------------------------------------
# resolve_current_weights
# ---------------------------------------------------------------------------


def test_resolve_current_weights_live_passthrough():
    acct = pd.Series({"AAPL": 0.3, "MSFT": 0.2})
    weights, approx = resolve_current_weights(["AAPL", "MSFT"], acct, account_present=True)
    assert approx is False
    assert weights.equals(acct)


def test_resolve_current_weights_equal_split_fallback():
    weights, approx = resolve_current_weights(
        ["AAA", "BBB", "CCC", "AAA"], pd.Series(dtype=float), account_present=False
    )
    assert approx is True
    # dedup → 3 names, equal split
    assert set(weights.index) == {"AAA", "BBB", "CCC"}
    assert abs(weights.sum() - 1.0) < 1e-9
    assert abs(weights["AAA"] - 1.0 / 3.0) < 1e-9


def test_resolve_current_weights_empty_account_falls_back():
    """account_present=True but empty weights → still equal-split fallback."""
    weights, approx = resolve_current_weights(
        ["X", "Y"], pd.Series(dtype=float), account_present=True
    )
    assert approx is True
    assert set(weights.index) == {"X", "Y"}


# ---------------------------------------------------------------------------
# synthetic preview
# ---------------------------------------------------------------------------


def test_build_synthetic_preview_is_complete():
    html = build_synthetic_preview_html()
    assert html.startswith("<!DOCTYPE") and "</html>" in html
    assert '<div class="exec-panel">' in html
    for cls in ("buy", "add", "trim", "sell", "hold"):
        assert f"act-grp act-grp--{cls}" in html, f"missing action group: {cls}"
    assert '<article class="card' in html  # matches "card" and "card card--action"
    assert "None" not in html
    assert "—" not in html
