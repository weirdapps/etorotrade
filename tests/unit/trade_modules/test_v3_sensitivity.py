"""TDD tests for scripts.v3_sensitivity — factor-weight 3-way sensitivity.

Covers the PURE pieces (no network): the three configs, summarize_book,
style_tilt, and the comparison HTML renderer. The network assembly in main()
is exercised only on the VPS, not here.
"""

import numpy as np
import pandas as pd

from scripts.v3_sensitivity import (
    CONFIGS,
    CORE,
    render_comparison_html,
    style_tilt,
    summarize_book,
)

# --------------------------------------------------------------------------- #
# configs
# --------------------------------------------------------------------------- #

_CLUSTER_KEYS = {"value", "quality", "momentum", "growth", "lowvol", "strength"}


def test_three_configs_present_and_well_formed():
    assert set(CONFIGS) == {"value_heavy", "balanced", "growth_forward"}
    for name, cfg in CONFIGS.items():
        assert set(cfg) == _CLUSTER_KEYS, name
        assert abs(sum(cfg.values()) - 1.0) < 1e-9, name


def test_config_growth_dial_increases_across_configs():
    """Growth weight steps up: value_heavy 0 -> balanced 0.15 -> growth_forward 0.20."""
    assert CONFIGS["value_heavy"]["growth"] == 0.0
    assert CONFIGS["balanced"]["growth"] == 0.15
    assert CONFIGS["growth_forward"]["growth"] == 0.20
    # and value steps DOWN as growth steps up
    assert (
        CONFIGS["value_heavy"]["value"]
        > CONFIGS["balanced"]["value"]
        > CONFIGS["growth_forward"]["value"]
    )


def test_core_list_is_the_mega_cap_growth_core():
    assert CORE == ["NVDA", "GOOG", "MSFT", "AAPL", "AMZN", "AVGO", "TSM", "META"]


# --------------------------------------------------------------------------- #
# summarize_book
# --------------------------------------------------------------------------- #


def _synthetic_actions():
    """A crafted action list covering every action group + core outcomes."""
    return [
        {
            "ticker": "NVDA",
            "name": "NVIDIA",
            "action": "ADD",
            "target_pct": 0.08,
            "current_pct": 0.05,
            "conviction": 2.5,
        },
        {
            "ticker": "NEW1",
            "name": "NewCo One",
            "action": "BUY",
            "target_pct": 0.07,
            "current_pct": 0.0,
            "conviction": 2.1,
        },
        {
            "ticker": "GOOG",
            "name": "Alphabet",
            "action": "BUY",
            "target_pct": 0.06,
            "current_pct": 0.0,
            "conviction": 2.0,
        },
        {
            "ticker": "MSFT",
            "name": "Microsoft",
            "action": "HOLD",
            "target_pct": 0.05,
            "current_pct": 0.05,
            "conviction": 1.8,
        },
        {
            "ticker": "AAPL",
            "name": "Apple",
            "action": "TRIM",
            "target_pct": 0.02,
            "current_pct": 0.06,
            "conviction": 0.3,
        },
        {
            "ticker": "META",
            "name": "Meta",
            "action": "SELL",
            "target_pct": 0.0,
            "current_pct": 0.04,
            "conviction": -0.7,
        },
    ]


def test_summarize_book_action_counts():
    actions = _synthetic_actions()
    summary = summarize_book(actions, pd.DataFrame(), CORE)
    assert summary["n_buy"] == 2  # NEW1, GOOG
    assert summary["n_add"] == 1  # NVDA
    assert summary["n_hold"] == 1  # MSFT
    assert summary["n_trim"] == 1  # AAPL
    assert summary["n_sell"] == 1  # META


def test_summarize_book_core_survival():
    actions = _synthetic_actions()
    summary = summarize_book(actions, pd.DataFrame(), CORE)
    surv = {row["ticker"]: row for row in summary["core_survival"]}
    # order preserved = the CORE order
    assert [r["ticker"] for r in summary["core_survival"]] == CORE
    assert surv["NVDA"]["action"] == "ADD" and surv["NVDA"]["kept"] is True
    assert surv["GOOG"]["action"] == "BUY" and surv["GOOG"]["kept"] is True
    assert surv["MSFT"]["action"] == "HOLD" and surv["MSFT"]["kept"] is True
    assert surv["AAPL"]["action"] == "TRIM" and surv["AAPL"]["kept"] is False
    assert surv["META"]["action"] == "SELL" and surv["META"]["kept"] is False
    # AMZN/AVGO/TSM never appear in actions -> ABSENT, not kept
    assert surv["AMZN"]["action"] == "ABSENT" and surv["AMZN"]["kept"] is False
    # kept-count = NVDA, GOOG, MSFT
    assert summary["core_kept"] == 3
    assert summary["core_total"] == 8


def test_summarize_book_top_names_sorted_by_target_weight():
    actions = _synthetic_actions()
    summary = summarize_book(actions, pd.DataFrame(), CORE, top_n=12)
    top = summary["top_names"]
    # SELL (target 0) excluded; sorted by target_pct desc.
    assert [t["ticker"] for t in top] == ["NVDA", "NEW1", "GOOG", "MSFT", "AAPL"]
    assert top[0]["name"] == "NVIDIA"
    assert abs(top[0]["target_pct"] - 0.08) < 1e-12
    assert abs(top[0]["conviction"] - 2.5) < 1e-12


def test_summarize_book_top_n_truncates():
    actions = [
        {
            "ticker": f"T{i}",
            "name": f"N{i}",
            "action": "BUY",
            "target_pct": 0.10 - i * 0.001,
            "current_pct": 0.0,
            "conviction": 1.0,
        }
        for i in range(20)
    ]
    summary = summarize_book(actions, pd.DataFrame(), CORE, top_n=12)
    assert len(summary["top_names"]) == 12
    assert summary["top_names"][0]["ticker"] == "T0"  # highest target


# --------------------------------------------------------------------------- #
# style_tilt
# --------------------------------------------------------------------------- #


def test_style_tilt_weighted_by_target_book_nan_safe():
    scored = pd.DataFrame(
        {"pe_trailing": [20.0, 40.0, np.nan], "mom_12_1": [0.10, 0.30, 0.20]},
        index=["A", "B", "C"],
    )
    actions = [
        {"ticker": "A", "target_pct": 0.10, "action": "BUY"},
        {"ticker": "B", "target_pct": 0.30, "action": "BUY"},
        {"ticker": "C", "target_pct": 0.10, "action": "BUY"},
    ]
    tilt = style_tilt(actions, scored)
    # P/E: C excluded (NaN), weights A:0.10 B:0.30 -> 0.25/0.75 -> 0.25*20+0.75*40 = 35
    assert abs(tilt["wavg_pe"] - 35.0) < 1e-9
    # Mom: all present, weights 0.2/0.6/0.2 -> 0.2*.10+0.6*.30+0.2*.20 = 0.24
    assert abs(tilt["wavg_mom"] - 0.24) < 1e-9


def test_style_tilt_empty_book_is_nan():
    tilt = style_tilt([], pd.DataFrame(columns=["pe_trailing", "mom_12_1"]))
    assert np.isnan(tilt["wavg_pe"])
    assert np.isnan(tilt["wavg_mom"])


# --------------------------------------------------------------------------- #
# comparison HTML
# --------------------------------------------------------------------------- #


def _synthetic_column(label, growth_kept):
    return {
        "label": label,
        "weights": CONFIGS[label],
        "n_buy": 3,
        "n_add": 2,
        "n_trim": 1,
        "n_sell": 4,
        "n_hold": 5,
        "core_survival": [
            {
                "ticker": t,
                "action": ("HOLD" if i < growth_kept else "SELL"),
                "kept": i < growth_kept,
            }
            for i, t in enumerate(CORE)
        ],
        "core_kept": growth_kept,
        "core_total": len(CORE),
        "top_names": [
            {"ticker": "NVDA", "name": "NVIDIA", "target_pct": 0.08, "conviction": 2.4},
            {"ticker": "MSFT", "name": "Microsoft", "target_pct": 0.07, "conviction": 2.0},
        ],
        "port_vol": 0.145,
        "deployment": 0.90,
        "wavg_pe": 30.5,
        "wavg_mom": 0.18,
    }


def test_render_comparison_html_three_columns_no_emdash():
    cols = [
        _synthetic_column("value_heavy", 3),
        _synthetic_column("balanced", 6),
        _synthetic_column("growth_forward", 8),
    ]
    meta = {"date": "2026-07-13", "nav": 1_117_940.29, "universe": 120, "regime": "NEUTRAL"}
    html = render_comparison_html(cols, meta)
    assert html.startswith("<!DOCTYPE")
    assert "</html>" in html
    # three config labels present
    for label in ("value_heavy", "balanced", "growth_forward"):
        assert label in html
    # core survival table names present
    for t in CORE:
        assert t in html
    # IBM Plex mono numerals + light theme cue
    assert "IBM Plex Mono" in html
    # no em-dash anywhere
    assert "—" not in html
    # renders the core-kept counts
    assert "8" in html and "3" in html
