"""Smoke tests for the Outlook-safe v3 email renderer (report_email)."""

from __future__ import annotations

import math

import pandas as pd

from trade_modules.v3 import report_email as rem

NAN = math.nan


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["AAA CORP TRUN", "BBB LIMITE", "CCC"],
            "description": [
                "Aaa Corporation, Inc. designs and sells widgets worldwide.",
                "Bbb Limited, together with its subsidiaries, provides services.",
                "",
            ],
            "sector": ["Technology", "Energy", "Financial Services"],
            "value_z": [0.5, -0.3, NAN],
            "quality_z": [0.2, 0.1, 0.0],
            "growth_z": [1.0, -0.2, 0.3],
            "momentum_z": [-0.1, 0.4, 0.0],
            "lowvol_z": [0.3, -0.5, 0.1],
            "pe_forward": [23.0, 10.0, NAN],
            "pb": [4.1, 1.5, 2.0],
            "ps_sector": [1.3, 0.6, 1.0],
            "roe": [18.9, 8.1, NAN],
            "op_margin": [0.22, 0.40, 0.10],
            "fcf": [0.4, NAN, 0.9],
            "earn_growth": [139.0, 24.0, 5.0],
            "rev_growth": [0.22, 0.65, 0.1],
            "mom_12_1": [0.30, 0.19, 0.0],
            "pct_52w_high": [84.0, 100.0, 50.0],
            "price_perf": [NAN, NAN, 12.0],
            "upside": [15.5, 12.8, 4.9],
            "buy_pct": [75.0, 100.0, 100.0],
            "analyst_mom": [35.0, 0.0, 0.0],
            "beta": [-0.2, 0.4, 1.0],
            "realized_vol": [0.35, 0.25, 0.30],
            "de": [40.4, NAN, 15.5],
        },
        index=["AAA", "BBB", "CCC"],
    )


def _meta() -> dict:
    return {
        "date": "2026-07-15",
        "generated_utc": "2026-07-15 17:22 UTC",
        "regime": "risk_on",
        "account": {
            "total_equity": 1_000_000,
            "unrealized_pnl": 50_000,
            "profit_pct": 5.0,
            "invested_cost": 900_000,
            "available": 100_000,
        },
        "social": {
            "copiers": 1000,
            "copiers_gain_pct": -1.0,
            "win_ratio": 55.0,
            "trades_ytd": 300,
            "gain_ytd": 3.0,
            "gain_mtd": 2.0,
            "risk_score": 3,
            "max_daily_risk": 4,
            "aum_tier_desc": "$1M-$2M",
        },
        "allocations": {
            "geography": {"North America": 0.7, "Europe": 0.3},
            "asset_type": {"Equity": 1.0},
            "sector": {"Technology": 0.5, "Other": 0.5},
        },
        "caps": {"name": 0.10, "sector": 0.35, "usd_bloc": 0.65, "region": 0.65},
        "coverage": {"n_scored": 3, "n_eligible": 3, "pct": 1.0, "adv_dropped": 0},
    }


def _actions() -> list:
    return [
        {
            "ticker": "AAA",
            "name": "AAA CORP TRUN",
            "sector": "Technology",
            "action": "BUY",
            "conviction": 1.66,
            "current_pct": 0.0,
            "target_pct": 0.064,
            "delta_pct": 0.064,
            "delta_usd": 64000,
            "price": 73.0,
        },
        {
            "ticker": "BBB",
            "name": "BBB LIMITE",
            "sector": "Energy",
            "action": "SELL",
            "conviction": -0.8,
            "current_pct": 0.016,
            "target_pct": 0.0,
            "delta_pct": -0.016,
            "delta_usd": -16000,
            "price": 45.0,
            "pnl": -2000,
            "pnl_pct": -10.0,
            "current_value": 16000,
        },
        {
            "ticker": "CCC",
            "name": "CCC",
            "sector": "Financial Services",
            "action": "HOLD",
            "conviction": 0.5,
            "current_pct": 0.01,
            "target_pct": 0.01,
            "delta_pct": 0.0,
            "delta_usd": 0.0,
            "price": 100.0,
            "pnl": 500,
            "pnl_pct": 5.0,
            "current_value": 10000,
        },
    ]


def _view() -> dict:
    return {
        "diagnostics": {
            "gate": {
                "gross_after": 0.84,
                "vol_after": 0.121,
                "vol_ceiling": 0.35,
                "cvar_after": 0.25,
                "net_beta": 0.9,
                "effective_bets": 20,
                "min_effective_bets": 10,
                "usd_bloc": 0.68,
            }
        }
    }


def test_render_email_is_outlook_safe():
    html = rem.render_email_report(
        _scores(),
        _meta(),
        portfolio=_view(),
        actions=_actions(),
        conditioning={"final_deployment": 0.98},
    )
    assert isinstance(html, str) and html.startswith("<!DOCTYPE")
    # Outlook / OWA sanitizer hostiles must be absent.
    for forbidden in ("var(--", "<details", "display:flex", "display:grid", "<svg"):
        assert forbidden not in html, forbidden
    assert "&amp;amp;" not in html  # no HTML-entity double-escaping
    # sections + all six factor clusters present
    for token in (
        "Factor Snapshot",
        "Value",
        "Quality",
        "PEAD",
        "Growth",
        "Momentum",
        "Strength",
        "Risk",
    ):
        assert token in html, token
    # USD-bloc breach surfaced (0.68 > 0.65 cap)
    assert "BREACH" in html
    # full name resolved from the description (not the truncated eToro name)
    assert "Aaa Corporation, Inc" in html
    # description shown (card's last element)
    assert "designs and sells widgets" in html
    for t in ("AAA", "BBB", "CCC"):
        assert t in html


def test_full_name_extraction():
    assert (
        rem._full_name("Micron Technology, Inc. designs memory.", "MICRON TECHNOL")
        == "Micron Technology, Inc"
    )
    assert (
        rem._full_name("Tencent Holdings Limited, an investment company.", "TENCENT HOLDIN")
        == "Tencent Holdings Limited"
    )
    assert rem._full_name("", "FALLBACK") == "FALLBACK"
    assert rem._full_name(None, "FB") == "FB"


def test_metric_formatting():
    assert rem._mfmt("mom_12_1", 0.30) == "+30%"  # fraction -> signed percent
    assert rem._mfmt("roe", 18.9) == "18.9%"  # already a percent
    assert rem._mfmt("pe_forward", 23.0) == "23.0"  # ratio
    assert rem._mfmt("beta", -0.2) == "-0.20"
    assert rem._mfmt("upside", 15.5) == "+15.5%"
    assert rem._mfmt("de", NAN) == "n/a"
