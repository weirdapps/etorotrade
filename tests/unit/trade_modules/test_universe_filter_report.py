"""Unit tests for universe_filter_report.py — pure aggregation logic only.

Tests cover:
  A) Summary/aggregation helpers on synthetic filter_universe output.
  B) compare_tier_region_filters — confirms a genuinely better tier subset ranks above
     the worse subsets when injected with a deterministic fake harness_evaluate.

All tests are pure: no I/O, no CSV reads, no filesystem access.
pragma: no cover is applied only to IO/live code in the report script itself.
"""

from __future__ import annotations

import pandas as pd

from scripts.universe_filter_report import (
    _format_report,
    aggregate_dropped_names,
    aggregate_gate_counts,
    compare_tier_region_filters,
    summarize_filter_result,
)

# ---------------------------------------------------------------------------
# Fixtures — synthetic filter_universe output
# ---------------------------------------------------------------------------


def _make_eligible_df(*tickers: str) -> pd.DataFrame:
    """Create a minimal eligible DataFrame with TKR and NAME columns."""
    return pd.DataFrame({"TKR": list(tickers), "NAME": [f"Co {t}" for t in tickers]})


def _make_filter_result(
    eligible_tickers: list[str],
    price_only: list[str],
    excluded: dict[str, str],
    reasons: dict[str, list[str]],
) -> dict:
    """Build a synthetic filter_universe result dict."""
    eligible_df = _make_eligible_df(*eligible_tickers)
    total = len(eligible_tickers) + len(price_only) + len(excluded) + len(reasons)
    gate_counts = {
        "min_cap": sum(1 for rs in reasons.values() if any("cap" in r for r in rs)),
        "min_analysts": sum(1 for rs in reasons.values() if any("analysts" in r for r in rs)),
        "positive_earnings": sum(
            1 for rs in reasons.values() if any("negative earnings" in r for r in rs)
        ),
        "trend": sum(1 for rs in reasons.values() if any("melting" in r for r in rs)),
    }
    summary = {
        "total": total,
        "fundamental": len(reasons),
        "eligible": len(eligible_tickers),
        "price_only": len(price_only),
        "excluded": len(excluded),
        "gate_failures": gate_counts,
    }
    return {
        "eligible": eligible_df,
        "price_only": price_only,
        "excluded": excluded,
        "reasons": reasons,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# summarize_filter_result
# ---------------------------------------------------------------------------


class TestSummarizeFilterResult:
    def test_basic_counts(self):
        result = _make_filter_result(
            eligible_tickers=["AAPL", "MSFT"],
            price_only=["BTC-USD", "GLD"],
            excluded={"BADPRC": "no usable price"},
            reasons={
                "AAPL": [],
                "MSFT": [],
                "TINY": ["cap $0.40B < $2B"],
            },
        )
        summary = summarize_filter_result(result)
        assert summary["eligible"] == 2
        assert summary["price_only"] == 2
        assert summary["excluded"] == 1
        assert summary["fundamental_total"] == 3  # AAPL + MSFT + TINY

    def test_zero_eligible(self):
        result = _make_filter_result(
            eligible_tickers=[],
            price_only=["BTC-USD"],
            excluded={},
            reasons={"FAIL": ["cap $0.01B < $2B"]},
        )
        summary = summarize_filter_result(result)
        assert summary["eligible"] == 0
        assert summary["fundamental_total"] == 1

    def test_eligible_pct_of_fundamental(self):
        result = _make_filter_result(
            eligible_tickers=["A", "B"],
            price_only=[],
            excluded={},
            reasons={"A": [], "B": [], "C": ["cap $0.50B < $2B"]},
        )
        summary = summarize_filter_result(result)
        # 2 eligible out of 3 fundamental rows = 66.7%
        pct = summary["eligible_pct"]
        assert 66 < pct < 68


# ---------------------------------------------------------------------------
# aggregate_gate_counts
# ---------------------------------------------------------------------------


class TestAggregateGateCounts:
    def test_counts_match_gate_failures(self):
        reasons = {
            "A": ["cap $0.40B < $2B"],
            "B": ["analysts 2 < 5"],
            "C": ["negative earnings (PET=-3.0, PEF=-2.0)"],
            "D": ["melting: 52W 12 < 30"],
            "E": ["cap $0.10B < $2B", "analysts 1 < 5"],
        }
        gate_failures = {
            "min_cap": 2,
            "min_analysts": 2,
            "positive_earnings": 1,
            "trend": 1,
        }
        result = _make_filter_result(
            eligible_tickers=["OK"],
            price_only=[],
            excluded={},
            reasons=reasons,
        )
        # Override with explicit counts to test the helper directly
        result["summary"]["gate_failures"] = gate_failures
        counts = aggregate_gate_counts(result)
        assert counts["min_cap"] == 2
        assert counts["min_analysts"] == 2
        assert counts["positive_earnings"] == 1
        assert counts["trend"] == 1

    def test_zero_failures_returns_zeros(self):
        result = _make_filter_result(
            eligible_tickers=["AAPL"],
            price_only=[],
            excluded={},
            reasons={"AAPL": []},
        )
        result["summary"]["gate_failures"] = {
            "min_cap": 0,
            "min_analysts": 0,
            "positive_earnings": 0,
            "trend": 0,
        }
        counts = aggregate_gate_counts(result)
        assert all(v == 0 for v in counts.values())


# ---------------------------------------------------------------------------
# aggregate_dropped_names
# ---------------------------------------------------------------------------


class TestAggregateDroppedNames:
    def test_dropped_names_have_ticker_and_reason(self):
        reasons = {
            "AAPL": [],
            "TINY": ["cap $0.30B < $2B"],
            "NOANA": ["analysts unknown (missing) < 5"],
        }
        result = _make_filter_result(
            eligible_tickers=["AAPL"],
            price_only=[],
            excluded={},
            reasons=reasons,
        )
        dropped = aggregate_dropped_names(result)
        tickers = [d["ticker"] for d in dropped]
        assert "TINY" in tickers
        assert "NOANA" in tickers
        # Passing row (AAPL) must NOT be in the dropped list
        assert "AAPL" not in tickers

    def test_excluded_names_included(self):
        result = _make_filter_result(
            eligible_tickers=[],
            price_only=[],
            excluded={"BADPRC": "no usable price"},
            reasons={},
        )
        dropped = aggregate_dropped_names(result)
        tickers = [d["ticker"] for d in dropped]
        assert "BADPRC" in tickers
        assert any("no usable price" in d["reason"] for d in dropped if d["ticker"] == "BADPRC")

    def test_empty_result_returns_empty_list(self):
        result = _make_filter_result(
            eligible_tickers=["AAPL"],
            price_only=[],
            excluded={},
            reasons={"AAPL": []},
        )
        dropped = aggregate_dropped_names(result)
        assert dropped == []

    def test_price_only_not_in_dropped(self):
        """Price-only assets (crypto, ETFs) are routed, not dropped — must not appear."""
        result = _make_filter_result(
            eligible_tickers=["AAPL"],
            price_only=["BTC-USD", "GLD"],
            excluded={},
            reasons={"AAPL": []},
        )
        dropped = aggregate_dropped_names(result)
        tickers = [d["ticker"] for d in dropped]
        assert "BTC-USD" not in tickers
        assert "GLD" not in tickers


# ---------------------------------------------------------------------------
# compare_tier_region_filters
# ---------------------------------------------------------------------------


def _make_backtest_rows(
    tiers: list[str],
    region: str = "us",
    alpha_val: float = 0.05,
    n_per_tier: int = 60,
) -> list[dict]:
    """Generate synthetic backtest rows with given tier labels and alpha value."""
    import itertools

    dates = [f"2026-0{(i // 10) + 1}-{(i % 10) + 1:02d}" for i in range(n_per_tier)]
    rows = []
    for i, (tier, date) in enumerate(itertools.product(tiers, dates[:n_per_tier])):
        rows.append(
            {
                "ticker": f"T{i}",
                "signal": "momentum",
                "signal_date": date,
                "tier": tier,
                "region": region,
                "horizon": 30,
                "alpha": alpha_val,
                "net_alpha": alpha_val - 0.005,
                "future_price": 100.0,
            }
        )
    return rows


def _fake_harness(rows: list[dict]) -> dict:
    """Deterministic fake harness that returns a score based on average net_alpha."""
    if not rows:
        return {"overall": {"passed": False, "dsr": None}, "_subset_score": -999.0}
    alphas = [r.get("net_alpha", 0.0) for r in rows if r.get("net_alpha") is not None]
    avg = sum(alphas) / len(alphas) if alphas else 0.0
    passed = avg > 0
    return {
        "overall": {"passed": passed, "dsr": avg},
        "_subset_score": avg,
    }


class TestCompareTierRegionFilters:
    def test_better_tier_subset_ranks_higher(self):
        """The large/mega subset has higher alpha → should rank above the full set
        and the micro/small subset (which has near-zero alpha)."""
        # Full set: mix of high alpha (large/mega) + low alpha (micro/small)
        good_rows = _make_backtest_rows(["large", "mega"], alpha_val=0.08)
        bad_rows = _make_backtest_rows(["micro", "small"], alpha_val=0.001)
        all_rows = good_rows + bad_rows

        rankings = compare_tier_region_filters(all_rows, _fake_harness)

        # Check we get at least 3 entries: full, excl-micro-small, regions
        assert len(rankings) >= 3

        # Find the relevant subsets
        names = [r["name"] for r in rankings]
        assert any(
            "large" in n.lower() or "excl" in n.lower() or "mega" in n.lower() for n in names
        )

        # The subset excluding micro/small should rank above the full set
        excl_rank = next((r["rank"] for r in rankings if "excl" in r["name"].lower()), None)
        full_rank = next((r["rank"] for r in rankings if r["name"] == "full"), None)
        assert excl_rank is not None
        assert full_rank is not None
        assert excl_rank < full_rank  # lower rank number = better

    def test_ranking_is_deterministic(self):
        """Same input → same ranking on repeated calls."""
        rows = _make_backtest_rows(["large", "mega", "mid", "small", "micro"], alpha_val=0.03)
        r1 = compare_tier_region_filters(rows, _fake_harness)
        r2 = compare_tier_region_filters(rows, _fake_harness)
        assert [e["name"] for e in r1] == [e["name"] for e in r2]

    def test_empty_rows_returns_results(self):
        """Empty input doesn't crash; all subsets have no data."""
        rankings = compare_tier_region_filters([], _fake_harness)
        assert isinstance(rankings, list)
        # All should have score of -999 or similar "no data" sentinel
        for r in rankings:
            assert "name" in r
            assert "score" in r

    def test_each_entry_has_required_fields(self):
        rows = _make_backtest_rows(["large", "mid"], alpha_val=0.04)
        rankings = compare_tier_region_filters(rows, _fake_harness)
        required = {"name", "subset_desc", "n_rows", "score", "rank", "passed"}
        for entry in rankings:
            assert required.issubset(set(entry.keys())), (
                f"Entry '{entry.get('name')}' missing fields: {required - set(entry.keys())}"
            )

    def test_regions_included_in_ranking(self):
        """Each region should appear as its own subset."""
        rows = _make_backtest_rows(["large"], region="us", alpha_val=0.06) + _make_backtest_rows(
            ["large"], region="eu", alpha_val=0.02
        )
        rankings = compare_tier_region_filters(rows, _fake_harness)
        names = [r["name"] for r in rankings]
        assert any("us" in n.lower() for n in names)
        assert any("eu" in n.lower() for n in names)

    def test_subsets_with_zero_rows_dont_crash(self):
        """If a region has zero rows, the harness sees an empty list — no crash."""
        rows = _make_backtest_rows(["large"], region="us", alpha_val=0.05)
        # hk region has no rows
        rankings = compare_tier_region_filters(rows, _fake_harness)
        assert isinstance(rankings, list)


# ---------------------------------------------------------------------------
# FIX 4 REGRESSION — Section-B history caption uses results_rows count
# ---------------------------------------------------------------------------


def _make_minimal_filter_result() -> dict:
    """Minimal synthetic filter result for _format_report tests."""
    return _make_filter_result(
        eligible_tickers=["AAPL"],
        price_only=["BTC-USD"],
        excluded={},
        reasons={"AAPL": []},
    )


def _make_minimal_rankings() -> list[dict]:
    """Minimal ranking list for _format_report tests."""
    return [
        {
            "rank": 1,
            "name": "full",
            "subset_desc": "All rows",
            "n_rows": 86368,
            "score": 0.05,
            "passed": True,
            "verdict": {},
        }
    ]


class TestFormatReportHistoryCaption:
    """Section-B caption must use results_rows count, not n_total (etoro.csv length)."""

    def test_section_b_uses_results_rows_not_n_total(self):
        """_format_report Section B caption must display results_rows count (86,368),
        NOT n_total (12,826 — etoro.csv row count)."""
        live_summary = summarize_filter_result(_make_minimal_filter_result())
        gate_counts = aggregate_gate_counts(_make_minimal_filter_result())
        report = _format_report(
            live_summary=live_summary,
            gate_counts=gate_counts,
            dropped_names=[],
            price_only_sample=[],
            tier_region_rankings=_make_minimal_rankings(),
            data_date="2026-07-01",
            n_total=12826,
            results_rows=86368,
            forward_gated_pass_rates={
                "analysts_pass_pct": "N/A",
                "earnings_pass_pct": "N/A",
                "trend_pass_pct": "N/A",
            },
        )
        # The Section B history caption must show results_rows count, not n_total
        assert "86,368" in report or "86368" in report
        # n_total only appears in Section A header, not the Section B history line
        # Confirm the Section B caveat line specifically mentions results_rows
        section_b_start = report.find("## B)")
        assert section_b_start != -1
        section_b = report[section_b_start:]
        # The history caveat inside Section B must show 86,368 rows
        assert "86,368" in section_b or "86368" in section_b
