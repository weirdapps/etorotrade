"""Tests for scripts/validate_brief.py.

The validator cross-checks every numerical claim in a midday-brief draft
against the yfinance market snapshot. It must reject invented numbers
without flagging legitimate news-sourced data.

Regression context (2026-05-12): the validator false-positived on
stock-specific percentages cited next to a $TICKER (e.g. "$NWG.L -4.7%")
and on market-cap notation like "$4T". Both came from MCP news, not from
the snapshot, and both should be skipped — only macro indices, futures,
commodities, FX, and yields are snapshot-validatable.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent.parent.parent / "scripts" / "validate_brief.py"


@pytest.fixture(scope="module")
def vb():
    spec = importlib.util.spec_from_file_location("validate_brief", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def snapshot():
    """Minimal snapshot mirroring the 2026-05-12 run."""
    return {
        "instruments": {
            "^GDAXI": {"price": 24087.91, "prev_close": 24350.28, "change_pct": -1.08},
            "^FCHI": {"price": 8008.94, "prev_close": 8056.38, "change_pct": -0.59},
            "^STOXX": {"price": 608.86, "prev_close": 612.79, "change_pct": -0.64},
            "^FTSE": {"price": 10231.34, "prev_close": 10269.4, "change_pct": -0.37},
            "^N225": {"price": 62742.57, "prev_close": 62417.88, "change_pct": 0.52},
            "^KS11": {"price": 7643.15, "prev_close": 7822.24, "change_pct": -2.29},
            "BZ=F": {"price": 107.84, "prev_close": 104.20, "change_pct": 3.49},
            "CL=F": {"price": 101.83, "prev_close": 98.10, "change_pct": 3.80},
        }
    }


class TestTickerAttributedPercentages:
    """Percentages cited next to a $TICKER are news-sourced (came from MCP
    news_reader) and not validatable against the macro snapshot. Skip them."""

    def test_uk_bank_moves_are_skipped(self, vb, snapshot):
        text = (
            "UK political crisis. $NWG.L -4.7%, $LLOY.L -4.3%, $BARC.L -4.1%. "
            "Sterling and gilts both wobbling."
        )
        result = vb.validate(text, snapshot)
        assert not result["pct_errors"], (
            f"Ticker-attributed percentages should be skipped, got: {result['pct_errors']}"
        )

    def test_ticker_with_plus_percentage(self, vb, snapshot):
        text = "Memory rally continues; $MU +37% on the week."
        result = vb.validate(text, snapshot)
        assert not result["pct_errors"]

    def test_distant_ticker_does_not_attribute(self, vb, snapshot):
        # Ticker far from the percentage — should NOT attribute.
        # -9.9% is not in the snapshot and not in news context, so this should fail.
        text = (
            "$NVDA had a rough day on the open. Sector rotation playing out "
            "through midday across the tape with broader semiconductor "
            "weakness pulling the cohort lower. The print landed at -9.9% "
            "by lunchtime."
        )
        result = vb.validate(text, snapshot)
        assert result["pct_errors"], "Distant ticker should not whitelist the percentage"


class TestMarketCapSuffix:
    """$NT / $NB / $NM is market-cap notation, not a price. Skip."""

    def test_dollar_4t_is_not_a_price(self, vb, snapshot):
        text = "$AVGO market is its own gravity now; the >$4T cap dwarfs everything else."
        result = vb.validate(text, snapshot)
        assert not result["price_errors"], (
            f"$4T cap notation should be skipped, got: {result['price_errors']}"
        )

    def test_dollar_800b_is_not_a_price(self, vb, snapshot):
        text = "$MU cleared $800B market cap this morning."
        result = vb.validate(text, snapshot)
        assert not result["price_errors"]

    def test_dollar_50m_is_not_a_price(self, vb, snapshot):
        text = "Buyback authorization lifted to $50M for the year."
        result = vb.validate(text, snapshot)
        assert not result["price_errors"]

    def test_lowercase_4tn_is_not_a_price(self, vb, snapshot):
        text = "$AVGO market cap pushed past $4tn this week."
        result = vb.validate(text, snapshot)
        assert not result["price_errors"]

    def test_genuine_price_still_validates(self, vb, snapshot):
        # Brent at $107.84 matches snapshot — should pass.
        text = "Oil rips. Brent at $107.84, WTI at $101.83."
        result = vb.validate(text, snapshot)
        assert not result["price_errors"], (
            f"Snapshot-matching prices should pass, got: {result['price_errors']}"
        )

    def test_invented_price_still_fails(self, vb, snapshot):
        # Brent at $200 doesn't match snapshot ($107.84) — should fail.
        text = "Oil rips to historic levels. Brent at $200 today."
        result = vb.validate(text, snapshot)
        assert result["price_errors"], "Invented price should be flagged"


class TestSnapshotMatchingStillWorks:
    """Regression guard: legitimate snapshot data must still validate."""

    def test_index_percentages_match_snapshot(self, vb, snapshot):
        text = (
            "Asia: Nikkei +0.5%, KOSPI -2.3% on profit-taking. "
            "Europe heavy: DAX -1.1%, CAC -0.6%, FTSE -0.4%, Stoxx 600 -0.6%."
        )
        result = vb.validate(text, snapshot)
        assert not result["pct_errors"], (
            f"Snapshot-matching index moves should pass, got: {result['pct_errors']}"
        )

    def test_invented_index_percentage_fails(self, vb, snapshot):
        # DAX at +9.9% is nowhere in the snapshot — should fail.
        text = "DAX ripped +9.9% on the open, blowing past every prior record."
        result = vb.validate(text, snapshot)
        assert result["pct_errors"], "Invented index move should be flagged"


class TestRealWorldDraft:
    """The 2026-05-12 brief that triggered this fix should now pass."""

    def test_full_post_passes(self, vb, snapshot):
        post = (
            "𝗠𝗶𝗱𝗱𝗮𝘆 𝗯𝗿𝗶𝗲𝗳 | 𝗧𝘂𝗲𝘀𝗱𝗮𝘆 𝟭𝟮 𝗠𝗮𝘆\n\n"
            "Asia mixed. Nikkei +0.5%, Hang Seng -0.2%, KOSPI -2.3%.\n\n"
            "Europe is heavy. Stoxx 600 -0.6%, DAX -1.1%, CAC -0.6%, FTSE -0.4%.\n\n"
            "1. Brent +3.5% to $107.84, WTI +3.8% to $101.83. "
            "Energy bid: $XOM, $CVX, $SHEL.L, $BP.L all working.\n\n"
            "2. UK crisis. $NWG.L -4.7%, $LLOY.L -4.3%, $BARC.L -4.1%.\n\n"
            "3. $MU cleared $800B market cap, +37% on the week.\n\n"
            "$AVGO and the >$4T cap is its own gravity now.\n"
        )
        result = vb.validate(post, snapshot)
        assert result["passed"], (
            f"Post should pass: pct_errors={result['pct_errors']}, "
            f"price_errors={result['price_errors']}, ticker_errors={result['ticker_errors']}"
        )
