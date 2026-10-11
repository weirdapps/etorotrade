"""AM is a signed analyst COUNT; every AM threshold is written in buy-share PERCENTAGE POINTS.

Since 2026-07-30 (``recommendations_parser.calculate_analyst_momentum``) the ``AM`` column is
the net change in the number of buy-rated analysts over ~3 months. The thresholds in
``config.yaml`` (``max_analyst_momentum``, ``min_analyst_momentum``), the interpolation knots
in ``calculate_sell_score`` / ``calculate_buy_score`` and the hard-coded ``-5`` guards were
all written for the old metric, the 3-month change in buy-% in percentage points. The fix
restates the count as a share of coverage (``#T``, the analyst-opinion count) before any
threshold reads it, so every rule keeps its documented meaning.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from trade_modules.analysis.analyst_momentum import am_share_pp, am_share_pp_series
from trade_modules.analysis_engine import calculate_action

_CLEAR_EARNINGS = {
    "earnings_date": None,
    "days_until": None,
    "status": "clear",
    "should_hold": False,
    "conviction_boost": False,
    "conviction_adjustment": 0,
}


@pytest.fixture(autouse=True)
def _mock_earnings_proximity():
    with patch(
        "trade_modules.earnings_proximity.check_earnings_proximity",
        return_value=_CLEAR_EARNINGS,
    ):
        yield


class TestAmSharePp:
    def test_count_over_coverage_in_percentage_points(self):
        assert am_share_pp(-2, 10) == pytest.approx(-20.0)
        assert am_share_pp(3, 40) == pytest.approx(7.5)
        assert am_share_pp(0, 12) == 0.0

    def test_clipped_to_the_old_metric_range(self):
        # The old metric was a difference of two percentages: it lived in [-100, +100].
        assert am_share_pp(-12, 1) == -100.0
        assert am_share_pp(19, 9) == 100.0

    @pytest.mark.parametrize(
        "am, cov", [(None, 10), (np.nan, 10), (-3, None), (-3, np.nan), (-3, 0), (-3, -1), ("x", 5)]
    )
    def test_unknown_is_nan_never_zero(self, am, cov):
        assert np.isnan(am_share_pp(am, cov))

    def test_series_matches_scalar(self):
        am = pd.Series([-2, 3, np.nan, -12, 1])
        cov = pd.Series([10, 40, 10, 1, 0])
        out = am_share_pp_series(am, cov)
        assert out.iloc[0] == pytest.approx(-20.0)
        assert out.iloc[1] == pytest.approx(7.5)
        assert np.isnan(out.iloc[2])
        assert out.iloc[3] == -100.0
        assert np.isnan(out.iloc[4])


def _sell_row(am_count, coverage, cap=600_000_000_000):
    """A US mega-cap whose only SELL-side defect is analyst momentum."""
    return {
        "upside": 15.0,
        "buy_percentage": 70.0,
        "pct_from_52w_high": 85,
        "above_200dma": True,
        "analyst_momentum": am_count,
        "market_cap": cap,
        "analyst_count": coverage,
        "total_ratings": coverage,
        "pe_forward": 25.0,
        "pe_trailing": 28.0,
        "EXRET": 10.5,
        "return_on_equity": 20.0,
        "debt_to_equity": 50.0,
        "fcf_yield": 3.0,
        "peg_ratio": 1.5,
        "short_percent": 2.0,
        "beta": 1.1,
        "revenue_growth": 15.0,
        "pe_vs_sector": 1.2,
        "target_dispersion": 30.0,
        "price": 150.0,
        "two_hundred_day_avg": 130.0,
    }


def _sell_conditions_logged(caplog, _ticker):
    return [
        r.getMessage()
        for r in caplog.records
        if "SELL TRIGGER - analyst momentum" in r.getMessage()
    ]


@pytest.fixture
def _legacy_sell_path():
    """``max_analyst_momentum`` is read only on the legacy ANY-trigger path; production runs
    ``use_signal_scoring: true``, where AM reaches SELL through the score knots instead."""
    with patch(
        "trade_modules.yaml_config_loader.YamlConfigLoader.is_signal_scoring_enabled",
        return_value=False,
    ):
        yield


@pytest.mark.usefixtures("_legacy_sell_path")
class TestSellTriggerOnCountScale:
    def test_raw_count_never_reaches_the_pp_trigger(self):
        """NEGATIVE CONTROL: the live AM minimum is -19 (2026-10-11 etoro.csv), so a raw count
        compared against ``max_analyst_momentum: -20`` can never fire."""
        live_min = -19
        assert not (live_min < -20)

    def test_exodus_on_thin_coverage_fires_the_sell_trigger(self, caplog):
        # 4 of 10 buy-rated analysts gone in 3 months is -40pp of buy share: the exodus the
        # -20pp trigger exists for. As a raw count (-4) it sat far above -20 and never fired.
        df = pd.DataFrame([_sell_row(-4, 10)], index=["EXOD"])
        with caplog.at_level("INFO"):
            calculate_action(df)
        assert _sell_conditions_logged(caplog, "EXOD")

    def test_same_count_on_broad_coverage_does_not_fire(self, caplog):
        # -4 of 60 is -6.7pp: a drift, not an exodus.
        df = pd.DataFrame([_sell_row(-4, 60)], index=["DRFT"])
        with caplog.at_level("INFO"):
            calculate_action(df)
        assert not _sell_conditions_logged(caplog, "DRFT")


class TestScoreKnotsReadShare:
    def test_sell_score_reads_the_share_not_the_count(self):
        from trade_modules.analysis.signals import calculate_sell_score

        # -2 of 10 is -20pp, past the -15pp knot (full AM penalty). The raw count -2 sat
        # between the -2 and 0 knots and scored 40 of 100.
        kw = {
            "upside": 15.0,
            "buy_pct": 70.0,
            "pct_52w": 85.0,
            "pef": 25.0,
            "pet": 28.0,
            "roe": 20.0,
            "de": 50.0,
        }
        _, thin = calculate_sell_score(
            **kw, sell_scoring_config={}, analyst_momentum=am_share_pp(-2, 10)
        )
        _, broad = calculate_sell_score(
            **kw, sell_scoring_config={}, analyst_momentum=am_share_pp(-2, 80)
        )
        assert "am_decline:-20.0pp" in thin
        assert not any(f.startswith("am_decline") for f in broad)
