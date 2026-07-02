"""Tests for S0 validation harness primitives.

Covers:
  - rolling_walk_forward (backtest_stats)
  - compute_ic_decay   (validation/ic_decay)
  - compute_turnover   (validation/turnover)
  - build_perf_matrix  (validation/perf_matrix)
"""

import datetime

import numpy as np
import pytest

from trade_modules.backtest_stats import rolling_walk_forward
from trade_modules.validation.ic_decay import compute_ic_decay
from trade_modules.validation.perf_matrix import build_perf_matrix
from trade_modules.validation.turnover import compute_turnover

# ---------------------------------------------------------------------------
# rolling_walk_forward
# ---------------------------------------------------------------------------


class TestRollingWalkForward:
    def _make_items(self, dates: list[str]) -> list[dict]:
        return [{"signal_date": d, "idx": i} for i, d in enumerate(dates)]

    def test_returns_list_of_tuples(self):
        items = self._make_items(
            [f"2026-01-{d:02d}" for d in range(1, 31)]  # 30 days
        )
        folds = rolling_walk_forward(items, n_folds=3, embargo_days=0)
        assert isinstance(folds, list)
        for train, test in folds:
            assert isinstance(train, list)
            assert isinstance(test, list)

    def test_train_expands_across_folds(self):
        items = self._make_items([f"2026-01-{d:02d}" for d in range(1, 31)])
        folds = rolling_walk_forward(items, n_folds=3, embargo_days=0)
        assert len(folds) >= 2
        train_sizes = [len(tr) for tr, _ in folds]
        assert train_sizes == sorted(train_sizes)

    def test_test_is_forward_of_train(self):
        items = self._make_items([f"2026-01-{d:02d}" for d in range(1, 31)])
        folds = rolling_walk_forward(items, n_folds=3, embargo_days=0)
        for train, test in folds:
            if not train or not test:
                continue
            max_train_date = max(r["signal_date"] for r in train)
            min_test_date = min(r["signal_date"] for r in test)
            assert max_train_date < min_test_date

    def test_embargo_respected(self):
        """No test item should be within embargo_days of the last train date."""
        items = self._make_items([f"2026-01-{d:02d}" for d in range(1, 31)])
        embargo_days = 5
        folds = rolling_walk_forward(items, n_folds=3, embargo_days=embargo_days)
        for train, test in folds:
            if not train or not test:
                continue
            max_train = datetime.date.fromisoformat(max(r["signal_date"] for r in train))
            for item in test:
                test_date = datetime.date.fromisoformat(item["signal_date"])
                gap = (test_date - max_train).days
                assert gap > embargo_days, (
                    f"Embargo violated: gap={gap} not > embargo={embargo_days}"
                )

    def test_train_test_disjoint(self):
        items = self._make_items([f"2026-01-{d:02d}" for d in range(1, 31)])
        folds = rolling_walk_forward(items, n_folds=3, embargo_days=0)
        for train, test in folds:
            train_idx = {r["idx"] for r in train}
            test_idx = {r["idx"] for r in test}
            assert train_idx.isdisjoint(test_idx)

    def test_no_empty_test_folds_returned(self):
        items = self._make_items([f"2026-01-{d:02d}" for d in range(1, 31)])
        folds = rolling_walk_forward(items, n_folds=3, embargo_days=0)
        for _, test in folds:
            assert len(test) > 0

    def test_degenerate_few_dates(self):
        """Fewer distinct dates than n_folds+1 must not crash."""
        items = self._make_items(["2026-01-01", "2026-01-02"])
        folds = rolling_walk_forward(items, n_folds=5, embargo_days=0)
        assert isinstance(folds, list)

    def test_empty_input(self):
        folds = rolling_walk_forward([], n_folds=3, embargo_days=0)
        assert folds == []

    def test_handles_date_objects(self):
        """items whose date_key is a datetime.date object must work."""
        items = [{"signal_date": datetime.date(2026, 1, d), "idx": d} for d in range(1, 21)]
        folds = rolling_walk_forward(items, n_folds=3, embargo_days=0)
        assert len(folds) > 0

    def test_custom_date_key(self):
        items = [{"ts": f"2026-02-{d:02d}", "idx": d} for d in range(1, 21)]
        folds = rolling_walk_forward(items, n_folds=3, embargo_days=0, date_key="ts")
        assert len(folds) > 0

    def test_last_date_included_in_test_set(self):
        """The item with the maximum date must appear in at least one test set."""
        items = self._make_items([f"2026-01-{d:02d}" for d in range(1, 31)])
        folds = rolling_walk_forward(items, n_folds=3, embargo_days=0)
        assert len(folds) > 0

        max_date = max(item["signal_date"] for item in items)
        test_items = [item for _, test in folds for item in test]
        test_dates = {item["signal_date"] for item in test_items}
        assert max_date in test_dates, (
            f"Last date {max_date} not found in any test set; test dates: {sorted(test_dates)}"
        )


# ---------------------------------------------------------------------------
# compute_ic_decay
# ---------------------------------------------------------------------------


class TestComputeIcDecay:
    def test_decaying_ic_gives_finite_positive_half_life(self):
        ic = {7: 0.10, 30: 0.06, 90: 0.02}
        result = compute_ic_decay(ic)
        assert result["half_life_days"] is not None
        assert result["half_life_days"] > 0
        assert result["ic0"] is not None
        assert result["ic0"] > 0

    def test_curve_is_sorted_by_horizon(self):
        ic = {90: 0.02, 7: 0.10, 30: 0.06}
        result = compute_ic_decay(ic)
        keys = list(result["curve"].keys())
        assert keys == sorted(keys)

    def test_flat_ic_returns_none_half_life_with_note(self):
        ic = {7: 0.10, 30: 0.10, 90: 0.10}
        result = compute_ic_decay(ic)
        # Flat: slope should be ~0, half_life should be None (or very large / infinite)
        # The spec says flat/rising → None; but an exactly flat log-linear gives slope≈0 →
        # half_life = None is the expected response.
        # Accept either None or a positive value >= 1000 (effectively non-decaying);
        # the spec says None, so enforce that:
        assert result["half_life_days"] is None
        assert result["note"] != ""

    def test_rising_ic_returns_none_half_life_with_note(self):
        """Rising IC (positive slope) → not a decay → half_life None."""
        ic = {7: 0.02, 30: 0.06, 90: 0.10}
        result = compute_ic_decay(ic)
        assert result["half_life_days"] is None
        assert result["note"] != ""

    def test_single_point_returns_none(self):
        result = compute_ic_decay({30: 0.05})
        assert result["half_life_days"] is None
        assert result["note"] != ""

    def test_empty_returns_none(self):
        result = compute_ic_decay({})
        assert result["half_life_days"] is None
        assert result["note"] != ""

    def test_all_nonpositive_returns_none(self):
        ic = {7: 0.0, 30: -0.05, 90: 0.0}
        result = compute_ic_decay(ic)
        assert result["half_life_days"] is None
        assert result["note"] != ""

    def test_keys_present(self):
        result = compute_ic_decay({7: 0.10, 30: 0.06, 90: 0.02})
        assert set(result.keys()) == {"half_life_days", "ic0", "curve", "note"}

    def test_note_empty_on_success(self):
        result = compute_ic_decay({7: 0.10, 30: 0.06, 90: 0.02})
        assert result["note"] == ""


# ---------------------------------------------------------------------------
# compute_turnover
# ---------------------------------------------------------------------------


class TestComputeTurnover:
    def _actions(self):
        return [
            {"date": "2026-01-01", "ticker": "AAPL", "weight_change": 0.05, "tier": "large"},
            {"date": "2026-01-05", "ticker": "MSFT", "weight_change": -0.03, "tier": "large"},
            {"date": "2026-01-10", "ticker": "NVDA", "weight_change": 0.02, "tier": "small"},
        ]

    def test_empty_actions_returns_zeros(self):
        result = compute_turnover([], window_days=90)
        assert result["turnover_annual_pct"] == 0.0
        assert result["annualized_drag_bps"] == 0.0
        assert result["n_trades"] == 0

    def test_n_trades_count(self):
        result = compute_turnover(self._actions(), window_days=90)
        assert result["n_trades"] == 3

    def test_turnover_formula(self):
        """sum|Δw| = 0.05+0.03+0.02 = 0.10; window=90; annual_pct = (0.10/90)*365*100."""
        actions = self._actions()
        result = compute_turnover(actions, window_days=90)
        expected_pct = (0.10 / 90) * 365 * 100
        assert result["turnover_annual_pct"] == pytest.approx(expected_pct, rel=1e-6)

    def test_flat_cost_no_tier_map(self):
        """Default 20 bps flat cost."""
        # drag = sum over actions: (|Δw| / window * 365 * 20)
        actions = self._actions()
        window = 90
        expected_drag = sum(abs(a["weight_change"]) for a in actions) / window * 365 * 20
        result = compute_turnover(actions, window_days=window, tier_cost_bps=None)
        assert result["annualized_drag_bps"] == pytest.approx(expected_drag, rel=1e-6)

    def test_per_tier_cost(self):
        """large=10 bps, small=30 bps; missing tier defaults to 20."""
        tier_cost = {"large": 10, "small": 30}
        actions = self._actions()
        window = 90
        # AAPL large=10, MSFT large=10, NVDA small=30
        costs = [10, 10, 30]
        expected_drag = sum(
            abs(a["weight_change"]) / window * 365 * c for a, c in zip(actions, costs)
        )
        result = compute_turnover(actions, window_days=window, tier_cost_bps=tier_cost)
        assert result["annualized_drag_bps"] == pytest.approx(expected_drag, rel=1e-6)

    def test_missing_tier_defaults_to_20_bps(self):
        """Action without 'tier' key should use 20 bps default."""
        actions = [{"date": "2026-01-01", "ticker": "X", "weight_change": 0.10}]
        result = compute_turnover(actions, window_days=365, tier_cost_bps={"large": 10})
        # No tier in action dict → default 20 bps
        expected_drag = 0.10 / 365 * 365 * 20
        assert result["annualized_drag_bps"] == pytest.approx(expected_drag, rel=1e-6)

    def test_keys_present(self):
        result = compute_turnover(self._actions(), window_days=90)
        assert set(result.keys()) == {"turnover_annual_pct", "annualized_drag_bps", "n_trades"}

    @pytest.mark.parametrize("window", [0, -1, -100])
    def test_window_days_nonpositive_raises(self, window):
        actions = [{"date": "2024-01-15", "ticker": "AAPL", "weight_change": 0.05}]
        with pytest.raises(ValueError, match="window_days must be positive"):
            compute_turnover(actions, window_days=window)


# ---------------------------------------------------------------------------
# build_perf_matrix
# ---------------------------------------------------------------------------


class TestBuildPerfMatrix:
    def _rows(self):
        return [
            {"signal_date": "2026-Q1", "ticker": "AAPL", "alpha": 0.10},
            {"signal_date": "2026-Q1", "ticker": "MSFT", "alpha": 0.05},
            {"signal_date": "2026-Q2", "ticker": "AAPL", "alpha": 0.08},
            {"signal_date": "2026-Q2", "ticker": "MSFT", "alpha": 0.03},
            {"signal_date": "2026-Q2", "ticker": "NVDA", "alpha": 0.12},
        ]

    def test_shape(self):
        row_labels, col_labels, mat = build_perf_matrix(self._rows())
        assert len(row_labels) == 2  # Q1, Q2
        assert len(col_labels) == 3  # AAPL, MSFT, NVDA
        assert mat.shape == (2, 3)

    def test_values_correct(self):
        row_labels, col_labels, mat = build_perf_matrix(self._rows())
        r_q1 = row_labels.index("2026-Q1")
        r_q2 = row_labels.index("2026-Q2")
        c_aapl = col_labels.index("AAPL")
        c_msft = col_labels.index("MSFT")
        c_nvda = col_labels.index("NVDA")

        assert mat[r_q1, c_aapl] == pytest.approx(0.10)
        assert mat[r_q1, c_msft] == pytest.approx(0.05)
        assert mat[r_q2, c_aapl] == pytest.approx(0.08)
        assert mat[r_q2, c_nvda] == pytest.approx(0.12)

    def test_missing_cell_is_nan(self):
        row_labels, col_labels, mat = build_perf_matrix(self._rows())
        r_q1 = row_labels.index("2026-Q1")
        c_nvda = col_labels.index("NVDA")
        assert np.isnan(mat[r_q1, c_nvda])

    def test_duplicate_rows_averaged(self):
        rows = [
            {"signal_date": "2026-Q1", "ticker": "AAPL", "alpha": 0.10},
            {"signal_date": "2026-Q1", "ticker": "AAPL", "alpha": 0.20},
        ]
        row_labels, col_labels, mat = build_perf_matrix(rows)
        c_aapl = col_labels.index("AAPL")
        r_q1 = row_labels.index("2026-Q1")
        assert mat[r_q1, c_aapl] == pytest.approx(0.15)

    def test_dtype_is_float64(self):
        _, _, mat = build_perf_matrix(self._rows())
        assert mat.dtype == np.float64

    def test_labels_sorted(self):
        row_labels, col_labels, mat = build_perf_matrix(self._rows())
        assert row_labels == sorted(row_labels)
        assert col_labels == sorted(col_labels)

    def test_custom_keys(self):
        rows = [
            {"period": "P1", "config": "A", "ret": 0.05},
            {"period": "P1", "config": "B", "ret": 0.10},
            {"period": "P2", "config": "A", "ret": 0.07},
        ]
        row_labels, col_labels, mat = build_perf_matrix(
            rows, period_key="period", config_key="config", value_key="ret"
        )
        assert mat.shape == (2, 2)
        r_p1 = row_labels.index("P1")
        c_a = col_labels.index("A")
        assert mat[r_p1, c_a] == pytest.approx(0.05)

    def test_empty_rows_returns_empty(self):
        row_labels, col_labels, mat = build_perf_matrix([])
        assert row_labels == []
        assert col_labels == []
        assert mat.shape == (0, 0)
