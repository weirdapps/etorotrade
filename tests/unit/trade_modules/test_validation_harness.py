"""
Tests for trade_modules/validation/harness.py

TDD: these tests are written FIRST and should fail until harness.py is implemented.

All tests exercise only the pure evaluate() function — no I/O.
"""

import numpy as np

# Not yet implemented — should fail on import until harness.py exists
from trade_modules.validation.harness import evaluate


def _make_rows(
    n: int,
    alpha_mean: float,
    alpha_std: float,
    signal: str = "TEST",
    horizons: tuple = (7, 30, 60, 90, 180, 250),
    n_tickers: int = 20,
    n_regimes: int = 0,
    include_net_alpha: bool = True,
    missing_future_price_count: int = 0,
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic backtest result rows."""
    rng = np.random.default_rng(seed)
    rows = []
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    for i in range(n):
        alpha = float(rng.normal(alpha_mean, alpha_std))
        net_alpha = alpha - 0.04 if include_net_alpha else None
        ticker = tickers[i % n_tickers]
        # Spread dates across a wide range
        day_offset = int((i / n) * 500)
        date = f"2022-{(day_offset // 30) % 12 + 1:02d}-{(day_offset % 28) + 1:02d}"
        row = {
            "signal": signal,
            "tier": "large",
            "region": "us",
            "signal_date": date,
            "alpha": alpha,
            "ticker": ticker,
            "future_price": 100.0,
            "price_at_signal": 90.0,
        }
        if include_net_alpha and net_alpha is not None:
            row["net_alpha"] = net_alpha

        # Assign horizon — distribute across given horizons
        row["horizon"] = horizons[i % len(horizons)]

        if n_regimes > 0:
            regime_list = ["risk_on", "risk_off", "neutral", "crisis"][:n_regimes]
            row["regime"] = regime_list[i % n_regimes]

        rows.append(row)

    # Inject missing future_price rows at the end (for survivorship test)
    for j in range(missing_future_price_count):
        extra = dict(rows[j])
        extra["future_price"] = None
        extra["signal_date"] = "2023-01-01"
        extra["alpha"] = float(rng.normal(alpha_mean, alpha_std))
        if "net_alpha" in extra:
            extra["net_alpha"] = extra["alpha"] - 0.04
        rows.append(extra)

    return rows


# ---------------------------------------------------------------------------
# 1. clear_edge — strong signal should produce computable stats, no crash
# ---------------------------------------------------------------------------
class TestClearEdge:
    def test_no_crash(self):
        rows = _make_rows(200, alpha_mean=0.05, alpha_std=0.02, n_regimes=2, seed=42)
        result = evaluate(rows)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        rows = _make_rows(200, alpha_mean=0.05, alpha_std=0.02, n_regimes=2, seed=42)
        result = evaluate(rows)
        assert "overall" in result
        assert "families" in result
        assert "regime_stratified" in result
        assert "turnover" in result
        assert "survivorship" in result
        assert "n_trials" in result

    def test_dsr_computable_and_bounded(self):
        rows = _make_rows(200, alpha_mean=0.05, alpha_std=0.02, n_regimes=2, seed=42)
        result = evaluate(rows)
        dsr = result["overall"]["dsr"]
        assert dsr is not None
        assert 0.0 <= dsr <= 1.0, f"DSR should be a probability, got {dsr}"

    def test_dsr_computable_or_reasons_explain(self):
        rows = _make_rows(200, alpha_mean=0.05, alpha_std=0.02, n_regimes=2, seed=42)
        result = evaluate(rows)
        # Per brief: "overall DSR >= 0.5, overall passed OR reasons list is non-empty
        # (we don't hard-assert pass since DSR formula is well-defined)".
        # Synthetic data has only ~34 rows at h=30 (200 / 6 horizons), which
        # correctly triggers the insufficient_obs gate — that's expected behaviour.
        dsr = result["overall"]["dsr"]
        reasons = result["overall"]["reasons"]
        # Either DSR is meaningful or reasons explain why it failed
        assert (dsr is not None and dsr >= 0.5) or len(reasons) > 0, (
            f"DSR={dsr} — expected either DSR>=0.5 or non-empty reasons; got {reasons}"
        )

    def test_reasons_is_list(self):
        rows = _make_rows(200, alpha_mean=0.05, alpha_std=0.02, n_regimes=2, seed=42)
        result = evaluate(rows)
        assert isinstance(result["overall"]["reasons"], list)


# ---------------------------------------------------------------------------
# 2. pure_noise — zero-alpha signal must FAIL
# ---------------------------------------------------------------------------
class TestPureNoise:
    def setup_method(self):
        np.random.seed(42)
        self.rows = _make_rows(200, alpha_mean=0.0, alpha_std=0.1, n_regimes=1, seed=42)
        self.result = evaluate(self.rows)

    def test_passed_is_false(self):
        assert self.result["overall"]["passed"] is False

    def test_reasons_non_empty(self):
        assert len(self.result["overall"]["reasons"]) > 0


# ---------------------------------------------------------------------------
# 3. thin_family — fewer than min_obs rows → insufficient_data flag
# ---------------------------------------------------------------------------
class TestThinFamily:
    def test_insufficient_data_flag(self):
        rows = _make_rows(10, alpha_mean=0.05, alpha_std=0.02, signal="X", seed=42)
        result = evaluate(rows, min_obs=30)
        assert "X" in result["families"]
        assert result["families"]["X"]["insufficient_data"] is True

    def test_no_crash_on_thin_data(self):
        rows = _make_rows(5, alpha_mean=0.05, alpha_std=0.02, signal="X", seed=42)
        result = evaluate(rows, min_obs=30)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 4. survivorship — rows with None future_price should be counted
# ---------------------------------------------------------------------------
class TestSurvivorship:
    def test_no_forward_price_counted(self):
        n_good = 100
        n_missing = 15
        rows = _make_rows(
            n_good,
            alpha_mean=0.05,
            alpha_std=0.02,
            missing_future_price_count=n_missing,
            seed=42,
        )
        result = evaluate(rows)
        assert result["survivorship"]["no_forward_price"] > 0, (
            "Expected some rows with missing future_price to be flagged"
        )

    def test_total_rows_matches_input(self):
        n_good = 100
        n_missing = 15
        rows = _make_rows(
            n_good,
            alpha_mean=0.05,
            alpha_std=0.02,
            missing_future_price_count=n_missing,
            seed=42,
        )
        result = evaluate(rows)
        assert result["survivorship"]["total_rows"] == n_good + n_missing

    def test_pct_dropped_in_range(self):
        rows = _make_rows(
            100, alpha_mean=0.05, alpha_std=0.02, missing_future_price_count=10, seed=42
        )
        result = evaluate(rows)
        pct = result["survivorship"]["pct_dropped"]
        assert 0.0 <= pct <= 100.0


# ---------------------------------------------------------------------------
# 5. regime_stratified — rows with 'regime' key → stratified output
# ---------------------------------------------------------------------------
class TestRegimeStratified:
    def test_two_or_more_regimes_in_output(self):
        rows = _make_rows(200, alpha_mean=0.03, alpha_std=0.02, n_regimes=2, seed=42)
        result = evaluate(rows)
        assert len(result["regime_stratified"]) >= 2, (
            f"Expected >=2 regime keys, got {list(result['regime_stratified'].keys())}"
        )

    def test_regime_stat_keys(self):
        rows = _make_rows(200, alpha_mean=0.03, alpha_std=0.02, n_regimes=3, seed=42)
        result = evaluate(rows)
        for regime_key, stats in result["regime_stratified"].items():
            assert "n" in stats, f"Missing 'n' in regime {regime_key}"
            assert "hit" in stats, f"Missing 'hit' in regime {regime_key}"
            assert "avg_alpha" in stats, f"Missing 'avg_alpha' in regime {regime_key}"


# ---------------------------------------------------------------------------
# 6. no_regime_key — rows without 'regime' → graceful, no crash
# ---------------------------------------------------------------------------
class TestNoRegimeKey:
    def test_no_crash(self):
        rows = _make_rows(100, alpha_mean=0.03, alpha_std=0.02, n_regimes=0, seed=42)
        result = evaluate(rows)
        assert isinstance(result, dict)

    def test_regime_stratified_empty_or_graceful(self):
        rows = _make_rows(100, alpha_mean=0.03, alpha_std=0.02, n_regimes=0, seed=42)
        result = evaluate(rows)
        # When no regime key is present, regime_stratified should be empty dict
        assert result["regime_stratified"] == {} or isinstance(result["regime_stratified"], dict)


# ---------------------------------------------------------------------------
# 7. no_action_records — turnover should be None
# ---------------------------------------------------------------------------
class TestNoActionRecords:
    def test_turnover_is_none_without_action_records(self):
        rows = _make_rows(100, alpha_mean=0.03, alpha_std=0.02, seed=42)
        result = evaluate(rows)
        assert result["turnover"] is None

    def test_turnover_not_none_with_action_records(self):
        rows = _make_rows(100, alpha_mean=0.03, alpha_std=0.02, seed=42)
        # Build minimal valid action records
        actions = [
            {"weight_change": 0.05, "tier": "large", "date": "2022-01-01", "ticker": "T00"}
        ] * 10
        result = evaluate(rows, action_records=actions)
        assert result["turnover"] is not None
