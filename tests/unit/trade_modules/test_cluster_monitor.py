import numpy as np
import pandas as pd

from trade_modules.cluster_monitor import check_cluster_alerts, compute_cluster_exposure


def _correlated_returns(seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.01, 250)
    return pd.DataFrame(
        {
            "A": base + rng.normal(0, 0.001, 250),  # A,B,C move together (~0.99)
            "B": base + rng.normal(0, 0.001, 250),
            "C": base + rng.normal(0, 0.001, 250),
            "D": rng.normal(0, 0.01, 250),  # independent
        }
    )


def test_cluster_exposure_sums_weights():
    weights = {"A": 0.15, "B": 0.14, "C": 0.10, "D": 0.05}  # A+B+C = 39%
    exposures = compute_cluster_exposure(
        weights, _correlated_returns(), threshold=0.7, min_cluster_size=3
    )
    assert exposures, "expected at least one cluster"
    top = max(exposures, key=lambda c: c["combined_weight_pct"])
    assert set(top["tickers"]) == {"A", "B", "C"}
    assert round(top["combined_weight_pct"], 0) == 39.0


def test_alerts_fire_on_thresholds():
    exposures = [{"tickers": ["A", "B", "C"], "combined_weight_pct": 39.0, "avg_correlation": 0.95}]
    alerts = check_cluster_alerts(exposures, soft_pct=30.0, hard_pct=35.0)
    assert alerts and alerts[0]["level"] == "HARD"


def test_no_alert_below_soft():
    exposures = [{"tickers": ["A", "B"], "combined_weight_pct": 12.0, "avg_correlation": 0.8}]
    assert check_cluster_alerts(exposures, soft_pct=30.0, hard_pct=35.0) == []
