import numpy as np
import pandas as pd

from trade_modules.cluster_monitor import (
    check_cluster_alerts,
    compute_cluster_exposure,
    exposure_from_known_clusters,
)


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


def test_exposure_from_known_clusters_sums_weights_stocks_key():
    weights = {"NVDA": 12.0, "MSFT": 11.0, "AVGO": 9.0, "JPM": 4.0}  # percent
    clusters = [{"stocks": ["NVDA", "MSFT", "AVGO"], "avg_correlation": 0.82}]
    out = exposure_from_known_clusters(weights, clusters)
    assert len(out) == 1
    assert out[0]["tickers"] == ["NVDA", "MSFT", "AVGO"]
    assert out[0]["combined_weight_pct"] == 32.0
    assert out[0]["avg_correlation"] == 0.82


def test_exposure_from_known_clusters_reads_tickers_key():
    weights = {"A": 20.0, "B": 16.0}
    clusters = [{"tickers": ["A", "B"], "avg_correlation": 0.9}]
    out = exposure_from_known_clusters(weights, clusters)
    assert out[0]["combined_weight_pct"] == 36.0


def test_exposure_from_known_clusters_fraction_mode():
    weights = {"A": 0.2, "B": 0.16}
    clusters = [{"stocks": ["A", "B"]}]
    out = exposure_from_known_clusters(weights, clusters, weights_in_pct=False)
    assert out[0]["combined_weight_pct"] == 36.0


def test_exposure_from_known_clusters_aligns_one_to_one():
    weights = {"A": 10.0}
    clusters = [{"stocks": []}, {"stocks": ["A"]}]
    out = exposure_from_known_clusters(weights, clusters)
    assert len(out) == 2
    assert out[0]["combined_weight_pct"] == 0.0
    assert out[1]["combined_weight_pct"] == 10.0


def test_exposure_then_alert_pipeline_fires_hard():
    weights = {"NVDA": 14.0, "MSFT": 13.0, "AVGO": 9.0}  # 36%
    clusters = [{"stocks": ["NVDA", "MSFT", "AVGO"], "avg_correlation": 0.85}]
    exposures = exposure_from_known_clusters(weights, clusters)
    alerts = check_cluster_alerts(exposures)
    assert alerts and alerts[0]["level"] == "HARD"
