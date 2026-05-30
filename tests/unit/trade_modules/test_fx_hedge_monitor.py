from trade_modules.fx_hedge_monitor import compute_currency_exposure, hedge_pct_table


def test_currency_buckets_and_usd_bloc():
    weights = {"NVDA": 30.0, "GLD": 10.0, "6758.T": 15.0, "DTE.DE": 20.0, "0700.HK": 25.0}
    exp = compute_currency_exposure(weights)
    by = {e["currency"]: e["weight_pct"] for e in exp["by_currency"]}
    assert round(by["USD"], 1) == 40.0  # NVDA + GLD
    assert round(by["JPY"], 1) == 15.0  # 6758.T
    assert round(by["EUR"], 1) == 20.0  # DTE.DE
    assert round(by["HKD"], 1) == 25.0  # 0700.HK
    assert round(exp["usd_bloc_pct"], 1) == 65.0  # USD(40) + HKD(25), USD-pegged
    assert round(exp["non_eur_pct"], 1) == 80.0  # everything except EUR(20)


def test_hedge_pct_table_ratios():
    rows = hedge_pct_table(usd_bloc_pct=65.0, ratios=(0.0, 0.5, 1.0))
    got = {r["ratio"]: r["hedge_pct_of_equity"] for r in rows}
    assert got[0.0] == 0.0
    assert got[0.5] == 32.5
    assert got[1.0] == 65.0


def test_empty_weights():
    exp = compute_currency_exposure({})
    assert exp["by_currency"] == [] and exp["usd_bloc_pct"] == 0.0
