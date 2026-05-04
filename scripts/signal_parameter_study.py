#!/usr/bin/env python3
"""
Signal Parameter T+30 Predictive Power Study

Loads signal_log.jsonl, fetches T+30 forward prices via yfinance,
and measures each parameter's ability to predict forward alpha.

For every parameter in the signal engine (BUY criteria + SELL triggers):
- Spearman rank correlation with T+30 alpha
- Quintile analysis (top vs bottom quintile alpha spread)
- Hit rate by quintile
- Statistical significance (p-value)

Output: console summary + JSON report.
"""

import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

# ─── Configuration ──────────────────────────────────────────────────────────

SIGNAL_LOG = Path.home() / "SourceCode/etorotrade/yahoofinance/output/signal_log.jsonl"
OUTPUT_DIR = Path.home() / ".weirdapps-trading"
PRICE_CACHE_DIR = OUTPUT_DIR / "price_cache_study"
RESULTS_FILE = OUTPUT_DIR / "parameter_study_results.json"

FORWARD_DAYS = 30  # T+30 horizon
MIN_SAMPLES = 30  # minimum observations for statistical tests
QUINTILES = 5

# Parameters to study
NUMERIC_PARAMS = [
    "upside",
    "buy_percentage",
    "exret",
    "pe_forward",
    "pe_trailing",
    "peg",
    "beta",
    "short_interest",
    "roe",
    "debt_equity",
    "pct_52w_high",
    "vix_level",
]

# ─── Price Fetching ─────────────────────────────────────────────────────────


def _load_price_cache() -> dict[str, pd.DataFrame]:
    """Load cached price data from parquet files."""
    cache = {}
    if PRICE_CACHE_DIR.exists():
        for f in PRICE_CACHE_DIR.glob("*.parquet"):
            ticker = f.stem
            try:
                cache[ticker] = pd.read_parquet(f)
            except Exception:
                pass
    return cache


def _save_price_cache(ticker: str, df: pd.DataFrame):
    """Save price data to parquet cache."""
    PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PRICE_CACHE_DIR / f"{ticker.replace('/', '_')}.parquet")


def fetch_prices_bulk(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Fetch price history for multiple tickers, using cache where available."""
    cache = _load_price_cache()
    results = {}
    to_fetch = []

    for t in tickers:
        safe_t = t.replace("/", "_")
        if safe_t in cache and len(cache[safe_t]) > 20:
            results[t] = cache[safe_t]
        else:
            to_fetch.append(t)

    if to_fetch:
        print(f"  Fetching {len(to_fetch)} tickers from yfinance (cached: {len(results)})...")
        # Batch download in chunks to avoid timeout
        chunk_size = 50
        for i in range(0, len(to_fetch), chunk_size):
            chunk = to_fetch[i : i + chunk_size]
            try:
                data = yf.download(
                    chunk,
                    start=start,
                    end=end,
                    progress=False,
                    threads=True,
                )
                if isinstance(data.columns, pd.MultiIndex):
                    for t in chunk:
                        if t in data["Close"].columns:
                            df = data["Close"][[t]].dropna()
                            df.columns = ["Close"]
                            results[t] = df
                            _save_price_cache(t, df)
                elif len(chunk) == 1:
                    t = chunk[0]
                    df = data[["Close"]].dropna()
                    results[t] = df
                    _save_price_cache(t, df)
            except Exception as e:
                print(f"  Batch {i}-{i+chunk_size} error: {e}")

    # SPY benchmark
    if "SPY" not in results:
        try:
            spy = yf.download("SPY", start=start, end=end, progress=False)
            results["SPY"] = spy[["Close"]].dropna()
            _save_price_cache("SPY", results["SPY"])
        except Exception:
            pass

    return results


def get_forward_return(prices: pd.DataFrame, signal_date: str, days: int) -> float | None:
    """Get T+N forward return from signal date."""
    try:
        dt = pd.Timestamp(signal_date[:10])
        # Find the closest trading day on or after signal date
        mask = prices.index >= dt
        if mask.sum() < 2:
            return None
        start_idx = prices.index[mask][0]
        start_pos = prices.index.get_loc(start_idx)
        end_pos = min(start_pos + days, len(prices) - 1)
        if end_pos <= start_pos:
            return None
        start_price = prices.iloc[start_pos]["Close"]
        end_price = prices.iloc[end_pos]["Close"]
        if start_price <= 0:
            return None
        return (end_price / start_price - 1) * 100
    except Exception:
        return None


# ─── Signal Log Loading ─────────────────────────────────────────────────────


def load_signal_log(max_age_days: int = 120) -> list[dict]:
    """Load signal log entries within the lookback window."""
    cutoff = datetime.now() - timedelta(days=max_age_days)
    cutoff_str = cutoff.isoformat()

    entries = []
    with open(SIGNAL_LOG) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                ts = entry.get("timestamp", "")
                if ts >= cutoff_str:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue
    return entries


def deduplicate_signals(entries: list[dict]) -> list[dict]:
    """Keep one signal per ticker per day (latest entry wins)."""
    by_ticker_day = {}
    for e in entries:
        ts = e.get("timestamp", "")[:10]
        ticker = e.get("ticker", "")
        key = f"{ticker}_{ts}"
        by_ticker_day[key] = e
    return list(by_ticker_day.values())


# ─── Statistical Analysis ───────────────────────────────────────────────────


def analyze_parameter(
    param_name: str,
    values: np.ndarray,
    alphas: np.ndarray,
) -> dict[str, Any]:
    """Compute predictive power metrics for a single parameter."""
    # Filter NaN
    mask = ~(np.isnan(values) | np.isnan(alphas))
    v = values[mask]
    a = alphas[mask]
    n = len(v)

    if n < MIN_SAMPLES:
        return {"param": param_name, "n": n, "status": "INSUFFICIENT_DATA"}

    # Spearman rank correlation
    rho, p_value = stats.spearmanr(v, a)

    # Quintile analysis
    try:
        quintile_labels = pd.qcut(v, QUINTILES, labels=False, duplicates="drop")
        q_unique = len(np.unique(quintile_labels))
    except ValueError:
        return {
            "param": param_name,
            "n": n,
            "spearman_rho": round(rho, 4),
            "p_value": round(p_value, 4),
            "status": "QUINTILE_FAIL",
        }

    q_alphas = {}
    q_hit_rates = {}
    for q in range(q_unique):
        q_mask = quintile_labels == q
        q_a = a[q_mask]
        if len(q_a) > 0:
            q_alphas[f"Q{q+1}"] = round(float(np.mean(q_a)), 2)
            q_hit_rates[f"Q{q+1}"] = round(float(np.mean(q_a > 0) * 100), 1)

    # Top vs bottom quintile spread
    q_keys = sorted(q_alphas.keys())
    if len(q_keys) >= 2:
        spread = q_alphas[q_keys[-1]] - q_alphas[q_keys[0]]
    else:
        spread = 0

    # Classification
    if abs(rho) >= 0.10 and p_value < 0.05:
        status = "PREDICTIVE"
    elif abs(rho) >= 0.05 and p_value < 0.10:
        status = "WEAK_SIGNAL"
    elif p_value >= 0.30:
        status = "NO_POWER"
    else:
        status = "MARGINAL"

    return {
        "param": param_name,
        "n": n,
        "spearman_rho": round(rho, 4),
        "p_value": round(p_value, 4),
        "quintile_spread": round(spread, 2),
        "quintile_alphas": q_alphas,
        "quintile_hit_rates": q_hit_rates,
        "top_quintile_alpha": q_alphas.get(q_keys[-1], 0) if q_keys else 0,
        "bottom_quintile_alpha": q_alphas.get(q_keys[0], 0) if q_keys else 0,
        "top_quintile_hit": q_hit_rates.get(q_keys[-1], 50) if q_keys else 50,
        "status": status,
    }


def analyze_sell_triggers(
    entries_with_alpha: list[dict],
) -> dict[str, dict]:
    """Analyze each sell trigger type's predictive power.

    CIO v36 N7 fix: include triggers regardless of signal value (the
    previous version filtered on signal=="S" only, which missed triggers
    that fire on H/I signals — the trigger LIST in signal_log is populated
    even when the final signal isn't SELL).
    """
    trigger_returns = defaultdict(list)
    no_trigger_returns = []

    for e in entries_with_alpha:
        alpha = e.get("alpha_30d")
        if alpha is None:
            continue
        triggers = e.get("sell_triggers", []) or []
        if triggers:
            for t in triggers:
                trigger_name = str(t).split(":")[0]
                trigger_returns[trigger_name].append(alpha)
        else:
            no_trigger_returns.append(alpha)

    results = {}
    baseline_alpha = np.mean(no_trigger_returns) if no_trigger_returns else 0

    for trigger, alphas in trigger_returns.items():
        n = len(alphas)
        if n < 5:
            results[trigger] = {"n": n, "status": "INSUFFICIENT_DATA"}
            continue
        avg_alpha = np.mean(alphas)
        hit_rate = np.mean(np.array(alphas) < 0) * 100  # SELL hit = stock went down
        results[trigger] = {
            "n": n,
            "avg_alpha": round(avg_alpha, 2),
            "hit_rate_down": round(hit_rate, 1),
            "vs_baseline": round(avg_alpha - baseline_alpha, 2),
            "status": "EFFECTIVE" if hit_rate > 55 else "WEAK" if hit_rate > 45 else "INVERTED",
        }

    return results


def analyze_by_signal_class(
    entries_with_alpha: list[dict],
    param_name: str,
) -> dict[str, dict]:
    """Run parameter analysis separately for BUY, HOLD, SELL signals."""
    by_signal = defaultdict(lambda: ([], []))
    for e in entries_with_alpha:
        val = e.get(param_name)
        alpha = e.get("alpha_30d")
        if val is None or alpha is None:
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue
        signal = e.get("signal", "H")
        by_signal[signal][0].append(val)
        by_signal[signal][1].append(alpha)

    results = {}
    for sig in ("B", "H", "S"):
        vals, alphas = by_signal.get(sig, ([], []))
        if len(vals) >= MIN_SAMPLES:
            results[sig] = analyze_parameter(
                f"{param_name}|{sig}",
                np.array(vals),
                np.array(alphas),
            )
    return results


# ─── Main Study ─────────────────────────────────────────────────────────────


def main():
    print("=" * 80)
    print("SIGNAL PARAMETER T+30 PREDICTIVE POWER STUDY")
    print("=" * 80)
    start_time = time.time()

    # Load signal log
    print("\n1. Loading signal log...")
    raw_entries = load_signal_log(max_age_days=120)
    print(f"   Raw entries: {len(raw_entries)}")
    entries = deduplicate_signals(raw_entries)
    print(f"   After dedup: {len(entries)}")

    # Date range
    dates = [e["timestamp"][:10] for e in entries if "timestamp" in e]
    print(f"   Date range: {min(dates)} to {max(dates)}")

    # Signal distribution
    sig_counts = defaultdict(int)
    for e in entries:
        sig_counts[e.get("signal", "?")] += 1
    print(f"   Signals: {dict(sig_counts)}")

    # Get unique tickers
    tickers = list({e["ticker"] for e in entries if e.get("ticker")})
    print(f"   Unique tickers: {len(tickers)}")

    # Fetch price data
    print("\n2. Fetching price data for T+30 forward returns...")
    price_start = min(dates)
    price_end = (datetime.strptime(max(dates), "%Y-%m-%d") + timedelta(days=45)).strftime(
        "%Y-%m-%d"
    )
    prices = fetch_prices_bulk(tickers + ["SPY"], price_start, price_end)
    print(f"   Prices loaded for {len(prices)} tickers")

    # Compute forward returns and alpha
    print("\n3. Computing T+30 forward returns and alpha...")
    entries_with_alpha = []
    no_price_count = 0
    for e in entries:
        ticker = e.get("ticker", "")
        ts = e.get("timestamp", "")
        if ticker not in prices or "SPY" not in prices:
            no_price_count += 1
            continue
        ret = get_forward_return(prices[ticker], ts, FORWARD_DAYS)
        spy_ret = get_forward_return(prices["SPY"], ts, FORWARD_DAYS)
        if ret is not None and spy_ret is not None:
            e["return_30d"] = ret
            e["spy_return_30d"] = spy_ret
            e["alpha_30d"] = ret - spy_ret
            entries_with_alpha.append(e)

    print(f"   Entries with T+30 alpha: {len(entries_with_alpha)}")
    print(f"   No price data: {no_price_count}")

    if len(entries_with_alpha) < MIN_SAMPLES:
        print("ERROR: Not enough data for analysis")
        return

    # Overall statistics
    alphas = [e["alpha_30d"] for e in entries_with_alpha]
    print(f"   Mean alpha: {np.mean(alphas):.2f}%")
    print(f"   Median alpha: {np.median(alphas):.2f}%")
    print(f"   Hit rate (alpha>0): {np.mean(np.array(alphas)>0)*100:.1f}%")

    # 4. Parameter-level analysis
    print("\n4. Analyzing parameter predictive power...")
    print("-" * 80)

    param_results = {}
    for param in NUMERIC_PARAMS:
        values = []
        param_alphas = []
        for e in entries_with_alpha:
            v = e.get(param)
            if v is not None:
                try:
                    values.append(float(v))
                    param_alphas.append(e["alpha_30d"])
                except (ValueError, TypeError):
                    pass
        if len(values) >= MIN_SAMPLES:
            result = analyze_parameter(param, np.array(values), np.array(param_alphas))
            param_results[param] = result
        else:
            param_results[param] = {"param": param, "n": len(values), "status": "INSUFFICIENT_DATA"}

    # Print results table
    print(
        f"\n{'Parameter':<20s} | {'ρ':>7s} | {'p-val':>7s} | {'Q5-Q1':>7s} | {'Hit Q5':>7s} | {'n':>5s} | {'Status'}"
    )
    print("-" * 80)
    for param in NUMERIC_PARAMS:
        r = param_results[param]
        if r.get("status") == "INSUFFICIENT_DATA":
            print(
                f"{param:<20s} | {'---':>7s} | {'---':>7s} | {'---':>7s} | {'---':>7s} | {r['n']:>5d} | INSUFFICIENT_DATA"
            )
            continue
        rho = r.get("spearman_rho", 0)
        p = r.get("p_value", 1)
        spread = r.get("quintile_spread", 0)
        hit = r.get("top_quintile_hit", 50)
        n = r.get("n", 0)
        status = r.get("status", "?")
        print(
            f"{param:<20s} | {rho:>+7.3f} | {p:>7.3f} | {spread:>+6.1f}% | {hit:>5.1f}% | {n:>5d} | {status}"
        )

    # 5. By-signal class analysis (BUY, HOLD, SELL separately)
    print("\n\n5. Parameter analysis BY SIGNAL CLASS:")
    print("=" * 80)
    for param in NUMERIC_PARAMS:
        by_sig = analyze_by_signal_class(entries_with_alpha, param)
        if by_sig:
            print(f"\n  {param}:")
            for sig, result in sorted(by_sig.items()):
                if result.get("status") != "INSUFFICIENT_DATA":
                    rho = result.get("spearman_rho", 0)
                    p = result.get("p_value", 1)
                    spread = result.get("quintile_spread", 0)
                    n = result.get("n", 0)
                    status = result.get("status", "?")
                    print(
                        f"    signal={sig}: ρ={rho:+.3f} p={p:.3f} Q5-Q1={spread:+.1f}% n={n} [{status}]"
                    )

    # 6. Sell trigger analysis
    print("\n\n6. SELL TRIGGER EFFECTIVENESS:")
    print("=" * 80)
    trigger_results = analyze_sell_triggers(entries_with_alpha)
    if trigger_results:
        print(f"\n{'Trigger':<35s} | {'n':>5s} | {'Avg α':>7s} | {'Hit%↓':>6s} | {'Status'}")
        print("-" * 75)
        for trigger, r in sorted(
            trigger_results.items(), key=lambda x: x[1].get("n", 0), reverse=True
        ):
            if r.get("status") == "INSUFFICIENT_DATA":
                print(f"{trigger:<35s} | {r['n']:>5d} | {'---':>7s} | {'---':>6s} | INSUFFICIENT")
                continue
            print(
                f"{trigger:<35s} | {r['n']:>5d} | {r['avg_alpha']:>+6.1f}% | {r['hit_rate_down']:>5.1f}% | {r['status']}"
            )
    else:
        print("  No sell triggers with sufficient data")

    # 7. PE trajectory analysis (PEF < PET as predictor)
    print("\n\n7. PE TRAJECTORY (PEF < PET) ANALYSIS:")
    print("=" * 80)
    improving = [
        (e["alpha_30d"])
        for e in entries_with_alpha
        if e.get("pe_forward")
        and e.get("pe_trailing")
        and float(e["pe_forward"]) > 0
        and float(e["pe_trailing"]) > 0
        and float(e["pe_forward"]) < float(e["pe_trailing"])
    ]
    deteriorating = [
        (e["alpha_30d"])
        for e in entries_with_alpha
        if e.get("pe_forward")
        and e.get("pe_trailing")
        and float(e["pe_forward"]) > 0
        and float(e["pe_trailing"]) > 0
        and float(e["pe_forward"]) >= float(e["pe_trailing"])
    ]
    if improving and deteriorating:
        print(
            f"  PE Improving (PEF<PET): n={len(improving)}, avg α={np.mean(improving):+.2f}%, hit={np.mean(np.array(improving)>0)*100:.1f}%"
        )
        print(
            f"  PE Deteriorating (PEF≥PET): n={len(deteriorating)}, avg α={np.mean(deteriorating):+.2f}%, hit={np.mean(np.array(deteriorating)>0)*100:.1f}%"
        )
        print(f"  Spread: {np.mean(improving) - np.mean(deteriorating):+.2f}pp")
        _, p = stats.mannwhitneyu(improving, deteriorating, alternative="two-sided")
        print(f"  Mann-Whitney p-value: {p:.4f}")

    # 8. EXRET composite vs components
    print("\n\n8. EXRET COMPOSITE vs COMPONENTS:")
    print("=" * 80)
    exret_r = param_results.get("exret", {})
    upside_r = param_results.get("upside", {})
    buypct_r = param_results.get("buy_percentage", {})
    print(
        f"  EXRET (upside×buy%): ρ={exret_r.get('spearman_rho', 'N/A')}, spread={exret_r.get('quintile_spread', 'N/A')}%"
    )
    print(
        f"  Upside alone:        ρ={upside_r.get('spearman_rho', 'N/A')}, spread={upside_r.get('quintile_spread', 'N/A')}%"
    )
    print(
        f"  Buy% alone:          ρ={buypct_r.get('spearman_rho', 'N/A')}, spread={buypct_r.get('quintile_spread', 'N/A')}%"
    )

    # Save results
    elapsed = time.time() - start_time
    report = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "horizon_days": FORWARD_DAYS,
        "total_entries": len(entries),
        "entries_with_alpha": len(entries_with_alpha),
        "date_range": {"start": min(dates), "end": max(dates)},
        "signal_distribution": dict(sig_counts),
        "overall_stats": {
            "mean_alpha": round(np.mean(alphas), 2),
            "median_alpha": round(np.median(alphas), 2),
            "hit_rate": round(np.mean(np.array(alphas) > 0) * 100, 1),
        },
        "parameter_results": param_results,
        "sell_trigger_results": trigger_results,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n\nResults saved to {RESULTS_FILE}")
    print(f"Elapsed: {elapsed:.1f}s")

    # Summary recommendations
    print(f"\n{'=' * 80}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 80}")
    predictive = [p for p, r in param_results.items() if r.get("status") == "PREDICTIVE"]
    weak = [p for p, r in param_results.items() if r.get("status") == "WEAK_SIGNAL"]
    no_power = [p for p, r in param_results.items() if r.get("status") == "NO_POWER"]
    print(f"  PREDICTIVE (keep, consider amplifying): {predictive or 'none'}")
    print(f"  WEAK SIGNAL (keep, monitor): {weak or 'none'}")
    print(f"  NO POWER (consider disabling): {no_power or 'none'}")


if __name__ == "__main__":
    main()
