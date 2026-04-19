"""
Factor Attribution Analysis for Committee Recommendations.

Tracks how often each conviction factor fires and whether
recommendations where that factor fired were correct.

Factors fall into two categories:
  A) Base Signal Factors — derived from signal parameters with thresholds
     (e.g., buy_pct >= 80%, exret >= 30%, RSI < 30)
  B) Conviction Modifiers — from the conviction waterfall
     (e.g., census_alignment, stale_targets, sector_concentration)

For each factor, at each time horizon (T+7 through T+365), we compute:
  - fires: how many recommendations had this factor active
  - hits: how many of those moved in the predicted direction
  - hit_rate: hits / fires
  - avg_return: mean return (alpha vs SPY) when factor fired
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# CIO v17 H7: T+30 is the system's target horizon, so it must be present.
# Older runs only had 7/14/21/28 — meaning attribution at the actual
# target horizon was never computed. T+30 added explicitly. T+7 retained
# for early-warning/leading-indicator analysis. T+90 for trend follow-up.
HORIZONS = [7, 14, 30, 60, 90]
# CIO v17 H7: Primary attribution horizon used in HTML reports + decisions.
PRIMARY_HORIZON_DAYS = 30

# Output path
ATTRIBUTION_OUTPUT_PATH = (
    Path.home() / ".weirdapps-trading" / "committee" / "factor_attribution.json"
)

# ---------------------------------------------------------------------------
# Factor Definitions
# ---------------------------------------------------------------------------

# Base signal factors: (name, description, lambda on action entry)
# Each returns True when the factor is "active" for that entry.
BASE_FACTORS = {
    "high_buy_pct": {
        "description": "Analyst buy consensus >= 80%",
        "category": "analyst",
        "test": lambda e: (e.get("buy_pct") or 0) >= 80,
    },
    "very_high_buy_pct": {
        "description": "Analyst buy consensus >= 90%",
        "category": "analyst",
        "test": lambda e: (e.get("buy_pct") or 0) >= 90,
    },
    "low_buy_pct": {
        "description": "Analyst buy consensus < 50%",
        "category": "analyst",
        "test": lambda e: 0 < (e.get("buy_pct") or 0) < 50,
    },
    "strong_exret": {
        "description": "Expected return >= 30%",
        "category": "valuation",
        "test": lambda e: (e.get("exret") or 0) >= 30,
    },
    "exceptional_exret": {
        "description": "Expected return >= 50%",
        "category": "valuation",
        "test": lambda e: (e.get("exret") or 0) >= 50,
    },
    "weak_exret": {
        "description": "Expected return < 15%",
        "category": "valuation",
        "test": lambda e: 0 < (e.get("exret") or 0) < 15,
    },
    "high_beta": {
        "description": "Beta >= 1.5 (volatile)",
        "category": "risk",
        "test": lambda e: (e.get("beta") or 1.0) >= 1.5,
    },
    "low_beta": {
        "description": "Beta <= 0.8 (defensive)",
        "category": "risk",
        "test": lambda e: 0 < (e.get("beta") or 1.0) <= 0.8,
    },
    "oversold_rsi": {
        "description": "RSI < 30 (oversold)",
        "category": "technical",
        "test": lambda e: 0 < (e.get("rsi") or 50) < 30,
    },
    "overbought_rsi": {
        "description": "RSI > 70 (overbought)",
        "category": "technical",
        "test": lambda e: (e.get("rsi") or 50) > 70,
    },
    "strong_fundamentals": {
        "description": "Fundamental score >= 70",
        "category": "fundamental",
        "test": lambda e: (e.get("fund_score") or 0) >= 70,
    },
    "weak_fundamentals": {
        "description": "Fundamental score <= 40",
        "category": "fundamental",
        "test": lambda e: 0 < (e.get("fund_score") or 0) <= 40,
    },
    "positive_technical": {
        "description": "Technical momentum >= 60",
        "category": "technical",
        "test": lambda e: (e.get("tech_momentum") or 0) >= 60,
    },
    "negative_technical": {
        "description": "Technical momentum <= 30",
        "category": "technical",
        "test": lambda e: 0 < (e.get("tech_momentum") or 0) <= 30,
    },
    "favorable_macro": {
        "description": "Macro environment favorable",
        "category": "macro",
        "test": lambda e: e.get("macro_fit") == "FAVORABLE",
    },
    "unfavorable_macro": {
        "description": "Macro environment unfavorable",
        "category": "macro",
        "test": lambda e: e.get("macro_fit") == "UNFAVORABLE",
    },
    "census_aligned": {
        "description": "Census aligned with signal",
        "category": "census",
        "test": lambda e: e.get("census") == "ALIGNED",
    },
    "census_divergent": {
        "description": "Census diverges from signal",
        "category": "census",
        "test": lambda e: "DIVERGENT" in str(e.get("census", "")),
    },
    "positive_news": {
        "description": "Positive news impact",
        "category": "news",
        "test": lambda e: "POSITIVE" in str(e.get("news_impact", "")),
    },
    "negative_news": {
        "description": "Negative news impact",
        "category": "news",
        "test": lambda e: "NEGATIVE" in str(e.get("news_impact", "")),
    },
    "high_conviction": {
        "description": "Conviction >= 70",
        "category": "composite",
        "test": lambda e: (e.get("conviction") or 0) >= 70,
    },
    "low_conviction": {
        "description": "Conviction <= 40",
        "category": "composite",
        "test": lambda e: 0 < (e.get("conviction") or 0) <= 40,
    },
}


def _is_hit(action: str, stock_return: float, spy_return: float) -> bool:
    """Determine if a recommendation was a 'hit' based on alpha vs SPY."""
    alpha = stock_return - spy_return
    if action in ("BUY", "ADD", "NEW_BUY"):
        return alpha > 0  # Stock outperformed SPY
    elif action in ("SELL", "TRIM"):
        return alpha < 0  # Stock underperformed SPY (sell was correct)
    else:  # HOLD
        return abs(alpha) < 2.0  # Stayed close to benchmark


def _fetch_prices(
    tickers: List[str],
    start_date: str,
    end_date: str,
) -> Dict[str, Dict[str, float]]:
    """
    Fetch historical close prices for multiple tickers.

    Returns: {ticker: {date_str: close_price, ...}, ...}
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed — cannot fetch prices for attribution")
        return {}

    prices = {}
    # Download in batch for efficiency
    ticker_str = " ".join(tickers)
    try:
        data = yf.download(
            ticker_str,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )
        if data.empty:
            return {}

        close = data["Close"] if "Close" in data.columns else data.get("Adj Close", data)

        for tkr in tickers:
            if len(tickers) == 1:
                col_data = close
            else:
                if tkr not in close.columns:
                    continue
                col_data = close[tkr]
            prices[tkr] = {
                d.strftime("%Y-%m-%d"): float(v)
                for d, v in col_data.dropna().items()
            }
    except Exception as exc:
        logger.error("yfinance download failed: %s", exc)

    return prices


def _nearest_price(
    price_series: Dict[str, float],
    target_date: str,
    max_gap_days: int = 5,
) -> Optional[float]:
    """Find the closest available price on or after target_date (skipping weekends/holidays)."""
    from datetime import datetime as dt

    target = dt.strptime(target_date, "%Y-%m-%d")
    for offset in range(max_gap_days):
        d = (target + timedelta(days=offset)).strftime("%Y-%m-%d")
        if d in price_series:
            return price_series[d]
    return None


def _fetch_prices_as_df(
    tickers: List[str],
    start_date: str,
    end_date: str,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Fetch prices using PriceService, returning both DataFrame and dict form.

    Returns:
        (prices_df, prices_dict) where prices_dict is {ticker: {date: price}}
        for backward compatibility with _nearest_price.
    """
    try:
        from trade_modules.price_service import PriceService
        svc = PriceService()
        df = svc.get_prices(tickers, start_date, end_date)
        # Convert to dict form for _nearest_price compatibility
        prices_dict = {}
        for col in df.columns:
            prices_dict[col] = {
                d.strftime("%Y-%m-%d"): float(v)
                for d, v in df[col].dropna().items()
            }
        return df, prices_dict
    except Exception as exc:
        logger.error("PriceService fetch failed, falling back: %s", exc)
        # Fallback to legacy _fetch_prices
        prices_dict = _fetch_prices(tickers, start_date, end_date)
        return pd.DataFrame(), prices_dict


def compute_factor_attribution(
    action_log_path: Optional[Path] = None,
    horizons: Optional[List[int]] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Main entry point: compute factor attribution from action log.

    Reads action log entries, fetches follow-up prices, and computes
    per-factor hit rates at each time horizon.

    Returns the full attribution dict (also saved to output_path).
    """
    horizons = horizons or HORIZONS
    output_path = output_path or ATTRIBUTION_OUTPUT_PATH

    # Load action log (try repo path first, then user cache)
    log_paths = [
        action_log_path,
        Path.home() / "SourceCode" / "etorotrade" / "data" / "committee" / "action_log.jsonl",
        Path.home() / ".weirdapps-trading" / "committee" / "action_log.jsonl",
    ]
    entries = []
    used_path = None
    for p in log_paths:
        if p and p.exists():
            with open(p) as f:
                entries = [json.loads(line) for line in f if line.strip()]
            used_path = p
            break

    if not entries:
        logger.warning("No action log found — cannot compute attribution")
        return {"error": "no_action_log"}

    # Filter to entries with price
    priced = [e for e in entries if e.get("price_at_recommendation") and e["price_at_recommendation"] > 0]
    logger.info(
        "Factor attribution: %d total entries, %d with price, from %s",
        len(entries), len(priced), used_path,
    )

    if not priced:
        return {"error": "no_entries_with_price", "total_entries": len(entries)}

    # Collect unique tickers and date range
    tickers = sorted(set(e["ticker"] for e in priced if e.get("ticker")))
    dates = sorted(set(e["committee_date"] for e in priced if e.get("committee_date")))
    earliest = dates[0]
    max_horizon = max(horizons)

    # Date range for price download: from earliest committee date to today + buffer
    from datetime import date as dt_date
    today = dt_date.today()
    end_date = (today + timedelta(days=5)).strftime("%Y-%m-%d")

    logger.info(
        "Fetching prices for %d tickers, %s to %s (max horizon T+%d)",
        len(tickers), earliest, end_date, max_horizon,
    )

    # Also fetch SPY for alpha computation
    all_tickers = tickers + ["SPY"]
    prices_df, prices = _fetch_prices_as_df(all_tickers, earliest, end_date)
    spy_prices = prices.get("SPY", {})

    if not spy_prices:
        logger.warning("Could not fetch SPY prices — using absolute returns")

    # -----------------------------------------------------------------------
    # Compute returns at each horizon for each entry
    # -----------------------------------------------------------------------
    enriched = []
    for entry in priced:
        tkr = entry["ticker"]
        rec_date = entry["committee_date"]
        rec_price = entry["price_at_recommendation"]
        action = entry.get("action", "HOLD")
        tkr_prices = prices.get(tkr, {})

        if not tkr_prices or rec_price <= 0:
            continue

        horizon_returns = {}
        for h in horizons:
            stock_return = None
            spy_return = 0.0

            if not prices_df.empty and tkr in prices_df.columns:
                # Trading-day offset via DataFrame index
                tkr_series = prices_df[tkr].dropna()
                sig_ts = pd.Timestamp(rec_date)
                future_dates = tkr_series.index[tkr_series.index >= sig_ts]
                if len(future_dates) <= h:
                    continue
                base_price_td = float(tkr_series.loc[future_dates[0]])
                future_price = float(tkr_series.loc[future_dates[h]])
                if base_price_td <= 0:
                    continue
                stock_return = (future_price - base_price_td) / base_price_td * 100

                # SPY benchmark with same trading-day offset
                if "SPY" in prices_df.columns:
                    spy_series = prices_df["SPY"].dropna()
                    spy_future = spy_series.index[spy_series.index >= sig_ts]
                    if len(spy_future) > h:
                        spy_base = float(spy_series.loc[spy_future[0]])
                        spy_fwd = float(spy_series.loc[spy_future[h]])
                        if spy_base > 0:
                            spy_return = (spy_fwd - spy_base) / spy_base * 100
            else:
                # Fallback: calendar-day offset with dict lookup
                target = (datetime.strptime(rec_date, "%Y-%m-%d") + timedelta(days=h)).strftime("%Y-%m-%d")
                if target > today.strftime("%Y-%m-%d"):
                    continue
                future_price = _nearest_price(tkr_prices, target)
                spy_rec_p = _nearest_price(spy_prices, rec_date) if spy_prices else None
                spy_future_p = _nearest_price(spy_prices, target) if spy_prices else None
                if future_price is None:
                    continue
                stock_return = (future_price - rec_price) / rec_price * 100
                if spy_rec_p and spy_future_p and spy_rec_p > 0:
                    spy_return = (spy_future_p - spy_rec_p) / spy_rec_p * 100

            if stock_return is None:
                continue

            alpha = stock_return - spy_return
            hit = _is_hit(action, stock_return, spy_return)

            horizon_returns[h] = {
                "stock_return": round(stock_return, 2),
                "spy_return": round(spy_return, 2),
                "alpha": round(alpha, 2),
                "hit": hit,
            }

        enriched.append({
            **entry,
            "horizon_returns": horizon_returns,
        })

    logger.info("Enriched %d entries with return data", len(enriched))

    # -----------------------------------------------------------------------
    # Factor attribution
    # -----------------------------------------------------------------------
    results = {}

    # A) Base signal factors
    for factor_name, factor_def in BASE_FACTORS.items():
        test_fn = factor_def["test"]
        results[factor_name] = _attribute_factor(
            factor_name, factor_def, enriched, horizons, test_fn,
        )

    # B) Conviction modifier factors (from waterfall)
    # Collect all unique modifier names
    all_modifiers = set()
    for e in enriched:
        wf = e.get("conviction_waterfall")
        if isinstance(wf, dict):
            all_modifiers.update(wf.keys())

    for mod_name in sorted(all_modifiers):
        factor_def = {
            "description": f"Conviction modifier: {mod_name.replace('_', ' ')}",
            "category": "modifier",
        }
        test_fn = lambda e, _m=mod_name: (
            isinstance(e.get("conviction_waterfall"), dict)
            and e["conviction_waterfall"].get(_m, 0) != 0
        )
        results[f"wf_{mod_name}"] = _attribute_factor(
            f"wf_{mod_name}", factor_def, enriched, horizons, test_fn,
        )

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    action_counts = defaultdict(int)
    for e in enriched:
        action_counts[e.get("action", "?")] += 1

    output = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data_range": f"{dates[0]} to {dates[-1]}",
        "total_entries": len(entries),
        "entries_with_price": len(priced),
        "entries_evaluated": len(enriched),
        "unique_tickers": len(tickers),
        "unique_dates": len(dates),
        "action_distribution": dict(action_counts),
        "horizons": horizons,
        "factors": results,
    }

    # Generate summary for HTML consumption and add meta key
    summary = generate_attribution_summary(output)
    output["summary"] = summary
    output["meta"] = {
        "entries_evaluated": len(enriched),
        "data_range": output["data_range"],
        "unique_tickers": len(tickers),
        "generated_at": output["generated_at"],
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Factor attribution saved to %s", output_path)

    return output


def _attribute_factor(
    name: str,
    definition: Dict[str, Any],
    entries: List[Dict],
    horizons: List[int],
    test_fn,
) -> Dict[str, Any]:
    """Compute attribution for a single factor across all entries and horizons."""
    fired_entries = [e for e in entries if test_fn(e)]
    total_fired = len(fired_entries)

    result = {
        "description": definition.get("description", name),
        "category": definition.get("category", "unknown"),
        "fires_total": total_fired,
        "fires_pct": round(total_fired / len(entries) * 100, 1) if entries else 0,
        "by_action": {},
    }

    # Group by action type
    action_groups = {"BUY": [], "ADD": [], "HOLD": [], "TRIM": [], "SELL": []}
    for e in fired_entries:
        action = e.get("action", "HOLD")
        # Normalize action names
        if action in ("NEW_BUY",):
            action = "BUY"
        if action in action_groups:
            action_groups[action].append(e)

    for action, group in action_groups.items():
        if not group:
            continue

        horizon_data = {}
        for h in horizons:
            evaluated = [e for e in group if h in e.get("horizon_returns", {})]
            if not evaluated:
                continue

            hits = sum(1 for e in evaluated if e["horizon_returns"][h]["hit"])
            alphas = [e["horizon_returns"][h]["alpha"] for e in evaluated]
            returns = [e["horizon_returns"][h]["stock_return"] for e in evaluated]

            horizon_data[f"T+{h}"] = {
                "evaluated": len(evaluated),
                "hits": hits,
                "hit_rate": round(hits / len(evaluated) * 100, 1),
                "avg_alpha": round(sum(alphas) / len(alphas), 2),
                "avg_return": round(sum(returns) / len(returns), 2),
            }

        if horizon_data:
            result["by_action"][action] = {
                "fires": len(group),
                **horizon_data,
            }

    return result


def generate_attribution_summary(
    attribution: Dict[str, Any],
    top_n: int = 15,
    primary_horizon_days: int = PRIMARY_HORIZON_DAYS,
) -> List[Dict[str, Any]]:
    """
    Generate a ranked summary of factors for the HTML report.

    CIO v17 H7: When data exists at the system's target horizon (default
    T+30), prefer it for the headline hit-rate / signal classification.
    Fall back to the next-best evaluated horizon only if T+30 has fewer
    than 3 evaluations.

    Returns a list of factor summaries sorted by predictive power, with
    the most actionable factors first. Each summary now also reports the
    *primary-horizon* hit rate and avg α explicitly so the HTML can
    surface the headline metric without extra logic.
    """
    factors = attribution.get("factors", {})
    if not factors:
        return []

    primary_key = f"T+{primary_horizon_days}"

    summaries = []
    for name, data in factors.items():
        fires = data.get("fires_total", 0)
        if fires < 3:
            continue

        # Per-horizon aggregates summed across actions for the primary horizon.
        primary_evaluated = 0
        primary_hits = 0
        primary_alpha_sum = 0.0
        primary_returns_sum = 0.0

        # Best (most-extreme) hit rate across all horizons (legacy fallback).
        best_hit_rate = None
        best_horizon = None
        best_action = None
        total_evaluated = 0

        for action, action_data in data.get("by_action", {}).items():
            for h_key, h_data in action_data.items():
                if not h_key.startswith("T+"):
                    continue
                evaluated = h_data.get("evaluated", 0)
                total_evaluated += evaluated
                if evaluated < 1:
                    continue

                # Track primary horizon (T+30 by default) aggregate.
                if h_key == primary_key:
                    primary_evaluated += evaluated
                    primary_hits += h_data.get("hits", 0)
                    primary_alpha_sum += (h_data.get("avg_alpha") or 0) * evaluated
                    primary_returns_sum += (h_data.get("avg_return") or 0) * evaluated

                # Best horizon — distance from 50%.
                if evaluated < 3:
                    continue
                hr = h_data.get("hit_rate", 0)
                if best_hit_rate is None or abs(hr - 50) > abs(best_hit_rate - 50):
                    best_hit_rate = hr
                    best_horizon = h_key
                    best_action = action

        if best_hit_rate is None or total_evaluated < 3:
            continue

        # Promote primary horizon if it has enough samples.
        primary_hit_rate = None
        primary_avg_alpha = None
        primary_avg_return = None
        primary_signal = None
        if primary_evaluated >= 3:
            primary_hit_rate = round(primary_hits / primary_evaluated * 100, 1)
            primary_avg_alpha = round(primary_alpha_sum / primary_evaluated, 2)
            primary_avg_return = round(primary_returns_sum / primary_evaluated, 2)
            if primary_hit_rate >= 60:
                primary_signal = "PREDICTIVE"
            elif primary_hit_rate >= 45:
                primary_signal = "WEAK"
            else:
                primary_signal = "CONTRARIAN"

        # Headline signal uses the primary horizon when available.
        signal = primary_signal or (
            "PREDICTIVE" if best_hit_rate >= 60 else
            "WEAK" if best_hit_rate >= 45 else "CONTRARIAN"
        )

        summaries.append({
            "name": name,
            "description": data.get("description", name),
            "category": data.get("category", "unknown"),
            "fires": fires,
            "fires_pct": data.get("fires_pct", 0),
            "best_hit_rate": best_hit_rate,
            "best_horizon": best_horizon,
            "best_action": best_action,
            "primary_horizon": primary_key if primary_hit_rate is not None else None,
            "primary_evaluated": primary_evaluated,
            "primary_hit_rate": primary_hit_rate,
            "primary_avg_alpha": primary_avg_alpha,
            "primary_avg_return": primary_avg_return,
            "primary_signal": primary_signal,
            "total_evaluated": total_evaluated,
            "signal": signal,
            "by_action": data.get("by_action", {}),
        })

    # Sort by primary horizon hit-rate when available; otherwise legacy best.
    def _sort_key(s):
        hr = s["primary_hit_rate"] if s["primary_hit_rate"] is not None else s["best_hit_rate"]
        return abs(hr - 50)

    summaries.sort(key=_sort_key, reverse=True)
    return summaries[:top_n]


def backfill_from_concordance(
    concordance_path: Optional[Path] = None,
    action_log_path: Optional[Path] = None,
) -> int:
    """
    Backfill action log entries with signal params from concordance data.

    For entries that have null buy_pct/exret/etc., look up the concordance
    entry for that ticker/date and fill in the missing fields.

    Returns count of entries updated.
    """
    conc_path = concordance_path or (
        Path.home() / ".weirdapps-trading" / "committee" / "concordance.json"
    )
    log_path = action_log_path or (
        Path.home() / "SourceCode" / "etorotrade" / "data" / "committee" / "action_log.jsonl"
    )

    if not conc_path.exists() or not log_path.exists():
        return 0

    with open(conc_path) as f:
        concordance = json.load(f)

    # Build lookup: ticker -> concordance entry
    conc_map = {}
    for entry in concordance:
        tkr = entry.get("ticker")
        if tkr:
            conc_map[tkr] = entry

    # Read action log
    with open(log_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    # Fields to backfill from concordance
    signal_fields = [
        "buy_pct", "exret", "beta", "rsi", "fund_score",
        "tech_momentum", "macro_fit", "census", "news_impact",
        "conviction_waterfall",
    ]

    updated = 0
    new_entries = []
    for entry in entries:
        tkr = entry.get("ticker")
        conc = conc_map.get(tkr, {})

        needs_update = False
        for field in signal_fields:
            if entry.get(field) is None and conc.get(field) is not None:
                entry[field] = conc[field]
                needs_update = True

        if needs_update:
            updated += 1
        new_entries.append(entry)

    if updated > 0:
        with open(log_path, "w") as f:
            for entry in new_entries:
                f.write(json.dumps(entry) + "\n")
        logger.info("Backfilled %d/%d entries from concordance", updated, len(entries))

    return updated
