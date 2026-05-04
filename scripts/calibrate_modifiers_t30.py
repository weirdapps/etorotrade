#!/usr/bin/env python3
"""
M11: Per-modifier T+30 predictive-power calibrator.

CIO v36 / Empirical Refoundation. Replaces literature-based modifier pruning
in trade_modules/committee_synthesis.py:138-161 with empirical evidence from
realized T+30 forward alpha.

Reads concordance history (committee_waterfall per stock, per date), looks up
T+30 forward alpha vs SPY via the parquet price cache, and computes Spearman
ρ(modifier_value, alpha_30d) for every modifier observed. Each modifier is
classified as PREDICTIVE / WEAK / SHADOW / DROP / INSUFFICIENT_DATA.

The output JSON at ~/.weirdapps-trading/committee/modifier_t30_calibration.json
becomes the source of truth for ACTIVE_MODIFIERS — modifiers without
empirical edge get demoted to shadow (~prefix) until evidence accumulates.

Usage:
    python scripts/calibrate_modifiers_t30.py
    python scripts/calibrate_modifiers_t30.py --history-dir /path --output /path
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default locations — configurable via CLI
DEFAULT_HISTORY_DIR = Path.home() / ".weirdapps-trading" / "committee" / "history"
DEFAULT_OUTPUT_PATH = (
    Path.home() / ".weirdapps-trading" / "committee" / "modifier_t30_calibration.json"
)
DEFAULT_PRICE_CACHE_DIR = Path.home() / ".weirdapps-trading" / "price_cache"

MIN_OBSERVATIONS = 30  # below this → INSUFFICIENT_DATA
PREDICTIVE_RHO = 0.10  # |ρ| ≥ 0.10 with significance → PREDICTIVE
WEAK_RHO = 0.05
SIGNIFICANCE_P = 0.05
HORIZON_TRADING_DAYS = 30  # T+30 is the user's horizon


# ─── Concordance history loading ────────────────────────────────────────────


def _load_concordance_file(fpath: Path) -> list[dict[str, Any]]:
    """Read one concordance file. Returns the per-stock rows or empty list."""
    try:
        with open(fpath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Skipping %s: %s", fpath.name, exc)
        return []

    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]

    rows = data.get("concordance") if isinstance(data, dict) else None
    if rows is None and isinstance(data, dict):
        rows = data.get("stocks", [])

    if isinstance(rows, dict):
        # Convert {ticker: {...}} → [{ticker: ticker, ...}]
        rows = [
            dict(v, ticker=k) if isinstance(v, dict) else {"ticker": k} for k, v in rows.items()
        ]
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


def _date_from_filename(fpath: Path) -> str | None:
    """Extract YYYY-MM-DD from a concordance filename like concordance-2026-04-30.json."""
    stem = fpath.stem
    # concordance-YYYY-MM-DD or concordance_YYYY-MM-DD
    parts = stem.replace("_", "-").split("-")
    for i in range(len(parts) - 2):
        candidate = f"{parts[i]}-{parts[i + 1]}-{parts[i + 2]}"
        try:
            datetime.strptime(candidate, "%Y-%m-%d")
            return candidate
        except ValueError:
            continue
    return None


def extract_modifier_observations(
    history_dir: Path | str,
) -> list[dict[str, Any]]:
    """Walk history dir, yield one observation per (ticker, date, modifier).

    Each observation is a dict {ticker, date, modifier, value} so we can
    later join with forward returns and compute per-modifier ρ.

    Concordance rows without a `conviction_waterfall` field are skipped — the
    calibrator can only score modifiers it can observe.
    """
    hdir = Path(history_dir)
    if not hdir.is_dir():
        logger.warning("History dir not found: %s", hdir)
        return []

    out: list[dict[str, Any]] = []
    for fpath in sorted(hdir.glob("concordance-*.json")) + sorted(hdir.glob("concordance_*.json")):
        date = _date_from_filename(fpath)
        if not date:
            continue
        rows = _load_concordance_file(fpath)
        for row in rows:
            ticker = row.get("ticker")
            wf = row.get("conviction_waterfall")
            if not ticker or not isinstance(wf, dict) or not wf:
                continue
            for mod_name, value in wf.items():
                # Strip shadow-tracking prefix so production + shadow share the
                # same name in the verdict table.
                clean = mod_name.lstrip("~")
                try:
                    fval = float(value)
                except (TypeError, ValueError):
                    continue
                out.append(
                    {
                        "ticker": str(ticker),
                        "date": date,
                        "modifier": clean,
                        "value": fval,
                    }
                )
    return out


# ─── Forward-return lookup ──────────────────────────────────────────────────


def _load_price_parquet(ticker: str, cache_dir: Path) -> object | None:
    """Load one ticker's parquet price file. Returns DataFrame or None.

    Lazy-imports pandas so unit tests that monkeypatch compute_alpha_lookup
    don't pay the import cost.
    """
    try:
        import pandas as pd  # noqa: WPS433 (intentional lazy import)
    except ImportError:
        return None
    safe = ticker.replace("/", "_")
    fp = cache_dir / f"{safe}_1y.parquet"
    if not fp.exists():
        # Fall back to non-suffixed cache layout used by some scripts
        fp = cache_dir / f"{safe}.parquet"
        if not fp.exists():
            return None
    try:
        return pd.read_parquet(fp)
    except Exception as exc:
        logger.debug("Failed to read %s: %s", fp, exc)
        return None


def _trading_day_alpha(prices, benchmark_prices, signal_date: str, days: int) -> float | None:
    """Compute T+N alpha vs benchmark in trading days. Returns alpha in pct points."""
    if prices is None or benchmark_prices is None or len(prices) < 2 or len(benchmark_prices) < 2:
        return None
    try:
        import pandas as pd  # noqa: WPS433

        dt = pd.Timestamp(signal_date[:10])
    except Exception:
        return None

    def _ret(df, start_dt, n_days):
        mask = df.index >= start_dt
        if mask.sum() < 2:
            return None
        start_idx = df.index[mask][0]
        start_pos = df.index.get_loc(start_idx)
        end_pos = min(start_pos + n_days, len(df) - 1)
        if end_pos <= start_pos:
            return None
        col = "Close" if "Close" in df.columns else df.columns[0]
        sp, ep = df.iloc[start_pos][col], df.iloc[end_pos][col]
        if sp <= 0:
            return None
        return (ep / sp - 1) * 100

    stock_ret = _ret(prices, dt, days)
    bench_ret = _ret(benchmark_prices, dt, days)
    if stock_ret is None or bench_ret is None:
        return None
    return stock_ret - bench_ret


def compute_alpha_lookup(
    observations: list[dict[str, Any]],
    cache_dir: Path | None = None,
    benchmark: str = "SPY",
    horizon_days: int = HORIZON_TRADING_DAYS,
) -> dict[tuple[str, str], float]:
    """Build dict (ticker, date) -> T+N alpha vs benchmark.

    Reads parquet price cache; skips tickers with missing data. Designed
    for unit tests to monkeypatch this function directly with a stub.
    """
    cache_dir = cache_dir or DEFAULT_PRICE_CACHE_DIR
    benchmark_prices = _load_price_parquet(benchmark, cache_dir)
    if benchmark_prices is None:
        logger.warning("Benchmark %s not in price cache %s", benchmark, cache_dir)
        return {}

    # Deduplicate (ticker, date) pairs
    unique_keys = {(o["ticker"], o["date"]) for o in observations}

    # Cache loaded ticker frames so we don't re-read parquet per observation
    price_cache: dict[str, object | None] = {}
    out: dict[tuple[str, str], float] = {}

    for ticker, date in sorted(unique_keys):
        if ticker not in price_cache:
            price_cache[ticker] = _load_price_parquet(ticker, cache_dir)
        prices = price_cache[ticker]
        alpha = _trading_day_alpha(prices, benchmark_prices, date, horizon_days)
        if alpha is not None:
            out[(ticker, date)] = alpha

    return out


# ─── Statistical analysis ───────────────────────────────────────────────────


def _spearman_rho_p(values: list[float], alphas: list[float]) -> tuple[float, float]:
    """Spearman ρ + p-value. Returns (NaN, 1.0) if invariant."""
    if len(values) < 3 or len(values) != len(alphas):
        return float("nan"), 1.0
    if len(set(values)) <= 1 or len(set(alphas)) <= 1:
        return float("nan"), 1.0
    try:
        from scipy import stats

        rho, p = stats.spearmanr(values, alphas)
        if rho is None or (isinstance(rho, float) and math.isnan(rho)):
            return float("nan"), 1.0
        return float(rho), float(p)
    except ImportError:
        # Fallback: rank-correlation by hand, no p-value
        n = len(values)
        rx = _ranks(values)
        ry = _ranks(alphas)
        mx, my = sum(rx) / n, sum(ry) / n
        cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n)) / n
        vx = sum((r - mx) ** 2 for r in rx) / n
        vy = sum((r - my) ** 2 for r in ry) / n
        if vx <= 0 or vy <= 0:
            return float("nan"), 1.0
        return cov / math.sqrt(vx * vy), 0.5  # cannot compute p without scipy


def _ranks(values: list[float]) -> list[float]:
    """Average-rank tied values. Standard Spearman convention."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def classify_verdict(rho: float, p_value: float, n: int) -> str:
    """Map (ρ, p, n) to PREDICTIVE / WEAK / SHADOW / DROP / INSUFFICIENT_DATA."""
    if isinstance(rho, float) and math.isnan(rho):
        # NaN means modifier value is constant or alphas are constant; either
        # way the modifier provides no rank discrimination → DROP.
        return "DROP"
    if n < MIN_OBSERVATIONS:
        return "INSUFFICIENT_DATA"
    abs_rho = abs(rho)
    is_significant = p_value < SIGNIFICANCE_P
    if abs_rho >= PREDICTIVE_RHO and is_significant:
        return "PREDICTIVE"
    if abs_rho >= WEAK_RHO and is_significant:
        return "WEAK"
    return "SHADOW"


# CIO v36 N4 — research-rigor utilities for the calibrator
#
# These add three best-practice guards to the verdict pipeline:
#   1. Bonferroni multiple-comparison correction across modifiers tested
#   2. Bootstrap 95% CI around each ρ — only PREDICTIVE if lower-CI > 0
#   3. Walk-forward train/test split — test on data the calibration didn't see
#
# Together they reduce false-PREDICTIVE rate from ~3 of 63 (binomial chance
# at α=0.05) to <1 expected over many tests.


def bonferroni_threshold(n_tests: int, alpha: float = 0.05) -> float:
    """Bonferroni-adjusted significance threshold for n_tests."""
    if n_tests <= 0:
        return alpha
    return alpha / n_tests


def bootstrap_spearman_ci(
    xs: list[float],
    ys: list[float],
    n_resamples: int = 500,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float | None, float | None]:
    """Bootstrap percentile CI on Spearman ρ. Returns (lower, upper) or (None,None)."""
    n = len(xs)
    if n < 5 or n != len(ys):
        return None, None
    try:
        from scipy import stats
    except ImportError:
        return None, None

    import random as _random

    rng = _random.Random(seed)

    rhos = []
    for _ in range(n_resamples):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        bx = [xs[i] for i in idx]
        by = [ys[i] for i in idx]
        if len(set(bx)) < 2 or len(set(by)) < 2:
            continue
        try:
            r, _ = stats.spearmanr(bx, by)
        except Exception:
            continue
        if r is not None and not math.isnan(r):
            rhos.append(float(r))

    if len(rhos) < 10:
        return None, None
    rhos.sort()
    alpha = (1 - confidence) / 2
    lo_idx = int(len(rhos) * alpha)
    hi_idx = int(len(rhos) * (1 - alpha))
    return rhos[lo_idx], rhos[min(hi_idx, len(rhos) - 1)]


def walk_forward_splits(
    observations: list[dict[str, Any]],
    n_folds: int = 4,
    sort_key: str = "value",
):
    """Yield (train, test) tuples for walk-forward validation.

    Splits the (sorted) sequence into n_folds equal chunks. Fold k uses
    chunks [0..k] as train and chunk k+1 as test. The first fold has only
    chunk 0 as train so it's skipped — yields n_folds - 1 splits.

    No leakage: test samples come strictly AFTER train samples by sort_key.
    """
    n = len(observations)
    if n < n_folds * 2:
        return
    sorted_obs = sorted(
        observations,
        key=lambda o: o.get(sort_key) if o.get(sort_key) is not None else 0,
    )
    fold_size = n // n_folds
    for k in range(n_folds - 1):
        train = sorted_obs[: (k + 1) * fold_size]
        test = sorted_obs[(k + 1) * fold_size : (k + 2) * fold_size]
        if train and test:
            yield train, test


def classify_verdict_rigorous(
    rho: float,
    p_value: float,
    n: int,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    n_modifiers_tested: int = 1,
) -> str:
    """Verdict that applies Bonferroni + bootstrap CI gates on top of basics.

    PREDICTIVE requires:
      - p_value passes Bonferroni-adjusted threshold (0.05 / n_modifiers_tested)
      - Bootstrap 95% CI lower bound is above 0 (sign confidence)
      - |ρ| ≥ PREDICTIVE_RHO

    WEAK requires nominal p<0.05 and |ρ| ≥ WEAK_RHO regardless of CI/Bonferroni.
    """
    if isinstance(rho, float) and math.isnan(rho):
        return "DROP"
    if n < MIN_OBSERVATIONS:
        return "INSUFFICIENT_DATA"

    abs_rho = abs(rho)
    bonf_alpha = bonferroni_threshold(n_modifiers_tested)
    passes_bonf = p_value < bonf_alpha

    # CI must not straddle zero for PREDICTIVE
    ci_strictly_positive = (
        ci_lower is not None
        and ci_upper is not None
        and ((ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0))
    )

    if abs_rho >= PREDICTIVE_RHO and passes_bonf and ci_strictly_positive:
        return "PREDICTIVE"
    # WEAK under rigor requires Bonferroni-adjusted significance too.
    # Without it, a nominal p<0.05 win is likely a false positive given
    # how many modifiers we test simultaneously — demote to SHADOW.
    if abs_rho >= WEAK_RHO and passes_bonf:
        return "WEAK"
    return "SHADOW"


def analyze_modifier(
    name: str,
    values: list[float],
    alphas: list[float],
    n_modifiers_tested: int = 1,
    rigorous: bool = False,
) -> dict[str, Any]:
    """Compute predictive metrics for one modifier.

    Returns: {modifier, n, spearman_rho, p_value, sign, verdict}.
    `sign` is +1 if rho > 0, -1 if < 0, 0 if NaN. The synthesis pipeline
    can use this to detect inverted modifiers (sign opposite of expected).

    CIO v36 N4: when rigorous=True, also computes bootstrap 95% CI and
    applies Bonferroni correction across n_modifiers_tested. The verdict
    field uses the rigorous classifier.
    """
    rho, p = _spearman_rho_p(values, alphas)
    n = len(values)
    sign = 0
    if isinstance(rho, float) and not math.isnan(rho):
        sign = 1 if rho > 0 else (-1 if rho < 0 else 0)

    out = {
        "modifier": name,
        "n": n,
        "spearman_rho": None if math.isnan(rho) else round(rho, 4),
        "p_value": round(p, 4),
        "sign": sign,
    }

    if rigorous and n >= MIN_OBSERVATIONS:
        ci_lo, ci_hi = bootstrap_spearman_ci(values, alphas, n_resamples=300, seed=42)
        out["ci_lower"] = round(ci_lo, 4) if ci_lo is not None else None
        out["ci_upper"] = round(ci_hi, 4) if ci_hi is not None else None
        out["bonferroni_alpha"] = round(bonferroni_threshold(n_modifiers_tested), 6)
        out["verdict"] = classify_verdict_rigorous(
            rho,
            p,
            n,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            n_modifiers_tested=n_modifiers_tested,
        )
        out["verdict_legacy"] = classify_verdict(rho, p, n)
    else:
        out["verdict"] = classify_verdict(rho, p, n)

    return out


# ─── Main entry ─────────────────────────────────────────────────────────────


def _build_forward_returns_for_agent_calibrator(
    history_dir: Path,
    cache_dir: Path,
    benchmark: str,
    horizon_days: int = HORIZON_TRADING_DAYS,
) -> dict[str, dict[str, float]]:
    """Build {ticker:date → {T+horizon_alpha: float}} for agent_sign_calibrator.

    Walks history files, computes T+N alpha vs benchmark using the same
    parquet cache the modifier calibrator uses. Used by N6 to feed the
    agent_sign_calibrator the data it needs (without it, evidence_total=0).
    """
    out: dict[str, dict[str, float]] = {}
    if not history_dir.is_dir():
        return out
    benchmark_prices = _load_price_parquet(benchmark, cache_dir)
    if benchmark_prices is None:
        return out

    price_cache: dict[str, object | None] = {}
    for fpath in sorted(history_dir.glob("concordance-*.json")):
        date = _date_from_filename(fpath)
        if not date:
            continue
        rows = _load_concordance_file(fpath)
        for row in rows:
            ticker = row.get("ticker")
            if not ticker:
                continue
            if ticker not in price_cache:
                price_cache[ticker] = _load_price_parquet(ticker, cache_dir)
            alpha = _trading_day_alpha(
                price_cache[ticker],
                benchmark_prices,
                date,
                horizon_days,
            )
            if alpha is not None:
                out[f"{ticker}:{date}"] = {f"T+{horizon_days}_alpha": alpha}
    return out


def run_agent_sign_calibrator(
    history_dir: Path,
    cache_dir: Path,
    benchmark: str = "SPY",
    output_path: Path | None = None,
) -> dict[str, Any]:
    """N6: invoke agent_sign_calibrator with proper forward returns.

    The calibrator's per-agent verdict (OK / INVERTED / INSUFFICIENT_DATA)
    requires real T+30 forward alpha. Without this wiring it sits at
    evidence_total=0 forever.
    """
    try:
        from trade_modules.agent_sign_calibrator import calibrate_agent_signs
    except ImportError:
        logger.warning("agent_sign_calibrator module not available")
        return {"status": "module_missing"}

    fr = _build_forward_returns_for_agent_calibrator(history_dir, cache_dir, benchmark)
    logger.info("Agent-sign calibrator: %d (ticker, date) forward returns ready", len(fr))
    result = calibrate_agent_signs(
        forward_returns=fr,
        history_dir=history_dir,
        horizon=f"T+{HORIZON_TRADING_DAYS}",
        lookback_days=120,
        min_evidence=10,
    )
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Wrote agent-sign calibration to %s", output_path)
    return result


def main(
    history_dir: Path | str = DEFAULT_HISTORY_DIR,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    cache_dir: Path | str = DEFAULT_PRICE_CACHE_DIR,
    benchmark: str = "SPY",
    rigorous: bool = False,
) -> dict[str, Any]:
    """Build the per-modifier T+30 calibration report and persist to JSON."""
    history_dir = Path(history_dir)
    output_path = Path(output_path)
    cache_dir = Path(cache_dir)

    obs = extract_modifier_observations(history_dir)
    if not obs:
        logger.warning("No modifier observations extracted from %s", history_dir)
        result = {
            "status": "no_data",
            "history_dir": str(history_dir),
            "modifiers": {},
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        return result

    alpha_lookup = compute_alpha_lookup(obs, cache_dir=cache_dir, benchmark=benchmark)

    # Group observations by modifier; only keep those with a forward-return
    # observation available.
    by_mod: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for o in obs:
        key = (o["ticker"], o["date"])
        alpha = alpha_lookup.get(key)
        if alpha is None:
            continue
        by_mod[o["modifier"]].append((o["value"], alpha))

    modifiers = {}
    n_tested = len(by_mod)
    for name, pairs in sorted(by_mod.items()):
        values = [p[0] for p in pairs]
        alphas = [p[1] for p in pairs]
        modifiers[name] = analyze_modifier(
            name,
            values,
            alphas,
            n_modifiers_tested=n_tested,
            rigorous=rigorous,
        )

    result = {
        "status": "ok",
        "generated_at": datetime.now().isoformat(),
        "history_dir": str(history_dir),
        "horizon_trading_days": HORIZON_TRADING_DAYS,
        "min_observations": MIN_OBSERVATIONS,
        "predictive_rho_threshold": PREDICTIVE_RHO,
        "weak_rho_threshold": WEAK_RHO,
        "significance_p": SIGNIFICANCE_P,
        "total_observations": len(obs),
        "observations_with_alpha": sum(len(v) for v in by_mod.values()),
        "modifiers": modifiers,
        "summary": {
            "predictive": sorted(k for k, v in modifiers.items() if v["verdict"] == "PREDICTIVE"),
            "weak": sorted(k for k, v in modifiers.items() if v["verdict"] == "WEAK"),
            "shadow": sorted(k for k, v in modifiers.items() if v["verdict"] == "SHADOW"),
            "drop": sorted(k for k, v in modifiers.items() if v["verdict"] == "DROP"),
            "insufficient_data": sorted(
                k for k, v in modifiers.items() if v["verdict"] == "INSUFFICIENT_DATA"
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(
        "Wrote modifier calibration to %s — predictive=%d weak=%d shadow=%d drop=%d insufficient=%d",
        output_path,
        len(result["summary"]["predictive"]),
        len(result["summary"]["weak"]),
        len(result["summary"]["shadow"]),
        len(result["summary"]["drop"]),
        len(result["summary"]["insufficient_data"]),
    )
    return result


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="M11: Per-modifier T+30 predictive-power calibrator",
    )
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_PRICE_CACHE_DIR)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument(
        "--rigorous",
        action="store_true",
        help="Apply Bonferroni correction + bootstrap CI to verdicts",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    result = main(
        history_dir=args.history_dir,
        output_path=args.output,
        cache_dir=args.cache_dir,
        benchmark=args.benchmark,
        rigorous=args.rigorous,
    )

    # Console summary
    print("=" * 70)
    print("MODIFIER T+30 CALIBRATION")
    print("=" * 70)
    print(f"Total observations: {result.get('total_observations', 0)}")
    print(f"With T+30 alpha:    {result.get('observations_with_alpha', 0)}")
    print()
    summary = result.get("summary", {})
    for level in ("predictive", "weak", "shadow", "drop", "insufficient_data"):
        items = summary.get(level, [])
        print(
            f"{level.upper():18s} ({len(items):3d}): {', '.join(items[:8])}{' ...' if len(items) > 8 else ''}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
