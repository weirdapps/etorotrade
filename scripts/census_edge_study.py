#!/usr/bin/env python3
"""Census Sub-Group Edge Study.

Research question: do any sub-groups of eToro "Popular Investors" have
predictive edge -- i.e., do the stocks a sub-group *collectively* holds
outperform the market (SPY) over the following 7 and 30 calendar days?

Design
------
For ~monthly census snapshots (first available snapshot per calendar month,
2025-06 .. 2026-06), we slice the ~1500 tracked investors into sub-groups,
build each group's top-15 "consensus book" from LONG positions (weighted by
``investmentPct``), then measure the equal-weighted forward return of that book
vs SPY over the same horizon. We aggregate the per-date *excess* returns per
(group, horizon) and test significance with Benjamini-Hochberg FDR (alpha=0.10)
and the Harvey-Liu-Zhu |t| >= 3.0 hurdle.

Schema deviations from the original task spec (verified against the archive)
---------------------------------------------------------------------------
1. ``instruments`` schema evolved. 2025-05/06 snapshots store a *flat list* of
   ``{instrumentId, symbol, ...}`` (no ``instrumentTypeID``); 2025-07 onward
   store ``instruments.details`` with ``symbolFull`` + ``instrumentTypeID``
   (5 == stock). ``build_id_symbol_map`` handles BOTH.
2. ``thisWeekGain`` is *never populated* in any of the 554 snapshots (0/1500 in
   every file checked). The ``hot_hand`` sub-group therefore falls back to
   ``dailyGain`` (populated 1500/1500) as the momentum proxy. This is a forced
   deviation, flagged in the report.
3. ``positionsCount`` lives at ``portfolio.positionsCount`` (top-level is null).
   The ``diversified`` group reads the nested field.

Pure functions (unit-tested) are kept free of IO/network. Snapshot loading,
the yfinance panel fetch, and ``main`` are marked ``# pragma: no cover``.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

CENSUS_DIR = Path.home() / "SourceCode" / "etoro_census" / "archive" / "data"
BENCHMARK = "SPY"
HORIZONS = (7, 30)
TOP_N = 15
DECILE = 0.10
OUTPUT_REPORT = Path.home() / "Downloads" / "census_edge_study_findings.md"

# Sub-group definitions. Each entry: (metric_field, nested_key_or_None, highest?)
# ``metric_field`` is read off each investor dict (or ``portfolio`` when nested).
# ``highest=True`` -> top decile; ``highest=False`` -> bottom decile.
# NOTE: hot_hand uses dailyGain because thisWeekGain is never populated (see
# module docstring, deviation #2).
SUBGROUP_SPECS: dict[str, tuple[str, str | None, bool]] = {
    "most_profitable": ("gain", None, True),
    "hot_hand": ("dailyGain", None, True),  # proxy for thisWeekGain (null in data)
    "diversified": ("positionsCount", "portfolio", True),
    "most_copied": ("copiers", None, True),
    "low_risk": ("riskScore", None, False),
    "crowd": (None, None, True),  # baseline: ALL investors
}

# The multiple-testing family excludes the crowd baseline.
FAMILY_GROUPS = [g for g in SUBGROUP_SPECS if g != "crowd"]

HLZ_HURDLE = 3.0  # Harvey-Liu-Zhu |t| hurdle for a "new factor"

# The OLD (2025-05/06) flat-list instrument schema carries no ``instrumentTypeID``,
# so the "stocks only (typeID==5)" filter can't be applied there. To keep those
# dates comparable to the type-filtered new-schema dates (a stock study), drop
# known crypto pseudo-tickers on the old-schema path only. (Bare crypto tickers
# like ``BTC``/``ETH`` also mis-resolve in yfinance to Grayscale equity trusts,
# not the coins, which would silently corrupt the book.)
_CRYPTO_TICKERS = frozenset(
    {
        "BTC",
        "ETH",
        "XRP",
        "ADA",
        "SOL",
        "DOGE",
        "LTC",
        "BCH",
        "DOT",
        "EOS",
        "XLM",
        "TRX",
        "NEO",
        "ZEC",
        "DASH",
        "MIOTA",
        "IOTA",
        "XTZ",
        "BNB",
        "LINK",
        "UNI",
        "AAVE",
        "ALGO",
        "ATOM",
        "AVAX",
        "COMP",
        "FIL",
        "GRT",
        "MKR",
        "SUSHI",
        "YFI",
        "MANA",
        "SAND",
        "SHIB",
        "MATIC",
        "LUNA",
        "FTT",
        "CRO",
        "NEAR",
        "ICP",
    }
)


# --------------------------------------------------------------------------- #
# Pure functions (unit-tested)
# --------------------------------------------------------------------------- #


def build_id_symbol_map(instruments: Any) -> dict[int, str]:
    """Map ``instrumentId -> symbol`` from a snapshot's ``instruments`` blob.

    Handles both archive schemas:
      * OLD (2025-05/06): a flat ``list`` of ``{instrumentId, symbol, ...}``.
      * NEW (2025-07+):   a ``dict`` with ``details`` -> list of
        ``{instrumentId, symbolFull, instrumentTypeID, ...}``. When
        ``instrumentTypeID`` is present we keep stocks only (typeID == 5).

    Symbols are upper-cased and stripped. Entries without a usable id or symbol
    are skipped.
    """
    id_to_symbol: dict[int, str] = {}

    if isinstance(instruments, dict):
        details = instruments.get("details") or []
        old_schema = False
    elif isinstance(instruments, list):
        details = instruments
        old_schema = True
    else:
        return id_to_symbol

    for entry in details:
        if not isinstance(entry, dict):
            continue
        iid = entry.get("instrumentId")
        if iid is None:
            iid = entry.get("instrumentID")
        if iid is None:
            continue
        # Restrict to stocks when the type is available (new schema).
        type_id = entry.get("instrumentTypeID")
        if type_id is not None and type_id != 5:
            continue
        sym = entry.get("symbolFull") or entry.get("symbol")
        if not sym:
            continue
        clean = str(sym).strip().upper()
        # Old schema has no type field -> approximate the stock filter by
        # dropping known crypto tickers (see _CRYPTO_TICKERS rationale).
        if old_schema and clean in _CRYPTO_TICKERS:
            continue
        try:
            id_to_symbol[int(iid)] = clean
        except (TypeError, ValueError):
            continue

    return id_to_symbol


def _metric_value(investor: dict, field: str, nested: str | None) -> float | None:
    """Extract a numeric metric off an investor dict, tolerating nesting/nulls."""
    if nested:
        parent = investor.get(nested)
        raw = parent.get(field) if isinstance(parent, dict) else None
    else:
        raw = investor.get(field)
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(val):
        return None
    return val


def select_subgroup(
    investors: list[dict],
    metric: str | None,
    *,
    nested: str | None = None,
    pct: float = DECILE,
    highest: bool = True,
    key_fn: Callable[[dict], float | None] | None = None,
) -> list[dict]:
    """Return the ``pct`` slice of ``investors`` by ``metric``.

    ``highest=True`` returns the *top* decile (largest metric), ``highest=False``
    the *bottom* decile (smallest metric). Investors whose metric is missing are
    excluded from ranking. When ``metric`` is ``None`` (the crowd baseline), all
    investors are returned unchanged.

    ``key_fn`` overrides field extraction (used by tests); otherwise the value is
    read via ``_metric_value(investor, metric, nested)``.
    """
    if metric is None and key_fn is None:
        return list(investors)

    extractor = key_fn if key_fn is not None else (lambda inv: _metric_value(inv, metric, nested))

    scored = [(extractor(inv), inv) for inv in investors]
    scored = [(v, inv) for v, inv in scored if v is not None]
    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=highest)
    n = max(1, int(round(len(scored) * pct)))
    return [inv for _v, inv in scored[:n]]


def build_consensus(
    subgroup: list[dict],
    id_to_symbol: dict[int, str],
    top_n: int = TOP_N,
) -> list[tuple[str, float]]:
    """Build a group's consensus book: top ``top_n`` symbols by summed weight.

    LONG positions only (``isBuy is True``). For each investor's each long
    position, add ``investmentPct`` to a per-symbol accumulator (keyed by the
    resolved ``symbolFull``). Positions whose ``instrumentId`` is not in
    ``id_to_symbol`` are dropped. Returns ``[(symbol, summed_weight), ...]``
    sorted by weight descending, truncated to ``top_n``.
    """
    weights: dict[str, float] = {}

    for inv in subgroup:
        portfolio = inv.get("portfolio")
        if not isinstance(portfolio, dict):
            continue
        positions = portfolio.get("positions") or []
        for pos in positions:
            if not isinstance(pos, dict):
                continue
            if pos.get("isBuy") is not True:  # long-only
                continue
            iid = pos.get("instrumentId")
            sym = id_to_symbol.get(iid)
            if sym is None:
                continue
            pct = pos.get("investmentPct")
            try:
                pct = float(pct)
            except (TypeError, ValueError):
                continue
            if math.isnan(pct):
                continue
            weights[sym] = weights.get(sym, 0.0) + pct

    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_n]


def _nearest_trading_close(series: pd.Series, target: date) -> float | None:
    """First close on or after ``target``; None if no such trading day exists."""
    if series is None or series.empty:
        return None
    ts = pd.Timestamp(target)
    # index is assumed sorted ascending (yfinance default)
    idx = series.index.searchsorted(ts, side="left")
    if idx >= len(series):
        return None
    val = series.iloc[idx]
    if pd.isna(val):
        # advance to next non-NaN on/after target
        for j in range(idx + 1, len(series)):
            v = series.iloc[j]
            if not pd.isna(v):
                return float(v)
        return None
    return float(val)


def forward_return(
    price_panel: pd.DataFrame,
    symbols: list[str],
    t_iso: str,
    horizon_days: int,
) -> float | None:
    """Equal-weighted forward return of ``symbols`` over ``[T, T+horizon]``.

    For each symbol resolvable in ``price_panel`` (a DataFrame of daily closes,
    columns = symbols, index = DatetimeIndex), the return is
    ``close[first trading day >= T+h] / close[first trading day >= T] - 1``,
    using the nearest trading day on/after each target date. The function
    returns the *equal-weighted mean* across resolvable symbols, or ``None`` if
    none resolve.
    """
    if price_panel is None or price_panel.empty or not symbols:
        return None

    t0 = datetime.fromisoformat(t_iso).date()
    t1 = t0 + timedelta(days=horizon_days)

    rets: list[float] = []
    for sym in symbols:
        if sym not in price_panel.columns:
            continue
        series = price_panel[sym].dropna()
        p0 = _nearest_trading_close(series, t0)
        p1 = _nearest_trading_close(series, t1)
        if p0 is None or p1 is None or p0 == 0:
            continue
        rets.append(p1 / p0 - 1.0)

    if not rets:
        return None
    return float(np.mean(rets))


def excess_stats(excess_values: list[float]) -> dict[str, float]:
    """Summary stats for a list of per-date excess returns.

    Returns ``{mean, std, t, hit, n}`` where ``std`` is the *sample* std
    (ddof=1), ``t = mean / (std / sqrt(n))`` (one-sample t vs 0), and
    ``hit`` is the fraction of values strictly > 0. NaNs are dropped. With
    n < 2 the t-stat is ``nan`` (undefined variance).
    """
    vals = [float(v) for v in excess_values if v is not None and not math.isnan(float(v))]
    n = len(vals)
    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "t": float("nan"),
            "hit": float("nan"),
            "n": 0,
        }

    arr = np.asarray(vals, dtype=float)
    mean = float(arr.mean())
    hit = float((arr > 0).mean())
    if n < 2:
        return {"mean": mean, "std": float("nan"), "t": float("nan"), "hit": hit, "n": n}

    std = float(arr.std(ddof=1))
    if std == 0:
        t = float("inf") if mean > 0 else (float("-inf") if mean < 0 else 0.0)
    else:
        t = mean / (std / math.sqrt(n))
    return {"mean": mean, "std": std, "t": t, "hit": hit, "n": n}


def benjamini_hochberg(pvals: list[float], alpha: float = 0.10) -> list[bool]:
    """Benjamini-Hochberg FDR at ``alpha``.

    Returns a boolean mask (same order/length as ``pvals``) flagging which
    hypotheses are rejected (significant). Uses the standard step-up rule:
    sort p-values ascending, find the largest rank ``k`` with
    ``p_(k) <= alpha * k / m``, reject all hypotheses with p-value <= p_(k).
    NaN p-values are treated as 1.0 (never significant).
    """
    m = len(pvals)
    if m == 0:
        return []

    clean = [1.0 if (p is None or math.isnan(p)) else float(p) for p in pvals]
    order = sorted(range(m), key=lambda i: clean[i])

    # Largest k (1-based) with p_(k) <= alpha*k/m
    k_max = 0
    for rank, idx in enumerate(order, start=1):
        if clean[idx] <= alpha * rank / m:
            k_max = rank
    if k_max == 0:
        return [False] * m

    threshold = clean[order[k_max - 1]]
    return [clean[i] <= threshold for i in range(m)]


def to_yf_symbol(symbol: str) -> str:
    """Map an eToro ``symbolFull`` to the yfinance ticker convention.

    eToro uses dotted class shares (``BRK.B``) and occasional ``.US`` suffixes;
    yfinance expects ``BRK-B`` and bare US tickers. Non-US exchange suffixes
    (``.L``, ``.DE``, ...) are left intact -- yfinance understands those. The
    transform is idempotent and only touches the two known-divergent forms so
    it never corrupts a symbol that is already valid.
    """
    if not symbol:
        return symbol
    s = str(symbol).strip().upper()
    if s.endswith(".US"):
        s = s[:-3]
    # US class shares: a trailing single-letter class after a dot -> dash
    # (BRK.B -> BRK-B, BF.B -> BF-B). yfinance itself uses single-letter
    # exchange suffixes for a few venues (.L London, .V TSX-V, .F Frankfurt) --
    # those must be preserved, so they are excluded from the rewrite.
    _SINGLE_LETTER_EXCHANGES = {"L", "V", "F"}
    if "." in s:
        base, _, suffix = s.rpartition(".")
        if len(suffix) == 1 and suffix.isalpha() and suffix not in _SINGLE_LETTER_EXCHANGES:
            s = f"{base}-{suffix}"
    return s


def two_sided_p_from_t(t: float, df: int) -> float:
    """Two-sided p-value for a t-statistic with ``df`` degrees of freedom.

    Uses the regularized incomplete beta function so we don't need scipy.
    Returns 1.0 when inputs are degenerate (df < 1 or non-finite t).
    """
    if df < 1 or t is None or math.isnan(t) or math.isinf(t):
        return 0.0 if (t is not None and math.isinf(t)) else 1.0
    x = df / (df + t * t)
    # p = I_x(df/2, 1/2) == two-sided tail
    return _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a,b) via continued fraction (Lentz)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    # continued fraction
    f, c, d = 1.0, 1.0, 0.0
    tiny = 1e-30
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        cd = c * d
        f *= cd
        if abs(1.0 - cd) < 1e-10:
            break
    result = front * (f - 1.0)
    return min(1.0, max(0.0, result))


# --------------------------------------------------------------------------- #
# IO / network (not unit-tested)
# --------------------------------------------------------------------------- #


def find_monthly_snapshots(  # pragma: no cover
    census_dir: Path = CENSUS_DIR,
    start_ym: str = "2025-06",
    end_ym: str = "2026-06",
) -> list[tuple[str, Path]]:
    """First available snapshot per calendar month in ``[start_ym, end_ym]``.

    Returns ``[(iso_date, path), ...]`` sorted by date.
    """
    import re
    from collections import OrderedDict

    files = sorted(census_dir.glob("etoro-data-*.json"))
    seen: OrderedDict[str, tuple[str, Path]] = OrderedDict()
    for f in files:
        m = re.match(r"etoro-data-(\d{4})-(\d{2})-(\d{2})", f.name)
        if not m:
            continue
        y, mo, day = m.groups()
        ym = f"{y}-{mo}"
        if start_ym <= ym <= end_ym and ym not in seen:
            seen[ym] = (f"{y}-{mo}-{day}", f)
    return list(seen.values())


def load_snapshot(path: Path) -> dict:  # pragma: no cover
    with open(path) as fh:
        return json.load(fh)


def fetch_price_panel(  # pragma: no cover
    symbols: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download daily adjusted closes for ``symbols`` into a single panel.

    Returns a DataFrame indexed by date with one column per resolvable symbol.
    Symbols yfinance can't resolve are simply absent from the columns.
    """
    import yfinance as yf

    uniq = sorted(set(symbols))
    print(f"[fetch] downloading {len(uniq)} symbols {start}..{end} via yfinance")
    raw = yf.download(
        uniq,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        threads=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    # Normalise to a flat closes DataFrame (columns = symbols).
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            closes = raw["Close"].copy()
        else:
            return pd.DataFrame()
    else:
        # single symbol -> Series-like frame
        col = uniq[0]
        closes = raw[["Close"]].copy()
        closes.columns = [col]

    closes = closes.sort_index()
    closes.columns = [str(c).upper() for c in closes.columns]
    # Drop columns yfinance returned empty (delisted / unresolvable): these come
    # back as all-NaN columns, which would otherwise be counted as "resolved".
    closes = closes.dropna(axis=1, how="all")
    return closes


def run_study(  # pragma: no cover
    census_dir: Path = CENSUS_DIR,
) -> dict[str, Any]:
    """Execute the full study; returns a results dict for reporting."""
    samples = find_monthly_snapshots(census_dir)
    print(f"[study] {len(samples)} monthly snapshots: {[iso for iso, _ in samples]}")

    # Pass 1: build each group's consensus book per date, collect all symbols.
    # books[date_iso][group] = [(symbol, weight), ...]
    books: dict[str, dict[str, list[tuple[str, float]]]] = {}
    coverage_rows: list[dict[str, Any]] = []
    all_symbols: set[str] = set()

    for iso, path in samples:
        snap = load_snapshot(path)
        investors = snap.get("investors", [])
        id_to_symbol = build_id_symbol_map(snap.get("instruments"))
        books[iso] = {}
        for group, (field, nested, highest) in SUBGROUP_SPECS.items():
            subgroup = select_subgroup(investors, field, nested=nested, pct=DECILE, highest=highest)
            book = build_consensus(subgroup, id_to_symbol, top_n=TOP_N)
            # Map eToro symbols to yfinance tickers for price resolution.
            book = [(to_yf_symbol(sym), w) for sym, w in book]
            books[iso][group] = book
            for sym, _w in book:
                all_symbols.add(sym)
        print(f"[study] {iso}: {len(investors)} investors, {len(id_to_symbol)} instruments mapped")

    all_symbols.add(BENCHMARK)

    # Pass 2: fetch one price panel spanning all dates + max horizon buffer.
    earliest = min(iso for iso, _ in samples)
    latest_t = max(iso for iso, _ in samples)
    end_dt = datetime.fromisoformat(latest_t).date() + timedelta(days=max(HORIZONS) + 10)
    panel = fetch_price_panel(list(all_symbols), start=earliest, end=end_dt.isoformat())
    resolved = set(panel.columns)
    print(f"[study] panel: {len(resolved)}/{len(all_symbols)} symbols resolved")

    # Coverage: how many of each book's symbols resolved.
    total_book_syms = 0
    total_resolved_syms = 0
    for iso, _ in samples:
        for group in SUBGROUP_SPECS:
            syms = [s for s, _w in books[iso][group]]
            got = [s for s in syms if s in resolved]
            total_book_syms += len(syms)
            total_resolved_syms += len(got)
            coverage_rows.append(
                {
                    "date": iso,
                    "group": group,
                    "book_size": len(syms),
                    "resolved": len(got),
                }
            )
    coverage_pct = 100.0 * total_resolved_syms / total_book_syms if total_book_syms else 0.0

    # Pass 3: excess returns per (group, horizon) across dates.
    # excess[group][h] = [excess_date1, excess_date2, ...]
    excess: dict[str, dict[int, list[float]]] = {
        g: {h: [] for h in HORIZONS} for g in SUBGROUP_SPECS
    }
    for iso, _ in samples:
        for h in HORIZONS:
            spy_ret = forward_return(panel, [BENCHMARK], iso, h)
            if spy_ret is None:
                continue
            for group in SUBGROUP_SPECS:
                syms = [s for s, _w in books[iso][group]]
                grp_ret = forward_return(panel, syms, iso, h)
                if grp_ret is None:
                    continue
                excess[group][h].append(grp_ret - spy_ret)

    # Pass 4: stats + multiple-testing corrections over the family.
    stats: dict[str, dict[int, dict[str, float]]] = {}
    family_keys: list[tuple[str, int]] = []
    family_pvals: list[float] = []
    for group in SUBGROUP_SPECS:
        stats[group] = {}
        for h in HORIZONS:
            st = excess_stats(excess[group][h])
            df = max(st["n"] - 1, 0)
            p = two_sided_p_from_t(st["t"], df)
            st["p"] = p
            stats[group][h] = st
            if group in FAMILY_GROUPS:
                family_keys.append((group, h))
                family_pvals.append(p)

    bh_mask = benjamini_hochberg(family_pvals, alpha=0.10)
    bh_sig = {k: bh_mask[i] for i, k in enumerate(family_keys)}

    return {
        "samples": [iso for iso, _ in samples],
        "stats": stats,
        "bh_sig": bh_sig,
        "family_keys": family_keys,
        "coverage_pct": coverage_pct,  # holding-slot weighted
        "unique_coverage_pct": (100.0 * len(resolved) / len(all_symbols) if all_symbols else 0.0),
        "coverage_rows": coverage_rows,
        "resolved_symbols": len(resolved),
        "total_symbols": len(all_symbols),
    }


def _fmt(v: float, nd: int = 4) -> str:  # pragma: no cover
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    return f"{v:.{nd}f}"


def write_report(results: dict[str, Any], out: Path = OUTPUT_REPORT) -> None:  # pragma: no cover
    stats = results["stats"]
    bh_sig = results["bh_sig"]
    samples = results["samples"]

    lines: list[str] = []
    lines.append("# Census Sub-Group Edge Study - Findings\n")
    lines.append(f"_Generated {datetime.now().isoformat(timespec='seconds')}_\n")
    lines.append("## Question\n")
    lines.append(
        "Do any sub-groups of eToro Popular Investors have predictive edge -- "
        "do the stocks a sub-group collectively holds (top-15 consensus book, "
        "long-only, weighted by investmentPct) outperform SPY over the next 7 "
        "and 30 calendar days?\n"
    )

    lines.append("## Method (as run)\n")
    lines.append(f"- Sample dates ({len(samples)} monthly snapshots): {', '.join(samples)}\n")
    lines.append(
        "- Sub-groups: top-decile by gain (`most_profitable`), "
        "dailyGain (`hot_hand`, proxy -- see caveats), "
        "portfolio.positionsCount (`diversified`), copiers "
        "(`most_copied`); bottom-decile by riskScore (`low_risk`); "
        "and ALL investors (`crowd`, baseline).\n"
    )
    lines.append(
        "- Consensus book: per group, sum `investmentPct` per "
        "instrument across all long positions, map to symbol, take "
        "top 15 by weight.\n"
    )
    lines.append(
        "- Forward return: equal-weighted mean over resolvable "
        "holdings of close[T+h]/close[T]-1 (nearest trading day >= "
        "target). Excess = group - SPY.\n"
    )
    lines.append(
        "- Family = 5 real groups x 2 horizons = 10 tests (crowd "
        "excluded from family, shown for reference). BH-FDR at "
        "alpha=0.10; HLZ hurdle |t| >= 3.0.\n"
    )

    lines.append("\n## Results\n")
    lines.append(
        "| Group | Horizon (d) | Mean excess | t-stat | Hit-rate | n | BH-sig? | \\|t\\|>=3? |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for group in SUBGROUP_SPECS:
        for h in HORIZONS:
            st = stats[group][h]
            is_family = (group, h) in bh_sig
            bh = ("yes" if bh_sig.get((group, h)) else "no") if is_family else "-"
            hlz = "yes" if (not math.isnan(st["t"]) and abs(st["t"]) >= HLZ_HURDLE) else "no"
            hitpct = "n/a" if math.isnan(st["hit"]) else f"{100 * st['hit']:.0f}%"
            tag = " (baseline)" if group == "crowd" else ""
            lines.append(
                f"| {group}{tag} | {h} | {_fmt(100 * st['mean'], 2)}% | "
                f"{_fmt(st['t'], 2)} | {hitpct} | {st['n']} | {bh} | {hlz} |"
            )

    lines.append("\n## Symbol-resolution coverage\n")
    lines.append(
        f"- Unique-symbol resolution: **{results['unique_coverage_pct']:.1f}%** "
        f"({results['resolved_symbols']}/{results['total_symbols']} distinct "
        f"symbols across all books + SPY resolved by yfinance). The unresolvable "
        f"remainder is delisted Russian ADRs (`.L`) and a couple of delisted US "
        f"names.\n"
    )
    lines.append(
        f"- Holding-slot-weighted resolution: **{results['coverage_pct']:.1f}%** "
        f"(each date x group top-15 slot counted; higher than the unique figure "
        f"because the unresolvable names are rare, low-weight holdings that "
        f"seldom enter a top-15 book).\n"
    )

    # Verdict
    survivors_bh = [k for k, v in bh_sig.items() if v]
    survivors_hlz = [
        (g, h)
        for g in FAMILY_GROUPS
        for h in HORIZONS
        if not math.isnan(stats[g][h]["t"]) and abs(stats[g][h]["t"]) >= HLZ_HURDLE
    ]
    lines.append("\n## Verdict\n")
    if not survivors_bh and not survivors_hlz:
        lines.append(
            "**No robust edge.** No (group, horizon) in the 10-test family "
            "survives Benjamini-Hochberg FDR at alpha=0.10, and none clears the "
            "Harvey-Liu-Zhu |t| >= 3.0 hurdle. Any apparent out/under-performance "
            "is statistically indistinguishable from noise at this sample size.\n"
        )
    else:
        lines.append(
            f"**Possible edge.** BH-FDR survivors: "
            f"{survivors_bh or 'none'}. HLZ |t|>=3 survivors: "
            f"{survivors_hlz or 'none'}. Treat with caution given the caveats "
            f"below (single regime, low n).\n"
        )

    lines.append("\n## Caveats\n")
    lines.append(
        "- **Single, mostly bull-ish regime** (2025-06 .. 2026-06). "
        "Edge that shows up in a rising market may vanish in a drawdown; "
        "no bear-market observation to test robustness.\n"
    )
    lines.append(
        f"- **Low power.** Only n~{len(samples)} monthly observations per "
        "cell -> wide confidence intervals; a real but modest edge could "
        "be missed (Type II), and BH/HLZ are deliberately conservative.\n"
    )
    lines.append(
        "- **Survivorship.** The census tracks *currently* prominent "
        "Popular Investors; investors who blew up and dropped off are not "
        "in the snapshots, biasing group returns upward.\n"
    )
    lines.append(
        "- **Long-only, equal-weight book.** We ignore shorts, leverage, "
        "and the investors' actual position sizing across their whole book; "
        "the top-15 equal-weight proxy is a coarse read of 'what the group "
        "holds'.\n"
    )
    lines.append(
        "- **`thisWeekGain` unavailable** -> `hot_hand` uses `dailyGain` as "
        "a momentum proxy. A one-day return is a noisier momentum signal "
        "than a one-week return.\n"
    )
    lines.append(
        "- **yfinance resolution gaps.** Non-US tickers, delistings, and "
        "symbol-format mismatches drop out of the panel; unique-symbol coverage "
        f"is ~{results['unique_coverage_pct']:.0f}%, so each book is measured on "
        "the subset it could resolve. Crypto holdings were excluded by design "
        "(stock study); bare crypto tickers also mis-resolve to Grayscale trusts "
        "in yfinance.\n"
    )
    lines.append(
        "- **Overlapping horizons across adjacent monthly dates** induce "
        "mild serial correlation in the excess series, which the plain "
        "one-sample t-stat does not adjust for.\n"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"[report] wrote {out}")


def main() -> None:  # pragma: no cover
    results = run_study()
    write_report(results)

    # Console summary
    stats = results["stats"]
    bh_sig = results["bh_sig"]
    print("\n=== RESULTS (group x horizon) ===")
    print(f"{'group':<16}{'h':>4}{'mean%':>9}{'t':>8}{'hit':>7}{'n':>4}{'BH':>5}{'|t|>=3':>8}")
    for group in SUBGROUP_SPECS:
        for h in HORIZONS:
            st = stats[group][h]
            is_family = (group, h) in bh_sig
            bh = ("Y" if bh_sig.get((group, h)) else "n") if is_family else "-"
            hlz = "Y" if (not math.isnan(st["t"]) and abs(st["t"]) >= HLZ_HURDLE) else "n"
            hit = "n/a" if math.isnan(st["hit"]) else f"{100 * st['hit']:.0f}%"
            print(
                f"{group:<16}{h:>4}{_fmt(100 * st['mean'], 2):>9}"
                f"{_fmt(st['t'], 2):>8}{hit:>7}{st['n']:>4}{bh:>5}{hlz:>8}"
            )
    print(
        f"\nCoverage: unique {results['unique_coverage_pct']:.1f}% "
        f"({results['resolved_symbols']}/{results['total_symbols']} symbols); "
        f"holding-slot-weighted {results['coverage_pct']:.1f}%"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
