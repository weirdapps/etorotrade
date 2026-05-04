"""
Census Time-Series Analysis

CIO Review v4 Finding F12: Compute holder-count trajectories over time
from the eToro census archive to detect accumulation and distribution
patterns among popular investors.

Each daily census snapshot records how many of the top-N popular investors
hold each instrument.  By comparing snapshots across days we can identify
which tickers are being accumulated (holder count rising) or distributed
(holder count falling).
"""

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CENSUS_ARCHIVE_DIR = Path.home() / "SourceCode" / "etoro_census" / "archive" / "data"

# Filename pattern: etoro-data-YYYY-MM-DD-HH-MM.json
_FILENAME_RE = re.compile(r"^etoro-data-(\d{4}-\d{2}-\d{2})-(\d{2}-\d{2})\.json$")

# Classification thresholds (percentage-point or count-based)
_STRONG_PP_THRESHOLD = 10.0  # holder-pct delta in pp
_STRONG_COUNT_PCT = 20.0  # holder-count change in %
_MODERATE_PP_THRESHOLD = 3.0
_MODERATE_COUNT_PCT = 5.0

# CIO v36 / M14: z-score-based classification thresholds (sigma multiples)
# Replaces absolute pp thresholds — 3pp matters when ticker is steady (z≈6)
# but is noise when ticker is volatile (z≈0.6). Z-score adapts to each
# ticker's normal churn.
_ZSCORE_STRONG = 3.0  # |z| ≥ 3 → strong move (1-in-370 chance)
_ZSCORE_MODERATE = 2.0  # |z| ≥ 2 → moderate move (1-in-22 chance)
# Default fallback volatility when history is insufficient (mimics legacy)
_DEFAULT_VOL_PP = 3.0


def _classify_trend_zscore(
    delta_pp: float,
    holder_volatility_pp: float,
) -> str:
    """Classify holder trend by z-score = delta_pp / σ_pp.

    More robust than absolute pp threshold: a 3pp move is huge for a
    steady ticker but noise for a volatile one. Z-score normalizes by
    each ticker's own historical volatility.

    When holder_volatility_pp is 0 (insufficient history), falls back to
    legacy absolute thresholds.
    """
    if not holder_volatility_pp or holder_volatility_pp <= 0:
        # Fallback to legacy classifier when no history
        return _classify_trend(delta_pp, 0.0)
    z = delta_pp / holder_volatility_pp
    if z >= _ZSCORE_STRONG:
        return "strong_accumulation"
    if z >= _ZSCORE_MODERATE:
        return "accumulation"
    if z <= -_ZSCORE_STRONG:
        return "strong_distribution"
    if z <= -_ZSCORE_MODERATE:
        return "distribution"
    return "stable"


def _holder_volatility(
    ticker: str,
    snapshots: list[dict[str, Any]],
    lookback_days: int = 90,
) -> float:
    """Return σ of holder-pct daily changes for a ticker over lookback_days.

    Computed as standard deviation of day-to-day pp changes in the ticker's
    holder share. Returns 0.0 when the ticker has fewer than 5 snapshots.
    """
    points: list[float] = []
    for snap in snapshots[-lookback_days:]:
        ic = snap.get("investor_count", 0)
        if not ic:
            continue
        held = snap.get("holdings", {}).get(ticker)
        if held is None:
            continue
        points.append(held / ic * 100)
    if len(points) < 5:
        return 0.0
    # std of day-to-day pp changes
    diffs = [points[i] - points[i - 1] for i in range(1, len(points))]
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    import math as _math

    return _math.sqrt(var) if var > 0 else 0.0


def _ema_holder_pct(
    ticker: str,
    snapshots: list[dict[str, Any]],
    span: int = 7,
) -> float:
    """EMA over recent snapshots' holder-pct for ticker. Returns 0.0 if no data."""
    pcts: list[float] = []
    for snap in snapshots[-span:]:
        ic = snap.get("investor_count", 0)
        if not ic:
            continue
        held = snap.get("holdings", {}).get(ticker)
        if held is None:
            continue
        pcts.append(held / ic * 100)
    if not pcts:
        return 0.0
    if len(pcts) == 1:
        return pcts[0]
    # Standard EMA: alpha = 2/(span+1)
    alpha = 2.0 / (span + 1)
    ema = pcts[0]
    for p in pcts[1:]:
        ema = alpha * p + (1 - alpha) * ema
    return ema


def _parse_filename(name: str) -> tuple[str, str] | None:
    """Extract (date, time) from a census filename, or None."""
    m = _FILENAME_RE.match(name)
    if m:
        return m.group(1), m.group(2)
    return None


def _select_latest_per_day(
    files: list[Path],
) -> list[tuple[str, Path]]:
    """
    Given census file paths, keep only the latest file for each calendar day.

    Returns a list of (date_str, path) sorted ascending by date.
    """
    by_date: dict[str, tuple[str, Path]] = {}
    for fpath in files:
        parsed = _parse_filename(fpath.name)
        if parsed is None:
            continue
        date_str, time_str = parsed
        existing = by_date.get(date_str)
        if existing is None or time_str > existing[0]:
            by_date[date_str] = (time_str, fpath)

    return sorted(
        [(date_str, entry[1]) for date_str, entry in by_date.items()],
        key=lambda x: x[0],
    )


def _extract_snapshot(
    fpath: Path,
    date_str: str,
    investor_tier: int,
) -> dict[str, Any] | None:
    """
    Load a single census JSON and extract holdings for the requested
    investor tier.

    Returns a snapshot dict or None on failure.
    """
    try:
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Skipping corrupt/unreadable census file %s: %s", fpath.name, exc)
        return None

    # Build instrumentId -> ticker mapping
    instrument_map: dict[int, str] = {}
    for detail in data.get("instruments", {}).get("details", []):
        iid = detail.get("instrumentId")
        symbol = detail.get("symbolFull")
        if iid is not None and symbol:
            instrument_map[iid] = symbol

    # Find the analysis matching the requested investor tier
    analysis = None
    for a in data.get("analyses", []):
        if a.get("investorCount") == investor_tier:
            analysis = a
            break

    if analysis is None:
        logger.warning("No analysis with investorCount=%d in %s", investor_tier, fpath.name)
        return None

    # Extract holdings — prefer the 'symbol' field already on each holding,
    # fall back to the instrument map.
    holdings: dict[str, int] = {}
    for h in analysis.get("topHoldings", []):
        ticker = h.get("symbol") or instrument_map.get(h.get("instrumentId"))
        count = h.get("holdersCount")
        if ticker and count is not None:
            holdings[ticker] = int(count)

    fear_greed = analysis.get("fearGreedIndex")
    investor_count = analysis.get("investorCount", investor_tier)

    return {
        "date": date_str,
        "holdings": holdings,
        "fear_greed": fear_greed,
        "investor_count": investor_count,
    }


def load_census_snapshots(
    archive_dir: Path | None = None,
    days_back: int = 30,
    investor_tier: int = 100,
) -> list[dict[str, Any]]:
    """
    Load census snapshots from the archive directory.

    Scans for ``etoro-data-YYYY-MM-DD-*.json`` files, keeps the latest file
    per calendar day, and extracts holdings for the specified investor tier.

    Args:
        archive_dir: Directory containing census JSON files.
            Defaults to ``CENSUS_ARCHIVE_DIR``.
        days_back: How many days of history to load.
        investor_tier: Which investor-count tier to extract (100, 500, 1000, 1500).

    Returns:
        List of snapshot dicts sorted ascending by date.  Each dict has keys
        ``date``, ``holdings``, ``fear_greed``, ``investor_count``.
    """
    directory = archive_dir or CENSUS_ARCHIVE_DIR

    if not directory.is_dir():
        logger.warning("Census archive directory does not exist: %s", directory)
        return []

    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    # Collect candidate files
    candidate_files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix == ".json" and p.name.startswith("etoro-data-")
    ]

    if not candidate_files:
        logger.warning("No census files found in %s", directory)
        return []

    daily = _select_latest_per_day(candidate_files)

    # Filter to requested window
    daily = [(d, p) for d, p in daily if d >= cutoff]

    snapshots: list[dict[str, Any]] = []
    for date_str, fpath in daily:
        snap = _extract_snapshot(fpath, date_str, investor_tier)
        if snap is not None:
            snapshots.append(snap)

    logger.info(
        "Loaded %d census snapshots (%s to %s)",
        len(snapshots),
        snapshots[0]["date"] if snapshots else "N/A",
        snapshots[-1]["date"] if snapshots else "N/A",
    )
    return snapshots


def _classify_trend(
    delta_pp: float,
    pct_change: float,
) -> str:
    """
    Classify a holder trend into one of five categories based on both
    percentage-point delta and count percentage change.
    """
    if delta_pp > _STRONG_PP_THRESHOLD or pct_change > _STRONG_COUNT_PCT:
        return "strong_accumulation"
    if delta_pp > _MODERATE_PP_THRESHOLD or pct_change > _MODERATE_COUNT_PCT:
        return "accumulation"
    if delta_pp < -_STRONG_PP_THRESHOLD or pct_change < -_STRONG_COUNT_PCT:
        return "strong_distribution"
    if delta_pp < -_MODERATE_PP_THRESHOLD or pct_change < -_MODERATE_COUNT_PCT:
        return "distribution"
    return "stable"


def compute_holder_trends(
    snapshots: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Compute holder-count trends for every ticker across snapshots.

    For each ticker appearing in at least two snapshots the function
    calculates absolute and percentage changes over the full window and
    the last 7 days, then classifies the trend.

    Args:
        snapshots: List of snapshot dicts (sorted ascending by date).

    Returns:
        Dict mapping ticker to trend data.
    """
    if len(snapshots) < 2:
        logger.info("Fewer than 2 snapshots — cannot compute trends")
        return {}

    earliest = snapshots[0]
    latest = snapshots[-1]
    latest_investor_count = latest["investor_count"]

    # Find the snapshot closest to 7 days ago
    latest_date = datetime.strptime(latest["date"], "%Y-%m-%d")
    seven_days_ago = latest_date - timedelta(days=7)
    snap_7d: dict[str, Any] | None = None
    for snap in reversed(snapshots):
        snap_date = datetime.strptime(snap["date"], "%Y-%m-%d")
        if snap_date <= seven_days_ago:
            snap_7d = snap
            break

    # Collect all tickers that appear in at least 2 snapshots
    ticker_counts: dict[str, int] = {}
    for snap in snapshots:
        for ticker in snap["holdings"]:
            ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

    trends: dict[str, dict[str, Any]] = {}
    for ticker, count in ticker_counts.items():
        if count < 2:
            continue

        current = latest["holdings"].get(ticker)
        first = earliest["holdings"].get(ticker)

        if current is None:
            # Ticker not in latest snapshot — skip
            continue

        holder_pct = current / latest_investor_count * 100 if latest_investor_count else 0.0

        # 30-day (full window) delta
        if first is not None and first > 0:
            delta_30d = current - first
            pct_change_30d = (current - first) / first * 100
        elif first is not None:
            delta_30d = current - first
            pct_change_30d = 0.0
        else:
            delta_30d = 0
            pct_change_30d = 0.0

        # 7-day delta
        delta_7d: int | None = None
        if snap_7d is not None:
            past_7d = snap_7d["holdings"].get(ticker)
            if past_7d is not None:
                delta_7d = current - past_7d

        # Percentage-point delta for classification
        earliest_investor_count = earliest["investor_count"]
        if first is not None and earliest_investor_count:
            earliest_pct = first / earliest_investor_count * 100
        else:
            earliest_pct = 0.0
        delta_pp = holder_pct - earliest_pct

        classification = _classify_trend(delta_pp, pct_change_30d)

        trends[ticker] = {
            "current_holders": current,
            "holder_pct": round(holder_pct, 1),
            "delta_30d": delta_30d,
            "delta_7d": delta_7d,
            "pct_change_30d": round(pct_change_30d, 1),
            "classification": classification,
        }

    return trends


def compute_fear_greed_trend(
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Track Fear and Greed index movement across snapshots.

    Args:
        snapshots: List of snapshot dicts (sorted ascending by date).

    Returns:
        Dict with current value, historical comparisons, and trend direction.
    """
    if not snapshots:
        return {
            "current": None,
            "7d_ago": None,
            "30d_ago": None,
            "trend": "unknown",
            "delta_7d": None,
            "delta_30d": None,
        }

    latest = snapshots[-1]
    current_fg = latest.get("fear_greed")

    latest_date = datetime.strptime(latest["date"], "%Y-%m-%d")
    seven_days_ago = latest_date - timedelta(days=7)
    thirty_days_ago = latest_date - timedelta(days=30)

    fg_7d: Any | None = None
    fg_30d: Any | None = None

    for snap in reversed(snapshots):
        snap_date = datetime.strptime(snap["date"], "%Y-%m-%d")
        fg = snap.get("fear_greed")
        if fg is None:
            continue
        if fg_7d is None and snap_date <= seven_days_ago:
            fg_7d = fg
        if fg_30d is None and snap_date <= thirty_days_ago:
            fg_30d = fg
        if fg_7d is not None and fg_30d is not None:
            break

    delta_7d = (current_fg - fg_7d) if (current_fg is not None and fg_7d is not None) else None
    delta_30d = (current_fg - fg_30d) if (current_fg is not None and fg_30d is not None) else None

    # Determine trend direction from the best available delta
    trend = "unknown"
    ref_delta = delta_7d if delta_7d is not None else delta_30d
    if ref_delta is not None:
        if ref_delta > 2:
            trend = "rising"
        elif ref_delta < -2:
            trend = "falling"
        else:
            trend = "flat"

    return {
        "current": current_fg,
        "7d_ago": fg_7d,
        "30d_ago": fg_30d,
        "trend": trend,
        "delta_7d": delta_7d,
        "delta_30d": delta_30d,
    }


def get_census_context(
    archive_dir: Path | None = None,
    days_back: int = 30,
    investor_tier: int = 100,
) -> dict[str, Any]:
    """
    High-level function combining snapshot loading, trend computation,
    and summary generation for committee consumption.

    Args:
        archive_dir: Census archive directory.
        days_back: Number of days to look back.
        investor_tier: Investor-count tier to analyse.

    Returns:
        Dict with accumulation/distribution rankings, fear-greed trend,
        full ticker trends, and a human-readable summary.
    """
    snapshots = load_census_snapshots(
        archive_dir=archive_dir,
        days_back=days_back,
        investor_tier=investor_tier,
    )

    if not snapshots:
        return {
            "data_available": False,
            "snapshots_loaded": 0,
            "date_range": {"start": None, "end": None},
            "fear_greed": compute_fear_greed_trend([]),
            "top_accumulating": [],
            "top_distributing": [],
            "ticker_trends": {},
            "summary": "Census: No snapshots available.",
        }

    trends = compute_holder_trends(snapshots)
    fear_greed = compute_fear_greed_trend(snapshots)

    # Rank by pct_change_30d
    sorted_tickers = sorted(
        trends.items(),
        key=lambda x: x[1]["pct_change_30d"],
        reverse=True,
    )

    top_accumulating = [
        {"ticker": t, **d} for t, d in sorted_tickers[:10] if d["pct_change_30d"] > 0
    ]
    top_distributing = [{"ticker": t, **d} for t, d in sorted_tickers if d["pct_change_30d"] < 0]
    # Take the 10 most negative
    top_distributing = sorted(
        top_distributing,
        key=lambda x: x["pct_change_30d"],
    )[:10]

    date_range = {
        "start": snapshots[0]["date"],
        "end": snapshots[-1]["date"],
    }
    n_days = (
        datetime.strptime(date_range["end"], "%Y-%m-%d")
        - datetime.strptime(date_range["start"], "%Y-%m-%d")
    ).days

    # Build human-readable summary
    fg_part = ""
    if fear_greed["current"] is not None:
        fg_part = f"F&G: {fear_greed['current']} ({fear_greed['trend']}"
        if fear_greed["delta_7d"] is not None:
            sign = "+" if fear_greed["delta_7d"] >= 0 else ""
            fg_part += f" {sign}{fear_greed['delta_7d']} in 7d"
        fg_part += ")."

    acc_part = ""
    if top_accumulating:
        names = ", ".join(
            f"{a['ticker']} ({'+' if a['pct_change_30d'] >= 0 else ''}{a['pct_change_30d']}%)"
            for a in top_accumulating[:3]
        )
        acc_part = f" Top accumulation: {names}."

    dist_part = ""
    if top_distributing:
        names = ", ".join(f"{d['ticker']} ({d['pct_change_30d']}%)" for d in top_distributing[:3])
        dist_part = f" Top distribution: {names}."

    summary = (
        f"Census: {len(snapshots)} snapshots over {n_days} days. " f"{fg_part}{acc_part}{dist_part}"
    )

    return {
        "data_available": True,
        "snapshots_loaded": len(snapshots),
        "date_range": date_range,
        "fear_greed": fear_greed,
        "top_accumulating": top_accumulating,
        "top_distributing": top_distributing,
        "ticker_trends": trends,
        "summary": summary.strip(),
    }
