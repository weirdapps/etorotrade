"""Trading Model v3 — Forward IC logger (Phase 5E).

Runs the v3 pipeline (same universe assembly as v3_full_report) then APPENDS
one row per eligible name to a persistent log:

    $V3_IC_LOG  or  ~/.weirdapps-trading/v3_ic_log.csv

Log schema (CSV): date, ticker, conviction, sector, close

Idempotent per day: if rows for today's date already exist, skip entirely.

Run (VPS / network allowed):   .venv/bin/python scripts/v3_ic_logger.py

No module-level yahoofinance.core.config import.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from trade_modules.v3.constants import V3_IC_HORIZONS  # noqa: E402
from trade_modules.validation.xsection_ic import cross_sectional_ic  # noqa: E402

PORTFOLIO_CSV = "yahoofinance/output/portfolio.csv"
BUY_CSV = "yahoofinance/output/buy.csv"
ETORO_CSV = "yahoofinance/output/etoro.csv"

DEFAULT_LOG = "~/.weirdapps-trading/v3_ic_log.csv"
# Per-cluster z's are logged alongside the composite conviction so forward IC can be
# measured PER CLUSTER from the live log, not just for the composite. Older logs that
# predate this lack the columns -> they read back as NaN (handled downstream).
_CLUSTER_Z = ("value_z", "quality_z", "momentum_z", "growth_z", "lowvol_z", "strength_z")
LOG_COLUMNS = ["date", "ticker", "conviction", "sector", "close", *_CLUSTER_Z]


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested, no network)
# ---------------------------------------------------------------------------


def append_snapshot(
    log_path: str | Path,
    date: str,
    scored: pd.DataFrame,
) -> int:
    """Append one row per name to the log, idempotent per day.

    Args:
        log_path: Path to the persistent CSV log (created on first run).
        date: UTC date string in ``%Y-%m-%d`` format.
        scored: DataFrame indexed by ticker with at least columns
            ``conviction``, ``sector``, and ``price`` (or ``close``).

    Returns:
        Number of rows appended.  0 when rows for ``date`` already exist
        (idempotent: the log is never modified a second time for the same day).
    """
    log_path = Path(log_path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Load or initialise the existing log.
    if log_path.exists() and log_path.stat().st_size > 0:
        try:
            existing = pd.read_csv(log_path, dtype={"date": str})
        except Exception:  # noqa: BLE001
            existing = pd.DataFrame(columns=LOG_COLUMNS)
    else:
        existing = pd.DataFrame(columns=LOG_COLUMNS)

    # Idempotency guard: skip if any row for this date already exists.
    if len(existing) > 0 and date in existing["date"].values:
        return 0

    # Build new rows from the scored frame.
    rows: list[dict] = []
    for ticker, row in scored.iterrows():
        # Support both 'price' (from enrich_features) and 'close' (direct).
        close_val = row.get("price") if "price" in row.index else row.get("close", np.nan)
        conv_val = row.get("conviction", np.nan)
        sect_val = row.get("sector", "")

        rec = {
            "date": date,
            "ticker": str(ticker),
            "conviction": float(conv_val) if pd.notna(conv_val) else np.nan,
            "sector": str(sect_val) if pd.notna(sect_val) else "",
            "close": float(close_val) if pd.notna(close_val) else np.nan,
        }
        for cz in _CLUSTER_Z:  # per-cluster z for per-cluster forward IC
            v = row.get(cz, np.nan)
            rec[cz] = float(v) if pd.notna(v) else np.nan
        rows.append(rec)

    if not rows:
        return 0

    new_df = pd.DataFrame(rows, columns=LOG_COLUMNS)
    # Skip concat when existing is empty to avoid a pandas FutureWarning about
    # concatenating DataFrames with all-NA (or no) entries on first write.
    combined = new_df if existing.empty else pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(log_path, index=False)
    return len(rows)


def forward_return_panel(
    log_df: pd.DataFrame, horizon: int, signal_col: str = "conviction"
) -> pd.DataFrame:
    """Build (signal_date, ticker, <signal_col>, forward_return) panel.

    ``signal_col`` selects which logged column is the signal (default the composite
    ``conviction``; pass e.g. ``"value_z"`` to measure a single cluster's IC).

    For each unique date at sorted position ``i`` in the log, the forward date
    is the date at position ``i + horizon`` (log-row steps, not calendar days).
    Forward return is defined as::

        fwd = close[date_{i+h}] / close[date_i] - 1

    matched per ticker across the two dates.

    Args:
        log_df: DataFrame with columns date, ticker, conviction, close.
        horizon: Number of log-date steps to look forward.

    Returns:
        Long-format panel DataFrame with columns
        ``[signal_date, ticker, conviction, forward_return]``.
        Empty when there are fewer than ``horizon + 1`` distinct dates.
    """
    empty = pd.DataFrame(columns=["signal_date", "ticker", signal_col, "forward_return"])
    if log_df is None or log_df.empty:
        return empty

    log = log_df.copy()
    log["date"] = log["date"].astype(str)

    dates = sorted(log["date"].unique())
    if len(dates) <= horizon:
        return empty

    # Build a (date, ticker) -> close pivot for fast lookups.
    pivot = log.pivot_table(index="date", columns="ticker", values="close", aggfunc="first")

    # Build (date, ticker) -> signal lookup (conviction, or a cluster-z).
    conv_pivot = log.pivot_table(index="date", columns="ticker", values=signal_col, aggfunc="first")

    records = []
    for i, signal_date in enumerate(dates):
        future_idx = i + horizon
        if future_idx >= len(dates):
            break
        future_date = dates[future_idx]

        for ticker in pivot.columns:
            if signal_date not in pivot.index or future_date not in pivot.index:
                continue
            close_sig = pivot.at[signal_date, ticker] if ticker in pivot.columns else np.nan
            close_fut = pivot.at[future_date, ticker] if ticker in pivot.columns else np.nan
            if not np.isfinite(close_sig) or not np.isfinite(close_fut) or close_sig <= 0:
                continue
            conviction = (
                conv_pivot.at[signal_date, ticker]
                if (signal_date in conv_pivot.index and ticker in conv_pivot.columns)
                else np.nan
            )
            records.append(
                {
                    "signal_date": signal_date,
                    "ticker": ticker,
                    signal_col: conviction,
                    "forward_return": close_fut / close_sig - 1.0,
                }
            )

    if not records:
        return empty
    return pd.DataFrame(records)


def ic_from_log(
    log_df: pd.DataFrame,
    horizons: tuple[int, ...] = V3_IC_HORIZONS,
    signal_col: str = "conviction",
) -> dict:
    """Compute cross-sectional IC for each horizon from the IC log.

    Args:
        log_df: DataFrame with columns date, ticker, conviction, sector, close.
        horizons: Forward-return horizons to evaluate (in log-date steps).

    Returns:
        ``{horizon: result}`` where ``result`` is either:

        - The dict returned by :func:`cross_sectional_ic` (keys: mean_ic,
          ic_std, t_stat, hit_rate, n_dates, ic_by_date), or
        - ``{"insufficient_history": True, "horizon": h,
             "n_dates_available": N, "n_dates_needed": h+1}``
          when fewer than ``h + 1`` distinct dates are in the log.
    """
    results: dict = {}

    if log_df is None or log_df.empty:
        for h in horizons:
            results[h] = {
                "insufficient_history": True,
                "horizon": h,
                "n_dates_available": 0,
                "n_dates_needed": h + 1,
            }
        return results

    dates = sorted(log_df["date"].astype(str).unique())
    n_dates = len(dates)

    for h in horizons:
        if n_dates <= h:
            results[h] = {
                "insufficient_history": True,
                "horizon": h,
                "n_dates_available": n_dates,
                "n_dates_needed": h + 1,
            }
            continue

        panel = forward_return_panel(log_df, h, signal_col=signal_col)
        if panel.empty or len(panel) == 0:
            results[h] = {
                "insufficient_history": True,
                "horizon": h,
                "n_dates_available": n_dates,
                "n_dates_needed": h + 1,
            }
            continue

        ic = cross_sectional_ic(
            panel,
            signal_col=signal_col,
            forward_col="forward_return",
            date_col="signal_date",
        )
        results[h] = ic

    return results


# ---------------------------------------------------------------------------
# Live pipeline (network required — runs on VPS)
# ---------------------------------------------------------------------------


def _read_tickers(path: str) -> list[str]:
    try:
        df = pd.read_csv(path, na_values=["--"])
        return df["TKR"].dropna().astype(str).tolist()
    except Exception as exc:  # noqa: BLE001
        print(f"warn: could not read {path}: {exc}", file=sys.stderr)
        return []


def main() -> None:
    # Lazy imports: avoid module-level yahoofinance.core.config import.
    from trade_modules.v3.combine import compute_scores  # noqa: PLC0415
    from trade_modules.v3.constants import MIN_FACTOR_COVERAGE  # noqa: PLC0415
    from trade_modules.v3.enrichment import select_enrichment_set  # noqa: PLC0415
    from trade_modules.v3.features import enrich_features  # noqa: PLC0415
    from trade_modules.v3.sectors import (  # noqa: PLC0415
        load_offline_sector_map,
        update_sector_cache,
    )
    from trade_modules.v3.universe import load_universe  # noqa: PLC0415

    # --- Universe: two-stage scoring (pre-rank full coverage universe, enrich top slice) ---
    port = _read_tickers(PORTFOLIO_CSV)
    buy = _read_tickers(BUY_CSV)
    port_set = set(port)
    sector_map = load_offline_sector_map()
    cap = int(os.environ.get("V3_ENRICH_CAP", "500"))
    pool = load_universe(ETORO_CSV, min_factor_coverage=MIN_FACTOR_COVERAGE)
    universe = select_enrichment_set(ETORO_CSV, port, buy, cap=cap, sector_map=sector_map)
    print(
        f"universe: pre-ranked {len(pool)} coverage names -> enriching {len(universe)} "
        f"({len(port_set)} held + {len(set(buy) - port_set)} analyst candidates + top coverage)"
    )

    # --- Feature enrichment + scoring ---
    feats = enrich_features(
        universe,
        ETORO_CSV,
        price_period="2y",
        accruals_fetch=lambda _tickers: {},
        sector_map=sector_map,
    )
    update_sector_cache(feats["sector"].dropna().to_dict())  # grow the cache from live sectors
    scores = compute_scores(feats, sector_neutral=True)

    # --- Filter to eligible names only ---
    elig = scores.get("eligible", pd.Series(True, index=scores.index)).fillna(False).astype(bool)
    eligible_scores = scores[elig]
    print(f"eligible: {len(eligible_scores)} / {len(scores)} names")

    # --- Log path (env override or default) ---
    log_path = os.environ.get("V3_IC_LOG", DEFAULT_LOG)

    # --- Today's UTC date ---
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # --- Append snapshot (idempotent) ---
    n_appended = append_snapshot(log_path, today, eligible_scores)
    expanded = os.path.expanduser(log_path)
    if n_appended == 0:
        print(f"skipped: rows for {today} already in log  ({expanded})")
    else:
        print(f"appended: {n_appended} rows for {today}  ->  {expanded}")


if __name__ == "__main__":
    main()
