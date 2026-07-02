"""Adapters for the S5 pipeline — PURE functions, no I/O.

Maps raw etoro.csv rows / portfolio DataFrames to the internal dicts
expected by S1 (filter_universe), S3 (cio.synthesize), and S4 (size_book).
"""

from __future__ import annotations

import pandas as pd

from .sectors import resolve_sector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_pct(value) -> float | None:
    """Parse a %-string like '56.1%' or '-5%' to float. Missing/-- → None."""
    if value is None:
        return None
    s = str(value).strip()
    if s in ("", "--", "nan", "NaN", "None"):
        return None
    s = s.rstrip("%").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _parse_float(value) -> float | None:
    """Parse a plain numeric string to float. Missing/-- → None."""
    if value is None:
        return None
    s = str(value).strip()
    if s in ("", "--", "nan", "NaN", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _safe_get(row: dict, key: str):
    """Get from dict; return None on missing or blank sentinel."""
    val = row.get(key)
    if val is None:
        return None
    s = str(val).strip()
    if s in ("", "--", "nan", "NaN", "None"):
        return None
    return val


# ---------------------------------------------------------------------------
# Public adapters
# ---------------------------------------------------------------------------


def etoro_row_to_candidate(row: dict, sector_map: dict | None = None) -> dict:
    """Map a real etoro.csv row dict → S3 candidate dict.

    etoro.csv columns:
        TKR, NAME, CAP, PRC, TGT, UP%, #T, %B, #A, AM, A, EXR, B, 52W,
        2H, PET, PEF, P/S, PEG, DV, SI, EG, PP, ROE, DE, FCF, ERN, SZ,
        BS, SIGNAL_TRACK, SIGNAL_HORIZON

    Parses "%" strings to float; missing/-- → None.
    composite_pct is NOT set here — the orchestrator attaches it after S2.
    Does NOT crash on any missing key.

    Args:
        row:        Dict of raw etoro.csv columns for one instrument.
        sector_map: Optional {symbol_upper: sector} dict from
                    load_sector_map().  When provided, ``sector`` is
                    resolved via exact-match (None if not covered —
                    international tickers and any absent symbol).
                    When None, ``sector`` is left as None (back-compat).
    """
    ticker = _safe_get(row, "TKR")
    sector: str | None = None
    if sector_map is not None and ticker:
        sector = resolve_sector(str(ticker), sector_map)
    return {
        "ticker": ticker,
        "name": _safe_get(row, "NAME"),
        "cap": _safe_get(row, "CAP"),
        # %-strings
        "UP%": _parse_pct(_safe_get(row, "UP%")),
        "%B": _parse_pct(_safe_get(row, "%B")),
        "FCF": _parse_pct(_safe_get(row, "FCF")),
        "EXR": _parse_pct(_safe_get(row, "EXR")),
        # Plain numerics
        "analysts": _parse_float(_safe_get(row, "#A")),
        "B": _parse_float(_safe_get(row, "B")),
        "52W": _parse_float(_safe_get(row, "52W")),
        "PET": _parse_float(_safe_get(row, "PET")),
        "PEF": _parse_float(_safe_get(row, "PEF")),
        "PEG": _parse_float(_safe_get(row, "PEG")),
        "DE": _parse_float(_safe_get(row, "DE")),
        "EG": _parse_float(_safe_get(row, "EG")),
        "ROE": _parse_float(_safe_get(row, "ROE")),
        # Sector: resolved from offline CSV map when provided; None otherwise.
        # size_book treats None sector as its own bucket (sector cap skipped).
        "sector": sector,
        # composite_pct: attached by orchestrator after S2; default None
        "composite_pct": None,
    }


def portfolio_to_weights(
    portfolio_df: pd.DataFrame,
) -> tuple[dict[str, float], set[str]]:
    """Extract current weights + held tickers from the portfolio DataFrame.

    Long-only: only rows where isBuy is True (or truthy) are included.
    totalInvestmentPct is percentage (e.g. 8.5 = 8.5%) → convert to fraction.

    Returns:
        (current_weights, held_tickers)
            current_weights: {ticker: fraction}  e.g. {"AAPL": 0.085}
            held_tickers: set of ticker strings (long only)
    """
    if portfolio_df.empty:
        return {}, set()

    weights: dict[str, float] = {}
    held: set[str] = set()

    for _, row in portfolio_df.iterrows():
        is_buy = row.get("isBuy", True)
        # Handle string "True"/"False" from CSV parsing
        if isinstance(is_buy, str):
            is_buy = is_buy.strip().lower() not in ("false", "0", "no")
        else:
            is_buy = bool(is_buy)

        if not is_buy:
            continue  # skip shorts

        ticker = str(row.get("symbol", "")).strip()
        if not ticker:
            continue

        pct_raw = row.get("totalInvestmentPct", 0.0)
        try:
            pct = float(pct_raw)
        except (TypeError, ValueError):
            pct = 0.0

        fraction = pct / 100.0
        weights[ticker] = fraction
        held.add(ticker)

    return weights, held


def universe_df_for_filter(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare an etoro.csv DataFrame for filter_universe.

    filter_universe expects the raw etoro.csv column names directly
    (TKR, NAME, CAP, PRC, #A, B, 52W, PET, PEF, FCF, EXR, UP%, %B, EG,
    PEG, ROE, DE, SIGNAL_TRACK, SIGNAL_HORIZON …).

    The etoro.csv already has these names, so this is mostly a passthrough
    with a defensive copy. If any expected column is missing, filter_universe
    handles it gracefully via its own _get/_is_blank helpers.

    Returns:
        A DataFrame (copy) suitable for filter_universe.
    """
    return raw_df.copy()
