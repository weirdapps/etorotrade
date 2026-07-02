"""Universe routing and quality-filter module (S1).

Pure functions — no I/O, no network.  All decisions are explicit and logged in
the returned 'reasons' dict so every pass/fail can be audited downstream.
"""

from __future__ import annotations

import pandas as pd

from trade_modules.analysis.tiers import _parse_market_cap

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tickers that are price-only by definition (leveraged/inverse ETPs, plain ETFs
# with no fundamental data, commodity vehicles, etc.)
KNOWN_PRICE_ONLY: frozenset[str] = frozenset(
    {
        "UVXY",
        "GLD",
        "LYXGRE.DE",
        "SH",
        "SDS",
        "QID",
        "TQQQ",
        "SQQQ",
        "SPXU",
        "SPXS",
        "XIV",
        "SVXY",
        "VXX",
        "VIXY",
        "AGG",
        "BND",
        "TLT",
        "IEF",
        "SLV",
        "IAU",
        "USO",
        "UNG",
    }
)

# Name fragments that indicate leveraged/inverse ETPs (case-insensitive)
_LEVERAGED_INVERSE_FRAGMENTS: tuple[str, ...] = (
    "ULTRA",
    "INVERSE",
    "2X",
    "3X",
    "LEVERAGED",
)

DEFAULT_CONFIG: dict = {
    "min_cap": 2e9,
    "min_analysts": 5,
    "require_positive_earnings": True,
    "min_52w": 30,
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_blank(value) -> bool:
    """Return True when value is missing, '--', empty string, or NaN."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).strip() in ("", "--", "nan", "NaN", "None")


def _get(row, key, default=None):
    """Safely retrieve a value from a dict-like or Series row."""
    try:
        val = row[key] if hasattr(row, "__getitem__") else getattr(row, key, default)
        return default if _is_blank(val) else val
    except (KeyError, AttributeError):
        return default


def _parse_fcf(fcf_str) -> float | None:
    """Parse FCF% string like '4.0%' to float.  Returns None when unparseable."""
    if _is_blank(fcf_str):
        return None
    s = str(fcf_str).strip().rstrip("%")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_pe(pe_val) -> float | None:
    """Parse a P/E value; returns None when missing/blank/non-numeric."""
    if _is_blank(pe_val):
        return None
    try:
        v = float(str(pe_val).strip())
        return None if pd.isna(v) else v
    except (ValueError, TypeError):
        return None


def _parse_52w(val) -> float | None:
    """Parse 52-week range position (0-100); returns None when missing."""
    if _is_blank(val):
        return None
    try:
        v = float(str(val).strip())
        return None if pd.isna(v) else v
    except (ValueError, TypeError):
        return None


def _has_any_fundamentals(row) -> bool:
    """Return True when the row has at least one non-blank fundamental field."""
    pet = _get(row, "PET")
    pef = _get(row, "PEF")
    fcf = _get(row, "FCF")
    n_analysts = _get(row, "#A")
    # Any of PET, PEF, FCF, #A present counts as "has fundamentals"
    return any(v is not None for v in [pet, pef, fcf, n_analysts])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def route_asset(row) -> str:
    """Classify a single instrument row.

    Returns
    -------
    str
        'fundamental' | 'price_only' | 'excluded'
    """
    # ------------------------------------------------------------------ #
    # 1. Excluded — no usable price
    # ------------------------------------------------------------------ #
    prc = _get(row, "PRC")
    if prc is None:
        return "excluded"
    try:
        prc_f = float(str(prc).strip())
        if pd.isna(prc_f) or prc_f <= 0:
            return "excluded"
    except (ValueError, TypeError):
        return "excluded"

    # ------------------------------------------------------------------ #
    # 2. Price-only — crypto
    # ------------------------------------------------------------------ #
    tkr = str(_get(row, "TKR", "")).strip()
    if tkr.endswith("-USD"):
        return "price_only"

    # ------------------------------------------------------------------ #
    # 3. Price-only — leveraged/inverse ETP by name
    # ------------------------------------------------------------------ #
    name = str(_get(row, "NAME", "")).strip().upper()
    if any(frag in name for frag in _LEVERAGED_INVERSE_FRAGMENTS):
        return "price_only"

    # ------------------------------------------------------------------ #
    # 4. Price-only — known ticker list
    # ------------------------------------------------------------------ #
    if tkr.upper() in KNOWN_PRICE_ONLY:
        return "price_only"

    # ------------------------------------------------------------------ #
    # 5. Price-only — plain ETF / index (no fundamentals at all)
    # ------------------------------------------------------------------ #
    if not _has_any_fundamentals(row):
        return "price_only"

    # ------------------------------------------------------------------ #
    # 6. Fundamental — single-name equity
    # ------------------------------------------------------------------ #
    return "fundamental"


def filter_universe(df: pd.DataFrame, config: dict | None = None) -> dict:
    """Route every row and apply quality gates to fundamental names.

    Parameters
    ----------
    df : pd.DataFrame
        etoro.csv-shaped data.  Not mutated.
    config : dict, optional
        Override any key in DEFAULT_CONFIG.

    Returns
    -------
    dict with keys:
        eligible   : DataFrame — fundamental rows passing ALL gates
        price_only : list of tickers
        excluded   : {ticker: reason}
        reasons    : {ticker: [reason_strings]}  (fundamental pass/fail log)
        summary    : {counts and per-gate stats}
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    working = df.copy()

    eligible_rows: list[int] = []
    price_only: list[str] = []
    excluded: dict[str, str] = {}
    reasons: dict[str, list[str]] = {}

    # per-gate failure counters
    gate_counts: dict[str, int] = {
        "min_cap": 0,
        "min_analysts": 0,
        "positive_earnings": 0,
        "trend": 0,
    }

    for idx, row in working.iterrows():
        tkr = str(row.get("TKR", idx) if hasattr(row, "get") else idx).strip()

        route = route_asset(row)

        if route == "excluded":
            excluded[tkr] = "no usable price"
            continue

        if route == "price_only":
            price_only.append(tkr)
            continue

        # ---- fundamental: apply quality gates ----
        row_reasons: list[str] = []
        passed = True

        # Gate 1 — min_cap
        cap_raw = _get(row, "CAP")
        cap_val = _parse_market_cap(cap_raw) if cap_raw is not None else 0.0
        if cap_val < cfg["min_cap"]:
            row_reasons.append(f"cap ${cap_val / 1e9:.2f}B < ${cfg['min_cap'] / 1e9:.0f}B")
            gate_counts["min_cap"] += 1
            passed = False

        # Gate 2 — min_analysts (fail-closed: blank/unparseable → FAIL)
        n_a_raw = _get(row, "#A")
        if n_a_raw is None:
            row_reasons.append(f"analysts unknown (missing) < {cfg['min_analysts']}")
            gate_counts["min_analysts"] += 1
            passed = False
        else:
            try:
                n_a = int(float(str(n_a_raw).strip()))
                if n_a < cfg["min_analysts"]:
                    row_reasons.append(f"analysts {n_a} < {cfg['min_analysts']}")
                    gate_counts["min_analysts"] += 1
                    passed = False
            except (ValueError, TypeError):
                row_reasons.append(
                    f"analysts unknown (unparseable: {n_a_raw!r}) < {cfg['min_analysts']}"
                )
                gate_counts["min_analysts"] += 1
                passed = False

        # Gate 3 — positive_earnings (fail-closed: all three missing → FAIL)
        if cfg.get("require_positive_earnings", True):
            pet = _parse_pe(_get(row, "PET"))
            pef = _parse_pe(_get(row, "PEF"))
            fcf = _parse_fcf(_get(row, "FCF"))

            all_missing = pet is None and pef is None and fcf is None
            if all_missing:
                row_reasons.append("negative earnings (all earnings fields missing)")
                gate_counts["positive_earnings"] += 1
                passed = False
            else:
                pet_ok = pet is not None and pet > 0
                pef_ok = pef is not None and pef > 0
                fcf_ok = fcf is not None and fcf > 0
                if not (pet_ok or pef_ok or fcf_ok):
                    parts = []
                    if pet is not None:
                        parts.append(f"PET={pet}")
                    if pef is not None:
                        parts.append(f"PEF={pef}")
                    if fcf is not None:
                        parts.append(f"FCF={fcf}%")
                    row_reasons.append(
                        f"negative earnings ({', '.join(parts) if parts else 'all ≤0'})"
                    )
                    gate_counts["positive_earnings"] += 1
                    passed = False

        # Gate 4 — trend / 52W (missing → leave open; only fail if present and below floor)
        w52_raw = _get(row, "52W")
        w52 = _parse_52w(w52_raw)
        if w52 is not None and w52 < cfg["min_52w"]:
            row_reasons.append(f"melting: 52W {w52:.0f} < {cfg['min_52w']}")
            gate_counts["trend"] += 1
            passed = False

        reasons[tkr] = row_reasons
        if passed:
            eligible_rows.append(idx)

    eligible_df = working.loc[eligible_rows].copy() if eligible_rows else working.iloc[0:0].copy()

    summary = {
        "total": len(working),
        "fundamental": len(reasons),
        "eligible": len(eligible_rows),
        "price_only": len(price_only),
        "excluded": len(excluded),
        "gate_failures": gate_counts,
    }

    return {
        "eligible": eligible_df,
        "price_only": price_only,
        "excluded": excluded,
        "reasons": reasons,
        "summary": summary,
    }
