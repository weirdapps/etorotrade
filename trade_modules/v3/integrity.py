"""BUILD ② (2026-07-18): fail-loud structural integrity gate for the etoro.csv panel.

The v3 loaders (``universe.load_universe``, ``features.enrich_features``) previously
did a bare ``pd.read_csv`` with no schema or dtype check, so a silently dropped
column or a same-field-count column-shift (a company name comma-splitting a row)
would degrade every downstream factor to NaN with no signal. This gate ASSERTS the
panel's shape so corruption STOPS the pipeline instead of quietly poisoning the book.

Two corruption modes, two catchers:
  * an EXTRA-field comma-shift produces more fields than the header -> pandas' C
    parser already raises ParserError at read time (loud, upstream of this gate);
  * a SAME-field-count misalignment (an upstream column dropped, values shifted)
    slips past pandas -> caught here: required columns must be present, and the
    confirmed-numeric factor columns must actually coerce to numbers.

Deliberately scoped to CORRUPTION, not COVERAGE:
  * CAP/SZ/ERN are legitimately non-numeric (suffix/date/flag) and are NEVER
    numeric-checked (CAP='39.2T');
  * a sparse or entirely-empty column is missingness, left to the coverage gate —
    only garbage IN a numeric column raises here.

The panel is clean today (RFC-4180 quoted, 31 cols, every factor column 100%
numeric); this is a forward guard, not a repair.
"""

from __future__ import annotations

import pandas as pd


class PanelIntegrityError(ValueError):
    """Raised when the etoro.csv panel fails a structural integrity check."""


# Columns v3 needs present to build the universe + the active factor set.
PANEL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "TKR",
    "NAME",
    "CAP",
    "PRC",
    "PET",
    "ROE",
    "FCF",
    "52W",
    "B",
    "AM",
    "EG",
)

# Confirmed plain-numeric factor columns (empirically 100% numeric on the clean
# panel). CAP is excluded — it is suffixed ('39.2T'); TKR/NAME are strings.
PANEL_REQUIRED_NUMERIC: tuple[str, ...] = ("PRC", "PET", "ROE", "FCF", "52W", "B", "AM", "EG")

# A confirmed-numeric column whose PRESENT values coerce below this fraction is
# corrupt (a column-shift injected strings). The clean panel is 1.000; corruption
# collapses toward 0, so the margin is deliberately wide.
NUMERIC_MIN_FRAC = 0.90

# The ticker column is the row identity; near-total nulls = catastrophic corruption.
TICKER_MIN_NONNULL_FRAC = 0.95


def _numeric_fraction(s: pd.Series) -> tuple[int, float]:
    """Return (#present values, fraction of them that coerce to a number).

    Mirrors ``features._num`` cleaning (strip ``%`` and thousands commas). "Present"
    = non-null and non-blank. Returns ``(0, 1.0)`` for an entirely-empty column so
    missingness (coverage) never trips the corruption gate.
    """
    present = s.dropna()
    present = present[present.astype(str).str.strip() != ""]
    if len(present) == 0:
        return 0, 1.0
    cleaned = pd.to_numeric(
        present.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )
    return len(present), float(cleaned.notna().mean())


def validate_panel(
    df: pd.DataFrame,
    *,
    source: str = "etoro.csv",
    required_columns: tuple[str, ...] = PANEL_REQUIRED_COLUMNS,
    required_numeric: tuple[str, ...] = PANEL_REQUIRED_NUMERIC,
) -> pd.DataFrame:
    """Fail-loud structural integrity gate. Returns ``df`` unchanged on success.

    Args:
        df: the raw panel as loaded by ``pd.read_csv`` (original column names).
        source: path / label used in error messages.
        required_columns: columns that must be present (``load_universe`` passes a
            minimal ``('TKR','PRC','CAP')``; the feature path uses the full default).
        required_numeric: confirmed-numeric columns whose present values must coerce.

    Raises:
        PanelIntegrityError: listing EVERY violation found in one pass.
    """
    if df is None or len(df) == 0:
        raise PanelIntegrityError(f"{source}: panel is empty (0 rows)")

    problems: list[str] = []

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        problems.append(f"missing required columns: {missing}")

    if "TKR" in df.columns:
        t = df["TKR"].astype(str).str.strip()
        nonnull_frac = float((t.ne("") & t.ne("nan") & t.ne("None")).mean())
        if nonnull_frac < TICKER_MIN_NONNULL_FRAC:
            problems.append(
                f"TKR only {nonnull_frac:.1%} non-null (< {TICKER_MIN_NONNULL_FRAC:.0%}) — "
                "likely a column shift"
            )

    for col in required_numeric:
        if col not in df.columns:
            continue  # already reported by the presence check
        n, frac = _numeric_fraction(df[col])
        if n > 0 and frac < NUMERIC_MIN_FRAC:
            problems.append(
                f"column {col!r} only {frac:.1%} numeric over {n} present values "
                f"(< {NUMERIC_MIN_FRAC:.0%}) — a column shift likely injected non-numeric data"
            )

    if problems:
        raise PanelIntegrityError(f"{source}: " + "; ".join(problems))
    return df
