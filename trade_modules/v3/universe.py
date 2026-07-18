import re

import pandas as pd

from trade_modules.v3.constants import COVERAGE_FACTORS, MIN_FACTOR_COVERAGE
from trade_modules.v3.integrity import validate_panel

US_TICKER = re.compile(r"^[A-Z]+$")  # US/USD listing = no exchange suffix
_SUFFIX = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3, "k": 1e3}


def parse_cap(v) -> float:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return float("nan")
    s = str(v).strip().replace(",", "")
    m = re.match(r"^([0-9.]+)\s*([TBMKk]?)$", s)
    if not m:
        return float("nan")
    return float(m.group(1)) * _SUFFIX.get(m.group(2), 1.0)


def _factor_coverage_count(
    df: pd.DataFrame, factors: tuple[str, ...] = COVERAGE_FACTORS
) -> pd.Series:
    """Per-row count of ``factors`` whose value is numeric-present.

    Mirrors ``features._num`` cleaning (strip ``%`` and thousands commas). Columns
    absent from ``df`` count as not-present for every row.
    """
    total = pd.Series(0, index=df.index)
    for col in factors:
        if col not in df.columns:
            continue
        cleaned = pd.to_numeric(
            df[col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip(),
            errors="coerce",
        )
        total = total + cleaned.notna().astype(int)
    return total


def load_universe(
    etoro_csv_path: str,
    min_price: float = 1.0,
    min_cap_usd: float = 5e8,
    min_factor_coverage: int = 0,
    us_only: bool = False,
) -> list[str]:
    """Cap/price-filtered ticker list — GLOBAL by default (multi-currency book).

    eToro reports CAP and PRC in the listing's LOCAL currency, so cap and price are
    FX-normalized to USD (``_usd_rate_for``) before the ``$500M`` / ``$1`` floors are
    applied — otherwise a ¥ or HK$ value would be mis-compared against a USD floor.
    For US listings the rate is 1.0, so their behaviour is unchanged.

    Args:
        us_only: restore the legacy US-listings-only filter (no exchange suffix).
        min_factor_coverage: > 0 keeps only names with that many panel-native factors
            present (``COVERAGE_FACTORS``) — the BUILD ④ coverage gate.
    """
    df = pd.read_csv(etoro_csv_path, na_values=["--"])
    df = validate_panel(
        df,
        source=str(etoro_csv_path),
        required_columns=("TKR", "PRC", "CAP"),
        required_numeric=("PRC",),
    )
    # Lazy import: features imports universe (parse_cap), so importing it at module
    # load would cycle; at call time features is already resolved.
    from trade_modules.v3.features import _usd_rate_for

    tkr = df["TKR"].astype(str)
    fx = tkr.map(_usd_rate_for)  # local currency -> USD (1.0 for US listings)
    price_usd = pd.to_numeric(df["PRC"], errors="coerce") * fx
    cap_usd = df["CAP"].map(parse_cap) * fx
    keep = (price_usd > min_price) & (cap_usd >= min_cap_usd)
    if us_only:
        keep = keep & tkr.str.match(US_TICKER)
    if min_factor_coverage > 0:
        keep = keep & (_factor_coverage_count(df) >= min_factor_coverage)
    return sorted(df.loc[keep, "TKR"].dropna().astype(str).unique().tolist())


def assemble_scored_universe(
    etoro_csv_path: str,
    holdings,
    candidates,
    *,
    min_factor_coverage: int = MIN_FACTOR_COVERAGE,
) -> list[str]:
    """Scored universe = coverage-gated broad set  ∪  holdings  ∪  candidates.

    Coverage-gating (>= ``min_factor_coverage`` of the panel-native factors) is the
    PRIMARY selector, replacing the analyst AND-gate. ``holdings`` (current positions)
    and ``candidates`` (analyst buy names) are ALWAYS included even below the
    threshold — a held name is never dropped for thin data. De-duped; holdings first,
    then candidates, then the coverage-gated names.
    """
    covered = load_universe(etoro_csv_path, min_factor_coverage=min_factor_coverage)
    ordered = [*(str(t) for t in holdings), *(str(t) for t in candidates), *covered]
    return list(dict.fromkeys(ordered))
