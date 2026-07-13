import re

import pandas as pd

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


def load_universe(
    etoro_csv_path: str, min_price: float = 1.0, min_cap_usd: float = 5e8
) -> list[str]:
    df = pd.read_csv(etoro_csv_path, na_values=["--"])
    tkr = df["TKR"].astype(str)
    mask_us = tkr.str.match(US_TICKER)
    price = pd.to_numeric(df["PRC"], errors="coerce")
    cap = df["CAP"].map(parse_cap)
    keep = mask_us & (price > min_price) & (cap >= min_cap_usd)
    return sorted(df.loc[keep, "TKR"].dropna().astype(str).unique().tolist())
