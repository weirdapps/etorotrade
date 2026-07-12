"""v3 feature enrichment.

Merges three sources into one per-ticker feature frame:
  1. NATIVE factors from the etoro CSV (31-col schema shared by
     portfolio.csv / buy.csv / etoro.csv).
  2. ADDED metrics from yfinance ``.info`` (injectable ``info_fetch``).
  3. DERIVED factors: target_dispersion, adv_usd, and price-spine
     mom_12_1 / realized_vol at the last available bar (injectable
     ``price_fetch``, defaults to the repo's robust batched fetcher).

Network access is confined to the default fetchers, which are only used
when no fake is injected — the unit tests inject fakes, so they never hit
the network.
"""

from __future__ import annotations

import pandas as pd

from trade_modules.v3.fetch import robust_fetch_prices
from trade_modules.v3.universe import parse_cap

# etoro CSV header -> feature name (NATIVE numeric factors)
_NATIVE_NUM = {
    "PRC": "price",
    "PET": "pe_trailing",
    "PEF": "pe_forward",
    "P/S": "ps_sector",
    "PEG": "peg",
    "ROE": "roe",
    "DE": "de",
    "FCF": "fcf",
    "B": "beta",
    "52W": "pct_52w_high",
    "PP": "price_perf",
    "SI": "short_interest",
    "AM": "analyst_mom",
    "EG": "earn_growth",
    "DV": "div_yield",
    "UP%": "upside",
    "%B": "buy_pct",
}

# yfinance .info key -> feature name (ADDED numeric metrics)
_INFO_NUM = {
    "priceToBook": "pb",
    "enterpriseToEbitda": "ev_ebitda",
    "returnOnAssets": "roa",
    "grossMargins": "gross_margin",
    "operatingMargins": "op_margin",
    "currentRatio": "current_ratio",
    "targetHighPrice": "target_high",
    "targetLowPrice": "target_low",
    "averageVolume": "avg_volume",
}
_INFO_STR = {"sector": "sector", "industry": "industry"}

# Price-spine windows (mirror trade_modules.v3.spine).
_MOM_SKIP = 21
_MOM_LOOKBACK = 252


def _num(s: pd.Series) -> pd.Series:
    """Coerce a mixed string/number series to float.

    Strips ``%`` and thousands commas; ``--`` / blanks / unparseable -> NaN.
    """
    return pd.to_numeric(
        s.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def _default_info_fetch(tickers: list[str]) -> dict[str, dict]:
    """Throttled per-ticker yfinance ``.info`` fetch.

    Sleeps ~0.3s between tickers, retries twice on exception, and skips
    (never raises) a ticker whose info cannot be retrieved.  Imported at
    call time so the module stays importable without yfinance installed.
    """
    import time  # noqa: PLC0415

    import yfinance as yf  # noqa: PLC0415

    out: dict[str, dict] = {}
    for t in tickers:
        for attempt in range(3):  # initial try + 2 retries
            try:
                info = yf.Ticker(t).info
                out[t] = info if isinstance(info, dict) else {}
                break
            except Exception:  # noqa: BLE001
                if attempt < 2:
                    time.sleep(0.3 * (attempt + 1))
        time.sleep(0.3)
    return out


def _price_factors(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Raw 12-1 momentum and annualized realized vol at the last bar, per ticker.

    Uses the same window logic as :mod:`trade_modules.v3.spine`
    (skip=21, lookback=252) but returns RAW values (not cross-sectional z)
    so the combiner can apply its own directional z-scoring — critically,
    realized_vol must stay a positive "high = bad" quantity so the combiner
    can negate it correctly.  Vol is annualized (× √252) for readability;
    the constant scaling is invariant under the combiner's z-score.
    """
    out = pd.DataFrame(index=tickers, columns=["mom_12_1", "realized_vol"], dtype=float)
    if prices is None or prices.empty:
        return out
    px = prices.sort_index()
    for tkr in tickers:
        if tkr not in px.columns:
            continue
        s = px[tkr].dropna()
        if len(s) < _MOM_LOOKBACK + 1:
            continue
        mom = s.iloc[-1 - _MOM_SKIP] / s.iloc[-1 - _MOM_LOOKBACK] - 1.0
        rets = s.iloc[-_MOM_LOOKBACK:].pct_change().dropna()
        vol = rets.std(ddof=0) * (252**0.5) if len(rets) > 1 else float("nan")
        out.loc[tkr, "mom_12_1"] = float(mom)
        out.loc[tkr, "realized_vol"] = float(vol)
    return out


def enrich_features(
    tickers,
    etoro_csv_path,
    price_period: str = "2y",
    info_fetch=None,
    price_fetch=None,
) -> pd.DataFrame:
    """Build the merged per-ticker feature frame (indexed by ticker).

    Args:
        tickers: Iterable of ticker symbols to enrich (de-duped, order kept).
        etoro_csv_path: Path to an etoro/portfolio/buy CSV (31-col schema).
        price_period: yfinance period for the price-derived factors.
        info_fetch: ``(tickers) -> {ticker: info_dict}``; defaults to a
            throttled yfinance fetcher.  Inject a fake in tests.
        price_fetch: ``(tickers, period=...) -> dates×tickers close frame``;
            defaults to :func:`robust_fetch_prices`.  Inject a fake in tests.

    Returns:
        pd.DataFrame indexed by ticker with native + added + derived columns.
        Tickers absent from ``.info`` still appear (native present, added NaN).
    """
    tickers = list(dict.fromkeys(str(t) for t in tickers))  # de-dupe, preserve order
    if info_fetch is None:
        info_fetch = _default_info_fetch
    if price_fetch is None:
        price_fetch = robust_fetch_prices

    # --- (1) native factors from the etoro CSV ---
    raw = pd.read_csv(etoro_csv_path, na_values=["--"])
    raw = raw.drop_duplicates(subset="TKR", keep="first").set_index("TKR")
    native = pd.DataFrame(index=raw.index)
    native["name"] = raw["NAME"].astype("object") if "NAME" in raw.columns else pd.NA
    native["cap"] = raw["CAP"].map(parse_cap) if "CAP" in raw.columns else float("nan")
    for col, feat in _NATIVE_NUM.items():
        native[feat] = _num(raw[col]) if col in raw.columns else float("nan")
    native = native.reindex(tickers)

    # --- (2) added metrics from yfinance .info ---
    info = info_fetch(tickers) or {}
    added = pd.DataFrame(index=tickers)
    for key, feat in _INFO_NUM.items():
        vals = pd.Series([info.get(t, {}).get(key) for t in tickers], index=tickers)
        added[feat] = pd.to_numeric(vals, errors="coerce")
    for key, feat in _INFO_STR.items():
        added[feat] = [info.get(t, {}).get(key) for t in tickers]

    feats = native.join(added)

    # --- (3) derived factors ---
    price = feats["price"]
    safe_price = price.where(price > 0)
    disp = (feats["target_high"] - feats["target_low"]) / safe_price
    feats["target_dispersion"] = disp.replace([float("inf"), float("-inf")], float("nan"))
    feats["adv_usd"] = feats["avg_volume"] * price

    prices = price_fetch(list(tickers), period=price_period)
    feats = feats.join(_price_factors(prices, tickers))

    feats.index.name = "ticker"
    return feats
