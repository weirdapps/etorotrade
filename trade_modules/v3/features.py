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

import math

import pandas as pd

from trade_modules.riskfirst.fx import currency_of
from trade_modules.v3.fetch import robust_fetch_prices
from trade_modules.v3.integrity import validate_panel
from trade_modules.v3.universe import parse_cap

# Approximate spot FX rates (local currency -> USD), for normalizing eToro's
# local-currency market cap and dollar-volume to USD. eToro reports CAP in the
# listing's local currency (yen for .T, EUR for .DE), and price is local too, so
# without this the mega/large/mid cap tiers, the cap_ordered vol mode, the report's
# Mkt-Cap display, and the runner's $1M ADV floor would mix currencies (a small yen
# name reading as a USD mega-cap). Note: universe.py::load_universe is US-only
# (^[A-Z]+$), so its $500M floor never sees a non-USD cap — this normalization serves
# the cross-market scoring/overlay paths, not that floor. Coarse by design (tiers +
# ADV floor are wide, so FX drift is immaterial); NOT used for P&L. Refresh occasionally.
_USD_RATE = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "CHF": 1.12,
    "JPY": 0.0067,
    "HKD": 0.128,
    "CAD": 0.73,
    "AUD": 0.66,
    "SEK": 0.095,
    "NOK": 0.094,
    "DKK": 0.145,
    "KRW": 0.00073,
    "TWD": 0.031,
    "SGD": 0.74,
    "INR": 0.012,
    "CNY": 0.138,
    "BRL": 0.18,
    "MXN": 0.058,
    "PLN": 0.25,
}


def _usd_rate_for(ticker: str) -> float:
    """Approximate local-currency -> USD spot rate for a ticker's listing currency."""
    return _USD_RATE.get(currency_of(str(ticker)), 1.0)


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
    "#A": "n_analysts",  # analyst count — coverage floor for the analyst-momentum buy veto
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
    "revenueGrowth": "rev_growth",  # growth cluster (yoy revenue growth, fraction)
}
_INFO_STR = {
    "sector": "sector",
    "industry": "industry",
    "country": "country",  # domicile — drives dual-listing (mother-market) dedup
    "quoteType": "quote_type",
    "longBusinessSummary": "description",
}

# Maximum chars for the business description shown on report cards.
_DESC_MAX = 220


def _truncate_description(text) -> str:
    """Truncate a business summary to ~220 chars ending at a clean boundary.

    Tries to end at the last sentence boundary (``. ! ?``) within the limit;
    falls back to the last word boundary; appends ``…`` only for word cuts.
    Returns ``""`` for missing / empty / non-string input (NaN-tolerant).
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.strip()
    if len(text) <= _DESC_MAX:
        return text
    # Last sentence-end punctuation within the window.
    window = text[: _DESC_MAX + 1]
    for ch in ".!?":
        pos = window.rfind(ch)
        if pos > 0:
            return text[: pos + 1].strip()
    # Word-boundary fallback.
    pos = window.rfind(" ")
    if pos > 0:
        return text[:pos].rstrip(",.;:") + "…"
    return text[:_DESC_MAX] + "…"


# Price-spine windows (mirror trade_modules.v3.spine).
_MOM_SKIP = 21
_MOM_LOOKBACK = 252

# Trade-level windows: monthly move ~= daily vol × √21; annualized vol ÷ √252.
_TRADING_DAYS_MONTH = 21
_TRADING_DAYS_YEAR = 252
# 2·sigma_m must stay below 1 so the stop is a positive price.
_SIGMA_M_CAP = 0.5


def trade_levels(entry, realized_vol) -> dict[str, float]:
    """Vol-scaled stop-loss / take-profit levels from annualized realized vol.

    ``realized_vol`` is the annualized (×√252) Close-based realized vol from
    :func:`_price_factors`. The approximate monthly move fraction is
    ``sigma_m = (realized_vol / √252) · √21``. Levels:

        stop_loss   = entry · (1 - 2·sigma_m)
        take_profit = entry · (1 + 3·sigma_m)
        rr          = (take_profit - entry) / (entry - stop_loss)  (= 1.5)

    ``entry`` echoes the price whenever the price itself is valid. Degenerate
    inputs (NaN/≤0 entry or vol, so ``sigma_m`` NaN/0, or a vol so large the
    stop would be non-positive) yield NaN levels; the report shows "n/a".
    """
    nan = float("nan")
    try:
        e = float(entry)
    except (TypeError, ValueError):
        e = nan
    e = e if (math.isfinite(e) and e > 0) else nan

    try:
        v = float(realized_vol)
    except (TypeError, ValueError):
        v = nan

    out = {"entry": e, "sigma_m": nan, "stop_loss": nan, "take_profit": nan, "rr": nan}
    if math.isnan(e) or not (math.isfinite(v) and v > 0):
        return out

    sigma_m = (v / math.sqrt(_TRADING_DAYS_YEAR)) * math.sqrt(_TRADING_DAYS_MONTH)
    if not (math.isfinite(sigma_m) and 0 < sigma_m < _SIGMA_M_CAP):
        return out

    stop = e * (1.0 - 2.0 * sigma_m)
    target = e * (1.0 + 3.0 * sigma_m)
    out.update(sigma_m=sigma_m, stop_loss=stop, take_profit=target, rr=(target - e) / (e - stop))
    return out


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


# yfinance row-label candidates for Sloan accruals (defensive: label varies by ticker/version).
_NI_LABELS = ("Net Income", "Net Income Common Stockholders")
_OCF_LABELS = ("Operating Cash Flow", "Total Cash From Operating Activities")
_TA_LABELS = ("Total Assets",)


def _first_label_value(df, labels: tuple) -> float:
    """Return the most-recent non-null value for the first matching row label."""
    nan = float("nan")
    if df is None or df.empty:
        return nan
    for label in labels:
        if label in df.index:
            row = pd.to_numeric(df.loc[label], errors="coerce").dropna()
            if not row.empty:
                return float(row.iloc[0])
    return nan


def _two_most_recent(df, labels: tuple) -> list:
    """Return up to 2 most-recent non-null floats for the first matching row label."""
    if df is None or df.empty:
        return []
    for label in labels:
        if label in df.index:
            row = pd.to_numeric(df.loc[label], errors="coerce").dropna()
            return [float(v) for v in row.iloc[:2]]
    return []


def _default_accruals_fetch(tickers: list[str]) -> dict[str, float]:
    """Throttled per-ticker yfinance accruals fetch (Hribar-Collins, cash-flow basis).

    accruals = (net_income - operating_cash_flow) / average_total_assets

    where average_total_assets = mean of the two most recent annual values.
    Lower / more-negative accruals = higher earnings quality = GOOD.
    Sleeps ~0.3 s between tickers, retries twice on exception, skips
    (never raises) tickers where data is unavailable.
    Imported at call time so the module stays importable without yfinance.
    """
    import math  # noqa: PLC0415
    import time  # noqa: PLC0415

    import yfinance as yf  # noqa: PLC0415

    from trade_modules.config_manager import get_config  # noqa: PLC0415

    resolve = get_config().get_data_fetch_ticker  # eToro ticker -> Yahoo symbol
    out: dict[str, float] = {}
    for t in tickers:
        for attempt in range(3):
            try:
                tkr = yf.Ticker(resolve(t))
                ni = _first_label_value(tkr.financials, _NI_LABELS)
                ocf = _first_label_value(tkr.cashflow, _OCF_LABELS)
                ta_vals = _two_most_recent(tkr.balance_sheet, _TA_LABELS)
                if len(ta_vals) >= 2:
                    avg_ta = (ta_vals[0] + ta_vals[1]) / 2.0
                elif len(ta_vals) == 1:
                    avg_ta = ta_vals[0]
                else:
                    avg_ta = float("nan")
                if (
                    math.isfinite(ni)
                    and math.isfinite(ocf)
                    and math.isfinite(avg_ta)
                    and avg_ta != 0
                ):
                    out[t] = (ni - ocf) / avg_ta
                break
            except Exception:  # noqa: BLE001
                if attempt < 2:
                    time.sleep(0.3 * (attempt + 1))
        time.sleep(0.3)
    return out


def _info_cache_path() -> str:
    """Daily on-disk cache for yfinance ``.info`` (keyed by Yahoo symbol)."""
    import os  # noqa: PLC0415
    from datetime import date  # noqa: PLC0415

    return os.path.expanduser(f"~/.weirdapps-trading/v3_info_cache_{date.today():%Y%m%d}.json")


def _load_info_cache() -> dict[str, dict]:
    import json  # noqa: PLC0415

    try:
        with open(_info_cache_path()) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001  (best-effort: missing/corrupt -> refetch)
        return {}


def _save_info_cache(cache: dict[str, dict]) -> None:
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415

    try:
        path = _info_cache_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(cache, fh, default=str)
    except Exception:  # noqa: BLE001  (best-effort cache; never fail the run)
        pass


def _info_is_populated(info) -> bool:
    """A rate-limited ``.info`` comes back as an empty/sparse dict WITHOUT raising.
    Treat 'no shortName and no price' as a fetch failure so it is retried."""
    return isinstance(info, dict) and bool(
        info.get("shortName") or info.get("regularMarketPrice") is not None
    )


def _default_info_fetch(tickers: list[str]) -> dict[str, dict]:
    """Throttled per-ticker yfinance ``.info`` fetch, provider-mapped + rate-limit-safe.

    Maps each eToro ticker to its Yahoo symbol (``get_data_fetch_ticker``: ``.NV``→``.AS``,
    ``.IM``→``.MI``, currency-line strip, class-share dash, …) before fetching, but keys the
    result by the ORIGINAL eToro ticker. Retries on exception AND on an empty/sparse dict
    (yfinance returns ``{}`` when rate-limited, without raising) with exponential back-off.
    A daily on-disk cache keyed by Yahoo symbol skips already-fetched names (the overlay runs
    every 4h — the cache collapses ~6 fetches/day to ~1). Never raises; unfetchable → ``{}``.
    """
    import time  # noqa: PLC0415

    import yfinance as yf  # noqa: PLC0415

    from trade_modules.config_manager import get_config  # noqa: PLC0415

    resolve = get_config().get_data_fetch_ticker
    cache = _load_info_cache()
    out: dict[str, dict] = {}
    dirty = False
    for t in tickers:
        y = resolve(t)
        if y in cache:  # cache hit (populated result stored earlier today)
            out[t] = cache[y]
            continue
        info: dict = {}
        for attempt in range(3):  # initial try + 2 retries
            try:
                got = yf.Ticker(y).info
                if _info_is_populated(got):
                    info = got
                    break
            except Exception:  # noqa: BLE001
                pass
            if attempt < 2:
                time.sleep(0.5 * (3**attempt))  # 0.5s, 1.5s back-off (rate-limit recovery)
        out[t] = info
        if info:  # cache only populated results, so rate-limited names retry next run
            cache[y] = info
            dirty = True
        time.sleep(0.3)
    if dirty:
        _save_info_cache(cache)
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
        rets = s.iloc[-_MOM_LOOKBACK:].pct_change(fill_method=None).dropna()
        vol = rets.std(ddof=0) * (252**0.5) if len(rets) > 1 else float("nan")
        out.loc[tkr, "mom_12_1"] = float(mom)
        out.loc[tkr, "realized_vol"] = float(vol)
    return out


# --- cross-listing ticker reconciliation ---------------------------------- #
# An account/holding key (from the eToro account API: NVDA, GILD, T.US, SBMO.NV)
# can differ from the signal-CSV key (NVDA.EUR, GILD.L, T, SBMO.AS). Native
# price/cap/factors come only from the CSV, so an unmatched key NaNs the whole
# row -> the name is ineligible -> spuriously SOLD. We resolve a target ticker to
# its CSV row by UNDERLYING-COMPANY ROOT, but only on a UNIQUE match, so an
# ambiguous or class-share ticker is never wrong-matched (it stays NaN and is
# HOLD-flagged downstream, never mis-priced).
_VENUE_SUFFIXES = frozenset(
    {
        # currency / price-unit lines eToro appends
        "EUR",
        "USD",
        "GBP",
        "GBX",
        "CHF",
        "JPY",
        "HKD",
        "SEK",
        "NOK",
        "DKK",
        "CAD",
        "AUD",
        "SGD",
        "PLN",
        "ZAR",
        "INR",
        "KRW",
        "TWD",
        "CNY",
        "ILS",
        "TRY",
        "AED",
        # exchange venues (eToro / Yahoo suffixes)
        "US",
        "L",
        "LN",
        "NV",
        "AS",
        "IM",
        "MI",
        "DE",
        "PA",
        "MC",
        "SW",
        "CH",
        "ST",
        "OL",
        "CO",
        "HE",
        "BR",
        "VI",
        "LS",
        "IR",
        "TO",
        "AX",
        "NZ",
        "SI",
        "HK",
        "T",
        "TW",
        "KS",
        "KQ",
        "SR",
        "SA",
        "MX",
        "JK",
        "BK",
        "F",
        "BE",
        "DU",
        "HM",
        "MU",
        "SG",
    }
)


def _underlying_root(ticker: str) -> str:
    """Strip trailing exchange/currency suffixes to the underlying-company root.

    ``NVDA.EUR`` -> ``NVDA``, ``T.US`` -> ``T``, ``SBMO.AS`` -> ``SBMO``,
    ``KSP.L.GBX`` -> ``KSP``. Class-share suffixes are preserved (``BRK.B`` stays
    ``BRK.B`` because ``B`` is not a venue token), so distinct share classes never
    collapse. Never strips below the first segment.
    """
    parts = str(ticker).upper().split(".")
    while len(parts) > 1 and parts[-1] in _VENUE_SUFFIXES:
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_csv_keys(raw_index, tickers) -> dict:
    """Map each target ticker to its signal-CSV row key.

    Exact key wins; otherwise a target resolves to a CSV key with the same
    underlying root — but ONLY when that root maps to exactly one CSV key
    (ambiguous roots are left unresolved). Unresolved targets map to themselves
    (-> a NaN native row -> HOLD-flagged downstream, never a wrong match).
    """
    keys = [str(k) for k in raw_index]
    exact = set(keys)
    counts: dict[str, int] = {}
    for k in keys:
        r = _underlying_root(k)
        counts[r] = counts.get(r, 0) + 1
    root_to_key = {_underlying_root(k): k for k in keys if counts[_underlying_root(k)] == 1}
    out: dict[str, str] = {}
    for t in tickers:
        ts = str(t)
        out[ts] = ts if ts in exact else root_to_key.get(_underlying_root(ts), ts)
    return out


def enrich_features(
    tickers,
    etoro_csv_path,
    price_period: str = "2y",
    info_fetch=None,
    price_fetch=None,
    accruals_fetch=None,
    sector_map: dict | None = None,
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
        accruals_fetch: ``(tickers) -> {ticker: float}``; computes Sloan
            accruals (net_income − ocf) / avg_total_assets.  Defaults to
            :func:`_default_accruals_fetch` (throttled yfinance, skip-not-raise).
            Pass ``lambda tickers: {}`` to skip the network call in report runs
            that supply a pre-cached value or don't need the column.
        sector_map: optional ``{TICKER_UPPER: sector}`` offline map (static index
            map + persistent cache, see ``trade_modules.v3.sectors``). When given,
            ``sector`` is resolved offline-first, falling back to live yfinance for
            names the map does not cover. When None/empty, behaviour is unchanged
            (live yfinance only).

    Returns:
        pd.DataFrame indexed by ticker with native + added + derived columns.
        Tickers absent from ``.info`` still appear (native present, added NaN).
        ``accruals`` is always present; NaN when unavailable.
    """
    tickers = list(dict.fromkeys(str(t) for t in tickers))  # de-dupe, preserve order
    if info_fetch is None:
        info_fetch = _default_info_fetch
    if price_fetch is None:
        price_fetch = robust_fetch_prices
    if accruals_fetch is None:
        accruals_fetch = _default_accruals_fetch

    # --- (1) native factors from the etoro CSV ---
    raw = pd.read_csv(etoro_csv_path, na_values=["--"])
    raw = validate_panel(raw, source=str(etoro_csv_path))
    raw = raw.drop_duplicates(subset="TKR", keep="first").set_index("TKR")
    native = pd.DataFrame(index=raw.index)
    native["name"] = raw["NAME"].astype("object") if "NAME" in raw.columns else pd.NA
    native["cap"] = raw["CAP"].map(parse_cap) if "CAP" in raw.columns else float("nan")
    # Normalize local-currency market cap -> USD (see _USD_RATE) so the floor + tiers
    # are single-currency (Toyota's yen cap no longer reads as a USD mega-cap).
    native["cap"] = native["cap"] * native.index.to_series().map(_usd_rate_for)
    for col, feat in _NATIVE_NUM.items():
        native[feat] = _num(raw[col]) if col in raw.columns else float("nan")
    # Resolve held/target keys to their CSV row (exact, else unique underlying-root
    # match) so a cross-listing label mismatch (NVDA vs NVDA.EUR) no longer NaNs the
    # row. Unresolved targets stay NaN (HOLD-flagged downstream, never mis-priced).
    _csv_key = _resolve_csv_keys(native.index, tickers)
    native = native.reindex([_csv_key[str(t)] for t in tickers])
    native.index = [str(t) for t in tickers]

    # --- (2) added metrics from yfinance .info ---
    info = info_fetch(tickers) or {}
    added = pd.DataFrame(index=tickers)
    for key, feat in _INFO_NUM.items():
        vals = pd.Series([info.get(t, {}).get(key) for t in tickers], index=tickers)
        added[feat] = pd.to_numeric(vals, errors="coerce")
    for key, feat in _INFO_STR.items():
        added[feat] = [info.get(t, {}).get(key) for t in tickers]
    # BUILD ③: prefer the offline/cached sector map (static index > cache) when
    # provided; fall back to the live yfinance .info sector for names it misses.
    if sector_map:
        added["sector"] = [
            sector_map.get(str(t).upper()) or info.get(t, {}).get("sector") for t in tickers
        ]
    # Truncate description to ~220 chars at a clean boundary; NaN/None → "".
    added["description"] = added["description"].apply(_truncate_description)

    feats = native.join(added)

    # --- (3) derived factors ---
    price = feats["price"]
    safe_price = price.where(price > 0)
    disp = (feats["target_high"] - feats["target_low"]) / safe_price
    feats["target_dispersion"] = disp.replace([float("inf"), float("-inf")], float("nan"))
    # earnings trajectory = forward/trailing P/E (PEF/PET, owner 2026-07-23): <1 = forward
    # cheaper = earnings expected to RISE; >1 = value-trap (earnings expected to fall).
    # Scored via the trajectory_z cluster with DIRECTION -1 (smaller is better).
    pet = pd.to_numeric(feats["pe_trailing"], errors="coerce")
    pef = pd.to_numeric(feats["pe_forward"], errors="coerce")
    feats["earn_trajectory"] = (pef / pet).where((pet > 0) & (pef > 0))
    # adv_usd = shares * local price * FX -> USD dollar-volume (for the ADV floor).
    feats["adv_usd"] = feats["avg_volume"] * price * feats.index.to_series().map(_usd_rate_for)

    prices = price_fetch(list(tickers), period=price_period)
    feats = feats.join(_price_factors(prices, tickers))

    # --- (4) vol-scaled trade levels (entry / stop / target / R:R) ---
    lv = pd.DataFrame(
        [trade_levels(p, v) for p, v in zip(feats["price"], feats["realized_vol"], strict=False)],
        index=feats.index,
    )
    for col in ("entry", "sigma_m", "stop_loss", "take_profit", "rr"):
        feats[col] = lv[col]

    # --- (5) Sloan accruals — earnings-quality factor (optional/lazy) ---
    # accruals = (net_income - operating_cash_flow) / average_total_assets.
    # Lower / more-negative = higher earnings quality = GOOD.
    # NaN-tolerant: missing tickers or fetch failures leave the column NaN.
    accruals_data = accruals_fetch(list(tickers)) or {}
    feats["accruals"] = pd.to_numeric(
        pd.Series([accruals_data.get(t) for t in feats.index], index=feats.index),
        errors="coerce",
    )

    # --- (6) PIT fundamentals from the Sharadar SF1 store: GP/assets (quality) + SUE
    # (PEAD). Latest filing per ticker; NaN for non-US / no-filing names (US-only store),
    # so the quality / PEAD clusters degrade gracefully. NaN-safe if the store is absent.
    from trade_modules.v3.fundamentals import live_fundamentals_factors  # noqa: PLC0415

    ff = live_fundamentals_factors(list(feats.index))
    for _c in ("gp_assets", "sue", "net_issuance", "earn_stability"):
        feats[_c] = ff[_c].reindex(feats.index)

    feats.index.name = "ticker"
    return feats
