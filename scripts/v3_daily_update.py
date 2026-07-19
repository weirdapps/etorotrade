"""Ingest Sharadar DAILY (survivorship-clean daily marketcap + pb/pe/ps) into a
month-end store — the survivorship fix that SEP-sample can't give us.

DAILY is in the SF1 bundle AND covers delisted names (verified: BBBY has full
2016-2026 DAILY history; SEP does not). We build the true US common-stock universe
INCLUDING delisted names from the TICKERS table, pull DAILY month-end marketcap
(a survivorship-clean return proxy) plus the point-in-time pb/pe/ps valuation
ratios, and store them append-only. This removes the survivorship bias from the
fundamentals backtest (my SF1 store was scoped to eToro's CURRENT universe, missing
the ~2,915 delisted names).

Caveat: monthly marketcap growth ≈ price return only while share count is stable;
it is contaminated by issuance/buybacks (mainly affecting the asset-growth factor).
Exact prices need SEP-full (a separate subscription) — flagged, not required here.

Auth: NDL_API_KEY.  Usage: .venv/bin/python scripts/v3_daily_update.py --since 2015-01-01
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

STORE = str(Path("~/.weirdapps-trading/v3_daily_monthly.parquet").expanduser())
UNIVERSE_FILE = str(Path("~/.weirdapps-trading/v3_us_universe.txt").expanduser())
BASE = "https://data.nasdaq.com/api/v3/datatables/SHARADAR"
VALUE_COLS = ["marketcap", "pb", "pe", "ps"]
_COLS = ["date", "ticker", *VALUE_COLS]


def _key() -> str:
    for v in ("NDL_API_KEY", "NASDAQ_DATA_LINK_API_KEY", "QUANDL_API_KEY"):
        if os.environ.get(v):
            return os.environ[v]
    print("ERROR: no NDL_API_KEY", file=sys.stderr)
    sys.exit(2)


def _pages(table: str, q: dict):
    key, cur = _key(), None
    while True:
        qq = {**q, "api_key": key}
        if cur:
            qq["qopts.cursor_id"] = cur
        url = f"{BASE}/{table}.json?{urllib.parse.urlencode(qq)}"
        for attempt in range(5):  # retry transient socket timeouts / 5xx
            try:
                with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310 (fixed host)
                    p = json.loads(r.read().decode())
                break
            except Exception:  # noqa: BLE001
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        dt = p["datatable"]
        yield dt["data"], [c["name"] for c in dt["columns"]]
        cur = p.get("meta", {}).get("next_cursor_id")
        if not cur:
            break


def build_universe(since: str) -> list[str]:
    """US Domestic Common Stock active in [since, now], INCLUDING delisted names."""
    rows = []
    q = {"table": "SF1", "qopts.columns": "ticker,category,isdelisted,lastpricedate"}
    for data, _cols in _pages("TICKERS", q):
        rows += data
    uni = sorted({r[0] for r in rows if r[1] == "Domestic Common Stock" and (r[3] or "") >= since})
    Path(UNIVERSE_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(UNIVERSE_FILE).write_text("\n".join(uni))
    print(f"universe: {len(uni)} US common stocks (incl. delisted) -> {UNIVERSE_FILE}")
    return uni


def _to_month_end(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=_COLS)
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in VALUE_COLS:
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d = d.dropna(subset=["date"])
    me = d.set_index("date").groupby("ticker")[VALUE_COLS].resample("ME").last().reset_index()
    me["date"] = me["date"].dt.strftime("%Y-%m-%d")
    return me[_COLS]


def _append(new: pd.DataFrame) -> int:
    if new.empty:
        return 0
    p = Path(STORE)
    existing = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=_COLS)
    keys = set(zip(existing["ticker"], existing["date"], strict=False)) if len(existing) else set()
    added = len({(t, d) for t, d in zip(new["ticker"], new["date"], strict=False)} - keys)
    combined = new if existing.empty else pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(["ticker", "date"], keep="last").sort_values(
        ["ticker", "date"]
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(p, index=False)
    return added


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest Sharadar DAILY -> month-end store.")
    ap.add_argument("--since", default="2015-01-01")
    args = ap.parse_args()
    uni = build_universe(args.since)
    # Resume: skip tickers already in the store (a prior run may have timed out).
    done: set = set()
    if Path(STORE).exists():
        done = set(pd.read_parquet(STORE, columns=["ticker"])["ticker"].astype(str).unique())
    todo = [t for t in uni if t not in done]
    print(f"resume: {len(done)} tickers already stored, {len(todo)} to fetch")
    total = 0
    for i in range(0, len(todo), 200):
        chunk = todo[i : i + 200]
        frames = []
        for data, cols in _pages(
            "DAILY",
            {
                "ticker": ",".join(chunk),
                "date.gte": args.since,
                "qopts.columns": "ticker,date," + ",".join(VALUE_COLS),
            },
        ):
            if data:
                frames.append(pd.DataFrame(data, columns=cols))
        daily = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        added = _append(_to_month_end(daily))
        total += added
        print(f"  chunk {i // 200 + 1}/{(len(uni) - 1) // 200 + 1}: {len(daily)} daily -> +{added}")
    print(f"done: +{total} month-end rows -> {STORE}")


if __name__ == "__main__":
    main()
