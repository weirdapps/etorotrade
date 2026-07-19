"""Ingest Sharadar SEP (survivorship-clean equity prices, incl. delisted) into a
month-end price store for the PIT fundamentals backtest.

SEP carries delisted tickers, which yfinance drops — the source of the survivorship
bias flagged in the first fundamentals backtest. We pull daily adjusted close
(``closeadj``, split+dividend adjusted), resample per ticker to month-end, and append
to a dedicated append-only store (``v3/price_store`` machinery) so a name that later
delisted still prices historically.

Auth: NDL_API_KEY (Sharadar SF1+SEP bundle). Usage:
    .venv/bin/python scripts/v3_sep_prices_update.py --from-universe --since 2015-01-01
    .venv/bin/python scripts/v3_sep_prices_update.py --tickers AAPL,MSFT --since 2015-01-01
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from trade_modules.v3.price_store import append_bars, store_coverage  # noqa: E402

SEP_STORE = str(Path("~/.weirdapps-trading/v3_sep_monthly.parquet").expanduser())
BASE = "https://data.nasdaq.com/api/v3/datatables/SHARADAR/SEP.json"
ETORO_CSV = "yahoofinance/output/etoro.csv"


def _api_key() -> str | None:
    for var in ("NDL_API_KEY", "NASDAQ_DATA_LINK_API_KEY", "QUANDL_API_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    return None


def _universe_tickers() -> list[str]:
    df = pd.read_csv(ETORO_CSV, na_values=["--"])
    return sorted(df["TKR"].dropna().astype(str).unique())


def _fetch_daily(key: str, tickers: list[str], since: str) -> pd.DataFrame:
    """Daily (ticker, date, closeadj) for a ticker batch, all pages."""
    frames: list[pd.DataFrame] = []
    for i in range(0, len(tickers), 100):
        batch = ",".join(tickers[i : i + 100])
        cursor: str | None = None
        while True:
            q = {
                "ticker": batch,
                "date.gte": since,
                "qopts.columns": "ticker,date,closeadj",
                "api_key": key,
            }
            if cursor:
                q["qopts.cursor_id"] = cursor
            url = f"{BASE}?{urllib.parse.urlencode(q)}"
            with urllib.request.urlopen(url, timeout=90) as resp:  # noqa: S310 (fixed https host)
                payload = json.loads(resp.read().decode())
            dt = payload.get("datatable", {})
            rows, cols = dt.get("data", []), [c["name"] for c in dt.get("columns", [])]
            if rows:
                frames.append(pd.DataFrame(rows, columns=cols))
            cursor = payload.get("meta", {}).get("next_cursor_id")
            if not cursor:
                break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _to_month_end(daily: pd.DataFrame) -> pd.DataFrame:
    """Long daily (ticker, date, closeadj) -> long month-end (date, ticker, close)."""
    if daily.empty:
        return pd.DataFrame(columns=["date", "ticker", "close"])
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["close"] = pd.to_numeric(d["closeadj"], errors="coerce")
    d = d.dropna(subset=["date", "close"])
    me = d.set_index("date").groupby("ticker")["close"].resample("ME").last().reset_index()
    me["date"] = me["date"].dt.strftime("%Y-%m-%d")
    return me[["date", "ticker", "close"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest Sharadar SEP -> month-end price store.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tickers")
    g.add_argument("--from-universe", action="store_true")
    ap.add_argument("--since", default="2015-01-01")
    args = ap.parse_args()

    key = _api_key()
    if not key:
        print("ERROR: no NDL_API_KEY (needs the Sharadar SEP bundle).", file=sys.stderr)
        sys.exit(2)
    tickers = (
        [t.strip() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else _universe_tickers()
    )
    print(f"SEP: fetching daily closeadj for {len(tickers)} tickers since {args.since}")
    total_added = 0
    # Process in ticker chunks so month-end resampling stays bounded in memory.
    for i in range(0, len(tickers), 200):
        chunk = tickers[i : i + 200]
        daily = _fetch_daily(key, chunk, args.since)
        me = _to_month_end(daily)
        added = append_bars(me, store_path=SEP_STORE) if not me.empty else 0
        total_added += added
        print(f"  chunk {i // 200 + 1}: {len(daily)} daily -> +{added} month-end rows")
    print(f"done: +{total_added} rows -> {store_coverage(SEP_STORE)}")


if __name__ == "__main__":
    main()
