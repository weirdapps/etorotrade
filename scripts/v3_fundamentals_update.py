"""Ingest Sharadar SF1 (Core US Fundamentals) into the PIT fundamentals store.

Point-in-time by construction: we keep Sharadar's ``datekey`` (the date each filing
became public) and store as-reported quarterly figures (``dimension=ARQ``), so
``fundamentals_store.read_asof(T)`` never sees a report filed after T.

Auth: set the Nasdaq Data Link API key in the environment (NOT the repo):
    export NDL_API_KEY=...            # or NASDAQ_DATA_LINK_API_KEY / QUANDL_API_KEY
(get it at https://data.nasdaq.com/account/profile ; needs a Sharadar SF1 subscription).

Usage:
    .venv/bin/python scripts/v3_fundamentals_update.py --from-universe [--since 2005-01-01]
    .venv/bin/python scripts/v3_fundamentals_update.py --tickers AAPL,MSFT,JPM

Scoped/incremental pulls use the paginated datatables API (implemented here). A
one-shot FULL-history load of the whole SF1 table is better done via the bulk
export endpoint (``qopts.export=true`` -> zip); wire that when we run the first
full backfill and have decided the ticker scope.
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

from trade_modules.v3.fundamentals_store import (  # noqa: E402
    NUM_FIELDS,
    STORE_PATH,
    append_records,
    store_coverage,
)

ETORO_CSV = "yahoofinance/output/etoro.csv"
BASE = "https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF1.json"
# Sharadar SF1 columns we keep (names already match NUM_FIELDS + keys).
_COLUMNS = ["ticker", "datekey", "reportperiod", *NUM_FIELDS]


def _api_key() -> str | None:
    for var in ("NDL_API_KEY", "NASDAQ_DATA_LINK_API_KEY", "QUANDL_API_KEY"):
        key = os.environ.get(var)
        if key:
            return key
    return None


def _universe_tickers() -> list[str]:
    try:
        df = pd.read_csv(ETORO_CSV, na_values=["--"])
        return sorted(df["TKR"].dropna().astype(str).unique())
    except Exception as exc:  # noqa: BLE001
        print(f"warn: could not read {ETORO_CSV}: {exc}", file=sys.stderr)
        return []


def _fetch_page(key: str, params: dict) -> tuple[list[list], list[str], str | None]:
    """One datatables page -> (rows, column_names, next_cursor)."""
    q = {**params, "api_key": key, "qopts.columns": ",".join(_COLUMNS), "dimension": "ARQ"}
    url = f"{BASE}?{urllib.parse.urlencode(q)}"
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (fixed https host)
        payload = json.loads(resp.read().decode())
    dt = payload.get("datatable", {})
    cols = [c["name"] for c in dt.get("columns", [])]
    nxt = payload.get("meta", {}).get("next_cursor_id")
    return dt.get("data", []), cols, nxt


def _fetch(key: str, tickers: list[str], since: str | None) -> pd.DataFrame:
    """Fetch all pages for a ticker batch (Sharadar caps tickers per request, so batch)."""
    frames: list[pd.DataFrame] = []
    for i in range(0, len(tickers), 100):  # batch tickers to stay within request limits
        batch = ",".join(tickers[i : i + 100])
        params: dict = {"ticker": batch}
        if since:
            params["datekey.gte"] = since
        cursor: str | None = None
        while True:
            page_params = dict(params)
            if cursor:
                page_params["qopts.cursor_id"] = cursor
            rows, cols, cursor = _fetch_page(key, page_params)
            if rows:
                frames.append(pd.DataFrame(rows, columns=cols))
            if not cursor:
                break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_COLUMNS)


def _catch_up_since(
    *, buffer_days: int = 90, backfill_since: str = "2015-01-01", store_path: str = STORE_PATH
) -> str:
    """Self-healing ``--since`` for the scheduled daily sync: the store's newest
    ``datekey`` minus ``buffer_days`` (re-pull recent filings idempotently — the buffer
    heals a missed run and picks up late/amended filings inside the window). Falls back
    to ``backfill_since`` when the store is empty.

    Note: this is ``datekey``-based, so a restatement of an OLDER period (same datekey,
    new value) is only re-fetched while it stays inside the buffer window, and a brand-new
    universe ticker gets only its recent history; a periodic full
    ``--from-universe --since <backfill>`` refresh remains the way to capture those.
    """
    last = store_coverage(store_path=store_path).get("last")
    if not last:
        return backfill_since
    return (pd.Timestamp(last) - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest Sharadar SF1 into the PIT fundamentals store.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tickers", help="comma-separated tickers")
    g.add_argument("--from-universe", action="store_true", help="use the etoro.csv universe")
    g.add_argument("--from-file", help="read tickers from a file (one per line)")
    ap.add_argument("--since", help="only filings with datekey >= this (YYYY-MM-DD)")
    ap.add_argument(
        "--catch-up",
        action="store_true",
        help="incremental self-healing sync: derive --since from the store's newest "
        "datekey minus a buffer (overrides --since). Use this for the scheduled job.",
    )
    args = ap.parse_args()

    key = _api_key()
    if not key:
        print(
            "ERROR: no Nasdaq Data Link API key. Set NDL_API_KEY (get it at "
            "https://data.nasdaq.com/account/profile, needs a Sharadar SF1 subscription).",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    elif args.from_file:
        tickers = [ln.strip() for ln in Path(args.from_file).read_text().splitlines() if ln.strip()]
    else:
        tickers = _universe_tickers()
    if not tickers:
        print("ERROR: no tickers to fetch.", file=sys.stderr)
        sys.exit(1)

    since = _catch_up_since() if args.catch_up else args.since
    if args.catch_up:
        print(f"catch-up: derived --since {since} from store coverage")
    print(f"fetching SF1 (ARQ) for {len(tickers)} tickers" + (f" since {since}" if since else ""))
    df = _fetch(key, tickers, since)
    print(f"fetched {len(df)} filing rows")
    added = append_records(df)
    cov = store_coverage()
    print(f"appended {added} new (ticker,datekey) rows -> store now: {cov}")


if __name__ == "__main__":
    main()
