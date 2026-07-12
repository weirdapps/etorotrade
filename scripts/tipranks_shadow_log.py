#!/usr/bin/env python3
"""Forward shadow-log of TipRanks signals for incremental-IC research.

WHY: A single same-day cross-section of TipRanks signals (smartScore, analyst
consensus, price-target upside, news sentiment, hedge-fund/insider scores) vs a
forward return is underpowered and, because the TipRanks MCP only exposes a TODAY
snapshot, cannot be tested against realized forward returns. To ever get a real
verdict on whether TipRanks/news add INCREMENTAL cross-sectional signal over the
v3 yfinance factor set, we must capture the snapshot daily and let forward returns
accrue. This script appends one JSONL row per (date, ticker) so incremental-IC can
be computed forward over time (join snapshot_t -> realized return_{t->t+h}).

OUTPUT: ~/.weirdapps-trading/tipranks_shadow/YYYY-MM-DD.jsonl  (one file per day)
Each line:
  {"date","ticker","smartScore","analystConsensus","bestAnalystConsensus",
   "priceTargetUpside","newsSentiment","hedgeFundsScore","insiderScore","price"}
`price` is the snapshot-day price, the anchor for later forward-return joins.

TWO INGEST PATHS (the writer is the same; only the source differs):
  1. --input snapshot.json  (PRIMARY, reliable): a Claude session runs the
     `mcp__tipranks__get_assets_data` tool (batched over the universe), dumps the
     result JSON, and pipes the file here. Accepts either {"assetsData":[...]} or a
     bare list of asset dicts.
  2. --fetch  (BEST-EFFORT, unofficial): hit the public TipRanks real-time
     endpoint over HTTP. May be blocked by TLS proxies (e.g. the NBG Mac) — it
     degrades gracefully (logs, skips) rather than crashing, mirroring the repo's
     inert-but-graceful pattern for gated data sources. Prefer the VPS for --fetch.

The universe defaults to the actionable names (portfolio.csv + buy.csv union).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path.home() / ".weirdapps-trading" / "tipranks_shadow"
DEFAULT_CSVS = [
    REPO / "yahoofinance" / "output" / "portfolio.csv",
    REPO / "yahoofinance" / "output" / "buy.csv",
]
US_RE = re.compile(r"^[A-Z]+$")

# TipRanks asset-dict key -> our JSONL field. Keeps the log stable if the API
# adds keys; unknown keys are ignored, missing keys become null.
FIELDS = {
    "ticker": "ticker",
    "smartScore": "smartScore",
    "analystConsensus": "analystConsensus",
    "bestAnalystConsensus": "bestAnalystConsensus",
    "priceTargetUpside": "priceTargetUpside",
    "newsSentiment": "newsSentiment",
    "hedgeFundsScore": "hedgeFundsScore",
    "insiderScore": "insiderScore",
    "price": "price",
}


def athens_today() -> str:
    return datetime.now(ZoneInfo("Europe/Athens")).strftime("%Y-%m-%d")


def load_universe(csv_paths: list[Path], us_only: bool = False) -> list[str]:
    """Union of the TKR column across the given CSVs, order-preserving, de-duped."""
    seen: dict[str, None] = {}
    for p in csv_paths:
        if not p.exists():
            print(f"[warn] universe CSV missing: {p}", file=sys.stderr)
            continue
        with p.open(newline="") as fh:
            for row in csv.DictReader(fh):
                t = (row.get("TKR") or "").strip()
                if not t or (us_only and not US_RE.match(t)):
                    continue
                seen.setdefault(t, None)
    return list(seen)


def parse_assets_data(obj) -> list[dict]:
    """Normalize an MCP get_assets_data payload to a list of asset dicts."""
    if isinstance(obj, str):
        obj = json.loads(obj)
    if isinstance(obj, dict):
        # MCP tool result may be {"result": "<json string>"} or {"assetsData": [...]}
        if "result" in obj and "assetsData" not in obj:
            return parse_assets_data(obj["result"])
        obj = obj.get("assetsData", obj)
    if not isinstance(obj, list):
        raise ValueError("expected a list of asset dicts or an {'assetsData': [...]} object")
    return obj


def to_record(asset: dict, date: str) -> dict | None:
    """Project a TipRanks asset dict onto the stable JSONL schema."""
    t = asset.get("ticker")
    if not t:
        return None
    rec = {"date": date}
    for src, dst in FIELDS.items():
        rec[dst] = asset.get(src)
    rec["ticker"] = t
    return rec


def fetch_tipranks_http(tickers: list[str], pause: float = 0.3) -> list[dict]:
    """Best-effort per-ticker fetch from the public TipRanks endpoint.

    Unofficial and undocumented; may be rate-limited or TLS-blocked. Returns
    whatever succeeds and skips the rest. Prefer the --input (MCP) path.
    """
    import urllib.request  # noqa: PLC0415

    out: list[dict] = []
    url = "https://www.tipranks.com/api/stocks/getData/?name={}"
    for t in tickers:
        try:
            req = urllib.request.Request(
                url.format(t), headers={"User-Agent": "Mozilla/5.0 (shadow-log)"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
                d = json.load(resp)
            out.append(
                {
                    "ticker": t,
                    "smartScore": (d.get("tipranksStockScore") or {}).get("score"),
                    "analystConsensus": (d.get("portfolioHoldingData") or {}).get(
                        "analystConsensus"
                    ),
                    "bestAnalystConsensus": (d.get("bestAnalystConsensus") or {}).get("consensus"),
                    "priceTargetUpside": d.get("priceTargetUpside"),
                    "newsSentiment": (d.get("newsSentiment") or {}).get("score"),
                    "hedgeFundsScore": (d.get("hedgeFundData") or {}).get("sentiment"),
                    "insiderScore": (d.get("insiderTrading") or {}).get("stockScore"),
                    "price": d.get("prices", [{}])[-1].get("p") if d.get("prices") else None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {t}: {type(exc).__name__} {exc}", file=sys.stderr)
        time.sleep(pause)
    return out


def append_snapshot(
    records: list[dict], date: str, out_dir: Path, force: bool = False
) -> tuple[int, int]:
    """Append records to <out_dir>/<date>.jsonl, skipping tickers already logged
    that day (unless force). Returns (written, skipped)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{date}.jsonl"
    existing: set[str] = set()
    if path.exists() and not force:
        with path.open() as fh:
            for line in fh:
                try:
                    existing.add(json.loads(line)["ticker"])
                except (json.JSONDecodeError, KeyError):
                    continue
    written = skipped = 0
    with path.open("a") as fh:
        for rec in records:
            if not force and rec["ticker"] in existing:
                skipped += 1
                continue
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            written += 1
    return written, skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--input", type=Path, help="JSON file from mcp get_assets_data (assetsData list or object)"
    )
    src.add_argument(
        "--fetch", action="store_true", help="best-effort HTTP fetch (unofficial; may be blocked)"
    )
    ap.add_argument(
        "--date", default=athens_today(), help="snapshot date YYYY-MM-DD (default: today, Athens)"
    )
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument(
        "--csv", type=Path, action="append", help="universe CSV(s); default portfolio.csv + buy.csv"
    )
    ap.add_argument(
        "--us-only", action="store_true", help="restrict fetch universe to ^[A-Z]+$ tickers"
    )
    ap.add_argument("--force", action="store_true", help="re-write tickers already logged today")
    ap.add_argument("--dry-run", action="store_true", help="parse + report counts, write nothing")
    args = ap.parse_args(argv)

    if args.input:
        assets = parse_assets_data(json.loads(args.input.read_text()))
    else:
        csvs = args.csv or DEFAULT_CSVS
        universe = load_universe(csvs, us_only=args.us_only)
        print(f"universe: {len(universe)} tickers", file=sys.stderr)
        assets = fetch_tipranks_http(universe)

    records = [r for a in assets if (r := to_record(a, args.date))]
    print(f"parsed {len(records)} records for {args.date}", file=sys.stderr)

    if args.dry_run:
        for r in records[:3]:
            print(json.dumps(r))
        print(f"[dry-run] would write {len(records)} records", file=sys.stderr)
        return 0

    written, skipped = append_snapshot(records, args.date, args.out_dir, force=args.force)
    print(
        f"wrote {written}, skipped {skipped} -> {args.out_dir / (args.date + '.jsonl')}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
