"""
Portfolio reconciliation: compare live eToro holdings against signal-generated portfolio.csv.

Detects:
- NEW positions opened after the last signal run (present on eToro, absent from portfolio.csv)
- CLOSED positions no longer on eToro (present in portfolio.csv, absent from eToro)
- SYMBOL DRIFT where eToro uses a different symbol than Yahoo Finance normalization

Uses the eToro public API (same as etorotrade's download.py pipeline) with
instrument metadata resolution for symbol mapping.

Usage:
    from trade_modules.portfolio_reconciler import reconcile_portfolio
    result = reconcile_portfolio()
"""

import csv
import json
import os
import subprocess
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any

# eToro public API base (the private api.etoro.com endpoints return 404)
ETORO_PUBLIC_API = "https://www.etoro.com/api/public/v1"
ETORO_USERNAME = "plessas"

# Known eToro symbolFull → Yahoo Finance mappings for edge cases
# The etorotrade pipeline uses normalize_ticker() from ticker_utils.py,
# but we keep a lightweight map here to avoid importing the full module.
ETORO_TO_YAHOO = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "BNB": "BNB-USD",
    "XRP": "XRP-USD",
    "SOL": "SOL-USD",
    "ADA": "ADA-USD",
    "DOGE": "DOGE-USD",
    "DOT": "DOT-USD",
    "AVAX": "AVAX-USD",
    "LINK": "LINK-USD",
}


def _get_etoro_credentials() -> tuple[str, str]:
    """Get eToro API credentials from macOS Keychain."""
    try:
        public_key = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                "etoro-api",
                "-s",
                "etoro-public-key",
                "-w",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()

        user_key = subprocess.run(
            ["security", "find-generic-password", "-a", "etoro-api", "-s", "etoro-user-key", "-w"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()

        return public_key, user_key
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"Failed to get eToro credentials from Keychain: {e}")


def _curl_json(url: str, api_key: str, user_key: str) -> dict:
    """Fetch JSON from eToro public API using curl (avoids urllib/file:// concerns)."""
    request_id = str(uuid.uuid4())

    result = subprocess.run(
        [
            "curl",
            "-s",
            "--fail",
            "--max-time",
            "30",
            "-H",
            f"X-API-KEY: {api_key}",
            "-H",
            f"X-USER-KEY: {user_key}",
            "-H",
            f"X-REQUEST-ID: {request_id}",
            "-H",
            "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            url,
        ],
        capture_output=True,
        text=True,
        timeout=35,
    )

    if result.returncode != 0:
        raise RuntimeError(f"eToro API request failed (code {result.returncode}): {result.stderr}")

    return json.loads(result.stdout)


def _fetch_live_portfolio(api_key: str, user_key: str) -> list[dict[str, Any]]:
    """Fetch current open positions from eToro public API."""
    url = f"{ETORO_PUBLIC_API}/user-info/people/{ETORO_USERNAME}/portfolio/live"
    data = _curl_json(url, api_key, user_key)
    return data.get("positions", [])


def _fetch_instrument_metadata(
    instrument_ids: list[int], api_key: str, user_key: str
) -> dict[int, dict]:
    """Fetch instrument metadata (symbols, names) for a list of instrument IDs."""
    if not instrument_ids:
        return {}

    unique_ids = sorted(set(instrument_ids))
    ids_param = ",".join(str(i) for i in unique_ids)
    url = f"{ETORO_PUBLIC_API}/market-data/instruments?instrumentIds={ids_param}"

    data = _curl_json(url, api_key, user_key)

    metadata = {}
    for instrument in data.get("instrumentDisplayDatas", []):
        iid = instrument.get("instrumentID")
        if iid is not None:
            metadata[iid] = instrument

    return metadata


def _normalize_etoro_symbol(symbol: str) -> str:
    """Normalize an eToro symbol to match Yahoo Finance / portfolio.csv format."""
    if not symbol:
        return symbol

    s = symbol.upper().strip()

    if s in ETORO_TO_YAHOO:
        return ETORO_TO_YAHOO[s]

    # Try using the etorotrade normalize_ticker if available
    try:
        from yahoofinance.utils.data.ticker_utils import normalize_ticker

        return normalize_ticker(s)
    except ImportError:
        pass

    return s


def _read_portfolio_csv(path: str) -> dict[str, dict[str, Any]]:
    """Read portfolio.csv (output format) and return dict keyed by ticker."""
    result = {}
    if not os.path.exists(path):
        return result

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tkr = row.get("TKR", "").strip()
            if tkr:
                result[tkr] = row

    return result


def _aggregate_positions(
    positions: list[dict], metadata: dict[int, dict]
) -> dict[str, dict[str, Any]]:
    """Aggregate eToro positions by resolved symbol."""
    grouped: dict[str, list[tuple[dict, dict]]] = defaultdict(list)

    for pos in positions:
        instrument_id = pos.get("instrumentId")
        meta = metadata.get(instrument_id, {})
        raw_symbol = meta.get("symbolFull", f"ID:{instrument_id}")
        normalized = _normalize_etoro_symbol(raw_symbol)
        grouped[normalized].append((pos, meta))

    result = {}
    for symbol, pos_meta_list in grouped.items():
        total_invest_pct = sum(p.get("investmentPct", 0) for p, _ in pos_meta_list)
        total_profit = sum(p.get("netProfit", 0) for p, _ in pos_meta_list)

        # Weighted average open rate
        avg_open = 0.0
        if total_invest_pct > 0:
            avg_open = (
                sum(p.get("openRate", 0) * p.get("investmentPct", 0) for p, _ in pos_meta_list)
                / total_invest_pct
            )

        first_meta = pos_meta_list[0][1]
        result[symbol] = {
            "symbol": symbol,
            "raw_etoro_symbol": first_meta.get("symbolFull", ""),
            "name": first_meta.get("instrumentDisplayName", ""),
            "instrument_id": pos_meta_list[0][0].get("instrumentId"),
            "num_positions": len(pos_meta_list),
            "invest_pct": round(total_invest_pct * 100, 2),
            "avg_open_rate": round(avg_open, 4),
            "total_profit": round(total_profit, 2),
        }

    return result


def reconcile_portfolio(
    portfolio_csv_path: str | None = None,
) -> dict[str, Any]:
    """
    Reconcile live eToro holdings against portfolio.csv.

    Returns dict with keys: timestamp, portfolio_csv_date, live_count, csv_count,
    matched, new_positions, closed_positions, summary, has_drift.
    """
    if portfolio_csv_path is None:
        portfolio_csv_path = os.path.expanduser(
            "~/SourceCode/etorotrade/yahoofinance/output/portfolio.csv"
        )

    # Get CSV modification date
    csv_mtime = ""
    if os.path.exists(portfolio_csv_path):
        mtime = os.path.getmtime(portfolio_csv_path)
        csv_mtime = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

    # Read portfolio.csv tickers
    csv_holdings = _read_portfolio_csv(portfolio_csv_path)
    csv_tickers = set(csv_holdings.keys())

    # Fetch live eToro positions with metadata
    api_key, user_key = _get_etoro_credentials()
    raw_positions = _fetch_live_portfolio(api_key, user_key)

    instrument_ids = [p.get("instrumentId") for p in raw_positions if p.get("instrumentId")]
    metadata = _fetch_instrument_metadata(instrument_ids, api_key, user_key)

    live_holdings = _aggregate_positions(raw_positions, metadata)
    live_tickers = set(live_holdings.keys())

    # Detect differences
    new_tickers = live_tickers - csv_tickers
    closed_tickers = csv_tickers - live_tickers
    matched_tickers = live_tickers & csv_tickers

    new_positions = []
    for tkr in sorted(new_tickers):
        info = live_holdings[tkr]
        new_positions.append(
            {
                "symbol": tkr,
                "name": info["name"],
                "raw_etoro_symbol": info["raw_etoro_symbol"],
                "invest_pct": info["invest_pct"],
                "num_positions": info["num_positions"],
                "total_profit": info["total_profit"],
            }
        )

    closed_positions = []
    for tkr in sorted(closed_tickers):
        row = csv_holdings[tkr]
        closed_positions.append(
            {
                "symbol": tkr,
                "name": row.get("NAME", ""),
                "signal": row.get("BS", ""),
                "last_price": row.get("PRC", ""),
                "upside": row.get("UP%", ""),
            }
        )

    has_drift = len(new_positions) > 0 or len(closed_positions) > 0

    # Build summary
    parts = []
    if new_positions:
        syms = ", ".join(p["symbol"] for p in new_positions)
        parts.append(f"{len(new_positions)} NEW position(s) not in signals: {syms}")
    if closed_positions:
        syms = ", ".join(p["symbol"] for p in closed_positions)
        parts.append(f"{len(closed_positions)} CLOSED position(s) still in signals: {syms}")
    if not has_drift:
        parts.append("Portfolio is in sync with eToro — no drift detected")

    summary = ". ".join(parts)

    result = {
        "timestamp": datetime.now().isoformat(),
        "portfolio_csv_date": csv_mtime,
        "live_count": len(live_tickers),
        "csv_count": len(csv_tickers),
        "matched": len(matched_tickers),
        "new_positions": new_positions,
        "closed_positions": closed_positions,
        "summary": summary,
        "has_drift": has_drift,
    }

    # Save reconciliation report
    output_dir = os.path.expanduser("~/.weirdapps-trading/committee/reports")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "reconciliation.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result
