# scripts/v3_account_snapshot.py
"""Write the live eToro account snapshot the v3 report anchors on.

Hits the eToro public API (getPortfolio + real/pnl) and writes
``~/Downloads/v3_live_account.json`` with per-position P/L, so the overlay
report can show profit/loss + current value on the names you already hold.

Schema written::

    {
      "nav": <total_equity USD>,
      "as_of": "<YYYY-MM-DD>",
      "source": "v3_account_snapshot (eToro API)",
      "weights":   {"<ticker>": <fraction 0..1>, ...},
      "positions": {"<ticker>": {current_value, base_value, cost, pnl,
                                 pnl_pct, weight_pct}, ...},
      "account":   {invested_cost, unrealized_pnl, profit_pct, total_equity,
                    available}
    }

Credentials: macOS Keychain (services ``etoro-public-key`` / ``etoro-user-key``)
or env ``ETORO_PUBLIC_KEY`` / ``ETORO_USER_KEY`` (used on the VPS). Self-contained
(no cross-repo import); mirrors the proven order-execution header recipe.

Run:  .venv/bin/python scripts/v3_account_snapshot.py [-o <path>]
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

_PORTFOLIO = "https://www.etoro.com/api/public/v1/trading/info/portfolio"
_PNL = "https://www.etoro.com/api/public/v1/trading/info/real/pnl"
_TRADEINFO = "https://www.etoro.com/api/public/v1/user-info/people/{u}/tradeinfo?period={p}"
_USERNAME = os.environ.get("ETORO_USERNAME", "plessas")
_INPUT_CSV = Path(os.path.expanduser("~/SourceCode/etorotrade/yahoofinance/input/portfolio.csv"))
_DEFAULT_OUT = "~/Downloads/v3_live_account.json"


def _key(service: str) -> str:
    """Read an eToro key from env first, then the macOS Keychain."""
    env = "ETORO_PUBLIC_KEY" if "public" in service else "ETORO_USER_KEY"
    if os.environ.get(env):
        return os.environ[env]
    if sys.platform == "darwin":
        try:
            r = subprocess.run(
                ["security", "find-generic-password", "-a", "etoro-api", "-s", service, "-w"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return r.stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return ""
    return ""


def _api_get(url: str) -> dict:
    """GET a JSON eToro endpoint with the Cloudflare-safe header set."""
    pub, usr = _key("etoro-public-key"), _key("etoro-user-key")
    if not pub or not usr:
        raise RuntimeError(
            "eToro API keys missing (set ETORO_PUBLIC_KEY / ETORO_USER_KEY or Keychain)"
        )
    r = subprocess.run(
        [
            "curl",
            "-s",
            url,
            "-H",
            f"x-api-key: {pub}",
            "-H",
            f"x-user-key: {usr}",
            "-H",
            f"x-request-id: {uuid.uuid4()}",
            "-H",
            "User-Agent: Mozilla/5.0",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"eToro API call failed ({url}): exit {r.returncode}")
    return json.loads(r.stdout)


def _instrument_map() -> dict[int, str]:
    """instrumentId -> ticker, from input/portfolio.csv."""
    if not _INPUT_CSV.exists():
        return {}
    out: dict[int, str] = {}
    with open(_INPUT_CSV) as f:
        for row in csv.DictReader(f):
            iid, sym = row.get("instrumentId"), (row.get("symbol") or "").strip()
            if iid and sym:
                try:
                    out[int(iid)] = sym
                except ValueError:
                    pass
    return out


_SEARCH = "https://www.etoro.com/api/public/v1/market-data/search"
_CRYPTO = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "XRP": "XRP-USD",
    "SOL": "SOL-USD",
    "ADA": "ADA-USD",
    "DOGE": "DOGE-USD",
}


def _normalize_ticker(sym: str) -> str:
    """Match the signal-pipeline ticker conventions (crypto suffix, HK zero-pad)."""
    if sym in _CRYPTO:
        return _CRYPTO[sym]
    if sym.endswith(".HK"):
        pfx = sym.split(".")[0].lstrip("0") or "0"
        return f"{pfx.zfill(4)}.HK"
    return sym


def _resolve_instrument(iid: int) -> str:
    """Resolve an instrumentId not in portfolio.csv to a ticker via the search API."""
    try:
        data = _api_get(f"{_SEARCH}?instrumentId={iid}")
    except Exception:  # noqa: BLE001 (best-effort fallback; skip on any failure)
        return ""
    for item in data.get("items", []):
        sym = (item.get("internalSymbolFull") or "").strip()
        if sym and not sym.endswith(".24-7"):
            return _normalize_ticker(sym)
    return ""


def build_snapshot() -> dict:
    """Fetch the account + per-position P/L and shape it into the v3 schema."""
    cp = fetch_portfolio()
    credit = float(cp.get("credit", 0))
    orders = cp.get("orders", [])
    pending = sum(float(o.get("amount", 0)) for o in orders)
    available = credit - pending

    imap = _instrument_map()
    pnl_map = fetch_pnl()

    pos: dict[str, dict] = {}
    total_pnl = total_exposure = 0.0
    for p in cp.get("positions", []):
        iid = int(p.get("instrumentID", 0))
        ticker = imap.get(iid) or _resolve_instrument(iid)
        if not ticker:
            continue  # unmapped + unresolvable instrument: skip
        pid = int(p.get("positionID", 0))
        cost = float(p.get("initialAmountInDollars", 0))
        base = float(p.get("unitsBaseValueDollars", cost))
        rp = pnl_map.get(pid, {})
        pnl = float(rp.get("pnl", 0.0))
        exposure = float(rp.get("exposure", base))
        total_pnl += pnl
        total_exposure += exposure
        d = pos.setdefault(
            ticker, {"cost": 0.0, "base_value": 0.0, "pnl": 0.0, "current_value": 0.0}
        )
        d["cost"] += cost
        d["base_value"] += base
        d["pnl"] += pnl
        d["current_value"] += exposure

    total_equity = available + total_exposure + pending
    invested = sum(d["base_value"] for d in pos.values()) + pending
    for d in pos.values():
        d["pnl_pct"] = round(d["pnl"] / d["base_value"] * 100, 2) if d["base_value"] > 0 else 0.0
        d["weight_pct"] = round(d["current_value"] / total_equity * 100, 2) if total_equity else 0.0
        for k in ("cost", "base_value", "pnl", "current_value"):
            d[k] = round(d[k], 2)

    weights = {
        t: round(d["current_value"] / total_equity, 6) for t, d in pos.items() if total_equity
    }
    raw_positions = cp.get("positions", [])
    social = fetch_social()
    social.update(
        {
            "unique_assets": len(pos),
            "open_positions": len(raw_positions),  # incl. lots
            "shorts": sum(1 for p in raw_positions if not p.get("isBuy", True)),
            "cash_pct": round(available / total_equity * 100, 1) if total_equity else 0.0,
        }
    )
    return {
        "nav": round(total_equity, 2),
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "source": "v3_account_snapshot (eToro API)",
        "weights": weights,
        "positions": pos,
        "account": {
            "invested_cost": round(invested, 2),
            "unrealized_pnl": round(total_pnl, 2),
            "profit_pct": round(total_pnl / invested * 100, 2) if invested > 0 else 0.0,
            "total_equity": round(total_equity, 2),
            "available": round(available, 2),
        },
        "social": social,
    }


def fetch_portfolio() -> dict:
    raw = _api_get(_PORTFOLIO)
    return raw.get("clientPortfolio", raw)


def fetch_pnl() -> dict[int, dict]:
    data = _api_get(_PNL)
    out: dict[int, dict] = {}
    for p in data.get("clientPortfolio", {}).get("positions", []):
        pid = int(p.get("positionID", 0))
        upnl = p.get("unrealizedPnL", {})
        if pid and upnl:
            out[pid] = {
                "pnl": float(upnl.get("pnL", 0)),
                "exposure": float(upnl.get("exposureInAccountCurrency", 0)),
            }
    return out


def fetch_social() -> dict:
    """PI performance + social metrics from the eToro tradeinfo endpoint.

    Two period calls: CurrYear (YTD gain, today, this-week, risk score, copiers,
    win ratio, trades) and CurrMonth (MTD gain). Gains and winRatio are already
    percentages (3.4 = 3.4%). Best-effort: returns {} if the endpoint is
    unavailable so the report degrades gracefully.
    """
    try:
        yr = _api_get(_TRADEINFO.format(u=_USERNAME, p="CurrYear"))
        mo = _api_get(_TRADEINFO.format(u=_USERNAME, p="CurrMonth"))
    except Exception:  # noqa: BLE001 (best-effort; report renders without social block)
        return {}
    return {
        "username": _USERNAME,
        "copiers": yr.get("copiers"),
        "baseline_copiers": yr.get("baseLineCopiers"),
        "copiers_gain_pct": yr.get("copiersGain"),
        "risk_score": yr.get("riskScore"),
        "max_daily_risk": yr.get("maxDailyRiskScore"),
        "win_ratio": yr.get("winRatio"),
        "trades_ytd": yr.get("trades"),
        "gain_ytd": yr.get("gain"),
        "gain_mtd": mo.get("gain"),
        "daily_gain": yr.get("dailyGain"),
        "week_gain": yr.get("thisWeekGain"),
        "aum_tier_desc": yr.get("aumTierDesc"),
    }


def main() -> int:
    out = os.path.expanduser(
        sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else _DEFAULT_OUT
    )
    snap = build_snapshot()
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=2)
    acc = snap["account"]
    print(
        f"snapshot -> {out}  |  equity ${acc['total_equity']:,.0f}  "
        f"P/L ${acc['unrealized_pnl']:+,.0f} ({acc['profit_pct']:+.2f}%)  "
        f"{len(snap['positions'])} positions"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
