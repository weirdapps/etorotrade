# scripts/v3_overlay_report.py
"""Trading Model v3 — OVERLAY report driver.

Runs the v3 pipeline in OVERLAY mode: instead of rebuilding the book from zero
(``v3_full_report.py`` / ``build_portfolio``), it anchors on the LIVE book and
applies a minimal, conviction-driven overlay via
:func:`trade_modules.v3.overlay.build_overlay` — sell only genuinely weak
holdings, keep the rest at their current weight, add only the strongest new
names, then risk-gate the combined book. Conviction uses the BALANCED cluster
weights ``{value:0.15, quality:0.25, momentum:0.25, growth:0.15, lowvol:0.12,
strength:0.08}`` via ``compute_scores(cluster_weights=...)``.

Writes ``~/Downloads/<UTCstamp>_v3_overlay_report.html`` (UTC ``%Y%m%d%H%M``) and
also emits a network-free synthetic preview to
``~/Downloads/v3_overlay_preview.html`` (structurally self-checked) so the layout
can be screenshotted without the live pipeline.

Live account: a JSON at ``$V3_ACCOUNT_JSON`` or ``~/Downloads/v3_live_account.json``
(schema ``{"nav": <float|null>, "weights": {"<ticker>": <fraction>, ...}}``)
supplies the REAL current book + NAV that the overlay anchors on. Absent -> the
current book is approximated as an equal split across portfolio.csv holdings.

Run (VPS / network allowed):   .venv/bin/python scripts/v3_overlay_report.py
Preview only (no network):     .venv/bin/python scripts/v3_overlay_report.py --preview

No module-level ``yahoofinance.core.config`` import.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.v3_full_report import (  # noqa: E402  (pure helpers, reused)
    _read_tickers,
    _synthetic_scores,
    _system_read,
    load_account_block,
    load_account_json,
    load_account_positions,
    load_account_social,
    resolve_current_weights,
)
from scripts.v3_portfolio import trend_regime  # noqa: E402  (pure, unit-tested)
from trade_modules.riskfirst.fx import USD_BLOC, currency_of  # noqa: E402
from trade_modules.v3.actions import build_actions  # noqa: E402
from trade_modules.v3.combine import compute_scores  # noqa: E402
from trade_modules.v3.conditioning import resolve_deployment  # noqa: E402
from trade_modules.v3.features import enrich_features  # noqa: E402
from trade_modules.v3.fetch import robust_fetch_prices  # noqa: E402
from trade_modules.v3.overlay import build_overlay  # noqa: E402
from trade_modules.v3.report import compute_regime, render_report  # noqa: E402

PORTFOLIO_CSV = "yahoofinance/output/portfolio.csv"
BUY_CSV = "yahoofinance/output/buy.csv"
ETORO_CSV = "yahoofinance/output/etoro.csv"

PREVIEW_OUT = "~/Downloads/v3_overlay_preview.html"

# Balanced factor tilt requested by the owner for the overlay.
BALANCED_WEIGHTS: dict[str, float] = {
    "value": 0.15,
    "quality": 0.25,
    "momentum": 0.25,
    "growth": 0.15,
    "lowvol": 0.12,
    "strength": 0.08,
}

# Mega-cap "core" whose kept-vs-sold status is always reported.
MEGA_CORE: list[str] = ["NVDA", "GOOG", "MSFT", "AAPL", "AMZN", "AVGO", "TSM", "META"]

# Overridable via env so the same runner produces any confirmed config.
_VOL_CEILING = float(os.environ.get("V3_VOL_CEILING", "0.18"))
_NAME_CAP = float(os.environ.get("V3_NAME_CAP", "0.10"))
_SECTOR_CAP = float(os.environ.get("V3_SECTOR_CAP", "0.35"))
_USD_BLOC_CAP = float(os.environ.get("V3_USD_BLOC_CAP", "0.60"))
_CAP_MODE: str | None = os.environ.get("V3_CAP_MODE") or None


# Owner rule (2026-07-13): sell negative-conviction NON-core names, protect the AI
# core from factor sells, and hold OUT the sleeves the owner manages himself
# (gold / vol / Greece) so the model neither sells nor re-deploys their capital.
def _envflag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "", "false", "no")


_SELL_NEG = _envflag("V3_SELL_NEGATIVE_NONCORE")
_NONCORE_SELL_FLOOR = float(os.environ.get("V3_NONCORE_SELL_FLOOR", "0.0"))  # sell deadband
_PROTECT_CORE = _envflag("V3_PROTECT_CORE")
_FLOOR_CORE = _envflag("V3_FLOOR_CORE", "0")  # floor AI core at current (thesis overlay)
_MANAGED_SLEEVES = [
    s.strip()
    for s in os.environ.get("V3_MANAGED_SLEEVES", "GLD,UVXY,LYXGRE.DE").split(",")
    if s.strip()
]

# Known ETF tickers for asset-type classification.
_ETFS: set[str] = {"LYXGRE.DE", "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "EEM", "EFA", "TLT"}


def _compute_allocations(positions: dict, scores: pd.DataFrame) -> dict:
    """Compute portfolio allocation fractions for Geography, Asset Type, and Sector.

    Returns {"geography": {label: frac}, "asset_type": {label: frac}, "sector": {label: frac}}
    where each inner dict is sorted descending by fraction, with buckets below 3% merged into
    an "Other" entry appended last. Returns empty inner dicts when positions is empty.
    """

    def _region_for(ticker: str) -> str:
        t = str(ticker).upper()
        if "-" in t:
            return "Crypto/Global"
        if "." not in t:
            return "North America"
        sfx = "." + t.rsplit(".", 1)[-1]
        if sfx in {".US"}:
            return "North America"
        if sfx in {".TO", ".V"}:
            return "North America"
        if sfx in {
            ".DE",
            ".PA",
            ".AS",
            ".MI",
            ".MC",
            ".L",
            ".SW",
            ".BR",
            ".LS",
            ".HE",
            ".ST",
            ".CO",
            ".OL",
            ".VI",
            ".IR",
            ".WA",
            ".AT",
        }:
            return "Europe"
        if sfx in {".HK", ".T", ".KS", ".KQ", ".TW", ".SI", ".AX", ".NS", ".BO", ".SS", ".SZ"}:
            return "Asia-Pacific"
        if sfx in {".SA", ".MX", ".BA", ".SN"}:
            return "Latin America"
        return "Other"

    if not positions:
        return {"geography": {}, "asset_type": {}, "sector": {}}

    total = sum(float(p.get("current_value", 0.0)) for p in positions.values())
    if total <= 0:
        return {"geography": {}, "asset_type": {}, "sector": {}}

    geo: dict[str, float] = {}
    atype: dict[str, float] = {}
    sec: dict[str, float] = {}

    for ticker, pos in positions.items():
        cv = float(pos.get("current_value", 0.0))
        if cv <= 0:
            continue
        t = str(ticker).upper()

        region = _region_for(ticker)
        geo[region] = geo.get(region, 0.0) + cv

        # Asset type: crypto > commodity > volatility > ETF > equity
        if "-" in t:
            at = "Crypto"
        elif t == "GLD":
            at = "Commodity"
        elif t in {"UVXY", "VXX"}:
            at = "Volatility"
        elif ticker in _ETFS:
            at = "ETF"
        else:
            at = "Equity"
        atype[at] = atype.get(at, 0.0) + cv

        # Sector from scores frame; fall back to "Other" for missing/NaN.
        sector_label = "Other"
        if scores is not None and hasattr(scores, "index") and ticker in scores.index:
            try:
                s = scores.loc[ticker, "sector"]
                if s and not pd.isna(s) and str(s).strip():
                    sector_label = str(s).strip()
            except Exception:  # noqa: BLE001
                pass
        sec[sector_label] = sec.get(sector_label, 0.0) + cv

    def _bucket(raw: dict) -> dict:
        """Convert raw dollar totals to fractions; merge <3% into Other; sort desc."""
        frac = {k: v / total for k, v in raw.items() if v > 0}
        other = frac.pop("Other", 0.0)
        small = [k for k, v in frac.items() if v < 0.03]
        for k in small:
            other += frac.pop(k)
        out = dict(sorted(frac.items(), key=lambda kv: -kv[1]))
        if other > 0:
            out["Other"] = other
        return out

    return {
        "geography": _bucket(geo),
        "asset_type": _bucket(atype),
        "sector": _bucket(sec),
    }


# ---------------------------------------------------------------------------
# Render-view adapter (pure) — overlay result -> exec-panel-compatible dict
# ---------------------------------------------------------------------------


def overlay_portfolio_view(overlay: dict, scored: pd.DataFrame) -> dict:
    """Adapt a ``build_overlay`` result into the dict shape ``render_report`` reads.

    ``build_overlay`` returns ``{weights, diagnostics}``; the report's exec panel
    expects a ``build_portfolio``-style dict (gross / cash / usd_bloc /
    sector_exposures / diagnostics with a ``gate`` block). This fills those from
    the gated book + gate diagnostics so the same template renders the overlay.
    """
    weights = overlay["weights"]
    gate = overlay["diagnostics"].get("gate") or {}
    gross = float(weights.sum()) if len(weights) else 0.0
    usd_bloc = float(sum(float(w) for t, w in weights.items() if currency_of(t) in USD_BLOC))

    sector_exp: dict[str, float] = {}
    if scored is not None and "sector" in scored.columns and len(weights):
        labs = (
            scored.reindex(list(weights.index))["sector"]
            .astype("string")
            .str.upper()
            .fillna("UNKNOWN")
        )
        for t, lab in zip(weights.index, labs.tolist(), strict=True):
            w = float(weights[t])
            if w > 1e-12:
                key = lab if lab else "UNKNOWN"
                sector_exp[key] = sector_exp.get(key, 0.0) + w

    cvar_dep = gate.get("cvar_after")
    cvar_rb = float(cvar_dep) / gross if (cvar_dep is not None and gross > 0) else None
    return {
        "weights": weights,
        "gross": gross,
        "cash": max(0.0, 1.0 - gross),
        "usd_bloc": usd_bloc,
        "sector_exposures": sector_exp,
        "selected": [t for t in weights.index if float(weights[t]) > 1e-9],
        "diagnostics": {
            "gate": gate,
            "cvar_95_deployed": cvar_dep,
            "cvar_95_risk_book": cvar_rb,
            "binding": {},
        },
    }


def _print_console_summary(diag: dict, actions: list, gross_target: float) -> None:
    """One-block console summary: n_sell/buy/keep, turnover, mega-core, vol, deploy."""
    n_hold = sum(1 for a in actions if a.get("action") == "HOLD")
    gate = diag.get("gate") or {}
    vol_after = gate.get("vol_after")
    print(
        f"overlay: {diag['n_sell']} sell / {diag['n_buy']} buy / "
        f"{diag['n_keep']} keep ({n_hold} HOLD actions)"
    )
    print(f"turnover: {diag['turnover']:.1%}  freed: {diag['freed_weight']:.1%}")
    print(
        f"deployment: {gross_target:.0%}  vol: {vol_after:.1%}"
        if vol_after is not None
        else f"deployment: {gross_target:.0%}"
    )
    print(
        f"thresholds: sell<= {diag['sell_threshold']:.3f}  buy>= {diag['buy_threshold']:.3f}"
        if diag["sell_threshold"] == diag["sell_threshold"]  # not NaN
        else "thresholds: n/a (no eligible universe)"
    )
    cfa = diag.get("core_floor_applied") or {}
    if cfa:
        pre = gate.get("vol_after_prefloor")
        raised = sum(cfa.values())
        vtxt = (
            f"  (vol {pre:.1%} -> {vol_after:.1%})"
            if pre is not None and vol_after is not None
            else ""
        )
        print(f"AI-core floor: +{raised:.1%} across {len(cfa)} names{vtxt}")
    cr = diag.get("core_retention")
    if cr:
        kept = [t for t, s in cr["per_name"].items() if s == "kept"]
        sold = [t for t, s in cr["per_name"].items() if s == "sold"]
        nh = [t for t, s in cr["per_name"].items() if s == "not_held"]
        print(f"mega-cap core: {cr['n_kept']} kept / {cr['n_sold']} sold")
        print(f"  kept:     {', '.join(kept) or '-'}")
        print(f"  sold:     {', '.join(sold) or '-'}")
        print(f"  not held: {', '.join(nh) or '-'}")


# ---------------------------------------------------------------------------
# Synthetic preview (network-free) + structural self-check
# ---------------------------------------------------------------------------


def build_overlay_preview_html() -> str:
    """Render the FULL overlay report from synthetic data (no network) for QA.

    Crafts a synthetic current book so BOTH overlay signatures appear: strong
    holdings are KEPT (some HOLD, some TRIM/ADD after the gate) and weak holdings
    are SOLD, while the strongest non-held names are BOUGHT.
    """
    scores = _synthetic_scores()
    conv = (
        pd.to_numeric(scores["conviction"], errors="coerce").dropna().sort_values(ascending=False)
    )
    # Leave the very strongest names NON-held so they become BUYs; hold a band of
    # solidly-above-median names (KEEP) and a band of the weakest names (SELL).
    keeps = list(conv.index[5:10])
    weak = list(conv.index[-3:])

    cur: dict[str, float] = dict.fromkeys(keeps, 0.12)
    cur.update(dict.fromkeys(weak, 0.1))
    current = pd.Series(cur, dtype=float)  # gross ~0.90

    overlay = build_overlay(
        scores,
        current,
        pd.DataFrame(),
        gross_target=0.90,
        max_new=6,
        core_list=keeps[:2],
    )
    view = overlay_portfolio_view(overlay, scores)
    actions = build_actions(overlay["weights"], current, scores, nav=250_000.0)
    _, cond = resolve_deployment("neutral")

    now = datetime.now(timezone.utc)
    meta = {
        "date": now.strftime("%Y-%m-%d"),
        "n_portfolio": int(scores["is_portfolio"].sum()),
        "n_candidates": int((~scores["is_portfolio"]).sum()),
        "regime": "NEUTRAL",
        "regime_detail": "synthetic offline overlay preview (no live fetch)",
        "system_read": _system_read(scores, "NEUTRAL"),
        "priced": int(scores["mom_12_1"].notna().sum()),
        "enriched": int(scores["pb"].notna().sum()),
        "generated_utc": now.strftime("%Y-%m-%d %H:%M UTC"),
        "current_weights_approx": False,
        "account": {
            "total_equity": 250_000.0,
            "unrealized_pnl": 12_500.0,
            "profit_pct": 5.0,
            "available": 25_000.0,
            "invested_cost": 212_500.0,
        },
        "social": {
            "copiers": 1375,
            "baseline_copiers": 1390,
            "copiers_gain_pct": -1.08,
            "risk_score": 3,
            "max_daily_risk": 4,
            "win_ratio": 54.57,
            "trades_ytd": 361,
            "gain_ytd": 3.4,
            "gain_mtd": 2.08,
            "daily_gain": -0.86,
            "week_gain": -0.87,
            "aum_tier_desc": "$1M-$2M",
            "unique_assets": 48,
            "open_positions": 175,
            "shorts": 0,
            "cash_pct": 11.0,
        },
        "allocations": {
            "geography": {
                "North America": 0.62,
                "Europe": 0.24,
                "Asia-Pacific": 0.09,
                "Other": 0.05,
            },
            "asset_type": {
                "Equity": 0.82,
                "ETF": 0.10,
                "Commodity": 0.05,
                "Volatility": 0.03,
            },
            "sector": {
                "Technology": 0.34,
                "Financials": 0.16,
                "Health Care": 0.14,
                "Industrials": 0.10,
                "Consumer": 0.09,
                "Energy": 0.07,
                "Other": 0.10,
            },
        },
    }
    return render_report(scores, meta, portfolio=view, actions=actions, conditioning=cond)


def _self_check(html: str) -> None:
    """Assert the overlay report has the exec panel, the SELL + BUY groups, factor
    cards, no literal 'None', and zero em-dashes (U+2014)."""
    problems: list[str] = []
    if '<div class="exec-panel">' not in html:
        problems.append("exec/risk panel missing")
    for cls in ("buy", "sell"):  # overlay's signature action groups
        if f"act-grp act-grp--{cls}" not in html:
            problems.append(f"action group {cls} missing")
    if '<article class="card card--action"' not in html:
        problems.append("action cards missing")
    if "None" in html:
        problems.append("literal 'None' present")
    if "—" in html:
        problems.append("em-dash (U+2014) present")
    if problems:
        raise AssertionError("overlay preview self-check failed: " + "; ".join(problems))


def write_preview(out: str = PREVIEW_OUT) -> str:
    """Build + self-check the synthetic overlay preview and write it to disk."""
    html = build_overlay_preview_html()
    _self_check(html)
    path = os.path.expanduser(out)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"synthetic overlay preview -> {path}  (self-check passed)")
    return path


# ---------------------------------------------------------------------------
# Live pipeline (smoke-tested by the VPS run, not unit tests)
# ---------------------------------------------------------------------------


def main() -> None:
    # --- Universe assembly (mirrors v3_full_report.py) ---
    port = _read_tickers(PORTFOLIO_CSV)
    buy = _read_tickers(BUY_CSV)
    port_set = set(port)
    universe = list(dict.fromkeys(port + buy))
    print(
        f"universe: {len(universe)} tickers ({len(port_set)} portfolio + "
        f"{len(set(buy) - port_set)} candidates)"
    )

    # --- Feature enrichment + scoring with BALANCED cluster weights ---
    feats = enrich_features(
        universe, ETORO_CSV, price_period="2y", accruals_fetch=lambda _tickers: {}
    )
    priced = int(feats["mom_12_1"].notna().sum())
    enriched = int(feats["pb"].notna().sum())
    scores = compute_scores(feats, sector_neutral=True, cluster_weights=BALANCED_WEIGHTS)
    scores["is_portfolio"] = scores.index.isin(port_set)

    elig = scores.get("eligible", pd.Series(True, index=scores.index)).fillna(False).astype(bool)
    n_port = int((scores["is_portfolio"] & elig).sum())
    n_cand = int((~scores["is_portfolio"] & elig).sum())

    # --- Prices + market regime ---
    prices: pd.DataFrame = pd.DataFrame()
    spx_close: pd.Series = pd.Series(dtype=float)
    try:
        prices = robust_fetch_prices(universe, period="2y")
    except Exception as exc:  # noqa: BLE001
        print(f"warn: universe price fetch failed ({exc})", file=sys.stderr)
    try:
        spx_raw = robust_fetch_prices(["^GSPC"], period="2y")
        if spx_raw is not None and not spx_raw.empty:
            spx_close = spx_raw.iloc[:, 0]
    except Exception as exc:  # noqa: BLE001
        print(f"warn: ^GSPC fetch failed ({exc}); defaulting to neutral", file=sys.stderr)

    regime, _mult = trend_regime(spx_close)
    regime_label, regime_detail = compute_regime(spx_close)
    gross_target, cond = resolve_deployment(regime, polymarket_signal=None)
    print(f"regime: {regime}  deployment: {gross_target:.0%}")

    # --- Current book (live account anchors the overlay; else equal-split) ---
    account_weights, nav, present = load_account_json()
    current_weights, approx = resolve_current_weights(port, account_weights, present)
    if present:
        print(f"current book: live account ({len(current_weights)} names, nav={nav})")
    else:
        print(f"current book: equal-split fallback ({len(current_weights)} names)")

    # --- Hold OUT owner-managed sleeves (gold / vol / Greece): the model neither
    #     sells nor re-deploys their capital; deployment shrinks by their weight. ---
    managed_upper = {s.upper() for s in _MANAGED_SLEEVES}
    held_managed = [t for t in current_weights.index if str(t).upper() in managed_upper]
    if held_managed:
        managed_weight = float(sum(float(current_weights[t]) for t in held_managed))
        current_weights = current_weights.drop(index=held_managed)
        gross_target = max(0.0, gross_target - managed_weight)
        print(
            f"managed sleeves held out: {', '.join(held_managed)} "
            f"({managed_weight:.1%}); model deployment -> {gross_target:.0%}"
        )

    # --- Owner thesis overlay: floor the AI core at its current weight (capped by the
    #     single-name limit) so the conviction gate can't trim the sleeve below where
    #     the owner holds it. Off by default; V3_FLOOR_CORE=1 for the owner config. ---
    core_floor = None
    if _FLOOR_CORE:
        cf = {
            t: min(float(current_weights[t]), _NAME_CAP)
            for t in MEGA_CORE
            if t in current_weights.index
        }
        core_floor = pd.Series(cf, dtype=float) if cf else None
        if core_floor is not None:
            print(
                f"AI-core floor ON: {len(core_floor)} names floored at current "
                f"(<= {_NAME_CAP:.0%}/name), sleeve target {core_floor.sum():.1%}"
            )

    # --- Overlay construction (keep book, sell weak, add strongest) ---
    overlay = build_overlay(
        scores,
        current_weights,
        prices,
        gross_target=gross_target,
        name_cap=_NAME_CAP,
        sector_cap=_SECTOR_CAP,
        usd_bloc_cap=_USD_BLOC_CAP,
        vol_ceiling=_VOL_CEILING,
        core_list=MEGA_CORE,
        cap_mode=_CAP_MODE,
        sell_negative_noncore=_SELL_NEG,
        noncore_sell_floor=_NONCORE_SELL_FLOOR,
        protect_core=_PROTECT_CORE,
        core_floor=core_floor,
    )
    view = overlay_portfolio_view(overlay, scores)
    actions = build_actions(overlay["weights"], current_weights, scores, nav=nav)
    # Attach live P/L from the eToro account snapshot to held names (P/L $ / % / value).
    _positions = load_account_positions()
    account_block = load_account_block()
    social_block = load_account_social()
    for _a in actions:
        _p = _positions.get(_a.get("ticker"))
        if _p:
            _a["pnl"] = _p.get("pnl")
            _a["pnl_pct"] = _p.get("pnl_pct")
            _a["current_value"] = _p.get("current_value")

    # --- Render the FULL report ---
    now = datetime.now(timezone.utc)
    meta = {
        "date": now.strftime("%Y-%m-%d"),
        "n_portfolio": n_port,
        "n_candidates": n_cand,
        "regime": regime_label,
        "regime_detail": regime_detail,
        "system_read": _system_read(scores, regime_label),
        "priced": priced,
        "enriched": enriched,
        "generated_utc": now.strftime("%Y-%m-%d %H:%M UTC"),
        "current_weights_approx": approx,
        "account": account_block,
        "social": social_block,
        "allocations": _compute_allocations(_positions, scores),
    }
    html = render_report(scores, meta, portfolio=view, actions=actions, conditioning=cond)

    stamp = now.strftime("%Y%m%d%H%M")
    out = os.path.expanduser(f"~/Downloads/{stamp}_v3_overlay_report.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)

    _print_console_summary(overlay["diagnostics"], actions, gross_target)
    print(f"overlay report -> {out}")

    # Always emit the offline synthetic preview alongside the live report.
    write_preview()


if __name__ == "__main__":
    if "--preview" in sys.argv:
        write_preview()
    else:
        main()
