# scripts/v3_full_report.py
"""Trading Model v3 — FULLY-WIRED report driver (Phase 5D).

Runs the full v3 pipeline (universe -> features -> scores -> deployment ->
build_portfolio -> build_actions) and renders the complete HTML via
``trade_modules.v3.report.render_report`` with the executive / risk panel and the
decision-support Suggested Actions section wired ABOVE the existing factor cards.
Writes ``~/Downloads/<UTCstamp>_v3_full_report.html`` (UTC ``%Y%m%d%H%M``).

Live account (optional): a JSON at ``$V3_ACCOUNT_JSON`` or
``~/Downloads/v3_live_account.json`` with schema::

    {"nav": <float|null>, "weights": {"<ticker>": <fraction 0..1>, ...}}

supplies REAL current weights + NAV for the action diff (delta$ shown when nav
is present). When it is absent, current weights fall back to an equal split
across held tickers and the report carries a clear "approximate" note.

Also emits a network-free synthetic preview to
``~/Downloads/v3_full_report_preview.html`` (structurally self-checked) so a
reviewer can screenshot the layout without running the live pipeline.

Run (VPS / network allowed):   .venv/bin/python scripts/v3_full_report.py
Preview only (no network):     .venv/bin/python scripts/v3_full_report.py --preview

Does NOT modify or import from the network path of v3_portfolio.py / v3_report.py
beyond reusing the pure ``trend_regime`` helper.  No module-level
``yahoofinance.core.config`` import.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.v3_portfolio import trend_regime  # noqa: E402  (pure, unit-tested)
from trade_modules.v3.actions import build_actions  # noqa: E402
from trade_modules.v3.combine import compute_scores  # noqa: E402
from trade_modules.v3.conditioning import resolve_deployment  # noqa: E402
from trade_modules.v3.construct import build_portfolio  # noqa: E402
from trade_modules.v3.features import enrich_features  # noqa: E402
from trade_modules.v3.fetch import robust_fetch_prices  # noqa: E402
from trade_modules.v3.report import compute_regime, render_report  # noqa: E402

PORTFOLIO_CSV = "yahoofinance/output/portfolio.csv"
BUY_CSV = "yahoofinance/output/buy.csv"
ETORO_CSV = "yahoofinance/output/etoro.csv"

DEFAULT_ACCOUNT_JSON = "~/Downloads/v3_live_account.json"
PREVIEW_OUT = "~/Downloads/v3_full_report_preview.html"


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested with synthetic data / temp files)
# ---------------------------------------------------------------------------


def load_account_json(path: str | None = None) -> tuple[pd.Series, float | None, bool]:
    """Load live-account current weights + NAV.

    Resolution order: explicit ``path`` arg -> ``$V3_ACCOUNT_JSON`` ->
    ``~/Downloads/v3_live_account.json``.  Schema::

        {"nav": <float|null>, "weights": {"<ticker>": <fraction 0..1>, ...}}

    Args:
        path: Optional explicit path (takes precedence over the env var).

    Returns:
        ``(weights, nav, present)``:
        - ``weights``: ``pd.Series`` of fractions indexed by ticker (empty when
          the file is absent / unreadable / has no ``weights``).
        - ``nav``: ``float`` NAV, or ``None`` when null / missing.
        - ``present``: ``True`` only when a JSON file was found AND parsed.
    """
    # An explicit path is authoritative (no env/default fallback); only when no
    # path is passed do we consult $V3_ACCOUNT_JSON then the default location.
    if path is not None:
        candidates = [path]
    else:
        candidates = [os.environ.get("V3_ACCOUNT_JSON"), DEFAULT_ACCOUNT_JSON]
    chosen = next(
        (os.path.expanduser(p) for p in candidates if p and os.path.exists(os.path.expanduser(p))),
        None,
    )
    if not chosen:
        return pd.Series(dtype=float), None, False

    try:
        with open(chosen, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:  # unreadable / malformed JSON
        print(f"warn: could not read account JSON {chosen}: {exc}", file=sys.stderr)
        return pd.Series(dtype=float), None, False

    raw_w = data.get("weights") or {}
    try:
        weights = pd.Series({str(k): float(v) for k, v in raw_w.items()}, dtype=float)
    except (TypeError, ValueError) as exc:
        print(f"warn: bad weights in account JSON {chosen}: {exc}", file=sys.stderr)
        weights = pd.Series(dtype=float)

    nav_raw = data.get("nav")
    nav = float(nav_raw) if nav_raw is not None else None
    return weights, nav, True


def load_account_positions(path: str | None = None) -> dict[str, dict]:
    """Load per-ticker P/L from the live-account JSON ``positions`` block.

    Mirrors :func:`load_account_json` path resolution. Returns
    ``{ticker: {"pnl", "pnl_pct", "current_value", ...}}`` — empty when the file
    is absent or carries no ``positions`` (older snapshots without P/L).
    """
    if path is not None:
        candidates: list[str | None] = [path]
    else:
        candidates = [os.environ.get("V3_ACCOUNT_JSON"), DEFAULT_ACCOUNT_JSON]
    chosen = next(
        (os.path.expanduser(p) for p in candidates if p and os.path.exists(os.path.expanduser(p))),
        None,
    )
    if not chosen:
        return {}
    try:
        with open(chosen, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    pos = data.get("positions") or {}
    return {str(k): v for k, v in pos.items() if isinstance(v, dict)}


def resolve_current_weights(
    port_tickers: list[str], account_weights: pd.Series, account_present: bool
) -> tuple[pd.Series, bool]:
    """Resolve the current-weights vector used for the action diff.

    When a live account file supplied non-empty weights, those are used verbatim
    (exact holdings).  Otherwise the current book is approximated as an EQUAL
    SPLIT across the (deduped) portfolio tickers and the ``approx`` flag is set so
    the report can warn that current weights are indicative.

    Args:
        port_tickers: Tickers held per portfolio.csv (order preserved).
        account_weights: Weights parsed from the live account JSON.
        account_present: Whether a live account JSON was found + parsed.

    Returns:
        ``(current_weights, approx)`` where ``approx`` is ``True`` when the
        equal-split fallback was used.
    """
    if account_present and not account_weights.empty:
        return account_weights, False
    tickers = list(dict.fromkeys(str(t) for t in port_tickers))
    if not tickers:
        return pd.Series(dtype=float), True
    eq = 1.0 / len(tickers)
    return pd.Series(eq, index=tickers, dtype=float), True


def _read_tickers(path: str) -> list[str]:
    try:
        df = pd.read_csv(path, na_values=["--"])
        return df["TKR"].dropna().astype(str).tolist()
    except Exception as exc:  # noqa: BLE001
        print(f"warn: could not read {path}: {exc}", file=sys.stderr)
        return []


def _system_read(scores: pd.DataFrame, regime: str) -> str:
    """One-sentence standfirst: portfolio ADD/TRIM + candidate buy-watch counts."""
    port = scores[scores.get("is_portfolio", False) == True]  # noqa: E712
    adds = int((port["conviction"] > 0.5).sum())
    trims = int((port["conviction"] < -0.5).sum())
    cand = scores[scores.get("is_portfolio", True) == False]  # noqa: E712
    watch = int((cand["conviction"] > 0.5).sum())
    return (
        f"Regime {regime}: {adds} portfolio ADD / {trims} TRIM signals, "
        f"{watch} candidate(s) clearing the buy-watch bar. Risk-gated book with "
        f"decision-support actions; weigh full-cluster conviction, not single factors."
    )


# ---------------------------------------------------------------------------
# Synthetic preview (network-free) + structural self-check
# ---------------------------------------------------------------------------


def _synthetic_scores() -> pd.DataFrame:
    """A deterministic, mixed-currency synthetic scored frame (no network)."""
    rng = np.random.default_rng(7)
    n = 22
    suffixes = ["", ".PA", ".L", ".DE", ".T", ""]  # mix so USD-bloc stays under cap
    idx = [f"SYN{i:02d}{suffixes[i % len(suffixes)]}" for i in range(n)]
    sectors = [
        ["Technology", "Financials", "Energy", "Health Care", "Industrials"][i % 5]
        for i in range(n)
    ]
    price = rng.uniform(20, 480, n).round(2)
    df = pd.DataFrame(
        {
            "name": [f"Synthetic Co {i:02d}" for i in range(n)],
            "sector": sectors,
            "description": ["Illustrative synthetic issuer for preview and QA." for _ in range(n)],
            "price": price,
            "pe_trailing": rng.normal(22, 7, n),
            "pe_forward": rng.normal(19, 6, n),
            "pb": rng.normal(6, 3, n),
            "ev_ebitda": rng.normal(15, 6, n),
            "roe": rng.normal(20, 10, n),
            "roa": rng.uniform(0.02, 0.30, n),
            "gross_margin": rng.uniform(0.20, 0.70, n),
            "op_margin": rng.uniform(0.05, 0.40, n),
            "fcf": rng.uniform(0.5, 6.0, n),
            "current_ratio": rng.uniform(0.8, 3.0, n),
            "de": rng.normal(90, 40, n),
            "mom_12_1": rng.normal(0.08, 0.25, n),
            "price_perf": rng.normal(10, 15, n),
            "pct_52w_high": rng.uniform(40, 100, n),
            "beta": rng.uniform(0.6, 1.6, n),
            "realized_vol": rng.uniform(0.15, 0.50, n),
            "analyst_mom": rng.normal(2, 3, n),
            "upside": rng.normal(12, 15, n),
            "buy_pct": rng.uniform(20, 100, n),
            "short_interest": rng.uniform(0.5, 6.0, n),
            "target_dispersion": rng.uniform(0.1, 0.6, n),
            "div_yield": rng.uniform(0, 4, n),
            "cap": rng.uniform(5e9, 3e12, n),
            "avg_volume": rng.uniform(3e5, 3e6, n),
            "adv_usd": rng.uniform(5e7, 5e8, n),
        },
        index=idx,
    )
    df["entry"] = price
    df["stop_loss"] = (price * 0.90).round(2)
    df["take_profit"] = (price * 1.15).round(2)
    df["rr"] = 1.5
    scores = compute_scores(df, sector_neutral=True)
    n_port = 6
    scores["is_portfolio"] = [True] * n_port + [False] * (n - n_port)
    return scores


def build_synthetic_preview_html() -> str:
    """Render the FULL report from synthetic data (no network) for screenshot/QA.

    Current weights are crafted so that every action group (BUY / ADD / TRIM /
    SELL / HOLD) is represented.
    """
    scores = _synthetic_scores()
    result = build_portfolio(scores, pd.DataFrame(), top_n=12)
    target = result["weights"]
    invested = list(target[target > 1e-6].index)

    cur: dict[str, float] = {}
    if len(invested) >= 3:
        cur[invested[0]] = max(float(target[invested[0]]) - 0.03, 0.001)  # ADD
        cur[invested[1]] = float(target[invested[1]]) + 0.05  # TRIM
        cur[invested[2]] = float(target[invested[2]])  # HOLD
    # invested[3:] absent -> BUY; a held name outside the target -> SELL.
    cur["SYNSELL"] = 0.05
    current = pd.Series(cur, dtype=float)

    actions = build_actions(target, current, scores, nav=250_000.0)
    _, cond = resolve_deployment("neutral")

    now = datetime.now(timezone.utc)
    meta = {
        "date": now.strftime("%Y-%m-%d"),
        "n_portfolio": int(scores["is_portfolio"].sum()),
        "n_candidates": int((~scores["is_portfolio"]).sum()),
        "regime": "NEUTRAL",
        "regime_detail": "synthetic offline preview (no live index fetch)",
        "system_read": _system_read(scores, "NEUTRAL"),
        "priced": int(scores["mom_12_1"].notna().sum()),
        "enriched": int(scores["pb"].notna().sum()),
        "generated_utc": now.strftime("%Y-%m-%d %H:%M UTC"),
        "current_weights_approx": False,
    }
    return render_report(scores, meta, portfolio=result, actions=actions, conditioning=cond)


def _self_check(html: str) -> None:
    """Assert the rendered HTML has the exec panel, every action group, factor
    cards, no literal 'None', and zero em-dashes (U+2014)."""
    problems: list[str] = []
    if '<div class="exec-panel">' not in html:
        problems.append("exec/risk panel missing")
    for cls in ("buy", "add", "trim", "sell", "hold"):
        if f"act-grp act-grp--{cls}" not in html:
            problems.append(f"action group {cls} missing")
    if '<article class="card"' not in html:
        problems.append("factor cards missing")
    if "None" in html:
        problems.append("literal 'None' present")
    if "—" in html:
        problems.append("em-dash (U+2014) present")
    if problems:
        raise AssertionError("preview self-check failed: " + "; ".join(problems))


def write_preview(out: str = PREVIEW_OUT) -> str:
    """Build + self-check the synthetic preview and write it to disk."""
    html = build_synthetic_preview_html()
    _self_check(html)
    path = os.path.expanduser(out)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"synthetic preview -> {path}  (self-check passed)")
    return path


# ---------------------------------------------------------------------------
# Live pipeline (smoke-tested by the VPS run, not unit tests)
# ---------------------------------------------------------------------------


def main() -> None:
    # --- Universe assembly (mirrors v3_portfolio.py) ---
    port = _read_tickers(PORTFOLIO_CSV)
    buy = _read_tickers(BUY_CSV)
    port_set = set(port)
    universe = list(dict.fromkeys(port + buy))
    print(
        f"universe: {len(universe)} tickers ({len(port_set)} portfolio + "
        f"{len(set(buy) - port_set)} candidates)"
    )

    # --- Feature enrichment + scoring ---
    feats = enrich_features(
        universe, ETORO_CSV, price_period="2y", accruals_fetch=lambda _tickers: {}
    )
    priced = int(feats["mom_12_1"].notna().sum())
    enriched = int(feats["pb"].notna().sum())
    scores = compute_scores(feats, sector_neutral=True)
    scores["is_portfolio"] = scores.index.isin(port_set)

    elig = scores.get("eligible", pd.Series(True, index=scores.index)).fillna(False).astype(bool)
    n_excluded = int((~elig).sum())
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

    regime, _mult = trend_regime(spx_close)  # lowercase key for deployment
    regime_label, regime_detail = compute_regime(spx_close)  # display label + detail
    gross_target, cond = resolve_deployment(regime, polymarket_signal=None)
    print(f"regime: {regime}  deployment: {gross_target:.0%}")

    # --- Risk-first construction ---
    result = build_portfolio(
        scores,
        prices,
        top_n=20,
        target_vol=0.12,
        name_cap=0.08,
        sector_cap=0.25,
        usd_bloc_cap=0.60,
        gross_target=gross_target,
    )

    # --- Current weights (live account or equal-split fallback) + actions ---
    account_weights, nav, present = load_account_json()
    current_weights, approx = resolve_current_weights(port, account_weights, present)
    if present:
        print(f"current weights: live account ({len(current_weights)} names, nav={nav})")
    else:
        print(f"current weights: equal-split fallback ({len(current_weights)} names)")
    actions = build_actions(result["weights"], current_weights, scores, nav=nav)

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
    }
    html = render_report(scores, meta, portfolio=result, actions=actions, conditioning=cond)

    stamp = now.strftime("%Y%m%d%H%M")
    out = os.path.expanduser(f"~/Downloads/{stamp}_v3_full_report.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"excluded (ineligible): {n_excluded}")
    print(f"full report -> {out}")

    # Always emit the offline synthetic preview alongside the live report.
    write_preview()


if __name__ == "__main__":
    if "--preview" in sys.argv:
        write_preview()
    else:
        main()
