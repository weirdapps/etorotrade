"""Shadow runner — wire the 5 real factors + live universe + edge gate.

Produces a risk-first target book and per-name recommendations in SHADOW mode
(it does NOT trade and does NOT drive sizing). The edge gate decides whether the
engine may ever be promoted out of shadow; with no forward track record for the
new composite (and a single-regime sample), it is expected to FAIL — which is the
correct, honest outcome: the rebuilt engine must now log its own signals forward
across >= one bear regime before any un-clamping.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from .edgegate import gate_verdict
from .engine import eligible_universe, recommend, select_and_construct
from .factors import lowvol, momentum, quality, size, value

FACTORS = [value.compute, quality.compute, momentum.compute, lowvol.compute, size.compute]
FACTOR_NAMES = ["value", "quality", "momentum", "lowvol", "size"]

DEFAULT_UNIVERSE = os.path.expanduser("~/SourceCode/etorotrade/yahoofinance/output/etoro.csv")
DEFAULT_PORTFOLIO = os.path.expanduser("~/SourceCode/etorotrade/yahoofinance/output/portfolio.csv")

_NUMERIC_COLS = [
    "PRC",
    "UP%",
    "%B",
    "AM",
    "B",
    "52W",
    "PET",
    "PEF",
    "P/S",
    "PEG",
    "DV",
    "SI",
    "EG",
    "ROE",
    "DE",
    "FCF",
]


def load_universe(path: str = DEFAULT_UNIVERSE) -> pd.DataFrame:
    """Load the processed universe, indexed by ticker, numerics coerced.
    CAP is kept as a string for the size factor's parser."""
    df = pd.read_csv(path)
    df = df.rename(columns={"TKR": "ticker"}).set_index("ticker")
    for c in _NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_current_weights(path: str = DEFAULT_PORTFOLIO) -> pd.Series:
    """Best-effort current portfolio weights (fraction of book) by ticker.
    Returns an empty Series if the file or a usable weight column is absent."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.Series(dtype=float)
    tcol = next((c for c in ("ticker", "TKR", "symbol", "Ticker") if c in df.columns), None)
    wcol = next(
        (
            c
            for c in ("weight", "Weight", "totalInvestmentPct", "allocation", "%")
            if c in df.columns
        ),
        None,
    )
    if tcol is None or wcol is None:
        return pd.Series(dtype=float)
    w = pd.to_numeric(df[wcol], errors="coerce")
    mx = w.max()
    if pd.notna(mx) and mx > 1.5:  # looks like percent, normalise to fraction
        w = w / 100.0
    return pd.Series(w.values, index=df[tcol].astype(str)).dropna()


def run(
    universe_path: str = DEFAULT_UNIVERSE,
    portfolio_path: str = DEFAULT_PORTFOLIO,
    *,
    top_n: int = 20,
    name_cap: float = 0.08,
    usd_bloc_cap: float = 0.60,
    target_vol: float = 0.12,
    market_vol: float = 0.18,
    idio_vol: float = 0.30,
    forward_obs: int = 0,
    n_regimes: int = 1,
    regime_overlay_enabled=None,
    regime_fn=None,
    regime_state_path=None,
    persistence_days=None,
    event_gate_enabled=None,
    event_risk_path=None,
) -> dict:
    """Run the shadow engine. Returns target weights, recommendations, edge verdict."""
    from .news_gate import load_config as _load_event_cfg
    from .regime_state import load_config as _load_regime_cfg

    _rcfg = _load_regime_cfg()
    _ecfg = _load_event_cfg()
    _regime_on = _rcfg["enabled"] if regime_overlay_enabled is None else regime_overlay_enabled
    _persist = _rcfg["persistence_days"] if persistence_days is None else persistence_days
    _regime_table = _rcfg["exposure"]
    _event_on = _ecfg["enabled"] if event_gate_enabled is None else event_gate_enabled
    _event_path = event_risk_path or _ecfg["event_risk_path"]

    df = load_universe(universe_path)
    df = eligible_universe(df, min_cap=2e9, min_factors=3)  # investability gate
    n_excluded = 0
    if _event_on:
        from .news_gate import apply_exclusions, load_event_risk

        excl = load_event_risk(_event_path)
        before = len(df)
        df = apply_exclusions(df, excl)
        n_excluded = before - len(df)
    regime = {"raw_regime": None, "confirmed_regime": None, "applied_multiplier": 1.0}
    if _regime_on:
        from .regime_state import DEFAULT_STATE_PATH, resolve_regime_multiplier

        _mult, regime = resolve_regime_multiplier(
            state_path=regime_state_path or DEFAULT_STATE_PATH,
            persistence_days=_persist,
            regime_fn=regime_fn,
            table=_regime_table,
        )
    built = select_and_construct(
        df,
        FACTORS,
        top_n=top_n,
        name_cap=name_cap,
        usd_bloc_cap=usd_bloc_cap,
        target_vol=target_vol,
        market_vol=market_vol,
        idio_vol=idio_vol,
        regime_multiplier=regime["applied_multiplier"],
    )
    target = built["weights"][built["weights"] > 1e-9]
    current = load_current_weights(portfolio_path)
    recs = recommend(target, current)

    # Edge gate for PROMOTION out of shadow. The rebuilt composite has no forward
    # track record of its own yet (forward_obs defaults 0) and the available sample
    # is single-regime -> gate FAILS -> stays in shadow (conviction clamp stays on).
    verdict = gate_verdict(
        sr=0.0,
        n_obs=forward_obs,
        n_trials=1,
        var_sr=0.02,
        n_regimes=n_regimes,
    )
    return {
        "mode": "SHADOW",
        "selected": built["selected"],
        "target_weights": target,
        "gross": built["gross"],
        "cash": built["cash"],
        "usd_bloc": built["usd_bloc"],
        "recommendations": recs,
        "edge_gate": verdict,
        "promotable": verdict["passed"],
        "regime": regime,
        "event_excluded": n_excluded,
    }


def build_report_md(res: dict) -> str:
    """Format a shadow run as a consumable markdown recommendation report."""
    g = res["edge_gate"]
    lines = [
        "# riskfirst SHADOW recommendations",
        "",
        f"**Mode:** {res['mode']} · gross {res['gross']:.1%} · cash {res['cash']:.1%} "
        f"· USD-bloc {res['usd_bloc']:.1%}"
        f" · regime {(res['regime']['confirmed_regime'] or 'neutral')} (×{res['regime']['applied_multiplier']:.2f})",
        "",
        "## Target book",
        "",
        "| Ticker | Weight |",
        "|---|---:|",
    ]
    for tkr, w in res["target_weights"].sort_values(ascending=False).items():
        lines.append(f"| {tkr} | {w:.2%} |")
    lines += [
        "",
        "## Recommendations vs current book",
        "",
        "| Ticker | Action | Current | Target | Delta |",
        "|---|---|---:|---:|---:|",
    ]
    recs = res["recommendations"]
    if len(recs):
        for _, r in recs.sort_values("delta", ascending=False).iterrows():
            lines.append(
                f"| {r['ticker']} | {r['action']} | {r['current']:.2%} | "
                f"{r['target']:.2%} | {r['delta']:+.2%} |"
            )
    lines += ["", f"## Edge gate: {'PASS' if g['passed'] else 'FAIL'} (DSR {g['dsr']:.3f})"]
    lines += [f"- {r}" for r in g["reasons"]]
    lines += ["", f"**Promotable out of shadow:** {res['promotable']}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:  # pragma: no cover - integration entry point
    import datetime

    res = run()
    print("=== riskfirst SHADOW run ===")
    print(
        f"selected {len(res['selected'])} names | gross {res['gross']:.2%} | "
        f"cash {res['cash']:.2%} | USD-bloc {res['usd_bloc']:.2%}"
    )
    print(
        f"Edge gate: {'PASS' if res['edge_gate']['passed'] else 'FAIL'} "
        f"(DSR {res['edge_gate']['dsr']:.3f}) | promotable: {res['promotable']}"
    )

    ts = datetime.datetime.now().strftime("%Y%m%d%H%M")
    md_path = os.path.expanduser(f"~/Downloads/{ts}_riskfirst_shadow.md")
    json_path = os.path.expanduser(f"~/Downloads/{ts}_riskfirst_shadow.json")
    with open(md_path, "w") as f:
        f.write(build_report_md(res))
    with open(json_path, "w") as f:
        json.dump(
            {
                "mode": res["mode"],
                "gross": res["gross"],
                "cash": res["cash"],
                "usd_bloc": res["usd_bloc"],
                "edge_gate": res["edge_gate"],
                "promotable": res["promotable"],
                "regime": res["regime"],
                "target_weights": res["target_weights"].round(4).to_dict(),
                "recommendations": res["recommendations"].to_dict(orient="records"),
            },
            f,
            indent=2,
        )
    print(f"Report:  {md_path}\nJSON:    {json_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
