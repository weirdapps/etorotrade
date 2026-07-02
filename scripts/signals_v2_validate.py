"""S2 signal-validation backtest — does a LEAN price-factor signal have OOS edge?

This script proves (or disproves) whether a price-only factor signal carries
genuine out-of-sample edge, judged by the S0 validation referee
(``trade_modules.validation.harness.evaluate``), against the old engine's known
negative OOS-Sharpe baseline.

Scope honesty
-------------
Only PRICE factors are historically computable from a daily price panel:
  * 12-1 (skip-month) momentum — ``riskfirst.prices.momentum_12_1``
  * low realized volatility     — ``riskfirst.prices.realized_vol`` (inverted)

Value / quality / size need POINT-IN-TIME fundamentals (trailing P/E, FCF, analyst
counts, market cap as-of the rebalance). We do NOT have a point-in-time
fundamental panel, so including them here would inject look-ahead (today's
fundamentals applied to a 2-year-old rebalance). They are therefore
FORWARD-GATED — deliberately OUT of this historical backtest. This backtest
validates the PRICE sleeve only; the fundamental composite must be validated live,
forward, once a point-in-time fundamental store exists.

Correctness
-----------
NO LOOK-AHEAD is the paramount property. ``price_factor_score(panel, t)`` reads
only ``prices[:t+1]``; ``build_signal_rows`` computes the forward return strictly
over ``[t, t+horizon]``. Both are pure and unit-tested (see
``tests/unit/trade_modules/test_signals_v2_validate.py``), the no-look-ahead test
being the hardest: mutating any bar after ``t`` must not change the score at ``t``.

The pure functions are unit-tested; ``main()`` is network glue (yfinance).
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd

from trade_modules.riskfirst.prices import momentum_12_1, realized_vol
from trade_modules.riskfirst.stats import zscore
from trade_modules.signals_v2.composite import map_to_signal

# Flat round-trip transaction cost netted from every alpha (20 bps).
ROUND_TRIP_COST = 0.0020


# --------------------------------------------------------------------------- #
# PURE functions — the correctness core
# --------------------------------------------------------------------------- #


def rebalance_dates(index: pd.DatetimeIndex, freq_days: int = 30, warmup: int = 253) -> list[int]:
    """Rebalance positions every ``freq_days`` trading bars, starting after warmup.

    Returns integer positions into ``index`` (not dates), so downstream code can
    slice the panel positionally without re-locating dates.

    The first rebalance is at position ``warmup`` (need 252+ bars for 12-1
    momentum), then every ``freq_days`` bars until the end of the index.

    PURE. Returns [] when the index is too short to reach warmup.
    """
    n = len(index)
    if n <= warmup or freq_days <= 0:
        return []
    return list(range(warmup, n, freq_days))


def price_factor_score(prices_panel: pd.DataFrame, as_of_pos: int) -> pd.Series:
    """Composite price-factor score per ticker using ONLY prices up to ``as_of_pos``.

    Score = z(momentum_12_1) + z(-realized_vol) across the cross-section, computed
    from ``prices_panel.iloc[: as_of_pos + 1]`` and NOTHING after it.

    NO LOOK-AHEAD: the slice ``[: as_of_pos + 1]`` is the only data touched, so any
    price at a position > as_of_pos is invisible to the score. Higher = more
    attractive (strong momentum + low vol). NaN for names with insufficient
    history at ``as_of_pos``.

    PURE — does not mutate ``prices_panel``.
    """
    hist = prices_panel.iloc[: as_of_pos + 1]

    mom = {t: momentum_12_1(hist[t]) for t in hist.columns}
    vol = {t: realized_vol(hist[t]) for t in hist.columns}

    mom_s = pd.Series(mom, dtype=float)
    vol_s = pd.Series(vol, dtype=float)

    # z-score momentum (higher better) + z-score of NEGATIVE vol (lower vol better).
    mom_z = zscore(mom_s)
    lowvol_z = zscore(-vol_s)

    return mom_z.add(lowvol_z, fill_value=0.0)


def build_signal_rows(
    prices_panel: pd.DataFrame,
    spy: pd.Series,
    freq_days: int = 30,
    horizon_days: int = 30,
    buy_pct: float = 0.2,
    sell_pct: float = 0.2,
) -> list[dict]:
    """Emit harness-ready rows across all valid rebalances.

    At each rebalance position ``t`` (from :func:`rebalance_dates`):
      * ``score = price_factor_score(panel, t)``  — uses only prices[:t+1]
      * ``signal = map_to_signal(score, buy_pct, sell_pct)``  — B/H/S per ticker
      * forward return per ticker = ``close[t+horizon] / close[t] - 1`` (skipped
        when ``t + horizon`` is out of range, or either endpoint is NaN)
      * ``spy_return = spy[t+horizon] / spy[t] - 1``
      * ``alpha = stock_return - spy_return``
      * ``net_alpha = alpha - ROUND_TRIP_COST``

    STRICT coupling contract: the score at ``t`` uses ONLY prices[:t+1]; the
    forward return uses ONLY [t, t+horizon]. There is no other coupling.

    Each row: ``{ticker, signal, signal_date(iso), alpha, net_alpha, horizon,
    tier:'NA'}``. PURE — does not mutate inputs.
    """
    n = len(prices_panel)
    idx = prices_panel.index
    rows: list[dict] = []

    for t in rebalance_dates(idx, freq_days=freq_days, warmup=253):
        fwd = t + horizon_days
        if fwd >= n:
            continue  # forward window falls off the end — no row

        score = price_factor_score(prices_panel, t)
        signal = map_to_signal(score, buy_pct=buy_pct, sell_pct=sell_pct)
        signal_date = idx[t].date().isoformat()

        # SPY forward return over [t, t+horizon].
        spy_t = float(spy.iloc[t])
        spy_fwd = float(spy.iloc[fwd])
        if spy_t <= 0 or pd.isna(spy_t) or pd.isna(spy_fwd):
            continue
        spy_return = spy_fwd / spy_t - 1.0

        for ticker in prices_panel.columns:
            sig = signal.get(ticker)
            if sig is None:
                continue
            p_t = prices_panel[ticker].iloc[t]
            p_fwd = prices_panel[ticker].iloc[fwd]
            if pd.isna(p_t) or pd.isna(p_fwd) or float(p_t) <= 0:
                continue
            stock_return = float(p_fwd) / float(p_t) - 1.0
            alpha = stock_return - spy_return
            rows.append(
                {
                    "ticker": ticker,
                    "signal": sig,
                    "signal_date": signal_date,
                    "alpha": alpha,
                    "net_alpha": alpha - ROUND_TRIP_COST,
                    "horizon": horizon_days,
                    "tier": "NA",
                }
            )

    return rows


def exit_bucket_alpha(rows: list[dict]) -> dict:
    """Exit-rule validation helper: does the EXIT bucket underperform the universe?

    Given harness rows (each with 'signal' and 'alpha'), compute:
      * exit_mean_alpha  — mean alpha for rows where signal == 'EXIT'
      * universe_mean_alpha — mean alpha across ALL rows (BUY + HOLD + EXIT)
      * exit_justified   — True if exit_mean_alpha < universe_mean_alpha (exiting
                           bottom-bucket names is justified by underperformance);
                           False if EXIT bucket matches or beats the universe mean
                           (no exit edge — reported honestly);
                           None if there are no EXIT rows (cannot determine).
      * exit_n           — number of EXIT rows
      * universe_n       — total rows

    This is the long-only honest answer to "should we avoid/exit bottom-bucket names?"
    It reports the truth even when the answer is "no material exit edge."

    PURE — does not mutate inputs.
    """
    if not rows:
        return {
            "exit_mean_alpha": None,
            "universe_mean_alpha": None,
            "exit_justified": None,
            "exit_n": 0,
            "universe_n": 0,
        }

    all_alphas = [r["alpha"] for r in rows]
    exit_alphas = [r["alpha"] for r in rows if r.get("signal") == "EXIT"]

    universe_mean = float(sum(all_alphas) / len(all_alphas))

    if not exit_alphas:
        return {
            "exit_mean_alpha": None,
            "universe_mean_alpha": universe_mean,
            "exit_justified": None,
            "exit_n": 0,
            "universe_n": len(rows),
        }

    exit_mean = float(sum(exit_alphas) / len(exit_alphas))
    exit_justified = exit_mean < universe_mean

    return {
        "exit_mean_alpha": exit_mean,
        "universe_mean_alpha": universe_mean,
        "exit_justified": exit_justified,
        "exit_n": len(exit_alphas),
        "universe_n": len(rows),
    }


# --------------------------------------------------------------------------- #
# main() — network glue (yfinance). Excluded from coverage.
# --------------------------------------------------------------------------- #


def _load_sample_tickers(csv_path: str, cap: int = 80) -> list[str]:  # pragma: no cover - I/O
    """S1-eligible fundamental tickers from etoro.csv, capped to bound yfinance."""
    from trade_modules.universe.filter import filter_universe

    df = pd.read_csv(csv_path)
    result = filter_universe(df)
    eligible = result["eligible"]
    tickers = [str(t).strip() for t in eligible["TKR"].tolist() if str(t).strip()]
    # Cap to the first N (already quality-gated) to bound the yfinance download.
    return tickers[:cap]


def _fmt_pct(x) -> str:  # pragma: no cover - formatting
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def _fmt_num(x, nd: int = 3) -> str:  # pragma: no cover - formatting
    return "n/a" if x is None else f"{x:.{nd}f}"


def _family_line(name: str, fam: dict) -> str:  # pragma: no cover - formatting
    if fam.get("insufficient_data"):
        return f"- **{name}**: insufficient data (n={fam.get('n')})"
    oos = fam.get("oos", {})
    oos_alpha = oos.get("oos_alpha") if oos.get("computed") else None
    return (
        f"- **{name}**: n={fam.get('n')}, mean_alpha={_fmt_pct(fam.get('mu_alpha'))}, "
        f"net_alpha≈{_fmt_pct((fam.get('mu_alpha') or 0) - ROUND_TRIP_COST) if fam.get('mu_alpha') is not None else 'n/a'}, "
        f"per_period_Sharpe={_fmt_num(fam.get('per_period_sharpe'))}, "
        f"DSR={_fmt_num(fam.get('dsr'))}, "
        f"OOS_alpha={_fmt_pct(oos_alpha)}, OOS_hit={_fmt_num(oos.get('oos_hit'))}, "
        f"passed={fam.get('passed')}"
    )


def main() -> int:  # pragma: no cover - network
    import sys

    csv_path = "yahoofinance/output/etoro.csv"
    period = "4y"  # multi-year daily history for price factors
    horizon_days = 30
    freq_days = 30

    print("[s2] loading S1-eligible sample tickers ...", file=sys.stderr)
    try:
        tickers = _load_sample_tickers(csv_path, cap=80)
    except Exception as exc:
        print(f"[s2] FAILED to load tickers: {exc}", file=sys.stderr)
        return 2
    print(f"[s2] sample size: {len(tickers)} tickers", file=sys.stderr)

    from trade_modules.riskfirst.prices import fetch_prices

    print(f"[s2] fetching {period} daily prices for sample + SPY (yfinance) ...", file=sys.stderr)
    panel = fetch_prices(tickers, period=period)
    spy_df = fetch_prices(["SPY"], period=period)
    if "SPY" not in spy_df.columns:
        print("[s2] FAILED: SPY price fetch empty (rate-limited?)", file=sys.stderr)
        return 3
    spy = spy_df["SPY"]

    # Align to common dates; drop tickers with too little history.
    panel = panel.reindex(spy.index).dropna(axis=1, thresh=int(0.6 * len(spy)))
    n_have = panel.shape[1]
    print(f"[s2] usable price series after alignment: {n_have}", file=sys.stderr)
    if n_have < 10 or len(spy) < 300:
        print(
            f"[s2] WARNING: thin data (tickers={n_have}, bars={len(spy)}) — "
            "yfinance likely rate-limited. Verdict may be unreliable; NOT fabricating.",
            file=sys.stderr,
        )

    # Base run at the default 20/20 split.
    from trade_modules.validation.harness import evaluate

    base_rows = build_signal_rows(
        panel, spy, freq_days=freq_days, horizon_days=horizon_days, buy_pct=0.2, sell_pct=0.2
    )
    print(f"[s2] built {len(base_rows)} signal rows @ 20/20 split", file=sys.stderr)
    base_report = evaluate(base_rows, family_key="signal", n_trials=5, min_obs=10)

    # Long-only exit-rule validation: does the bottom bucket underperform the universe?
    exit_verdict = exit_bucket_alpha(base_rows)

    # Sweep buy_pct splits; rank by OOS net alpha of the B family.
    # NOTE: v2 is LONG-ONLY — there is no short sleeve. 'sell_pct' in map_to_signal
    # maps to the EXIT bucket; we sweep it symmetrically for signal calibration only.
    sweep: list[tuple[float, float, float | None, dict]] = []
    for pct in (0.1, 0.2, 0.3):
        rows = build_signal_rows(
            panel, spy, freq_days=freq_days, horizon_days=horizon_days, buy_pct=pct, sell_pct=pct
        )
        rep = evaluate(rows, family_key="signal", n_trials=5, min_obs=10)
        b = rep["families"].get("B", {})
        oos = b.get("oos", {})
        # Prefer OOS net alpha; net = gross OOS alpha minus round-trip cost.
        oos_alpha = oos.get("oos_alpha") if oos.get("computed") else None
        oos_net = (oos_alpha - ROUND_TRIP_COST) if oos_alpha is not None else None
        sweep.append((pct, pct, oos_net, rep))

    ranked = sorted(
        sweep, key=lambda x: (x[2] is not None, x[2] if x[2] is not None else -1e9), reverse=True
    )
    best = ranked[0]

    ts = _dt.datetime.now().strftime("%Y%m%d%H%M")
    out_path = f"{_home()}/Downloads/{ts}_signals_v2_validation.md"
    _write_report(
        out_path, tickers, n_have, len(spy), period, base_report, exit_verdict, sweep, best
    )
    print(f"[s2] report written: {out_path}", file=sys.stderr)

    # Console verdict summary — LONG-ONLY framing.
    b = base_report["families"].get("B", {})
    print("\n===== S2 VERDICT — LONG-ONLY (20/20 split) =====")
    print("Strategy: LONG-ONLY. BUY = enter/hold long; EXIT = close/avoid long. NO short sleeve.")
    print(_family_line("BUY side (B)", b))
    # Exit-rule finding (honest, even if no edge).
    ej = exit_verdict.get("exit_justified")
    exit_str = (
        f"EXIT bucket mean alpha={_fmt_pct(exit_verdict.get('exit_mean_alpha'))}, "
        f"universe mean alpha={_fmt_pct(exit_verdict.get('universe_mean_alpha'))}, "
        f"exit_justified={ej} (n={exit_verdict.get('exit_n')} EXIT rows)"
    )
    print(f"Exit-rule: {exit_str}")
    print(
        f"best split by BUY OOS net alpha: buy={best[0]}/exit={best[1]} (OOS net {_fmt_pct(best[2])})"
    )
    print(f"report: {out_path}")
    return 0


def _home() -> str:  # pragma: no cover - env
    import os

    return os.path.expanduser("~")


def _write_report(  # pragma: no cover - I/O
    path, tickers, n_have, n_bars, period, base_report, exit_verdict, sweep, best
) -> None:
    b = base_report["families"].get("B", {})
    overall = base_report.get("overall", {})
    dsr_assumptions = base_report.get("dsr_assumptions", {})

    lines: list[str] = []
    lines.append("# S2 Signal-Validation Backtest — Long-Only Price-Factor Signal")
    lines.append("")
    lines.append(f"_Generated {_dt.datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append("## Strategy framing — LONG-ONLY")
    lines.append("")
    lines.append(
        "**The v2 strategy is LONG-ONLY. There is no short sleeve.**  "
        "The earlier 'S' (sell/short) framing was a mis-reading of the signal: "
        "the factor composite's bottom quantile showed POSITIVE forward returns in "
        "the backtest, so there is no short edge. 'EXIT' means close or avoid a long "
        "position — it does NOT mean enter a short. `long_only_signal` in "
        "`signals_v2.composite` is the canonical mapper with explicit BUY/HOLD/EXIT labels."
    )
    lines.append("")
    lines.append("## What was tested")
    lines.append("")
    lines.append(
        "A LEAN, price-only factor signal: cross-sectional z-score of 12-1 "
        "(skip-month) momentum + z-score of inverse realized volatility (low-vol). "
        "Top quantile → BUY; bottom quantile → EXIT (close/avoid long). "
        "Judged by the S0 referee (`validation.harness.evaluate`)."
    )
    lines.append("")
    lines.append(
        f"- Sample: {n_have} usable price series (from up to {len(tickers)} S1-eligible "
        f"fundamental names), plus SPY as the benchmark."
    )
    lines.append(f"- History: `{period}` daily bars ({n_bars} aligned trading days).")
    lines.append("- Horizon: 30 trading days. Rebalance: every 30 trading days. Warmup: 253 bars.")
    lines.append(
        f"- Costs: flat {ROUND_TRIP_COST * 1e4:.0f} bps round-trip netted into `net_alpha`."
    )
    lines.append("")
    lines.append("## BUY-side verdict — OOS alpha (20% top quantile)")
    lines.append("")
    lines.append(_family_line("BUY (top 20%)", b))
    lines.append(_family_line("HOLD (middle 60%)", base_report["families"].get("H", {})))
    lines.append("")
    lines.append(f"- Overall gate passed: **{overall.get('passed')}**")
    if overall.get("reasons"):
        for r in overall["reasons"]:
            lines.append(f"  - {r}")
    lines.append(
        f"- DSR assumptions: n_trials={dsr_assumptions.get('n_trials')}, "
        f"var_sr={_fmt_num(dsr_assumptions.get('var_sr'))} "
        f"({dsr_assumptions.get('var_sr_source')})."
    )
    lines.append("")
    lines.append("## Exit-rule validation")
    lines.append("")
    ej = exit_verdict.get("exit_justified")
    lines.append(
        f"- EXIT bucket mean alpha: {_fmt_pct(exit_verdict.get('exit_mean_alpha'))} "
        f"(n={exit_verdict.get('exit_n')} rows)"
    )
    lines.append(f"- Universe mean alpha: {_fmt_pct(exit_verdict.get('universe_mean_alpha'))}")
    if ej is True:
        lines.append(
            "- **exit_justified=True**: EXIT names underperform the universe mean — "
            "exiting/avoiding this bucket is historically supported."
        )
    elif ej is False:
        lines.append(
            "- **exit_justified=False**: EXIT names do NOT underperform the universe mean — "
            "no material exit edge in this sample. Honest finding; reported as-is."
        )
    else:
        lines.append("- exit_justified=None: insufficient EXIT rows to determine.")
    lines.append("")
    lines.append("## BUY/EXIT split sweep (ranked by BUY-family OOS net alpha)")
    lines.append("")
    lines.append(
        "| buy% | exit% | BUY mean alpha | BUY OOS alpha | BUY OOS net | BUY DSR | BUY passed |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for bp, sp, oos_net, rep in sweep:
        fam = rep["families"].get("B", {})
        oos = fam.get("oos", {})
        oos_alpha = oos.get("oos_alpha") if oos.get("computed") else None
        lines.append(
            f"| {bp:.1f} | {sp:.1f} | {_fmt_pct(fam.get('mu_alpha'))} | "
            f"{_fmt_pct(oos_alpha)} | {_fmt_pct(oos_net)} | "
            f"{_fmt_num(fam.get('dsr'))} | {fam.get('passed')} |"
        )
    lines.append("")
    lines.append(
        f"**Best split by BUY OOS net alpha:** buy={best[0]:.1f} / exit={best[1]:.1f} "
        f"(OOS net {_fmt_pct(best[2])})."
    )
    lines.append("")
    lines.append("## Baseline reminder — OLD engine")
    lines.append("")
    lines.append(
        "The prior signal engine (snapshot proxies: 52W-proximity for momentum, "
        "beta for low-vol, ~220 parameters tuned on ~60 observations in a single "
        "bull regime) produced a **NEGATIVE out-of-sample Sharpe** under this same "
        "S0 referee — i.e. no demonstrable OOS edge. That is the bar this rebuild "
        "must clear."
    )
    lines.append("")
    lines.append("## Honest caveats")
    lines.append("")
    lines.append(
        "- **Long-only only.** There is NO short sleeve. The factor composite's "
        "bottom quantile showed positive forward returns in the backtest — no short "
        "edge exists. 'EXIT' is a long-exit signal, not a short-entry signal."
    )
    lines.append(
        "- **Price-factor ONLY.** Value / quality / size require point-in-time "
        "fundamentals (as-of trailing P/E, FCF, analyst counts, market cap). We "
        "have no point-in-time fundamental store, so those factors are "
        "**forward-gated** — deliberately excluded here to avoid look-ahead. This "
        "backtest validates the price sleeve; the fundamental composite must be "
        "validated live/forward once such a store exists."
    )
    lines.append(
        "- History was extended from the old ~4-month window to a multi-year daily "
        "panel specifically so the PRICE factors (which need 252+ bars) are "
        "actually computable out-of-sample."
    )
    lines.append(
        "- Costs are a flat 20 bps round-trip; real execution slippage is not "
        "modelled and would be worse in practice."
    )
    lines.append(
        "- Sample is capped (~80 liquid names) to bound yfinance; it is not the "
        "full universe. A single 30-day horizon and a mostly-one-regime window "
        "limit the strength of any PASS — treat a positive result as *encouraging*, "
        "not *proven across regimes*."
    )
    lines.append(
        "- No-look-ahead is enforced by construction and unit-tested "
        "(`test_no_lookahead_score_invariant_to_future_bars`)."
    )
    lines.append("")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
