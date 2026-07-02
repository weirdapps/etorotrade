"""Pure signal-composite module for the S2 signal rebuild.

All functions are PURE — no I/O, no network, no mutation.

V2 STRATEGY IS LONG-ONLY
-------------------------
``long_only_signal`` is the **canonical mapper** for the v2 production strategy.
Labels: 'BUY' (top quantile — attractive long), 'EXIT' (bottom quantile — close
or avoid a long; NOT a short position), 'HOLD' (middle). 'EXIT' means the factor
composite ranks this name in the bottom bucket; the correct action is to sell *if
currently held* or skip *if not held*. There is NO short sleeve: the backtest
showed the factor composite's bottom quantile had POSITIVE forward returns, so no
short edge exists.

``map_to_signal`` is kept for back-compat (returns 'B'/'H'/'S'). Under the v2
strategy its 'S' label must be interpreted as "exit long" NOT "enter short". For
new code, prefer ``long_only_signal`` whose labels make the intent unambiguous.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from trade_modules.riskfirst.engine import composite_score as _composite_score
from trade_modules.riskfirst.prices import momentum_12_1


def _default_factor_fns() -> list[Callable[[pd.DataFrame], pd.Series]]:
    """The five default riskfirst factor functions."""
    from trade_modules.riskfirst.factors.lowvol import compute as lowvol_compute
    from trade_modules.riskfirst.factors.momentum import compute as momentum_compute
    from trade_modules.riskfirst.factors.quality import compute as quality_compute
    from trade_modules.riskfirst.factors.size import compute as size_compute
    from trade_modules.riskfirst.factors.value import compute as value_compute

    return [value_compute, quality_compute, momentum_compute, lowvol_compute, size_compute]


def factor_composite(
    df: pd.DataFrame,
    factor_fns: list[Callable[[pd.DataFrame], pd.Series]] | None = None,
    weights: list[float] | None = None,
) -> pd.Series:
    """Composite factor score per name.

    If factor_fns is None, uses the 5 default riskfirst factors (value, quality,
    momentum, lowvol, size). Thin delegation to riskfirst.engine.composite_score.
    PURE — does not mutate df.

    Args:
        df: DataFrame indexed by ticker. Columns from the processed universe.
        factor_fns: list of callables ``(df) -> pd.Series``. None = defaults.
        weights: optional list of floats, one per factor fn. None = equal weight.

    Returns:
        pd.Series indexed to df, higher = more attractive. NaN where all factors
        are NaN for a name.
    """
    fns = factor_fns if factor_fns is not None else _default_factor_fns()
    return _composite_score(df, fns, weights=weights)


def map_to_signal(
    composite: pd.Series,
    buy_pct: float = 0.20,
    sell_pct: float = 0.20,
) -> pd.Series:
    """Map a composite Series to a Series of 'B'/'H'/'S'.

    Top ``buy_pct`` fraction (highest composite) → 'B';
    bottom ``sell_pct`` → 'S'; middle → 'H'.

    NaN composite → 'H' (not tradable on this signal; don't crash).
    Uses quantile thresholds. Ties at the boundary resolve deterministically
    (>= for buy cut, <= for sell cut).

    Guards:
        - empty → empty
        - buy_pct + sell_pct > 1 → raise ValueError
        - pct = 0 → none in that class
        - does not mutate the input

    Args:
        composite: pd.Series of floats, index = ticker names.
        buy_pct: fraction of universe to label 'B' (top).
        sell_pct: fraction of universe to label 'S' (bottom).

    Returns:
        pd.Series of str ('B'/'H'/'S'), same index as composite.
    """
    if buy_pct + sell_pct > 1.0 + 1e-9:
        raise ValueError(
            f"buy_pct ({buy_pct}) + sell_pct ({sell_pct}) > 1.0 — buy and sell zones would overlap"
        )

    if len(composite) == 0:
        return pd.Series(dtype=str)

    # Work on a copy — never mutate input
    comp = composite.copy()

    # NaN → will be mapped to 'H' at the end
    valid = comp.dropna()

    if len(valid) == 0 or buy_pct == 0.0 and sell_pct == 0.0:
        return pd.Series("H", index=composite.index)

    # Compute quantile cut-points from the non-NaN values
    buy_cut: float = float(valid.quantile(1.0 - buy_pct)) if buy_pct > 0.0 else float("inf")
    sell_cut: float = float(valid.quantile(sell_pct)) if sell_pct > 0.0 else float("-inf")

    # Assign labels: NaN → 'H', above buy_cut → 'B', below sell_cut → 'S', else 'H'
    labels = pd.Series("H", index=composite.index)
    if buy_pct > 0.0:
        labels = labels.where(~(comp >= buy_cut), other="B")
    if sell_pct > 0.0:
        labels = labels.where(~(comp <= sell_cut), other="S")

    # Ties: a ticker that qualifies for BOTH B and S (only possible at a point mass)
    # resolves to 'B' (top wins). This preserves the original behavior.
    # NaN inputs always 'H' (override any boundary assignment)
    nan_mask = composite.isna()
    labels[nan_mask] = "H"

    return labels


def long_only_signal(
    composite: pd.Series,
    buy_pct: float = 0.20,
    exit_pct: float = 0.20,
) -> pd.Series:
    """Long-only signal. Top ``buy_pct`` (highest composite) → 'BUY' (attractive long);
    bottom ``exit_pct`` → 'EXIT' (do NOT hold — sell if currently held, never short);
    middle → 'HOLD'. NaN → 'HOLD'. This replaces the symmetric B/S/H short interpretation:
    'EXIT' means close/avoid a long, NOT a short position. Reuse the same percentile logic
    as map_to_signal. PURE.

    Guards:
        - empty → empty
        - buy_pct + exit_pct > 1 → raise ValueError
        - exit_pct = 0 → no EXIT labels
        - buy_pct = 0 → no BUY labels
        - does not mutate the input

    Args:
        composite: pd.Series of floats, index = ticker names.
        buy_pct: fraction of universe to label 'BUY' (top quantile).
        exit_pct: fraction of universe to label 'EXIT' (bottom quantile).

    Returns:
        pd.Series of str ('BUY'/'HOLD'/'EXIT'), same index as composite.
        'S' / short labels are NEVER emitted.
    """
    if buy_pct + exit_pct > 1.0 + 1e-9:
        raise ValueError(
            f"buy_pct ({buy_pct}) + exit_pct ({exit_pct}) > 1.0 — BUY and EXIT zones would overlap"
        )

    if len(composite) == 0:
        return pd.Series(dtype=str)

    comp = composite.copy()
    valid = comp.dropna()

    if len(valid) == 0 or (buy_pct == 0.0 and exit_pct == 0.0):
        return pd.Series("HOLD", index=composite.index)

    buy_cut: float = float(valid.quantile(1.0 - buy_pct)) if buy_pct > 0.0 else float("inf")
    exit_cut: float = float(valid.quantile(exit_pct)) if exit_pct > 0.0 else float("-inf")

    labels = pd.Series("HOLD", index=composite.index)
    if buy_pct > 0.0:
        labels = labels.where(~(comp >= buy_cut), other="BUY")
    if exit_pct > 0.0:
        labels = labels.where(~(comp <= exit_cut), other="EXIT")

    # Ties at the boundary: BUY wins over EXIT (mirrors map_to_signal tie rule).
    # NaN always → 'HOLD'.
    nan_mask = composite.isna()
    labels[nan_mask] = "HOLD"

    return labels


def price_sleeve_signal(
    prices_df: pd.DataFrame,
    ma_window: int = 200,
) -> pd.Series:
    """For the price-only sleeve (crypto/ETFs — no fundamentals): per-ticker
    'B'/'H'/'S' from trend.

    For each column (ticker) in prices_df:
        mom = momentum_12_1(series)   — 12-1 Jegadeesh-Titman momentum
        last = last close
        ma   = mean of last ma_window closes

    'B' if mom > 0 AND last > ma;
    'S' if mom < 0 AND last < ma;
    'H' otherwise (includes: insufficient history, flat, choppy, NaN).

    Insufficient history (fewer than ~253 observations for momentum or
    ma_window for MA) → 'H'. PURE, NaN-safe. Does not mutate inputs.

    Args:
        prices_df: DataFrame of daily closing prices, columns = tickers,
                   index = dates (typically DatetimeIndex). May be empty.
        ma_window: lookback for the moving average (default 200 days).

    Returns:
        pd.Series indexed by ticker name, values in {'B', 'H', 'S'}.
    """
    if prices_df.empty:
        return pd.Series(dtype=str)

    results: dict[str, str] = {}
    min_mom_obs = 253  # year + 1 for JT momentum

    for ticker in prices_df.columns:
        series = prices_df[ticker].dropna()
        n = len(series)

        # Insufficient history guard
        if n < max(min_mom_obs, ma_window):
            results[ticker] = "H"
            continue

        mom = momentum_12_1(series)

        # MA guard
        if n < ma_window:
            results[ticker] = "H"
            continue

        last_close = float(series.iloc[-1])
        ma = float(series.iloc[-ma_window:].mean())

        # NaN guards
        if not np.isfinite(mom) or not np.isfinite(last_close) or not np.isfinite(ma):
            results[ticker] = "H"
            continue

        if mom > 0 and last_close > ma:
            results[ticker] = "B"
        elif mom < 0 and last_close < ma:
            results[ticker] = "S"
        else:
            results[ticker] = "H"

    return pd.Series(results)
