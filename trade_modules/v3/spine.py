"""Price-spine sleeves: 12-1 momentum and low-volatility factors."""

import pandas as pd


def _z(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


def momentum_12_1(
    eur_close: pd.DataFrame,
    asof,
    skip: int = 21,
    lookback: int = 252,
) -> pd.Series:
    """12-1 momentum: cross-sectional z of (price[i-skip] / price[i-lookback]) - 1.

    Returns an empty Series when there are fewer than `lookback` rows before `asof`.
    """
    idx = eur_close.index
    if asof not in idx:
        return pd.Series(dtype=float)
    i = idx.get_loc(asof)
    if i - skip < 0 or i - lookback < 0:
        return pd.Series(dtype=float)
    mom = (eur_close.iloc[i - skip] / eur_close.iloc[i - lookback]) - 1.0
    return _z(mom.dropna())


def low_vol(
    eur_close: pd.DataFrame,
    asof,
    lookback: int = 252,
) -> pd.Series:
    """Low-volatility factor: cross-sectional z of negative realized vol.

    Low-vol stocks score high (negative of vol → higher = more stable).
    Returns an empty Series when there are fewer than `lookback` rows before `asof`.
    """
    idx = eur_close.index
    if asof not in idx:
        return pd.Series(dtype=float)
    i = idx.get_loc(asof)
    if i - lookback < 0:
        return pd.Series(dtype=float)
    rets = eur_close.iloc[i - lookback : i + 1].pct_change(fill_method=None).iloc[1:]
    vol = rets.std(ddof=0)
    return _z(-vol.dropna())


def spine_scores(
    eur_close: pd.DataFrame,
    asof_dates,
    w_mom: float = 0.5,
    w_lv: float = 0.5,
) -> pd.DataFrame:
    """Combine momentum and low-vol into a single per-date cross-sectional z-score.

    Returns a long DataFrame with columns [as_of, ticker, score].
    """
    rows = []
    for asof in asof_dates:
        both = pd.concat(
            [
                momentum_12_1(eur_close, asof).rename("mom"),
                low_vol(eur_close, asof).rename("lv"),
            ],
            axis=1,
        ).dropna()
        if both.empty:
            continue
        score = _z(w_mom * both["mom"] + w_lv * both["lv"])
        for tkr, sc in score.items():
            rows.append({"as_of": asof, "ticker": tkr, "score": float(sc)})
    return pd.DataFrame(rows, columns=["as_of", "ticker", "score"])
