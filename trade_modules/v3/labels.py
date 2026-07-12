import pandas as pd


def forward_returns(
    eur_close: pd.DataFrame,
    asof_dates: list,
    horizons: list[int],
) -> pd.DataFrame:
    """Compute strictly-forward returns for each (as_of, horizon) pair.

    Uses iloc[i+h] so the future leg never reads data at or before as_of.
    Rows where the future bar does not exist are silently skipped.

    Returns
    -------
    pd.DataFrame
        Long-form with columns [as_of, ticker, horizon, fwd_ret].
    """
    idx = eur_close.index
    rows = []
    for asof in asof_dates:
        if asof not in idx:
            continue
        i = idx.get_loc(asof)
        base = eur_close.iloc[i]
        for h in horizons:
            j = i + h
            if j >= len(idx):
                continue
            fr = (eur_close.iloc[j] / base) - 1.0
            for tkr, r in fr.dropna().items():
                rows.append(
                    {
                        "as_of": asof,
                        "ticker": tkr,
                        "horizon": int(h),
                        "fwd_ret": float(r),
                    }
                )
    return pd.DataFrame(rows, columns=["as_of", "ticker", "horizon", "fwd_ret"])


def demean_by_date(fwd: pd.DataFrame) -> pd.DataFrame:
    """Add net_alpha = fwd_ret minus per-(as_of, horizon) cross-sectional mean."""
    out = fwd.copy()
    out["net_alpha"] = out.groupby(["as_of", "horizon"])["fwd_ret"].transform(
        lambda s: s - s.mean()
    )
    return out
