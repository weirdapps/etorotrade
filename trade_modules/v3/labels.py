import numpy as np
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


def cross_sectional_ic(scores: pd.DataFrame, fwd: pd.DataFrame, horizon: int) -> pd.Series:
    """Spearman rank IC per as_of date.

    Parameters
    ----------
    scores:
        Long-form DataFrame with columns [as_of, ticker, score].
    fwd:
        Long-form DataFrame with columns [as_of, ticker, horizon, fwd_ret].
    horizon:
        The forecast horizon (in bars) to select from fwd.

    Returns
    -------
    pd.Series
        Index = as_of dates; value = Spearman rank correlation between score
        and fwd_ret for that date.  Dates with fewer than 3 names are dropped
        (NaN-filtered).
    """
    f = fwd[fwd["horizon"] == horizon].merge(scores, on=["as_of", "ticker"], how="inner")

    def _ic(g):
        if len(g) < 3:
            return np.nan
        return g["score"].corr(g["fwd_ret"], method="spearman")

    return f.groupby("as_of").apply(_ic, include_groups=False).dropna()


def ic_summary(ic: pd.Series) -> dict:
    """Summarise an IC series.

    Parameters
    ----------
    ic:
        pd.Series of per-date Spearman IC values.

    Returns
    -------
    dict
        {n, mean_ic, t_stat, hit_rate}
    """
    n = int(len(ic))
    mean = float(ic.mean()) if n else float("nan")
    sd = float(ic.std(ddof=1)) if n > 1 else float("nan")
    t = mean / (sd / np.sqrt(n)) if (n > 1 and sd and sd > 0) else float("nan")
    hit = float((ic > 0).mean()) if n else float("nan")
    return {"n": n, "mean_ic": mean, "t_stat": t, "hit_rate": hit}
