import pandas as pd

from trade_modules.v3.fetch import robust_fetch_prices


def to_eur(usd_close: pd.DataFrame, eurusd: pd.Series) -> pd.DataFrame:
    fx = eurusd.reindex(usd_close.index).ffill().bfill()
    return usd_close.div(fx, axis=0)


def load_eur_close(
    tickers: list[str], period: str = "5y", fetch=robust_fetch_prices
) -> pd.DataFrame:
    usd = fetch(list(tickers), period=period)
    fx_df = fetch(["EURUSD=X"], period=period)
    eurusd = fx_df["EURUSD=X"] if "EURUSD=X" in fx_df.columns else fx_df.iloc[:, 0]
    eur = to_eur(usd, eurusd)
    return eur.dropna(how="all")
