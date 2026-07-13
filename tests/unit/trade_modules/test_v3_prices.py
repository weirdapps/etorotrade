import numpy as np
import pandas as pd

from trade_modules.v3.prices import load_eur_close, to_eur


def test_to_eur_divides_by_usd_per_eur():
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    usd = pd.DataFrame({"AAPL": [110.0, 121.0, 132.0]}, index=idx)
    eurusd = pd.Series([1.10, 1.10, 1.10], index=idx)  # USD per EUR
    eur = to_eur(usd, eurusd)
    assert np.allclose(eur["AAPL"].values, [100.0, 110.0, 120.0])


def test_load_eur_close_uses_fetch_seam():
    idx = pd.date_range("2026-01-01", periods=2, freq="D")

    def fake_fetch(tickers, period="2y"):
        if tickers == ["EURUSD=X"]:
            return pd.DataFrame({"EURUSD=X": [1.25, 1.25]}, index=idx)
        return pd.DataFrame({"AAPL": [125.0, 250.0]}, index=idx)

    eur = load_eur_close(["AAPL"], fetch=fake_fetch)
    assert np.allclose(eur["AAPL"].values, [100.0, 200.0])
