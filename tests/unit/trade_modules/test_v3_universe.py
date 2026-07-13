import pandas as pd

from trade_modules.v3.universe import load_universe, parse_cap


def test_parse_cap_suffixes():
    assert parse_cap("3.5T") == 3.5e12
    assert parse_cap("800B") == 800e9
    assert parse_cap("1.2M") == 1.2e6
    assert pd.isna(parse_cap("--"))


def test_load_universe_filters(tmp_path):
    csv = tmp_path / "etoro.csv"
    pd.DataFrame(
        {
            "TKR": ["AAPL", "PENNY", "SMALL", "7203.T", "MSFT"],
            "PRC": [200.0, 0.5, 50.0, 3000.0, 400.0],
            "CAP": ["3.5T", "10B", "100M", "40T", "2.5T"],
        }
    ).to_csv(csv, index=False)
    u = load_universe(str(csv), min_price=1.0, min_cap_usd=5e8)
    assert u == ["AAPL", "MSFT"]  # PENNY (price), SMALL (cap), 7203.T (non-USD) excluded
