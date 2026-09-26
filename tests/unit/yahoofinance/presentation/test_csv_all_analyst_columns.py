"""
The all-analyst columns (%BA, #TA) must survive the CSV write path.

STANDARD_DISPLAY_COLUMNS lists them, but save_to_csv only keeps columns that
format_dataframe produced, and format_dataframe builds a fresh frame from its
column_mapping. A raw field missing from that mapping is dropped silently.
"""

import pandas as pd

from yahoofinance.presentation.console_modules.data_manager import save_to_csv
from yahoofinance.presentation.console_modules.table_renderer import (
    add_position_size_column,
    format_dataframe,
    sort_market_data,
)


def _row(**overrides):
    row = {
        "symbol": "AAA",
        "company": "Example Corp",
        "current_price": 100.0,
        "target_price": 120.0,
        "upside": 20.0,
        "analyst_count": 12,
        "total_ratings": 5,
        "buy_percentage": 80.0,
        "buy_percentage_all": 65.0,
        "total_ratings_all": 14,
        "market_cap": 5_000_000_000,
        "A": "E",
        "BS": "H",
    }
    row.update(overrides)
    return row


def test_format_dataframe_maps_all_analyst_fields():
    df = format_dataframe(pd.DataFrame([_row()]), truncate_name=False)

    # Formatted exactly like the %B / #A twins.
    assert df["%B"].iloc[0] == "80%"
    assert df["%BA"].iloc[0] == "65%"
    assert df["#TA"].iloc[0] == 14


def test_zero_all_analyst_buy_ratio_is_a_value_and_zero_ratings_is_unknown():
    row = _row(buy_percentage=0, total_ratings=0, buy_percentage_all=0, total_ratings_all=0)
    df = format_dataframe(pd.DataFrame([row]), truncate_name=False)

    # A genuine 0% buy ratio is a value. "--" reads as unknown, and a reader
    # that skips unknowns would let the name through a low-buy screen.
    assert df["%BA"].iloc[0] == "0%"
    # Zero ratings means no reading at all, so it stays unknown.
    assert df["#TA"].iloc[0] == "--"
    # The older twins keep their existing behaviour.
    assert df["%B"].iloc[0] == "--"
    assert df["#A"].iloc[0] == "--"


def test_missing_all_analyst_buy_ratio_stays_unknown():
    df = format_dataframe(pd.DataFrame([_row(buy_percentage_all=None)]), truncate_name=False)

    assert df["%BA"].iloc[0] == "--"


def test_written_csv_carries_zero_all_analyst_buy_ratio(tmp_path):
    save_to_csv(
        [_row(buy_percentage_all=0, total_ratings_all=0)],
        "etoro.csv",
        output_dir=str(tmp_path),
        _format_dataframe_fn=lambda df: format_dataframe(df, truncate_name=False),
        _add_position_size_fn=add_position_size_column,
        _sort_market_data_fn=sort_market_data,
    )

    written = pd.read_csv(tmp_path / "etoro.csv", keep_default_na=False)

    assert written["%BA"].iloc[0] == "0%"
    assert written["#TA"].iloc[0] == "--"


def test_format_dataframe_keeps_all_analyst_display_columns():
    raw = pd.DataFrame([{"TKR": "AAA", "%BA": 65.0, "#TA": 14}])
    df = format_dataframe(raw, truncate_name=False)

    assert df["%BA"].iloc[0] == "65%"
    assert df["#TA"].iloc[0] == 14


def test_written_etoro_csv_header_carries_all_analyst_columns(tmp_path):
    save_to_csv(
        [_row(), _row(symbol="BBB", company="Other Corp", market_cap=9_000_000_000)],
        "etoro.csv",
        output_dir=str(tmp_path),
        _format_dataframe_fn=lambda df: format_dataframe(df, truncate_name=False),
        _add_position_size_fn=add_position_size_column,
        _sort_market_data_fn=sort_market_data,
    )

    written = pd.read_csv(tmp_path / "etoro.csv")

    assert "%BA" in written.columns
    assert "#TA" in written.columns
    # Column order follows STANDARD_DISPLAY_COLUMNS: %BA after %B, #TA after #A.
    cols = list(written.columns)
    assert cols.index("%B") < cols.index("%BA")
    assert cols.index("#A") < cols.index("#TA")
    aaa = written.loc[written["TKR"] == "AAA"].iloc[0]
    assert aaa["%BA"] == "65%"
    assert aaa["#TA"] == 14
