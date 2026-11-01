"""dtype checks that behave the same under pandas 2 and pandas 3."""

import pandas as pd


def is_text_series(series: pd.Series) -> bool:
    """True for a column holding text.

    pandas 2 stores strings as dtype ``object``; pandas 3 infers the dedicated
    ``str`` dtype instead, so a bare ``dtype == object`` check silently skips
    string columns there. Mixed columns stay ``object`` in both.
    """
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
