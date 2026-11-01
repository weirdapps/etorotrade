"""Text columns must be treated as text under pandas 2 (dtype object) and pandas 3 (dtype str).

pandas 3 infers a column of strings as the new ``str`` dtype, so a bare
``dtype == object`` check silently skips it. These cover the call sites whose
fallback path would otherwise turn "10%" into NaN or skip formatting.
"""

import numpy as np
import pandas as pd
import pytest

from trade_modules.improvement_analyzer import ImprovementAnalyzer
from trade_modules.output_manager import _apply_color_to_dataframe
from trade_modules.portfolio_risk import PortfolioRiskAnalyzer
from yahoofinance.analysis.market import get_short_interest_condition


def test_improvement_analyzer_to_numeric_parses_percent_strings():
    series = pd.Series(["10%", "--", "2.5%"])
    result = ImprovementAnalyzer()._to_numeric(series)
    assert result.iloc[0] == pytest.approx(10.0)
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.5)


def test_short_interest_condition_parses_percent_strings():
    df = pd.DataFrame({"SI": ["1.5%", "8%", "nan"]})
    keep = get_short_interest_condition(df, short_interest_threshold=5.0, is_maximum=True)
    assert keep.tolist() == [True, False, True]


def test_color_applies_to_string_columns_only():
    df = pd.DataFrame({"TKR": ["AAPL"], "PRICE": [1.0]})
    result = _apply_color_to_dataframe(df, "\033[92m")
    assert result["TKR"].iloc[0].startswith("\033[92m")
    assert result["PRICE"].iloc[0] == 1.0


def test_portfolio_risk_parses_string_market_cap_weights():
    df = pd.DataFrame(
        {
            "ticker": ["AAPL", "JPM"],
            "sector": ["Technology", "Financial Services"],
            "region": ["US", "US"],
            "CAP": ["3.5T", "500B"],
        }
    )
    analyzer = PortfolioRiskAnalyzer()
    sectors = analyzer.get_sector_concentration(df)
    assert sectors["Technology"] > sectors["Financial Services"]


def test_portfolio_beta_weights_by_string_market_cap():
    df = pd.DataFrame({"ticker": ["A", "B"], "beta": [2.0, 1.0], "CAP": ["3T", "1T"]})
    beta = PortfolioRiskAnalyzer().calculate_portfolio_beta(df)
    assert beta == pytest.approx(1.75)
