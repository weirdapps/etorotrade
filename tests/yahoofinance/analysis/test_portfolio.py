import pytest
from bs4 import BeautifulSoup
from unittest.mock import patch, Mock
from yahoofinance.analysis.portfolio import (
    format_percentage_value,
    extract_summary_data,
    extract_metric,
    extract_cash_percentage,
    extract_data,
    color_value,
    update_html
)

@pytest.fixture
def sample_html():
    return """
    <div class="relative">
        <div class="flex flex-col items-center">
            <span class="font-semibold text-green-600">+5.23%</span>
            <div class="text-sm text-slate-400">TODAY</div>
        </div>
        <div class="flex flex-col items-center">
            <span class="font-semibold text-red-600">-2.15%</span>
            <div class="text-sm text-slate-400">MTD</div>
        </div>
        <div class="flex flex-col items-center">
            <span class="font-semibold text-green-600">+8.45%</span>
            <div class="text-sm text-slate-400">YTD</div>
        </div>
        <div class="flex flex-col items-center">
            <span class="font-semibold text-green-600">+15.30%</span>
            <div class="text-sm text-slate-400">2YR</div>
        </div>
    </div>
    <h2 class="font-semibold text-slate-100">Beta</h2>
    <span class="text-5xl">1.25</span>
    <h2 class="font-semibold text-slate-100">Jensen's Alpha</h2>
    <span class="text-5xl">0.45</span>
    <h2 class="font-semibold text-slate-100">Sharpe Ratio</h2>
    <span class="text-5xl">1.8</span>
    <h2 class="font-semibold text-slate-100">Sortino Ratio</h2>
    <span class="text-5xl">2.1</span>
    <div class="relative flex justify-between space-x-2">
        <div>Cash</div>
        <div class="font-medium">15.5%</div>
    </div>
    """

@pytest.fixture
def soup(sample_html):
    return BeautifulSoup(sample_html, 'html.parser')

def test_format_percentage_value():
    assert format_percentage_value("+5.23%") == "+5.23%"
    assert format_percentage_value("-2.15%") == "-2.15%"
    assert format_percentage_value("0.00%") == "+0.00%"
    assert format_percentage_value("invalid") == "invalid"

def test_extract_summary_data(soup):
    data = extract_summary_data(soup)
    assert data["TODAY"] == "+5.23%"
    assert data["MTD"] == "-2.15%"
    assert data["YTD"] == "+8.45%"
    assert data["2YR"] == "+15.30%"

def test_extract_metric(soup):
    beta_result = extract_metric(soup, "Beta", "Beta")
    assert beta_result == ("Beta", "1.25")

    alpha_result = extract_metric(soup, "Alpha", "Jensen's Alpha")
    assert alpha_result == ("Alpha", "0.45")

    sharpe_result = extract_metric(soup, "Sharpe", "Sharpe Ratio")
    assert sharpe_result == ("Sharpe", "1.8")

    sortino_result = extract_metric(soup, "Sortino", "Sortino Ratio")
    assert sortino_result == ("Sortino", "2.1")

def test_extract_cash_percentage(soup):
    result = extract_cash_percentage(soup)
    assert result == ("Cash", "15.5%")

def test_extract_data(soup):
    data = extract_data(soup)
    
    # Check summary data
    assert data["TODAY"] == "+5.23%"
    assert data["MTD"] == "-2.15%"
    assert data["YTD"] == "+8.45%"
    assert data["2YR"] == "+15.30%"
    
    # Check metrics
    assert data["Beta"] == "1.25"
    assert data["Alpha"] == "0.45"
    assert data["Sharpe"] == "1.8"
    assert data["Sortino"] == "2.1"
    
    # Check cash percentage
    assert data["Cash"] == "15.5%"

def test_color_value():
    assert "\033[92m" in color_value("+5.23%")  # Green for positive
    assert "\033[91m" in color_value("-2.15%")  # Red for negative
    assert "\033[93m" in color_value("0.00%")   # Yellow for zero
    assert "invalid" == color_value("invalid")   # No color for invalid

@patch('yahoofinance.analysis.portfolio.FormatUtils')
def test_update_html(mock_format_utils_class, tmp_path):
    html_file = tmp_path / "test.html"
    
    # Test data to update
    data = {
        "TODAY": "+5.23%",
        "MTD": "-2.15%",
        "YTD": "+8.45%",
        "2YR": "+15.30%",
        "Beta": "1.25",
        "Alpha": "0.45",
        "Sharpe": "1.8",
        "Sortino": "2.1"
    }
    
    # Set up the mock FormatUtils instance
    mock_utils = Mock()
    mock_utils.format_market_metrics = Mock(return_value=[
        {'id': 'TODAY', 'value': '+5.23%', 'label': 'Today'},
        {'id': 'MTD', 'value': '-2.15%', 'label': 'MTD'},
        {'id': 'YTD', 'value': '+8.45%', 'label': 'YTD'},
        {'id': '2YR', 'value': '+15.30%', 'label': '2YR'},
        {'id': 'Beta', 'value': '1.25', 'label': 'Beta'},
        {'id': 'Alpha', 'value': '0.45', 'label': 'Alpha'},
        {'id': 'Sharpe', 'value': '1.8', 'label': 'Sharpe'},
        {'id': 'Sortino', 'value': '2.1', 'label': 'Sortino'}
    ])
    mock_utils.generate_market_html = Mock(return_value="<html>Mocked HTML</html>")
    
    # Configure the class mock to return our instance mock
    mock_format_utils_class.return_value = mock_utils
    
    # Create a file for testing
    html_file.touch()
    
    # Call the function
    with patch('builtins.open'), patch('builtins.print'):
        update_html(data, str(html_file))
    
    # Verify the mocks were called correctly
    assert mock_format_utils_class.call_count == 1
    assert mock_utils.generate_market_html.call_count == 1
    assert "Portfolio Performance" in mock_utils.generate_market_html.call_args[1]['title']

def test_extract_data_empty_soup():
    empty_soup = BeautifulSoup("", 'html.parser')
    data = extract_data(empty_soup)
    assert isinstance(data, dict)
    assert len(data) == 0

def test_extract_data_missing_elements(soup):
    # Remove some elements to test robustness
    for element in soup.select("span.text-5xl"):
        element.decompose()
    
    data = extract_data(soup)
    
    # Summary data should still be present
    assert "TODAY" in data
    assert data["TODAY"] == "+5.23%"
    assert "MTD" in data
    assert data["MTD"] == "-2.15%"
    assert "YTD" in data
    assert data["YTD"] == "+8.45%"
    assert "2YR" in data
    assert data["2YR"] == "+15.30%"
    
    # Metrics should be missing
    assert "Beta" not in data
    assert "Alpha" not in data
    assert "Sharpe" not in data
    assert "Sortino" not in data