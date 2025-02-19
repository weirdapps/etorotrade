import requests
from bs4 import BeautifulSoup
import os
from tabulate import tabulate
from . import templates
from .utils import FormatUtils

# Constants for time periods
THIS_MONTH = 'This Month'
YEAR_TO_DATE = 'Year To Date'
TWO_YEARS = '2 Years'

def get_soup(url: str) -> BeautifulSoup:
    """
    Fetch and parse HTML content from a URL.
    
    Args:
        url (str): The URL to fetch data from
        
    Returns:
        BeautifulSoup: Parsed HTML content
        
    Raises:
        requests.exceptions.RequestException: If the request fails
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        # Don't specify Accept-Encoding to let requests handle it
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Connection': 'keep-alive',
    }
    
    session = requests.Session()
    try:
        # First attempt with default SSL verification
        response = session.get(url, headers=headers, verify=True, timeout=30)
        response.raise_for_status()
        
        # Force response encoding to UTF-8
        response.encoding = 'utf-8'
        return BeautifulSoup(response.text, "html.parser")
        
    except requests.exceptions.SSLError:
        try:
            # Second attempt with SSL verification disabled
            session.verify = False
            response = session.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Force response encoding to UTF-8
            response.encoding = 'utf-8'
            return BeautifulSoup(response.text, "html.parser")
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch data from {url} (SSL retry failed): {str(e)}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch data from {url}: {str(e)}")
    finally:
        session.close()

def format_percentage_value(value: str) -> str:
    """Format a percentage value with proper sign and decimals."""
    try:
        # Remove % symbol but keep signs
        clean_value = value.replace('%', '').strip()
        value_float = float(clean_value)
        # Keep original sign for negative values, add + for positive
        if value_float < 0:
            return f"{value_float:.2f}%"
        return f"+{value_float:.2f}%"
    except ValueError:
        return value.strip()

def extract_summary_data(soup) -> dict:
    """Extract summary metrics (TODAY, MTD, YTD, 2YR)."""
    data = {}
    summary_items = soup.select("div.relative div.flex.flex-col.items-center")
    
    if not summary_items:
        return data
        
    for item in summary_items:
        value_span = (item.find("span", class_="font-semibold text-green-600") or
                     item.find("span", class_="font-semibold text-red-600"))
        label_div = item.find("div", class_="text-sm text-slate-400")
        
        if label_div and value_span:
            label = label_div.text.strip()
            value = format_percentage_value(value_span.text.strip())
            data[label] = value
            
    return data

from typing import Optional, Tuple

def extract_metric(soup, label: str, contains_text: str) -> Optional[Tuple[str, str]]:
    """Extract a metric value given its label and containing text."""
    container = soup.find('h2',
                        class_=['font-semibold', 'text-slate-100'],
                        string=lambda s: contains_text in str(s))
    if container:
        value_span = container.find_next("span", class_="text-5xl")
        if value_span:
            return label, value_span.text.strip()
    return None

def extract_cash_percentage(soup) -> Optional[Tuple[str, str]]:
    """Extract cash percentage value."""
    cash_container = soup.select_one("div.relative.flex.justify-between.space-x-2:-soup-contains('Cash')")
    if cash_container:
        cash_value_span = cash_container.find("div", class_="font-medium")
        if cash_value_span:
            return "Cash", cash_value_span.text.strip()
    return None

def extract_data(soup):
    """Extract all metrics from the webpage."""
    data = extract_summary_data(soup)
    
    # Extract other metrics
    metrics = [
        ("Beta", "Beta"),
        ("Alpha", "Jensen's Alpha"),
        ("Sharpe", "Sharpe Ratio"),
        ("Sortino", "Sortino Ratio")
    ]
    
    for label, contains_text in metrics:
        result = extract_metric(soup, label, contains_text)
        if result:
            data[result[0]] = result[1]
    
    # Extract cash percentage
    cash_result = extract_cash_percentage(soup)
    if cash_result:
        data[cash_result[0]] = cash_result[1]
    
    return data

def update_html(data, html_path):
    """Update HTML file with the extracted data."""
    # Map the scraped data fields to portfolio.html fields
    field_mapping = {
        THIS_MONTH: [THIS_MONTH, 'MTD'],
        YEAR_TO_DATE: [YEAR_TO_DATE, 'YTD'],
        TWO_YEARS: [TWO_YEARS, '2YR'],
        'Beta': ['Beta'],
        'Sharpe': ['Sharpe'],
        'Cash': ['Cash']
    }
    
    # Create portfolio data using the mapping
    portfolio_data = {}
    for target_field, source_fields in field_mapping.items():
        # Try each possible source field
        for source in source_fields:
            if source in data:
                portfolio_data[target_field] = data[source]
                break
        # If no matching field found, use default
        if target_field not in portfolio_data:
            portfolio_data[target_field] = '0.00%' if target_field in [THIS_MONTH, YEAR_TO_DATE, TWO_YEARS, 'Cash'] else '0.00'
    
    # Create metrics dictionary for formatting
    metrics_dict = {}
    for key, value in portfolio_data.items():
        metrics_dict[key] = {
            'value': value,
            'label': key,
            'is_percentage': '%' in str(value)
        }
    
    # Format metrics using FormatUtils instance
    utils = FormatUtils()
    formatted_metrics = utils.format_market_metrics(metrics_dict)
    
    # Generate the HTML using FormatUtils
    sections = [{
        'title': "Portfolio Performance",
        'metrics': formatted_metrics,
        'columns': 3,
        'rows': 2,
        'width': "800px"
    }]
    html_content = utils.generate_market_html(
        title="Portfolio Performance",
        sections=sections
    )

    try:
        with open(html_path, 'w', encoding='utf-8') as file:
            file.write(html_content)
        print("\nHTML file updated successfully.")
    except IOError as e:
        print(f"\nError: Could not write to file {html_path}. {e}")

def color_value(value):
    """Color code the values for console output."""
    try:
        num = float(value.replace('%', '').replace('+', ''))
        if num > 0:
            return f"\033[92m{value}\033[0m"  # Green for positive
        elif num < 0:
            return f"\033[91m{value}\033[0m"  # Red for negative
        else:
            return f"\033[93m{value}\033[0m"  # Yellow for zero
    except ValueError:
        return value  # If conversion fails, return the original value

if __name__ == "__main__":
    try:
        # URL of the page to scrape
        url = "https://bullaware.com/etoro/plessas"
        
        # Get the parsed HTML
        soup = get_soup(url)
        
        # Extract data from the webpage
        data = extract_data(soup)
        
        # Define the path to the HTML file
        html_path = os.path.join(os.path.dirname(__file__), 'output', 'portfolio.html')

        # Update the HTML file with the extracted data
        if data:
            # Display the data in the console
            table = [[key, color_value(value)] for key, value in data.items()]
            print(tabulate(table, headers=["Metric", "Value"], tablefmt="fancy_grid", colalign=("left", "right")))

            # Update the HTML file
            update_html(data, html_path)
        else:
            print("No data found.")
            
    except Exception as e:
        print(f"Error: {str(e)}")