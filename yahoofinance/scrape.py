import requests
from bs4 import BeautifulSoup
import os
from tabulate import tabulate

# URL of the page to scrape
url = "https://bullaware.com/etoro/plessas"

# Send a GET request to the page
response = requests.get(url)

# Parse the page content with BeautifulSoup
soup = BeautifulSoup(response.content, "html.parser")

# Function to extract the data
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
    # Use find instead of select_one for better handling of special characters
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

# Function to update the HTML file
def update_html(data, html_path):
    # Read the HTML file
    with open(html_path, 'r') as file:
        html_content = file.read()

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Update the values in the HTML
    for key, value in data.items():
        element = soup.find(id=key)
        if element:
            element.string = value

    # Write the updated HTML back to the file
    with open(html_path, 'w') as file:
        file.write(str(soup))

# Function to color code the values for console output
def color_value(value):
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

# Main script execution
if __name__ == "__main__":
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
        print("\nHTML file updated successfully.")

    else:
        print("No data found.")