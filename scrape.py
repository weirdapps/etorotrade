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
def extract_data(soup):
    data = {}
    labels = ["TODAY", "MTD", "YTD", "1YR", "2YR"]  # "5YR" excluded
    
    # Extract summary data
    summary_items = soup.select("div.relative div.flex.flex-col.items-center")
    if summary_items:
        for index, item in enumerate(summary_items[:-1]):  # Exclude the last item
            value_span = item.find("span", class_="font-semibold text-green-600")
            if value_span is None:  # Handle case where negative values might have a different class
                value_span = item.find("span", class_="font-semibold text-red-600")
            label_div = item.find("div", class_="text-sm text-slate-400")
            if label_div and value_span:
                label = labels[index] if index < len(labels) else label_div.text.strip()
                value = value_span.text.strip()
                # Convert to float and format to 2 decimal places
                try:
                    value = float(value.replace('%', ''))
                    sign = "+" if value > 0 else ""
                    value = f"{sign}{value:.2f}%"
                except ValueError:
                    value = value_span.text.strip()  # If conversion fails, keep original text
                data[label] = value

    # Extract beta value
    beta_label = "Beta"
    beta_value_container = soup.select_one("h2.font-semibold.text-slate-100:-soup-contains('Beta')")
    if beta_value_container:
        beta_value_span = beta_value_container.find_next("span", class_="text-5xl")
        if beta_value_span:
            beta_value = beta_value_span.text.strip()
            data[beta_label] = beta_value

    # Extract Jensen's Alpha
    alpha_label = "Alpha"
    alpha_container = soup.select_one("h2.font-semibold.text-slate-100:-soup-contains(\"Jensen's Alpha\")")
    if alpha_container:
        alpha_span = alpha_container.find_next("span", class_="text-5xl")
        if alpha_span:
            alpha_value = alpha_span.text.strip()
            data[alpha_label] = alpha_value
            
    # Extract Sharpe Ratio
    sharpe_label = "Sharpe"
    sharpe_ratio_container = soup.select_one("h2.font-semibold.text-slate-100:-soup-contains('Sharpe Ratio')")
    if sharpe_ratio_container:
        sharpe_ratio_span = sharpe_ratio_container.find_next("span", class_="text-5xl")
        if sharpe_ratio_span:
            sharpe_ratio = sharpe_ratio_span.text.strip()
            data[sharpe_label] = sharpe_ratio

    # Extract Sortino Ratio
    sortino_label = "Sortino"
    sortino_ratio_container = soup.select_one("h2.font-semibold.text-slate-100:-soup-contains('Sortino Ratio')")
    if sortino_ratio_container:
        sortino_ratio_span = sortino_ratio_container.find_next("span", class_="text-5xl")
        if sortino_ratio_span:
            sortino_ratio = sortino_ratio_span.text.strip()
            data[sortino_label] = sortino_ratio

    # Extract Cash percentage
    cash_label = "Cash"
    cash_container = soup.select_one("div.relative.flex.justify-between.space-x-2:-soup-contains('Cash')")
    if cash_container:
        cash_value_span = cash_container.find("div", class_="font-medium")
        if cash_value_span:
            cash_value = cash_value_span.text.strip()
            data[cash_label] = cash_value

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
    html_path = os.path.join('finprep', 'output', 'portfolio.html')

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