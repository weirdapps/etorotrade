import requests
from bs4 import BeautifulSoup
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
    labels = ["TODAY", "MTD", "YTD", "1YR", "2YR", "5YR"]
    
    # Extract summary data
    summary_items = soup.select("div.relative div.flex.flex-col.items-center")
    if summary_items:
        for index, item in enumerate(summary_items):
            value_span = item.find("span", class_="font-semibold text-green-600")
            label_div = item.find("div", class_="text-sm text-slate-400")
            if label_div and value_span:
                label = labels[index] if index < len(labels) else label_div.text.strip()
                value = value_span.text.strip()
                # Convert to float and format to 2 decimal places
                try:
                    value = float(value.replace('%', ''))
                    value = f"{value:.2f}%"
                except ValueError:
                    value = value_span.text.strip()  # If conversion fails, keep original text
                data[label] = value

    return data

# Function to color code the values
def color_value(value):
    try:
        num = float(value.replace('%', ''))
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
    data = extract_data(soup)
    if data:
        table = [[key, color_value(value)] for key, value in data.items()]
        print(tabulate(table, headers=["Time", "Change"], tablefmt="fancy_grid", colalign=("left", "right")))
    else:
        print("No data found")