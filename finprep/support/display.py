import csv
from tabulate import tabulate

def format_value(value, format_spec, default='-'):
    """
    Formats a value according to the given format specification.
    Returns a default value if the value is None or cannot be formatted.
    """
    try:
        if value is None:
            return default
        if value == 0:
            return '0'
        if '.0f' in format_spec:
            return format_spec.format(int(value))
        return format_spec.format(float(value))
    except ValueError:
        return default

def get_institutional_change_display(institutional_change):
    """
    Returns the display string for institutional change based on its value.
    """
    if institutional_change is None:
        return '-'
    elif institutional_change > 100:
        return '> +100%'
    elif institutional_change < -100:
        return '< -100%'
    else:
        return f"{float(institutional_change):.2f}%"

def determine_color(row):
    """Determine the color for a row based on its values"""
    def safe_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    target_percent_diff = safe_float(row.get('target_percent_diff'))
    num_targets = safe_float(row.get('num_targets'))
    analyst_rating = safe_float(row.get('analyst_rating'))
    total_recommendations = safe_float(row.get('total_recommendations'))

    # Conditions for green color
    is_green = ((target_percent_diff is not None and target_percent_diff > 15 and 
                num_targets is not None and num_targets > 2) and 
                (analyst_rating is not None and analyst_rating > 65 and 
                total_recommendations is not None and total_recommendations > 2))

    # Conditions for red color
    is_red = ((target_percent_diff is not None and target_percent_diff < 5 and 
            num_targets is not None and num_targets > 2) or 
            (analyst_rating is not None and analyst_rating < 55 and 
            total_recommendations is not None and total_recommendations > 2))

    # Conditions for yellow color
    is_yellow = not is_green and not is_red and (
        (num_targets is not None and num_targets < 3) or 
        (total_recommendations is not None and total_recommendations < 3))

    # Determine the color
    if is_red:
        return '\033[91m'  # Red
    elif is_green:
        return '\033[92m'  # Green
    elif is_yellow:
        return '\033[93m'  # Yellow
    else:
        return '\033[0m'   # Default (No color)

def format_rows(numbered_data):
    """Format rows with color and handle None values"""
    rows = []
    for row_data, color in numbered_data:
        if color:
            row = [f"{color}{item}\033[0m" if item is not None else item for item in row_data]
        else:
            row = row_data
        rows.append(row)

    formatted_rows = [
        [
            str(item) if item is not None else '-'
            for item in row
        ]
        for row in rows
    ]
    return formatted_rows

def display_table(data):
    """
    Displays the table using the tabulate library, with formatted rows and columns.
    """
    numbered_data = []

    for i, row in enumerate(data):
        institutional_change_display = get_institutional_change_display(row.get('institutional_change'))

        row_data = [
            i + 1,  # Adding row index as the first element
            row.get('ticker', '-'),
            format_value(row.get('stock_price'), "{:.2f}"),
            format_value(row.get('dcf_price'), "{:.2f}"),
            format_value(row.get('dcf_percent_diff'), "{:.1f}"),
            format_value(row.get('target_consensus'), "{:.2f}"),
            format_value(row.get('target_percent_diff'), "{:.1f}"),
            row.get('num_targets', '-'),
            format_value(row.get('analyst_rating'), "{:.0f}%"),
            row.get('total_recommendations', '-'),
            format_value(row.get('expected_return'), "{:.2f}%"),
            row.get('financial_score', '-'),
            row.get('piotroski_score', '-'),
            format_value(row.get('pe_ratio_ttm'), "{:.1f}"),
            format_value(row.get('peg_ratio_ttm'), "{:.2f}"),
            format_value(row.get('buysell'), "{:.2f}%"),
            institutional_change_display,
            format_value(row.get('senate_sentiment'), "{:.0f}%")
        ]

        color = determine_color(row)
        numbered_data.append((row_data, color))

    headers = [
        "#", "Ticker", "Price", "P DCF", "% DCF", "Target", "% T", "# T", "Rate", "# R", "ER", "FinS", "Piotr", "PE", "PEG", "Inside", "Inst", "Senate"
    ]

    rows = format_rows(numbered_data)

    print(tabulate(rows, headers=headers, tablefmt="fancy_grid", colalign=("right", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right")))

def save_to_csv(filename, data):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        headers = [
            "#", "Ticker", "Price", "DCF Price", "DCF Diff %", "Target Price", "Target Diff %", "# Targets", "Consensus Rating", "# Ratings", "Expected Return", "Financial Score", "Piotroski Score", "PE ratio", "PEG ratio", "Insiders Change", "Institutional Change", "Senate Change"
        ]
        writer.writerow(headers)
        for i, row in enumerate(data):
            row_data = [
                i + 1,
                row.get('ticker', ''),
                f"{float(row['stock_price']):.2f}" if row.get('stock_price') not in [None, '-'] else '-',
                f"{float(row['dcf_price']):.2f}" if row.get('dcf_price') not in [None, '-'] else '-',
                f"{float(row['dcf_percent_diff']):.1f}" if row.get('dcf_percent_diff') not in [None, '-'] else '-',
                f"{float(row['target_consensus']):.2f}" if row.get('target_consensus') not in [None, '-'] else '-',
                f"{float(row['target_percent_diff']):.1f}" if row.get('target_percent_diff') not in [None, '-'] else '-',
                row.get('num_targets', ''),
                f"{int(row['analyst_rating']):.0f}" if row.get('analyst_rating') not in [None, '-'] else '-',
                row.get('total_recommendations', ''),
                f"{float(row['expected_return']):.2f}" if row.get('expected_return') not in [None, '-'] else '-',
                row.get('financial_score', ''),
                row.get('piotroski_score', ''),
                f"{float(row['pe_ratio_ttm']):.1f}" if row.get('pe_ratio_ttm') not in [None, '-'] else '-',
                f"{float(row['peg_ratio_ttm']):.2f}" if row.get('peg_ratio_ttm') not in [None, '-'] else '-',
                f"{float(row['buysell']):.2f}%" if row.get('buysell') not in [None, '-'] else '-',
                f"{float(row['institutional_change']):.2f}%" if row.get('institutional_change') not in [None, '-'] else '-',
                f"{int(row['senate_sentiment']):.0f}" if row.get('senate_sentiment') not in [None, '-'] else '-'
            ]
            writer.writerow(row_data)