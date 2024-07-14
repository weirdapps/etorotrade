# support/display.py

from tabulate import tabulate
from support.row_format import format_rows, determine_color

def display_table(data):
    numbered_data = []
    for i, row in enumerate(data):
        institutional_change = row.get('institutional_change')
        if institutional_change and institutional_change > 100:
            institutional_change_display = '> +100%'
        elif institutional_change and institutional_change < -100:
            institutional_change_display = '< -100%'
        elif institutional_change is not None:
            institutional_change_display = f"{float(institutional_change):.2f}%"
        else:
            institutional_change_display = '-'

        def format_value(value, format_spec, default='-'):
            try:
                if value in [None, '-']:
                    return default
                if '.0f' in format_spec:
                    return format_spec.format(int(value))
                return format_spec.format(float(value))
            except ValueError:
                return default
        
        row_data = [
            i + 1,
            row.get('ticker', ''),
            format_value(row.get('stock_price'), "{:.2f}"),
            format_value(row.get('dcf_price'), "{:.2f}"),
            format_value(row.get('dcf_percent_diff'), "{:.1f}"),
            format_value(row.get('target_consensus'), "{:.2f}"),
            format_value(row.get('target_percent_diff'), "{:.1f}"),
            row.get('num_targets', ''),
            format_value(row.get('analyst_rating'), "{:.0f}"),
            row.get('total_recommendations', ''),
            format_value(row.get('expected_return'), "{:.2f}"),
            row.get('financial_score', ''),
            row.get('piotroski_score', ''),
            format_value(row.get('pe_ratio_ttm'), "{:.1f}"),
            format_value(row.get('peg_ratio_ttm'), "{:.2f}"),
            format_value(row.get('buysell'), "{:.2f}"),
            institutional_change_display,
            format_value(row.get('senate_sentiment'), "{:.0f}")
        ]
        color = determine_color(row)
        numbered_data.append((row_data, color))

    headers = [
        "#", "Ticker", "Price", "DCF P", "DCF %", "Target", "Target %", "# T", "Rating", "# R", "ER", "Score", "Piotr", "PE", "PEG", "Inside", "Institute", "Senate", 
    ]
    rows = format_rows(numbered_data)

    print(tabulate(rows, headers=headers, tablefmt="fancy_grid", colalign=("right", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right")))