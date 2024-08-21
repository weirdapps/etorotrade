def format_rows(numbered_data):
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

def determine_color(row):
    
    def safe_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # target_percent_diff = safe_float(row.get('target_percent_diff'))
    expected_return = safe_float(row.get('expected_return'))
    num_targets = safe_float(row.get('num_targets'))
    analyst_rating = safe_float(row.get('analyst_rating'))
    total_recommendations = safe_float(row.get('total_recommendations'))

    # Conditions for green color
    is_green = ((expected_return is not None and expected_return > 15 and 
                num_targets is not None and num_targets > 2) and 
                (analyst_rating is not None and analyst_rating > 65 and 
                total_recommendations is not None and total_recommendations > 2))

    # Conditions for red color
    is_red = ((expected_return is not None and expected_return < 5 and 
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
    