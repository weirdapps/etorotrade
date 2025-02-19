"""HTML templates for generating market reports."""

BASE_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <meta name="description" content="{description}"/>
    <title>{title}</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;600;700&display=swap" rel="stylesheet"/>
    <link href="styles.css" rel="stylesheet"/>
    {custom_styles}
</head>
<body>
    <main role="main" class="container">{content}</main>
    <script src="script.js"></script>
</body>
</html>
'''

def metric_item(id: str, value: str, label: str) -> str:
    """Generate HTML for a metric item with improved accessibility."""
    numeric_value = value.replace('%', '').replace('$', '').replace(',', '')
    return f'''<article class="flex" role="article" tabindex="0">
        <div class="text-3xl font-bold" id="{id}" data-value="{numeric_value}" role="text" aria-label="{label}: {value}">{value}</div>
        <div class="metric-label">
            <div class="text-base font-semibold">{label}</div>
        </div>
    </article>'''

def metrics_grid(title: str, metrics: list, columns: int = 3, rows: int = 2, width: str = "800px", date_range: str = "") -> str:
    """Generate HTML for a grid of metrics with improved structure."""
    # Determine grid class based on metrics length and provided columns
    if len(metrics) == 4 and columns == 3:
        grid_class = "grid-4x1"
    elif columns != 3:
        grid_class = f"grid-{columns}x1"
    elif columns == 2 and rows == 2:
        grid_class = "grid-2x2"
    elif rows == 2:
        grid_class = "grid-3x2"
    else:
        grid_class = "grid-3x1"
    
    # Extract main label without date range
    metrics_html = []
    for m in metrics:
        label = m['label']
        if '(' in label and ')' in label:
            label = label.split('(', 1)[0].strip()
        metrics_html.append(metric_item(m['id'], m['value'], label))
    
    # Extract date range from first metric if not provided
    if not date_range and '(' in metrics[0]['label'] and ')' in metrics[0]['label']:
        date_range = f"({metrics[0]['label'].split('(', 1)[1]}"
    
    return f'''<section class="summary-container" style="width: {width}">
        <h2 id="{title.lower().replace(' ', '-')}-title">{title}</h2>
        <div class="grid {grid_class}" role="group" aria-labelledby="{title.lower().replace(' ', '-')}-title">
            {"\n".join(metrics_html)}
        </div>
        {f'<div class="date-range">{date_range}</div>' if date_range else ''}
    </section>'''

def generate_html(title: str, content: str, custom_styles: str = "", description: str = "") -> str:
    """Generate complete HTML document with improved metadata."""
    if not description:
        description = f"Dashboard displaying {title.lower()} metrics and statistics"
    
    custom_styles_html = f"<style>{custom_styles}</style>" if custom_styles else ""
    
    return BASE_TEMPLATE.format(
        title=title,
        description=description,
        content=content,
        custom_styles=custom_styles_html
    )