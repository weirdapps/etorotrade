"""HTML templates for generating market reports."""

BASE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
    <title>{title}</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet"/>
    <link href="styles.css" rel="stylesheet"/>
    {custom_styles}
</head>
<body>
    {content}
    <script src="script.js"></script>
</body>
</html>
"""

def metric_item(id: str, value: str, label: str) -> str:
    """Generate HTML for a metric item."""
    numeric_value = value.replace('%', '').replace('$', '').replace(',', '')
    return f"""<div class="flex">
        <div class="text-3xl font-bold" id="{id}" data-value="{numeric_value}">{value}</div>
        <div class="text-sm text-slate-400">{label}</div>
    </div>"""

def metrics_grid(title: str, metrics: list, columns: int = 3, rows: int = 2, width: str = "700px") -> str:
    """Generate HTML for a grid of metrics."""
    if columns == 2 and rows == 2:
        grid_class = "grid-2x2"
    elif rows == 2:
        grid_class = "grid-3x2"
    else:
        grid_class = f"grid-{columns}x1"
    metrics_html = "\n".join(metric_item(m['id'], m['value'], m['label']) for m in metrics)
    
    return f"""<div class="summary-container" style="width: {width}">
        <h2>{title}</h2>
        <div class="grid {grid_class}">
            {metrics_html}
        </div>
    </div>"""

def generate_html(title: str, content: str, custom_styles: str = "") -> str:
    """Generate complete HTML document."""
    custom_styles_html = f"<style>{custom_styles}</style>" if custom_styles else ""
    return BASE_TEMPLATE.format(
        title=title,
        content=content,
        custom_styles=custom_styles_html
    )