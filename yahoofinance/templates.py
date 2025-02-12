"""HTML templates for generating market reports."""

BASE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
    <title>{title}</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet"/>
    <style>
        body {{
            background-color: #111827;
            color: white;
            font-family: 'Roboto', sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }}
        .container {{
            display: flex;
            flex-direction: column;
            gap: 2.5px;
        }}
        .summary-container {{
            background-color: #1f2937;
            padding: 5px 2.5px;
            border-radius: 4px;
            box-shadow: 0 0 5px rgba(0, 0, 0, 0.5);
            text-align: center;
        }}
        .summary-container h2 {{
            margin-top: 0;
            font-size: 24px;
            font-weight: normal;
            margin-bottom: 5px;
        }}
        .grid {{
            display: grid;
            gap: 2.5px;
        }}
        .grid-4x1 {{ grid-template-columns: repeat(4, 1fr); }}
        .grid-5x1 {{ grid-template-columns: repeat(5, 1fr); }}
        .grid-3x2 {{
            grid-template-columns: repeat(3, 1fr);
            grid-template-rows: repeat(2, 1fr);
            gap: 10px;
            aspect-ratio: 16/9;
            padding: 2.5px;
            width: 100%;
            max-width: 960px;
            margin: 0 auto;
        }}
        .grid-2x2 {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            grid-template-rows: repeat(2, 1fr);
            gap: 10px;
            aspect-ratio: 1;
            padding: 2.5px;
            width: 100%;
            max-width: 500px;
            margin: 0 auto;
        }}
        .grid-2x2 .flex {{
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }}
        .flex {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2.5px;
        }}
        .text-3xl {{ font-size: 1.875rem; }}
        .font-bold {{ font-weight: bold; }}
        .text-green-600 {{ color: #10b981; }}
        .text-red-600 {{ color: #ef4444; }}
        .text-gray-600 {{ color: #4b5563; }}
        .text-slate-400 {{ color: #94a3b8; }}
        .text-sm {{ font-size: 0.875rem; }}
        {custom_styles}
    </style>
</head>
<body>
    {content}
    <script>
        function updateColors() {{
            const elements = document.querySelectorAll('[data-value]');
            elements.forEach(el => {{
                const value = parseFloat(el.getAttribute('data-value'));
                if (value > 0) {{
                    el.classList.add('text-green-600');
                }} else if (value < 0) {{
                    el.classList.add('text-red-600');
                }} else {{
                    el.classList.add('text-gray-600');
                }}
            }});
        }}
        updateColors();
    </script>
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

def metrics_grid(title: str, metrics: list, columns: int = 4, rows: int = 1, width: str = "700px") -> str:
    """Generate HTML for a grid of metrics."""
    if rows == 2:
        grid_class = "grid-2x2" if columns == 2 else "grid-3x2"
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
    return BASE_TEMPLATE.format(
        title=title,
        content=content,
        custom_styles=custom_styles
    )