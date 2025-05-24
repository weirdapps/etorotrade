"""
HTML templates for Yahoo Finance data presentation.

This module provides template strings and utilities for generating HTML content
from financial data. Templates are organized by category (market, portfolio, etc.)
and can be customized with parameters.
"""

from typing import Any, Dict

from yahoofinance.core.logging import get_logger


logger = get_logger(__name__)


def get_template(template_key: str, default: str = "") -> str:
    """
    Get a template by key from the Templates class.

    Args:
        template_key: The key of the template to get
        default: Default value if the template key doesn't exist

    Returns:
        Template string
    """
    if hasattr(Templates, template_key):
        return getattr(Templates, template_key)
    else:
        return default


class Templates:
    """Container for HTML templates"""

    # Base HTML document template
    BASE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="styles.css">
    {extra_head}
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        {content}
    </div>
    <script src="script.js"></script>
    {extra_scripts}
</body>
</html>
"""

    # Dashboard container template
    DASHBOARD_CONTAINER = """
<div class="dashboard">
    {sections}
</div>
"""

    # Dashboard section template
    DASHBOARD_SECTION = """
<div class="section" style="width: {width}">
    <h2>{title}</h2>
    <div class="metrics-grid" style="grid-template-columns: repeat({columns}, 1fr)">
        {metrics}
    </div>
</div>
"""

    # Metric card template
    METRIC_CARD = """
<div class="metric-card">
    <div class="metric-label">{label}</div>
    <div class="metric-value {color}">{value}</div>
</div>
"""

    # Table container template
    TABLE_CONTAINER = """
<div class="table-container">
    {table}
</div>
"""

    # Chart container template
    CHART_CONTAINER = """
<div class="chart-container" id="{chart_id}">
    <h2>{title}</h2>
    <canvas id="{canvas_id}" width="800" height="400"></canvas>
</div>
"""

    # Chart script template (using Chart.js)
    CHART_SCRIPT = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    const ctx = document.getElementById('{canvas_id}').getContext('2d');
    new Chart(ctx, {
        type: '{chart_type}',
        data: {
            labels: {labels},
            datasets: [{
                label: '{dataset_label}',
                data: {data},
                backgroundColor: {colors},
                borderColor: {border_colors},
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: {begin_at_zero}
                }
            }
        }
    });
});
</script>
"""

    # Default CSS for styling
    DEFAULT_CSS = """/* Default styles for financial dashboards */
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f5f5f7;
    color: #333;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

h1 {
    color: #333;
    text-align: center;
    margin-bottom: 30px;
    font-weight: 500;
}

h2 {
    color: #444;
    font-weight: 500;
    margin-bottom: 20px;
    font-size: 1.5rem;
}

.dashboard {
    display: flex;
    flex-direction: column;
    gap: 30px;
}

.section {
    background-color: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.metrics-grid {
    display: grid;
    grid-gap: 15px;
}

.metric-card {
    background-color: #f9f9f9;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    transition: transform 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.metric-label {
    font-size: 0.9rem;
    color: #777;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-value {
    font-size: 1.4rem;
    font-weight: 600;
    color: #333;
}

.metric-value.positive {
    color: #34c759;
}

.metric-value.negative {
    color: #ff3b30;
}

/* Table styling */
.table-container {
    overflow-x: auto;
    margin-top: 20px;
}

.stock-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.95rem;
}

.stock-table th {
    background-color: #f2f2f2;
    color: #555;
    font-weight: 600;
    text-align: left;
    padding: 12px 15px;
    border-bottom: 2px solid #ddd;
}

.stock-table td {
    padding: 10px 15px;
    border-bottom: 1px solid #eee;
}

.stock-table tr:hover {
    background-color: #f9f9f9;
}

/* Chart styling */
.chart-container {
    background-color: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    margin-bottom: 30px;
}

/* Responsive design */
@media (max-width: 768px) {
    .metrics-grid {
        grid-template-columns: repeat(2, 1fr) !important;
    }
    
    .metric-card {
        padding: 12px;
    }
    
    .metric-value {
        font-size: 1.2rem;
    }
}

@media (max-width: 480px) {
    .metrics-grid {
        grid-template-columns: 1fr !important;
    }
}"""

    # Default JavaScript
    DEFAULT_JS = """// Dashboard functionality
document.addEventListener('DOMContentLoaded', function() {
    // Add any interactive behaviors here
    console.log('Dashboard loaded');
    
    // Example: Add click animation to metric cards
    const metricCards = document.querySelectorAll('.metric-card');
    metricCards.forEach(card => {
        card.addEventListener('click', function() {
            this.style.transform = 'scale(0.98)';
            setTimeout(() => {
                this.style.transform = '';
            }, 150);
        });
    });
});"""


class TemplateEngine:
    """Engine for rendering templates with context data"""

    def __init__(self):
        """Initialize template engine"""
        self.templates = Templates()

    def render_base_html(
        self, title: str, content: str, extra_head: str = "", extra_scripts: str = ""
    ) -> str:
        """
        Render base HTML document.

        Args:
            title: Page title
            content: Main content for the page
            extra_head: Additional head content (CSS, meta tags, etc.)
            extra_scripts: Additional script tags

        Returns:
            Complete HTML document as string
        """
        return self.templates.BASE_HTML.format(
            title=title, content=content, extra_head=extra_head, extra_scripts=extra_scripts
        )

    def render_dashboard(self, sections: list) -> str:
        """
        Render dashboard container with sections.

        Args:
            sections: List of rendered section HTML strings

        Returns:
            Dashboard HTML as string
        """
        sections_html = "\n".join(sections)
        return self.templates.DASHBOARD_CONTAINER.format(sections=sections_html)

    def render_section(
        self, title: str, metrics: list, columns: int = 4, width: str = "100%"
    ) -> str:
        """
        Render dashboard section.

        Args:
            title: Section title
            metrics: List of rendered metric HTML strings
            columns: Number of columns in the grid
            width: Width of the section

        Returns:
            Section HTML as string
        """
        metrics_html = "\n".join(metrics)
        return self.templates.DASHBOARD_SECTION.format(
            title=title, metrics=metrics_html, columns=columns, width=width
        )

    def render_metric(self, label: str, value: str, color: str = "normal") -> str:
        """
        Render metric card.

        Args:
            label: Metric label
            value: Metric value
            color: Color class for the value

        Returns:
            Metric card HTML as string
        """
        return self.templates.METRIC_CARD.format(label=label, value=value, color=color)

    def render_table(self, table_html: str) -> str:
        """
        Render table container.

        Args:
            table_html: HTML table content

        Returns:
            Table container HTML as string
        """
        return self.templates.TABLE_CONTAINER.format(table=table_html)

    def render_chart(self, chart_id: str, title: str, canvas_id: str = "") -> str:
        """
        Render chart container.

        Args:
            chart_id: ID for the chart container
            title: Chart title
            canvas_id: ID for the canvas element (defaults to chart_id + "_canvas")

        Returns:
            Chart container HTML as string
        """
        canvas_id = canvas_id or f"{chart_id}_canvas"
        return self.templates.CHART_CONTAINER.format(
            chart_id=chart_id, title=title, canvas_id=canvas_id
        )

    def render_chart_script(
        self,
        canvas_id: str,
        chart_type: str,
        labels: list,
        data: list,
        dataset_label: str,
        colors: list = None,
        border_colors: list = None,
        begin_at_zero: bool = True,
    ) -> str:
        """
        Render chart script.

        Args:
            canvas_id: ID of the canvas element
            chart_type: Type of chart (bar, line, pie, etc.)
            labels: List of labels for data points
            data: List of data values
            dataset_label: Label for the dataset
            colors: List of background colors
            border_colors: List of border colors
            begin_at_zero: Whether y-axis should begin at zero

        Returns:
            Chart script HTML as string
        """
        # Default colors if not provided
        if colors is None:
            colors = [
                "rgba(75, 192, 192, 0.2)",
                "rgba(54, 162, 235, 0.2)",
                "rgba(255, 206, 86, 0.2)",
                "rgba(255, 99, 132, 0.2)",
                "rgba(153, 102, 255, 0.2)",
            ]
            # Repeat colors if needed
            colors = [colors[i % len(colors)] for i in range(len(data))]

        if border_colors is None:
            border_colors = [
                "rgba(75, 192, 192, 1)",
                "rgba(54, 162, 235, 1)",
                "rgba(255, 206, 86, 1)",
                "rgba(255, 99, 132, 1)",
                "rgba(153, 102, 255, 1)",
            ]
            # Repeat colors if needed
            border_colors = [border_colors[i % len(border_colors)] for i in range(len(data))]

        return self.templates.CHART_SCRIPT.format(
            canvas_id=canvas_id,
            chart_type=chart_type,
            labels=str(labels),
            data=str(data),
            dataset_label=dataset_label,
            colors=str(colors),
            border_colors=str(border_colors),
            begin_at_zero=str(begin_at_zero).lower(),
        )

    def get_default_css(self) -> str:
        """Get default CSS styles"""
        return self.templates.DEFAULT_CSS

    def get_default_js(self) -> str:
        """Get default JavaScript"""
        return self.templates.DEFAULT_JS
