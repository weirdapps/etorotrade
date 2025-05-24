import unittest

from yahoofinance.presentation.templates import TemplateEngine, Templates


class TestTemplates(unittest.TestCase):
    def setUp(self):
        """Set up test environment with template engine."""
        self.engine = TemplateEngine()
        self.templates = Templates()

    def test_template_constants(self):
        """Test that template constants are available."""
        self.assertIn("<!DOCTYPE html>", self.templates.BASE_HTML)
        self.assertIn('<div class="dashboard">', self.templates.DASHBOARD_CONTAINER)
        self.assertIn('<div class="section"', self.templates.DASHBOARD_SECTION)
        self.assertIn('<div class="metric-card">', self.templates.METRIC_CARD)
        self.assertIn('<div class="table-container">', self.templates.TABLE_CONTAINER)
        self.assertIn('<div class="chart-container"', self.templates.CHART_CONTAINER)

    def test_render_metric(self):
        """Test rendering metric card."""
        # Test with normal value
        html = self.engine.render_metric("Test Label", "10.5%")
        self.assertIn('<div class="metric-label">Test Label</div>', html)
        self.assertIn('<div class="metric-value normal">10.5%</div>', html)

        # Test with positive value
        html = self.engine.render_metric("Test Label", "10.5%", "positive")
        self.assertIn('<div class="metric-value positive">10.5%</div>', html)

        # Test with negative value
        html = self.engine.render_metric("Test Label", "-5.2%", "negative")
        self.assertIn('<div class="metric-value negative">-5.2%</div>', html)

    def test_render_section(self):
        """Test rendering dashboard section with metrics."""
        # Create some metrics
        metrics = [
            self.engine.render_metric("Metric 1", "10.5%", "positive"),
            self.engine.render_metric("Metric 2", "-5.2%", "negative"),
        ]

        # Test default 4-column grid
        html = self.engine.render_section("Test Section", metrics)
        self.assertIn("<h2>Test Section</h2>", html)
        self.assertIn("grid-template-columns: repeat(4, 1fr)", html)
        self.assertIn("width: 100%", html)

        # Test custom column count
        html = self.engine.render_section("Test Section", metrics, columns=3, width="80%")
        self.assertIn("grid-template-columns: repeat(3, 1fr)", html)
        self.assertIn("width: 80%", html)

    def test_render_base_html(self):
        """Test rendering base HTML document."""
        title = "Test Page"
        content = "<div>Test Content</div>"
        extra_head = "<style>.custom { color: red; }</style>"
        extra_scripts = '<script>console.log("test");</script>'

        html = self.engine.render_base_html(title, content, extra_head, extra_scripts)

        # Test basic structure
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn('<html lang="en">', html)
        self.assertIn("<head>", html)
        self.assertIn("<body>", html)

        # Test title and content
        self.assertIn(f"<title>{title}</title>", html)
        self.assertIn(content, html)

        # Test extra head and scripts
        self.assertIn(extra_head, html)
        self.assertIn(extra_scripts, html)

    def test_render_dashboard(self):
        """Test rendering complete dashboard with sections."""
        # Create metrics
        metrics1 = [
            self.engine.render_metric("Metric 1", "10.5%", "positive"),
            self.engine.render_metric("Metric 2", "-5.2%", "negative"),
        ]

        metrics2 = [
            self.engine.render_metric("Metric 3", "$100.50"),
            self.engine.render_metric("Metric 4", "N/A"),
        ]

        # Create sections
        section1 = self.engine.render_section("Market Metrics", metrics1, columns=2)
        section2 = self.engine.render_section("Portfolio Metrics", metrics2, columns=2)

        # Render dashboard
        dashboard = self.engine.render_dashboard([section1, section2])

        # Verify dashboard structure
        self.assertIn('<div class="dashboard">', dashboard)
        self.assertIn("<h2>Market Metrics</h2>", dashboard)
        self.assertIn("<h2>Portfolio Metrics</h2>", dashboard)
        self.assertIn('<div class="metric-value positive">10.5%</div>', dashboard)
        self.assertIn('<div class="metric-value negative">-5.2%</div>', dashboard)
        self.assertIn('<div class="metric-value normal">$100.50</div>', dashboard)

    def test_render_chart(self):
        """Test rendering chart container and script."""
        # Test chart container
        chart_html = self.engine.render_chart("test-chart", "Test Chart")
        self.assertIn('<div class="chart-container" id="test-chart">', chart_html)
        self.assertIn("<h2>Test Chart</h2>", chart_html)
        self.assertIn('<canvas id="test-chart_canvas"', chart_html)

        # Skip chart script test since it uses private implementation details
        # We're only testing the public API of the template engine

    def test_default_styles(self):
        """Test default CSS and JavaScript are available."""
        css = self.engine.get_default_css()
        js = self.engine.get_default_js()

        self.assertIn("body {", css)
        self.assertIn(".metric-value.positive {", css)
        self.assertIn(".metric-value.negative {", css)

        self.assertIn("document.addEventListener('DOMContentLoaded'", js)
        self.assertIn("console.log('Dashboard loaded');", js)


if __name__ == "__main__":
    unittest.main()
