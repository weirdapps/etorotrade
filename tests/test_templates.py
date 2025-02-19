import unittest
from yahoofinance.templates import metric_item, metrics_grid, generate_html

class TestTemplates(unittest.TestCase):
    def test_metric_item(self):
        """Test metric item HTML generation with accessibility features."""
        # Test with numeric value
        html = metric_item('test_id', '10.5%', 'Test Label')
        self.assertIn('id="test_id"', html)
        self.assertIn('data-value="10.5"', html)
        self.assertIn('class="text-3xl font-bold"', html)
        self.assertIn('class="text-base font-semibold"', html)
        self.assertIn('>Test Label<', html)
        self.assertIn('role="article"', html)
        self.assertIn('tabindex="0"', html)
        self.assertIn('aria-label="Test Label: 10.5%"', html)
        
        # Test with negative value
        html = metric_item('test_id', '-5.2%', 'Test Label')
        self.assertIn('data-value="-5.2"', html)
        self.assertIn('>-5.2%<', html)
        self.assertIn('aria-label="Test Label: -5.2%"', html)
        
        # Test with currency value
        html = metric_item('test_id', '$100.50', 'Test Label')
        self.assertIn('data-value="100.50"', html)
        self.assertIn('>$100.50<', html)
        self.assertIn('aria-label="Test Label: $100.50"', html)

    def test_metrics_grid(self):
        """Test metrics grid HTML generation with semantic structure."""
        metrics = [
            {'id': 'metric1', 'value': '10.5%', 'label': 'DJI30 (2025-02-07 to 2025-02-14)'},
            {'id': 'metric2', 'value': '-5.2%', 'label': 'SP500 (2025-02-07 to 2025-02-14)'},
            {'id': 'metric3', 'value': '$100.50', 'label': 'NQ100 (2025-02-07 to 2025-02-14)'},
            {'id': 'metric4', 'value': 'N/A', 'label': 'VIX (2025-02-07 to 2025-02-14)'}
        ]
        
        # Test default 4-column grid
        html = metrics_grid('Test Grid', metrics)
        self.assertIn('<h2 id="test-grid-title">Test Grid</h2>', html)
        self.assertIn('class="grid grid-4x1"', html)
        self.assertIn('width: 800px', html)
        self.assertIn('role="group"', html)
        self.assertIn('aria-labelledby="test-grid-title"', html)
        
        # Verify label structure
        self.assertIn('class="text-base font-semibold">DJI30<', html)
        self.assertIn('class="date-range">(2025-02-07 to 2025-02-14)</div>', html)
        
        # Test custom column count
        html = metrics_grid('Test Grid', metrics, columns=5)
        self.assertIn('class="grid grid-5x1"', html)
        
        # Test custom width
        html = metrics_grid('Test Grid', metrics, width='900px')
        self.assertIn('width: 900px', html)
        
        # Test with explicit date range
        html = metrics_grid('Test Grid', metrics, date_range='(Custom Range)')
        self.assertIn('class="date-range">(Custom Range)</div>', html)

    def test_generate_html(self):
        """Test complete HTML document generation with metadata and accessibility."""
        title = 'Test Page'
        content = '<div>Test Content</div>'
        custom_styles = '.custom { color: red; }'
        description = 'Test page description'
        
        html = generate_html(title, content, custom_styles, description)
        
        # Test basic structure
        self.assertIn('<!DOCTYPE html>', html)
        self.assertIn('<html lang="en">', html)
        self.assertIn('<head>', html)
        self.assertIn('<body>', html)
        self.assertIn('<main role="main"', html)
        
        # Test metadata
        self.assertIn(f'<title>{title}</title>', html)
        self.assertIn(f'<meta name="description" content="{description}"', html)
        self.assertIn('<meta charset="utf-8"', html)
        self.assertIn('<meta name="viewport"', html)
        
        # Test content and styles
        self.assertIn(content, html)
        self.assertIn(custom_styles, html)
        
        # Test without custom styles and description
        html = generate_html(title, content)
        self.assertNotIn('.custom { color: red; }', html)
        self.assertIn('content="Dashboard displaying test page metrics and statistics"', html)

    def test_template_integration(self):
        """Test integration of all template components with accessibility features."""
        metrics = [
            {'id': 'metric1', 'value': '10.5%', 'label': 'DJI30 (2025-02-07 to 2025-02-14)'},
            {'id': 'metric2', 'value': '-5.2%', 'label': 'SP500 (2025-02-07 to 2025-02-14)'}
        ]
        
        grid = metrics_grid('Test Grid', metrics)
        html = generate_html('Test Page', grid)
        
        # Verify complete structure
        self.assertIn('<!DOCTYPE html>', html)
        self.assertIn('<title>Test Page</title>', html)
        self.assertIn('<h2 id="test-grid-title">Test Grid</h2>', html)
        self.assertIn('role="main"', html)
        self.assertIn('role="group"', html)
        self.assertIn('role="article"', html)
        
        # Verify metric structure
        self.assertIn('id="metric1"', html)
        self.assertIn('data-value="10.5"', html)
        self.assertIn('class="text-base font-semibold">DJI30<', html)
        self.assertIn('class="date-range">(2025-02-07 to 2025-02-14)</div>', html)

if __name__ == '__main__':
    unittest.main()