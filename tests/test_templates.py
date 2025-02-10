import unittest
from yahoofinance.templates import metric_item, metrics_grid, generate_html

class TestTemplates(unittest.TestCase):
    def test_metric_item(self):
        """Test metric item HTML generation."""
        # Test with positive numeric value
        html = metric_item('test_id', '10.5%', 'Test Label')
        self.assertIn('id="test_id"', html)
        self.assertIn('data-value="10.5"', html)
        self.assertIn('>10.5%<', html)
        self.assertIn('>Test Label<', html)
        self.assertIn('class="text-3xl font-bold"', html)
        self.assertIn('class="text-sm text-slate-400"', html)
        
        # Test with negative value
        html = metric_item('test_id', '-5.2%', 'Test Label')
        self.assertIn('data-value="-5.2"', html)
        self.assertIn('>-5.2%<', html)
        
        # Test with currency value
        html = metric_item('test_id', '$100.50', 'Test Label')
        self.assertIn('data-value="100.50"', html)
        self.assertIn('>$100.50<', html)
        
        # Test with zero value
        html = metric_item('test_id', '0.0%', 'Test Label')
        self.assertIn('data-value="0.0"', html)
        self.assertIn('>0.0%<', html)
        
        # Test with non-numeric value
        html = metric_item('test_id', 'N/A', 'Test Label')
        self.assertIn('data-value="N/A"', html)
        self.assertIn('>N/A<', html)

    def test_metrics_grid(self):
        """Test metrics grid HTML generation."""
        metrics = [
            {'id': 'metric1', 'value': '10.5%', 'label': 'Metric 1'},
            {'id': 'metric2', 'value': '-5.2%', 'label': 'Metric 2'},
            {'id': 'metric3', 'value': '$100.50', 'label': 'Metric 3'},
            {'id': 'metric4', 'value': 'N/A', 'label': 'Metric 4'}
        ]
        
        # Test default 4-column grid
        html = metrics_grid('Test Grid', metrics)
        self.assertIn('<h2>Test Grid</h2>', html)
        self.assertIn('class="grid grid-4x1"', html)
        self.assertIn('width: 700px', html)
        for metric in metrics:
            self.assertIn(f'id="{metric["id"]}"', html)
            self.assertIn(f'>{metric["value"]}<', html)
            self.assertIn(f'>{metric["label"]}<', html)
        
        # Test custom column count
        html = metrics_grid('Test Grid', metrics, columns=5)
        self.assertIn('class="grid grid-5x1"', html)
        
        # Test custom width
        html = metrics_grid('Test Grid', metrics, width='900px')
        self.assertIn('width: 900px', html)
        
        # Test with single metric
        html = metrics_grid('Test Grid', [metrics[0]])
        self.assertIn('id="metric1"', html)
        self.assertIn('>10.5%<', html)
        self.assertIn('>Metric 1<', html)

    def test_generate_html(self):
        """Test complete HTML document generation."""
        title = 'Test Page'
        content = '<div>Test Content</div>'
        custom_styles = '.custom { color: red; }'
        
        html = generate_html(title, content, custom_styles)
        
        # Test basic structure
        self.assertIn('<!DOCTYPE html>', html)
        self.assertIn('<html lang="en">', html)
        self.assertIn('<head>', html)
        self.assertIn('<body>', html)
        
        # Test title and content
        self.assertIn(f'<title>{title}</title>', html)
        self.assertIn(content, html)
        
        # Test custom styles
        self.assertIn(custom_styles, html)
        
        # Test required elements
        self.assertIn('<meta charset="utf-8"', html)
        self.assertIn('<meta content="width=device-width', html)
        self.assertIn('font-family: \'Roboto\'', html)
        
        # Test color scheme
        self.assertIn('background-color: #111827', html)
        self.assertIn('color: white', html)
        
        # Test JavaScript
        self.assertIn('function updateColors()', html)
        self.assertIn('document.querySelectorAll(\'[data-value]\')', html)
        
        # Test without custom styles
        html = generate_html(title, content)
        self.assertNotIn('.custom { color: red; }', html)

    def test_template_integration(self):
        """Test integration of all template components."""
        metrics = [
            {'id': 'metric1', 'value': '10.5%', 'label': 'Metric 1'},
            {'id': 'metric2', 'value': '-5.2%', 'label': 'Metric 2'}
        ]
        
        grid = metrics_grid('Test Grid', metrics)
        html = generate_html('Test Page', grid)
        
        # Verify complete structure
        self.assertIn('<!DOCTYPE html>', html)
        self.assertIn('<title>Test Page</title>', html)
        self.assertIn('<h2>Test Grid</h2>', html)
        self.assertIn('id="metric1"', html)
        self.assertIn('data-value="10.5"', html)
        self.assertIn('>10.5%<', html)
        self.assertIn('>Metric 1<', html)
        self.assertIn('updateColors();', html)

if __name__ == '__main__':
    unittest.main()