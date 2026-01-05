"""
CSS and JavaScript assets for HTML reports.

This module provides default styling and scripts for generated HTML reports.
"""

# Default CSS for financial dashboards
DEFAULT_CSS = """/* Default styles for financial dashboards */
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f8f9fa;
    color: #333;
    line-height: 1.6;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 30px 20px;
}

h1 {
    color: #1a365d;
    text-align: center;
    margin-bottom: 30px;
    font-weight: 600;
    font-size: 2rem;
    padding-bottom: 15px;
    border-bottom: 1px solid #e2e8f0;
}

h2.section-title {
    color: #2d3748;
    font-weight: 600;
    margin-bottom: 20px;
    font-size: 1.5rem;
    position: relative;
    padding-bottom: 8px;
}

h2.section-title:after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    width: 50px;
    height: 3px;
    background-color: #4a5568;
    border-radius: 3px;
}

.dashboard {
    display: flex;
    flex-direction: column;
    gap: 30px;
}

.section {
    background-color: white;
    border-radius: 12px;
    padding: 25px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}

.section:hover {
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
}

.metrics-grid {
    display: grid;
    grid-gap: 20px;
}

.metric-card {
    position: relative;
    background-color: #ffffff;
    border-radius: 10px;
    padding: 20px 15px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
    transition: all 0.3s ease;
    overflow: hidden;
    border: 1px solid #edf2f7;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 15px rgba(0, 0, 0, 0.08);
}

.metric-border {
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background-color: #e2e8f0;
}

.metric-border.positive-border {
    background-color: #38a169;
}

.metric-border.negative-border {
    background-color: #e53e3e;
}

.metric-label {
    font-size: 0.85rem;
    color: #718096;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    font-weight: 500;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #2d3748;
    margin-bottom: 5px;
}

.metric-value.positive {
    color: #38a169;
}

.metric-value.negative {
    color: #e53e3e;
}

/* Table styling */
.table-container {
    overflow-x: auto;
    margin-top: 20px;
}

.stock-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
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
}
"""

# Default JavaScript for dashboards
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
});
"""
