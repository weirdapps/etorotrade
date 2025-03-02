/**
 * Script for enhancing the display of financial metrics
 */
document.addEventListener('DOMContentLoaded', function() {
    // Color positive/negative values appropriately
    var metricValues = document.querySelectorAll('.metric-value');
    metricValues.forEach(function(element) {
        var value = element.textContent.trim();
        
        // Reset classes first
        element.classList.remove('positive', 'negative', 'normal');
        
        // Apply appropriate class based on the value's content
        if (value.includes('+')) {
            element.classList.add('positive');
        } else if (value.includes('-') && value !== '--') {
            element.classList.add('negative');
        } else {
            element.classList.add('normal');
        }
    });
    
    // Add subtle hover effect to cards
    var cards = document.querySelectorAll('.metric-card');
    cards.forEach(function(card) {
        card.addEventListener('mouseenter', function() {
            this.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.backgroundColor = 'rgba(255, 255, 255, 0.05)';
        });
    });
});