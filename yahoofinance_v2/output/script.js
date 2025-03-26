// Dashboard functionality
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
