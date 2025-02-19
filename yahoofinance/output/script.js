function updateColors() {
    const elements = document.querySelectorAll('[data-value]');
    elements.forEach(el => {
        const value = parseFloat(el.getAttribute('data-value'));
        el.classList.remove('text-green-500', 'text-red-500', 'text-gray-500');
        
        if (value > 0) {
            el.classList.add('text-green-500');
            el.setAttribute('aria-label', `Positive value: ${el.textContent}`);
        } else if (value < 0) {
            el.classList.add('text-red-500');
            el.setAttribute('aria-label', `Negative value: ${el.textContent}`);
        } else {
            el.classList.add('text-gray-500');
            el.setAttribute('aria-label', `Neutral value: ${el.textContent}`);
        }
    });
}

function initializeMetrics() {
    // Add role and tabindex for accessibility
    const containers = document.querySelectorAll('.flex');
    containers.forEach(container => {
        container.setAttribute('role', 'article');
        container.setAttribute('tabindex', '0');
    });

    // Initialize colors
    updateColors();

    // Add hover effect listeners
    containers.forEach(container => {
        container.addEventListener('mouseenter', () => {
            container.style.transform = 'scale(1.02)';
        });
        container.addEventListener('mouseleave', () => {
            container.style.transform = 'scale(1)';
        });
    });
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', initializeMetrics);

// Update on dynamic content changes
const observer = new MutationObserver(updateColors);
observer.observe(document.body, { 
    childList: true, 
    subtree: true 
});