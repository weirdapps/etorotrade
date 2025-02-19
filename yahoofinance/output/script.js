function updateColors() {
    const elements = document.querySelectorAll('[data-value]');
    elements.forEach(el => {
        const value = parseFloat(el.getAttribute('data-value'));
        if (value > 0) {
            el.classList.add('text-green-600');
        } else if (value < 0) {
            el.classList.add('text-red-600');
        } else {
            el.classList.add('text-gray-600');
        }
    });
}
document.addEventListener('DOMContentLoaded', updateColors);