// Auto-dismiss success messages after 3 seconds
(function() {
    const message = document.getElementById('success-message');
    if (message) {
        setTimeout(() => {
            message.style.transition = 'opacity 0.3s ease-out';
            message.style.opacity = '0';
            setTimeout(() => message.remove(), 300);
        }, 3000);
    }
    
    // Smooth status updates - prevent flickering
    document.body.addEventListener('htmx:beforeSwap', function(evt) {
        if (evt.detail.target?.id === 'status-info') {
            evt.detail.target.style.opacity = '0.98';
        }
    });
    
    document.body.addEventListener('htmx:afterSwap', function(evt) {
        if (evt.detail.target?.id === 'status-info') {
            requestAnimationFrame(() => {
                evt.detail.target.style.opacity = '1';
            });
        }
    });
})();
