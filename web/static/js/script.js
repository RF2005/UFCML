// UFC Fight Predictor JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const loadingDiv = document.getElementById('loading');
    const resultsDiv = document.getElementById('results');
    const errorDiv = document.getElementById('error');
    const predictBtn = document.getElementById('predict-btn');

    // Handle form submission
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        makePrediction();
    });

    // Fighter name inputs - capitalize first letter of each word
    const fighterInputs = document.querySelectorAll('#fighter_a, #fighter_b');
    fighterInputs.forEach(input => {
        input.addEventListener('blur', function() {
            this.value = capitalizeWords(this.value);
        });
    });

    function capitalizeWords(str) {
        return str.split(' ').map(word =>
            word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
        ).join(' ');
    }

    function showLoading() {
        form.style.display = 'none';
        resultsDiv.style.display = 'none';
        errorDiv.style.display = 'none';
        loadingDiv.style.display = 'block';
        predictBtn.disabled = true;
    }

    function hideLoading() {
        loadingDiv.style.display = 'none';
        predictBtn.disabled = false;
    }

    function showError(message) {
        hideLoading();
        form.style.display = 'none';
        resultsDiv.style.display = 'none';
        errorDiv.style.display = 'block';
        document.getElementById('error-message').textContent = message;
    }

    function showResults(data) {
        hideLoading();
        form.style.display = 'none';
        errorDiv.style.display = 'none';
        resultsDiv.style.display = 'block';

        // Winner announcement
        document.getElementById('winner-name').textContent = data.predicted_winner;
        document.getElementById('confidence-value').textContent = data.confidence + '%';

        // Fighter results
        document.getElementById('fighter-a-name').textContent = data.fighter_a;
        document.getElementById('fighter-b-name').textContent = data.fighter_b;

        // Probability bars with animation
        const fighterABar = document.getElementById('fighter-a-bar');
        const fighterBBar = document.getElementById('fighter-b-bar');
        const fighterAProb = document.getElementById('fighter-a-prob');
        const fighterBProb = document.getElementById('fighter-b-prob');

        // Set probability text
        fighterAProb.textContent = data.fighter_a_probability + '%';
        fighterBProb.textContent = data.fighter_b_probability + '%';

        // Animate probability bars
        setTimeout(() => {
            fighterABar.style.width = data.fighter_a_probability + '%';
            fighterBBar.style.width = data.fighter_b_probability + '%';
        }, 100);

        // Fight details
        document.getElementById('result-weight-class').textContent =
            data.weight_class || 'Not specified';
        document.getElementById('result-title-fight').textContent =
            data.title_fight ? 'Yes' : 'No';

        // Highlight winner
        const winnerCard = data.predicted_winner === data.fighter_a ?
            document.getElementById('fighter-a-result') :
            document.getElementById('fighter-b-result');

        winnerCard.style.border = '2px solid #ff0000';
        winnerCard.style.background = 'linear-gradient(45deg, #1a0000, #2a0000)';
    }

    function makePrediction() {
        const formData = new FormData(form);

        // Validate fighter names
        const fighterA = formData.get('fighter_a').trim();
        const fighterB = formData.get('fighter_b').trim();

        if (!fighterA || !fighterB) {
            showError('Please enter both fighter names.');
            return;
        }

        if (fighterA.toLowerCase() === fighterB.toLowerCase()) {
            showError('Fighter names must be different.');
            return;
        }

        showLoading();

        // Make AJAX request
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showResults(data);
            } else {
                showError(data.error || 'Prediction failed. Please try again.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showError('Network error. Please check your connection and try again.');
        });
    }

    // Reset form function
    window.resetForm = function() {
        form.style.display = 'flex';
        resultsDiv.style.display = 'none';
        errorDiv.style.display = 'none';
        loadingDiv.style.display = 'none';

        // Reset form fields
        form.reset();

        // Remove winner highlighting
        const fighterResults = document.querySelectorAll('.fighter-result');
        fighterResults.forEach(result => {
            result.style.border = '1px solid #333333';
            result.style.background = '#1a1a1a';
        });

        // Reset probability bars
        document.getElementById('fighter-a-bar').style.width = '0%';
        document.getElementById('fighter-b-bar').style.width = '0%';

        // Focus on first input
        document.getElementById('fighter_a').focus();
    };

    // Add some interactive effects
    const inputs = document.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.parentElement.style.transform = 'scale(1.02)';
            this.parentElement.style.transition = 'transform 0.2s ease';
        });

        input.addEventListener('blur', function() {
            this.parentElement.style.transform = 'scale(1)';
        });
    });

    // Add hover effects to buttons
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.addEventListener('mouseenter', function() {
            if (!this.disabled) {
                this.style.transform = 'translateY(-2px)';
            }
        });

        button.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });

    // Focus on first input when page loads
    document.getElementById('fighter_a').focus();
});

// Add some UFC-style sound effects (optional)
function playBellSound() {
    // This would play a bell sound if audio files were added
    // const audio = new Audio('/static/sounds/bell.mp3');
    // audio.play();
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Press Enter on results to start new prediction
    if (e.key === 'Enter' && document.getElementById('results').style.display === 'block') {
        resetForm();
    }

    // Press Escape to reset form
    if (e.key === 'Escape') {
        resetForm();
    }
});