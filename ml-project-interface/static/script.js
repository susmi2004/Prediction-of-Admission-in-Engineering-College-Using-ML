// User management functions
function isLoggedIn() {
    // Check if there's a user session on the server
    return document.body.classList.contains('logged-in');
}

function getCurrentUser() {
    // Get username from the DOM if available
    const welcomeMsg = document.getElementById('welcomeMsg');
    if (welcomeMsg) {
        const username = welcomeMsg.getAttribute('data-username');
        return { name: username };
    }
    return null;
}

// Function to show/hide sections with authentication check
function showSection(sectionName) {
    // Check if user is logged in for protected sections
    if ((sectionName === 'home' || sectionName === 'about') && !isLoggedIn()) {
        alert('Please login first');
        window.location.href = '/login';
        return;
    }

    // For client-side navigation (if needed)
    if (sectionName === 'login') {
        window.location.href = '/login';
    } else if (sectionName === 'signup') {
        window.location.href = '/signup';
    } else if (sectionName === 'home') {
        window.location.href = '/home';
    }
}

// Login form handling
const loginForm = document.getElementById('loginForm');
if (loginForm) {
    loginForm.addEventListener('submit', (e) => {
        // Let the form submit normally to the server
        // The server will handle authentication
    });
}

// Signup form handling
const signupForm = document.getElementById('signupForm');
if (signupForm) {
    signupForm.addEventListener('submit', (e) => {
        // Form validation only
        const password = document.getElementById('password').value;
        const confirmPassword = document.getElementById('confirm_password').value;

        if (password !== confirmPassword) {
            e.preventDefault(); // Prevent form submission
            alert('Passwords do not match!');
            return false;
        }
        
        // Let the form submit to the server for processing
        return true;
    });
}

// Prediction form handling
const predictionForm = document.getElementById('predictionForm');
if (predictionForm) {
    predictionForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            const resultBox = document.getElementById('resultBox');
            const resultMessage = document.getElementById('resultMessage');
            const collegeDetails = document.getElementById('collegeDetails');
            const collegeInfo = document.getElementById('collegeInfo');
            const alternativeBranches = document.getElementById('alternativeBranches');
            const alternativesInfo = document.getElementById('alternativesInfo');

            // Clear previous results
            resultBox.className = 'result-box';
            collegeDetails.style.display = 'none';
            alternativeBranches.style.display = 'none';

            if (result.status === 'success') {
                if (result.message === 'Eligible for preferred branch') {
                    resultBox.classList.add('success');
                    resultMessage.textContent = 'Congratulations! You are eligible for your preferred branch.';
                    
                    // Display college details if available
                    if (result.college_details && Object.keys(result.college_details).length > 0) {
                        collegeInfo.innerHTML = `
                            <p><strong>College Name:</strong> ${result.college_details.name || ''}</p>
                            <p><strong>Location:</strong> ${result.college_details.place || ''}</p>
                            <p><strong>District:</strong> ${result.college_details.district || ''}</p>
                            <p><strong>Fee:</strong> ₹${result.college_details.fee || ''}</p>
                        `;
                        collegeDetails.style.display = 'block';
                    }
                } else {
                    resultBox.classList.add('error');
                    resultMessage.textContent = 'Sorry, you are not eligible for your preferred branch.';
                    
                    // Display alternative branches if available
                    if (result.alternative_branches) {
                        alternativesInfo.textContent = result.alternative_branches;
                        alternativeBranches.style.display = 'block';
                    } else {
                        alternativesInfo.textContent = 'No alternative branches available';
                        alternativeBranches.style.display = 'block';
                    }
                }
            } else {
                resultBox.classList.add('error');
                resultMessage.textContent = `Error: ${result.message}`;
            }

            resultBox.style.display = 'block';
            resultBox.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            console.error('Error:', error);
            const resultBox = document.getElementById('resultBox');
            resultBox.className = 'result-box error';
            document.getElementById('resultMessage').textContent = 'An error occurred while processing your request.';
            resultBox.style.display = 'block';
        }
    });
}

// Logout function
function logout() {
    // Redirect to the logout route
    window.location.href = '/logout';
}

// Initialize the page
document.addEventListener('DOMContentLoaded', () => {
    // Check for flash messages and display them
    const flashMessages = document.querySelectorAll('.alert');
    if (flashMessages.length > 0) {
        setTimeout(() => {
            flashMessages.forEach(message => {
                message.style.opacity = '0';
                setTimeout(() => {
                    message.style.display = 'none';
                }, 500);
            });
        }, 5000);
    }
    
    // Add event listeners to any logout buttons
    const logoutButtons = document.querySelectorAll('.logout-btn');
    logoutButtons.forEach(button => {
        button.addEventListener('click', (e) => {
            e.preventDefault();
            logout();
        });
    });
});
