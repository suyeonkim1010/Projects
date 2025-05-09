document.getElementById('contactForm').addEventListener('submit', async function(event) {
    event.preventDefault(); // Prevent default form submission behavior

    const form = event.target;
    const formData = new FormData(form); // Create FormData from form

    try {
        const response = await fetch(form.action, {
            method: form.method,
            body: formData, // Automatically sets correct headers
        });

        if (response.ok) {
            alert('Your message has been sent successfully!');
            form.reset(); // Clear the form on success
        } else {
            const errorText = await response.text(); // Fetch the response text for debugging
            console.error('Response error:', errorText);
            alert('There was a problem with your submission. Please try again.');
        }
    } catch (error) {
        console.error('Submission error:', error);
        alert('An error occurred. Please try again later.');
    }
});

