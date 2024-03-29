const executeBtn = document.getElementById('executeBtn');
const outputContainer = document.getElementById('outputContainer');
const spinner = document.getElementById('spinner');
let eventSource;

executeBtn.addEventListener('click', () => {
    const captchaType = document.getElementById('captchaTypeSelect').value;
    // Clear output container
    outputContainer.innerText = 'Executing Python code...\n';
    spinner.style.display = 'block';

    // Close the existing event source if it exists
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }

    // Create a new event source
    eventSource = new EventSource('/attackPanel/streamOutput?captchaType='+captchaType.toString());

    // Event listener for receiving messages from server
    eventSource.onmessage = function(event) {
        // Append received data to output container
        outputContainer.innerText += event.data + '\n';
        // Automatically scroll to the bottom of the container
        outputContainer.scrollTop = outputContainer.scrollHeight;

        if (event.data.trim() === 'End of data') {
            // Close the event source
            eventSource.close();
        }
    };

    // Event listener for when the connection is closed
    eventSource.onclose = function() {
        console.log('Connection closed.');
        // Hide spinner
        spinner.style.display = 'none';
    };

    // Event listener for errors
    eventSource.onerror = function(error) {
        console.error('Error occurred:', error);
    };
});


// Close the event source connection when leaving the page
window.onbeforeunload = function() {
    if (eventSource) {
        eventSource.close();
    }
};