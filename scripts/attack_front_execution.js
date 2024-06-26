import { extractInformation, updatePieCharts, displayMetricsTable, erasePreviousMetrics } from './statistics_dashboard.js';


const executeBtn = document.getElementById('executeBtn');
const outputContainer = document.getElementById('outputContainer');
const spinner = document.getElementById('spinner');

let eventSource;



window.onload = function() {
    var executeBtnWidth = document.getElementById("executeBtn").offsetWidth;
    document.getElementById("captchaTypeSelect").style.width = executeBtnWidth + "px";
    var executeBtnHeight = document.getElementById("executeBtn").offsetHeight;
    document.getElementById("captchaTypeSelect").style.height = executeBtnHeight + "px";
};


executeBtn.addEventListener('click', () => {
    const captchaType = document.getElementById('captchaTypeSelect').value;
    const lr = document.getElementById('lr').value;
    const iteration = document.getElementById('iteration').value;
    const debug_mode = document.getElementById('debug_mode').checked;
    erasePreviousMetrics();
    // Clear output container
    outputContainer.innerText = 'Executing Python code...\n';
    spinner.style.visibility = 'visible';

    // Close the existing event source if it exists
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }

    // Create a new event source
    eventSource = new EventSource('/attackPanel/streamOutput?captchaType='+captchaType.toString()+'&iteration='+iteration+'&lr='+lr+'&debug='+debug_mode);

    // Event listener for receiving messages from server
    eventSource.onmessage = function(event) {
        if (event.data.trim() === 'End of data') {
            const extractedInfo = extractInformation(outputContainer.innerText, captchaType);
            // Close the event source
            eventSource.close();
            // Hide spinner
            spinner.style.visibility = 'hidden';
            
            updatePieCharts(extractedInfo);
            displayMetricsTable(extractedInfo, captchaType);

        }
        else {
            // Append received data to output container
            outputContainer.innerText += event.data + '\n';
            // Automatically scroll to the bottom of the container
            outputContainer.scrollTop = outputContainer.scrollHeight;   
        }     
    };

    // Event listener for when the connection is closed
    eventSource.onclose = function() {
        console.log('Connection closed.');
        // Hide spinner
        spinner.style.visibility = 'hidden';
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