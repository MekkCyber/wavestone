// STATISTICS DISPLAY


export function extractInformation(data) {
    // Regular expressions to match the desired information
    const lossRegex = /loss: ([\d.]+)/;
    const accuracyRegex = /accuracy: ([\d.]+)/;
    // Extract loss and accuracy from the data
    const lossMatch = data.match(lossRegex);
    const accuracyMatch = data.match(accuracyRegex);

    // Initialize variables to store extracted information
    let loss = null;
    let accuracy = null;

    // Check if the information was found and extract it
    if (lossMatch && lossMatch.length > 1) {
        loss = parseFloat(lossMatch[1]);
    }
    if (accuracyMatch && accuracyMatch.length > 1) {
        accuracy = parseFloat(accuracyMatch[1]);
    }

    // Return the extracted information
    return { accuracy};
}

let accuracyChart;


// Function to update pie charts
export function updatePieCharts(extractedInfo) {
    // Update or create loss pie chart
    const { accuracy } = extractedInfo;
    
    if (accuracyChart) {
        accuracyChart.data.datasets[0].data[0] = accuracy;
        accuracyChart.update();
    } else {
        accuracyChart = new Chart(document.getElementById('accuracyPieChart').getContext('2d'), {
            type: 'pie',
            data: {
                labels: ['Accuracy'],
                datasets: [{
                    data: [accuracy],
                    backgroundColor: ['#36A2EB']
                }]
            },
            options: {
                responsive: true
            }
        });
    }

    
}