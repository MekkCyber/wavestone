// STATISTICS DISPLAY


export function extractInformation(data) {
    // Regular expressions to match the desired information
    const lossRegex = /loss: ([\d.]+)/;
    const accuracyRegex = /accuracy: ([\d.]+)/;
    const sparse_categorical_accuracyRegex = /sparse_categorical_accuracy: ([\d.]+)/;
    // Extract loss and accuracy from the data
    const lossMatch = data.match(lossRegex);
    const accuracyMatch = data.match(accuracyRegex);
    const sparse_categorical_accuracyMatch = data.match(sparse_categorical_accuracyRegex);

    // Initialize variables to store extracted information
    let loss = null;
    let accuracy = null;
    let sparse_categorical_accuracy = null;

    // Check if the information was found and extract it
    if (lossMatch && lossMatch.length > 1) {
        loss = parseFloat(lossMatch[1]);
    }
    if (accuracyMatch && accuracyMatch.length > 1) {
        accuracy = parseFloat(accuracyMatch[1]);
    }
    if (sparse_categorical_accuracyMatch && sparse_categorical_accuracyMatch.length > 1) {
        sparse_categorical_accuracy = parseFloat(sparse_categorical_accuracyMatch[1]);
    }

    // Return the extracted information
    return { loss, accuracy, sparse_categorical_accuracy };
}

let lossChart, accuracyChart, sparseAccuracyChart;


// Function to update pie charts
export function updatePieCharts(extractedInfo) {
    // Update or create loss pie chart
    const { loss, accuracy, sparse_categorical_accuracy: sparseAccuracy } = extractedInfo;
    if (lossChart) {
        lossChart.data.datasets[0].data[0] = loss;
        lossChart.update();
    } else {
        lossChart = new Chart(document.getElementById('lossPieChart').getContext('2d'), {
            type: 'pie',
            data: {
                labels: ['Loss'],
                datasets: [{
                    data: [loss],
                    backgroundColor: ['#FF6384']
                }]
            },
            options: {
                responsive: true
            }
        });
    }

    // Update or create accuracy pie chart
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

    // Update or create sparse accuracy pie chart
    if (sparseAccuracyChart) {
        sparseAccuracyChart.data.datasets[0].data[0] = sparseAccuracy;
        sparseAccuracyChart.update();
    } else {
        sparseAccuracyChart = new Chart(document.getElementById('sparseAccuracyPieChart').getContext('2d'), {
            type: 'pie',
            data: {
                labels: ['Sparse Accuracy'],
                datasets: [{
                    data: [sparseAccuracy],
                    backgroundColor: ['#FFCE56']
                }]
            },
            options: {
                responsive: true
            }
        });
    }
}