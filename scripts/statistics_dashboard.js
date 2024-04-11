// STATISTICS DISPLAY


export function extractInformation(data) {
    // Regular expressions to match the desired information
    // const lossRegex = /Loss: ([\d.]+)/;
    // const accuracyRegex = /Accuracy: ([\d.]+)/;
    // const FalsePositiveRegex = /False Positive Rate: ([\d.]+)/;
    // const FalseNegativeRegex = /False Negative Rate: ([\d.]+)/;
    // const PrecisionRegex = /Precision: ([\d.]+)/;
    // const RecallRegex = /Recall: ([\d.]+)/;
    // const F1_ScoreRegex = /F1-Score: ([\d.]+)/;
    // // Extract loss and accuracy from the data
    // const lossMatch = data.match(lossRegex);
    // const accuracyMatch = data.match(accuracyRegex);
    // const FalsePositiveMatch = data.match(FalsePositiveRegex);
    // const FalseNegativeMatch = data.match(FalseNegativeRegex);
    // const PrecisionMatch = data.match(PrecisionRegex);
    // const RecallMatch = data.match(RecallRegex);
    // const F1_ScoreMatch = data.match(F1_ScoreRegex);

    // // Initialize variables to store extracted information
    // let loss = null;
    // let accuracy = null;
    // let falsePositive = null;
    // let falseNegative = null;
    // let precision = null;
    // let recall = null;
    // let f1_score = null;

    // // Check if the information was found and extract it
    // if (lossMatch && lossMatch.length > 1) {
    //     loss = parseFloat(lossMatch[1]);
    // }
    // if (accuracyMatch && accuracyMatch.length > 1) {
    //     accuracy = parseFloat(accuracyMatch[1]);
    // }
    // if (FalsePositiveMatch && FalsePositiveMatch.length > 1) {
    //     falsePositive = parseFloat(FalsePositiveMatch[1]);
    // }
    // if (FalseNegativeMatch && FalseNegativeMatch.length > 1) {
    //     falseNegative = parseFloat(FalseNegativeMatch[1]);
    // }
    // if (PrecisionMatch && PrecisionMatch.length > 1) {
    //     precision = parseFloat(PrecisionMatch[1]);
    // }
    // if (RecallMatch && RecallMatch.length > 1) {
    //     recall = parseFloat(RecallMatch[1]);
    // }
    // if (F1_ScoreMatch && F1_ScoreMatch.length > 1) {
    //     f1_score = parseFloat(F1_ScoreMatch[1]);
    // }


    const patterns = [
        { name: 'loss', regex: /Loss: ([\d.]+)/ },
        { name: 'accuracy', regex: /Accuracy: ([\d.]+)/ },
        { name: 'falsePositive', regex: /False Positive Rate: ([\d.]+)/ },
        { name: 'falseNegative', regex: /False Negative Rate: ([\d.]+)/ },
        { name: 'precision', regex: /Precision: ([\d.]+)/ },
        { name: 'recall', regex: /Recall: ([\d.]+)/ },
        { name: 'f1_score', regex: /F1-Score: ([\d.]+)/ }
    ];
      
    const info = {};
      
    patterns.forEach(({ name, regex }) => {
      const match = data.match(regex);
      info[name] = match && match.length > 1 ? parseFloat(match[1]) : null;
    });
      
    const { loss, accuracy, falsePositive, falseNegative, precision, recall, f1_score } = info;
      
    console.log('Loss:', loss);
    console.log('Accuracy:', accuracy);
    console.log('False Positive Rate:', falsePositive);
    console.log('False Negative Rate:', falseNegative);
    console.log('Precision:', precision);
    console.log('Recall:', recall);
    console.log('F1-Score:', f1_score);

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