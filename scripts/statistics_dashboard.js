// STATISTICS DISPLAY


export function extractInformation(data, debug_mode) {


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
      
    // if (debug_mode === true) {
    console.log('Loss:', loss);
    console.log('Accuracy:', accuracy);
    console.log('False Positive Rate:', falsePositive);
    console.log('False Negative Rate:', falseNegative);
    console.log('Precision:', precision);
    console.log('Recall:', recall);
    console.log('F1-Score:', f1_score);
    // }

    // Return the extracted information
    return info;
}

let recallChart, precisionChart, F1Chart;


// Function to update pie charts
export function updatePieCharts(extractedInfo) {
    // Update or create loss pie chart
    const { precision, recall, f1_score } = extractedInfo;
    
    if (precisionChart) {
        precisionChart.data.datasets[0].data[0] = precision;
        precisionChart.update();
    } else {
        precisionChart = new Chart(document.getElementById('precisionPieChart').getContext('2d'), {
            type: 'pie',
            data: {
                labels: ['Precision', ''],
                datasets: [{
                    data: [precision, 100-precision],
                    backgroundColor: ['#FF6384', '#444444']
                }]
            },
            options: {
                plugins: {
                    responsive: true,
                    legend: {
                        labels: {
                            color: 'white' // Specify the color of the labels
                        }
                    },
                    title: {
                        display: true,
                        text: 'Precision : ' + precision.toFixed(2) + '%',
                        color: 'white', // Specify the color of the title
                        position: 'bottom' // Position the title at the bottom
                    }
                }
            }
        });

    }


    if (recallChart) {
        recallChart.data.datasets[0].data[0] = accuracy;
        recallChart.update();
    } else {
        recallChart = new Chart(document.getElementById('recallPieChart').getContext('2d'), {
            type: 'pie',
            data: {
                labels: ['Recall', ''],
                datasets: [{
                    data: [recall, 100-recall],
                    backgroundColor: ['#36A2EB', '#444444']
                }]
            },
            options: {
                plugins: {
                    responsive: true,
                    legend: {
                        labels: {
                            color: 'white' // Specify the color of the labels
                        }
                    },
                    title: {
                        display: true,
                        text: 'Recall : ' + recall.toFixed(2) + '%',
                        color: 'white', // Specify the color of the title
                        position: 'bottom' // Position the title at the bottom
                    }
                }
            }
        });
        
    }


    if (F1Chart) {
        F1Chart.data.datasets[0].data[0] = accuracy;
        F1Chart.updaccuracyChartate();
    } else {
        F1Chart = new Chart(document.getElementById('f1PieChart').getContext('2d'), {
            type: 'pie',
            data: {
                labels: ['F1-Score', ''],
                datasets: [{
                    data: [f1_score, 100-f1_score],
                    backgroundColor: ['#FFCE56', '#444444']
                }]
            },
            options: {
                plugins: {
                    responsive: true,
                    legend: {
                        labels: {
                            color: 'white' // Specify the color of the labels
                        }
                    },
                    title: {
                        display: true,
                        text: 'F1-Score : ' + f1_score.toFixed(2) + '%',
                        color: 'white', // Specify the color of the title
                        position: 'bottom' // Position the title at the bottom
                    }
                }
            }
        });
        
    }

    
}