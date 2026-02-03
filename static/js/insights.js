// static/js/insights.js

async function loadInsights() {
    const res = await fetch("/insights-data");
    const data = await res.json();

    // Extract summary data
    const avgScore = data.average_score || 0;
    const avgStudy = data.average_study_hours || 0;
    const avgMarks = data.average_marks || 0;

    const riskDist = data.risk_distribution || {};
    const learnerDist = data.learner_distribution || {};

    // Update total predictions count
    document.getElementById("totalStudents").textContent = data.total_students || 0;

    // Register the plugin to all charts
    Chart.register(ChartDataLabels);

    // Common plugin configuration
    const commonPlugins = {
        legend: {
            position: 'top',
        },
        tooltip: {
            enabled: true
        },
        datalabels: {
            color: '#fff',
            font: {
                weight: 'bold',
                size: 12
            },
            formatter: function(value, context) {
                return value.toFixed(1);
            }
        }
    };

    // ---- 1️⃣ Average Score Chart ----
    new Chart(document.getElementById("scoreChart"), {
        type: "bar",
        data: {
            labels: ["Average Predicted Score"],
            datasets: [{
                label: "Final Score",
                data: [avgScore],
                backgroundColor: "rgba(56, 189, 248, 0.7)",
                borderColor: "rgba(56, 189, 248, 1)",
                borderWidth: 2
            }]
        },
        plugins: [ChartDataLabels],
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        display: false
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                ...commonPlugins,
                datalabels: {
                    ...commonPlugins.datalabels,
                    anchor: 'end',
                    align: 'top',
                    offset: 5
                }
            }
        }
    });

    // ---- 2️⃣ Risk Distribution Pie Chart ----
    new Chart(document.getElementById("riskChart"), {
        type: "pie",
        data: {
            labels: ["Low", "Medium", "High", "Critical"],
            datasets: [{
                data: [
                    riskDist["Low"] || 0,
                    riskDist["Medium"] || 0,
                    riskDist["High"] || 0,
                    riskDist["Critical"] || 0
                ],
                backgroundColor: [
                    "rgba(34,197,94,0.7)",   // green
                    "rgba(234,179,8,0.7)",   // yellow
                    "rgba(244,63,94,0.7)",   // red
                    "rgba(127,29,29,0.8)"    // dark red
                ],
                borderColor: '#0f172a',
                borderWidth: 2
            }]
        },
        plugins: [ChartDataLabels],
        options: {
            responsive: true,
            plugins: {
                ...commonPlugins,
                legend: {
                    position: 'right',
                    labels: {
                        color: '#e5e7eb',
                        padding: 20,
                        font: {
                            size: 12
                        }
                    }
                },
                datalabels: {
                    ...commonPlugins.datalabels,
                    formatter: function(value, context) {
                        const total = context.dataset.data.reduce((a, b) => a + b, 0);
                        const percentage = Math.round((value / total) * 100);
                        return value > 0 ? `${value} (${percentage}%)` : '';
                    }
                }
            }
        }
    });

    // ---- 3️⃣ Study Hours vs Avg Marks ----
    new Chart(document.getElementById("studyChart"), {
        type: "bar",
        data: {
            labels: ["Avg Study Hours", "Avg Marks"],
            datasets: [{
                label: "Average Values",
                data: [avgStudy, avgMarks],
                backgroundColor: [
                    "rgba(59,130,246,0.7)",   // blue
                    "rgba(236,72,153,0.7)"    // pink
                ],
                borderColor: [
                    "rgba(59,130,246,1)",
                    "rgba(236,72,153,1)"
                ],
                borderWidth: 2
            }]
        },
        plugins: [ChartDataLabels],
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        display: false
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                ...commonPlugins,
                datalabels: {
                    ...commonPlugins.datalabels,
                    anchor: 'end',
                    align: 'top',
                    offset: 5
                }
            }
        }
    });

}

document.addEventListener("DOMContentLoaded", loadInsights);
