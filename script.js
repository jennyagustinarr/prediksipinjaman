const chartInstances = {};
let analyticsData = null;
let visualizationRendered = false;
let modelRendered = false;
let datasetLoaded = false;

const C = {
    green950: '#052e16',
    green900: '#064e3b',
    green800: '#065f46',
    green700: '#047857',
    green600: '#059669',
    green500: '#10b981',
    green200: '#a7f3d0',
    green100: '#d1fae5',
    green50: '#ecfdf5',
    red: '#dc2626',
    orange: '#f97316',
    yellow: '#f59e0b',
    slate: '#64748b',
    border: '#d7f3e3'
};

const tabTitles = {
    overview: 'Ringkasan Dataset',
    predict: 'Prediksi Persetujuan Pinjaman',
    visualization: 'Visualisasi EDA',
    model: 'Evaluasi Model Random Forest',
    dataset: 'Preview Dataset'
};

const rupiahCompact = new Intl.NumberFormat('id-ID', {
    notation: 'compact',
    maximumFractionDigits: 1
});

const numberFormat = new Intl.NumberFormat('id-ID');

function destroyChart(id) {
    if (chartInstances[id]) {
        chartInstances[id].destroy();
        delete chartInstances[id];
    }
}

function showLoading(show) {
    document.getElementById('loadingOverlay').classList.toggle('hidden', !show);
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type === 'error' ? 'error' : ''}`;
    toast.classList.remove('hidden');
    setTimeout(() => toast.classList.add('hidden'), 3600);
}

function getBestEvaluation(modelName = 'Random Forest') {
    if (!analyticsData?.evaluation?.length) return null;
    return analyticsData.evaluation.find(item => item.model.toLowerCase() === modelName.toLowerCase()) || analyticsData.evaluation[0];
}

function activateTab(tabName) {
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });

    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.toggle('active', tab.id === `tab-${tabName}`);
    });

    document.getElementById('topbarTitle').textContent = tabTitles[tabName] || 'Dashboard';

    if (tabName === 'visualization' && analyticsData && !visualizationRendered) {
        renderVisualizationCharts(analyticsData);
        visualizationRendered = true;
    }

    if (tabName === 'model' && analyticsData && !modelRendered) {
        renderModelTab(analyticsData);
        modelRendered = true;
    }

    if (tabName === 'dataset' && !datasetLoaded) {
        loadDatasetSample();
    }

    if (window.innerWidth <= 760) {
        document.getElementById('sidebar').classList.remove('open');
    }
}

function setupNavigation() {
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.addEventListener('click', () => activateTab(btn.dataset.tab));
    });

    document.getElementById('sidebarToggle').addEventListener('click', () => {
        document.getElementById('sidebar').classList.toggle('open');
    });
}

function applyDefaultInputs(defaults) {
    if (!defaults) return;
    Object.entries(defaults).forEach(([key, value]) => {
        const input = document.querySelector(`[name="${key}"]`);
        if (input) input.value = value;
    });
}

async function loadAnalytics() {
    showLoading(true);
    try {
        const response = await fetch('/api/analytics');
        if (!response.ok) throw new Error('API analytics tidak merespons dengan benar.');
        analyticsData = await response.json();

        const d = analyticsData;
        document.getElementById('totalRows').textContent = numberFormat.format(d.dataset_info.total_rows);
        document.getElementById('modelStatus').textContent = d.model_status;
        document.getElementById('kpiTotal').textContent = numberFormat.format(d.dataset_info.total_rows);
        document.getElementById('kpiApproved').textContent = numberFormat.format(d.dataset_info.approved_count);
        document.getElementById('kpiRejected').textContent = numberFormat.format(d.dataset_info.rejected_count);
        document.getElementById('kpiAcc').textContent = `${d.accuracy}%`;

        applyDefaultInputs(d.defaults);
        renderPieChart(d);
        renderEvaluationOverviewChart(d);
    } catch (error) {
        console.error(error);
        showToast('Gagal memuat data dashboard. Pastikan Flask sudah berjalan.', 'error');
    } finally {
        showLoading(false);
    }
}

function baseChartOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: C.green950,
                    font: { size: 12, weight: '700' }
                }
            },
            tooltip: {
                backgroundColor: C.green950,
                titleFont: { weight: '800' },
                bodyFont: { weight: '600' },
                padding: 11,
                cornerRadius: 10
            }
        }
    };
}

function renderPieChart(d) {
    destroyChart('chartPie');
    const ctx = document.getElementById('chartPie').getContext('2d');
    const labels = Object.keys(d.target_counts);
    const values = Object.values(d.target_counts);

    chartInstances.chartPie = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: [C.green600, C.red],
                borderColor: '#fff',
                borderWidth: 3,
                hoverOffset: 10
            }]
        },
        options: {
            ...baseChartOptions(),
            cutout: '62%',
            plugins: {
                ...baseChartOptions().plugins,
                legend: { position: 'bottom' },
                tooltip: {
                    callbacks: {
                        label: ctx => {
                            const total = values.reduce((a, b) => a + b, 0);
                            const pct = ((ctx.parsed / total) * 100).toFixed(1);
                            return ` ${ctx.label}: ${numberFormat.format(ctx.parsed)} (${pct}%)`;
                        }
                    }
                }
            }
        }
    });
}

function renderEvaluationOverviewChart(d) {
    destroyChart('chartEvaluationOverview');
    const ctx = document.getElementById('chartEvaluationOverview').getContext('2d');
    const labels = d.evaluation.map(row => row.model);
    const values = d.evaluation.map(row => +(row.accuracy * 100).toFixed(2));

    chartInstances.chartEvaluationOverview = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Accuracy (%)',
                data: values,
                backgroundColor: labels.map(label => label.toLowerCase().includes('random') ? C.green600 : C.green200),
                borderRadius: 9,
                borderSkipped: false
            }]
        },
        options: {
            ...baseChartOptions(),
            scales: {
                y: {
                    min: Math.max(0, Math.min(...values) - 5),
                    max: 100,
                    grid: { color: '#e8f7ef' },
                    ticks: { color: C.slate, callback: value => `${value}%` }
                },
                x: { grid: { display: false }, ticks: { color: C.green950, font: { weight: '800' } } }
            },
            plugins: { ...baseChartOptions().plugins, legend: { display: false } }
        }
    });
}

function renderVisualizationCharts(d) {
    renderCorrelationChart(d);
    renderImportanceChart(d);
    renderOutlierChart(d);
    renderCibilDistributionChart(d);
    renderScatterChart(d);
}

function renderCorrelationChart(d) {
    destroyChart('chartCorr');
    const ctx = document.getElementById('chartCorr').getContext('2d');
    const labels = d.correlation.map(item => item.label);
    const values = d.correlation.map(item => item.value);

    chartInstances.chartCorr = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Korelasi terhadap Approved',
                data: values,
                backgroundColor: values.map(value => value >= 0 ? C.green600 : C.red),
                borderRadius: 8,
                borderSkipped: false
            }]
        },
        options: {
            ...baseChartOptions(),
            indexAxis: 'y',
            scales: {
                x: {
                    min: -1,
                    max: 1,
                    grid: { color: '#e8f7ef' },
                    ticks: { color: C.slate }
                },
                y: { grid: { display: false }, ticks: { color: C.green950, font: { size: 11, weight: '700' } } }
            },
            plugins: { ...baseChartOptions().plugins, legend: { display: false } }
        }
    });
}

function renderImportanceChart(d) {
    destroyChart('chartImportance');
    const ctx = document.getElementById('chartImportance').getContext('2d');
    const top = d.feature_importance.slice(0, 10).reverse();

    chartInstances.chartImportance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: top.map(item => item.label),
            datasets: [{
                label: 'Importance',
                data: top.map(item => item.importance),
                backgroundColor: C.green600,
                borderRadius: 8,
                borderSkipped: false
            }]
        },
        options: {
            ...baseChartOptions(),
            indexAxis: 'y',
            scales: {
                x: { grid: { color: '#e8f7ef' }, ticks: { color: C.slate } },
                y: { grid: { display: false }, ticks: { color: C.green950, font: { size: 11, weight: '700' } } }
            },
            plugins: { ...baseChartOptions().plugins, legend: { display: false } }
        }
    });
}

function renderOutlierChart(d) {
    destroyChart('chartOutlier');
    const ctx = document.getElementById('chartOutlier').getContext('2d');
    const data = d.outlier_counts;

    chartInstances.chartOutlier = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(item => item.label),
            datasets: [{
                label: 'Jumlah Outlier',
                data: data.map(item => item.count),
                backgroundColor: data.map(item => item.count > 0 ? C.orange : C.green200),
                borderRadius: 8,
                borderSkipped: false
            }]
        },
        options: {
            ...baseChartOptions(),
            indexAxis: 'y',
            scales: {
                x: { grid: { color: '#e8f7ef' }, ticks: { color: C.slate } },
                y: { grid: { display: false }, ticks: { color: C.green950, font: { size: 11, weight: '700' } } }
            },
            plugins: { ...baseChartOptions().plugins, legend: { display: false } }
        }
    });
}

function buildHistogram(values, minValue = null, maxValue = null, binCount = 10) {
    const clean = values.filter(value => Number.isFinite(value));
    const min = minValue ?? Math.min(...clean);
    const max = maxValue ?? Math.max(...clean);
    const width = (max - min) / binCount || 1;
    const bins = Array.from({ length: binCount }, () => 0);

    clean.forEach(value => {
        let index = Math.floor((value - min) / width);
        if (index < 0) index = 0;
        if (index >= binCount) index = binCount - 1;
        bins[index] += 1;
    });

    const labels = Array.from({ length: binCount }, (_, i) => {
        const start = Math.round(min + i * width);
        const end = Math.round(min + (i + 1) * width);
        return `${start}-${end}`;
    });

    return { labels, bins };
}

function renderCibilDistributionChart(d) {
    destroyChart('chartCibil');
    const ctx = document.getElementById('chartCibil').getContext('2d');
    const approved = d.distribution_data.cibil_score.Approved;
    const rejected = d.distribution_data.cibil_score.Rejected;
    const histApproved = buildHistogram(approved, 300, 900, 10);
    const histRejected = buildHistogram(rejected, 300, 900, 10);

    chartInstances.chartCibil = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: histApproved.labels,
            datasets: [
                {
                    label: 'Approved',
                    data: histApproved.bins,
                    backgroundColor: C.green600,
                    borderRadius: 5
                },
                {
                    label: 'Rejected',
                    data: histRejected.bins,
                    backgroundColor: C.red,
                    borderRadius: 5
                }
            ]
        },
        options: {
            ...baseChartOptions(),
            scales: {
                x: { stacked: false, grid: { display: false }, ticks: { color: C.slate, maxRotation: 45 } },
                y: { grid: { color: '#e8f7ef' }, ticks: { color: C.slate } }
            }
        }
    });
}

function renderScatterChart(d) {
    destroyChart('chartScatter');
    const ctx = document.getElementById('chartScatter').getContext('2d');

    chartInstances.chartScatter = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Approved',
                    data: d.scatter_data.Approved,
                    backgroundColor: C.green600,
                    pointRadius: 3,
                    pointHoverRadius: 5
                },
                {
                    label: 'Rejected',
                    data: d.scatter_data.Rejected,
                    backgroundColor: C.red,
                    pointRadius: 3,
                    pointHoverRadius: 5
                }
            ]
        },
        options: {
            ...baseChartOptions(),
            scales: {
                x: {
                    title: { display: true, text: 'CIBIL Score', color: C.green950, font: { weight: '800' } },
                    grid: { color: '#e8f7ef' },
                    ticks: { color: C.slate }
                },
                y: {
                    title: { display: true, text: 'Jumlah Pinjaman', color: C.green950, font: { weight: '800' } },
                    grid: { color: '#e8f7ef' },
                    ticks: { color: C.slate, callback: value => rupiahCompact.format(value) }
                }
            },
            plugins: {
                ...baseChartOptions().plugins,
                tooltip: {
                    callbacks: {
                        label: ctx => ` ${ctx.dataset.label}: CIBIL ${ctx.parsed.x}, Pinjaman ${rupiahCompact.format(ctx.parsed.y)}`
                    }
                }
            }
        }
    });
}

function renderModelTab(d) {
    renderMetricCards(d);
    renderConfusionMatrix(d);
    renderReportChart(d);
    renderModelComparisonChart(d);
}

function renderMetricCards(d) {
    const evalRow = getBestEvaluation('Random Forest');
    const metrics = [
        ['Accuracy', evalRow ? evalRow.accuracy : 0],
        ['Precision', evalRow ? evalRow.precision : 0],
        ['Recall', evalRow ? evalRow.recall : 0],
        ['F1-Score', evalRow ? evalRow.f1 : 0]
    ];

    document.getElementById('metricGrid').innerHTML = metrics.map(([label, value]) => `
        <div class="metric-card">
            <span>${label}</span>
            <strong>${(value * 100).toFixed(2)}%</strong>
        </div>
    `).join('');
}

function renderConfusionMatrix(d) {
    const cm = d.confusion_matrix;
    document.getElementById('cmContainer').innerHTML = `
        <div class="cm-grid">
            <div></div>
            <div class="cm-axis">Prediksi<br>Rejected</div>
            <div class="cm-axis">Prediksi<br>Approved</div>
            <div class="cm-label">Aktual<br>Rejected</div>
            <div class="cm-cell good">${cm[0][0]}</div>
            <div class="cm-cell bad">${cm[0][1]}</div>
            <div class="cm-label">Aktual<br>Approved</div>
            <div class="cm-cell bad">${cm[1][0]}</div>
            <div class="cm-cell good">${cm[1][1]}</div>
        </div>
    `;
}

function renderReportChart(d) {
    destroyChart('chartReport');
    const ctx = document.getElementById('chartReport').getContext('2d');
    const labels = Object.keys(d.classification_report);
    const precision = labels.map(label => +(d.classification_report[label].precision * 100).toFixed(2));
    const recall = labels.map(label => +(d.classification_report[label].recall * 100).toFixed(2));
    const f1 = labels.map(label => +(d.classification_report[label].f1 * 100).toFixed(2));

    chartInstances.chartReport = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                { label: 'Precision', data: precision, backgroundColor: C.green800, borderRadius: 7 },
                { label: 'Recall', data: recall, backgroundColor: C.green500, borderRadius: 7 },
                { label: 'F1-Score', data: f1, backgroundColor: C.green200, borderRadius: 7 }
            ]
        },
        options: {
            ...baseChartOptions(),
            scales: {
                y: { min: 0, max: 100, grid: { color: '#e8f7ef' }, ticks: { color: C.slate, callback: v => `${v}%` } },
                x: { grid: { display: false }, ticks: { color: C.green950, font: { weight: '800' } } }
            }
        }
    });
}

function renderModelComparisonChart(d) {
    destroyChart('chartModelComparison');
    const ctx = document.getElementById('chartModelComparison').getContext('2d');
    const labels = d.evaluation.map(row => row.model);

    chartInstances.chartModelComparison = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                { label: 'Accuracy', data: d.evaluation.map(row => +(row.accuracy * 100).toFixed(2)), backgroundColor: C.green800, borderRadius: 7 },
                { label: 'Precision', data: d.evaluation.map(row => +(row.precision * 100).toFixed(2)), backgroundColor: C.green500, borderRadius: 7 },
                { label: 'Recall', data: d.evaluation.map(row => +(row.recall * 100).toFixed(2)), backgroundColor: C.green200, borderRadius: 7 },
                { label: 'F1-Score', data: d.evaluation.map(row => +(row.f1 * 100).toFixed(2)), backgroundColor: C.yellow, borderRadius: 7 }
            ]
        },
        options: {
            ...baseChartOptions(),
            scales: {
                y: { min: 0, max: 100, grid: { color: '#e8f7ef' }, ticks: { color: C.slate, callback: v => `${v}%` } },
                x: { grid: { display: false }, ticks: { color: C.green950, font: { weight: '800' } } }
            }
        }
    });
}

function collectFormData(formElement) {
    const formData = new FormData(formElement);
    const data = {};
    formData.forEach((value, key) => {
        data[key] = value;
    });
    return data;
}

function showPredictionError(message) {
    const errorBox = document.getElementById('errorBox');
    errorBox.textContent = message;
    errorBox.classList.remove('hidden');
}

function hidePredictionError() {
    const errorBox = document.getElementById('errorBox');
    errorBox.textContent = '';
    errorBox.classList.add('hidden');
}

function renderPrediction(data) {
    const resultBox = document.getElementById('resultBox');
    const resultIcon = document.getElementById('resultIcon');
    const resultLabel = document.getElementById('resultLabel');
    const resultDesc = document.getElementById('resultDesc');
    const approvedProbability = document.getElementById('approvedProbability');
    const rejectedProbability = document.getElementById('rejectedProbability');
    const approvedBar = document.getElementById('approvedBar');
    const rejectedBar = document.getElementById('rejectedBar');

    const isApproved = data.prediction === 'Approved';
    resultBox.classList.remove('hidden');
    resultIcon.classList.toggle('rejected', !isApproved);
    resultIcon.innerHTML = isApproved ? '<i class="fa-solid fa-check"></i>' : '<i class="fa-solid fa-xmark"></i>';
    resultLabel.textContent = data.prediction;
    resultDesc.textContent = data.note;
    approvedProbability.textContent = `${data.probability_approved}%`;
    rejectedProbability.textContent = `${data.probability_rejected}%`;
    approvedBar.style.width = `${data.probability_approved}%`;
    rejectedBar.style.width = `${data.probability_rejected}%`;
}

function setupPredictionForm() {
    const form = document.getElementById('predictionForm');
    const button = document.getElementById('btnPredict');

    form.addEventListener('submit', async event => {
        event.preventDefault();
        hidePredictionError();
        button.disabled = true;
        button.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Memproses...';

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(collectFormData(form))
            });

            const data = await response.json();
            if (!response.ok || data.status !== 'success') {
                throw new Error(data.message || 'Prediksi gagal diproses.');
            }

            renderPrediction(data);
            showToast('Prediksi berhasil diproses.');
        } catch (error) {
            showPredictionError(error.message);
        } finally {
            button.disabled = false;
            button.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Mulai Prediksi';
        }
    });

    form.addEventListener('reset', () => {
        hidePredictionError();
        document.getElementById('resultBox').classList.add('hidden');
        setTimeout(() => applyDefaultInputs(analyticsData?.defaults), 0);
    });
}

async function loadDatasetSample() {
    showLoading(true);
    try {
        const response = await fetch('/api/dataset-sample');
        if (!response.ok) throw new Error('Dataset sample gagal dimuat.');
        const data = await response.json();
        const thead = document.getElementById('dataTableHead');
        const tbody = document.getElementById('dataTableBody');

        thead.innerHTML = `<tr>${data.columns.map(column => `<th>${column}</th>`).join('')}</tr>`;
        tbody.innerHTML = data.rows.map(row => `
            <tr>
                ${data.columns.map(column => `<td>${row[column] ?? ''}</td>`).join('')}
            </tr>
        `).join('');

        document.getElementById('datasetMeta').textContent = `Menampilkan ${data.rows.length} baris awal dari total ${numberFormat.format(analyticsData.dataset_info.total_rows)} data dan ${analyticsData.dataset_info.total_cols} kolom.`;
        datasetLoaded = true;
    } catch (error) {
        console.error(error);
        showToast('Gagal memuat preview dataset.', 'error');
    } finally {
        showLoading(false);
    }
}

window.addEventListener('DOMContentLoaded', () => {
    setupNavigation();
    setupPredictionForm();
    loadAnalytics();
});
