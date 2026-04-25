/* ── ChurnGuard Frontend ──────────────────────────────────────── */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ── State ───────────────────────────────────────────────────────
let scoringData = null;
let currentPage = 1;
const PAGE_SIZE = 20;
let riskDonutChart = null;
let featureBarChart = null;
let globalFeatureChart = null;

// ── Navigation ──────────────────────────────────────────────────
const sections = { dashboard: 'Dashboard', customers: 'Customers', explainability: 'Explainability' };
const subtitles = {
    dashboard: 'Overview of churn risk and revenue impact',
    customers: 'Detailed customer scores and retention strategies',
    explainability: 'SHAP feature importance and individual explanations',
};

function showSection(name) {
    // Hide all sections + welcome
    $$('.content-section').forEach(s => s.style.display = 'none');
    $('#welcomeState').style.display = scoringData ? 'none' : 'flex';

    if (scoringData) {
        $(`#section-${name}`).style.display = 'flex';
    }

    // Update nav
    $$('.nav-item').forEach(n => n.classList.remove('active'));
    $(`.nav-item[data-section="${name}"]`).classList.add('active');

    // Update topbar
    $('#pageTitle').textContent = sections[name];
    $('#pageSubtitle').textContent = subtitles[name];

    // Close mobile sidebar
    $('#sidebar').classList.remove('open');
}

$$('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
        e.preventDefault();
        showSection(item.dataset.section);
    });
});

$('#menuToggle').addEventListener('click', () => {
    $('#sidebar').classList.toggle('open');
});

// ── File Upload & Demo ──────────────────────────────────────────
function handleUpload(file) {
    const formData = new FormData();
    formData.append('file', file);
    runScoring(formData);
}

function handleDemo() {
    runScoring(new FormData());
}

$('#csvUpload').addEventListener('change', (e) => {
    if (e.target.files[0]) handleUpload(e.target.files[0]);
});
$('#csvUpload2').addEventListener('change', (e) => {
    if (e.target.files[0]) handleUpload(e.target.files[0]);
});
$('#demoBtn').addEventListener('click', handleDemo);
$('#demoBtn2').addEventListener('click', handleDemo);

// ── Scoring API Call ────────────────────────────────────────────
async function runScoring(formData) {
    const overlay = $('#loadingOverlay');
    overlay.classList.add('active');

    try {
        const resp = await fetch('/api/score', { method: 'POST', body: formData });
        const data = await resp.json();

        if (!resp.ok) {
            alert(data.error || 'Scoring failed');
            return;
        }

        scoringData = data;
        renderDashboard();
        renderCustomerTable();
        renderExplainability();
        showSection('dashboard');

        // Update model badge
        const badge = $('#modelBadge');
        badge.querySelector('.model-badge-dot').classList.add('ready');
        badge.querySelector('span').textContent = `${data.model_info.model_type} v${data.model_info.version}`;

    } catch (err) {
        alert('Network error: ' + err.message);
    } finally {
        overlay.classList.remove('active');
    }
}

// ── Render Dashboard ────────────────────────────────────────────
function renderDashboard() {
    const s = scoringData.summary;
    const total = s.total_customers;

    $('#kpiTotal').textContent = total.toLocaleString();
    $('#kpiHigh').textContent = s.high_risk.toLocaleString();
    $('#kpiHighPct').textContent = `${((s.high_risk / total) * 100).toFixed(1)}%`;
    $('#kpiMedium').textContent = s.medium_risk.toLocaleString();
    $('#kpiMediumPct').textContent = `${((s.medium_risk / total) * 100).toFixed(1)}%`;
    $('#kpiLow').textContent = s.low_risk.toLocaleString();
    $('#kpiLowPct').textContent = `${((s.low_risk / total) * 100).toFixed(1)}%`;

    // Impact
    const imp = scoringData.impact;
    $('#impactRevenue').textContent = `$${imp.total_revenue_at_risk.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    $('#impactTargeted').textContent = imp.targeted_customer_count.toLocaleString();
    $('#impactSaved').textContent = `$${imp.projected_revenue_saved.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    $('#impactROI').textContent = `${imp.retention_roi.toFixed(2)}x`;

    // Model info
    const mi = scoringData.model_info;
    const infoGrid = $('#modelInfoGrid');
    infoGrid.innerHTML = [
        { label: 'Model Type', value: mi.model_type },
        { label: 'Version', value: mi.version },
        { label: 'Training Date', value: new Date(mi.training_date).toLocaleDateString() },
        { label: 'Training Rows', value: mi.dataset_row_count.toLocaleString() },
        { label: 'AUC-ROC', value: mi.auc_roc ? mi.auc_roc.toFixed(4) : 'N/A' },
        { label: 'Accuracy', value: mi.accuracy ? (mi.accuracy * 100).toFixed(1) + '%' : 'N/A' },
    ].map(i => `<div class="model-info-item"><span class="label">${i.label}</span><span class="value">${i.value}</span></div>`).join('');

    // Charts
    renderRiskDonut(s);
    renderFeatureBar(scoringData.feature_importance.slice(0, 8));
}

function renderRiskDonut(summary) {
    const ctx = $('#riskDonutChart').getContext('2d');
    if (riskDonutChart) riskDonutChart.destroy();

    riskDonutChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['High Risk', 'Medium Risk', 'Low Risk'],
            datasets: [{
                data: [summary.high_risk, summary.medium_risk, summary.low_risk],
                backgroundColor: ['#ef4444', '#f59e0b', '#22c55e'],
                borderColor: '#1a1d27',
                borderWidth: 3,
                hoverOffset: 6,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#8b8fa3', padding: 16, font: { family: "'Inter', sans-serif", size: 12 } },
                },
            },
        },
    });
}

function renderFeatureBar(features) {
    const ctx = $('#featureBarChart').getContext('2d');
    if (featureBarChart) featureBarChart.destroy();

    const labels = features.map(f => cleanFeatureName(f.feature));
    const values = features.map(f => f.importance);

    featureBarChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Mean |SHAP|',
                data: values,
                backgroundColor: 'rgba(99, 102, 241, 0.7)',
                borderColor: '#6366f1',
                borderWidth: 1,
                borderRadius: 4,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: { display: false },
            },
            scales: {
                x: {
                    grid: { color: 'rgba(42, 45, 58, 0.5)' },
                    ticks: { color: '#8b8fa3', font: { size: 11 } },
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#e4e6f0', font: { size: 11 } },
                },
            },
        },
    });
}

// ── Customer Table ──────────────────────────────────────────────
function getFilteredCustomers() {
    if (!scoringData) return [];
    let customers = scoringData.customers;

    const search = $('#customerSearch').value.toLowerCase();
    const risk = $('#riskFilter').value;

    if (search) {
        customers = customers.filter(c => c.customer_id.toLowerCase().includes(search));
    }
    if (risk) {
        customers = customers.filter(c => c.risk_segment === risk);
    }
    return customers;
}

function renderCustomerTable() {
    const customers = getFilteredCustomers();
    const totalPages = Math.max(1, Math.ceil(customers.length / PAGE_SIZE));
    if (currentPage > totalPages) currentPage = totalPages;

    const start = (currentPage - 1) * PAGE_SIZE;
    const page = customers.slice(start, start + PAGE_SIZE);

    const tbody = $('#customerTableBody');
    tbody.innerHTML = page.map(c => {
        const prob = c.churn_probability;
        const barColor = prob >= 0.7 ? '#ef4444' : prob >= 0.4 ? '#f59e0b' : '#22c55e';
        const badgeClass = c.risk_segment === 'High' ? 'risk-high' : c.risk_segment === 'Medium' ? 'risk-medium' : 'risk-low';
        const driver = c.drivers.length > 0 ? cleanFeatureName(c.drivers[0]) : '—';

        return `<tr>
            <td><strong>${c.customer_id}</strong></td>
            <td>
                <div class="prob-bar-wrapper">
                    <div class="prob-bar"><div class="prob-bar-fill" style="width:${prob * 100}%;background:${barColor}"></div></div>
                    <span>${(prob * 100).toFixed(1)}%</span>
                </div>
            </td>
            <td><span class="risk-badge ${badgeClass}">${c.risk_segment}</span></td>
            <td>${driver}</td>
            <td>${c.retention_strategy || '—'}</td>
            <td>$${c.annual_revenue.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
        </tr>`;
    }).join('');

    // Pagination
    renderPagination(totalPages);
}

function renderPagination(totalPages) {
    const container = $('#pagination');
    if (totalPages <= 1) { container.innerHTML = ''; return; }

    let html = `<button class="page-btn" ${currentPage === 1 ? 'disabled' : ''} onclick="goToPage(${currentPage - 1})">‹ Prev</button>`;

    const maxVisible = 7;
    let startPage = Math.max(1, currentPage - Math.floor(maxVisible / 2));
    let endPage = Math.min(totalPages, startPage + maxVisible - 1);
    if (endPage - startPage < maxVisible - 1) startPage = Math.max(1, endPage - maxVisible + 1);

    if (startPage > 1) {
        html += `<button class="page-btn" onclick="goToPage(1)">1</button>`;
        if (startPage > 2) html += `<span style="color:var(--text-dim);padding:0 4px">…</span>`;
    }

    for (let i = startPage; i <= endPage; i++) {
        html += `<button class="page-btn ${i === currentPage ? 'active' : ''}" onclick="goToPage(${i})">${i}</button>`;
    }

    if (endPage < totalPages) {
        if (endPage < totalPages - 1) html += `<span style="color:var(--text-dim);padding:0 4px">…</span>`;
        html += `<button class="page-btn" onclick="goToPage(${totalPages})">${totalPages}</button>`;
    }

    html += `<button class="page-btn" ${currentPage === totalPages ? 'disabled' : ''} onclick="goToPage(${currentPage + 1})">Next ›</button>`;
    container.innerHTML = html;
}

window.goToPage = function(page) {
    currentPage = page;
    renderCustomerTable();
};

$('#customerSearch').addEventListener('input', () => { currentPage = 1; renderCustomerTable(); });
$('#riskFilter').addEventListener('change', () => { currentPage = 1; renderCustomerTable(); });

// ── Explainability ──────────────────────────────────────────────
function renderExplainability() {
    // Global feature chart (full list)
    const ctx = $('#globalFeatureChart').getContext('2d');
    if (globalFeatureChart) globalFeatureChart.destroy();

    const features = scoringData.feature_importance;
    const labels = features.map(f => cleanFeatureName(f.feature));
    const values = features.map(f => f.importance);

    globalFeatureChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Mean |SHAP|',
                data: values,
                backgroundColor: values.map((_, i) => {
                    const colors = ['#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe'];
                    return colors[Math.min(Math.floor(i / 4), colors.length - 1)];
                }),
                borderRadius: 4,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    grid: { color: 'rgba(42, 45, 58, 0.5)' },
                    ticks: { color: '#8b8fa3', font: { size: 11 } },
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#e4e6f0', font: { size: 11 } },
                },
            },
        },
    });

    // Populate waterfall customer selector
    const select = $('#waterfallCustomerSelect');
    select.innerHTML = '<option value="">Select a customer...</option>';
    scoringData.customers.forEach((c, i) => {
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = `${c.customer_id} (${c.risk_segment} — ${(c.churn_probability * 100).toFixed(1)}%)`;
        select.appendChild(opt);
    });
}

$('#waterfallCustomerSelect').addEventListener('change', async (e) => {
    const idx = e.target.value;
    const container = $('#waterfallContainer');

    if (!idx && idx !== 0) {
        container.innerHTML = '<div class="waterfall-placeholder"><p>Select a customer to view their SHAP waterfall plot</p></div>';
        return;
    }

    container.innerHTML = '<div class="waterfall-placeholder"><div class="spinner"></div><p>Generating waterfall plot...</p></div>';

    try {
        const resp = await fetch(`/api/waterfall/${idx}`);
        if (!resp.ok) {
            const err = await resp.json();
            container.innerHTML = `<div class="waterfall-placeholder"><p style="color:var(--red)">${err.error}</p></div>`;
            return;
        }
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        container.innerHTML = `<img src="${url}" alt="SHAP Waterfall Plot">`;
    } catch (err) {
        container.innerHTML = `<div class="waterfall-placeholder"><p style="color:var(--red)">Failed to load plot</p></div>`;
    }
});

// ── Export CSV ───────────────────────────────────────────────────
$('#exportCsvBtn').addEventListener('click', async () => {
    try {
        const resp = await fetch('/api/export/csv');
        if (!resp.ok) {
            alert('Export not available');
            return;
        }
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'customer_summary.csv';
        a.click();
        URL.revokeObjectURL(url);
    } catch (err) {
        alert('Export failed: ' + err.message);
    }
});

// ── Helpers ─────────────────────────────────────────────────────
function cleanFeatureName(name) {
    // Remove prefixes like num__, cat__ and clean up
    return name
        .replace(/^(num__|cat__)/, '')
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());
}

// ── Init: Load model info ───────────────────────────────────────
(async function init() {
    try {
        const resp = await fetch('/api/model-info');
        if (resp.ok) {
            const info = await resp.json();
            const badge = $('#modelBadge');
            badge.querySelector('.model-badge-dot').classList.add('ready');
            badge.querySelector('span').textContent = `${info.model_type} v${info.version}`;
        }
    } catch (e) {
        // Model not loaded yet, that's fine
    }
})();
