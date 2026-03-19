const COLORS = [
    '#3b82f6', '#ef4444', '#10b981', '#f59e0b', 
    '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'
];

let globalClasses = {};
let activeImageIdx = null;

// Pagination state
let currentPage = 1;
const currentLimit = 50;
let currentSearch = "";
let searchTimeout = null;

// Chart instances
let classDistChartInstance = null;
let bboxSizeChartInstance = null;
let splitDistChartInstance = null;
// Store dynamic advanced charts to destroy them on re-run
let advancedChartInstances = [];

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    loadDashboard();
    loadImages();

    // Search and Pagination
    document.getElementById('search-input').addEventListener('input', handleSearch);
    document.getElementById('page-prev-btn').addEventListener('click', () => {
        if (currentPage > 1) { currentPage--; loadImages(); }
    });
    document.getElementById('page-next-btn').addEventListener('click', () => {
        currentPage++; loadImages();
    });

    // Augmentation modal wiring
    document.getElementById('open-augment-btn').addEventListener('click', openAugmentModal);
    document.getElementById('close-augment-btn').addEventListener('click', closeAugmentModal);
    document.getElementById('augment-modal').addEventListener('click', (e) => {
        if (e.target === e.currentTarget) closeAugmentModal();
    });
    document.getElementById('apply-augment-btn').addEventListener('click', applyAugmentations);

    // Advanced analytics modal wiring
    document.getElementById('open-analytics-btn').addEventListener('click', openAnalyticsModal);
    document.getElementById('close-analytics-btn').addEventListener('click', closeAnalyticsModal);
    document.getElementById('analytics-modal').addEventListener('click', (e) => {
        if (e.target === e.currentTarget) closeAnalyticsModal();
    });
    document.getElementById('run-analytics-btn').addEventListener('click', runAdvancedAnalytics);
});

// ─── Data Loading ──────────────────────────────────────────────────────────

function loadDashboard() {
    fetch('/api/analytics')
        .then(res => res.json())
        .then(data => {
            renderSidebarStats(data.summary);
            renderCharts(data);
        })
        .catch(err => console.error("Error fetching analytics:", err));
}

function loadImages() {
    const query = new URLSearchParams({
        page: currentPage,
        limit: currentLimit,
        search: currentSearch
    });
    
    fetch(`/api/images?${query.toString()}`)
        .then(res => res.json())
        .then(data => {
            globalClasses = data.classes;
            renderImageList(data.images, data.pagination);
        })
        .catch(err => console.error("Error fetching images:", err));
}

function handleSearch(e) {
    currentSearch = e.target.value.trim();
    currentPage = 1;
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(loadImages, 300); // 300ms debounce
}

function renderSidebarStats(summary) {
    const statsContainer = document.getElementById('sidebar-stats');
    statsContainer.innerHTML = `
        <div class="stat-box">
            <div class="value">${summary.total_images.toLocaleString()}</div>
            <div class="label">Images</div>
        </div>
        <div class="stat-box">
            <div class="value">${summary.total_bboxes.toLocaleString()}</div>
            <div class="label">BBoxes</div>
        </div>
    `;
}

function renderCharts(data) {
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Inter', sans-serif";
    
    if (classDistChartInstance) classDistChartInstance.destroy();
    if (bboxSizeChartInstance) bboxSizeChartInstance.destroy();
    if (splitDistChartInstance) splitDistChartInstance.destroy();

    // 1. Class Distribution Chart
    const distCtx = document.getElementById('classDistChart').getContext('2d');
    const classNames = Object.keys(data.class_distribution);
    const classCounts = Object.values(data.class_distribution);
    
    classDistChartInstance = new Chart(distCtx, {
        type: 'bar',
        data: {
            labels: classNames,
            datasets: [{
                label: 'Instances per Class',
                data: classCounts,
                backgroundColor: COLORS,
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                title: { display: true, text: 'Class Distribution', color: '#f8fafc', font: { size: 14 } }
            },
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true },
                x: { grid: { display: false } }
            }
        }
    });

    // 2. Average BBox Sizes Chart
    const sizeCtx = document.getElementById('bboxSizeChart').getContext('2d');
    const sortedClassNames = Object.keys(data.avg_bbox_sizes);
    const avgWidths = sortedClassNames.map(c => data.avg_bbox_sizes[c].w);
    const avgHeights = sortedClassNames.map(c => data.avg_bbox_sizes[c].h);

    bboxSizeChartInstance = new Chart(sizeCtx, {
        type: 'bar',
        data: {
            labels: sortedClassNames,
            datasets: [
                { label: 'Avg Width', data: avgWidths, backgroundColor: '#3b82f6', borderRadius: 4 },
                { label: 'Avg Height', data: avgHeights, backgroundColor: '#10b981', borderRadius: 4 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Average BBox Sizes (Relative)', color: '#f8fafc', font: { size: 14 } }
            },
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true, max: 1.0 },
                x: { grid: { display: false } }
            }
        }
    });

    // 3. Split Distribution Chart
    const splitData = data.split_distribution || {};
    const splitNames = Object.keys(splitData).map(s => s.charAt(0).toUpperCase() + s.slice(1));
    const splitImages = Object.values(splitData).map(s => s.images);

    if (splitNames.length > 0) {
        const splitCtx = document.getElementById('splitDistChart').getContext('2d');
        splitDistChartInstance = new Chart(splitCtx, {
            type: 'doughnut',
            data: {
                labels: splitNames,
                datasets: [{
                    data: splitImages,
                    backgroundColor: COLORS.slice(0, splitNames.length),
                    borderColor: '#1e293b',
                    borderWidth: 2,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: { display: true, text: 'Split Distribution (Images)', color: '#f8fafc', font: { size: 14 } },
                    legend: { position: 'bottom', labels: { padding: 16 } },
                    tooltip: {
                        callbacks: {
                            label: function(ctx) {
                                const split = Object.values(splitData)[ctx.dataIndex];
                                return ` ${ctx.label}: ${split.images} images (${split.percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }
}

function renderImageList(images, pagination) {
    const listContainer = document.getElementById('image-list');
    listContainer.innerHTML = ''; 
    
    images.forEach(img => {
        const li = document.createElement('li');
        li.innerHTML = `
            <span class="filename" title="${img.filename}">${img.filename}</span>
            <span class="badge">${img.bboxes.length > 0 ? img.bboxes.length : ''}</span>
        `;
        li.dataset.idx = img.idx;
        
        li.addEventListener('click', () => {
            document.querySelectorAll('.image-list li').forEach(el => el.classList.remove('active'));
            li.classList.add('active');
            loadImageIntoViewer(img);
        });
        
        listContainer.appendChild(li);
    });

    // Pagination elements
    const prevBtn = document.getElementById('page-prev-btn');
    const nextBtn = document.getElementById('page-next-btn');
    const indicator = document.getElementById('page-indicator');

    prevBtn.disabled = pagination.page <= 1;
    nextBtn.disabled = pagination.page >= pagination.total_pages;
    indicator.textContent = `Page ${pagination.page} of ${Math.max(1, pagination.total_pages)}`;
}

function loadImageIntoViewer(imgData) {
    document.getElementById('current-image-name').textContent = imgData.filename;
    document.getElementById('current-image-details').innerHTML = 
        `${imgData.width > 0 ? imgData.width : '?'} x ${imgData.height > 0 ? imgData.height : '?'} <br/> ${imgData.bboxes.length} Annotations`;
    
    const wrapper = document.getElementById('canvas-wrapper');
    const canvas = document.getElementById('annotation-canvas');
    const ctx = canvas.getContext('2d');
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const imgObj = new Image();
    imgObj.onload = () => {
        const maxWidth = wrapper.clientWidth - 48; 
        const maxHeight = wrapper.clientHeight - 48;
        
        const scale = Math.min(maxWidth / imgObj.width, maxHeight / imgObj.height);
        const finalWidth = imgObj.width * scale;
        const finalHeight = imgObj.height * scale;
        
        canvas.width = finalWidth;
        canvas.height = finalHeight;
        
        ctx.drawImage(imgObj, 0, 0, finalWidth, finalHeight);
        drawAnnotations(ctx, imgData.bboxes, finalWidth, finalHeight);
    };
    imgObj.src = `/api/image/${imgData.idx}`;
}

function drawAnnotations(ctx, bboxes, width, height) {
    bboxes.forEach(bbox => {
        const color = COLORS[bbox.class_id % COLORS.length];
        
        const boxW = bbox.width * width;
        const boxH = bbox.height * height;
        const startX = (bbox.x_center * width) - (boxW / 2);
        const startY = (bbox.y_center * height) - (boxH / 2);
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(startX, startY, boxW, boxH);
        
        const className = globalClasses[bbox.class_id] || `Class ${bbox.class_id}`;
        ctx.font = '12px Inter, sans-serif';
        const textWidth = ctx.measureText(className).width;
        
        ctx.fillStyle = color;
        ctx.fillRect(startX, startY - 18, textWidth + 8, 18);
        
        ctx.fillStyle = '#ffffff';
        ctx.fillText(className, startX + 4, startY - 5);
    });
}

// ─── Augmentation Modal ────────────────────────────────────────────────────

function openAugmentModal() {
    const modal = document.getElementById('augment-modal');
    modal.style.display = 'flex';
    document.getElementById('augment-status').textContent = '';

    fetch('/api/analytics')
        .then(res => res.json())
        .then(data => {
            const container = document.getElementById('class-select-list');
            container.innerHTML = '';
            for (const [className, count] of Object.entries(data.class_distribution)) {
                let classId = null;
                for (const [id, name] of Object.entries(globalClasses)) {
                    if (name === className || `Class ${id}` === className) { classId = parseInt(id); break; }
                }
                container.innerHTML += `
                    <label>
                        <input type="checkbox" name="aug-class" value="${classId !== null ? classId : className}" />
                        <span>${className}</span><span class="class-count">${count}</span>
                    </label>
                `;
            }
        });

    fetch('/api/augmentations')
        .then(res => res.json())
        .then(augList => {
            const container = document.getElementById('augmentation-select-list');
            container.innerHTML = '';
            augList.forEach(aug => {
                container.innerHTML += `
                    <label title="${aug.description}">
                        <input type="checkbox" name="aug-type" value="${aug.name}" /><span>${aug.label}</span>
                    </label>
                `;
            });
        });
}

function closeAugmentModal() {
    document.getElementById('augment-modal').style.display = 'none';
}

function applyAugmentations() {
    const classCbs = document.querySelectorAll('input[name="aug-class"]:checked');
    const augCbs = document.querySelectorAll('input[name="aug-type"]:checked');
    const numImages = parseInt(document.getElementById('aug-count-input').value) || 5;
    const strictFilter = document.getElementById('aug-strict-input').checked;
    const statusEl = document.getElementById('augment-status');
    const btn = document.getElementById('apply-augment-btn');

    if (classCbs.length === 0 || augCbs.length === 0) {
        statusEl.textContent = 'Select at least one class and augmentation.';
        statusEl.className = 'augment-status error';
        return;
    }

    const targetClasses = Array.from(classCbs).map(cb => parseInt(cb.value));
    const augmentations = Array.from(augCbs).map(cb => cb.value);

    statusEl.textContent = 'Generating…';
    statusEl.className = 'augment-status';
    btn.classList.add('loading');

    fetch('/api/augment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_classes: targetClasses, augmentations, num_images: numImages, strict_filter: strictFilter }),
    })
    .then(res => res.json())
    .then(data => {
        btn.classList.remove('loading');
        if (data.error) throw new Error(data.error);

        let msg = `✓ Generated ${data.generated_count} images.`;
        if (data.skipped_classes && data.skipped_classes.length > 0) {
            msg += ` ⚠ Skipped ${data.skipped_classes.length} class(es).`;
        }
        statusEl.textContent = msg;
        statusEl.className = 'augment-status success';
        
        loadDashboard();
        currentPage = 1;
        loadImages();
    })
    .catch(err => {
        btn.classList.remove('loading');
        statusEl.textContent = `Error: ${err.message}`;
        statusEl.className = 'augment-status error';
    });
}

// ─── Advanced Analytics Modal & Rendering ────────────────────────────────────

function openAnalyticsModal() {
    document.getElementById('analytics-modal').style.display = 'flex';
    document.getElementById('analytics-status').textContent = '';
}

function closeAnalyticsModal() {
    document.getElementById('analytics-modal').style.display = 'none';
}

function runAdvancedAnalytics() {
    const checked = document.querySelectorAll('input[name="analytic-opt"]:checked');
    const statusEl = document.getElementById('analytics-status');
    const btn = document.getElementById('run-analytics-btn');
    
    if (checked.length === 0) {
        statusEl.textContent = 'Select at least one metric.';
        statusEl.className = 'augment-status error';
        return;
    }

    const options = {};
    checked.forEach(cb => options[cb.value] = true);

    statusEl.textContent = 'Submitting job...';
    statusEl.className = 'augment-status';
    btn.classList.add('loading');

    fetch('/api/analytics/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ options })
    })
    .then(res => res.json())
    .then(data => {
        if (!data.job_id) throw new Error("No job ID returned");
        pollAnalyticsJob(data.job_id);
    })
    .catch(err => {
        btn.classList.remove('loading');
        statusEl.textContent = `Error: ${err.message}`;
        statusEl.className = 'augment-status error';
    });
}

function pollAnalyticsJob(jobId) {
    const statusEl = document.getElementById('analytics-status');
    const btn = document.getElementById('run-analytics-btn');
    
    fetch(`/api/analytics/jobs/${jobId}`)
        .then(res => res.json())
        .then(data => {
            if (data.status === 'completed') {
                btn.classList.remove('loading');
                statusEl.textContent = 'Success!';
                statusEl.className = 'augment-status success';
                setTimeout(() => {
                    closeAnalyticsModal();
                    renderAdvancedAnalyticsResults(data.result);
                }, 500);
            } else if (data.status === 'failed') {
                btn.classList.remove('loading');
                statusEl.textContent = `Job Failed: ${data.error}`;
                statusEl.className = 'augment-status error';
            } else {
                statusEl.textContent = 'Processing...';
                setTimeout(() => pollAnalyticsJob(jobId), 1000);
            }
        })
        .catch(err => {
            btn.classList.remove('loading');
            statusEl.textContent = `Polling Error: ${err.message}`;
            statusEl.className = 'augment-status error';
        });
}

function renderAdvancedAnalyticsResults(data) {
    const section = document.getElementById('advanced-analytics-section');
    const container = document.getElementById('advanced-results-container');
    
    section.style.display = 'flex';
    container.innerHTML = '';
    
    // Clear old dynamic charts
    advancedChartInstances.forEach(c => c.destroy());
    advancedChartInstances = [];

    const createHTMLCard = (title, content) => {
        const d = document.createElement('div');
        d.className = 'metric-card';
        d.innerHTML = `<h3>${title}</h3>${content}`;
        container.appendChild(d);
        return d;
    };

    const createChartCard = (title) => {
        const d = document.createElement('div');
        d.className = 'metric-card';
        d.style.height = '300px';
        d.innerHTML = `<h3>${title}</h3><div style="position:relative; flex:1;"><canvas></canvas></div>`;
        container.appendChild(d);
        return d.querySelector('canvas');
    };

    if (data.images_per_class) {
        let rows = Object.entries(data.images_per_class).map(([c, v]) => `<tr><td>${c}</td><td>${v}</td></tr>`).join('');
        createHTMLCard('Images per Class', `<table class="metric-table"><tr><th>Class</th><th>Images</th></tr>${rows}</table>`);
    }

    if (data.bbox_count_per_image) {
        const b = data.bbox_count_per_image;
        createHTMLCard('BBox Count per Image', `
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top:8px;">
                <div><span style="color:var(--text-secondary);font-size:0.8rem;">Min:</span> <br><span class="metric-value">${b.min}</span></div>
                <div><span style="color:var(--text-secondary);font-size:0.8rem;">Max:</span> <br><span class="metric-value">${b.max}</span></div>
                <div><span style="color:var(--text-secondary);font-size:0.8rem;">Mean:</span> <br><span class="metric-value">${b.mean}</span></div>
                <div><span style="color:var(--text-secondary);font-size:0.8rem;">Median:</span> <br><span class="metric-value">${b.median}</span></div>
            </div>
        `);
    }

    if (data.bbox_size_dist) {
        const ctx = createChartCard('BBox Size Distribution');
        advancedChartInstances.push(new Chart(ctx, {
            type: 'pie',
            data: { labels: ['Small', 'Medium', 'Large'], datasets: [{ data: [data.bbox_size_dist.small, data.bbox_size_dist.medium, data.bbox_size_dist.large], backgroundColor: COLORS }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } }
        }));
    }

    if (data.image_resolution_dist) {
        const ctx = createChartCard('Image Resolution Distribution');
        const resLabels = Object.keys(data.image_resolution_dist).slice(0, 10);
        const resData = Object.values(data.image_resolution_dist).slice(0, 10);
        advancedChartInstances.push(new Chart(ctx, {
            type: 'bar',
            data: { labels: resLabels, datasets: [{ label: 'Images', data: resData, backgroundColor: COLORS[0], borderRadius: 4 }] },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } } }
        }));
    }

    if (data.duplicate_detection) {
        const dd = data.duplicate_detection;
        let html = `
            <p style="margin-bottom:12px; font-size:0.9rem;">Found <strong>${dd.total_duplicate_images}</strong> instances spanning <strong>${dd.duplicate_groups}</strong> groups.</p>
            <table class="metric-table"><tr><th>MD5 Hash</th><th>Files</th></tr>`;
        Object.keys(dd.groups || {}).slice(0, 5).forEach(h => {
             html += `<tr><td style="font-family:monospace;font-size:0.7rem;">${h.slice(0,10)}...</td><td>${dd.groups[h].length} files</td></tr>`;
        });
        if (dd.duplicate_groups > 5) html += `<tr><td colspan="2">... and ${dd.duplicate_groups - 5} more</td></tr>`;
        html += `</table>`;
        createHTMLCard('Duplicate Detection (MD5)', html);
    }
    
    if (data.outlier_detection) {
        const od = data.outlier_detection;
        createHTMLCard('Outlier Detection (>2σ)', `
            <p style="margin-bottom:12px; font-size:0.9rem;">Found <strong>${od.total_outliers}</strong> extreme outliers.</p>
            <table class="metric-table"><tr><th>#</th><th>Area</th><th>Reason</th></tr>
            ${(od.outliers || []).slice(0, 5).map(o => `<tr><td>${o.index}</td><td>${o.area.toFixed(4)}</td><td style="color:#f59e0b;">${o.reasons.join(', ')}</td></tr>`).join('')}
            </table>
        `);
    }

    if (data.anchor_analysis) {
        createHTMLCard(`Anchor Box Analysis (K=${data.anchor_analysis.k})`, `
            <table class="metric-table"><tr><th>Idx</th><th>Width</th><th>Height</th><th>Area</th></tr>
            ${data.anchor_analysis.anchors.map((a, i) => `<tr><td>${i+1}</td><td>${a.width.toFixed(3)}</td><td>${a.height.toFixed(3)}</td><td>${(a.width*a.height).toFixed(4)}</td></tr>`).join('')}
            </table>
        `);
    }

    section.scrollIntoView({ behavior: 'smooth' });
}
