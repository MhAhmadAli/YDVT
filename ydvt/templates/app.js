const COLORS = [
    '#3b82f6', '#ef4444', '#10b981', '#f59e0b', 
    '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'
];

let globalClasses = {};
let activeImageIdx = null;

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    // 1. Fetch and render Analytics
    fetch('/api/analytics')
        .then(res => res.json())
        .then(data => {
            renderSidebarStats(data.summary);
            renderCharts(data);
        })
        .catch(err => console.error("Error fetching analytics:", err));

    // 2. Fetch and render Image List
    fetch('/api/images')
        .then(res => res.json())
        .then(data => {
            globalClasses = data.classes;
            renderImageList(data.images);
            setupSearch(data.images);
        })
        .catch(err => console.error("Error fetching images:", err));
});

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
    // Styling constants for charts matching CSS dark mode
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Inter', sans-serif";
    
    // 1. Class Distribution Chart
    const distCtx = document.getElementById('classDistChart').getContext('2d');
    const classNames = Object.keys(data.class_distribution);
    const classCounts = Object.values(data.class_distribution);
    
    new Chart(distCtx, {
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
    
    // Sort class names for consistency
    const sortedClassNames = Object.keys(data.avg_bbox_sizes);
    const avgWidths = sortedClassNames.map(c => data.avg_bbox_sizes[c].w);
    const avgHeights = sortedClassNames.map(c => data.avg_bbox_sizes[c].h);

    new Chart(sizeCtx, {
        type: 'bar',
        data: {
            labels: sortedClassNames,
            datasets: [
                {
                    label: 'Avg Width',
                    data: avgWidths,
                    backgroundColor: '#3b82f6',
                    borderRadius: 4,
                },
                {
                    label: 'Avg Height',
                    data: avgHeights,
                    backgroundColor: '#10b981',
                    borderRadius: 4,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Average Bounding Box Sizes (Relative)', color: '#f8fafc', font: { size: 14 } }
            },
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true, max: 1.0 },
                x: { grid: { display: false } }
            }
        }
    });
}

function renderImageList(images) {
    const listContainer = document.getElementById('image-list');
    listContainer.innerHTML = ''; // Clear
    
    images.forEach(img => {
        const li = document.createElement('li');
        li.innerHTML = `
            <span class="filename" title="${img.filename}">${img.filename}</span>
            <span class="badge">${img.bboxes.length > 0 ? img.bboxes.length : ''}</span>
        `;
        li.dataset.idx = img.idx;
        
        li.addEventListener('click', () => {
            // Update active state
            document.querySelectorAll('.image-list li').forEach(el => el.classList.remove('active'));
            li.classList.add('active');
            
            loadImageIntoViewer(img);
        });
        
        listContainer.appendChild(li);
    });
}

function setupSearch(images) {
    const searchInput = document.getElementById('search-input');
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        const filtered = images.filter(img => img.filename.toLowerCase().includes(query));
        renderImageList(filtered);
    });
}

function loadImageIntoViewer(imgData) {
    document.getElementById('current-image-name').textContent = imgData.filename;
    document.getElementById('current-image-details').innerHTML = 
        `${imgData.width > 0 ? imgData.width : '?'} x ${imgData.height > 0 ? imgData.height : '?'} <br/> ${imgData.bboxes.length} Annotations`;
    
    const wrapper = document.getElementById('canvas-wrapper');
    const canvas = document.getElementById('annotation-canvas');
    const ctx = canvas.getContext('2d');
    
    // Clear previous
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Create an invisible image to load the source
    const imgObj = new Image();
    imgObj.onload = () => {
        // We want to fit the image inside the wrapper while maintaining aspect ratio
        const maxWidth = wrapper.clientWidth - 48; // padding
        const maxHeight = wrapper.clientHeight - 48;
        
        const scale = Math.min(maxWidth / imgObj.width, maxHeight / imgObj.height);
        const finalWidth = imgObj.width * scale;
        const finalHeight = imgObj.height * scale;
        
        canvas.width = finalWidth;
        canvas.height = finalHeight;
        
        // Draw the image onto the canvas (no img tag needed if we draw everything on canvas)
        ctx.drawImage(imgObj, 0, 0, finalWidth, finalHeight);
        
        // Draw annotations
        drawAnnotations(ctx, imgData.bboxes, finalWidth, finalHeight);
    };
    imgObj.src = `/api/image/${imgData.idx}`;
}

function drawAnnotations(ctx, bboxes, width, height) {
    bboxes.forEach(bbox => {
        const color = COLORS[bbox.class_id % COLORS.length];
        
        // YOLO coords are relative center_x, center_y, width, height
        const boxW = bbox.width * width;
        const boxH = bbox.height * height;
        const startX = (bbox.x_center * width) - (boxW / 2);
        const startY = (bbox.y_center * height) - (boxH / 2);
        
        // Draw Box
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(startX, startY, boxW, boxH);
        
        // Draw Label Background
        const className = globalClasses[bbox.class_id] || `Class ${bbox.class_id}`;
        ctx.font = '12px Inter, sans-serif';
        const textWidth = ctx.measureText(className).width;
        
        ctx.fillStyle = color;
        ctx.fillRect(startX, startY - 18, textWidth + 8, 18);
        
        // Draw Label Text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(className, startX + 4, startY - 5);
    });
}
