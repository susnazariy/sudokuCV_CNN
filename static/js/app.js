/**
 * Sudoku CV Solver - Web Interface
 */

// DOM Elements
const elements = {
    // Tabs
    tabBtns: document.querySelectorAll('.tab-btn'),
    tabContents: document.querySelectorAll('.tab-content'),
    
    // Image upload
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('file-input'),
    previewImage: document.getElementById('preview-image'),
    solveImageBtn: document.getElementById('solve-image-btn'),
    imageResult: document.getElementById('image-result'),
    resultImage: document.getElementById('result-image'),
    warpedResult: document.getElementById('warped-result'),
    detectedGrid: document.getElementById('detected-grid'),
    solutionGrid: document.getElementById('solution-grid'),
    
    // Manual input
    inputGrid: document.getElementById('input-grid'),
    solveManualBtn: document.getElementById('solve-manual-btn'),
    clearBtn: document.getElementById('clear-btn'),
    sampleBtn: document.getElementById('sample-btn'),
    manualResult: document.getElementById('manual-result'),
    manualSolutionGrid: document.getElementById('manual-solution-grid'),
    
    // UI
    loading: document.getElementById('loading'),
    errorMessage: document.getElementById('error-message')
};

// State
let currentImage = null;

// Sample puzzle for testing
const SAMPLE_PUZZLE = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initDropZone();
    initManualInput();
    initButtons();
});

// Tab Navigation
function initTabs() {
    elements.tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;
            
            // Update buttons
            elements.tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update content
            elements.tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === tabId) {
                    content.classList.add('active');
                }
            });
            
            hideError();
        });
    });
}

// Image Upload
function initDropZone() {
    const dropZone = elements.dropZone;
    
    // Drag events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults);
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('drag-over');
        });
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('drag-over');
        });
    });
    
    // Drop handler
    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
    
    // Click to upload
    dropZone.addEventListener('click', () => {
        elements.fileInput.click();
    });
    
    elements.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please upload an image file');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        currentImage = e.target.result;
        
        // Show preview
        elements.previewImage.src = currentImage;
        elements.previewImage.classList.remove('hidden');
        elements.dropZone.querySelector('.drop-zone-content').classList.add('hidden');
        elements.solveImageBtn.classList.remove('hidden');
        
        // Hide previous results
        elements.imageResult.classList.add('hidden');
        hideError();
    };
    reader.readAsDataURL(file);
}

// Manual Input Grid
function initManualInput() {
    createEditableGrid(elements.inputGrid);
}

function createEditableGrid(container) {
    container.innerHTML = '';
    
    for (let i = 0; i < 81; i++) {
        const cell = document.createElement('div');
        cell.className = 'sudoku-cell';
        
        const input = document.createElement('input');
        input.type = 'text';
        input.maxLength = 1;
        input.dataset.index = i;
        
        input.addEventListener('input', handleCellInput);
        input.addEventListener('keydown', handleCellKeydown);
        
        cell.appendChild(input);
        container.appendChild(cell);
    }
}

function handleCellInput(e) {
    const value = e.target.value;
    
    // Only allow 1-9
    if (value && (!/^[1-9]$/.test(value))) {
        e.target.value = '';
    }
}

function handleCellKeydown(e) {
    const index = parseInt(e.target.dataset.index);
    const row = Math.floor(index / 9);
    const col = index % 9;
    
    let newIndex = index;
    
    switch (e.key) {
        case 'ArrowUp':
            if (row > 0) newIndex = index - 9;
            e.preventDefault();
            break;
        case 'ArrowDown':
            if (row < 8) newIndex = index + 9;
            e.preventDefault();
            break;
        case 'ArrowLeft':
            if (col > 0) newIndex = index - 1;
            e.preventDefault();
            break;
        case 'ArrowRight':
            if (col < 8) newIndex = index + 1;
            e.preventDefault();
            break;
        case 'Backspace':
        case 'Delete':
            if (!e.target.value && col > 0) {
                newIndex = index - 1;
            }
            break;
    }
    
    if (newIndex !== index) {
        const inputs = elements.inputGrid.querySelectorAll('input');
        inputs[newIndex].focus();
    }
}

function getGridFromInputs() {
    const inputs = elements.inputGrid.querySelectorAll('input');
    const grid = [];
    
    for (let i = 0; i < 9; i++) {
        const row = [];
        for (let j = 0; j < 9; j++) {
            const value = inputs[i * 9 + j].value;
            row.push(value ? parseInt(value) : 0);
        }
        grid.push(row);
    }
    
    return grid;
}

function setGridToInputs(grid) {
    const inputs = elements.inputGrid.querySelectorAll('input');
    
    for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {
            const value = grid[i][j];
            inputs[i * 9 + j].value = value ? value : '';
        }
    }
}

// Display Grid (non-editable)
function displayGrid(container, grid, originalGrid = null) {
    container.innerHTML = '';
    
    for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {
            const cell = document.createElement('div');
            cell.className = 'sudoku-cell';
            
            const value = grid[i][j];
            cell.textContent = value || '';
            
            if (originalGrid) {
                if (originalGrid[i][j] !== 0) {
                    cell.classList.add('original');
                } else if (value !== 0) {
                    cell.classList.add('solved');
                }
            }
            
            container.appendChild(cell);
        }
    }
}

// Button Handlers
function initButtons() {
    elements.solveImageBtn.addEventListener('click', solveFromImage);
    elements.solveManualBtn.addEventListener('click', solveManual);
    elements.clearBtn.addEventListener('click', clearGrid);
    elements.sampleBtn.addEventListener('click', loadSample);
}

async function solveFromImage() {
    if (!currentImage) {
        showError('Please upload an image first');
        return;
    }
    
    showLoading();
    hideError();
    
    try {
        const response = await fetch('/api/solve/image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: currentImage })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Display results
            elements.resultImage.src = data.result_image;
            if (data.warped_result) {
                elements.warpedResult.src = data.warped_result;
                elements.warpedResult.style.display = 'block';
            } else {
                elements.warpedResult.style.display = 'none';
            }
            displayGrid(elements.detectedGrid, data.puzzle);
            displayGrid(elements.solutionGrid, data.solution, data.puzzle);
            elements.imageResult.classList.remove('hidden');
        } else {
            showError(data.error || 'Failed to solve puzzle');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function solveManual() {
    const grid = getGridFromInputs();
    
    // Check if grid has any numbers
    const hasNumbers = grid.some(row => row.some(cell => cell !== 0));
    if (!hasNumbers) {
        showError('Please enter some numbers first');
        return;
    }
    
    showLoading();
    hideError();
    
    try {
        const response = await fetch('/api/solve/grid', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ grid })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayGrid(elements.manualSolutionGrid, data.solution, grid);
            elements.manualResult.classList.remove('hidden');
        } else {
            showError(data.error || 'Failed to solve puzzle');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        hideLoading();
    }
}

function clearGrid() {
    const inputs = elements.inputGrid.querySelectorAll('input');
    inputs.forEach(input => input.value = '');
    elements.manualResult.classList.add('hidden');
    hideError();
}

function loadSample() {
    setGridToInputs(SAMPLE_PUZZLE);
    elements.manualResult.classList.add('hidden');
    hideError();
}

// UI Helpers
function showLoading() {
    elements.loading.classList.remove('hidden');
}

function hideLoading() {
    elements.loading.classList.add('hidden');
}

function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorMessage.classList.remove('hidden');
}

function hideError() {
    elements.errorMessage.classList.add('hidden');
}
