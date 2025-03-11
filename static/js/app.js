/**
 * PDF Question-Answer System Main JavaScript File
 * This file handles user interactions and API requests in the web interface.
 */

// API URLs
const API_BASE_URL = '';  // If empty, same origin is used
const API_ENDPOINTS = {
    UPLOAD: `${API_BASE_URL}/upload`,
    ASK: `${API_BASE_URL}/ask`,
    DOCUMENTS: `${API_BASE_URL}/documents`
};

// DOM Elements
const uploadForm = document.getElementById('upload-form');
const uploadStatus = document.getElementById('upload-status');
const questionForm = document.getElementById('question-form');
const answerContainer = document.getElementById('answer-container');
const sourcesList = document.getElementById('sources-list');
const documentsContainer = document.getElementById('documents-container');

// When page loads
document.addEventListener('DOMContentLoaded', () => {
    // Load existing documents
    loadDocuments();
    
    // Capture form submissions
    uploadForm.addEventListener('submit', handleUpload);
    questionForm.addEventListener('submit', handleQuestion);
});

/**
 * Handles file upload process
 */
async function handleUpload(event) {
    event.preventDefault();
    
    const fileInput = document.getElementById('file');
    const file = fileInput.files[0];
    
    if (!file) {
        showMessage(uploadStatus, 'Please select a PDF file', 'danger');
        return;
    }
    
    // Check file type
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        showMessage(uploadStatus, 'Please upload PDF files only', 'danger');
        return;
    }
    
    // Show upload status
    showMessage(uploadStatus, 'Uploading document...', 'info');
    showLoading(uploadStatus);
    
    try {
        // Upload the file
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(API_ENDPOINTS.UPLOAD, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Upload error: ${response.status} ${response.statusText}`);
        }
        
        const result = await response.json();
        
        // Log API response
        console.log("Upload API response:", result);
        console.log("Upload API response type:", typeof result);
        console.log("Upload API response has doc_id:", result.hasOwnProperty('doc_id'));
        
        // Successful upload message
        if (result && result.doc_id) {
            showMessage(uploadStatus, `Document uploaded successfully. ID: ${result.doc_id}`, 'success');
        } else if (result && typeof result === 'object') {
            // Response is an object but has no doc_id - find the first key and use it (for possible data structure issues)
            const firstKey = Object.keys(result)[0];
            const possibleDocId = result[firstKey]?.doc_id || 'unknown';
            showMessage(uploadStatus, `Document uploaded. ID: ${possibleDocId}`, 'success');
            
            // Update original result
            result.doc_id = possibleDocId;
        } else {
            showMessage(uploadStatus, `Document uploaded but couldn't retrieve ID.`, 'warning');
        }
        
        // Update document list
        loadDocuments();
        
        // Clear form
        uploadForm.reset();
        
    } catch (error) {
        console.error('Upload error:', error);
        showMessage(uploadStatus, `Error: ${error.message}`, 'danger');
    } finally {
        hideLoading(uploadStatus);
    }
}

/**
 * Handles question processing
 */
async function handleQuestion(event) {
    event.preventDefault();
    
    const questionInput = document.getElementById('question');
    const question = questionInput.value.trim();
    
    if (!question) {
        showMessage(answerContainer, 'Please enter a question', 'danger');
        return;
    }
    
    // Show loading status
    showMessage(answerContainer, 'Searching for answer...', 'info');
    showLoading(answerContainer);
    sourcesList.innerHTML = '';
    
    try {
        // Submit the question
        const response = await fetch(API_ENDPOINTS.ASK, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }
        
        const result = await response.json();
        
        // Display answer
        displayAnswer(result);
        
    } catch (error) {
        console.error('Question error:', error);
        showMessage(answerContainer, `Error: ${error.message}`, 'danger');
        sourcesList.innerHTML = '';
    } finally {
        hideLoading(answerContainer);
    }
}

/**
 * Displays the answer and sources
 */
function displayAnswer(result) {
    // Clear previous content
    answerContainer.innerHTML = '';
    sourcesList.innerHTML = '';
    
    if (!result) {
        showMessage(answerContainer, 'No answer received from the API', 'warning');
        return;
    }
    
    // Create answer container
    const answerDiv = document.createElement('div');
    answerDiv.className = 'answer-text';
    
    // If the result has an 'answer' property, use it
    if (result.answer) {
        answerDiv.textContent = result.answer;
    } 
    // If the result has a 'text' property, use it
    else if (result.text) {
        answerDiv.textContent = result.text;
    }
    // If the result itself is a string
    else if (typeof result === 'string') {
        answerDiv.textContent = result;
    }
    // If the result is an array (probably search results)
    else if (Array.isArray(result)) {
        answerDiv.textContent = "Found relevant content in the document:";
        
        // Display sources
        displaySources(result);
    }
    // Fallback: stringify the result
    else {
        answerDiv.textContent = JSON.stringify(result, null, 2);
    }
    
    answerContainer.appendChild(answerDiv);
    
    // If the result has documents/sources, display them
    if (result.documents) {
        displaySources(result.documents, result.distances);
    }
    
    // If explicit sources are provided
    if (result.sources) {
        displaySources(result.sources);
    }
}

/**
 * Displays source documents and their relevance
 */
function displaySources(sources, distances = []) {
    // Clear previous sources
    sourcesList.innerHTML = '';
    
    if (!sources || sources.length === 0) {
        const noSourcesEl = document.createElement('p');
        noSourcesEl.className = 'text-muted';
        noSourcesEl.textContent = 'No source documents found';
        sourcesList.appendChild(noSourcesEl);
        return;
    }
    
    // Create source list
    const sourcesUl = document.createElement('ul');
    sourcesUl.className = 'list-group';
    
    sources.forEach((source, index) => {
        const sourceItem = document.createElement('li');
        sourceItem.className = 'list-group-item';
        
        // Handle different source formats
        let sourceText = '';
        let sourceMetadata = '';
        
        if (typeof source === 'string') {
            sourceText = source;
        } else if (source.text) {
            sourceText = source.text;
            
            // Add metadata if available
            if (source.metadata) {
                sourceMetadata = `
                    <small class="text-muted d-block mt-1">
                        Document: ${source.metadata.filename || 'N/A'}, 
                        Page: ${source.metadata.page_num || 'N/A'}
                    </small>
                `;
            }
        } else if (source.content) {
            sourceText = source.content;
        } else {
            sourceText = JSON.stringify(source);
        }
        
        // Add distance/relevance if available
        let relevance = '';
        if (distances && distances[index] !== undefined) {
            const distance = distances[index];
            const percent = Math.round((1 - Math.min(distance, 1)) * 100);
            relevance = `
                <div class="progress mt-2" style="height: 5px;">
                    <div class="progress-bar bg-success" role="progressbar" 
                        style="width: ${percent}%;" 
                        aria-valuenow="${percent}" aria-valuemin="0" aria-valuemax="100">
                    </div>
                </div>
                <small class="text-muted">Relevance: ${percent}%</small>
            `;
        }
        
        sourceItem.innerHTML = `
            <div class="source-text">${sourceText}</div>
            ${sourceMetadata}
            ${relevance}
        `;
        
        sourcesUl.appendChild(sourceItem);
    });
    
    sourcesList.appendChild(sourcesUl);
}

/**
 * Lists currently uploaded documents
 */
async function loadDocuments() {
    try {
        const response = await fetch('/documents');
        
        const documents = await response.json();
        console.log("Documents API response:", documents); // Debug log
        
        // Show document list
        if (documents && documents.length > 0) {
            documentsContainer.innerHTML = '';
            
            documents.forEach(doc => {
                console.log("Processing document:", doc); // Debug log for each document
                
                // Skip if doc is undefined or missing required fields
                if (!doc || typeof doc !== 'object') {
                    console.warn("Invalid document data:", doc);
                    return; // Skip this document
                }
                
                const docDiv = document.createElement('div');
                docDiv.className = 'document-item mb-3 p-3 border rounded';
                
                docDiv.innerHTML = `
                    <h6>${doc.filename || 'Unnamed Document'}</h6>
                    <div class="document-info">
                        <span class="badge bg-info me-2">ID: ${doc.doc_id ? doc.doc_id.substring(0, 8) : 'N/A'}...</span>
                        <span class="badge bg-secondary me-2">${doc.page_count || 0} pages</span>
                        <span class="badge bg-primary me-2">${doc.file_size ? formatBytes(doc.file_size) : '0 Bytes'}</span>
                        <span class="badge ${doc.status === 'completed' ? 'bg-success' : 'bg-warning'}">${doc.status || 'unknown'}</span>
                    </div>
                    <div class="mt-2">
                        <small class="text-muted">Uploaded: ${doc.upload_time || 'unknown'}</small>
                    </div>
                `;
                
                documentsContainer.appendChild(docDiv);
            });
        } else {
            documentsContainer.innerHTML = '<p class="text-muted">No documents uploaded yet.</p>';
        }
        
    } catch (error) {
        console.error('Document listing error:', error);
        documentsContainer.innerHTML = `<p class="text-danger">Error loading documents: ${error.message}</p>`;
    }
}

/**
 * Displays a message in the specified container
 */
function showMessage(container, message, type = 'info') {
    container.innerHTML = '';
    
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;
    
    container.appendChild(alertDiv);
}

/**
 * Shows loading animation
 */
function showLoading(element) {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading mt-2';
    element.appendChild(loadingDiv);
}

/**
 * Hides loading animation
 */
function hideLoading(element) {
    const loadingEl = element.querySelector('.loading');
    if (loadingEl) {
        loadingEl.remove();
    }
}

/**
 * Formats bytes to human-readable format
 */
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
} 