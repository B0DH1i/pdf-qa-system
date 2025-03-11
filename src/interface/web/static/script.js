document.addEventListener('DOMContentLoaded', () => {
    // API endpoints
    const API_URL = '';
    const UPLOAD_ENDPOINT = '/upload';
    const DOCUMENTS_ENDPOINT = '/documents';
    const QUESTION_ENDPOINT = '/ask';

    // DOM elements
    const uploadForm = document.getElementById('upload-form');
    const fileDropArea = document.querySelector('.file-drop-area');
    const fileInput = document.querySelector('.file-input');
    const uploadStatus = document.getElementById('upload-status');
    const documentsList = document.getElementById('documents-list');
    const docSelector = document.getElementById('doc-selector');
    const questionForm = document.getElementById('question-form');
    const questionInput = document.getElementById('question-input');
    const answerContent = document.getElementById('answer-content');

    // Drag & drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        fileDropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        fileDropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        fileDropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        fileDropArea.classList.add('is-active');
    }

    function unhighlight() {
        fileDropArea.classList.remove('is-active');
    }

    fileDropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length > 0) {
            const fileMsg = document.querySelector('.file-msg');
            fileMsg.textContent = files[0].name;
        }
    }

    fileInput.addEventListener('change', () => {
        handleFiles(fileInput.files);
    });

    // PDF upload process
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!fileInput.files[0]) {
            showMessage(uploadStatus, 'Please select a PDF file', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        showMessage(uploadStatus, 'Uploading...', 'info');

        try {
            const response = await fetch(API_URL + UPLOAD_ENDPOINT, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.status}`);
            }

            const data = await response.json();
            showMessage(uploadStatus, 'PDF uploaded successfully!', 'success');
            loadDocuments();
        } catch (error) {
            showMessage(uploadStatus, `Upload error: ${error.message}`, 'error');
        }
    });

    // Load documents
    async function loadDocuments() {
        try {
            const response = await fetch(API_URL + DOCUMENTS_ENDPOINT);
            
            if (!response.ok) {
                throw new Error(`Error: ${response.status}`);
            }

            const documents = await response.json();
            
            // Update document list
            if (documents.length === 0) {
                documentsList.innerHTML = '<p class="empty-message">No documents uploaded yet</p>';
                docSelector.innerHTML = '<option value="">Select a document</option>';
            } else {
                documentsList.innerHTML = '';
                docSelector.innerHTML = '<option value="">Select a document</option>';
                
                documents.forEach(doc => {
                    // Create document card
                    const docCard = document.createElement('div');
                    docCard.className = 'document-card';
                    
                    const docInfo = document.createElement('div');
                    docInfo.className = 'document-info';
                    
                    const docTitle = document.createElement('h3');
                    docTitle.textContent = doc.filename;
                    
                    const docMeta = document.createElement('div');
                    docMeta.className = 'document-meta';
                    docMeta.innerHTML = `
                        <span>Upload: ${doc.upload_time}</span> | 
                        <span>Page: ${doc.page_count}</span> | 
                        <span>Size: ${formatFileSize(doc.file_size)}</span>
                    `;
                    
                    docInfo.appendChild(docTitle);
                    docInfo.appendChild(docMeta);
                    docCard.appendChild(docInfo);
                    
                    documentsList.appendChild(docCard);
                    
                    // Add document selector
                    const option = document.createElement('option');
                    option.value = doc.doc_id;
                    option.textContent = doc.filename;
                    docSelector.appendChild(option);
                });
            }
        } catch (error) {
            documentsList.innerHTML = `<p class="error-message">Documents could not be loaded: ${error.message}</p>`;
        }
    }

    // Ask a question
    questionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const docId = docSelector.value;
        const question = questionInput.value.trim();
        
        if (!docId) {
            showMessage(answerContent, 'Please select a document', 'error');
            return;
        }
        
        if (!question) {
            showMessage(answerContent, 'Please enter a question', 'error');
            return;
        }
        
        // Show loading
        answerContent.innerHTML = '<p class="info-message">Answering...</p>';
        
        try {
            const response = await fetch(API_URL + QUESTION_ENDPOINT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question,
                    doc_id: docId,
                    threshold: 0.7,
                    max_results: 5
                }),
            });
            
            if (!response.ok) {
                throw new Error(`Error: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Display answer
            displayAnswer(data);
        } catch (error) {
            showMessage(answerContent, `Question-answer error: ${error.message}`, 'error');
        }
    });

    // Display answer
    function displayAnswer(data) {
        let html = `
            <div class="answer-box">
                <p>${data.answer}</p>
            </div>
        `;
        
        if (data.context && data.context.length > 0) {
            html += `
                <div class="context-box">
                    <h4>Relevant Content:</h4>
                    <ul>
            `;
            
            data.context.forEach(ctx => {
                html += `<li>${ctx}</li>`;
            });
            
            html += `
                    </ul>
                </div>
            `;
        }
        
        if (data.sources && data.sources.length > 0) {
            html += `
                <div class="sources-box">
                    <h4>Sources:</h4>
                    <ul>
            `;
            
            data.sources.forEach(source => {
                html += `<li>Page ${source.page}: ${source.text.substring(0, 100)}...</li>`;
            });
            
            html += `
                    </ul>
                </div>
            `;
        }
        
        html += `<p class="processing-time">Processing time: ${data.processing_time.toFixed(2)} seconds</p>`;
        
        answerContent.innerHTML = html;
    }

    // Helper functions
    function showMessage(element, message, type) {
        element.innerHTML = `<p class="${type}-message">${message}</p>`;
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        else return (bytes / 1048576).toFixed(1) + ' MB';
    }

    // Load documents when page loads
    loadDocuments();
}); 