# PDF Question Answering System - User Guide

This guide provides instructions on how to install, configure, and use the PDF Question Answering System.

## Installation

### Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)
- Sufficient disk space for document storage and vector database
- 4GB+ RAM recommended for processing larger documents

### Installation Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/pdf-qa-system.git
   cd pdf-qa-system
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install Spacy language model**

   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Verify installation**

   ```bash
   python run_api.py
   ```

   If successful, you should see output indicating that the API is running on http://127.0.0.1:8000.

## Configuration

The system can be configured via the `config.yaml` file in the root directory. Important configuration options include:

### Memory Settings

```yaml
memory:
  max_usage_mb: 400
  cleanup_threshold_mb: 350
  tracking_interval_sec: 1.0
```

- `max_usage_mb`: Maximum memory usage before cleanup
- `cleanup_threshold_mb`: Memory threshold to trigger cleanup
- `tracking_interval_sec`: Interval for memory tracking

### Processing Settings

```yaml
processing:
  batch_size: 1000
  max_workers: 4
  chunk_size: 512
  embedding_size: 768
```

- `batch_size`: Number of text chunks to process in a batch
- `max_workers`: Maximum number of concurrent processing workers
- `chunk_size`: Size of text chunks for processing
- `embedding_size`: Dimension of the embedding vectors

## Basic Usage

### Starting the System

1. From the project directory, run:

   ```bash
   python run_api.py
   ```

2. Open your web browser and navigate to:

   ```
   http://localhost:8000
   ```

### Uploading Documents

1. From the main interface, click the "Upload Document" button.
2. Select a PDF file from your computer.
3. Wait for the document to be processed. Processing time depends on the document size.
4. Once processing is complete, the document will appear in your document list.

### Asking Questions

1. Select a document from your document list.
2. Type your question in the question input field.
3. Click "Ask" or press Enter.
4. The system will search the document and display relevant answers.

### Adjusting Search Parameters

You can adjust search parameters for each query:

- **Similarity Threshold**: Controls how closely the content must match your question (default: 0.45)
- **Max Results**: Maximum number of text chunks to return (default: 5)

Lower similarity thresholds will return more results but may be less relevant.

## Multilingual Support

The system automatically detects the language of your questions and documents. You can:

- Upload documents in any supported language
- Ask questions in any supported language
- Receive answers in the same language as your question

No special configuration is needed for multilingual support.

## Advanced Features

### Document Management

- **Document List**: View all uploaded documents
- **Document Details**: View metadata including upload time, size, and page count
- **Document Deletion**: Remove documents from the system

### Response Options

- **Context View**: See the surrounding text for each answer
- **Page References**: View the page numbers where answers were found
- **Confidence Scores**: See how confident the system is about each answer

## Troubleshooting

### Common Issues

1. **Document Processing Failures**
   - Ensure the PDF is not password-protected
   - Check that the file is a valid PDF
   - Try splitting very large documents into smaller files

2. **No Relevant Answers**
   - Try rephrasing your question
   - Lower the similarity threshold
   - Check that the document contains information related to your question

3. **High Memory Usage**
   - Adjust the memory settings in the configuration file
   - Process fewer documents simultaneously
   - Use smaller chunk sizes for very large documents

4. **API Connection Issues**
   - Verify the server is running
   - Check for firewall or network restrictions
   - Ensure the correct port is being used (default: 8000)

## API Usage

For developers, the system provides a REST API that can be integrated with other applications:

- `POST /upload`: Upload a document
- `GET /documents`: List all documents
- `POST /ask`: Ask a question about a document
- `DELETE /document/{doc_id}`: Delete a document

See the API documentation for more details on these endpoints. 