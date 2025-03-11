# PDF Question Answering System

A powerful document question-answering system that allows users to upload PDF documents and ask questions about their content. The system uses vector similarity search and natural language processing to provide accurate answers based on the document content.

## Features

- **PDF Document Processing**: Upload and process PDF documents for question answering
- **Semantic Search**: Advanced vector-based similarity search to find relevant content
- **Multilingual Support**: Works with multiple languages through a multilingual embedding model
- **Web Interface**: User-friendly web interface for easy interaction
- **Detailed Logging**: Comprehensive logging for troubleshooting

## Installation

### Prerequisites

- Python 3.8+
- Git

### Step 1: Clone the repository

```bash
git clone https://github.com/B0DH1i/pdf-qa-system.git
cd pdf-qa-system
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Starting the API

Run the following command to start the API:

```bash
python run_api.py
```

This will start the server on http://localhost:8000

### Using the Web Interface

1. Open your browser and navigate to http://localhost:8000
2. Upload a PDF document using the upload button
3. Once the document is processed, you can ask questions about its content
4. The system will search for relevant information and provide answers based on the document content

### Configuration

You can configure the system by modifying the following parameters:

- **Similarity Threshold**: The default similarity threshold is 0.45, which can be adjusted for more or less strict matching
- **Max Results**: The default number of results returned is 5, which can be increased for more comprehensive answers

## Technical Details

The system consists of several components:

1. **Document Processor**: Handles PDF parsing and text extraction
2. **Text Processor**: Converts text to embeddings using a multilingual transformer model
3. **Vector Store**: Stores and retrieves document embeddings for similarity search
4. **API**: Provides an interface for the web client

## Troubleshooting

If you encounter issues with the system:

1. **No information found**: Try rephrasing your question or using more specific terms
2. **Slow processing**: Large documents may take longer to process initially, but subsequent queries should be faster
3. **Memory issues**: If you're processing very large documents, ensure your system has sufficient RAM

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
