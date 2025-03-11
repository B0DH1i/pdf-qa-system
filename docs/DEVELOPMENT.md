# Development Guide for PDF Question Answering System

This document provides guidance for developers who want to set up, develop, and contribute to the PDF Question Answering System.

## Setting Up the Development Environment

### Prerequisites

- Python 3.8 or higher
- Git
- pip (Python package manager)
- Virtual environment tool (venv or conda)

### Clone the Repository

```bash
git clone https://github.com/yourusername/pdf-qa-system.git
cd pdf-qa-system
```

### Create and Activate a Virtual Environment

#### Using venv (Python's built-in virtual environment)

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

#### Using conda

```bash
# Create conda environment
conda create -n pdf-qa python=3.9
conda activate pdf-qa
```

### Install Dependencies

```bash
# Install development dependencies
pip install -e ".[dev]"

# Or install all dependencies directly
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Project Structure

```
pdf-qa-system/
├── src/                 # Source code
│   ├── interface/       # User interface components
│   │   ├── web/         # Web API and UI
│   │   └── cli/         # Command-line interface
│   ├── processing/      # Data processing modules
│   │   ├── document.py  # Document processing
│   │   └── text.py      # Text processing
│   ├── storage/         # Storage components
│   │   ├── document_store.py  # Document storage
│   │   └── vector_store.py    # Vector storage
│   └── models/          # ML models and interfaces
│       ├── embedding.py # Embedding models
│       └── qa.py        # Question answering models
├── tests/               # Test suite
├── docs/                # Documentation
├── examples/            # Example code and notebooks
├── scripts/             # Utility scripts
├── requirements.txt     # Production dependencies
├── requirements-dev.txt # Development dependencies
└── setup.py             # Package configuration
```

## Development Workflow

### Running in Development Mode

```bash
# Start the web interface in development mode
python run_api.py --dev

# Development mode enables:
# - Auto-reload on code changes
# - Detailed error messages
# - Debug logging
```

### Code Style and Linting

The project follows PEP 8 style guidelines with a few modifications defined in the pyproject.toml file.

```bash
# Run linting checks
flake8 src tests

# Format code automatically
black src tests

# Sort imports
isort src tests
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src

# Run specific test file
pytest tests/test_document_processor.py

# Run tests in parallel
pytest -xvs -n auto
```

## Making Changes

### Branching Strategy

We use a feature branch workflow:

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # or 
   git checkout -b fix/bug-description
   ```

2. Make your changes and commit them with descriptive messages:
   ```bash
   git add .
   git commit -m "Add feature: detailed description of your changes"
   ```

3. Push your branch to the remote repository:
   ```bash
   git push -u origin feature/your-feature-name
   ```

4. Create a pull request from your branch to the main branch.

### Commit Guidelines

- Use clear, descriptive commit messages
- Start with a verb in the imperative mood (e.g., "Add", "Fix", "Update")
- Reference issue numbers when applicable (e.g., "Fix #123: Improve PDF text extraction")

## Documentation

### Code Documentation

- Document all public modules, classes, and functions using Google-style docstrings
- Document complex algorithms and decisions inline using comments

Example:

```python
def extract_text_from_pdf(pdf_path, extraction_method='pdfminer'):
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        extraction_method (str): Method to use for extraction ('pdfminer' or 'pypdf')
        
    Returns:
        list: List of strings, where each string is the text content of a page
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        ValueError: If the extraction method is not supported
    """
    # Implementation...
```

### Updating Documentation

When making changes, make sure to update the relevant documentation:

1. Update docstrings for modified code
2. Update relevant Markdown files in the `docs/` directory
3. Update examples if your changes affect the API

## Working with Models

### Using Custom Embedding Models

The system is designed to work with different embedding models:

```python
from src.models.embedding import EmbeddingModel

# Create a custom embedding model by extending the base class
class CustomEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        # Custom initialization
        
    def embed_texts(self, texts):
        # Custom embedding logic
        return embeddings
```

### Testing with Smaller Models

For development, you can use smaller models to speed up testing:

```python
# In your development configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Smaller, faster model
```

## Database Setup for Development

### Local Vector Database

For development, the system uses a local ChromaDB instance:

```python
# This is set up automatically, but you can customize:
from src.storage.vector_store import VectorStore

vector_store = VectorStore(
    collection_name="dev_collection",
    persist_directory="./data/chroma_db",
    embedding_function=your_embedding_function
)
```

### Working with Test Data

Sample data is provided in the `examples/data/` directory:

```bash
# Load sample data for development
python scripts/load_sample_data.py
```

## Debugging

### Logging

The system uses Python's built-in logging module with different levels:

```python
import logging

# Set log level
logging.basicConfig(level=logging.DEBUG)

# In your code
logger = logging.getLogger(__name__)
logger.debug("Detailed debugging information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
```

### Profiling

For performance optimization:

```bash
# Profile the API
python -m cProfile -o profile_output.prof run_api.py

# Analyze results
python scripts/analyze_profile.py profile_output.prof
```

## Building and Packaging

### Creating a Distribution Package

```bash
# Build the package
python -m build

# This creates:
# - dist/*.whl (wheel package)
# - dist/*.tar.gz (source distribution)
```

### Creating a Docker Image

```bash
# Build Docker image
docker build -t pdf-qa-system:latest .

# Run the Docker container
docker run -p 8000:8000 pdf-qa-system:latest
```

## Contributing

### Contribution Process

1. Check existing issues or create a new one describing your planned contribution
2. Fork the repository and create a feature branch
3. Make your changes, following the code style and adding tests
4. Submit a pull request targeting the main branch
5. Address any feedback from the code review

### Pull Request Checklist

- [ ] Code follows the project's style guidelines
- [ ] Unit tests added/updated for new functionality
- [ ] Documentation updated
- [ ] All tests pass
- [ ] No new warnings introduced
- [ ] Changes are backward compatible (or breaking changes are clearly documented)

## Troubleshooting Common Development Issues

### PDF Processing Issues

If you encounter issues with PDF text extraction:

- Try different extraction methods (`pdfminer.six` vs `PyPDF2`)
- Check if the PDF is scanned or contains images (may require OCR)
- Verify PDF permissions (some PDFs are secured)

### Vector Database Issues

If ChromaDB is causing issues:

- Check the ChromaDB log at `./data/chroma_db/chroma.log`
- Verify the compatibility of the embedding dimensions
- Try refreshing the collection by deleting `./data/chroma_db` and rebuilding

### Model Loading Problems

If you have issues loading models:

- Check internet connectivity (models may need to be downloaded)
- Verify enough disk space for model storage
- Check compatibility between the transformers library and the model version

## Performance Optimization Tips

- Use batching for document processing and embedding generation
- Cache embeddings of frequently accessed documents
- Use smaller models during development
- Profile your code to identify bottlenecks

## Resources

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [PyPDF2 Documentation](https://pypdf2.readthedocs.io/)
- [Sentence-Transformers Documentation](https://www.sbert.net/) 