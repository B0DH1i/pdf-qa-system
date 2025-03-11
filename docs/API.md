# PDF Question Answering System - API Documentation

This document provides comprehensive details about the REST API endpoints available in the PDF QA System.

## API Overview

The API is built with FastAPI and provides endpoints for document management and question answering. All endpoints use JSON for data exchange except for the file upload endpoint, which uses multipart/form-data.

Base URL: `http://localhost:8000`

## Authentication

The API currently does not implement authentication. If you are deploying this in a production environment, it is recommended to implement an authentication mechanism.

## API Endpoints

### Document Management

#### Upload a Document

Upload a PDF document for processing.

- **URL**: `/upload`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`

**Request Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | File | Yes | The PDF file to upload |

**Response**:

```json
{
  "doc_id": "f82a7d8e-6b3c-4d2e-9b5a-3a7f8c92e1a5",
  "filename": "sample.pdf",
  "upload_time": "2023-03-11T15:30:45",
  "page_count": 15,
  "status": "processed",
  "file_size": 1245678
}
```

**Status Codes**:

- `200 OK`: The document was successfully uploaded and processed
- `400 Bad Request`: Invalid request format or invalid file type
- `500 Internal Server Error`: Processing error

#### List Documents

Retrieve a list of all uploaded documents.

- **URL**: `/documents`
- **Method**: `GET`

**Response**:

```json
{
  "documents": [
    {
      "doc_id": "f82a7d8e-6b3c-4d2e-9b5a-3a7f8c92e1a5",
      "filename": "sample.pdf",
      "upload_time": "2023-03-11T15:30:45",
      "page_count": 15,
      "status": "processed",
      "file_size": 1245678
    },
    {
      "doc_id": "a1b2c3d4-e5f6-7890-abcd-1234567890ab",
      "filename": "another.pdf",
      "upload_time": "2023-03-10T12:15:30",
      "page_count": 8,
      "status": "processed",
      "file_size": 567890
    }
  ]
}
```

**Status Codes**:

- `200 OK`: Successfully retrieved documents
- `500 Internal Server Error`: Server error

#### Get Document Details

Retrieve detailed information about a specific document.

- **URL**: `/document/{doc_id}`
- **Method**: `GET`

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| doc_id | UUID | Yes | The ID of the document |

**Response**:

```json
{
  "doc_id": "f82a7d8e-6b3c-4d2e-9b5a-3a7f8c92e1a5",
  "filename": "sample.pdf",
  "upload_time": "2023-03-11T15:30:45",
  "page_count": 15,
  "status": "processed",
  "file_size": 1245678,
  "metadata": {
    "language": "en",
    "title": "Sample Document",
    "author": "John Doe",
    "creation_date": "2023-01-15T10:00:00"
  }
}
```

**Status Codes**:

- `200 OK`: Successfully retrieved document details
- `404 Not Found`: Document not found
- `500 Internal Server Error`: Server error

#### Delete Document

Delete a document from the system.

- **URL**: `/document/{doc_id}`
- **Method**: `DELETE`

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| doc_id | UUID | Yes | The ID of the document to delete |

**Response**:

```json
{
  "status": "success",
  "message": "Document deleted successfully"
}
```

**Status Codes**:

- `200 OK`: Document successfully deleted
- `404 Not Found`: Document not found
- `500 Internal Server Error`: Server error

### Question Answering

#### Ask a Question

Ask a question about a document.

- **URL**: `/ask`
- **Method**: `POST`
- **Content-Type**: `application/json`

**Request Body**:

```json
{
  "question": "What is the main topic of chapter 3?",
  "doc_id": "f82a7d8e-6b3c-4d2e-9b5a-3a7f8c92e1a5",
  "threshold": 0.45,
  "max_results": 5
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| question | String | Yes | The question to ask |
| doc_id | UUID | Yes | The ID of the document to query |
| threshold | Float | No | Similarity threshold (0.0-1.0, default 0.45) |
| max_results | Integer | No | Maximum number of results to return (default 5) |

**Response**:

```json
{
  "question": "What is the main topic of chapter 3?",
  "answers": [
    {
      "text": "Chapter 3 focuses on the economic impacts of climate change, particularly in developing countries.",
      "source": "page 24, paragraph 2",
      "similarity": 0.87,
      "context": "Chapter 3: Economic Impacts\n\nChapter 3 focuses on the economic impacts of climate change, particularly in developing countries. The analysis shows that agricultural sectors are most vulnerable, with potential GDP losses ranging from 2% to 8% by 2050."
    },
    {
      "text": "The third chapter examines economic consequences including changes in agriculture, tourism, and infrastructure costs.",
      "source": "page 5, paragraph 3",
      "similarity": 0.72,
      "context": "The report is structured as follows: Chapter 1 introduces the methodology, Chapter 2 explains the science behind climate change, and the third chapter examines economic consequences including changes in agriculture, tourism, and infrastructure costs."
    }
  ],
  "language": "en",
  "processing_time": 0.253
}
```

**Status Codes**:

- `200 OK`: Successfully processed the question
- `400 Bad Request`: Invalid request format or parameters
- `404 Not Found`: Document not found
- `500 Internal Server Error`: Server error

## Data Models

### DocumentResponse

```json
{
  "doc_id": "UUID",
  "filename": "String",
  "upload_time": "DateTime",
  "page_count": "Integer",
  "status": "String",
  "file_size": "Integer"
}
```

### QuestionRequest

```json
{
  "question": "String",
  "doc_id": "UUID",
  "threshold": "Float",
  "max_results": "Integer"
}
```

### AnswerResponse

```json
{
  "question": "String",
  "answers": [
    {
      "text": "String",
      "source": "String",
      "similarity": "Float",
      "context": "String"
    }
  ],
  "language": "String",
  "processing_time": "Float"
}
```

## Error Handling

The API returns standard HTTP status codes along with a JSON object containing error details:

```json
{
  "error": "Error message",
  "details": "Additional error details"
}
```

## Rate Limiting

The current implementation does not include rate limiting. For production use, consider implementing appropriate rate limiting to prevent abuse.

## API Versioning

The current API does not implement versioning. Future versions should follow a proper versioning scheme, such as `/api/v1/endpoint`.

## Examples

### cURL Examples

1. **Upload a document**:

```bash
curl -X POST -F "file=@/path/to/document.pdf" http://localhost:8000/upload
```

2. **Ask a question**:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "question": "What is the main topic of chapter 3?",
  "doc_id": "f82a7d8e-6b3c-4d2e-9b5a-3a7f8c92e1a5",
  "threshold": 0.45,
  "max_results": 5
}' http://localhost:8000/ask
```

3. **List documents**:

```bash
curl -X GET http://localhost:8000/documents
```

4. **Delete a document**:

```bash
curl -X DELETE http://localhost:8000/document/f82a7d8e-6b3c-4d2e-9b5a-3a7f8c92e1a5
``` 