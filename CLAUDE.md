# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Course Materials RAG (Retrieval-Augmented Generation) System - a full-stack web application that enables semantic search and AI-powered Q&A over course documents. It uses FastAPI for the backend, ChromaDB for vector storage, and Anthropic's Claude for response generation.

## Key Architecture Components

### Backend Architecture (Python/FastAPI)
- **app.py**: FastAPI application entry point with CORS setup and API endpoints
- **rag_system.py**: Main orchestrator combining document processing, vector storage, and AI generation
- **vector_store.py**: ChromaDB wrapper managing course catalog and content collections
- **ai_generator.py**: Claude AI integration with tool-based function calling
- **document_processor.py**: Handles course document parsing and chunking
- **search_tools.py**: Tool manager for structured course searches
- **session_manager.py**: Manages conversation history for context-aware responses
- **config.py**: Central configuration management

### Data Flow
1. Documents are processed into Course objects with Lessons
2. Content is chunked and stored in ChromaDB collections (catalog + content)
3. Queries trigger semantic search with optional course/lesson filtering
4. AI uses tool-based search to find relevant content
5. Responses include sources and maintain conversation context

## Common Development Commands

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend
uv run uvicorn app:app --reload --port 8000
```

### Dependency Management
```bash
# Install dependencies
uv sync

# Add new dependency
uv add <package-name>
```

### Environment Setup
Create a `.env` file in the root directory:
```
ANTHROPIC_API_KEY=your_api_key_here
```

## Key Implementation Details

### Vector Store Structure
- **course_catalog**: Stores course metadata for semantic course name resolution
- **course_content**: Stores actual course material chunks with metadata filtering

### Search Capabilities
- Semantic search across all courses
- Course-specific filtering using fuzzy name matching
- Lesson-specific filtering within courses
- Tool-based search integration for structured queries

### Document Processing
- Supports .txt, .pdf, .docx formats
- Extracts course structure (title, instructor, lessons)
- Creates overlapping chunks for better context preservation
- Avoids duplicate course imports on startup

## Important Notes

- The system uses ChromaDB for persistence - data survives restarts
- Course documents are loaded from `/docs` folder on startup
- Frontend is served statically from the FastAPI backend
- API documentation available at `http://localhost:8000/docs`
- Session management enables multi-turn conversations with context
- dont run the server using ./run.sh I will start it myself