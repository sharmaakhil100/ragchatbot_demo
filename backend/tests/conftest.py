"""
Shared fixtures and mock data for testing the RAG system
"""

import json
import os
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import List, Dict, Any
from fastapi.testclient import TestClient
import asyncio

# Add backend to path for imports
import sys
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from config import Config
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults, VectorStore


@pytest.fixture
def mock_course_data():
    """Create mock course data for testing"""
    courses = [
        Course(
            title="Introduction to Python Programming",
            course_link="https://example.com/python-course",
            instructor="Jane Doe",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Getting Started",
                    lesson_link="https://example.com/python/lesson1",
                ),
                Lesson(
                    lesson_number=2,
                    title="Variables and Data Types",
                    lesson_link="https://example.com/python/lesson2",
                ),
                Lesson(
                    lesson_number=3,
                    title="Control Flow",
                    lesson_link="https://example.com/python/lesson3",
                ),
            ],
        ),
        Course(
            title="Machine Learning Fundamentals",
            course_link="https://example.com/ml-course",
            instructor="John Smith",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Introduction to ML",
                    lesson_link="https://example.com/ml/lesson1",
                ),
                Lesson(
                    lesson_number=2,
                    title="Supervised Learning",
                    lesson_link="https://example.com/ml/lesson2",
                ),
            ],
        ),
        Course(
            title="Advanced Data Science",
            course_link="https://example.com/ds-course",
            instructor="Emily Johnson",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Data Preprocessing",
                    lesson_link="https://example.com/ds/lesson1",
                ),
            ],
        ),
    ]
    return courses


@pytest.fixture
def mock_course_chunks():
    """Create mock course chunks for testing"""
    chunks = [
        CourseChunk(
            content="Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, and automation.",
            course_title="Introduction to Python Programming",
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Variables in Python are used to store data. Python supports various data types including integers, floats, strings, lists, and dictionaries.",
            course_title="Introduction to Python Programming",
            lesson_number=2,
            chunk_index=1,
        ),
        CourseChunk(
            content="Control flow statements like if-else conditions and loops allow you to control the execution of your Python programs.",
            course_title="Introduction to Python Programming",
            lesson_number=3,
            chunk_index=2,
        ),
        CourseChunk(
            content="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            course_title="Machine Learning Fundamentals",
            lesson_number=1,
            chunk_index=3,
        ),
        CourseChunk(
            content="Supervised learning is a type of machine learning where the model is trained on labeled data. Common algorithms include linear regression and decision trees.",
            course_title="Machine Learning Fundamentals",
            lesson_number=2,
            chunk_index=4,
        ),
        CourseChunk(
            content="Data preprocessing is a crucial step in any data science project. It involves cleaning, transforming, and preparing raw data for analysis.",
            course_title="Advanced Data Science",
            lesson_number=1,
            chunk_index=5,
        ),
    ]
    return chunks


@pytest.fixture
def mock_vector_store(mock_course_data, mock_course_chunks):
    """Create a mock vector store with test data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(tmpdir, "all-MiniLM-L6-v2", max_results=5)

        # Add mock courses to catalog
        for course in mock_course_data:
            store.add_course_metadata(course)

        # Add mock chunks to content
        store.add_course_content(mock_course_chunks)

        yield store


@pytest.fixture
def mock_search_results():
    """Create mock search results for testing"""

    def _create_results(documents=None, metadata=None, distances=None, error=None):
        if documents is None:
            documents = []
        if metadata is None:
            metadata = []
        if distances is None:
            distances = []
        return SearchResults(
            documents=documents, metadata=metadata, distances=distances, error=error
        )

    return _create_results


@pytest.fixture
def mock_config():
    """Create a mock configuration object"""
    config = Config()
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-test-model"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client"""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="This is a test response from the AI.")]
    mock_response.stop_reason = "stop"
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_tool_manager():
    """Create a mock tool manager"""
    manager = ToolManager()
    return manager


@pytest.fixture
def sample_course_document():
    """Create a sample course document for testing document processing"""
    content = """Course Title: Test Course for Processing
Course Link: https://example.com/test-course
Course Instructor: Test Instructor

Lesson 1: Introduction to Testing
Lesson Link: https://example.com/test/lesson1
This is the content of lesson 1. It contains information about testing fundamentals.
Testing is important for ensuring code quality and reliability.

Lesson 2: Advanced Testing Techniques
Lesson Link: https://example.com/test/lesson2
This lesson covers advanced testing techniques including unit tests, integration tests,
and end-to-end tests. We'll explore various testing frameworks and best practices.

Lesson 3: Test Automation
Lesson Link: https://example.com/test/lesson3
Automation is key to efficient testing. This lesson explores continuous integration,
continuous deployment, and automated testing pipelines.
"""
    return content


@pytest.fixture
def temp_course_file(sample_course_document):
    """Create a temporary course file for testing"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(sample_course_document)
        temp_path = f.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def test_app():
    """Create a test FastAPI app without static file mounting to avoid path issues"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Import the Pydantic models from app module
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app import QueryRequest, QueryResponse, SourceLink, CourseStats, ClearSessionRequest
    
    # Create test app
    app = FastAPI(title="Course Materials RAG System Test", root_path="")
    
    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Mock RAG system for testing
    mock_rag = MagicMock()
    
    # API Endpoints (same as in app.py but using mock RAG system)
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or "test-session-id"
            answer, sources = mock_rag.query(request.query, session_id)
            
            source_links = []
            for source in sources:
                if isinstance(source, dict):
                    source_links.append(SourceLink(
                        text=source.get('text', ''),
                        link=source.get('link')
                    ))
                else:
                    source_links.append(SourceLink(text=str(source), link=None))
            
            return QueryResponse(
                answer=answer,
                sources=source_links,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/session/clear")
    async def clear_session(request: ClearSessionRequest):
        try:
            mock_rag.session_manager.clear_session(request.session_id)
            return {"status": "success", "message": "Session cleared"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Store the mock RAG system on the app for test access
    app.state.mock_rag = mock_rag
    
    return app


@pytest.fixture
def test_client(test_app):
    """Create a test client for the FastAPI app"""
    return TestClient(test_app)


@pytest.fixture
def mock_rag_responses():
    """Mock responses for RAG system queries"""
    return {
        "query_response": (
            "Python is a versatile programming language known for its simplicity and readability.",
            [
                {"text": "Python is a high-level programming language", "link": "https://example.com/python/lesson1"},
                {"text": "Python supports various data types", "link": "https://example.com/python/lesson2"}
            ]
        ),
        "analytics_response": {
            "total_courses": 3,
            "course_titles": [
                "Introduction to Python Programming",
                "Machine Learning Fundamentals", 
                "Advanced Data Science"
            ]
        }
    }


@pytest.fixture
def mock_session_manager():
    """Create a mock session manager"""
    manager = MagicMock()
    manager.create_session.return_value = "test-session-123"
    manager.clear_session.return_value = None
    return manager


@pytest.fixture
def api_test_data():
    """Test data for API endpoint testing"""
    return {
        "valid_query": {
            "query": "What is Python programming?",
            "session_id": "test-session-123"
        },
        "query_without_session": {
            "query": "Explain machine learning basics"
        },
        "invalid_query": {
            "query": ""
        },
        "clear_session_request": {
            "session_id": "test-session-123"
        }
    }


@pytest.fixture
async def async_test_client(test_app):
    """Create an async test client for testing async endpoints"""
    from httpx import AsyncClient
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client
