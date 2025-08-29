"""
Shared fixtures and mock data for testing the RAG system
"""
import pytest
import os
import tempfile
import json
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

# Add backend to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from ai_generator import AIGenerator
from config import Config
from rag_system import RAGSystem


@pytest.fixture
def mock_course_data():
    """Create mock course data for testing"""
    courses = [
        Course(
            title="Introduction to Python Programming",
            course_link="https://example.com/python-course",
            instructor="Jane Doe",
            lessons=[
                Lesson(lesson_number=1, title="Getting Started", lesson_link="https://example.com/python/lesson1"),
                Lesson(lesson_number=2, title="Variables and Data Types", lesson_link="https://example.com/python/lesson2"),
                Lesson(lesson_number=3, title="Control Flow", lesson_link="https://example.com/python/lesson3"),
            ]
        ),
        Course(
            title="Machine Learning Fundamentals",
            course_link="https://example.com/ml-course",
            instructor="John Smith",
            lessons=[
                Lesson(lesson_number=1, title="Introduction to ML", lesson_link="https://example.com/ml/lesson1"),
                Lesson(lesson_number=2, title="Supervised Learning", lesson_link="https://example.com/ml/lesson2"),
            ]
        ),
        Course(
            title="Advanced Data Science",
            course_link="https://example.com/ds-course",
            instructor="Emily Johnson",
            lessons=[
                Lesson(lesson_number=1, title="Data Preprocessing", lesson_link="https://example.com/ds/lesson1"),
            ]
        )
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
            chunk_index=0
        ),
        CourseChunk(
            content="Variables in Python are used to store data. Python supports various data types including integers, floats, strings, lists, and dictionaries.",
            course_title="Introduction to Python Programming",
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Control flow statements like if-else conditions and loops allow you to control the execution of your Python programs.",
            course_title="Introduction to Python Programming",
            lesson_number=3,
            chunk_index=2
        ),
        CourseChunk(
            content="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            course_title="Machine Learning Fundamentals",
            lesson_number=1,
            chunk_index=3
        ),
        CourseChunk(
            content="Supervised learning is a type of machine learning where the model is trained on labeled data. Common algorithms include linear regression and decision trees.",
            course_title="Machine Learning Fundamentals",
            lesson_number=2,
            chunk_index=4
        ),
        CourseChunk(
            content="Data preprocessing is a crucial step in any data science project. It involves cleaning, transforming, and preparing raw data for analysis.",
            course_title="Advanced Data Science",
            lesson_number=1,
            chunk_index=5
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
            documents=documents,
            metadata=metadata,
            distances=distances,
            error=error
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
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_course_document)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)