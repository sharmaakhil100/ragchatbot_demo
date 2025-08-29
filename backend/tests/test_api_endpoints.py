"""
API endpoint tests for the Course Materials RAG System

Tests all FastAPI endpoints for proper request/response handling, error cases,
and integration with the RAG system components.
"""
import pytest
import json
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException


@pytest.mark.api
class TestQueryEndpoint:
    """Test the /api/query endpoint"""
    
    def test_query_with_session_id(self, test_client, mock_rag_responses, api_test_data):
        """Test querying with a provided session ID"""
        # Setup mock response
        test_client.app.state.mock_rag.query.return_value = mock_rag_responses["query_response"]
        
        response = test_client.post("/api/query", json=api_test_data["valid_query"])
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"
        assert isinstance(data["sources"], list)
        
        # Verify RAG system was called correctly
        test_client.app.state.mock_rag.query.assert_called_once_with(
            "What is Python programming?", "test-session-123"
        )
    
    def test_query_without_session_id(self, test_client, mock_rag_responses, api_test_data):
        """Test querying without a session ID creates a new session"""
        # Setup mock response
        test_client.app.state.mock_rag.query.return_value = mock_rag_responses["query_response"]
        
        response = test_client.post("/api/query", json=api_test_data["query_without_session"])
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-id"  # Default from test app
    
    def test_query_with_dict_sources(self, test_client, api_test_data):
        """Test query response with dictionary sources containing links"""
        mock_sources = [
            {"text": "Python is versatile", "link": "https://example.com/lesson1"},
            {"text": "Python has simple syntax", "link": "https://example.com/lesson2"}
        ]
        test_client.app.state.mock_rag.query.return_value = ("Test answer", mock_sources)
        
        response = test_client.post("/api/query", json=api_test_data["valid_query"])
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["sources"]) == 2
        for source in data["sources"]:
            assert "text" in source
            assert "link" in source
            assert source["link"] is not None
    
    def test_query_with_string_sources(self, test_client, api_test_data):
        """Test query response with string sources (backward compatibility)"""
        mock_sources = ["Python is versatile", "Python has simple syntax"]
        test_client.app.state.mock_rag.query.return_value = ("Test answer", mock_sources)
        
        response = test_client.post("/api/query", json=api_test_data["valid_query"])
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["sources"]) == 2
        for source in data["sources"]:
            assert "text" in source
            assert "link" in source
            assert source["link"] is None  # String sources have no link
    
    def test_query_empty_query(self, test_client, api_test_data):
        """Test query with empty query string"""
        response = test_client.post("/api/query", json=api_test_data["invalid_query"])
        
        assert response.status_code == 200  # Should still process empty queries
    
    def test_query_missing_required_field(self, test_client):
        """Test query with missing required 'query' field"""
        response = test_client.post("/api/query", json={"session_id": "test"})
        
        assert response.status_code == 422  # Validation error
    
    def test_query_invalid_json(self, test_client):
        """Test query with invalid JSON payload"""
        response = test_client.post(
            "/api/query", 
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_query_rag_system_error(self, test_client, api_test_data):
        """Test query when RAG system raises an exception"""
        test_client.app.state.mock_rag.query.side_effect = Exception("RAG system error")
        
        response = test_client.post("/api/query", json=api_test_data["valid_query"])
        
        assert response.status_code == 500
        assert "RAG system error" in response.json()["detail"]


@pytest.mark.api
class TestCoursesEndpoint:
    """Test the /api/courses endpoint"""
    
    def test_get_course_stats_success(self, test_client, mock_rag_responses):
        """Test successful retrieval of course statistics"""
        test_client.app.state.mock_rag.get_course_analytics.return_value = mock_rag_responses["analytics_response"]
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Introduction to Python Programming" in data["course_titles"]
        
        # Verify RAG system was called
        test_client.app.state.mock_rag.get_course_analytics.assert_called_once()
    
    def test_get_course_stats_empty_response(self, test_client):
        """Test course stats with empty analytics response"""
        test_client.app.state.mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    def test_get_course_stats_rag_system_error(self, test_client):
        """Test course stats when RAG system raises an exception"""
        test_client.app.state.mock_rag.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 500
        assert "Analytics error" in response.json()["detail"]
    
    def test_get_course_stats_method_not_allowed(self, test_client):
        """Test course stats endpoint with invalid HTTP method"""
        response = test_client.post("/api/courses")
        
        assert response.status_code == 405  # Method Not Allowed


@pytest.mark.api
class TestSessionClearEndpoint:
    """Test the /api/session/clear endpoint"""
    
    def test_clear_session_success(self, test_client, api_test_data):
        """Test successful session clearing"""
        response = test_client.post("/api/session/clear", json=api_test_data["clear_session_request"])
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "message" in data
        
        # Verify session manager was called
        test_client.app.state.mock_rag.session_manager.clear_session.assert_called_once_with("test-session-123")
    
    def test_clear_session_missing_session_id(self, test_client):
        """Test clearing session without session_id"""
        response = test_client.post("/api/session/clear", json={})
        
        assert response.status_code == 422  # Validation error
    
    def test_clear_session_invalid_session_id_type(self, test_client):
        """Test clearing session with invalid session_id type"""
        response = test_client.post("/api/session/clear", json={"session_id": 123})
        
        assert response.status_code == 422  # Validation error
    
    def test_clear_session_rag_system_error(self, test_client, api_test_data):
        """Test session clearing when RAG system raises an exception"""
        test_client.app.state.mock_rag.session_manager.clear_session.side_effect = Exception("Session error")
        
        response = test_client.post("/api/session/clear", json=api_test_data["clear_session_request"])
        
        assert response.status_code == 500
        assert "Session error" in response.json()["detail"]


@pytest.mark.api
class TestCORSAndMiddleware:
    """Test CORS and middleware configuration"""
    
    def test_cors_headers_on_options_request(self, test_client):
        """Test CORS headers are present on OPTIONS request"""
        response = test_client.options("/api/query")
        
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers
    
    def test_cors_headers_on_post_request(self, test_client, api_test_data):
        """Test CORS headers are present on POST request"""
        test_client.app.state.mock_rag.query.return_value = ("Test", [])
        
        response = test_client.post("/api/query", json=api_test_data["valid_query"])
        
        assert "access-control-allow-origin" in response.headers


@pytest.mark.api
class TestRequestResponseModels:
    """Test Pydantic request/response model validation"""
    
    def test_query_request_validation(self, test_client):
        """Test QueryRequest model validation"""
        # Valid request
        valid_request = {"query": "test query", "session_id": "session-123"}
        test_client.app.state.mock_rag.query.return_value = ("Test", [])
        
        response = test_client.post("/api/query", json=valid_request)
        assert response.status_code == 200
        
        # Request without optional session_id
        minimal_request = {"query": "test query"}
        response = test_client.post("/api/query", json=minimal_request)
        assert response.status_code == 200
        
        # Invalid request - missing query
        invalid_request = {"session_id": "session-123"}
        response = test_client.post("/api/query", json=invalid_request)
        assert response.status_code == 422
    
    def test_query_response_structure(self, test_client, api_test_data):
        """Test QueryResponse model structure"""
        mock_sources = [
            {"text": "Source 1", "link": "https://example.com/1"},
            {"text": "Source 2", "link": None}
        ]
        test_client.app.state.mock_rag.query.return_value = ("Test answer", mock_sources)
        
        response = test_client.post("/api/query", json=api_test_data["valid_query"])
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        
        # Check source structure
        for source in data["sources"]:
            assert "text" in source
            assert "link" in source
    
    def test_course_stats_response_structure(self, test_client):
        """Test CourseStats model structure"""
        mock_analytics = {
            "total_courses": 5,
            "course_titles": ["Course 1", "Course 2", "Course 3", "Course 4", "Course 5"]
        }
        test_client.app.state.mock_rag.get_course_analytics.return_value = mock_analytics
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)


@pytest.mark.api
@pytest.mark.integration
class TestEndToEndAPIFlow:
    """Test complete API workflows"""
    
    def test_complete_query_session_flow(self, test_client, mock_rag_responses):
        """Test complete flow: query -> get courses -> clear session"""
        # Step 1: Make a query
        test_client.app.state.mock_rag.query.return_value = mock_rag_responses["query_response"]
        
        query_response = test_client.post("/api/query", json={
            "query": "What is Python?",
            "session_id": "flow-test-session"
        })
        
        assert query_response.status_code == 200
        query_data = query_response.json()
        session_id = query_data["session_id"]
        
        # Step 2: Get course statistics
        test_client.app.state.mock_rag.get_course_analytics.return_value = mock_rag_responses["analytics_response"]
        
        courses_response = test_client.get("/api/courses")
        
        assert courses_response.status_code == 200
        courses_data = courses_response.json()
        assert courses_data["total_courses"] > 0
        
        # Step 3: Clear the session
        clear_response = test_client.post("/api/session/clear", json={
            "session_id": session_id
        })
        
        assert clear_response.status_code == 200
        clear_data = clear_response.json()
        assert clear_data["status"] == "success"
        
        # Verify all RAG system methods were called
        test_client.app.state.mock_rag.query.assert_called()
        test_client.app.state.mock_rag.get_course_analytics.assert_called()
        test_client.app.state.mock_rag.session_manager.clear_session.assert_called()


@pytest.mark.api
class TestErrorHandling:
    """Test comprehensive error handling across endpoints"""
    
    def test_404_on_invalid_endpoint(self, test_client):
        """Test 404 response for non-existent endpoints"""
        response = test_client.get("/api/nonexistent")
        assert response.status_code == 404
    
    def test_405_on_invalid_method(self, test_client):
        """Test 405 response for invalid HTTP methods"""
        response = test_client.delete("/api/query")
        assert response.status_code == 405
    
    def test_422_on_invalid_content_type(self, test_client):
        """Test 422 response for invalid content type"""
        response = test_client.post(
            "/api/query",
            data="not json",
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 422