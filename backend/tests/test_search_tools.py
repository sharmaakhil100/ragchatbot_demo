"""
Test suite for search_tools.py - CourseSearchTool and CourseOutlineTool
"""
import pytest
import json
from unittest.mock import Mock, MagicMock, patch
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test CourseSearchTool functionality"""
    
    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is properly structured"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["required"]
        assert "properties" in definition["input_schema"]
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
    
    def test_execute_basic_query(self, mock_vector_store):
        """Test basic query execution without filters"""
        tool = CourseSearchTool(mock_vector_store)
        
        # Mock the search method to return results
        mock_results = SearchResults(
            documents=["Python is a programming language", "Machine learning basics"],
            metadata=[
                {"course_title": "Introduction to Python Programming", "lesson_number": 1},
                {"course_title": "Machine Learning Fundamentals", "lesson_number": 1}
            ],
            distances=[0.1, 0.2]
        )
        mock_vector_store.search = Mock(return_value=mock_results)
        
        result = tool.execute(query="programming languages")
        
        # Verify search was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="programming languages",
            course_name=None,
            lesson_number=None
        )
        
        # Check result formatting
        assert "[Introduction to Python Programming - Lesson 1]" in result
        assert "Python is a programming language" in result
        assert "[Machine Learning Fundamentals - Lesson 1]" in result
        assert "Machine learning basics" in result
    
    def test_execute_with_course_filter(self, mock_vector_store):
        """Test query execution with course name filter"""
        tool = CourseSearchTool(mock_vector_store)
        
        # Mock search to return filtered results
        mock_results = SearchResults(
            documents=["Variables in Python are used to store data"],
            metadata=[{"course_title": "Introduction to Python Programming", "lesson_number": 2}],
            distances=[0.1]
        )
        mock_vector_store.search = Mock(return_value=mock_results)
        
        result = tool.execute(
            query="variables",
            course_name="Python"
        )
        
        # Verify search was called with course filter
        mock_vector_store.search.assert_called_once_with(
            query="variables",
            course_name="Python",
            lesson_number=None
        )
        
        assert "[Introduction to Python Programming - Lesson 2]" in result
        assert "Variables in Python" in result
    
    def test_execute_with_lesson_filter(self, mock_vector_store):
        """Test query execution with lesson number filter"""
        tool = CourseSearchTool(mock_vector_store)
        
        mock_results = SearchResults(
            documents=["Content from lesson 2"],
            metadata=[{"course_title": "Test Course", "lesson_number": 2}],
            distances=[0.15]
        )
        mock_vector_store.search = Mock(return_value=mock_results)
        
        result = tool.execute(
            query="test query",
            course_name="Test Course",
            lesson_number=2
        )
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Test Course",
            lesson_number=2
        )
        
        assert "[Test Course - Lesson 2]" in result
        assert "Content from lesson 2" in result
    
    def test_execute_with_error(self, mock_vector_store):
        """Test error handling in execute method"""
        tool = CourseSearchTool(mock_vector_store)
        
        # Mock search to return an error
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="No course found matching 'NonExistent'"
        )
        mock_vector_store.search = Mock(return_value=mock_results)
        
        result = tool.execute(query="test", course_name="NonExistent")
        
        assert result == "No course found matching 'NonExistent'"
    
    def test_execute_empty_results(self, mock_vector_store):
        """Test handling of empty search results"""
        tool = CourseSearchTool(mock_vector_store)
        
        # Mock empty results
        mock_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search = Mock(return_value=mock_results)
        
        # Test without filters
        result = tool.execute(query="xyz123notfound")
        assert "No relevant content found" in result
        
        # Test with course filter
        result = tool.execute(query="xyz123notfound", course_name="Python")
        assert "No relevant content found in course 'Python'" in result
        
        # Test with lesson filter
        result = tool.execute(query="xyz123notfound", lesson_number=5)
        assert "No relevant content found in lesson 5" in result
        
        # Test with both filters
        result = tool.execute(query="xyz123notfound", course_name="Python", lesson_number=5)
        assert "No relevant content found in course 'Python' in lesson 5" in result
    
    def test_source_tracking(self, mock_vector_store):
        """Test that sources are properly tracked with links"""
        tool = CourseSearchTool(mock_vector_store)
        
        # Mock search results
        mock_results = SearchResults(
            documents=["Test content 1", "Test content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": None}
            ],
            distances=[0.1, 0.2]
        )
        mock_vector_store.search = Mock(return_value=mock_results)
        
        # Mock get_lesson_link and get_course_link
        mock_vector_store.get_lesson_link = Mock(return_value="https://example.com/lesson1")
        mock_vector_store.get_course_link = Mock(return_value="https://example.com/course")
        
        result = tool.execute(query="test")
        
        # Check sources were tracked
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "Course A - Lesson 1"
        assert tool.last_sources[0]["link"] == "https://example.com/lesson1"
        assert tool.last_sources[1]["text"] == "Course B"
        assert tool.last_sources[1]["link"] == "https://example.com/course"
    
    def test_format_results_with_unknown_course(self, mock_vector_store):
        """Test formatting when course title is unknown"""
        tool = CourseSearchTool(mock_vector_store)
        
        mock_results = SearchResults(
            documents=["Some content"],
            metadata=[{"course_title": "unknown", "lesson_number": None}],
            distances=[0.3]
        )
        
        formatted = tool._format_results(mock_results)
        assert "[unknown]" in formatted
        assert "Some content" in formatted


class TestCourseOutlineTool:
    """Test CourseOutlineTool functionality"""
    
    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is properly structured"""
        tool = CourseOutlineTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "course_name" in definition["input_schema"]["required"]
        assert "properties" in definition["input_schema"]
        assert "course_name" in definition["input_schema"]["properties"]
    
    def test_execute_valid_course(self, mock_vector_store):
        """Test getting outline for a valid course"""
        tool = CourseOutlineTool(mock_vector_store)
        
        # Mock course resolution
        tool._resolve_course_name = Mock(return_value="Introduction to Python Programming")
        
        # Mock course catalog data
        mock_catalog_results = {
            'metadatas': [{
                'title': 'Introduction to Python Programming',
                'course_link': 'https://example.com/python',
                'instructor': 'Jane Doe',
                'lessons_json': json.dumps([
                    {"lesson_number": 1, "lesson_title": "Getting Started", "lesson_link": "https://example.com/lesson1"},
                    {"lesson_number": 2, "lesson_title": "Variables", "lesson_link": "https://example.com/lesson2"}
                ])
            }]
        }
        mock_vector_store.course_catalog.get = Mock(return_value=mock_catalog_results)
        
        result = tool.execute(course_name="Python")
        
        # Verify the outline contains expected information
        assert "**Course Title:** Introduction to Python Programming" in result
        assert "**Course Link:** https://example.com/python" in result
        assert "**Instructor:** Jane Doe" in result
        assert "**Total Lessons:** 2" in result
        assert "Lesson 1: Getting Started" in result
        assert "Lesson 2: Variables" in result
        
        # Check sources
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Introduction to Python Programming - Course Outline"
        assert tool.last_sources[0]["link"] == "https://example.com/python"
    
    def test_execute_nonexistent_course(self, mock_vector_store):
        """Test handling of non-existent course"""
        tool = CourseOutlineTool(mock_vector_store)
        
        # Mock failed course resolution
        tool._resolve_course_name = Mock(return_value=None)
        
        result = tool.execute(course_name="NonExistent")
        
        assert result == "No course found matching 'NonExistent'"
    
    def test_execute_with_missing_metadata(self, mock_vector_store):
        """Test handling of missing metadata fields"""
        tool = CourseOutlineTool(mock_vector_store)
        
        # Mock course resolution
        tool._resolve_course_name = Mock(return_value="Test Course")
        
        # Mock catalog with minimal data
        mock_catalog_results = {
            'metadatas': [{
                'title': 'Test Course',
                'course_link': '',
                'instructor': '',
                'lessons_json': '[]'
            }]
        }
        mock_vector_store.course_catalog.get = Mock(return_value=mock_catalog_results)
        
        result = tool.execute(course_name="Test")
        
        assert "**Course Title:** Test Course" in result
        assert "**Total Lessons:** 0" in result
        # Should not include empty course link or instructor
        assert "**Course Link:**  " not in result or "**Course Link:** " not in result
    
    def test_resolve_course_name(self, mock_vector_store):
        """Test course name resolution using semantic search"""
        tool = CourseOutlineTool(mock_vector_store)
        
        # Mock query results with distances
        mock_query_results = {
            'documents': [['Introduction to Python Programming']],
            'metadatas': [[{'title': 'Introduction to Python Programming'}]],
            'distances': [[0.5]]  # Good match distance
        }
        mock_vector_store.course_catalog.query = Mock(return_value=mock_query_results)
        
        resolved = tool._resolve_course_name("Python")
        
        assert resolved == "Introduction to Python Programming"
        mock_vector_store.course_catalog.query.assert_called_once_with(
            query_texts=["Python"],
            n_results=1
        )
    
    def test_resolve_course_name_failure(self, mock_vector_store):
        """Test course name resolution failure"""
        tool = CourseOutlineTool(mock_vector_store)
        
        # Mock empty query results
        mock_query_results = {
            'documents': [[]],
            'metadatas': [[]]
        }
        mock_vector_store.course_catalog.query = Mock(return_value=mock_query_results)
        
        resolved = tool._resolve_course_name("NonExistent")
        
        assert resolved is None
    
    def test_execute_with_exception(self, mock_vector_store):
        """Test exception handling in execute method"""
        tool = CourseOutlineTool(mock_vector_store)
        
        # Mock successful resolution but catalog.get raises exception
        tool._resolve_course_name = Mock(return_value="Test Course")
        mock_vector_store.course_catalog.get = Mock(side_effect=Exception("Database error"))
        
        result = tool.execute(course_name="Test")
        
        assert "Error retrieving course outline" in result
        assert "Database error" in result


class TestToolManager:
    """Test ToolManager functionality"""
    
    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        
        manager.register_tool(tool)
        
        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == tool
    
    def test_register_tool_without_name(self, mock_vector_store):
        """Test registration fails for tool without name"""
        manager = ToolManager()
        
        # Create a mock tool with invalid definition
        mock_tool = Mock()
        mock_tool.get_tool_definition = Mock(return_value={"description": "No name"})
        
        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            manager.register_tool(mock_tool)
    
    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)
        
        definitions = manager.get_tool_definitions()
        
        assert len(definitions) == 2
        names = [d["name"] for d in definitions]
        assert "search_course_content" in names
        assert "get_course_outline" in names
    
    def test_execute_tool(self, mock_vector_store):
        """Test tool execution through manager"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        tool.execute = Mock(return_value="Search results")
        
        manager.register_tool(tool)
        
        result = manager.execute_tool("search_course_content", query="test")
        
        assert result == "Search results"
        tool.execute.assert_called_once_with(query="test")
    
    def test_execute_nonexistent_tool(self):
        """Test executing a non-existent tool"""
        manager = ToolManager()
        
        result = manager.execute_tool("nonexistent_tool", param="value")
        
        assert result == "Tool 'nonexistent_tool' not found"
    
    def test_get_last_sources(self, mock_vector_store):
        """Test retrieving sources from tools"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        search_tool.last_sources = [
            {"text": "Source 1", "link": "https://example.com/1"},
            {"text": "Source 2", "link": "https://example.com/2"}
        ]
        
        manager.register_tool(search_tool)
        
        sources = manager.get_last_sources()
        
        assert len(sources) == 2
        assert sources[0]["text"] == "Source 1"
        assert sources[1]["text"] == "Source 2"
    
    def test_get_last_sources_empty(self):
        """Test getting sources when no tools have sources"""
        manager = ToolManager()
        
        sources = manager.get_last_sources()
        
        assert sources == []
    
    def test_reset_sources(self, mock_vector_store):
        """Test resetting sources in all tools"""
        manager = ToolManager()
        
        search_tool = CourseSearchTool(mock_vector_store)
        search_tool.last_sources = [{"text": "Source", "link": "link"}]
        
        outline_tool = CourseOutlineTool(mock_vector_store)
        outline_tool.last_sources = [{"text": "Outline", "link": "link2"}]
        
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)
        
        manager.reset_sources()
        
        assert search_tool.last_sources == []
        assert outline_tool.last_sources == []
        assert manager.get_last_sources() == []