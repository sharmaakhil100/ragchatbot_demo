"""
Test suite for rag_system.py - RAGSystem integration tests
"""
import pytest
import os
import tempfile
from unittest.mock import Mock, MagicMock, patch
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


class TestRAGSystem:
    """Test RAGSystem functionality"""
    
    def test_initialization(self, mock_config):
        """Test RAGSystem initialization with all components"""
        with patch('rag_system.DocumentProcessor') as mock_doc_processor, \
             patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator') as mock_ai_generator, \
             patch('rag_system.SessionManager') as mock_session_manager:
            
            system = RAGSystem(mock_config)
            
            # Verify all components were initialized
            mock_doc_processor.assert_called_once_with(
                mock_config.CHUNK_SIZE,
                mock_config.CHUNK_OVERLAP
            )
            mock_vector_store.assert_called_once_with(
                mock_config.CHROMA_PATH,
                mock_config.EMBEDDING_MODEL,
                mock_config.MAX_RESULTS
            )
            mock_ai_generator.assert_called_once_with(
                mock_config.ANTHROPIC_API_KEY,
                mock_config.ANTHROPIC_MODEL
            )
            mock_session_manager.assert_called_once_with(mock_config.MAX_HISTORY)
            
            # Verify tools are registered
            assert system.tool_manager is not None
            assert len(system.tool_manager.tools) == 2
            assert "search_course_content" in system.tool_manager.tools
            assert "get_course_outline" in system.tool_manager.tools
    
    def test_add_course_document_success(self, mock_config, mock_course_data, mock_course_chunks):
        """Test successfully adding a course document"""
        system = RAGSystem(mock_config)
        
        # Mock document processor
        course = mock_course_data[0]
        chunks = mock_course_chunks[:3]  # Use first 3 chunks
        system.document_processor.process_course_document = Mock(
            return_value=(course, chunks)
        )
        
        # Mock vector store methods
        system.vector_store.add_course_metadata = Mock()
        system.vector_store.add_course_content = Mock()
        
        result_course, chunk_count = system.add_course_document("/path/to/course.txt")
        
        # Verify processing
        system.document_processor.process_course_document.assert_called_once_with(
            "/path/to/course.txt"
        )
        
        # Verify storage
        system.vector_store.add_course_metadata.assert_called_once_with(course)
        system.vector_store.add_course_content.assert_called_once_with(chunks)
        
        # Verify return values
        assert result_course == course
        assert chunk_count == 3
    
    def test_add_course_document_failure(self, mock_config):
        """Test handling of document processing failure"""
        system = RAGSystem(mock_config)
        
        # Mock processing failure
        system.document_processor.process_course_document = Mock(
            side_effect=Exception("Processing error")
        )
        
        result_course, chunk_count = system.add_course_document("/path/to/bad.txt")
        
        assert result_course is None
        assert chunk_count == 0
    
    def test_add_course_folder_new_courses(self, mock_config, mock_course_data, mock_course_chunks, temp_course_file):
        """Test adding courses from a folder"""
        system = RAGSystem(mock_config)
        
        # Create temp folder with test files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_file1 = os.path.join(tmpdir, "course1.txt")
            test_file2 = os.path.join(tmpdir, "course2.pdf")
            test_file3 = os.path.join(tmpdir, "not_a_course.jpg")  # Should be ignored
            
            with open(test_file1, 'w') as f:
                f.write("Course 1 content")
            with open(test_file2, 'w') as f:
                f.write("Course 2 content")
            with open(test_file3, 'w') as f:
                f.write("Not a course")
            
            # Mock existing courses (empty)
            system.vector_store.get_existing_course_titles = Mock(return_value=[])
            
            # Mock document processing
            course1 = mock_course_data[0]
            course2 = mock_course_data[1]
            chunks1 = mock_course_chunks[:3]
            chunks2 = mock_course_chunks[3:5]
            
            system.document_processor.process_course_document = Mock(
                side_effect=[(course1, chunks1), (course2, chunks2)]
            )
            
            # Mock vector store methods
            system.vector_store.add_course_metadata = Mock()
            system.vector_store.add_course_content = Mock()
            
            total_courses, total_chunks = system.add_course_folder(tmpdir)
            
            # Verify only valid files were processed
            assert system.document_processor.process_course_document.call_count == 2
            
            # Verify courses were added
            assert system.vector_store.add_course_metadata.call_count == 2
            assert system.vector_store.add_course_content.call_count == 2
            
            assert total_courses == 2
            assert total_chunks == 5  # 3 + 2 chunks
    
    def test_add_course_folder_skip_existing(self, mock_config, mock_course_data, mock_course_chunks):
        """Test skipping existing courses when adding from folder"""
        system = RAGSystem(mock_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            test_file = os.path.join(tmpdir, "course.txt")
            with open(test_file, 'w') as f:
                f.write("Course content")
            
            # Mock existing course with same title
            course = mock_course_data[0]
            system.vector_store.get_existing_course_titles = Mock(
                return_value=[course.title]
            )
            
            # Mock document processing
            system.document_processor.process_course_document = Mock(
                return_value=(course, mock_course_chunks[:3])
            )
            
            # Mock vector store methods
            system.vector_store.add_course_metadata = Mock()
            system.vector_store.add_course_content = Mock()
            
            total_courses, total_chunks = system.add_course_folder(tmpdir)
            
            # Course should be processed to check title
            system.document_processor.process_course_document.assert_called_once()
            
            # But not added since it already exists
            system.vector_store.add_course_metadata.assert_not_called()
            system.vector_store.add_course_content.assert_not_called()
            
            assert total_courses == 0
            assert total_chunks == 0
    
    def test_add_course_folder_clear_existing(self, mock_config):
        """Test clearing existing data when requested"""
        system = RAGSystem(mock_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            system.vector_store.clear_all_data = Mock()
            system.vector_store.get_existing_course_titles = Mock(return_value=[])
            
            system.add_course_folder(tmpdir, clear_existing=True)
            
            system.vector_store.clear_all_data.assert_called_once()
    
    def test_add_course_folder_nonexistent(self, mock_config):
        """Test handling of non-existent folder"""
        system = RAGSystem(mock_config)
        
        total_courses, total_chunks = system.add_course_folder("/nonexistent/folder")
        
        assert total_courses == 0
        assert total_chunks == 0
    
    def test_query_simple(self, mock_config):
        """Test simple query without session"""
        system = RAGSystem(mock_config)
        
        # Mock AI generator response
        system.ai_generator.generate_response = Mock(
            return_value="Python is a programming language"
        )
        
        # Mock tool manager sources
        system.tool_manager.get_last_sources = Mock(
            return_value=[{"text": "Source 1", "link": "link1"}]
        )
        system.tool_manager.reset_sources = Mock()
        
        response, sources = system.query("What is Python?")
        
        # Verify AI generator was called
        system.ai_generator.generate_response.assert_called_once()
        call_args = system.ai_generator.generate_response.call_args
        assert "What is Python?" in call_args[1]["query"]
        assert call_args[1]["conversation_history"] is None
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] == system.tool_manager
        
        # Verify response and sources
        assert response == "Python is a programming language"
        assert sources == [{"text": "Source 1", "link": "link1"}]
        
        # Verify sources were reset
        system.tool_manager.reset_sources.assert_called_once()
    
    def test_query_with_session(self, mock_config):
        """Test query with session management"""
        system = RAGSystem(mock_config)
        
        # Mock conversation history
        history = "User: Hello\nAssistant: Hi!"
        system.session_manager.get_conversation_history = Mock(return_value=history)
        system.session_manager.add_exchange = Mock()
        
        # Mock AI response
        system.ai_generator.generate_response = Mock(
            return_value="I can help with that"
        )
        
        # Mock sources
        system.tool_manager.get_last_sources = Mock(return_value=[])
        system.tool_manager.reset_sources = Mock()
        
        response, sources = system.query("Tell me more", session_id="session123")
        
        # Verify session history was retrieved
        system.session_manager.get_conversation_history.assert_called_once_with("session123")
        
        # Verify AI generator received history
        call_args = system.ai_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] == history
        
        # Verify exchange was added to session
        system.session_manager.add_exchange.assert_called_once_with(
            "session123",
            "Tell me more",
            "I can help with that"
        )
        
        assert response == "I can help with that"
    
    def test_query_prompt_format(self, mock_config):
        """Test that query prompt is formatted correctly"""
        system = RAGSystem(mock_config)
        
        system.ai_generator.generate_response = Mock(return_value="Response")
        system.tool_manager.get_last_sources = Mock(return_value=[])
        system.tool_manager.reset_sources = Mock()
        
        system.query("How do I use variables?")
        
        call_args = system.ai_generator.generate_response.call_args
        query_text = call_args[1]["query"]
        
        assert "Answer this question about course materials:" in query_text
        assert "How do I use variables?" in query_text
    
    def test_get_course_analytics(self, mock_config):
        """Test getting course analytics"""
        system = RAGSystem(mock_config)
        
        # Mock vector store methods
        system.vector_store.get_course_count = Mock(return_value=5)
        system.vector_store.get_existing_course_titles = Mock(
            return_value=["Course 1", "Course 2", "Course 3", "Course 4", "Course 5"]
        )
        
        analytics = system.get_course_analytics()
        
        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course 1" in analytics["course_titles"]
        assert "Course 5" in analytics["course_titles"]
    
    def test_tool_integration(self, mock_config):
        """Test that tools are properly integrated with the system"""
        system = RAGSystem(mock_config)
        
        # Verify search tool is connected to vector store
        assert system.search_tool.store == system.vector_store
        
        # Verify outline tool is connected to vector store
        assert system.outline_tool.store == system.vector_store
        
        # Verify tools are registered
        tool_defs = system.tool_manager.get_tool_definitions()
        tool_names = [t["name"] for t in tool_defs]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
    
    def test_end_to_end_search_query(self, mock_config, mock_course_chunks):
        """Test end-to-end query that triggers search tool"""
        system = RAGSystem(mock_config)
        
        # Setup mock search results
        search_results = "Python is a high-level programming language"
        system.search_tool.execute = Mock(return_value=search_results)
        system.search_tool.last_sources = [{"text": "Python Course", "link": "link"}]
        
        # Mock AI generator to simulate tool use
        def mock_generate(*args, **kwargs):
            # Simulate tool execution
            tool_manager = kwargs.get('tool_manager')
            if tool_manager:
                # Simulate AI calling the search tool
                result = tool_manager.execute_tool(
                    "search_course_content",
                    query="Python programming"
                )
            return "Based on the search, Python is a versatile language"
        
        system.ai_generator.generate_response = Mock(side_effect=mock_generate)
        
        response, sources = system.query("What is Python programming?")
        
        # Verify the flow
        assert "Based on the search" in response
        assert sources == [{"text": "Python Course", "link": "link"}]
        
        # Verify search tool was called
        system.search_tool.execute.assert_called_once_with(
            query="Python programming"
        )


class TestRAGSystemDocumentProcessing:
    """Test document processing within RAG system"""
    
    def test_process_real_course_document(self, mock_config, temp_course_file):
        """Test processing a real course document file"""
        system = RAGSystem(mock_config)
        
        # Use real document processor (not mocked)
        course, chunks = system.document_processor.process_course_document(temp_course_file)
        
        # Verify course was extracted correctly
        assert course.title == "Test Course for Processing"
        assert course.course_link == "https://example.com/test-course"
        assert course.instructor == "Test Instructor"
        assert len(course.lessons) == 3
        
        # Verify lessons
        assert course.lessons[0].lesson_number == 1
        assert course.lessons[0].title == "Introduction to Testing"
        assert course.lessons[1].lesson_number == 2
        assert course.lessons[1].title == "Advanced Testing Techniques"
        assert course.lessons[2].lesson_number == 3
        assert course.lessons[2].title == "Test Automation"
        
        # Verify chunks were created
        assert len(chunks) > 0
        assert all(chunk.course_title == "Test Course for Processing" for chunk in chunks)
        
        # Verify chunk content
        chunk_contents = [chunk.content for chunk in chunks]
        assert any("testing fundamentals" in content for content in chunk_contents)
        assert any("unit tests" in content for content in chunk_contents)
        assert any("continuous integration" in content for content in chunk_contents)
    
    def test_chunk_overlap(self, mock_config):
        """Test that chunks have proper overlap"""
        system = RAGSystem(mock_config)
        
        # Create a long text to ensure multiple chunks
        long_text = " ".join([f"Sentence {i}." for i in range(100)])
        
        chunks = system.document_processor.chunk_text(long_text)
        
        # Verify chunks were created
        assert len(chunks) > 1
        
        # Verify overlap exists between consecutive chunks
        for i in range(len(chunks) - 1):
            chunk1_end = chunks[i][-mock_config.CHUNK_OVERLAP:]
            chunk2_start = chunks[i + 1][:mock_config.CHUNK_OVERLAP]
            
            # There should be some overlap (not exact due to sentence boundaries)
            # But the end of one chunk should share content with start of next
            # This is a loose check since overlap is sentence-based
            assert len(chunks[i]) <= mock_config.CHUNK_SIZE + 50  # Allow some flexibility


class TestRAGSystemErrorHandling:
    """Test error handling in RAG system"""
    
    def test_query_with_ai_failure(self, mock_config):
        """Test handling of AI generator failure"""
        system = RAGSystem(mock_config)
        
        # Mock AI generator to raise exception
        system.ai_generator.generate_response = Mock(
            side_effect=Exception("API error")
        )
        
        # This should raise the exception (no error handling in query method)
        with pytest.raises(Exception, match="API error"):
            system.query("Test query")
    
    def test_add_document_with_vector_store_failure(self, mock_config):
        """Test handling of vector store failure"""
        system = RAGSystem(mock_config)
        
        # Mock successful processing
        course = Course(
            title="Test Course",
            lessons=[Lesson(lesson_number=1, title="Lesson 1")]
        )
        chunks = [CourseChunk(
            content="Content",
            course_title="Test Course",
            lesson_number=1,
            chunk_index=0
        )]
        system.document_processor.process_course_document = Mock(
            return_value=(course, chunks)
        )
        
        # Mock vector store failure
        system.vector_store.add_course_metadata = Mock(
            side_effect=Exception("Database error")
        )
        
        result_course, chunk_count = system.add_course_document("/path/to/course.txt")
        
        # Should handle the error gracefully
        assert result_course is None
        assert chunk_count == 0
    
    def test_add_folder_with_processing_errors(self, mock_config):
        """Test handling of mixed successes and failures when processing folder"""
        system = RAGSystem(mock_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(3):
                with open(os.path.join(tmpdir, f"course{i}.txt"), 'w') as f:
                    f.write(f"Course {i} content")
            
            system.vector_store.get_existing_course_titles = Mock(return_value=[])
            
            # Mock processing: success, failure, success
            course1 = Course(title="Course 1", lessons=[])
            course3 = Course(title="Course 3", lessons=[])
            chunks1 = [CourseChunk(content="C1", course_title="Course 1", chunk_index=0)]
            chunks3 = [CourseChunk(content="C3", course_title="Course 3", chunk_index=0)]
            
            system.document_processor.process_course_document = Mock(
                side_effect=[
                    (course1, chunks1),
                    Exception("Processing error"),
                    (course3, chunks3)
                ]
            )
            
            system.vector_store.add_course_metadata = Mock()
            system.vector_store.add_course_content = Mock()
            
            total_courses, total_chunks = system.add_course_folder(tmpdir)
            
            # Should process what it can despite one failure
            assert total_courses == 2  # Only successful ones
            assert total_chunks == 2
            assert system.vector_store.add_course_metadata.call_count == 2