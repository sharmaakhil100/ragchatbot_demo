"""
Test suite to verify that the identified issues have been fixed
"""

import os
import tempfile

import pytest
from config import Config
from rag_system import RAGSystem


class TestBugFixes:
    """Test that previously identified bugs are now fixed"""

    @pytest.fixture
    def test_system(self):
        """Create a test RAG system with sample data"""
        config = Config()
        config.CHROMA_PATH = tempfile.mkdtemp()
        config.ANTHROPIC_API_KEY = "test-key"

        system = RAGSystem(config)

        # Add sample course
        with tempfile.TemporaryDirectory() as tmpdir:
            course_content = """Course Title: Python Programming
Course Link: https://example.com/python
Course Instructor: Jane Doe

Lesson 1: Introduction
Lesson Link: https://example.com/lesson1
Python is a programming language.
"""
            file_path = os.path.join(tmpdir, "python.txt")
            with open(file_path, "w") as f:
                f.write(course_content)

            system.add_course_document(file_path)

        return system

    def test_nonexistent_course_search(self, test_system):
        """Test that searching for non-existent course returns proper error"""
        # Search with a completely unrelated course name
        result = test_system.search_tool.execute(
            query="programming",
            course_name="Completely Random Nonexistent Course XYZ123",
        )

        # Should return "No course found" message
        assert (
            "No course found matching" in result
        ), f"Expected 'No course found' message, got: {result[:200]}"

        # Should NOT return any actual content
        assert (
            "Python" not in result
        ), "Should not return Python content when course doesn't exist"

    def test_nonexistent_course_outline(self, test_system):
        """Test that getting outline for non-existent course returns proper error"""
        result = test_system.outline_tool.execute(
            course_name="Random Nonexistent Course ABC789"
        )

        # Should return "No course found" message
        assert (
            "No course found matching" in result
        ), f"Expected 'No course found' message, got: {result[:200]}"

        # Should NOT return any course data
        assert (
            "**Course Title:**" not in result
        ), "Should not return course structure for non-existent course"

    def test_partial_match_still_works(self, test_system):
        """Test that partial/fuzzy matching still works for valid courses"""
        # Test partial match
        result = test_system.search_tool.execute(
            query="language",
            course_name="Python",  # Partial match for "Python Programming"
        )

        # Should find the course and return results
        assert (
            "Python Programming" in result
        ), "Partial match 'Python' should find 'Python Programming'"
        assert (
            "No course found" not in result
        ), "Should not show error for valid partial match"

    def test_course_resolution_threshold(self, test_system):
        """Test that course resolution uses proper similarity threshold"""
        # Try various search terms
        test_cases = [
            ("Python", True),  # Should match
            ("Programming", True),  # Should match
            ("Pyth", True),  # Close enough
            ("XYZ123ABC", False),  # Too different
            ("Completely Different", False),  # No match
            ("!@#$%^&*()", False),  # Special chars, no match
        ]

        for search_term, should_match in test_cases:
            resolved = test_system.vector_store._resolve_course_name(search_term)

            if should_match:
                assert (
                    resolved == "Python Programming"
                ), f"'{search_term}' should resolve to 'Python Programming', got {resolved}"
            else:
                assert (
                    resolved is None
                ), f"'{search_term}' should not resolve to any course, got {resolved}"

    def test_error_messages_are_user_friendly(self, test_system):
        """Test that error messages are clear and user-friendly"""
        # Test search error
        search_result = test_system.search_tool.execute(
            query="test", course_name="Nonexistent"
        )

        # Check message is clear
        if "No course found" in search_result:
            assert (
                "Nonexistent" in search_result
            ), "Error should mention the course name that wasn't found"

        # Test outline error
        outline_result = test_system.outline_tool.execute(course_name="Nonexistent")

        if "No course found" in outline_result:
            assert (
                "Nonexistent" in outline_result
            ), "Error should mention the course name that wasn't found"
