from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "What to search for in the course content"
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                    }
                },
                "required": ["query"]
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the search tool with given parameters.
        
        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter
            
        Returns:
            Formatted search results or error message
        """
        
        # Use the vector store's unified search interface
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )
        
        # Handle errors
        if results.error:
            return results.error
        
        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."
        
        # Format and return results
        return self._format_results(results)
    
    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI with links
        
        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')
            
            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"
            
            # Track source for the UI
            source_text = course_title
            if lesson_num is not None:
                source_text += f" - Lesson {lesson_num}"
            
            # Get the lesson link if available
            lesson_link = None
            if lesson_num is not None:
                lesson_link = self.store.get_lesson_link(course_title, lesson_num)
            elif course_title != 'unknown':
                # If no specific lesson, try to get course link
                lesson_link = self.store.get_course_link(course_title)
            
            # Create source object with text and optional link
            source_obj = {
                "text": source_text,
                "link": lesson_link
            }
            sources.append(source_obj)
            
            formatted.append(f"{header}\n{doc}")
        
        # Store sources for retrieval
        self.last_sources = sources
        
        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for retrieving course outlines with lesson details"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": "Get the complete outline of a course including course title, link, and all lessons",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    }
                },
                "required": ["course_name"]
            }
        }
    
    def execute(self, course_name: str) -> str:
        """
        Execute the course outline tool.
        
        Args:
            course_name: Course name/title to get outline for
            
        Returns:
            Formatted course outline or error message
        """
        import json
        
        # Step 1: Resolve course name using semantic search
        resolved_title = self._resolve_course_name(course_name)
        if not resolved_title:
            return f"No course found matching '{course_name}'"
        
        # Step 2: Get course metadata from catalog
        try:
            results = self.store.course_catalog.get(ids=[resolved_title])
            if not results or not results.get('metadatas'):
                return f"Could not retrieve outline for course '{course_name}'"
            
            metadata = results['metadatas'][0]
            
            # Extract course information
            course_title = metadata.get('title', 'Unknown Course')
            course_link = metadata.get('course_link', '')
            instructor = metadata.get('instructor', '')
            lessons_json = metadata.get('lessons_json', '[]')
            
            # Parse lessons
            lessons = json.loads(lessons_json)
            
            # Format the outline
            outline = []
            outline.append(f"**Course Title:** {course_title}")
            if course_link:
                outline.append(f"**Course Link:** {course_link}")
            if instructor:
                outline.append(f"**Instructor:** {instructor}")
            outline.append(f"**Total Lessons:** {len(lessons)}")
            outline.append("\n**Lessons:**")
            
            for lesson in lessons:
                lesson_num = lesson.get('lesson_number', 0)
                lesson_title = lesson.get('lesson_title', 'Untitled')
                outline.append(f"  - Lesson {lesson_num}: {lesson_title}")
            
            # Track source for the UI
            self.last_sources = [{
                "text": f"{course_title} - Course Outline",
                "link": course_link if course_link else None
            }]
            
            return "\n".join(outline)
            
        except Exception as e:
            return f"Error retrieving course outline: {str(e)}"
    
    def _resolve_course_name(self, course_name: str) -> Optional[str]:
        """Use vector search to find best matching course by name"""
        try:
            results = self.store.course_catalog.query(
                query_texts=[course_name],
                n_results=1
            )
            
            if results['documents'][0] and results['metadatas'][0]:
                # Check distance threshold for match quality
                if results['distances'] and results['distances'][0]:
                    distance = results['distances'][0][0]
                    if distance > 1.8:  # Poor match threshold (increased for better partial matching)
                        return None
                
                # Return the title (which is now the ID)
                return results['metadatas'][0][0]['title']
        except Exception as e:
            print(f"Error resolving course name: {e}")
        
        return None

class ToolManager:
    """Manages available tools for the AI"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"
        
        return self.tools[tool_name].execute(**kwargs)
    
    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources') and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []