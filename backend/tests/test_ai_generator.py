"""
Test suite for ai_generator.py - AIGenerator class
"""
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test AIGenerator functionality"""
    
    def test_initialization(self, mock_config):
        """Test AIGenerator initialization"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            
            # Verify Anthropic client was created with correct API key
            mock_anthropic.assert_called_once_with(api_key=mock_config.ANTHROPIC_API_KEY)
            
            # Verify model and base params are set
            assert generator.model == mock_config.ANTHROPIC_MODEL
            assert generator.base_params["model"] == mock_config.ANTHROPIC_MODEL
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800
    
    def test_generate_response_simple(self, mock_anthropic_client):
        """Test simple response generation without tools"""
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This is a simple response")]
        mock_response.stop_reason = "stop"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        result = generator.generate_response("What is Python?")
        
        # Verify API call
        mock_anthropic_client.messages.create.assert_called_once()
        call_args = mock_anthropic_client.messages.create.call_args[1]
        
        assert call_args["model"] == "test-model"
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "What is Python?"
        assert "system" in call_args
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        
        # Verify response
        assert result == "This is a simple response"
    
    def test_generate_response_with_conversation_history(self, mock_anthropic_client):
        """Test response generation with conversation history"""
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response with history")]
        mock_response.stop_reason = "stop"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        conversation_history = "User: Hello\nAssistant: Hi there!"
        
        result = generator.generate_response(
            query="How are you?",
            conversation_history=conversation_history
        )
        
        # Verify system content includes history
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert "Previous conversation:" in call_args["system"]
        assert conversation_history in call_args["system"]
        
        assert result == "Response with history"
    
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic_client, mock_tool_manager):
        """Test response generation with tools available but not used"""
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        # Mock response without tool use
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Direct response without tools")]
        mock_response.stop_reason = "stop"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {"type": "object", "properties": {}}
            }
        ]
        
        result = generator.generate_response(
            query="What is 2+2?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify tools were included in API call
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tools"] == tools
        assert call_args["tool_choice"] == {"type": "auto"}
        
        assert result == "Direct response without tools"
    
    def test_generate_response_with_tool_use(self, mock_anthropic_client):
        """Test response generation with tool usage"""
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        # Create mock tool use content block
        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.id = "tool_123"
        mock_tool_use.input = {"query": "Python basics"}
        
        # Mock initial response with tool use
        mock_initial_response = MagicMock()
        mock_initial_response.content = [mock_tool_use]
        mock_initial_response.stop_reason = "tool_use"
        
        # Mock final response after tool execution
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock(text="Based on the search results, Python is...")]
        mock_final_response.stop_reason = "stop"
        
        # Set up the mock to return different responses
        mock_anthropic_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool = Mock(return_value="Search results: Python is a programming language")
        
        tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {"type": "object", "properties": {}}
            }
        ]
        
        result = generator.generate_response(
            query="What is Python?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python basics"
        )
        
        # Verify two API calls were made
        assert mock_anthropic_client.messages.create.call_count == 2
        
        # Verify second call includes tool results
        second_call_args = mock_anthropic_client.messages.create.call_args_list[1][1]
        messages = second_call_args["messages"]
        assert len(messages) >= 2
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"][0]["type"] == "tool_result"
        assert messages[-1]["content"][0]["tool_use_id"] == "tool_123"
        assert messages[-1]["content"][0]["content"] == "Search results: Python is a programming language"
        
        assert result == "Based on the search results, Python is..."
    
    def test_handle_tool_execution_multiple_tools(self, mock_anthropic_client):
        """Test handling multiple tool calls in one response"""
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        # Create multiple tool use blocks
        mock_tool_use_1 = MagicMock()
        mock_tool_use_1.type = "tool_use"
        mock_tool_use_1.name = "search_course_content"
        mock_tool_use_1.id = "tool_1"
        mock_tool_use_1.input = {"query": "Python"}
        
        mock_tool_use_2 = MagicMock()
        mock_tool_use_2.type = "tool_use"
        mock_tool_use_2.name = "get_course_outline"
        mock_tool_use_2.id = "tool_2"
        mock_tool_use_2.input = {"course_name": "Python"}
        
        # Mock text block (not a tool)
        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Let me search for that..."
        
        # Mock initial response
        mock_initial_response = MagicMock()
        mock_initial_response.content = [mock_text, mock_tool_use_1, mock_tool_use_2]
        
        # Mock final response
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock(text="Combined results...")]
        
        mock_anthropic_client.messages.create.return_value = mock_final_response
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool = Mock(side_effect=[
            "Python search results",
            "Python course outline"
        ])
        
        base_params = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Tell me about Python"}],
            "system": "System prompt",
            "temperature": 0,
            "max_tokens": 800
        }
        
        result = generator._handle_tool_execution(
            mock_initial_response,
            base_params,
            mock_tool_manager
        )
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content",
            query="Python"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline",
            course_name="Python"
        )
        
        # Verify final API call includes both tool results
        call_args = mock_anthropic_client.messages.create.call_args[1]
        messages = call_args["messages"]
        tool_results = messages[-1]["content"]
        
        assert len(tool_results) == 2
        assert tool_results[0]["tool_use_id"] == "tool_1"
        assert tool_results[0]["content"] == "Python search results"
        assert tool_results[1]["tool_use_id"] == "tool_2"
        assert tool_results[1]["content"] == "Python course outline"
        
        assert result == "Combined results..."
    
    def test_system_prompt_content(self):
        """Test that system prompt contains expected content"""
        generator = AIGenerator("test-key", "test-model")
        
        # Check key elements of system prompt
        assert "AI assistant specialized in course materials" in generator.SYSTEM_PROMPT
        assert "search_course_content" in generator.SYSTEM_PROMPT
        assert "get_course_outline" in generator.SYSTEM_PROMPT
        assert "Tool Usage Guidelines" in generator.SYSTEM_PROMPT
        assert "Response Protocol" in generator.SYSTEM_PROMPT
        assert "Brief, Concise and focused" in generator.SYSTEM_PROMPT
    
    def test_base_params_configuration(self):
        """Test that base parameters are correctly configured"""
        generator = AIGenerator("test-key", "test-model")
        
        assert generator.base_params["model"] == "test-model"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800
        
        # Ensure no extra unexpected parameters
        expected_keys = {"model", "temperature", "max_tokens"}
        assert set(generator.base_params.keys()) == expected_keys
    
    def test_handle_tool_execution_no_tools(self, mock_anthropic_client):
        """Test handle_tool_execution when no tool_use blocks present"""
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        # Mock response with only text, no tools
        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Just text, no tools"
        
        mock_initial_response = MagicMock()
        mock_initial_response.content = [mock_text]
        
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock(text="Final response")]
        
        mock_anthropic_client.messages.create.return_value = mock_final_response
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool = Mock()
        
        base_params = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "System prompt",
            "temperature": 0,
            "max_tokens": 800
        }
        
        result = generator._handle_tool_execution(
            mock_initial_response,
            base_params,
            mock_tool_manager
        )
        
        # Tool manager should not be called
        mock_tool_manager.execute_tool.assert_not_called()
        
        # Final call should still be made
        mock_anthropic_client.messages.create.assert_called_once()
        
        assert result == "Final response"
    
    def test_generate_response_with_empty_query(self, mock_anthropic_client):
        """Test handling of empty query"""
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response to empty query")]
        mock_response.stop_reason = "stop"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        result = generator.generate_response("")
        
        # Should still make API call with empty content
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert call_args["messages"][0]["content"] == ""
        
        assert result == "Response to empty query"
    
    def test_two_round_sequential_tool_calling(self, mock_anthropic_client):
        """Test successful two-round sequential tool execution"""
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        # Create mock tool use blocks for first round
        mock_tool_use_1 = MagicMock()
        mock_tool_use_1.type = "tool_use"
        mock_tool_use_1.name = "get_course_outline"
        mock_tool_use_1.id = "tool_1"
        mock_tool_use_1.input = {"course_name": "Python"}
        
        # Mock first response with tool use
        mock_first_response = MagicMock()
        mock_first_response.content = [mock_tool_use_1]
        mock_first_response.stop_reason = "tool_use"
        
        # Create mock tool use blocks for second round
        mock_tool_use_2 = MagicMock()
        mock_tool_use_2.type = "tool_use"
        mock_tool_use_2.name = "search_course_content"
        mock_tool_use_2.id = "tool_2"
        mock_tool_use_2.input = {"query": "lesson 4 content"}
        
        # Mock second response with tool use
        mock_second_response = MagicMock()
        mock_second_response.content = [mock_tool_use_2]
        mock_second_response.stop_reason = "tool_use"
        
        # Mock final response after two rounds
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock(text="Combined analysis of course outline and content")]
        mock_final_response.stop_reason = "stop"
        
        # Set up the mock to return different responses
        mock_anthropic_client.messages.create.side_effect = [
            mock_first_response,
            mock_second_response,
            mock_final_response
        ]
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool = Mock(side_effect=[
            "Course outline: Lesson 1, Lesson 2, Lesson 3, Lesson 4: Advanced Topics",
            "Content found: Advanced Python concepts including decorators and metaclasses"
        ])
        
        tools = [
            {
                "name": "get_course_outline",
                "description": "Get course outline",
                "input_schema": {"type": "object", "properties": {}}
            },
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {"type": "object", "properties": {}}
            }
        ]
        
        result = generator.generate_response(
            query="What topics are covered in lesson 4 of the Python course?",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline",
            course_name="Python"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content",
            query="lesson 4 content"
        )
        
        # Verify three API calls were made (2 with tools, 1 final without)
        assert mock_anthropic_client.messages.create.call_count == 3
        
        # Verify first two calls had tools, last one didn't
        first_call_args = mock_anthropic_client.messages.create.call_args_list[0][1]
        assert "tools" in first_call_args
        
        second_call_args = mock_anthropic_client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_args
        
        third_call_args = mock_anthropic_client.messages.create.call_args_list[2][1]
        assert "tools" not in third_call_args
        
        assert result == "Combined analysis of course outline and content"
    
    def test_max_rounds_enforcement(self, mock_anthropic_client):
        """Test that execution stops at max_rounds even if Claude wants more tools"""
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        # Create mock tool use blocks
        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.id = "tool_id"
        mock_tool_use.input = {"query": "search query"}
        
        # Mock responses that always want to use tools
        mock_tool_response = MagicMock()
        mock_tool_response.content = [mock_tool_use]
        mock_tool_response.stop_reason = "tool_use"
        
        # Mock final response
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock(text="Final response after max rounds")]
        mock_final_response.stop_reason = "stop"
        
        # Set up mock to return tool use twice, then final response
        mock_anthropic_client.messages.create.side_effect = [
            mock_tool_response,  # Round 1
            mock_tool_response,  # Round 2
            mock_final_response  # Final call after max rounds
        ]
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool = Mock(return_value="Search result")
        
        tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {"type": "object", "properties": {}}
            }
        ]
        
        result = generator.generate_response(
            query="Complex query",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        # Verify exactly 2 tool executions (max_rounds)
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Verify 3 API calls (2 rounds + 1 final)
        assert mock_anthropic_client.messages.create.call_count == 3
        
        assert result == "Final response after max rounds"
    
    def test_single_round_early_termination(self, mock_anthropic_client):
        """Test that execution stops early if Claude doesn't use tools"""
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        # Create mock tool use for first round
        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.id = "tool_1"
        mock_tool_use.input = {"query": "Python basics"}
        
        # Mock first response with tool use
        mock_first_response = MagicMock()
        mock_first_response.content = [mock_tool_use]
        mock_first_response.stop_reason = "tool_use"
        
        # Mock second response without tool use (early termination)
        mock_second_response = MagicMock()
        mock_second_response.content = [MagicMock(text="Complete response without needing more tools")]
        mock_second_response.stop_reason = "stop"
        
        # Set up the mock
        mock_anthropic_client.messages.create.side_effect = [
            mock_first_response,
            mock_second_response
        ]
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool = Mock(return_value="Python is a programming language")
        
        tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {"type": "object", "properties": {}}
            }
        ]
        
        result = generator.generate_response(
            query="What is Python?",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        # Verify only 1 tool execution
        assert mock_tool_manager.execute_tool.call_count == 1
        
        # Verify only 2 API calls (not 3)
        assert mock_anthropic_client.messages.create.call_count == 2
        
        assert result == "Complete response without needing more tools"
    
    def test_tool_error_handling_in_rounds(self, mock_anthropic_client):
        """Test error handling during tool execution in sequential rounds"""
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        # Create mock tool use blocks
        mock_tool_use_1 = MagicMock()
        mock_tool_use_1.type = "tool_use"
        mock_tool_use_1.name = "search_course_content"
        mock_tool_use_1.id = "tool_1"
        mock_tool_use_1.input = {"query": "test query"}
        
        mock_tool_use_2 = MagicMock()
        mock_tool_use_2.type = "tool_use"
        mock_tool_use_2.name = "get_course_outline"
        mock_tool_use_2.id = "tool_2"
        mock_tool_use_2.input = {"course_name": "Python"}
        
        # Mock first response with tool use
        mock_first_response = MagicMock()
        mock_first_response.content = [mock_tool_use_1, mock_tool_use_2]
        mock_first_response.stop_reason = "tool_use"
        
        # Mock second response (continues despite error)
        mock_second_response = MagicMock()
        mock_second_response.content = [MagicMock(text="Response handling the error gracefully")]
        mock_second_response.stop_reason = "stop"
        
        mock_anthropic_client.messages.create.side_effect = [
            mock_first_response,
            mock_second_response
        ]
        
        # Create mock tool manager that throws error for one tool
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool = Mock(side_effect=[
            Exception("Tool execution failed"),
            "Course outline retrieved successfully"
        ])
        
        tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {"type": "object", "properties": {}}
            },
            {
                "name": "get_course_outline",
                "description": "Get course outline",
                "input_schema": {"type": "object", "properties": {}}
            }
        ]
        
        result = generator.generate_response(
            query="Test query",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify both tools were attempted
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Verify error was handled and included in messages
        second_call_args = mock_anthropic_client.messages.create.call_args_list[1][1]
        messages = second_call_args["messages"]
        
        # Find the tool results message
        tool_results_msg = None
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                if any(item.get("type") == "tool_result" for item in msg["content"]):
                    tool_results_msg = msg
                    break
        
        assert tool_results_msg is not None
        
        # Check that error message was included
        error_found = False
        for item in tool_results_msg["content"]:
            if "Tool execution error" in item.get("content", ""):
                error_found = True
                break
        
        assert error_found
        assert result == "Response handling the error gracefully"
    
    def test_message_accumulation_across_rounds(self, mock_anthropic_client):
        """Test that message history accumulates correctly across rounds"""
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        # Create mock tool use for first round
        mock_tool_use_1 = MagicMock()
        mock_tool_use_1.type = "tool_use"
        mock_tool_use_1.name = "get_course_outline"
        mock_tool_use_1.id = "tool_1"
        mock_tool_use_1.input = {"course_name": "ML"}
        
        # Mock first response
        mock_first_response = MagicMock()
        mock_first_response.content = [mock_tool_use_1]
        mock_first_response.stop_reason = "tool_use"
        
        # Create mock tool use for second round
        mock_tool_use_2 = MagicMock()
        mock_tool_use_2.type = "tool_use"
        mock_tool_use_2.name = "search_course_content"
        mock_tool_use_2.id = "tool_2"
        mock_tool_use_2.input = {"query": "supervised learning", "course_name": "ML"}
        
        # Mock second response
        mock_second_response = MagicMock()
        mock_second_response.content = [mock_tool_use_2]
        mock_second_response.stop_reason = "tool_use"
        
        # Mock final response
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock(text="Final comprehensive response")]
        mock_final_response.stop_reason = "stop"
        
        mock_anthropic_client.messages.create.side_effect = [
            mock_first_response,
            mock_second_response,
            mock_final_response
        ]
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool = Mock(side_effect=[
            "ML Course: Lesson 1: Intro, Lesson 2: Supervised Learning",
            "Supervised learning uses labeled data for training"
        ])
        
        tools = [
            {
                "name": "get_course_outline",
                "description": "Get course outline",
                "input_schema": {"type": "object", "properties": {}}
            },
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {"type": "object", "properties": {}}
            }
        ]
        
        result = generator.generate_response(
            query="Tell me about supervised learning in the ML course",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify message accumulation in final call
        final_call_args = mock_anthropic_client.messages.create.call_args_list[2][1]
        final_messages = final_call_args["messages"]
        
        # Should have:
        # 1. Initial user message
        # 2. First assistant response (tool use)
        # 3. First tool results (user)
        # 4. Second assistant response (tool use)
        # 5. Second tool results (user)
        assert len(final_messages) == 5
        
        # Verify message roles alternate correctly
        assert final_messages[0]["role"] == "user"  # Initial query
        assert final_messages[1]["role"] == "assistant"  # First tool use
        assert final_messages[2]["role"] == "user"  # First tool results
        assert final_messages[3]["role"] == "assistant"  # Second tool use
        assert final_messages[4]["role"] == "user"  # Second tool results
        
        # Verify initial query is preserved
        assert final_messages[0]["content"] == "Tell me about supervised learning in the ML course"
        
        assert result == "Final comprehensive response"