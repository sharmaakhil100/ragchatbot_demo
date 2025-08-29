from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
1. **search_course_content**: Search within course materials for specific content
   - Use for questions about lesson content, concepts, or detailed educational materials
   - Can filter by course name and/or lesson number

2. **get_course_outline**: Get complete course structure and lesson list
   - Use for questions about course organization, structure, or available lessons
   - Returns course title, course link, and all lesson numbers with titles
   - Perfect for "what lessons are in...", "show me the outline of...", or "list the topics in..." queries

Multi-Round Tool Strategy:
- You can make up to 2 sequential tool calls to gather comprehensive information
- After each tool result, assess if additional searches would improve your response
- Use follow-up tool calls for:
  * Getting course structure after content search
  * Searching different courses or lessons for comparisons
  * Finding related topics across multiple courses
  * Clarifying or expanding on initial results
- Each round allows one tool call maximum

Tool Usage Guidelines:
- **First round**: Use the most relevant tool for the primary question
- **Second round**: Use additional tools if the first results suggest more information is needed
- **Course outline/structure questions**: Use get_course_outline tool
- **Content-specific questions**: Use search_course_content tool
- Synthesize all tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course-specific questions**: Use appropriate tool(s) strategically
- **Complex queries**: Use multiple rounds to gather complete information before responding
- **For outline queries**: Always include the course title, course link (if available), and complete lesson list with numbers and titles
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the outline tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Complete** - For outlines, include all course structure details
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
        max_rounds: int = 2,
    ) -> str:
        """
        Generate AI response with sequential tool calling support.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds (default 2)

        Returns:
            Generated response as string
        """

        # Build system content with round guidance
        system_content = self._build_system_content(conversation_history, max_rounds)

        # Initialize messages and round tracking
        messages = [{"role": "user", "content": query}]
        current_round = 0

        # Execute up to max_rounds of tool calling
        while current_round < max_rounds:
            # Make API call with tools available
            response = self._make_api_call(messages, system_content, tools)

            # If no tool use, we're done
            if response.stop_reason != "tool_use" or not tool_manager:
                return response.content[0].text

            # Execute tools and update messages for next round
            messages = self._execute_tools_and_update_messages(
                response, messages, tool_manager
            )
            current_round += 1

        # Max rounds reached - make final call without tools
        return self._make_final_call(messages, system_content)

    def _build_system_content(
        self, conversation_history: Optional[str], max_rounds: int
    ) -> str:
        """
        Build system content with conversation history and round guidance.

        Args:
            conversation_history: Previous messages for context
            max_rounds: Maximum number of tool calling rounds

        Returns:
            Complete system content string
        """
        base_content = self.SYSTEM_PROMPT

        # Add round limit context if using tools
        if max_rounds > 1:
            round_context = f"\n\nNote: You have up to {max_rounds} rounds of tool calls available for comprehensive information gathering."
            base_content += round_context

        # Add conversation history if available
        if conversation_history:
            return f"{base_content}\n\nPrevious conversation:\n{conversation_history}"
        return base_content

    def _make_api_call(
        self,
        messages: List[Dict[str, Any]],
        system_content: str,
        tools: Optional[List] = None,
    ) -> Any:
        """
        Make API call with consistent tool availability.

        Args:
            messages: Current message history
            system_content: System prompt content
            tools: Available tools (if any)

        Returns:
            API response object
        """
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        # Include tools if available (key change: always include during rounds)
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        return self.client.messages.create(**api_params)

    def _execute_tools_and_update_messages(
        self, response, messages: List[Dict[str, Any]], tool_manager
    ) -> List[Dict[str, Any]]:
        """
        Execute tools and update message history for next round.

        Args:
            response: API response containing tool use requests
            messages: Current message history
            tool_manager: Manager to execute tools

        Returns:
            Updated message history
        """
        # Create a copy to avoid modifying original
        updated_messages = messages.copy()

        # Add AI's tool use response
        updated_messages.append({"role": "assistant", "content": response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )
                except Exception as e:
                    # Handle individual tool execution errors
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Tool execution error: {str(e)}",
                        }
                    )

        # Add tool results as user message
        if tool_results:
            updated_messages.append({"role": "user", "content": tool_results})

        return updated_messages

    def _make_final_call(
        self, messages: List[Dict[str, Any]], system_content: str
    ) -> str:
        """
        Make final API call without tools when rounds are exhausted.

        Args:
            messages: Complete message history
            system_content: System prompt content

        Returns:
            Final response text
        """
        # Make API call without tools
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
            # Note: No tools parameter - this is the final response
        }

        response = self.client.messages.create(**api_params)
        return response.content[0].text

    # Keep _handle_tool_execution for backward compatibility
    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Legacy method for backward compatibility.
        Redirects to new sequential approach with max_rounds=1.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Extract messages and add the initial response
        messages = base_params["messages"].copy()

        # Execute tools and update messages
        messages = self._execute_tools_and_update_messages(
            initial_response, messages, tool_manager
        )

        # Make final call without tools (matching original behavior)
        return self._make_final_call(messages, base_params["system"])
