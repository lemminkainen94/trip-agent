"""Tests for the User Interface Agent implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.user_interface import UserInterfaceAgent
from src.models.preferences import UserPreferences


class TestUserInterfaceAgent:
    """Tests for the UserInterfaceAgent class."""
    
    def test_init(self):
        """Test initialization of the UserInterfaceAgent."""
        # Create a mock language model
        mock_llm = MagicMock(spec=BaseChatModel)
        
        # Create a UserInterfaceAgent
        agent = UserInterfaceAgent(llm=mock_llm)
        
        # Check that the agent was initialized correctly
        assert agent.name == "User Interface Agent"
        assert "manages user communication" in agent.description.lower()
        assert agent.llm == mock_llm
        assert agent.tools == []
        assert agent.memory is None
        assert agent.conversation_history == []
    
    def test_get_system_prompt(self):
        """Test getting the system prompt."""
        # Create a mock language model
        mock_llm = MagicMock(spec=BaseChatModel)
        
        # Create a UserInterfaceAgent
        agent = UserInterfaceAgent(llm=mock_llm)
        
        # Get the system prompt
        system_prompt = agent.get_system_prompt()
        
        # Check that the system prompt contains the agent's name and description
        assert "User Interface Agent" in system_prompt
        assert "manages user communication" in system_prompt.lower()
        
        # Check that the system prompt contains additional instructions
        assert "be conversational" in system_prompt.lower()
        assert "extract key preferences" in system_prompt.lower()
        assert "ask clarifying questions" in system_prompt.lower()
    
    @pytest.mark.asyncio
    async def test_process_without_history(self):
        """Test processing input without conversation history."""
        # Create a mock language model with async invoke method
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = MagicMock()
        mock_response.content = "Hello! How can I help you plan your trip?"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Create a UserInterfaceAgent
        agent = UserInterfaceAgent(llm=mock_llm)
        
        # Process input
        result = await agent.process("I want to plan a trip to Paris")
        
        # Check that the result contains the expected keys
        assert "response" in result
        assert "preferences" in result
        
        # Check that the response is correct
        assert result["response"] == "Hello! How can I help you plan your trip?"
    
    @pytest.mark.asyncio
    async def test_process_with_history(self):
        """Test processing input with conversation history."""
        # Create a mock language model with async invoke method
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = MagicMock()
        mock_response.content = "Paris is a great choice! When are you planning to go?"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Create a UserInterfaceAgent
        agent = UserInterfaceAgent(llm=mock_llm)
        
        # Add some conversation history
        agent.conversation_history = [
            HumanMessage(content="I want to plan a trip"),
            AIMessage(content="Great! Where would you like to go?")
        ]
        
        # Process input
        result = await agent.process("I'm thinking of Paris")
        
        # Check that the result contains the expected keys
        assert "response" in result
        assert "preferences" in result
        
        # Check that the response is correct
        assert result["response"] == "Paris is a great choice! When are you planning to go?"
    
    @pytest.mark.asyncio
    async def test_process_with_preferences(self):
        """Test processing input with user preferences."""
        # Create a mock language model with async invoke method
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = MagicMock()
        mock_response.content = "I see you're interested in art and history. Paris has many museums!"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Create a UserInterfaceAgent
        agent = UserInterfaceAgent(llm=mock_llm)
        
        # Create user preferences
        preferences = UserPreferences(
            name="John",
            interests=["art", "history"],
            activity_level="moderate"
        )
        
        # Process input
        result = await agent.process("Tell me about museums in Paris", preferences)
        
        # Check that the result contains the expected keys
        assert "response" in result
        assert "preferences" in result
        
        # Check that the response is correct
        assert result["response"] == "I see you're interested in art and history. Paris has many museums!"
    
    @pytest.mark.asyncio
    async def test_extract_preferences(self):
        """Test extracting preferences from input text."""
        # Create a mock language model
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = MagicMock()
        mock_response.content = '{"name": "John", "interests": ["art", "history"], "activity_level": "moderate"}'
        mock_llm.invoke = MagicMock(return_value=mock_response)
        
        # Create a UserInterfaceAgent
        agent = UserInterfaceAgent(llm=mock_llm)
        
        # Extract preferences
        with patch('langchain_core.prompts.ChatPromptTemplate.from_messages'):
            preferences = agent._extract_preferences("I'm John and I like art and history. I prefer moderate activity.")
        
        # Check that the preferences were extracted correctly
        assert preferences.name == "John"
        assert "art" in preferences.interests
        assert "history" in preferences.interests
        assert preferences.activity_level == "moderate"