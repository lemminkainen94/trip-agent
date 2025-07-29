"""Tests for the base agent implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.base import BaseAgent


class TestBaseAgent:
    """Tests for the BaseAgent class."""
    
    def test_init(self):
        """Test initialization of the BaseAgent."""
        # Create a mock language model
        mock_llm = MagicMock(spec=BaseChatModel)
        
        # Create a BaseAgent
        agent = BaseAgent(
            name="Test Agent",
            description="A test agent",
            llm=mock_llm
        )
        
        # Check that the agent was initialized correctly
        assert agent.name == "Test Agent"
        assert agent.description == "A test agent"
        assert agent.llm == mock_llm
        assert agent.tools == []
        assert agent.memory is None
    
    def test_get_system_prompt(self):
        """Test getting the system prompt."""
        # Create a mock language model
        mock_llm = MagicMock(spec=BaseChatModel)
        
        # Create a BaseAgent
        agent = BaseAgent(
            name="Test Agent",
            description="A test agent",
            llm=mock_llm
        )
        
        # Get the system prompt
        system_prompt = agent.get_system_prompt()
        
        # Check that the system prompt contains the agent's name and description
        assert "Test Agent" in system_prompt
        assert "A test agent" in system_prompt
    
    def test_create_prompt(self):
        """Test creating a prompt template."""
        # Create a mock language model
        mock_llm = MagicMock(spec=BaseChatModel)
        
        # Create a BaseAgent
        agent = BaseAgent(
            name="Test Agent",
            description="A test agent",
            llm=mock_llm
        )
        
        # Create a prompt template
        prompt_template = agent.create_prompt()
        
        # Check that the prompt template was created correctly
        assert prompt_template is not None
        assert len(prompt_template.messages) == 2
        assert prompt_template.messages[0][0] == "system"
        assert prompt_template.messages[1][0] == "human"
    
    def test_format_message_history(self):
        """Test formatting message history."""
        # Create a mock language model
        mock_llm = MagicMock(spec=BaseChatModel)
        
        # Create a BaseAgent
        agent = BaseAgent(
            name="Test Agent",
            description="A test agent",
            llm=mock_llm
        )
        
        # Create some test messages
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
            HumanMessage(content="How are you?"),
            AIMessage(content="I'm doing well, thanks for asking!")
        ]
        
        # Format the message history
        formatted = agent._format_message_history(messages)
        
        # Check that the message history was formatted correctly
        assert "Human: Hello" in formatted
        assert "AI: Hi there" in formatted
        assert "Human: How are you?" in formatted
        assert "AI: I'm doing well, thanks for asking!" in formatted
    
    def test_process_not_implemented(self):
        """Test that process raises NotImplementedError."""
        # Create a mock language model
        mock_llm = MagicMock(spec=BaseChatModel)
        
        # Create a BaseAgent
        agent = BaseAgent(
            name="Test Agent",
            description="A test agent",
            llm=mock_llm
        )
        
        # Check that process raises NotImplementedError
        with pytest.raises(NotImplementedError):
            agent.process("Hello")
    
    def test_str(self):
        """Test string representation of the agent."""
        # Create a mock language model
        mock_llm = MagicMock(spec=BaseChatModel)
        
        # Create a BaseAgent
        agent = BaseAgent(
            name="Test Agent",
            description="A test agent",
            llm=mock_llm
        )
        
        # Check the string representation
        assert str(agent) == "Test Agent - A test agent"