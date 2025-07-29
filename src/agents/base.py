"""Base agent implementation for the Trip Agent system."""

from typing import Any, Dict, List, Optional, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.models.preferences import UserPreferences


class BaseAgent(BaseModel):
    """Base agent class with common functionality for all agents."""
    
    name: str = Field(description="Name of the agent")
    description: str = Field(description="Description of the agent's role")
    llm: BaseChatModel = Field(description="Language model to use for this agent")
    tools: List[BaseTool] = Field(default_factory=list, description="Tools available to this agent")
    memory: Optional[Any] = Field(None, description="Memory system for this agent")
    
    model_config = {"arbitrary_types_allowed": True}
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent.
        
        Returns:
            str: The system prompt for this agent.
        """
        base_prompt = (
            f"You are {self.name}, {self.description}. "
            f"You are part of a multi-agent trip planning system. "
            f"Your goal is to provide helpful, accurate information within your domain of expertise. "
            f"Always be polite, professional, and focused on the user's needs."
        )
        
        if self.tools:
            tool_descriptions = "\n".join(
                [f"- {tool.name}: {tool.description}" for tool in self.tools]
            )
            base_prompt += f"\n\nYou have access to the following tools:\n{tool_descriptions}"
        
        return base_prompt
    
    def create_prompt(self) -> ChatPromptTemplate:
        """Create a prompt template for this agent.
        
        Returns:
            ChatPromptTemplate: The prompt template for this agent.
        """
        system_prompt = self.get_system_prompt()
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
    
    async def process(self, input_text: str, user_preferences: Optional[UserPreferences] = None) -> str:
        """Process input and generate a response.
        
        Args:
            input_text: The input text to process.
            user_preferences: Optional user preferences to consider.
            
        Returns:
            str: The agent's response.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _format_message_history(self, messages: List[BaseMessage]) -> str:
        """Format message history for inclusion in prompts.
        
        Args:
            messages: List of messages in the conversation history.
            
        Returns:
            str: Formatted message history.
        """
        formatted = []
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                formatted.append(f"AI: {message.content}")
            else:
                formatted.append(f"{message.type}: {message.content}")
        return "\n".join(formatted)
    
    def __str__(self) -> str:
        """Return string representation of the agent."""
        return f"{self.name} - {self.description}"