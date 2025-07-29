"""User Interface Agent implementation for the Trip Agent system."""

from typing import Any, Dict, List, Optional, Set
from enum import Enum
import json
import re
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from pydantic import Field, BaseModel

from src.agents.base import BaseAgent
from src.models.preferences import UserPreferences


class ConversationState(str, Enum):
    """Enum representing the state of the conversation with the user."""
    GREETING = "greeting"
    COLLECTING_DESTINATION = "collecting_destination"
    COLLECTING_DATES = "collecting_dates"
    COLLECTING_INTERESTS = "collecting_interests"
    COLLECTING_BUDGET = "collecting_budget"
    COLLECTING_ACCOMMODATION = "collecting_accommodation"
    COLLECTING_ADDITIONAL_PREFERENCES = "collecting_additional_preferences"
    CONFIRMATION = "confirmation"
    READY = "ready"
    REFINEMENT = "refinement"


class ConversationContext(BaseModel):
    """Model to track the conversation context and state."""
    state: ConversationState = ConversationState.GREETING
    missing_info: Set[str] = Field(default_factory=set)
    attempts: Dict[str, int] = Field(default_factory=dict)
    confirmed: bool = False


class UserInterfaceAgent(BaseAgent):
    """Agent responsible for managing user communication and preference capture."""
    
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="History of conversation with the user"
    )
    
    conversation_context: ConversationContext = Field(
        default_factory=ConversationContext,
        description="Context tracking the state of the conversation"
    )
    
    def __init__(
        self,
        llm: BaseChatModel,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[Any] = None,
        **data
    ):
        """Initialize the User Interface Agent.
        
        Args:
            llm: Language model to use for this agent.
            tools: Optional list of tools available to this agent.
            memory: Optional memory system for this agent.
            **data: Additional data for the agent.
        """
        super().__init__(
            name="User Interface Agent",
            description=(
                "an agent that manages user communication and preference capture. "
                "You lead conversations with users, extract and store their preferences, "
                "ask follow-up questions for clarification, and ensure all necessary "
                "information is collected for trip planning."
            ),
            llm=llm,
            tools=tools or [],
            memory=memory,
            **data
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent.
        
        Returns:
            str: The system prompt for this agent.
        """
        base_prompt = super().get_system_prompt()
        additional_instructions = (
            "\n\nWhen interacting with users:"
            "\n1. Be conversational, friendly, and helpful."
            "\n2. Extract key preferences like destination, dates, interests, budget, etc."
            "\n3. Ask clarifying questions when information is missing or ambiguous."
            "\n4. Summarize the user's preferences to confirm understanding."
            "\n5. Store relevant information for later retrieval."
            "\n6. Handle follow-up questions and changes to preferences gracefully."
            "\n7. Guide the conversation to collect all necessary information for trip planning."
            "\n8. Respond to user questions while still guiding the conversation forward."
        )
        return base_prompt + additional_instructions
    
    async def process(
        self, 
        input_text: str, 
        user_preferences: Optional[UserPreferences] = None
    ) -> Dict[str, Any]:
        """Process user input and generate a response.
        
        Args:
            input_text: The input text from the user.
            user_preferences: Optional existing user preferences.
            
        Returns:
            Dict containing the agent's response and updated preferences.
        """
        # Initialize or use existing preferences
        preferences = user_preferences or UserPreferences()
        
        # Update conversation history
        if input_text:
            self.conversation_history.append({"role": "user", "content": input_text})
        
        # Check if we're in confirmation state and user is confirming
        if (self.conversation_context.state == ConversationState.CONFIRMATION and input_text):
            # Look for affirmative responses in the user's message
            user_response = input_text.lower()
            affirmative_words = ["yes", "correct", "right", "sure", "confirm", "proceed", "looks good", "sounds good", "go on", "carry on"]
            
            if any(word in user_response for word in affirmative_words):
                self.conversation_context.confirmed = True
                # Immediately update state to READY
                self.conversation_context.state = ConversationState.READY
        
        # Extract preferences from the input text
        updated_preferences = self._extract_preferences(input_text, preferences)
        
        # Update the conversation context based on the preferences
        self._update_conversation_context(updated_preferences)
        
        # Generate a response based on the conversation state
        response = await self._generate_response(updated_preferences)
        
        # Add the response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return {
            "response": response,
            "preferences": updated_preferences
        }
    
    def _extract_preferences(
        self, 
        input_text: str, 
        existing_preferences: Optional[UserPreferences] = None
    ) -> UserPreferences:
        """Extract user preferences from input text.
        
        Args:
            input_text: The input text to extract preferences from.
            existing_preferences: Optional existing user preferences to update.
            
        Returns:
            UserPreferences: Updated user preferences.
        """
        # Start with existing preferences or create new ones
        preferences = existing_preferences or UserPreferences()
        
        if not input_text:
            return preferences
        
        # Use the LLM to extract preferences from the input text
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Extract travel preferences from the user's message. "
                "Return a JSON object with the following fields if mentioned: "
                "destination, start_date, end_date, name, travel_style, interests (as a list), "
                "activity_level, accommodation_type, budget_range, dietary_restrictions (as a list), "
                "accessibility_needs (as a list), preferred_transportation (as a list), "
                "excluded_categories (as a list). "
                "For dates, convert to ISO format (YYYY-MM-DD) if possible. "
                "Only include fields that are explicitly mentioned or can be directly inferred."
            )),
            ("human", input_text)
        ])
        
        # Extract preferences using the LLM
        chain = extraction_prompt | self.llm
        extraction_response = chain.invoke({})
        
        try:
            # Try to parse the response as JSON
            # Extract JSON from the response if it's wrapped in markdown or other text
            json_match = re.search(r'```(?:json)?\n(.*?)\n```', extraction_response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = extraction_response.content
            
            # Clean up the JSON string
            json_str = re.sub(r'[\n\r\t]', '', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)
            
            extracted = json.loads(json_str)
            
            # Update preferences with extracted values
            for key, value in extracted.items():
                if hasattr(preferences, key) and value:
                    # Handle date parsing
                    if key in ['start_date', 'end_date'] and isinstance(value, str):
                        try:
                            # Try to parse various date formats and store as ISO format string
                            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%B %d, %Y', '%b %d, %Y']:
                                try:
                                    date_value = datetime.strptime(value, fmt)
                                    # Store as string in ISO format
                                    setattr(preferences, key, date_value.strftime('%Y-%m-%d'))
                                    break
                                except ValueError:
                                    continue
                        except Exception:
                            # If date parsing fails, store as original string
                            setattr(preferences, key, value)
                    else:
                        setattr(preferences, key, value)
        
        except Exception as e:
            # If parsing fails, log the error and continue with existing preferences
            print(f"Error extracting preferences: {e}")
        
        return preferences
    
    def _update_conversation_context(self, preferences: UserPreferences) -> None:
        """Update the conversation context based on the current preferences.
        
        Args:
            preferences: The current user preferences.
        """
        context = self.conversation_context
        
        # Check for missing required information
        missing_info = set()
        
        # Check if destination is missing
        if not getattr(preferences, "destination", None):
            missing_info.add("destination")
        
        # Check if start_date is missing
        if not getattr(preferences, "start_date", None):
            missing_info.add("start_date")
        
        # Check if end_date is missing
        if not getattr(preferences, "end_date", None):
            missing_info.add("end_date")
        
        # Check if interests are missing
        if not getattr(preferences, "interests", None):
            missing_info.add("interests")
        
        # Update the missing info in the context
        context.missing_info = missing_info
        
        # Determine the conversation state based on missing info
        if not context.missing_info and not context.confirmed:
            context.state = ConversationState.CONFIRMATION
        elif not context.missing_info and context.confirmed:
            context.state = ConversationState.READY
        elif "destination" in context.missing_info:
            context.state = ConversationState.COLLECTING_DESTINATION
        elif "start_date" in context.missing_info or "end_date" in context.missing_info:
            context.state = ConversationState.COLLECTING_DATES
        elif "interests" in context.missing_info:
            context.state = ConversationState.COLLECTING_INTERESTS
        elif not getattr(preferences, "budget_range", None):
            context.state = ConversationState.COLLECTING_BUDGET
        elif not getattr(preferences, "accommodation_type", None):
            context.state = ConversationState.COLLECTING_ACCOMMODATION
        else:
            context.state = ConversationState.COLLECTING_ADDITIONAL_PREFERENCES
    
    async def _generate_response(self, preferences: UserPreferences) -> str:
        """Generate a response based on the conversation state and preferences.
        
        Args:
            preferences: The current user preferences.
            
        Returns:
            str: The generated response.
        """
        context = self.conversation_context
        state = context.state
        
        # Format the conversation history for context
        history_text = self._format_message_history(self.conversation_history)
        
        # Create a prompt based on the conversation state
        if state == ConversationState.GREETING:
            prompt_template = (
                "You are a helpful travel assistant. The user is starting a conversation with you. "
                "Greet them warmly and ask about their travel plans. "
                "Ask specifically about their destination and travel dates. "
                "Be conversational and friendly.\n\n"
                "Conversation history:\n{history}\n\n"
                "Your response:"
            )
        
        elif state == ConversationState.COLLECTING_DESTINATION:
            prompt_template = (
                "You are a helpful travel assistant. The user hasn't specified a clear destination yet. "
                "Ask them about where they want to go for their trip. "
                "Be conversational and friendly.\n\n"
                "Current preferences: {preferences}\n\n"
                "Conversation history:\n{history}\n\n"
                "Your response (ask about destination):"
            )
        
        elif state == ConversationState.COLLECTING_DATES:
            prompt_template = (
                "You are a helpful travel assistant. The user has mentioned {destination} as their destination, "
                "but we need to know their travel dates. "
                "Ask them when they plan to travel and for how long. "
                "Be conversational and friendly.\n\n"
                "Current preferences: {preferences}\n\n"
                "Conversation history:\n{history}\n\n"
                "Your response (ask about travel dates):"
            )
        
        elif state == ConversationState.COLLECTING_INTERESTS:
            prompt_template = (
                "You are a helpful travel assistant. The user is planning a trip to {destination} "
                "from {start_date} to {end_date}. "
                "Ask them about their interests and what they'd like to do during their trip. "
                "Suggest some common interests like museums, outdoor activities, food experiences, etc. "
                "Be conversational and friendly.\n\n"
                "Current preferences: {preferences}\n\n"
                "Conversation history:\n{history}\n\n"
                "Your response (ask about interests):"
            )
        
        elif state == ConversationState.COLLECTING_BUDGET:
            prompt_template = (
                "You are a helpful travel assistant. The user is planning a trip to {destination} "
                "from {start_date} to {end_date} with interests in {interests}. "
                "Ask them about their budget range for the trip. "
                "Be conversational and friendly.\n\n"
                "Current preferences: {preferences}\n\n"
                "Conversation history:\n{history}\n\n"
                "Your response (ask about budget):"
            )
        
        elif state == ConversationState.COLLECTING_ACCOMMODATION:
            prompt_template = (
                "You are a helpful travel assistant. The user is planning a trip to {destination} "
                "from {start_date} to {end_date} with a budget of {budget}. "
                "Ask them about their preferred accommodation type (hotel, hostel, Airbnb, etc.). "
                "Be conversational and friendly.\n\n"
                "Current preferences: {preferences}\n\n"
                "Conversation history:\n{history}\n\n"
                "Your response (ask about accommodation):"
            )
        
        elif state == ConversationState.COLLECTING_ADDITIONAL_PREFERENCES:
            prompt_template = (
                "You are a helpful travel assistant. The user is planning a trip to {destination} "
                "from {start_date} to {end_date}. "
                "Ask them about any additional preferences they might have, such as dietary restrictions, "
                "accessibility needs, or preferred transportation methods. "
                "Be conversational and friendly.\n\n"
                "Current preferences: {preferences}\n\n"
                "Conversation history:\n{history}\n\n"
                "Your response (ask about additional preferences):"
            )
        
        elif state == ConversationState.CONFIRMATION:
            prompt_template = (
                "You are a helpful travel assistant. The user has provided all the necessary information "
                "for their trip to {destination} from {start_date} to {end_date}. "
                "Summarize their preferences and ask them to confirm if everything is correct. "
                "Let them know that once they confirm, you'll start planning their trip. "
                "Be conversational and friendly.\n\n"
                "Current preferences: {preferences}\n\n"
                "Conversation history:\n{history}\n\n"
                "Your response (summarize and ask for confirmation):"
            )
            
            # Mark as confirmed if this is the second time in confirmation state
            if len(self.conversation_history) >= 4:
                self.conversation_context.confirmed = True
        
        elif state == ConversationState.READY:
            prompt_template = (
                "You are a helpful travel assistant. The user has confirmed their trip details "
                "for {destination} from {start_date} to {end_date}. "
                "Let them know that you're now going to create a personalized trip plan for them "
                "based on their preferences. Explain that this might take a moment. "
                "Be conversational and friendly.\n\n"
                "Current preferences: {preferences}\n\n"
                "Conversation history:\n{history}\n\n"
                "Your response (acknowledge and proceed with trip planning):"
            )
        
        else:  # Default/fallback prompt
            prompt_template = (
                "You are a helpful travel assistant. Continue the conversation with the user "
                "about their trip planning. "
                "Be conversational and friendly.\n\n"
                "Current preferences: {preferences}\n\n"
                "Conversation history:\n{history}\n\n"
                "Your response:"
            )
        
        # Format the prompt with the current context
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Prepare the variables for the prompt
        variables = {
            "history": history_text,
            "preferences": str(preferences),
            "destination": getattr(preferences, "destination", "their destination"),
            "start_date": getattr(preferences, "start_date", "the start date"),
            "end_date": getattr(preferences, "end_date", "the end date"),
            "interests": getattr(preferences, "interests", []),
            "budget": getattr(preferences, "budget_range", "their budget")
        }
        
        # Generate the response
        chain = prompt | self.llm
        response = chain.invoke(variables)
        
        return response.content
    
    def _format_message_history(self, messages: List[Dict[str, str]]) -> str:
        """Format message history for inclusion in prompts.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            
        Returns:
            str: Formatted message history.
        """
        formatted_history = ""
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                formatted_history += f"User: {content}\n"
            elif role == "assistant":
                formatted_history += f"Assistant: {content}\n"
        
        return formatted_history
