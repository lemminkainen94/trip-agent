"""Destination Information Agent implementation for the Trip Agent system."""

from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field

from src.agents.base import BaseAgent
from src.models.preferences import UserPreferences
from src.models.trip import Attraction, Location


class DestinationInformationAgent(BaseAgent):
    """Agent responsible for researching and providing destination information."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[Any] = None,
        **data
    ):
        """Initialize the Destination Information Agent.
        
        Args:
            llm: Language model to use for this agent.
            tools: Optional list of tools available to this agent.
            memory: Optional memory system for this agent.
            **data: Additional data for the agent.
        """
        super().__init__(
            name="Destination Information Agent",
            description=(
                "an agent that researches and provides comprehensive information about travel destinations. "
                "You compile data on main attractions, historical sites, cultural context, local cuisine, "
                "and other information that doesn't change frequently."
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
            "\n\nWhen providing destination information:"
            "\n1. Focus on evergreen information that doesn't change frequently."
            "\n2. Provide detailed information about main attractions, historical sites, and cultural context."
            "\n3. Include information about local cuisine and dining options."
            "\n4. Organize information by category (attractions, history, culture, food, etc.)."
            "\n5. Provide specific details rather than general statements."
            "\n6. Consider the user's preferences and interests when highlighting attractions."
        )
        return base_prompt + additional_instructions
    
    async def process(
        self, 
        input_text: str, 
        user_preferences: Optional[UserPreferences] = None
    ) -> Dict[str, Any]:
        """Process input and generate destination information.
        
        Args:
            input_text: The input text containing the destination to research.
            user_preferences: Optional user preferences to consider.
            
        Returns:
            Dict containing the agent's response and structured destination data.
        """
        # Create a prompt with user preferences if available
        prompt = self.create_prompt()
        
        # Add user preferences context if available
        if user_preferences:
            preferences_text = str(user_preferences)
            context = (
                f"The user has the following preferences: {preferences_text}\n\n"
                f"Research request: {input_text}"
            )
        else:
            context = f"Research request: {input_text}"
        
        # Generate response using the chain approach
        chain = prompt | self.llm
        response = await chain.ainvoke({"input": context})
        
        # Extract structured destination data
        destination_data = await self._extract_destination_data(input_text, response.content)
        
        return {
            "response": response.content,
            "destination_data": destination_data
        }
    
    async def _extract_destination_data(
        self, 
        destination_name: str, 
        research_text: str
    ) -> Dict[str, Any]:
        """Extract structured destination data from research text.
        
        Args:
            destination_name: The name of the destination.
            research_text: The research text containing destination information.
            
        Returns:
            Dict containing structured destination data.
        """
        # Use the LLM to extract structured data from the research text
        system_message = (
            "Extract structured destination data from the research text. "
            "Return a JSON object with the following structure:\n"
            "{\n"
            "  \"location\": {\n"
            "    \"name\": \"destination name\",\n"
            "    \"latitude\": latitude as float,\n"
            "    \"longitude\": longitude as float\n"
            "  },\n"
            "  \"attractions\": [\n"
            "    {\n"
            "      \"name\": \"attraction name\",\n"
            "      \"description\": \"detailed description\",\n"
            "      \"category\": \"category (museum, park, etc.)\",\n"
            "      \"visit_duration\": estimated visit duration in minutes\n"
            "    },\n"
            "    ...\n"
            "  ],\n"
            "  \"cuisine\": [\"cuisine 1\", \"cuisine 2\", ...],\n"
            "  \"cultural_notes\": [\"cultural note 1\", \"cultural note 2\", ...],\n"
            "  \"historical_context\": \"brief historical context\"\n"
            "}\n"
            "If exact coordinates are not known, provide reasonable estimates."
        )
        
        human_message = f"Destination: {destination_name}\n\nResearch: {research_text}"
        
        # Use direct messages instead of a template
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ]
        
        # Extract structured data using the LLM directly with messages
        extraction_response = await self.llm.ainvoke(messages)
        
        try:
            # Try to parse the response as JSON
            import json
            from json import JSONDecodeError
            
            # Extract JSON from the response if it's wrapped in markdown or other text
            import re
            json_match = re.search(r'```json\n(.*?)\n```', extraction_response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = extraction_response.content
            
            structured_data = json.loads(json_str)
            
            # Convert to our data models
            location = Location(
                name=structured_data["location"]["name"],
                latitude=structured_data["location"]["latitude"],
                longitude=structured_data["location"]["longitude"]
            )
            
            attractions = []
            for attraction_data in structured_data.get("attractions", []):
                attraction = Attraction(
                    name=attraction_data["name"],
                    description=attraction_data["description"],
                    location=location,  # Use the main location for now
                    category=attraction_data["category"],
                    visit_duration=attraction_data["visit_duration"]
                )
                attractions.append(attraction)
            
            return {
                "location": location,
                "attractions": attractions,
                "cuisine": structured_data.get("cuisine", []),
                "cultural_notes": structured_data.get("cultural_notes", []),
                "historical_context": structured_data.get("historical_context", "")
            }
        
        except (JSONDecodeError, KeyError, AttributeError) as e:
            # If parsing fails, log the error and return a minimal structure
            print(f"Error extracting destination data: {e}")
            return {
                "location": Location(name=destination_name, latitude=0.0, longitude=0.0),
                "attractions": [],
                "cuisine": [],
                "cultural_notes": [],
                "historical_context": ""
            }
