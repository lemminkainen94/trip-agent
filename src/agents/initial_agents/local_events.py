"""Local Events & Logistics Agent implementation for the Trip Agent system."""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from pydantic import Field

from src.agents.base import BaseAgent
from src.models.preferences import UserPreferences
from src.models.trip import Location


class LocalEventsAgent(BaseAgent):
    """Agent responsible for handling time-sensitive and logistical information."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[Any] = None,
        **data
    ):
        """Initialize the Local Events & Logistics Agent.
        
        Args:
            llm: Language model to use for this agent.
            tools: Optional list of tools available to this agent.
            memory: Optional memory system for this agent.
            **data: Additional data for the agent.
        """
        super().__init__(
            name="Local Events & Logistics Agent",
            description=(
                "an agent that handles time-sensitive and logistical information for trip planning. "
                "You track event schedules, venue information, opening hours, calculate visit durations, "
                "flag time-sensitive considerations, and provide updated information on seasonal activities."
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
            "\n\nWhen providing logistics and event information:"
            "\n1. Focus on time-sensitive information like events, festivals, and seasonal activities."
            "\n2. Provide detailed information about opening hours, ticket availability, and booking requirements."
            "\n3. Calculate realistic visit durations and travel times between locations."
            "\n4. Flag any time-sensitive considerations like peak hours or seasonal closures."
            "\n5. Consider weather patterns and seasonal factors that might affect the trip."
            "\n6. Provide specific, actionable information rather than general statements."
        )
        return base_prompt + additional_instructions
    
    async def process(
        self, 
        input_text: str, 
        user_preferences: Optional[UserPreferences] = None,
        location: Optional[Location] = None,
        date_range: Optional[tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """Process input and generate logistics and event information.
        
        Args:
            input_text: The input text containing the logistics request.
            user_preferences: Optional user preferences to consider.
            location: Optional location to research events for.
            date_range: Optional date range (start_date, end_date) to research events for.
            
        Returns:
            Dict containing the agent's response and structured logistics data.
        """
        # Create a prompt with user preferences and location if available
        prompt = self.create_prompt()
        
        # Build context with all available information
        context_parts = [f"Logistics request: {input_text}"]
        
        if user_preferences:
            preferences_text = str(user_preferences)
            context_parts.append(f"User preferences: {preferences_text}")
        
        if location:
            context_parts.append(f"Location: {location.name} (Coordinates: {location.latitude}, {location.longitude})")
        
        if date_range:
            start_date, end_date = date_range
            context_parts.append(
                f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Generate response using the chain approach
        chain = prompt | self.llm
        response = await chain.ainvoke({"input": context})
        
        # Extract structured logistics data
        logistics_data = await self._extract_logistics_data(
            input_text, response.content, location, date_range
        )
        
        return {
            "response": response.content,
            "logistics_data": logistics_data
        }
    
    async def _extract_logistics_data(
        self, 
        request_text: str, 
        research_text: str,
        location: Optional[Location] = None,
        date_range: Optional[tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """Extract structured logistics data from research text.
        
        Args:
            request_text: The original request text.
            research_text: The research text containing logistics information.
            location: Optional location the logistics are for.
            date_range: Optional date range (start_date, end_date) the logistics are for.
            
        Returns:
            Dict containing structured logistics data.
        """
        # Use the LLM to extract structured data from the research text
        system_message = (
            "Extract structured logistics and event data from the research text. "
            "Return a JSON object with the following structure:\n"
            "{\n"
            "  \"events\": [\n"
            "    {\n"
            "      \"name\": \"event name\",\n"
            "      \"description\": \"detailed description\",\n"
            "      \"date\": \"YYYY-MM-DD\",\n"
            "      \"start_time\": \"HH:MM\",\n"
            "      \"end_time\": \"HH:MM\",\n"
            "      \"location\": \"event location\",\n"
            "      \"ticket_required\": true/false,\n"
            "      \"price\": price as float or null\n"
            "    },\n"
            "    ...\n"
            "  ],\n"
            "  \"opening_hours\": {\n"
            "    \"attraction_name\": {\n"
            "      \"Monday\": \"10:00-18:00\",\n"
            "      ...\n"
            "    },\n"
            "    ...\n"
            "  },\n"
            "  \"seasonal_notes\": [\"seasonal note 1\", \"seasonal note 2\", ...],\n"
            "  \"travel_times\": {\n"
            "    \"location_pair\": \"estimated time in minutes\",\n"
            "    ...\n"
            "  }\n"
            "}\n"
        )
        
        human_message = (
            f"Request: {request_text}\n\n"
            f"Location: {location.name if location else 'Unknown'}\n\n"
            f"Date Range: {date_range[0].strftime('%Y-%m-%d') if date_range else 'Unknown'} to "
            f"{date_range[1].strftime('%Y-%m-%d') if date_range else 'Unknown'}\n\n"
            f"Research: {research_text}"
        )
        
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
            
            return {
                "events": structured_data.get("events", []),
                "opening_hours": structured_data.get("opening_hours", {}),
                "seasonal_notes": structured_data.get("seasonal_notes", []),
                "travel_times": structured_data.get("travel_times", {})
            }
        
        except (JSONDecodeError, KeyError, AttributeError) as e:
            # If parsing fails, log the error and return a minimal structure
            print(f"Error extracting logistics data: {e}")
            return {
                "events": [],
                "opening_hours": {},
                "seasonal_notes": [],
                "travel_times": {}
            }