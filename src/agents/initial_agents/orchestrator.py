"""Orchestrator Agent implementation for the Trip Agent system."""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import Field

from src.agents.base import BaseAgent
from src.agents.user_interface import UserInterfaceAgent
from src.agents.destination_information import DestinationInformationAgent
from src.agents.local_events import LocalEventsAgent
from src.agents.itinerary_optimization import ItineraryOptimizationAgent
from src.models.preferences import UserPreferences, TripRequest
from src.models.trip import Trip


class OrchestratorAgent(BaseAgent):
    """Agent responsible for coordinating agent activities and compiling final output."""
    
    user_interface_agent: Optional[UserInterfaceAgent] = Field(
        default=None, description="Agent for user communication and preference capture"
    )
    destination_info_agent: Optional[DestinationInformationAgent] = Field(
        default=None, description="Agent for destination information research"
    )
    local_events_agent: Optional[LocalEventsAgent] = Field(
        default=None, description="Agent for local events and logistics information"
    )
    itinerary_agent: Optional[ItineraryOptimizationAgent] = Field(
        default=None, description="Agent for itinerary optimization"
    )
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(
        self,
        llm: BaseChatModel,
        user_interface_agent: Optional[UserInterfaceAgent] = None,
        destination_info_agent: Optional[DestinationInformationAgent] = None,
        local_events_agent: Optional[LocalEventsAgent] = None,
        itinerary_agent: Optional[ItineraryOptimizationAgent] = None,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[Any] = None,
        **data
    ):
        """Initialize the Orchestrator Agent.
        
        Args:
            llm: Language model to use for this agent.
            user_interface_agent: User Interface Agent instance.
            destination_info_agent: Destination Information Agent instance.
            local_events_agent: Local Events & Logistics Agent instance.
            itinerary_agent: Itinerary Optimization Agent instance.
            tools: Optional list of tools available to this agent.
            memory: Optional memory system for this agent.
            **data: Additional data for the agent.
        """
        super().__init__(
            name="Orchestrator Agent",
            description=(
                "an agent that coordinates the activities of all other agents and compiles the final output. "
                "You delegate specific tasks to specialized agents, manage workflow and information flow "
                "between agents, resolve conflicts between agent outputs, and ensure the final trip plan "
                "is comprehensive and coherent."
            ),
            llm=llm,
            tools=tools or [],
            memory=memory,
            **data
        )
        
        self.user_interface_agent = user_interface_agent
        self.destination_info_agent = destination_info_agent
        self.local_events_agent = local_events_agent
        self.itinerary_agent = itinerary_agent
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent.
        
        Returns:
            str: The system prompt for this agent.
        """
        base_prompt = super().get_system_prompt()
        additional_instructions = (
            "\n\nWhen orchestrating the trip planning process:"
            "\n1. Coordinate the activities of all specialized agents."
            "\n2. Delegate specific tasks to the appropriate agents based on their expertise."
            "\n3. Manage the flow of information between agents."
            "\n4. Resolve any conflicts or inconsistencies in agent outputs."
            "\n5. Compile the final trip plan, ensuring it is comprehensive and coherent."
            "\n6. Ensure the final output meets the user's preferences and requirements."
            "\n7. Provide a clear, well-structured final trip plan to the user."
        )
        return base_prompt + additional_instructions
    
    async def process_user_input(self, input_text: str, user_preferences: Optional[UserPreferences] = None) -> Dict[str, Any]:
        """Process user input and coordinate agent activities.
        
        Args:
            input_text: The input text from the user.
            user_preferences: Optional existing user preferences.
            
        Returns:
            Dict containing the response and any updated information.
        """
        # First, use the User Interface Agent to process the input and extract preferences
        ui_result = await self.user_interface_agent.process(input_text, user_preferences)
        response = ui_result["response"]
        updated_preferences = ui_result.get("preferences", user_preferences)
        
        # Check if this is a trip planning request
        if await self._is_trip_planning_request(input_text, response):
            # Extract trip request details
            trip_request = await self._extract_trip_request(input_text, updated_preferences)
            
            # Generate a complete trip plan
            trip_plan = await self.generate_trip_plan(trip_request)
            
            return {
                "response": response,
                "preferences": updated_preferences,
                "trip_request": trip_request,
                "trip_plan": trip_plan
            }
        
        # Otherwise, just return the response and updated preferences
        return {
            "response": response,
            "preferences": updated_preferences
        }
    
    async def generate_trip_plan(self, trip_request: TripRequest) -> Trip:
        """Generate a complete trip plan based on the trip request.
        
        Args:
            trip_request: The trip request details.
            
        Returns:
            Trip: The generated trip plan.
        """
        # Parse dates
        start_date = datetime.strptime(trip_request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(trip_request.end_date, "%Y-%m-%d")
        date_range = (start_date, end_date)
        
        # Step 1: Get destination information
        destination_result = await self.destination_info_agent.process(
            f"Research information about {trip_request.destination}",
            trip_request.preferences
        )
        destination_data = destination_result.get("destination_data", {})
        
        # Step 2: Get local events and logistics information
        logistics_result = await self.local_events_agent.process(
            f"Research events and logistics for {trip_request.destination} from {trip_request.start_date} to {trip_request.end_date}",
            trip_request.preferences,
            destination_data.get("location"),
            date_range
        )
        logistics_data = logistics_result.get("logistics_data", {})
        
        # Step 3: Generate optimized itinerary
        itinerary_result = await self.itinerary_agent.process(
            f"Create an optimized itinerary for {trip_request.destination} from {trip_request.start_date} to {trip_request.end_date}",
            trip_request.preferences,
            destination_data,
            logistics_data,
            date_range
        )
        trip_plan = itinerary_result.get("itinerary")
        
        # Step 4: Review and finalize the trip plan
        finalized_plan = await self._review_and_finalize(trip_plan, trip_request)
        
        return finalized_plan
    
    async def _is_trip_planning_request(self, input_text: str, response: str) -> bool:
        """Determine if the input is a trip planning request.
        
        Args:
            input_text: The input text from the user.
            response: The response from the User Interface Agent.
            
        Returns:
            bool: True if this is a trip planning request, False otherwise.
        """
        # Use the LLM to determine if this is a trip planning request
        system_message = (
            "Determine if the user's message is a trip planning request. "
            "A trip planning request typically includes or asks about destinations, dates, "
            "or travel itineraries. Return 'yes' if it's a trip planning request, 'no' otherwise."
        )
        
        human_message = f"User message: {input_text}\nAgent response: {response}"
        
        # Use direct messages instead of a template
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ]
        
        # Get response using direct messages
        detection_response = await self.llm.ainvoke(messages)
        
        return "yes" in detection_response.content.lower()
    
    async def _extract_trip_request(
        self, 
        input_text: str, 
        user_preferences: Optional[UserPreferences]
    ) -> TripRequest:
        """Extract trip request details from the input text.
        
        Args:
            input_text: The input text from the user.
            user_preferences: Optional user preferences.
            
        Returns:
            TripRequest: The extracted trip request details.
        """
        # Use the LLM to extract trip request details
        system_message = (
            "Extract trip request details from the user's message. "
            "Return a JSON object with the following fields: "
            "destination, start_date (YYYY-MM-DD), end_date (YYYY-MM-DD), "
            "travelers (number), additional_notes. "
            "If any field is not explicitly mentioned, make a reasonable assumption "
            "or use placeholder values."
        )
        
        # Use direct messages instead of a template
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=input_text)
        ]
        
        # Get response using direct messages
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
            
            trip_data = json.loads(json_str)
            
            # Create a TripRequest object
            preferences = user_preferences or UserPreferences()
            
            return TripRequest(
                destination=trip_data.get("destination", "Unknown"),
                start_date=trip_data.get("start_date", datetime.now().strftime("%Y-%m-%d")),
                end_date=trip_data.get("end_date", (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")),
                travelers=trip_data.get("travelers", 1),
                preferences=preferences,
                additional_notes=trip_data.get("additional_notes")
            )
        
        except (JSONDecodeError, KeyError, AttributeError) as e:
            # If parsing fails, create a minimal trip request
            print(f"Error extracting trip request: {e}")
            
            preferences = user_preferences or UserPreferences()
            
            return TripRequest(
                destination="Unknown",
                start_date=datetime.now().strftime("%Y-%m-%d"),
                end_date=(datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
                travelers=1,
                preferences=preferences
            )
    
    async def _review_and_finalize(self, trip_plan: Trip, trip_request: TripRequest) -> Trip:
        """Review and finalize the trip plan.
        
        Args:
            trip_plan: The generated trip plan.
            trip_request: The original trip request.
            
        Returns:
            Trip: The finalized trip plan.
        """
        # Use the LLM to review and suggest improvements to the trip plan
        system_message = (
            "Review the generated trip plan and suggest improvements. "
            "Consider the following aspects:\n"
            "1. Does the plan match the user's preferences and interests?\n"
            "2. Is the schedule realistic and balanced?\n"
            "3. Are there any conflicts or inconsistencies?\n"
            "4. Are there any missing elements that would enhance the trip?\n"
            "Return your assessment and specific suggestions for improvement."
        )
        
        human_message = (
            f"Trip Request: {trip_request.destination} from {trip_request.start_date} to {trip_request.end_date}\n"
            f"Travelers: {trip_request.travelers}\n"
            f"Preferences: {trip_request.preferences}\n"
            f"Additional Notes: {trip_request.additional_notes}\n\n"
            f"Trip Plan: {trip_plan}"
        )
        
        # Use direct messages instead of a template
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ]
        
        # Get response using direct messages
        review_response = await self.llm.ainvoke(messages)
        
        # For now, we'll just return the original trip plan
        # In a more advanced implementation, we could use the review to make improvements
        return trip_plan