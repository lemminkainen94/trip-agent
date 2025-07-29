"""Itinerary Optimization Agent implementation for the Trip Agent system."""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from pydantic import Field

from src.agents.base import BaseAgent
from src.models.preferences import UserPreferences
from src.models.trip import Activity, Attraction, DayPlan, Location, Trip, TravelLeg


class ItineraryOptimizationAgent(BaseAgent):
    """Agent responsible for building optimized travel schedules."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[Any] = None,
        **data
    ):
        """Initialize the Itinerary Optimization Agent.
        
        Args:
            llm: Language model to use for this agent.
            tools: Optional list of tools available to this agent.
            memory: Optional memory system for this agent.
            **data: Additional data for the agent.
        """
        super().__init__(
            name="Itinerary Optimization Agent",
            description=(
                "an agent that builds optimized travel schedules. "
                "You calculate travel times between locations, optimize visit order for efficiency, "
                "balance activities throughout the day, account for operating hours and peak times, "
                "and create realistic daily schedules."
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
            "\n\nWhen optimizing itineraries:"
            "\n1. Calculate realistic travel times between locations."
            "\n2. Optimize the visit order to minimize travel time and maximize efficiency."
            "\n3. Balance activities throughout the day to avoid exhaustion."
            "\n4. Account for opening hours, peak times, and seasonal factors."
            "\n5. Include buffer time for unexpected delays, rest, and meals."
            "\n6. Consider the user's preferences, interests, and activity level."
            "\n7. Create a day-by-day schedule with specific times for each activity."
            "\n8. Ensure the itinerary is realistic and achievable."
        )
        return base_prompt + additional_instructions
    
    async def process(
        self, 
        input_text: str, 
        user_preferences: Optional[UserPreferences] = None,
        destination_data: Optional[Dict[str, Any]] = None,
        logistics_data: Optional[Dict[str, Any]] = None,
        date_range: Optional[tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """Process input and generate an optimized itinerary.
        
        Args:
            input_text: The input text containing the itinerary request.
            user_preferences: Optional user preferences to consider.
            destination_data: Optional destination data to use for planning.
            logistics_data: Optional logistics data to use for planning.
            date_range: Optional date range (start_date, end_date) to plan for.
            
        Returns:
            Dict containing the agent's response and structured itinerary.
        """
        # Create a prompt with all available information
        prompt = self.create_prompt()
        
        # Build context with all available information
        context_parts = [f"Itinerary request: {input_text}"]
        
        if user_preferences:
            preferences_text = str(user_preferences)
            context_parts.append(f"User preferences: {preferences_text}")
        
        if destination_data:
            location = destination_data.get("location")
            attractions = destination_data.get("attractions", [])
            
            if location:
                context_parts.append(f"Destination: {location.name}")
            
            if attractions:
                attractions_text = "\n".join([
                    f"- {attraction.name} ({attraction.category}): {attraction.description} "
                    f"(Visit duration: {attraction.visit_duration} minutes)"
                    for attraction in attractions
                ])
                context_parts.append(f"Attractions:\n{attractions_text}")
        
        if logistics_data:
            events = logistics_data.get("events", [])
            opening_hours = logistics_data.get("opening_hours", {})
            seasonal_notes = logistics_data.get("seasonal_notes", [])
            travel_times = logistics_data.get("travel_times", {})
            
            if events:
                events_text = "\n".join([
                    f"- {event['name']} on {event['date']} from {event['start_time']} to {event['end_time']}: "
                    f"{event['description']}"
                    for event in events
                ])
                context_parts.append(f"Events:\n{events_text}")
            
            if opening_hours:
                hours_text = "\n".join([
                    f"- {attraction}: {', '.join([f'{day}: {hours}' for day, hours in hours.items()])}"
                    for attraction, hours in opening_hours.items()
                ])
                context_parts.append(f"Opening Hours:\n{hours_text}")
            
            if seasonal_notes:
                notes_text = "\n".join([f"- {note}" for note in seasonal_notes])
                context_parts.append(f"Seasonal Notes:\n{notes_text}")
            
            if travel_times:
                times_text = "\n".join([
                    f"- {locations}: {time} minutes"
                    for locations, time in travel_times.items()
                ])
                context_parts.append(f"Travel Times:\n{times_text}")
        
        if date_range:
            start_date, end_date = date_range
            context_parts.append(
                f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Generate response using the chain approach
        chain = prompt | self.llm
        response = await chain.ainvoke({"input": context})
        
        # Create structured itinerary
        itinerary = await self._create_itinerary(
            input_text, response.content, destination_data, logistics_data, date_range
        )
        
        return {
            "response": response.content,
            "itinerary": itinerary
        }
    
    async def _create_itinerary(
        self, 
        request_text: str, 
        planning_text: str,
        destination_data: Optional[Dict[str, Any]] = None,
        logistics_data: Optional[Dict[str, Any]] = None,
        date_range: Optional[tuple[datetime, datetime]] = None
    ) -> Trip:
        """Create a structured itinerary from planning text.
        
        Args:
            request_text: The original request text.
            planning_text: The planning text containing the itinerary.
            destination_data: Optional destination data to use for planning.
            logistics_data: Optional logistics data to use for planning.
            date_range: Optional date range (start_date, end_date) to plan for.
            
        Returns:
            Trip: Structured itinerary.
        """
        # Use the LLM to extract structured data from the planning text
        system_message = (
            "Extract a structured itinerary from the planning text. "
            "Return a JSON object with the following structure:\n"
            "{\n"
            "  \"title\": \"trip title\",\n"
            "  \"destination\": {\n"
            "    \"name\": \"destination name\",\n"
            "    \"latitude\": latitude as float,\n"
            "    \"longitude\": longitude as float\n"
            "  },\n"
            "  \"days\": [\n"
            "    {\n"
            "      \"date\": \"YYYY-MM-DD\",\n"
            "      \"activities\": [\n"
            "        {\n"
            "          \"start_time\": \"HH:MM\",\n"
            "          \"end_time\": \"HH:MM\",\n"
            "          \"description\": \"detailed description\",\n"
            "          \"attraction\": {\n"
            "            \"name\": \"attraction name\",\n"
            "            \"category\": \"category\",\n"
            "            \"visit_duration\": duration in minutes\n"
            "          },\n"
            "          \"travel\": {\n"
            "            \"origin\": \"origin name\",\n"
            "            \"destination\": \"destination name\",\n"
            "            \"mode\": \"transportation mode\",\n"
            "            \"duration\": duration in minutes,\n"
            "            \"distance\": distance in kilometers\n"
            "          }\n"
            "        },\n"
            "        ...\n"
            "      ]\n"
            "    },\n"
            "    ...\n"
            "  ]\n"
            "}\n"
            "For each activity, include either 'attraction' or 'travel' or neither, but not both. "
            "If the activity is visiting an attraction, include 'attraction'. "
            "If the activity is traveling between locations, include 'travel'. "
            "If the activity is something else (like dining, rest, etc.), include neither."
        )
        
        human_message = (
            f"Request: {request_text}\n\n"
            f"Date Range: {date_range[0].strftime('%Y-%m-%d') if date_range else 'Unknown'} to "
            f"{date_range[1].strftime('%Y-%m-%d') if date_range else 'Unknown'}\n\n"
            f"Planning: {planning_text}"
        )
        
        # Use direct messages instead of a template
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
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
            
            itinerary_data = json.loads(json_str)
            
            # Create a Location object for the destination
            destination = Location(
                name=itinerary_data["destination"]["name"],
                latitude=itinerary_data["destination"]["latitude"],
                longitude=itinerary_data["destination"]["longitude"]
            )
            
            # Create DayPlan objects for each day
            days = []
            for day_data in itinerary_data["days"]:
                date = datetime.strptime(day_data["date"], "%Y-%m-%d")
                
                # Create Activity objects for each activity
                activities = []
                for activity_data in day_data["activities"]:
                    start_time = datetime.strptime(
                        f"{day_data['date']} {activity_data['start_time']}", 
                        "%Y-%m-%d %H:%M"
                    )
                    end_time = datetime.strptime(
                        f"{day_data['date']} {activity_data['end_time']}", 
                        "%Y-%m-%d %H:%M"
                    )
                    
                    # Create an Attraction object if this activity is visiting an attraction
                    attraction = None
                    if "attraction" in activity_data and activity_data["attraction"]:
                        attraction_data = activity_data["attraction"]
                        attraction = Attraction(
                            name=attraction_data["name"],
                            description=activity_data["description"],
                            location=destination,  # Use the main destination for now
                            category=attraction_data["category"],
                            visit_duration=attraction_data["visit_duration"]
                        )
                    
                    # Create a TravelLeg object if this activity is traveling
                    travel = None
                    if "travel" in activity_data and activity_data["travel"]:
                        travel_data = activity_data["travel"]
                        travel = TravelLeg(
                            origin=Location(
                                name=travel_data["origin"],
                                latitude=destination.latitude,  # Use destination coordinates for now
                                longitude=destination.longitude
                            ),
                            destination=Location(
                                name=travel_data["destination"],
                                latitude=destination.latitude,  # Use destination coordinates for now
                                longitude=destination.longitude
                            ),
                            mode=travel_data["mode"],
                            duration=travel_data["duration"],
                            distance=travel_data["distance"]
                        )
                    
                    # Create the Activity object
                    activity = Activity(
                        start_time=start_time,
                        end_time=end_time,
                        attraction=attraction,
                        travel=travel,
                        description=activity_data["description"]
                    )
                    activities.append(activity)
                
                # Create the DayPlan object
                day_plan = DayPlan(
                    date=date,
                    activities=activities
                )
                days.append(day_plan)
            
            # Create the Trip object
            start_date = date_range[0] if date_range else days[0].date if days else datetime.now()
            end_date = date_range[1] if date_range else days[-1].date if days else (datetime.now() + timedelta(days=7))
            
            trip = Trip(
                title=itinerary_data["title"],
                destination=destination,
                start_date=start_date,
                end_date=end_date,
                days=days
            )
            
            return trip
        
        except (JSONDecodeError, KeyError, AttributeError, ValueError) as e:
            # If parsing fails, log the error and return a minimal trip
            print(f"Error creating itinerary: {e}")
            
            # Create a minimal trip with the available information
            destination_name = "Unknown Destination"
            if destination_data and "location" in destination_data:
                destination_name = destination_data["location"].name
            
            destination = Location(
                name=destination_name,
                latitude=0.0,
                longitude=0.0
            )
            
            start_date = date_range[0] if date_range else datetime.now()
            end_date = date_range[1] if date_range else (datetime.now() + timedelta(days=7))
            
            return Trip(
                title=f"Trip to {destination_name}",
                destination=destination,
                start_date=start_date,
                end_date=end_date,
                days=[]
            )
