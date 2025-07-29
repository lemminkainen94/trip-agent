"""
ReAct-based Trip Planning Agent that plans all days at once.

This module provides an implementation of the trip planning agent using
the ReAct (Reasoning and Acting) pattern from LangGraph. The agent creates
a complete trip plan for all days at once and verifies constraints like
opening hours through an iterative process.
"""

import json
import os
import re
from datetime import datetime, timedelta, time
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.callbacks.base import Callbacks
from langgraph.prebuilt import create_react_agent
from langsmith.run_helpers import traceable
from pydantic import Field

from src.agents.base import BaseAgent
from src.models.trip import (
    Attraction, 
    Location, 
    Activity, 
    DayPlan, 
    Trip, 
    TravelLeg,
    AttractionRanking,
    CategoryRankings,
    TripPlanningState
)
from .tools import TripPlanningTools, is_attraction_open_at_time, is_attraction_available_on_date

# Load environment variables
load_dotenv()

# Configure LangSmith tracing if API key is set
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")


class TripPlanningReactAgent(BaseAgent):
    """
    ReAct-based Trip Planning Agent that plans all days at once.
    
    This agent uses the ReAct (Reasoning and Acting) pattern from LangGraph
    to create a complete trip plan for all days at once. It uses tools to
    validate constraints like opening hours and date ranges.
    """
    
    state: Optional[TripPlanningState] = Field(None, description="Current state of the trip planning process")
    react_agent: Any = Field(None, description="ReAct agent for planning the trip")
    
    def __init__(self, llm: BaseChatModel):
        """
        Initialize the agent with a language model.
        
        Args:
            llm: Language model to use for planning
        """
        super().__init__(
            name="Trip Planning ReAct Agent",
            description="An agent that creates detailed trip plans using the ReAct pattern",
            llm=llm
        )
        self.state = None
    
    @traceable(run_type="chain", name="TripPlanningReactAgent.process")
    async def process(
        self,
        destination_name: str,
        attractions: List[Attraction],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        preferences: Dict[str, Any],
        excluded_categories: List[str] = None,
        destination_report: Optional[str] = None,
        callbacks: Callbacks = None
    ) -> Trip:
        """
        Process a trip planning request.
        
        Args:
            destination_name: Name of the destination
            attractions: List of attractions at the destination
            start_date: Start date of the trip (string in YYYY-MM-DD format or datetime)
            end_date: End date of the trip (string in YYYY-MM-DD format or datetime)
            preferences: User preferences for the trip
            excluded_categories: Categories to exclude from the trip
            destination_report: Optional destination report with additional information
            callbacks: Optional callbacks for LangSmith tracing
            
        Returns:
            A Trip object with the planned itinerary
        """
        # Initialize state
        state = TripPlanningState()
        state["destination_name"] = destination_name
        state["attractions"] = attractions
        
        # Store dates as strings for consistent handling
        if isinstance(start_date, datetime):
            state["start_date"] = start_date.strftime("%Y-%m-%d")
        else:
            state["start_date"] = start_date
            
        if isinstance(end_date, datetime):
            state["end_date"] = end_date.strftime("%Y-%m-%d")
        else:
            state["end_date"] = end_date
            
        state["preferences"] = preferences
        state["excluded_categories"] = excluded_categories or []
        state["destination_report"] = destination_report
        
        # Rank attractions
        await self.rank_attractions(state, callbacks=callbacks)
        
        # Plan the trip
        await self.plan_trip(state, callbacks=callbacks)
        
        # Create the trip
        await self.create_trip(state, callbacks=callbacks)
        
        # Return the trip
        return state["trip"]
    
    @traceable(run_type="chain", name="TripPlanningReactAgent.rank_attractions")
    async def rank_attractions(self, state: TripPlanningState, callbacks: Callbacks = None) -> Dict[str, Any]:
        """
        Rank attractions by category based on user preferences.
        
        Args:
            state: Current trip planning state
            callbacks: Optional callbacks for LangSmith tracing
            
        Returns:
            Updated state with ranked attractions
        """
        # Extract state variables
        destination_name = state["destination_name"]
        attractions = state["attractions"]
        preferences = state["preferences"]
        excluded_categories = state["excluded_categories"]
        
        # Group attractions by category
        categories = {}
        for attraction in attractions:
            if attraction.category not in excluded_categories:
                if attraction.category not in categories:
                    categories[attraction.category] = []
                categories[attraction.category].append(attraction)
        
        # Create a prompt for the LLM to rank attractions
        prompt = f"""
        You are an expert travel planner for {destination_name}.
        
        I need you to rank the attractions in each category based on the user's preferences.
        
        Here are the user preferences:
        """
        
        # Convert preferences to a serializable format
        serializable_preferences = {}
        for key, value in preferences.items():
            if isinstance(value, datetime):
                serializable_preferences[key] = value.strftime('%Y-%m-%d')
            elif isinstance(value, dict):
                serializable_dict = {}
                for k, v in value.items():
                    if isinstance(v, datetime):
                        serializable_dict[k] = v.strftime('%Y-%m-%d')
                    else:
                        serializable_dict[k] = v
                serializable_preferences[key] = serializable_dict
            else:
                serializable_preferences[key] = value
        
        preferences_text = json.dumps(serializable_preferences, indent=2)
        prompt += preferences_text
        
        # Format attractions by category for the prompt
        attractions_by_category = ""
        for category, category_attractions in categories.items():
            attractions_by_category += f"\n\n{category}:\n"
            for attraction in category_attractions:
                opening_hours_text = ""
                if attraction.opening_hours:
                    opening_hours_text = f"Opening Hours: {json.dumps(attraction.opening_hours)}"
                
                date_range_text = ""
                if attraction.date_range:
                    date_range_text = f"Available Dates: {attraction.date_range}"
                
                attractions_by_category += f"- {attraction.name}: {attraction.description}\n  {opening_hours_text}\n  {date_range_text}\n"
        
        # Create the human message
        human_message_content = f"""
        Please rank the attractions within each category based on the following user preferences:
        
        {preferences_text}
        
        Destination Information:
        {state.get("destination_report", "No additional information provided.")}
        
        Here are the attractions by category:
        {attractions_by_category}
        
        Please return your rankings in the following JSON format:
        ```json
        {{
            "rankings": [
                {{
                    "category": "Category Name",
                    "attractions": [
                        {{
                            "name": "Attraction Name",
                            "score": 9.5,
                            "reasoning": "Brief explanation of why this attraction is ranked highly"
                        }},
                        {{
                            "name": "Another Attraction",
                            "score": 8.7,
                            "reasoning": "Brief explanation of why this attraction is ranked second"
                        }}
                    ]
                }},
                {{
                    "category": "Another Category",
                    "attractions": [
                        // attractions for this category
                    ]
                }}
            ]
        }}
        ```
        
        Make sure to include ALL attractions from each category in your rankings.
        """
        
        # Call the LLM
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=human_message_content)
        ]
        
        response = await self.llm.ainvoke(messages)
        response_text = response.content
        
        # Parse the JSON response
        try:
            # Extract JSON from the response
            match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                json_str = response_text
            
            # Remove any non-JSON text before or after
            json_str = re.sub(r'^[^{]*', '', json_str)
            json_str = re.sub(r'[^}]*$', '', json_str)
            
            rankings_data = json.loads(json_str)
            
            # Convert to CategoryRankings objects
            ranked_categories = []
            
            for category_data in rankings_data.get("rankings", []):
                category_name = category_data.get("category")
                attraction_rankings = []
                
                for attraction_data in category_data.get("attractions", []):
                    attraction_name = attraction_data.get("name")
                    score = attraction_data.get("score", attraction_data.get("rank", 5.0))  # Handle both formats
                    reasoning = attraction_data.get("reasoning", attraction_data.get("reason", ""))  # Handle both formats
                    
                    # Find the attraction object
                    attraction = None
                    for a in attractions:
                        if a.name == attraction_name and a.category == category_name:
                            attraction = a
                            break
                    
                    if attraction:
                        attraction_rankings.append(
                            AttractionRanking(
                                attraction=attraction,
                                score=float(score),  # Convert to float for score
                                reasoning=reasoning
                            )
                        )
                
                # Sort by score (higher is better)
                attraction_rankings.sort(key=lambda x: x.score, reverse=True)
                
                ranked_categories.append(
                    CategoryRankings(
                        category=category_name,
                        attractions=attraction_rankings
                    )
                )
            
            # Update state
            state["ranked_categories"] = ranked_categories
            
            return {"ranked_categories": ranked_categories}
        
        except Exception as e:
            print(f"Error parsing rankings response: {e}")
            print(f"Response was: {response_text}")
            
            # Fallback: create simple rankings
            ranked_categories = []
            
            for category, category_attractions in categories.items():
                attraction_rankings = []
                
                for i, attraction in enumerate(category_attractions):
                    attraction_rankings.append(
                        AttractionRanking(
                            attraction=attraction,
                            score=float(10 - i),  # Higher score for first attractions
                            reasoning=f"Fallback ranking for {attraction.name}"
                        )
                    )
                
                # Sort by score (higher is better)
                attraction_rankings.sort(key=lambda x: x.score, reverse=True)
                
                ranked_categories.append(
                    CategoryRankings(
                        category=category,
                        attractions=attraction_rankings
                    )
                )
            
            # Update state
            state["ranked_categories"] = ranked_categories
            
            return {"ranked_categories": ranked_categories}
    
    def _format_ranked_attractions(self, ranked_categories: List[CategoryRankings]) -> str:
        """
        Format ranked attractions for the prompt.
        
        Args:
            ranked_categories: List of CategoryRankings objects
            
        Returns:
            Formatted text of ranked attractions
        """
        result = ""
        
        for category in ranked_categories:
            result += f"\n\n{category.category}:\n"
            
            for ranking in category.attractions:
                attraction = ranking.attraction
                
                # Format opening hours
                opening_hours_text = ""
                if attraction.opening_hours:
                    opening_hours_text = f"Opening Hours: {json.dumps(attraction.opening_hours)}"
                
                # Format date range
                date_range_text = ""
                if attraction.date_range:
                    date_range_text = f"Available Dates: {attraction.date_range}"
                
                # Format visit duration
                visit_duration = attraction.visit_duration
                
                result += f"{ranking.score:.1f}. {attraction.name}: {attraction.description}\n"
                result += f"   Visit Duration: {visit_duration}\n"
                result += f"   {opening_hours_text}\n" if opening_hours_text else ""
                result += f"   {date_range_text}\n" if date_range_text else ""
        
        return result
    
    # Helper methods moved to tools.py
    
    @traceable(run_type="chain", name="TripPlanningReactAgent.plan_trip")
    async def plan_trip(self, state: TripPlanningState, callbacks: Callbacks = None) -> Dict[str, Any]:
        """
        Plan the trip using a ReAct agent that validates and refines the plan.
        
        Args:
            state: Current trip planning state
            callbacks: Optional callbacks for LangSmith tracing
            
        Returns:
            Updated state with planned trip
        """
        # Store the state for tool access
        self.state = state
        
        # Extract state variables
        destination_name = state["destination_name"]
        start_date = state["start_date"]
        end_date = state["end_date"]
        preferences = state["preferences"]
        ranked_categories = state["ranked_categories"]
        destination_report = state.get("destination_report", "")
        
        # Format the ranked attractions for the prompt
        ranked_attractions_text = self._format_ranked_attractions(ranked_categories)
        
        # Create a list of all attraction names for validation
        available_attractions_list = "Available Attractions (use EXACT names from this list):\n"
        for category in ranked_categories:
            for ranking in category.attractions:
                available_attractions_list += f"- \"{ranking.attraction.name}\"\n"
        
        # Create the system prompt for the ReAct agent
        system_prompt = f"""
        You are an expert travel planner for {destination_name}.
        
        Your task is to create a detailed day-by-day itinerary for a trip from {start_date} to {end_date}.
        
        Use the ranked attractions provided to create a balanced and realistic schedule.
        
        PLANNING GUIDELINES:
        - ALWAYS USE PROVIDED VISIT DURATION for each attraction!
        - Schedule 3-5 activities per day, with appropriate breaks for meals and rest.
        - Group attractions by location to minimize travel time.
        - Schedule museums and galleries earlier in the day when they're open.
        - Include at least one meal (lunch or dinner) at a local restaurant each day.
        - Ensure that attractions are open at the scheduled times.
        - Only include attractions on days they are available (check date ranges for seasonal events).
        - Distribute highly-ranked attractions evenly across the trip.
        - Avoid scheduling activities too close together - allow for travel time between locations.
        - Include a mix of categories each day (don't put all museums on one day).
        - CRITICAL: Do not leave gaps longer than 1 hours in the schedule during daytime (9:00-19:00).
        - If there would be a gap longer than 1 hours, add another attraction, activity, or suggest free time for shopping, walking, or relaxing in a specific location.
        
        IMPORTANT: Before finalizing your plan, validate it to ensure:
        1. All attractions are open at the scheduled times.
        2. All attractions are available on the scheduled dates.
        3. The schedule is realistic and balanced.
        4. Attractions are not repeated.
        5. There are no large gaps (>1 hours) in the daytime schedule (9:00-19:00).
        
        User Preferences:
        {json.dumps(preferences, indent=2)}
        
        Destination Information:
        {destination_report if destination_report else "No additional information provided."}
        
        {available_attractions_list}
        
        Here are the ranked attractions to use in your plan:
        {ranked_attractions_text}
        
        Please create a detailed itinerary in the following JSON format:
        ```json
        {{
            "days": [
                {{
                    "date": "YYYY-MM-DD",
                    "activities": [
                        {{
                            "start_time": "09:00",
                            "end_time": "10:30",
                            "attraction_name": "Name of attraction",
                            "description": "Brief description of the activity"
                        }},
                        {{
                            "start_time": "11:00",
                            "end_time": "12:30",
                            "attraction_name": "Another attraction",
                            "description": "Another description"
                        }}
                    ]
                }},
                {{
                    "date": "YYYY-MM-DD",
                    "activities": [
                        // activities for next day
                    ]
                }}
            ]
        }}
        ```
        
        The times should be in 24-hour format (HH:MM).
        
        Make sure to validate your plan against opening hours and date availability before finalizing it.
        Don't suggest people-watching! That's creepy!
        """
        
        # Create tools for the ReAct agent
        tools_instance = TripPlanningTools(state)
        
        # Define the tool functions
        tool_functions = [
            tools_instance.check_opening_hours,
            tools_instance.validate_itinerary,
            tools_instance.check_attraction_availability,
            tools_instance.check_schedule_gaps,  # Add the new schedule gap checking tool
        ]
        
        # Create the ReAct agent using the create_react_agent function
        self.react_agent = create_react_agent(
            model=self.llm,
            tools=tool_functions,
            prompt=system_prompt
        )
        
        # Create the human message content
        human_message_content = f"""Please create a detailed itinerary for a trip to {destination_name} from {start_date} to {end_date}.

        Here are the user preferences:
        {json.dumps(preferences, indent=2)}

        Here are the top-ranked attractions in each category:
        {ranked_attractions_text}

        {available_attractions_list}

        IMPORTANT REQUIREMENTS:
        1. Create a day-by-day plan with 3-5 activities per day.
        2. Include specific start and end times for each activity.
        3. Ensure activities are open at the scheduled times.
        4. Include at least one meal (lunch or dinner) each day.
        5. Do not leave large gaps (>1 hours) in the daytime schedule (9:00-19:00).
        6. If there would be a gap, add an activity or suggest free time in a specific location.
        7. Return your plan in the following JSON format:

        ```json
        {{
          "days": [
            {{
              "date": "YYYY-MM-DD",
              "activities": [
                {{
                  "start_time": "HH:MM",
                  "end_time": "HH:MM",
                  "attraction_name": "Exact name from the provided list",
                  "description": "Brief description of the activity"
                }},
                ...
              ]
            }},
            ...
          ]
        }}
        ```

        Use the check_opening_hours, validate_itinerary, check_attraction_availability, and check_schedule_gaps tools to validate your plan before finalizing it.
        """
        
        # Run the ReAct agent
        result = await self.react_agent.ainvoke({"messages": [HumanMessage(content=human_message_content)]})
        
        # Extract the final plan from the result
        final_response = result["messages"][-1].content
        
        # Parse the JSON response
        try:
            # Extract JSON from the response
            match = re.search(r'```json\s*(.*?)\s*```', final_response, re.DOTALL)
            if match:
                final_response = match.group(1)
            
            # Remove any non-JSON text before or after
            final_response = re.sub(r'^[^{]*', '', final_response)
            final_response = re.sub(r'[^}]*$', '', final_response)
            
            plan = json.loads(final_response)
            
            # Update state
            state["plan"] = plan
            
            return {"plan": plan}
        
        except Exception as e:
            print(f"Error parsing plan response: {e}")
            print(f"Response was: {final_response}")
            
            # Fallback: create a simple plan
            plan = self._create_fallback_plan(state)
            
            # Update state
            state["plan"] = plan
            
            return {"plan": plan}
    
    @traceable(run_type="chain", name="TripPlanningReactAgent._create_fallback_plan")
    async def _create_fallback_plan(self, state: TripPlanningState, callbacks: Callbacks = None) -> Dict[str, Any]:
        """
        Create a fallback plan if the LLM response cannot be parsed.
        
        Args:
            state: Current trip planning state
            callbacks: Optional callbacks for LangSmith tracing
            
        Returns:
            A simple plan with one activity per day
        """
        print("Creating fallback plan...")
        
        # Extract state variables
        start_date = state["start_date"]
        end_date = state["end_date"]
        ranked_categories = state["ranked_categories"]
        
        # Create a list of all attractions, sorted by score
        all_attractions = []
        for category in ranked_categories:
            for ranking in category.attractions:
                all_attractions.append((ranking.attraction, ranking.score))
        
        # Sort by score (higher is better)
        all_attractions.sort(key=lambda x: x[1], reverse=True)
        attractions = [a[0] for a in all_attractions]
        
        # Create a day for each date in the range
        days = []
        
        # Convert string dates to datetime for date calculations
        if isinstance(start_date, str):
            start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start_date_dt = start_date
            
        if isinstance(end_date, str):
            end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end_date_dt = end_date
        
        current_date = start_date_dt
        attraction_index = 0
        
        while current_date <= end_date_dt:
            date_str = current_date.strftime("%Y-%m-%d")
            activities = []
            
            # Morning activity (9:00 - 11:00)
            if attraction_index < len(attractions):
                attraction = attractions[attraction_index]
                
                # Check if the attraction is available on this date
                if is_attraction_available_on_date(attraction, current_date):
                    # Check if the attraction is open at this time
                    if is_attraction_open_at_time(attraction, current_date, "09:00"):
                        activities.append({
                            "start_time": "09:00",
                            "end_time": "11:00",
                            "attraction_name": attraction.name,
                            "description": f"Visit {attraction.name}. {attraction.description}"
                        })
                    else:
                        # Add with a warning
                        activities.append({
                            "start_time": "09:00",
                            "end_time": "11:00",
                            "attraction_name": attraction.name,
                            "description": f"Visit {attraction.name}. {attraction.description} (WARNING: This attraction may not be open at this time)"
                        })
                
                attraction_index += 1
            
            # Lunch (12:00 - 13:30)
            if attraction_index < len(attractions):
                attraction = attractions[attraction_index]
                
                # Look for a food-related attraction
                food_attraction = None
                for i in range(attraction_index, min(attraction_index + 10, len(attractions))):
                    if attractions[i].category.lower() in ["restaurants", "cafes", "food"]:
                        food_attraction = attractions[i]
                        attraction_index = i + 1
                        break
                
                if food_attraction:
                    attraction = food_attraction
                
                # Check if the attraction is available on this date
                if is_attraction_available_on_date(attraction, current_date):
                    # Check if the attraction is open at this time
                    if is_attraction_open_at_time(attraction, current_date, "12:00"):
                        activities.append({
                            "start_time": "12:00",
                            "end_time": "13:30",
                            "attraction_name": attraction.name,
                            "description": f"Lunch at {attraction.name}. {attraction.description}"
                        })
                    else:
                        # Add with a warning
                        activities.append({
                            "start_time": "12:00",
                            "end_time": "13:30",
                            "attraction_name": attraction.name,
                            "description": f"Lunch at {attraction.name}. {attraction.description} (WARNING: This attraction may not be open at this time)"
                        })
                
                attraction_index += 1
            
            # Afternoon activity (14:00 - 16:00)
            if attraction_index < len(attractions):
                attraction = attractions[attraction_index]
                
                # Check if the attraction is available on this date
                if is_attraction_available_on_date(attraction, current_date):
                    # Check if the attraction is open at this time
                    if is_attraction_open_at_time(attraction, current_date, "14:00"):
                        activities.append({
                            "start_time": "14:00",
                            "end_time": "16:00",
                            "attraction_name": attraction.name,
                            "description": f"Visit {attraction.name}. {attraction.description}"
                        })
                    else:
                        # Add with a warning
                        activities.append({
                            "start_time": "14:00",
                            "end_time": "16:00",
                            "attraction_name": attraction.name,
                            "description": f"Visit {attraction.name}. {attraction.description} (WARNING: This attraction may not be open at this time)"
                        })
                
                attraction_index += 1
            
            # Add the day to the plan
            days.append({
                "date": date_str,
                "activities": activities
            })
            
            # Move to the next day
            current_date += timedelta(days=1)
        
        return {"days": days}
    
    @traceable(run_type="chain", name="TripPlanningReactAgent.create_trip")
    async def create_trip(self, state: TripPlanningState, callbacks: Callbacks = None) -> Dict[str, Any]:
        """
        Create the final Trip object from the plan.
        
        Args:
            state: Current trip planning state
            callbacks: Optional callbacks for LangSmith tracing
            
        Returns:
            Updated state with the created Trip object
        """
        plan = state["plan"]
        
        # Create a DayPlan for each day in the plan
        day_plans = []
        
        for day_data in plan["days"]:
            date_str = day_data["date"]
            date = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Create activities for this day
            activities = []
            
            for activity_data in day_data["activities"]:
                # Parse start and end times
                start_time_str = activity_data["start_time"]
                end_time_str = activity_data["end_time"]
                
                # Create datetime objects for start and end times
                start_hour, start_minute = map(int, start_time_str.split(":"))
                end_hour, end_minute = map(int, end_time_str.split(":"))
                
                start_datetime = datetime.combine(date, time(start_hour, start_minute))
                end_datetime = datetime.combine(date, time(end_hour, end_minute))
                
                # Find the attraction by name
                attraction_name = activity_data["attraction_name"]
                attraction = None
                
                for a in state["attractions"]:
                    if a.name == attraction_name:
                        attraction = a
                        break
                
                # Create the activity
                activity = Activity(
                    start_time=start_datetime,
                    end_time=end_datetime,
                    attraction=attraction,  # May be None if attraction not found
                    description=activity_data["description"]
                )
                
                activities.append(activity)
            
            # Create the day plan
            day_plan = DayPlan(
                date=date,
                activities=activities
            )
            
            day_plans.append(day_plan)
        
        # Convert string dates to datetime objects for the Trip model
        start_date = state["start_date"]
        end_date = state["end_date"]
        
        # Convert to datetime if they are strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Create the trip
        trip = Trip(
            title=f"Trip to {state['destination_name']}",
            destination=Location(name=state["destination_name"]),
            start_date=start_date,
            end_date=end_date,
            days=day_plans
        )
        
        # Update state
        state["trip"] = trip
        
        return {"trip": trip}
