"""
Alternative Trip Planning Agent that plans all days at once.

This module provides an alternative implementation of the trip planning agent
that creates a complete trip plan for all days at once, rather than planning
each day separately. This approach may produce more balanced plans across
multiple days.
"""

import os
from pydoc import describe
import re
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder

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

class TripPlanningAltAgent(BaseAgent):
    """
    Alternative Trip Planning Agent that plans all days at once.
    
    This agent creates a complete trip plan for all days at once, rather than
    planning each day separately. This approach may produce more balanced plans
    across multiple days.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[Any] = None,
        **data
    ):
        """
        Initialize the TripPlanningAltAgent.
        
        Args:
            llm: Language model to use for planning
            tools: Optional list of tools to use
                memory: Optional memory system for this agent.
            **data: Additional data for the agent.
        """
        super().__init__(
            name="Trip Planning Agent",
            description=(
                "an agent that creates optimized trip itineraries based on destination "
                "reports and attraction data. You rank attractions by category and create "
                "detailed daily schedules considering logistical constraints and user preferences."
            ),
            llm=llm,
            tools=tools or [],
            memory=memory,
            **data
        )
    
    def _build_graph(self):
        """Build the workflow graph for trip planning."""
        return {
            "rank_attractions": {
                "func": self.rank_attractions,
                "next": "plan_trip"
            },
            "plan_trip": {
                "func": self.plan_trip,
                "next": "create_trip"
            },
            "create_trip": {
                "func": self.create_trip,
                "next": None
            }
        }
    
    async def process(
        self,
        destination_name: str,
        attractions: List[Attraction],
        start_date: datetime,
        end_date: datetime,
        preferences: Optional[Dict[str, Any]] = None,
        excluded_categories: Optional[List[str]] = None,
        destination_report: Optional[str] = None
    ) -> Trip:
        """
        Process the trip planning request.
        
        Args:
            destination_name: Name of the destination
            attractions: List of attractions to consider
            start_date: Start date of the trip
            end_date: End date of the trip
            preferences: Optional user preferences
            excluded_categories: Optional categories to exclude
            destination_report: Optional destination report with additional context
            
        Returns:
            A Trip object with the planned itinerary
        """
        # Initialize state
        state = {
            "destination_name": destination_name,
            "attractions": attractions,
            "start_date": start_date,
            "end_date": end_date,
            "preferences": preferences or {},
            "excluded_categories": excluded_categories or [],
            "destination_report": destination_report or "",
            "used_attractions": set(),
            "ranked_categories": [],
            "day_plans": []
        }
        
        # Execute the workflow steps sequentially
        # Step 1: Rank attractions
        state = await self.rank_attractions(state)
        
        # Step 2: Plan the trip
        state = await self.plan_trip(state)
        
        # Step 3: Create the trip object
        state = await self.create_trip(state)
        
        # Return the trip
        return state["trip"]
    
    async def rank_attractions(self, state: TripPlanningState) -> Dict[str, Any]:
        """
        Rank attractions by category based on user preferences.
        
        Args:
            state: Current state of the trip planning process
            
        Returns:
            Updated state with ranked attractions
        """
        # Group attractions by category
        attractions_by_category = {}
        for attraction in state["attractions"]:
            if attraction.category in state.get("excluded_categories", []):
                continue
                
            if attraction.category not in attractions_by_category:
                attractions_by_category[attraction.category] = []
                
            attractions_by_category[attraction.category].append(attraction)
        
        # Create a prompt for the LLM to rank attractions
        prompt = SystemMessage(content=f"""
        You are a travel expert specializing in {state["destination_name"]}. Your task is to rank attractions 
        within each category from most to least attractive based on the destination report and user preferences.
        
        For each attraction, provide:
        1. A score from 0-10 (10 being the most attractive)
        2. A brief reasoning for the score
        
        Consider factors like:
        - Historical/cultural significance
        - Uniqueness and must-see status
        - Visitor experience and reviews
        - Alignment with user preferences
        - Seasonal relevance
        
        IMPORTANT: You must respond with a valid JSON object in the following format:
        ```json
        {{
          "rankings": [
            {{
              "category": "Category Name",
              "attractions": [
                {{
                  "name": "Attraction Name",
                  "score": 8.5,
                  "reasoning": "Brief explanation for the score"
                }},
                ...
              ]
            }},
            ...
          ]
        }}
        ```
        
        Do not include any text outside of the JSON structure. Ensure all attraction names exactly match the provided names.
        """)
        
        # Add user preferences if available
        user_prefs = ""
        if state.get("preferences"):
            user_prefs = f"User Preferences: {json.dumps(state['preferences'], indent=2)}"
        
        # Create the human message with the attractions to rank
        categories_text = "\n\n".join([
            f"Category: {category}\nAttractions:\n" + 
            "\n".join([f"- {a.name}: {a.description}" for a in attractions])
            for category, attractions in attractions_by_category.items()
        ])
        
        human_message = HumanMessage(content=f"""
        Destination: {state["destination_name"]}
        
        Destination Report:
        {state["destination_report"]}
        
        {user_prefs}
        
        Please rank the following attractions by category:
        
        {categories_text}
        """)
        
        # Get response from LLM
        response = await self.llm.ainvoke([prompt, human_message])
        
        # Parse the response
        try:
            # Extract JSON from the response (in case there's markdown code block formatting)
            json_content = response.content
            if "```json" in response.content:
                json_content = response.content.split("```json")[1].split("```")[0].strip()
            elif "```" in response.content:
                json_content = response.content.split("```")[1].split("```")[0].strip()
            
            # Parse the JSON response
            ranking_data = json.loads(json_content)
            
            # Convert to CategoryRankings objects
            ranked_categories = []
            for category_data in ranking_data.get("rankings", []):  # Note: changed from "ranked_categories" to "rankings"
                category_name = category_data.get("category", "")
                attraction_rankings = []
                
                for attraction_data in category_data.get("attractions", []):
                    attraction_name = attraction_data.get("name", "")
                    score = attraction_data.get("score", 0)
                    reasoning = attraction_data.get("reasoning", "")
                    
                    # Find the matching attraction
                    matching_attraction = None
                    for attraction in state["attractions"]:
                        if attraction.name.lower() == attraction_name.lower():
                            matching_attraction = attraction
                            break
                    
                    if matching_attraction:
                        attraction_rankings.append(
                            AttractionRanking(
                                attraction=matching_attraction,
                                score=score,
                                reasoning=reasoning
                            )
                        )
                
                if attraction_rankings:
                    ranked_categories.append(
                        CategoryRankings(
                            category=category_name,
                            attractions=sorted(attraction_rankings, key=lambda x: x.score, reverse=True)
                        )
                    )
            
            # Update state with ranked categories
            state["ranked_categories"] = ranked_categories
            
        except Exception as e:
            print(f"Error parsing attraction rankings: {e}")
            # Fallback: create simple rankings based on categories
            ranked_categories = []
            for category, attractions in attractions_by_category.items():
                attraction_rankings = []
                for i, attraction in enumerate(attractions):
                    score = max(10 - i, 1)  # Simple scoring: first gets 10, second gets 9, etc.
                    attraction_rankings.append(
                        AttractionRanking(
                            attraction=attraction,
                            score=score,
                            reasoning=f"Default ranking for {category}"
                        )
                    )
                
                ranked_categories.append(
                    CategoryRankings(
                        category=category,
                        attractions=attraction_rankings
                    )
                )
            
            state["ranked_categories"] = ranked_categories
        
        return state
    
    async def plan_trip(self, state: TripPlanningState) -> Dict[str, Any]:
        """
        Plan the entire trip at once for all days.
        
        Args:
            state: Current state of the trip planning process
            
        Returns:
            Updated state with day plans
        """
        # Calculate the number of days in the trip
        num_days = (state["end_date"] - state["start_date"]).days + 1
        
        # Create a prompt for the LLM to plan the entire trip
        prompt = SystemMessage(content=f"""
        You are a travel itinerary expert specializing in creating detailed multi-day trip schedules.
        
        Your task is to create a realistic and enjoyable itinerary for a {num_days}-day trip to {state["destination_name"]} 
        from {state["start_date"].strftime('%A, %B %d, %Y')} to {state["end_date"].strftime('%A, %B %d, %Y')}.
        
        Guidelines:
        - Plan activities from approximately 9:00 AM to 9:00 PM each day, unless user preferences specify otherwise.
        - ALWAYS USE PROVIDED VISIT DURATION for each attraction!
        - Include 4-5 landmarks/monuments/parks per day
        - Allocate 3-5 hours for museums/galleries
        - Museums/galleries and palaces, churches etc... usually have earlieer opening hours, so try to put them in the earlier part of the day. 
        - Include about 2-3 food/coffee places per day, spread evenly. Treat breakfast, lunch and dinner as separate activities.
        - Allow at least 30 minutes between activities for travel
        - Check opening hours and visit durations to ensure feasibility
        - If there's a festival or special event during the trip and the user hasn't opted out, include it
        - IMPORTANT: Group attractions that are close to each other (within walking distance) to minimize travel time
        - Try to include a mix of different types of attractions across the trip
        - Distribute top-rated attractions evenly across all days of the trip
        
        CRITICAL INSTRUCTIONS:
        1. You MUST use EXACT attraction names from the provided list. Do not modify, abbreviate, or paraphrase attraction names.
        2. For each attraction you include, verify that its exact name appears in the provided attractions list.
        3. Only create custom activities (not tied to specific attractions) if there are NO MORE unused attractions available.
        4. If you run out of attractions in a particular category, use attractions from other categories instead.
        5. Optimize the itinerary by grouping attractions that are close to each other (walking distance).
        6. User preferences override general guidelines.
        7. IMPORTANT: Only include attractions on days that fall within their date range. If an attraction has a specific date range, it should only be scheduled during that period.
        8. For festivals or events without a specified date range, include a note in the description that the date availability is unknown.
        9. Create a balanced schedule across all days - don't front-load the best attractions.
        
        IMPORTANT: You must return your response in a specific JSON format as follows:
        
        ```json
        {{
            "day_plans": [
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
        
        VALIDATION REQUIREMENTS:
        1. Before finalizing your response, verify that EVERY attraction_name in your JSON exactly matches one of the provided attractions.
        2. If an attraction name doesn't match exactly, replace it with the correct name from the provided list.
        3. Only use custom activities if there are NO MORE unused attractions available. Always use actual existing places in the destination. Don't use general descriptions.
        4. Create a balanced, realistic schedule that doesn't rush or overpack any day. It's preferable to have activities of different types spread evenly through each day.
        5. Make sure each day has a proper date in YYYY-MM-DD format that matches the trip's date range.
        6. Make sure the attraction is open at the time of the activity. Check opening hours in the provided ranked attractions list.
        
        ONLY RESPOND WITH THE JSON. Do not include any other text before or after the JSON.
        """)
        
        # Format ranked attractions for the prompt
        ranked_attractions_text = ""
        
        # Add a section specifically listing all available attraction names for easy reference
        available_attractions_list = "Available Attractions (use EXACT names from this list):\n"
        for category in state["ranked_categories"]:
            for ranking in category.attractions:
                available_attractions_list += f"- \"{ranking.attraction.name}\"\n"
        
        # Add detailed attraction information by category
        for category in state["ranked_categories"]:
            ranked_attractions_text += f"\n\nCategory: {category.category}\n"
            for i, ranking in enumerate(category.attractions):
                attraction = ranking.attraction
                
                # Format opening hours if available
                opening_hours = "Not specified"
                if attraction.opening_hours:
                    opening_hours = str(attraction.opening_hours)
                
                # Add travel distance information if available
                travel_info_text = ""
                if hasattr(attraction, 'travel_info') and attraction.travel_info:
                    travel_info_text = "\n   Walking distances to other attractions:\n"
                    for other_name, info in attraction.travel_info.items():
                        travel_info_text += f"     - To {other_name}: {info['distance']} meters, {info['time']} minutes\n"
                
                # Add date range information
                date_range_text = "Available year-round"
                if attraction.date_range:
                    date_range_text = f"Available during: {attraction.date_range}"
                
                ranked_attractions_text += (
                    f"{i+1}. {attraction.name} (Score: {ranking.score}/10)\n"
                    f"   Description: {attraction.description}\n"
                    f"   Visit Duration: {attraction.visit_duration}\n"
                    f"   Opening Hours: {opening_hours}\n"
                    f"   Date Range: {date_range_text}\n"
                    f"   Reasoning: {ranking.reasoning}\n"
                    f"{travel_info_text}"
                )
        
        # Format date range
        date_range = f"{state['start_date'].strftime('%Y-%m-%d')} to {state['end_date'].strftime('%Y-%m-%d')}"
        
        # Create the human message with the planning request
        human_message = HumanMessage(content=f"""
        Plan a {num_days}-day trip to {state["destination_name"]} from {date_range}
        
        User preferences:
        {json.dumps(state["preferences"], indent=2) if state["preferences"] else "No specific preferences provided."}
        
        Destination information:
        {state["destination_report"] if "destination_report" in state else ""}
        
        Available Attractions:
        {available_attractions_list}
        
        Ranked Attractions:
        {ranked_attractions_text}
        
        Please create a detailed itinerary for all {num_days} days using the available attractions.
        Remember to use EXACT attraction names from the list above and verify each name before including it.
        Only create custom activities if you've used all available attractions.
        """)
        
        # Call the LLM to plan the trip
        response = await self.llm.ainvoke([prompt, human_message])
        
        # Create a list of dates for the trip (moved here for access in exception handler)
        trip_dates = [state["start_date"] + timedelta(days=i) for i in range(num_days)]
        print("response: ", response.content)
        # Parse the response
        try:
            # Extract JSON from the response
            json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON with different delimiters or without delimiters
                json_match = re.search(r'\{.*"day_plans".*\}', response.content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # Fallback to creating a default structure
                    print("Could not extract JSON from response, using fallback structure")
                    json_str = '{"day_plans": []}'
            
            # Parse the JSON
            try:
                trip_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Error parsing trip plan JSON: {e}")
                print(f"Response content: {response.content[:200]}...")
                # Create a default structure
                trip_data = {"day_plans": []}
                
                # Since JSON parsing failed, go directly to fallback
                raise Exception("JSON parsing failed, using fallback activities")
            
            # Process each day plan
            day_plans = []
            used_attractions = set()
            
            # Create a mapping of attraction names to attraction objects for quick lookup
            attraction_map = {}
            for category in state["ranked_categories"]:
                for ranking in category.attractions:
                    attraction_map[ranking.attraction.name] = ranking.attraction
            
            for day_data in trip_data.get("day_plans", []):
                date_str = day_data.get("date", "")
                try:
                    date = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    # If date parsing fails, use the next available date
                    if day_plans:
                        date = day_plans[-1].date + timedelta(days=1)
                    else:
                        date = state["start_date"]
                
                activities = []
                for activity_data in day_data.get("activities", []):
                    # Get attraction name
                    attraction_name = activity_data.get("attraction_name", "")
                    
                    # Find the attraction object
                    attraction = None
                    if attraction_name in attraction_map:
                        attraction = attraction_map[attraction_name]
                        
                        # Check if the attraction is available on this date
                        if not self._is_attraction_available_on_date(attraction, date):
                            # Skip this attraction if it's not available on this date
                            print(f"Skipping {attraction_name} as it's not available on {date.strftime('%Y-%m-%d')}")
                            continue
                            
                        used_attractions.add(attraction_name)
                    
                    # Get start and end times
                    start_time_str = activity_data.get("start_time", "")
                    end_time_str = activity_data.get("end_time", "")
                    
                    # Parse times
                    start_time = self._parse_time_from_json(start_time_str, date)
                    end_time = self._parse_time_from_json(end_time_str, date)
                    
                    if not start_time or not end_time:
                        continue
                    
                    # Get description
                    description = activity_data.get("description", f"Visit {attraction_name}" if attraction_name else "")
                    
                    # Add warning for festivals/events without date range
                    if attraction and attraction.category and attraction.category.lower() in ["festival", "event", "festivals", "events"] and not attraction.date_range:
                        description += " (WARNING: Date availability unknown for this event/festival)"
                    
                    # Create the activity
                    activity = Activity(
                        start_time=start_time,
                        end_time=end_time,
                        attraction=attraction,
                        description=description
                    )
                    
                    activities.append(activity)
                
                # Sort activities by start time
                activities.sort(key=lambda a: a.start_time)
                
                # Create the day plan
                day_plan = DayPlan(date=date, activities=activities)
                day_plans.append(day_plan)
            
            # Sort day plans by date
            day_plans.sort(key=lambda d: d.date)
            
            # Update state with day plans and used attractions
            state["day_plans"] = day_plans
            state["used_attractions"] = used_attractions
            
        except Exception as e:
            print(f"Error parsing trip plan: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: create default day plans
            day_plans = []
            used_attractions = set()
            
            for date in trip_dates:
                print(f"Creating fallback activities for {date.strftime('%Y-%m-%d')}")
                activities = self._create_fallback_activities(
                    date, 
                    state["ranked_categories"], 
                    used_attractions
                )
                
                if activities:  # Only add day plan if there are activities
                    print(f"Created {len(activities)} fallback activities for {date.strftime('%Y-%m-%d')}")
                    day_plan = DayPlan(date=date, activities=activities)
                    day_plans.append(day_plan)
                else:
                    print(f"No activities created for {date.strftime('%Y-%m-%d')}")
                    
            # If we still have no day plans, create at least one with a placeholder activity
            if not day_plans and trip_dates:
                print("Creating placeholder activity since no fallback activities were created")
                first_date = trip_dates[0]
                placeholder_activity = Activity(
                    start_time=datetime.combine(first_date, datetime.min.time()) + timedelta(hours=9),
                    end_time=datetime.combine(first_date, datetime.min.time()) + timedelta(hours=10),
                    attraction=None,
                    description="Free time to explore the city"
                )
                day_plan = DayPlan(date=first_date, activities=[placeholder_activity])
                day_plans.append(day_plan)
            
            state["day_plans"] = day_plans
            state["used_attractions"] = used_attractions
        
        return state
    
    def _create_fallback_activities(
        self, 
        date: datetime, 
        ranked_categories: List[CategoryRankings],
        used_attractions: set
    ) -> List[Activity]:
        """
        Create fallback activities for a day when LLM planning fails.
        
        Args:
            date: Date for the activities
            ranked_categories: Ranked categories of attractions
            used_attractions: Set of already used attraction names
            
        Returns:
            List of activities for the day
        """
        activities = []
        current_time = datetime.combine(date, datetime.min.time()) + timedelta(hours=9)  # Start at 9 AM
        
        # Collect top attractions from each category that haven't been used yet
        top_attractions = []
        for category in ranked_categories:
            for ranking in category.attractions:
                attraction = ranking.attraction
                if attraction.name not in used_attractions:
                    # Check if the attraction is available on this date
                    if self._is_attraction_available_on_date(attraction, date):
                        top_attractions.append(attraction)
                        used_attractions.add(attraction.name)
                        break
                    else:
                        print(f"Skipping {attraction.name} in fallback as it's not available on {date.strftime('%Y-%m-%d')}")

        
        # Add more attractions if needed to reach at least 6
        if len(top_attractions) < 6:
            for category in ranked_categories:
                for ranking in category.attractions:
                    attraction = ranking.attraction
                    if attraction.name not in used_attractions:
                        # Check if the attraction is available on this date
                        if self._is_attraction_available_on_date(attraction, date):
                            top_attractions.append(attraction)
                            used_attractions.add(attraction.name)
                            if len(top_attractions) >= 6:
                                break
                        else:
                            print(f"Skipping {attraction.name} in fallback as it's not available on {date.strftime('%Y-%m-%d')}")
                if len(top_attractions) >= 6:
                    break
        
        # Limit to 6 attractions total
        top_attractions = top_attractions[:6]
        
        # Sort attractions to minimize travel distance
        sorted_attractions = self._sort_attractions_by_proximity(top_attractions)
        
        # Create activities for each top attraction
        for attraction in sorted_attractions:
            # Determine duration
            duration_minutes = 60
            if attraction.visit_duration:
                try:
                    # Try to parse the duration string (e.g., "2 hours", "90 minutes")
                    duration_str = attraction.visit_duration.lower()
                    if "hour" in duration_str:
                        hours = float(re.search(r'(\d+(\.\d+)?)', duration_str).group(1))
                        duration_minutes = int(hours * 60)
                    elif "minute" in duration_str:
                        duration_minutes = int(re.search(r'(\d+)', duration_str).group(1))
                    else:
                        # Try to parse as a number (assumed to be hours)
                        duration_minutes = int(float(duration_str) * 60)
                except (ValueError, AttributeError):
                    # Default to 1 hour if parsing fails
                    duration_minutes = 60
            
            # Create the activity
            end_time = current_time + timedelta(minutes=duration_minutes)
            
            # Prepare description with warning for festivals/events without date range
            description = f"Visit {attraction.name}"
            if attraction.category and attraction.category.lower() in ["festival", "event", "festivals", "events"] and not attraction.date_range:
                description += " (WARNING: Date availability unknown for this event/festival)"
            
            activity = Activity(
                start_time=current_time,
                end_time=end_time,
                attraction=attraction,
                description=description
            )
            
            activities.append(activity)
            
            # Move to the next time slot (add 30 minutes for travel)
            current_time = end_time + timedelta(minutes=30)
        
        return activities
    
    def _sort_attractions_by_proximity(self, attractions: List[Attraction]) -> List[Attraction]:
        """
        Sort attractions to minimize travel distance between them.
        
        This function implements a greedy algorithm to find a path through attractions
        that minimizes the total walking distance.
        
        Args:
            attractions: List of attractions to sort
            
        Returns:
            Sorted list of attractions to minimize travel distance
        """
        if not attractions:
            return []
            
        # If there's only one attraction or none have travel_info, return as is
        if len(attractions) <= 1 or not any(hasattr(a, 'travel_info') and a.travel_info for a in attractions):
            return attractions
            
        # Start with the first attraction
        sorted_attractions = [attractions[0]]
        remaining = attractions[1:]
        
        # Greedy algorithm to find the next closest attraction
        while remaining:
            current = sorted_attractions[-1]
            
            if not hasattr(current, 'travel_info') or not current.travel_info:
                # If current attraction has no travel info, just add the next one
                sorted_attractions.append(remaining.pop(0))
                continue
                
            # Find the closest remaining attraction
            closest_idx = 0
            closest_distance = float('inf')
            
            for i, attraction in enumerate(remaining):
                if attraction.name in current.travel_info:
                    distance = current.travel_info[attraction.name]['distance']
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_idx = i
            
            # Add the closest attraction to our path
            sorted_attractions.append(remaining.pop(closest_idx))
        
        return sorted_attractions
    
    async def create_trip(self, state: TripPlanningState) -> Dict[str, Any]:
        """
        Create the final Trip object from day plans.
        
        Args:
            state: Current state of the trip planning process
            
        Returns:
            Updated state with the final Trip object
        """
        # Create the Trip object
        # First, create a Location object for the destination
        destination_location = Location(
            name=state["destination_name"],
            address=""  # We don't have the address in the state
        )
        
        trip = Trip(
            title=f"Trip to {state['destination_name']}",
            destination=destination_location,
            start_date=state["start_date"],
            end_date=state["end_date"],
            days=state["day_plans"]
        )
        
        # Update state with the trip
        state["trip"] = trip
        
        return state
    
    def _parse_time_from_json(self, time_str: str, date: datetime) -> Optional[datetime]:
        """
        Parse a time string from JSON into a datetime object.
        
        Args:
            time_str: Time string from JSON (e.g., "09:00", "9:00 AM")
            date: Date to combine with the time
            
        Returns:
            Datetime object if parsing succeeds, None otherwise
        """
        if not time_str:
            return None
            
        try:
            # Try different time formats
            formats = ["%H:%M", "%I:%M %p", "%I:%M%p"]
            
            for fmt in formats:
                try:
                    time_obj = datetime.strptime(time_str, fmt).time()
                    return datetime.combine(date, time_obj)
                except ValueError:
                    continue
            
            # If all formats fail, try extracting hours and minutes with regex
            match = re.search(r'(\d+):(\d+)', time_str)
            if match:
                hours = int(match.group(1))
                minutes = int(match.group(2))
                
                # Adjust for PM if specified
                if "pm" in time_str.lower() and hours < 12:
                    hours += 12
                
                return datetime.combine(date, datetime.min.time()) + timedelta(hours=hours, minutes=minutes)
            
            return None
        except Exception as e:
            print(f"Error parsing time from JSON: {e}")
            return None
            
    def _is_attraction_available_on_date(self, attraction: Attraction, date: datetime) -> bool:
        """
        Check if an attraction is available on a specific date based on its date range.
        
        Args:
            attraction: The attraction to check
            date: The date to check availability for
            
        Returns:
            True if the attraction is available on the date, False otherwise
        """
        # If no date range is specified, assume it's available year-round
        if not attraction.date_range:
            return True
            
        try:
            # Common date formats to try parsing
            date_formats = [
                # Month day-day, year
                r"([A-Za-z]+)\s+(\d+)-(\d+),\s+(\d{4})",  # "July 1-15, 2025"
                # Month day - Month day, year
                r"([A-Za-z]+)\s+(\d+)\s*-\s*([A-Za-z]+)\s+(\d+),\s+(\d{4})",  # "July 1 - August 15, 2025"
                # Month - Month, year
                r"([A-Za-z]+)\s*-\s*([A-Za-z]+),\s+(\d{4})",  # "July-August, 2025"
            ]
            
            # Convert date to string format for comparison
            date_str = date.strftime("%B %d, %Y")  # e.g., "July 15, 2025"
            month_str = date.strftime("%B")  # e.g., "July"
            year_str = date.strftime("%Y")  # e.g., "2025"
            
            # Try to parse the date range
            for pattern in date_formats:
                match = re.search(pattern, attraction.date_range)
                if match:
                    # Different handling based on the pattern matched
                    if len(match.groups()) == 4:  # "July 1-15, 2025"
                        month = match.group(1)
                        start_day = int(match.group(2))
                        end_day = int(match.group(3))
                        year = int(match.group(4))
                        
                        # Check if date falls within the range
                        if (date.year == year and 
                            date.strftime("%B").lower() == month.lower() and
                            start_day <= date.day <= end_day):
                            return True
                    
                    elif len(match.groups()) == 5:  # "July 1 - August 15, 2025"
                        start_month = match.group(1)
                        start_day = int(match.group(2))
                        end_month = match.group(3)
                        end_day = int(match.group(4))
                        year = int(match.group(5))
                        
                        # Convert month names to numbers
                        start_month_num = datetime.strptime(start_month, "%B").month
                        end_month_num = datetime.strptime(end_month, "%B").month
                        
                        # Check if date falls within the range
                        if (date.year == year and 
                            ((date.month > start_month_num and date.month < end_month_num) or
                             (date.month == start_month_num and date.day >= start_day) or
                             (date.month == end_month_num and date.day <= end_day))):
                            return True
                    
                    elif len(match.groups()) == 3:  # "July-August, 2025"
                        start_month = match.group(1)
                        end_month = match.group(2)
                        year = int(match.group(3))
                        
                        # Convert month names to numbers
                        start_month_num = datetime.strptime(start_month, "%B").month
                        end_month_num = datetime.strptime(end_month, "%B").month
                        
                        # Check if date falls within the range
                        if (date.year == year and 
                            start_month_num <= date.month <= end_month_num):
                            return True
            
            # Simple string matching for more flexible handling
            # Check if the current month and year are mentioned in the date range
            if (month_str.lower() in attraction.date_range.lower() and 
                year_str in attraction.date_range):
                return True
                
            # Check if just the current year is mentioned (for annual events)
            if year_str in attraction.date_range:
                return True
                
            # If we couldn't parse the date range but it exists, be conservative and return False
            return False
            
        except Exception as e:
            print(f"Error checking date range for {attraction.name}: {e}")
            # If there's an error parsing, assume it's not available to be safe
            return False