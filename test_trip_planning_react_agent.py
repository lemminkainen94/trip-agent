"""
Test script for the ReAct-based Trip Planning Agent.

This script demonstrates how to use the TripPlanningReactAgent to create
a trip plan with improved validation of opening hours and date ranges.
"""

import asyncio
import json
from datetime import datetime, timedelta
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.agents.trip_planning_react import TripPlanningReactAgent
from src.models.trip import Attraction, Location

load_dotenv()

def load_attractions(file_path="extracted_attractions.json"):
    """Load attractions from a JSON file."""
    with open(file_path, "r") as f:
        attractions_data = json.load(f)
    
    attractions = []
    for attraction_data in attractions_data:
        # Convert location dictionary to Location object
        location = Location(**attraction_data["location"])
        
        # Create Attraction object
        attraction = Attraction(
            name=attraction_data["name"],
            description=attraction_data["description"],
            location=location,
            category=attraction_data["category"],
            visit_duration=attraction_data["visit_duration"],
            opening_hours=attraction_data.get("opening_hours"),
            date_range=attraction_data.get("date_range")
        )
        attractions.append(attraction)
    
    return attractions


def load_destination_report(file_path="cph_test_report.md"):
    """Load destination report from a file."""
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, "r") as f:
        return f.read()


async def main():
    # Load attractions
    attractions = load_attractions()
    print(f"Loaded {len(attractions)} attractions")
    
    # Load destination report
    destination_report = load_destination_report()
    if destination_report:
        print(f"Loaded destination report ({len(destination_report)} characters)")
    else:
        print("No destination report found")
    
    # Set up trip dates
    start_date = datetime.now() + timedelta(days=1)  # Tomorrow
    end_date = start_date + timedelta(days=2)  # 3-day trip
    
    print(f"Planning trip from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Set up user preferences
    preferences = {
        "interests": ["history", "art", "food"],
        "pace": "moderate",
        "budget": "mid-range"
    }
    
    # Create the agent
    llm = ChatOpenAI(model="o4-mini")
    agent = TripPlanningReactAgent(llm=llm)
    
    # Process the trip planning request
    trip = await agent.process(
        destination_name="Copenhagen",
        attractions=attractions,
        start_date=start_date,
        end_date=end_date,
        preferences=preferences,
        excluded_categories=[],
        destination_report=destination_report
    )
    
    # Print the trip plan
    print("\n=== Trip Plan ===")
    print(f"Destination: {trip.destination}")
    print(f"Dates: {trip.start_date.strftime('%Y-%m-%d')} to {trip.end_date.strftime('%Y-%m-%d')}")
    
    with open("trip_plan_alt.md", "w", encoding="utf-8") as f:
        f.write(f"Trip: {trip.title}\n")
        f.write(f"Destination: {trip.destination.name}\n")
        f.write(f"Duration: {trip.start_date.strftime('%Y-%m-%d')} to {trip.end_date.strftime('%Y-%m-%d')}\n")
        f.write("=" * 50 + "\n")
        for day_plan in trip.days:
            print(f"\n--- Day: {day_plan.date.strftime('%Y-%m-%d')} ---")
            f.write(f"\n--- Day: {day_plan.date.strftime('%Y-%m-%d')} ---\n")
            
            for activity in day_plan.activities:
                activity_str = f"  {activity.start_time.strftime('%H:%M')} - {activity.end_time.strftime('%H:%M')}, {activity.attraction.name}: {activity.description}"
                print(activity_str)
                f.write(f"{activity_str}\n")


if __name__ == "__main__":
    asyncio.run(main())