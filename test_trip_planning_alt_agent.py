#!/usr/bin/env python
"""
Test script for TripPlanningAltAgent with Copenhagen trip data.

This script tests the alternative trip planning agent that plans all days at once
instead of day-by-day, which may produce more balanced itineraries.
"""

import json
import os
import asyncio
import uuid
from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from dotenv import load_dotenv

from src.agents.trip_planning_alt.trip_planning_alt import TripPlanningAltAgent
from src.models.trip import Location, Attraction

load_dotenv()

# Configure LangSmith project name if enabled
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "trip-planning-alt")
LANGSMITH_ENABLED = bool(os.getenv("LANGSMITH_API_KEY"))

def load_attractions(file_path):
    """Load attractions from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            attractions_data = json.load(f)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return []
    
    attractions = []
    for attraction_data in attractions_data:
        # Create location object
        location_data = attraction_data.get("location", {})
        location = Location(
            name=location_data.get("name", ""),
            address=location_data.get("address")
        )
        
        # Create attraction object
        attraction = Attraction(
            name=attraction_data.get("name", ""),
            description=attraction_data.get("description", ""),
            location=location,
            category=attraction_data.get("category", ""),
            visit_duration=attraction_data.get("visit_duration", "60"),
            opening_hours=attraction_data.get("opening_hours", {}),
            price=attraction_data.get("price"),
            rating=attraction_data.get("rating"),
            travel_info=attraction_data.get("travel_info", {}),
            date_range=attraction_data.get("date_range", "")
        )
        attractions.append(attraction)
    
    return attractions


def load_destination_report(file_path):
    """Load destination report from a markdown file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Destination report file {file_path} not found.")
        return ""


async def main():
    """Run the TripPlanningAltAgent test."""
    # Set up paths
    base_dir = Path(__file__).parent
    attractions_path = base_dir / "extracted_attractions.json"
    report_path = base_dir / "cph_test_report.md"
    
    # Load data
    attractions = load_attractions(attractions_path)
    if not attractions:
        print("No attractions loaded. Exiting.")
        return
    
    print(f"Loaded {len(attractions)} attractions")
    
    # Load destination report
    destination_report = load_destination_report(report_path)
    if not destination_report:
        print("Warning: No destination report loaded. Proceeding without it.")
    else:
        print("Loaded destination report")
    
    # Create user preferences
    preferences = {
        "travel_style": "cultural",
        "interests": ["food", "art", "nightlife", "festivals"],
        "activity_level": "moderate",
        "preferred_transportation": ["walking", "public transport"]
    }
    
    # Set up dates
    start_date = datetime.strptime("2025-06-30", "%Y-%m-%d")
    end_date = datetime.strptime("2025-07-01", "%Y-%m-%d")
    
    # Initialize the agent
    # Make sure to set your OpenAI API key in the environment
    llm = ChatOpenAI(model="o4-mini")
    agent = TripPlanningAltAgent(llm=llm)
    
    # Set up LangSmith tracing if enabled
    callbacks = None
    if LANGSMITH_ENABLED:
        run_id = str(uuid.uuid4())
        tracer = LangChainTracer(
            project_name=LANGSMITH_PROJECT,
            run_id=run_id
        )
        callbacks = CallbackManager([tracer])
        print(f"LangSmith tracing enabled with run ID: {run_id}")
    
    # Plan the trip
    print("Planning trip to Copenhagen using the alternative trip planning agent...")
    trip = await agent.process(
        destination_name="Copenhagen",
        attractions=attractions,
        start_date=start_date,
        end_date=end_date,
        preferences=preferences,
        destination_report=destination_report,
        callbacks=callbacks
    )
    
    # Print trip summary
    print("\n" + "=" * 50)
    print(f"Trip to {trip.destination}")
    print(f"Duration: {trip.start_date.strftime('%Y-%m-%d')} to {trip.end_date.strftime('%Y-%m-%d')}")
    print("=" * 50)
    
    # Print daily itinerary
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
    
    print("\nTrip planning completed! Results saved to trip_plan_alt.md")


if __name__ == "__main__":
    asyncio.run(main())