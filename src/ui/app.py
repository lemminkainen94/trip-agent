"""Streamlit UI for the Trip Agent system."""

import os
import json
import uuid
import requests
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import pandas as pd

from src.models.trip import Trip, Location, Attraction, Activity, DayPlan
from src.models.preferences import UserPreferences
from src.agents.user_interface import UserInterfaceAgent, ConversationState
from src.agents.destination_research_assistant.destination_report import DestinationReportAgent
from src.agents.attraction_extraction.attraction_extraction import AttractionExtractionAgent
from src.agents.trip_planning_react.trip_planning_react import TripPlanningReactAgent


# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000")
USE_LOCAL_AGENTS = os.getenv("USE_LOCAL_AGENTS", "true").lower() == "true"


# Session state initialization
def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    if "trip_plan" not in st.session_state:
        st.session_state.trip_plan = None
    
    if "user_preferences" not in st.session_state:
        st.session_state.user_preferences = None
    
    if "destination_report" not in st.session_state:
        st.session_state.destination_report = None
    
    if "attractions" not in st.session_state:
        st.session_state.attractions = None
    
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = None
    
    if "agents" not in st.session_state and USE_LOCAL_AGENTS:
        # Initialize agents if using local processing
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model_name=os.getenv("OPENAI_MODEL_NAME", "o4-mini")
        )
        
        st.session_state.agents = {
            "user_interface": UserInterfaceAgent(llm=llm),
            "destination_report": DestinationReportAgent(llm=llm),
            "attraction_extraction": AttractionExtractionAgent(llm=llm),
            "trip_planning": TripPlanningReactAgent(llm=llm)
        }


# Helper functions
def send_message(message: str) -> Dict:
    """Send a message to the API and get the response.
    
    Args:
        message: Message to send.
        
    Returns:
        Dict: Response from the API.
    """
    if USE_LOCAL_AGENTS:
        import asyncio
        return asyncio.run(process_message_locally(message))
    else:
        return process_message_via_api(message)


def process_message_via_api(message: str) -> Dict:
    """Process a message via the API.
    
    Args:
        message: Message to send.
        
    Returns:
        Dict: Response from the API.
    """
    # Prepare the request
    messages = st.session_state.messages + [{"role": "user", "content": message}]
    
    # Convert messages to the format expected by the API
    api_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
    ]
    
    # Send the request to the API
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "messages": api_messages,
                "user_id": st.session_state.user_id
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with the API: {e}")
        return {"message": {"role": "assistant", "content": "I'm sorry, I'm having trouble connecting to the server."}}


async def process_message_locally(message: str) -> Dict:
    """Process a message using local agents.
    
    Args:
        message: Message to send.
        
    Returns:
        Dict: Response with assistant message and optional trip plan.
    """
    agents = st.session_state.agents
    
    # Process the user input with the user interface agent
    result = await agents["user_interface"].process(
        message,
        st.session_state.user_preferences
    )
    
    # Update the session with the preferences
    if "preferences" in result and result["preferences"]:
        st.session_state.user_preferences = result["preferences"]
        
        # Check if we have enough information to start trip planning
        # Only start planning if the agent has confirmed we're ready
        ui_agent = agents["user_interface"]
        if (ui_agent.conversation_context.state == ConversationState.READY and
            hasattr(result["preferences"], "destination") and 
            result["preferences"].destination and 
            hasattr(result["preferences"], "start_date") and 
            result["preferences"].start_date and
            hasattr(result["preferences"], "end_date") and 
            result["preferences"].end_date):
            
            # Start background trip planning
            st.session_state.processing_status = "destination_report"
    
    # Create response
    return {
        "message": {"role": "assistant", "content": result["response"]},
        "trip_plan": st.session_state.trip_plan
    }


async def continue_trip_planning_process():
    """Continue the trip planning process based on current status."""
    if not st.session_state.processing_status:
        return
    
    agents = st.session_state.agents
    preferences = st.session_state.user_preferences
    
    if st.session_state.processing_status == "destination_report":
        # Step 1: Generate destination report
        destination_name = preferences.destination
        with st.spinner("Researching destination information..."):
            # Run synchronous process method in a thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            destination_report = await loop.run_in_executor(
                None,
                lambda: agents["destination_report"].process(
                    destination_name=destination_name,
                    user_preferences=preferences
                )
            )
            st.session_state.destination_report = destination_report["report"]
            with open(f"{destination_name}-{preferences.start_date}-{preferences.end_date}.md", "w") as f:
                f.write(destination_report["report"])
    
            st.session_state.processing_status = "attraction_extraction"
            # Force a rerun to continue the process
            st.rerun()
    
    
    elif st.session_state.processing_status == "attraction_extraction":
        # Step 2: Extract attractions from the destination report
        destination_name = preferences.destination
        with st.spinner("Extracting attractions..."):
            # Check if the process method is async or sync
            if asyncio.iscoroutinefunction(agents["attraction_extraction"].process):
                attractions = await agents["attraction_extraction"].process(
                    report_content=st.session_state.destination_report,
                    destination_name=destination_name
                )
            else:
                # Run synchronous process method in a thread pool
                loop = asyncio.get_running_loop()
                attractions = await loop.run_in_executor(
                    None,
                    lambda: agents["attraction_extraction"].process(
                        report_content=st.session_state.destination_report,
                        destination_name=destination_name
                    )
                )
            st.session_state.attractions = attractions
            st.session_state.processing_status = "trip_planning"
            # Force a rerun to continue the process
            st.rerun()
    
    
    elif st.session_state.processing_status == "trip_planning":
        # Step 3: Generate trip plan using the trip planning agent
        destination_name = preferences.destination
        start_date = preferences.start_date
        end_date = preferences.end_date
        
        # Get excluded categories if available
        excluded_categories = []
        if hasattr(preferences, "excluded_categories") and preferences.excluded_categories:
            excluded_categories = preferences.excluded_categories
        
        with st.spinner("Creating your personalized trip plan..."):
            # Check if the process method is async or sync
            if asyncio.iscoroutinefunction(agents["trip_planning"].process):
                trip_plan = await agents["trip_planning"].process(
                    destination_name=destination_name,
                    attractions=st.session_state.attractions,
                    start_date=start_date,
                    end_date=end_date,
                    preferences=preferences.dict() if hasattr(preferences, "dict") else {},
                    excluded_categories=excluded_categories,
                    destination_report=st.session_state.destination_report
                )
            else:
                # Run synchronous process method in a thread pool
                loop = asyncio.get_running_loop()
                trip_plan = await loop.run_in_executor(
                    None,
                    lambda: agents["trip_planning"].process(
                        destination_name=destination_name,
                        attractions=st.session_state.attractions,
                        start_date=start_date,
                        end_date=end_date,
                        preferences=preferences.dict() if hasattr(preferences, "dict") else {},
                        excluded_categories=excluded_categories,
                        destination_report=st.session_state.destination_report
                    )
                )
            st.session_state.trip_plan = trip_plan
            with open(f"trip-plan-{destination_name}-{start_date}-{end_date}.md", "w") as f:
                f.write(f"Trip: {trip_plan.title}\n")
                f.write(f"Destination: {trip_plan.destination.name}\n")
                f.write(f"Duration: {trip_plan.start_date.strftime('%Y-%m-%d')} to {trip_plan.end_date.strftime('%Y-%m-%d')}\n")
                f.write("=" * 50 + "\n")
                for day_plan in trip_plan.days:
                    print(f"\n--- Day: {day_plan.date.strftime('%Y-%m-%d')} ---")
                    f.write(f"\n--- Day: {day_plan.date.strftime('%Y-%m-%d')} ---\n")
                    
                    for activity in day_plan.activities:
                        activity_str = f"  {activity.start_time.strftime('%H:%M')} - {activity.end_time.strftime('%H:%M')}, {activity.attraction.name}: {activity.description}"
                        print(activity_str)
                        f.write(f"{activity_str}\n")
            
            # Clear processing status after trip_plan planning is complete
            st.session_state.processing_status = "completed"
            # Force a rerun to update the UI with the completed trip plan
            st.rerun()
    
    elif st.session_state.processing_status == "completed":
        pass


def get_trip_plan() -> Optional[Trip]:
    """Get the trip plan from the API or local state.
    
    Returns:
        Optional[Trip]: The trip plan if available.
    """
    if USE_LOCAL_AGENTS:
        return st.session_state.trip_plan
    
    try:
        response = requests.get(f"{API_URL}/trip_plan/{st.session_state.user_id}")
        response.raise_for_status()
        
        if response.status_code == 200 and response.content:
            # Parse the trip plan
            trip_data = response.json()
            
            if trip_data:
                # Convert the JSON to a Trip object
                return Trip(**trip_data)
        
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting trip plan: {e}")
        return None


def create_trip_map(trip: Trip) -> folium.Map:
    """Create a folium map for the trip.
    
    Args:
        trip: Trip to create a map for.
        
    Returns:
        folium.Map: Map of the trip.
    """
    # Create a map centered on the trip destination
    trip_map = folium.Map(
        location=[trip.destination.latitude, trip.destination.longitude],
        zoom_start=12
    )
    
    # Create a marker cluster for attractions
    marker_cluster = MarkerCluster().add_to(trip_map)
    
    # Add markers for attractions
    attractions = []
    for day in trip.days:
        for activity in day.activities:
            if activity.attraction:
                attractions.append({
                    "name": activity.attraction.name,
                    "location": activity.attraction.location,
                    "day": day.date.strftime("%Y-%m-%d"),
                    "time": f"{activity.start_time.strftime('%H:%M')} - {activity.end_time.strftime('%H:%M')}"
                })
    
    # Add markers for each attraction
    for attraction in attractions:
        folium.Marker(
            location=[attraction["location"].latitude, attraction["location"].longitude],
            popup=folium.Popup(
                f"<b>{attraction['name']}</b><br>"
                f"Day: {attraction['day']}<br>"
                f"Time: {attraction['time']}",
                max_width=300
            ),
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(marker_cluster)
    
    return trip_map


def display_trip_details(trip: Trip):
    """Display trip details in the UI.
    
    Args:
        trip: Trip to display details for.
    """
    # Display trip title and dates
    st.write(f"**{trip.title}**")
    st.write(f"*{trip.start_date.strftime('%B %d, %Y')} to {trip.end_date.strftime('%B %d, %Y')}*")
    
    # Create tabs for each day
    day_tabs = st.tabs([f"Day {i+1}" for i in range(len(trip.days))])
    
    # Fill each tab with the day's activities
    for i, (tab, day) in enumerate(zip(day_tabs, trip.days)):
        with tab:
            st.write(f"**Day {i+1}: {day.date.strftime('%A, %B %d, %Y')}**")
            
            # Create a dataframe for the day's activities
            activities_data = []
            for activity in day.activities:
                activity_type = "Visit" if activity.attraction else "Travel" if activity.travel else "Other"
                name = (
                    activity.attraction.name if activity.attraction 
                    else f"{activity.travel.origin.name} to {activity.travel.destination.name}" if activity.travel 
                    else activity.description
                )
                
                activities_data.append({
                    "Time": f"{activity.start_time.strftime('%H:%M')} - {activity.end_time.strftime('%H:%M')}",
                    "Activity": name,
                    "Type": activity_type,
                    "Description": activity.description
                })
            
            # Display the activities as a table
            if activities_data:
                activities_df = pd.DataFrame(activities_data)
                st.table(activities_df)
            else:
                st.write("No activities planned for this day.")


def display_processing_status():
    """Display the current processing status."""
    if not st.session_state.processing_status:
        return
    
    status = st.session_state.processing_status
    
    if status == "destination_report":
        st.info("Researching destination information...")
    elif status == "attraction_extraction":
        st.info("Extracting attractions...")
    elif status == "trip_planning":
        st.info("Creating your personalized trip plan...")
    elif status == "completed":
        st.success("Trip plan completed!")


# Main UI
def main():
    """Main UI function."""
    # Set page config
    st.set_page_config(
        page_title="Trip Agent",
        page_icon="🧳",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Page title
    st.title("Trip Agent 🧳")
    st.write("Your AI-powered travel planning assistant")
    
    # Create a two-column layout
    col1, col2 = st.columns([1, 1])
    
    # Chat interface in the first column
    with col1:
        st.subheader("Chat with Trip Agent")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Display processing status
        display_processing_status()
        
        # Chat input
        user_input = st.chat_input("Ask about trip planning...")
        
        if user_input:
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Add user message to session state
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Get response from API or local agents
            with st.spinner("Thinking..."):
                if USE_LOCAL_AGENTS:
                    import asyncio
                    response = asyncio.run(process_message_locally(user_input))
                else:
                    response = process_message_via_api(user_input)
            
            # Display assistant message
            with st.chat_message("assistant"):
                st.write(response["message"]["content"])
            
            # Add assistant message to session state
            st.session_state.messages.append({"role": "assistant", "content": response["message"]["content"]})
            
            # Check if we have a trip plan
            if "trip_plan" in response and response["trip_plan"]:
                st.session_state.trip_plan = Trip(**response["trip_plan"]) if isinstance(response["trip_plan"], dict) else response["trip_plan"]
        
        # Continue trip planning process if needed
        if USE_LOCAL_AGENTS and st.session_state.processing_status and st.session_state.processing_status != "completed":
            import asyncio
            asyncio.run(continue_trip_planning_process())
    
    # Trip details and map in the second column
    with col2:
        # Check if we have a trip plan
        trip_plan = st.session_state.trip_plan or get_trip_plan()
        
        if trip_plan:
            st.subheader("Trip Plan")
            
            # Create tabs for map and details
            map_tab, details_tab = st.tabs(["Map", "Details"])
            
            with map_tab:
                # Create and display the map
                # TODO: attraction geolocations and map creation
                """trip_map = create_trip_map(trip_plan)
                folium_static(trip_map, width=600, height=500)"""
            
            with details_tab:
                # Display trip details
                display_trip_details(trip_plan)
        else:
            st.info(
                "No trip plan yet. Start by telling Trip Agent about your travel plans. "
                "For example: 'I want to plan a trip to Paris for 5 days in June.'"
            )


if __name__ == "__main__":
    main()