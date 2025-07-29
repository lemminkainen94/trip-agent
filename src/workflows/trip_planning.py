"""Trip planning workflow implementation using langgraph."""

from typing import Annotated, Any, Dict, List, Optional, TypedDict, Union
from datetime import datetime

from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from src.agents.orchestrator import OrchestratorAgent
from src.models.preferences import UserPreferences, TripRequest
from src.models.trip import Trip


class TripPlanningState(TypedDict):
    """State for the trip planning workflow."""
    
    messages: List[Union[HumanMessage, AIMessage]]  # Conversation history
    user_preferences: Optional[UserPreferences]  # User preferences
    trip_request: Optional[TripRequest]  # Trip request details
    destination_data: Optional[Dict[str, Any]]  # Destination information
    logistics_data: Optional[Dict[str, Any]]  # Logistics information
    trip_plan: Optional[Trip]  # Generated trip plan


def create_trip_planning_graph(orchestrator: OrchestratorAgent) -> StateGraph:
    """Create a trip planning workflow graph.
    
    Args:
        orchestrator: Orchestrator agent to coordinate the workflow.
        
    Returns:
        StateGraph: The trip planning workflow graph.
    """
    # Define the workflow graph
    workflow = StateGraph(TripPlanningState)
    
    # Define the nodes in the graph
    
    # Node for processing user input
    @workflow.node("process_user_input")
    async def process_user_input(state: TripPlanningState) -> Dict[str, Any]:
        """Process user input and update the state.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Dict: Updated state values.
        """
        # Get the last message from the user
        last_human_message = None
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                last_human_message = message
                break
        
        if not last_human_message:
            # No user message found, return unchanged state
            return {}
        
        # Process the user input using the orchestrator
        result = await orchestrator.process_user_input(
            last_human_message.content, 
            state.get("user_preferences")
        )
        
        # Update the state with the result
        updates = {
            "messages": add_messages(state["messages"], [AIMessage(content=result["response"])])
        }
        
        if "preferences" in result and result["preferences"]:
            updates["user_preferences"] = result["preferences"]
        
        if "trip_request" in result and result["trip_request"]:
            updates["trip_request"] = result["trip_request"]
        
        if "trip_plan" in result and result["trip_plan"]:
            updates["trip_plan"] = result["trip_plan"]
        
        return updates
    
    # Node for generating destination information
    @workflow.node("generate_destination_info")
    async def generate_destination_info(state: TripPlanningState) -> Dict[str, Any]:
        """Generate destination information and update the state.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Dict: Updated state values.
        """
        trip_request = state.get("trip_request")
        if not trip_request:
            # No trip request found, return unchanged state
            return {}
        
        # Generate destination information using the orchestrator's destination info agent
        result = await orchestrator.destination_info_agent.process(
            f"Research information about {trip_request.destination}",
            trip_request.preferences
        )
        
        # Update the state with the result
        return {
            "destination_data": result.get("destination_data", {})
        }
    
    # Node for generating logistics information
    @workflow.node("generate_logistics_info")
    async def generate_logistics_info(state: TripPlanningState) -> Dict[str, Any]:
        """Generate logistics information and update the state.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Dict: Updated state values.
        """
        trip_request = state.get("trip_request")
        destination_data = state.get("destination_data")
        
        if not trip_request or not destination_data:
            # Missing required data, return unchanged state
            return {}
        
        # Parse dates
        start_date = datetime.strptime(trip_request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(trip_request.end_date, "%Y-%m-%d")
        date_range = (start_date, end_date)
        
        # Generate logistics information using the orchestrator's local events agent
        result = await orchestrator.local_events_agent.process(
            f"Research events and logistics for {trip_request.destination} from {trip_request.start_date} to {trip_request.end_date}",
            trip_request.preferences,
            destination_data.get("location"),
            date_range
        )
        
        # Update the state with the result
        return {
            "logistics_data": result.get("logistics_data", {})
        }
    
    # Node for generating the trip plan
    @workflow.node("generate_trip_plan")
    async def generate_trip_plan(state: TripPlanningState) -> Dict[str, Any]:
        """Generate the trip plan and update the state.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Dict: Updated state values.
        """
        trip_request = state.get("trip_request")
        destination_data = state.get("destination_data")
        logistics_data = state.get("logistics_data")
        
        if not trip_request or not destination_data or not logistics_data:
            # Missing required data, return unchanged state
            return {}
        
        # Parse dates
        start_date = datetime.strptime(trip_request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(trip_request.end_date, "%Y-%m-%d")
        date_range = (start_date, end_date)
        
        # Generate trip plan using the orchestrator's itinerary agent
        result = await orchestrator.itinerary_agent.process(
            f"Create an optimized itinerary for {trip_request.destination} from {trip_request.start_date} to {trip_request.end_date}",
            trip_request.preferences,
            destination_data,
            logistics_data,
            date_range
        )
        
        # Update the state with the result
        trip_plan = result.get("itinerary")
        
        # Add a message to the conversation with the trip plan summary
        trip_summary = f"I've created a trip plan for {trip_request.destination} from {trip_request.start_date} to {trip_request.end_date}."
        
        if trip_plan and trip_plan.days:
            day_count = len(trip_plan.days)
            attraction_count = sum(
                1 for day in trip_plan.days 
                for activity in day.activities 
                if activity.attraction
            )
            
            trip_summary += f" The plan includes {day_count} days with {attraction_count} attractions to visit."
        
        trip_summary += " Would you like me to explain the details of this trip plan?"
        
        return {
            "trip_plan": trip_plan,
            "messages": add_messages(state["messages"], [AIMessage(content=trip_summary)])
        }
    
    # Define the edges in the graph
    
    # Start with processing user input
    workflow.set_entry_point("process_user_input")
    
    # Define the conditional edges
    @workflow.conditional_edge("process_user_input")
    def route_after_user_input(state: TripPlanningState) -> str:
        """Determine the next node after processing user input.
        
        Args:
            state: Current workflow state.
            
        Returns:
            str: Name of the next node.
        """
        if state.get("trip_plan"):
            # If we already have a trip plan, we're done
            return "END"
        elif state.get("trip_request") and not state.get("destination_data"):
            # If we have a trip request but no destination data, generate destination info
            return "generate_destination_info"
        else:
            # Otherwise, wait for more user input
            return "END"
    
    workflow.add_conditional_edges(
        "process_user_input",
        route_after_user_input
    )
    
    # After generating destination info, generate logistics info
    workflow.add_edge("generate_destination_info", "generate_logistics_info")
    
    # After generating logistics info, generate the trip plan
    workflow.add_edge("generate_logistics_info", "generate_trip_plan")
    
    # After generating the trip plan, we're done
    workflow.add_edge("generate_trip_plan", "END")
    
    return workflow