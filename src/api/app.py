"""FastAPI application for the Trip Agent system."""

import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from langchain.callbacks.base import Callbacks
from langchain.callbacks.manager import CallbackManager
from langsmith.run_helpers import traceable
from langsmith.run_trees import RunTree
from langsmith.callbacks import LangChainTracer

from src.models.preferences import UserPreferences, TripRequest
from src.models.trip import Trip, Location, Attraction, Activity, DayPlan
from src.agents.user_interface import UserInterfaceAgent
from src.agents.destination_research_assistant.destination_report import DestinationReportAgent
from src.agents.attraction_extraction.attraction_extraction import AttractionExtractionAgent
from src.agents.trip_planning_react.trip_planning_react import TripPlanningReactAgent


# Create FastAPI app
app = FastAPI(
    title="Trip Agent API",
    description="API for the Multi-Agent Trip Planner system",
    version="0.1.0",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# API models
class ChatMessage(BaseModel):
    """Chat message model for the API."""
    
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")


class ChatRequest(BaseModel):
    """Chat request model for the API."""
    
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    user_id: Optional[str] = Field(None, description="User ID for session management")


class ChatResponse(BaseModel):
    """Chat response model for the API."""
    
    message: ChatMessage = Field(..., description="Response message from the assistant")
    trip_plan: Optional[Trip] = Field(None, description="Generated trip plan if available")


# Global state
sessions: Dict[str, Dict[str, Any]] = {}


# Setup LangSmith tracing if API key is available
LANGSMITH_PROJECT = "trip-agent"
ENABLE_TRACING = os.getenv("LANGSMITH_API_KEY") is not None


# Dependency to get or create agents
async def get_agents() -> Dict[str, Any]:
    """Get or create the agent instances.
    
    Returns:
        Dict[str, Any]: Dictionary containing agent instances.
    """
    # Import here to avoid circular imports
    from langchain_openai import ChatOpenAI
    
    # Create language model
    llm = ChatOpenAI(
        model_name=os.getenv("OPENAI_MODEL_NAME", "o4-mini"),
        temperature=0.2
    )
    
    # Create agents
    user_interface_agent = UserInterfaceAgent(llm=llm)
    destination_report_agent = DestinationReportAgent(llm=llm)
    attraction_extraction_agent = AttractionExtractionAgent(llm=llm)
    trip_planning_agent = TripPlanningReactAgent(llm=llm)
    
    return {
        "user_interface": user_interface_agent,
        "destination_report": destination_report_agent,
        "attraction_extraction": attraction_extraction_agent,
        "trip_planning": trip_planning_agent
    }


# Dependency to get or create a session
def get_session(user_id: str) -> Dict[str, Any]:
    """Get or create a session for the user.
    
    Args:
        user_id: User ID for session management.
        
    Returns:
        Dict[str, Any]: The session state.
    """
    if user_id not in sessions:
        sessions[user_id] = {
            "messages": [],
            "user_preferences": None,
            "destination_name": None,
            "destination_report": None,
            "attractions": None,
            "trip_request": None,
            "trip_plan": None
        }
    
    return sessions[user_id]


# API endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    agents: Dict[str, Any] = Depends(get_agents)
) -> Dict[str, Any]:
    """Process a chat message and return a response.
    
    Args:
        request: Chat request containing messages and user ID.
        background_tasks: FastAPI background tasks.
        agents: Dictionary of agent instances.
        
    Returns:
        Dict: Response containing the assistant's message and optional trip plan.
    """
    # Get or create a session for the user
    user_id = request.user_id or "default_user"
    session = get_session(user_id)
    
    # Convert API messages to LangChain messages
    from langchain_core.messages import HumanMessage, AIMessage
    
    lc_messages = []
    for msg in request.messages:
        if msg.role == "user":
            lc_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            lc_messages.append(AIMessage(content=msg.content))
    
    # Update the session with the new messages
    session["messages"] = lc_messages
    
    # Setup LangSmith tracing if enabled
    callbacks = None
    if ENABLE_TRACING:
        run_id = str(uuid.uuid4())
        tracer = LangChainTracer(
            project_name=LANGSMITH_PROJECT,
            run_id=run_id
        )
        callback_manager = CallbackManager([tracer])
        callbacks = callback_manager
    
    # Process the user input with the user interface agent
    result = await agents["user_interface"].process(
        lc_messages[-1].content if lc_messages else "",
        session.get("user_preferences")
    )
    
    # Update the session with the preferences
    if "preferences" in result and result["preferences"]:
        session["user_preferences"] = result["preferences"]
        
        # Check if we have enough information to start trip planning
        if (hasattr(result["preferences"], "destination") and 
            result["preferences"].destination and 
            hasattr(result["preferences"], "start_date") and 
            result["preferences"].start_date and
            hasattr(result["preferences"], "end_date") and 
            result["preferences"].end_date):
            
            # Start background task to generate the trip plan
            background_tasks.add_task(
                generate_trip_plan_background,
                agents,
                session,
                callbacks
            )
    
    # Create response
    response = {
        "message": ChatMessage(role="assistant", content=result["response"]),
        "trip_plan": session.get("trip_plan")
    }
    
    return response


@app.get("/trip_plan/{user_id}", response_model=Optional[Trip])
async def get_trip_plan(user_id: str) -> Optional[Trip]:
    """Get the generated trip plan for a user.
    
    Args:
        user_id: User ID for session management.
        
    Returns:
        Optional[Trip]: The generated trip plan if available.
    """
    session = get_session(user_id)
    return session.get("trip_plan")


# Background task to generate the trip plan
async def generate_trip_plan_background(
    agents: Dict[str, Any],
    session: Dict[str, Any],
    callbacks: Optional[Callbacks] = None
) -> None:
    """Generate a trip plan in the background using the sequential flow.
    
    Args:
        agents: Dictionary of agent instances.
        session: Session state.
        callbacks: Optional callbacks for LangSmith tracing.
    """
    preferences = session["user_preferences"]
    destination_name = preferences.destination
    start_date = preferences.start_date
    end_date = preferences.end_date
    
    # Step 1: Generate destination report
    session["destination_name"] = destination_name
    destination_report = await agents["destination_report"].process(
        destination_name=destination_name,
        user_preferences=preferences,
        callbacks=callbacks
    )
    session["destination_report"] = destination_report["report"]
    
    with open(f"{destination_name}-{start_date}-{end_date}.md", "w") as f:
        f.write(destination_report["report"])
    
    # Step 2: Extract attractions from the destination report
    attractions = await agents["attraction_extraction"].process(
        report_content=destination_report["report"],
        destination_name=destination_name,
        callbacks=callbacks
    )
    session["attractions"] = attractions
    
    # Step 3: Generate trip plan using the trip planning agent
    excluded_categories = []
    if hasattr(preferences, "excluded_categories"):
        excluded_categories = preferences.excluded_categories
    
    trip_plan = await agents["trip_planning"].process(
        destination_name=destination_name,
        attractions=attractions,
        start_date=start_date,
        end_date=end_date,
        preferences=preferences.dict() if hasattr(preferences, "dict") else {},
        excluded_categories=excluded_categories,
        destination_report=destination_report["report"],
        callbacks=callbacks
    )

    with open(f"trip-plan-{destination_name}-{start_date}-{end_date}.md", "w") as f:
        f.write(trip_plan)
    
    # Update the session with the trip plan
    session["trip_plan"] = trip_plan


# Run the application
if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("PORT", "8000"))
    
    # Run the application
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=port, reload=True)