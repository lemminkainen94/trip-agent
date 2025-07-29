#!/usr/bin/env python
"""
Test script for the Destination Research Assistant.

This script demonstrates how to use the DestinationReportAgent in isolation
to generate comprehensive destination reports.
"""

import os
import asyncio
import json
import uuid
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langsmith import traceable
from langchain_core.tracers import LangChainTracer
from langchain_core.tracers.langchain import wait_for_all_tracers

from src.agents.destination_research_assistant.destination_report import DestinationReportAgent
from src.models.preferences import UserPreferences

load_dotenv()

# Configure LangSmith if API key is available
LANGSMITH_ENABLED = bool(os.getenv("LANGSMITH_API_KEY"))
if LANGSMITH_ENABLED:
    # Enable LangChain tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    # Configure LangSmith API endpoint
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    # Set LangChain API key
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    print("LangSmith tracing enabled")


@traceable(run_type="chain", name="DestinationResearchTest")
async def test_destination_research(
    destination_name: str,
    user_preferences: Optional[UserPreferences] = None,
    output_file: Optional[str] = None,
    project_name: Optional[str] = None
) -> None:
    """
    Test the Destination Research Assistant by generating a report for a destination.
    
    Args:
        destination_name: Name of the destination to research.
        user_preferences: Optional user preferences to consider.
        output_file: Optional file path to save the report as JSON.
    """
    print(f"Initializing Destination Research Assistant for: {destination_name}")
    
    # Set up LangSmith tracing callbacks
    callbacks = []
    run_id = None
    
    if LANGSMITH_ENABLED:
        # Create a unique run ID for this research session
        run_id = str(uuid.uuid4())
        project = project_name or f"destination-research-{destination_name.lower().replace(' ', '-')}"
        
        # Set the project name in environment variables
        os.environ["LANGCHAIN_PROJECT"] = project
        
        # Create a LangChain tracer with the project name
        tracer = LangChainTracer(project_name=project)
        callbacks.append(tracer)
        print(f"LangSmith project: {project}")
    
    # Create language model with tracing
    llm = ChatOpenAI(
        model_name=os.getenv("OPENAI_MODEL_NAME", "o4-mini"),
        #temperature=0.2,
        #callbacks=callbacks
    )

    # Create the Destination Report Agent with tracing
    destination_agent = DestinationReportAgent(llm=llm, callbacks=callbacks)
    
    print(f"Starting research process for {destination_name}...")
    print("This may take a few minutes as the agent researches various aspects of the destination.")
    
    # Process the destination
    start_time = datetime.now()
    report_data = destination_agent.process(
        destination_name=destination_name, 
        user_preferences=user_preferences,
        callbacks=callbacks
    )
    end_time = datetime.now()
    
    # Calculate processing time
    processing_time = (end_time - start_time).total_seconds()
    
    print(f"\nResearch completed in {processing_time:.2f} seconds!")
    print(f"\n{'=' * 80}")
    print(f"DESTINATION REPORT: {destination_name.upper()}")
    print(f"{'=' * 80}")
    
    # Print the full report
    print("\nFULL REPORT:")
    print("-" * 50)
    print(report_data["report"])

    # sections
    print("\nREPORT SECTIONS:")
    print("-" * 50)
    for section in report_data["sections"]:
        print(f"\n{section}")

    with open(output_file, "w") as f:
        f.write(report_data["report"])
    # Wait for all LangSmith traces to be submitted
    if LANGSMITH_ENABLED:
        print("Waiting for LangSmith traces to be submitted...")
        try:
            wait_for_all_tracers()
        except AttributeError:
            # Handle the case when the flush method is not available
            import time
            # Give some time for traces to be submitted asynchronously
            time.sleep(5)
        print(f"All traces submitted to LangSmith project: {project_name or f'destination-research-{destination_name.lower().replace(' ', '-')}'}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Destination Research Assistant")
    parser.add_argument("destination", help="Name of the destination to research")
    parser.add_argument("--output", "-o", help="Output file path to save the report as JSON")
    parser.add_argument("--project", "-p", help="LangSmith project name for tracing (requires LANGSMITH_API_KEY)")
    parser.add_argument("--disable-langsmith", action="store_true", help="Disable LangSmith tracing even if API key is available")
    
    args = parser.parse_args()
    
    # Override LangSmith settings if requested
    if args.disable_langsmith:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        print("LangSmith tracing disabled by command line argument")
    
    # Run the test
    asyncio.run(test_destination_research(
        destination_name=args.destination, 
        output_file=args.output,
        project_name=args.project
    ))