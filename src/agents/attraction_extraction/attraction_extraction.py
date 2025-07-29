"""Attraction Extraction Agent implementation for the Trip Agent system.

This agent extracts attractions from destination reports and enriches them with
additional information using web search.
"""

import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from pydantic import Field

from src.agents.base import BaseAgent
from src.agents.attraction_extraction.models import (
    AttractionCandidate,
    AttractionExtractionState
)
from src.models.trip import Attraction, Location
from src.utils.distance_calculator import calculate_attraction_distances


class AttractionExtractionAgent(BaseAgent):
    """Agent responsible for extracting and enriching attraction information.
    
    This agent analyzes destination reports to extract attractions and then uses
    web search to enrich the information about each attraction, including details
    like opening hours, visit duration, etc.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[Any] = None,
        **data
    ):
        """Initialize the Attraction Extraction Agent.
        
        Args:
            llm: Language model to use for this agent.
            tools: Optional list of tools available to this agent.
            memory: Optional memory system for this agent.
            **data: Additional data for the agent.
        """
        super().__init__(
            name="Attraction Extraction Agent",
            description=(
                "an agent that extracts attractions from destination reports and enriches "
                "them with additional information using web search. You analyze text to identify "
                "attractions like museums, landmarks, restaurants, and events, then gather "
                "detailed information about each one."
            ),
            llm=llm,
            tools=tools or [],
            memory=memory,
            **data
        )
        
        # Store the LLM for later use in node functions
        self.llm = llm
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent.
        
        Returns:
            str: The system prompt for this agent.
        """
        base_prompt = super().get_system_prompt()
        additional_instructions = (
            "You are an attraction extraction agent that identifies and enriches information "
            "about points of interest from destination reports. Your goal is to create a "
            "comprehensive database of attractions with detailed, accurate information "
            "to help travelers plan their trips.\n\n"
            "You carefully analyze text to identify attractions like museums, landmarks, "
            "historical sites, restaurants, cafes, bars, parks, theaters, and events. "
            "For each attraction, you gather information about its location, description, "
            "category, opening hours, visit duration, price, and ratings."
        )
        return f"{base_prompt}\n\n{additional_instructions}"
    
    async def process(
        self, 
        report_content: str, 
        destination_name: str, 
        callbacks: Optional[List[Any]] = None
    ) -> List[Attraction]:
        """Extract and enrich attractions from a destination report.
        
        Args:
            report_content: The content of the destination report.
            destination_name: The name of the destination.
            callbacks: Optional callbacks for the agent.
            
        Returns:
            List[Attraction]: A list of attractions with enriched information.
        """
        # Create and run the attraction extraction graph
        graph = self._create_extraction_graph()
        
        # Initialize the state
        state = AttractionExtractionState(
            destination_name=destination_name,
            report_content=report_content,
            messages=[],
            current_attraction_index=0
        )
        
        # Run the graph
        thread = {"configurable": {"thread_id": f"{destination_name}_report"}, "recursion_limit": 100}
        result = graph.invoke(state, thread)
        
        # Get the enriched attractions
        attractions = result["enriched_attractions"]
        
        # Calculate walking distances and times between attractions
        try:
            attractions = await calculate_attraction_distances(destination_name, attractions)
        except Exception as e:
            print(f"Warning: Failed to calculate distances between attractions: {str(e)}")
        
        # Return the enriched attractions with distance information
        return attractions
    
    def _create_extraction_graph(self) -> StateGraph:
        """Create the LangGraph workflow for attraction extraction and enrichment.
        
        Returns:
            StateGraph: The compiled graph for attraction extraction.
        """
        # Create the graph builder
        builder = StateGraph(AttractionExtractionState)
        
        # Define node functions
        def extract_attractions(state: AttractionExtractionState) -> Dict[str, Any]:
            """Extract attractions from the destination report."""
            destination_name = state["destination_name"]
            report_content = state["report_content"]
            
            prompt = f"""Extract all attractions mentioned in the following destination report for {destination_name}.
            
            Focus on:
            - Museums, galleries, and cultural sites
            - Historical landmarks and monuments
            - Parks, gardens, and natural attractions
            - Restaurants, cafes, and food places
            - Bars, pubs, and nightlife venues
            - Theaters and performance venues
            - Festivals and events
            
            For each attraction, identify:
            1. Name (required)
            2. Category (required) - e.g., museum, restaurant, park, etc.
            3. Brief description (if available)
            4. Location name (if mentioned)
            
            REPORT CONTENT:
            {report_content}
            
            Return the results as a JSON list of objects with the following structure:
            [
                {{
                    "name": "Attraction Name",
                    "category": "Category",
                    "description": "Brief description if available",
                    "location_name": "Location name if mentioned",
                    "extracted_from": "Section name where this was found"
                }}
            ]
            
            Only include attractions that are clearly mentioned in the report. Do not make up or infer attractions that aren't explicitly mentioned.
            """
            
            extraction_result = self.llm.invoke([
                SystemMessage(content="You are an AI assistant that specializes in extracting structured information about attractions from travel reports."),
                HumanMessage(content=prompt)
            ])
            
            # Extract JSON from the response
            json_match = re.search(r'```json\n(.*?)\n```', extraction_result.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without markdown formatting
                json_match = re.search(r'\[\s*\{.*\}\s*\]', extraction_result.content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = extraction_result.content
            
            import json
            try:
                attractions_data = json.loads(json_str)
                extracted_attractions = [AttractionCandidate(**attraction) for attraction in attractions_data]
            except (json.JSONDecodeError, TypeError):
                # Fallback to a more lenient approach if JSON parsing fails
                extracted_attractions = []
                
            return {"extracted_attractions": extracted_attractions}
        
        def should_enrich_next_attraction(state: AttractionExtractionState) -> str:
            """Determine if there are more attractions to enrich."""
            extracted_attractions = state["extracted_attractions"]
            current_index = state["current_attraction_index"]
            
            if current_index < len(extracted_attractions):
                return "enrich_attraction"
            else:
                return "finalize"
        
        def enrich_attraction(state: AttractionExtractionState) -> Dict[str, Any]:
            """Enrich the current attraction with additional information from web search."""
            extracted_attractions = state["extracted_attractions"]
            current_index = state["current_attraction_index"]
            destination_name = state["destination_name"]
            
            if current_index >= len(extracted_attractions):
                return {}
            
            current_attraction = extracted_attractions[current_index]
            tavily_search = TavilySearchResults(max_results=2)
    
            # Perform web search for the attraction
            search_query = f"{current_attraction.name} {destination_name} attraction information opening hours visit duration"
            search_results = tavily_search.invoke(search_query)
            print(search_query)
            print(search_results)
            # Create prompt for enriching the attraction
            prompt = f"""Enrich the following attraction information using the search results provided:
            
            ATTRACTION:
            - Name: {current_attraction.name}
            - Category: {current_attraction.category}
            - Description: {current_attraction.description or "Not available"}
            - Location: {current_attraction.location_name or "Not available"}
            - Date Range: {current_attraction.date_range or "Not available"}
            - Destination: {destination_name}
            
            SEARCH RESULTS:
            {search_results}
            
            Based on the search results, provide the following information in JSON format:
            1. Full description of the attraction
            2. Precise location (address if available)
            3. Visit duration in minutes (average time spent). Can also be a time range. Convert to string.
            4. Opening hours by day of week (if available). If a specific day is not mentioned in the opening hours, fill with "Closed".
            Prefer precise opening hours over descriptions like "till sunsets" or "all day". If the search_results provide different
            opening hours for different seasons, provide opening hours for the current season.
            5. Price of admission (if applicable). Convert to string.
            6. Rating out of 5 (if available)
            7. Date range when the attraction is available (especially for festivals and events). If not specified, leave as null.
            
            Return ONLY the JSON object with this structure:
            {{
                "description": "Detailed description",
                "address": "Full address",
                "visit_duration": "120",
                "opening_hours": {{"Monday": "9:00 AM - 5:00 PM", "Tuesday": "9:00 AM - 5:00 PM", ...}},
                "price": "15.50",
                "rating": 4.5,
                "date_range": "YYYY-MM-DD - YYYY-MM-DD"
            }}
            
            If you cannot find specific information, use null for that field. Make your best estimate for visit_duration based on the type of attraction if not specified.
            For festivals and events, it's especially important to identify the specific dates or date range when they occur.
            """
            
            enrichment_result = self.llm.invoke([
                SystemMessage(content="You are an AI assistant that specializes in enriching information about attractions using search results."),
                HumanMessage(content=prompt)
            ])
            
            # Extract JSON from the response
            json_match = re.search(r'```json\n(.*?)\n```', enrichment_result.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without markdown formatting
                json_match = re.search(r'\{.*\}', enrichment_result.content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = enrichment_result.content
            
            import json
            try:
                enriched_data = json.loads(json_str)
                print(enriched_data)
                
                # Create location object
                location = Location(
                    name=current_attraction.location_name or current_attraction.name,
                    address=enriched_data.get("address"),
                )
                
                # Clean opening_hours to ensure no None values in the dictionary
                opening_hours = enriched_data.get("opening_hours")
                if opening_hours and isinstance(opening_hours, dict):
                    # Replace None values with "Missing"
                    opening_hours = {day: hours if hours is not None else "Missing" for day, hours in opening_hours.items()}
                
                # Create attraction object
                attraction = Attraction(
                    name=current_attraction.name,
                    description=enriched_data.get("description", current_attraction.description or ""),
                    location=location,
                    category=current_attraction.category,
                    visit_duration=str(enriched_data.get("visit_duration") or "60"),  # Ensure it's always a string
                    opening_hours=opening_hours,
                    price=enriched_data.get("price"),
                    rating=enriched_data.get("rating"),
                    date_range=enriched_data.get("date_range")
                )
                
                # Add to enriched attractions list
                enriched_attractions = state.get("enriched_attractions", [])
                enriched_attractions.append(attraction)
                
                return {
                    "current_attraction_index": current_index + 1,
                    "enriched_attractions": enriched_attractions,
                    "search_results": search_results
                }
                
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                # If parsing fails, create a basic attraction with minimal information
                location = Location(
                    name=current_attraction.location_name or current_attraction.name,
                    address=None
                )
                
                # Create a dictionary for opening_hours with "Missing" values for each day
                opening_hours = {
                    "Monday": "Missing",
                    "Tuesday": "Missing",
                    "Wednesday": "Missing",
                    "Thursday": "Missing",
                    "Friday": "Missing",
                    "Saturday": "Missing",
                    "Sunday": "Missing"
                }
                
                attraction = Attraction(
                    name=current_attraction.name,
                    description=current_attraction.description or "",
                    location=location,
                    category=current_attraction.category,
                    visit_duration="60",  # Default to 60 minutes
                    opening_hours=opening_hours,
                    price=None,
                    rating=None,
                    date_range=None
                )
                
                # Add to enriched attractions list
                enriched_attractions = state.get("enriched_attractions", [])
                enriched_attractions.append(attraction)
                
                return {
                    "current_attraction_index": current_index + 1,
                    "enriched_attractions": enriched_attractions,
                    "search_results": search_results
                }
        
        def finalize_attractions(state: AttractionExtractionState) -> Dict[str, Any]:
            """Finalize the list of attractions."""
            return {"enriched_attractions": state["enriched_attractions"]}
        
        # Add nodes to the graph
        builder.add_node("extract_attractions", extract_attractions)
        builder.add_node("enrich_attraction", enrich_attraction)
        builder.add_node("finalize", finalize_attractions)
        
        # Define edges
        builder.add_edge(START, "extract_attractions")
        builder.add_conditional_edges(
            "extract_attractions",
            should_enrich_next_attraction,
            {
                "enrich_attraction": "enrich_attraction",
                "finalize": "finalize"
            }
        )
        builder.add_conditional_edges(
            "enrich_attraction",
            should_enrich_next_attraction,
            {
                "enrich_attraction": "enrich_attraction",
                "finalize": "finalize"
            }
        )
        builder.add_edge("finalize", END)
        
        # Compile the graph
        memory = MemorySaver()
        return builder.compile(checkpointer=memory)