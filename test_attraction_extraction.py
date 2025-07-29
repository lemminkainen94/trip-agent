"""
Test script for the Attraction Extraction Agent.

This script loads the Copenhagen test report and extracts attractions
using the AttractionExtractionAgent.
"""

import asyncio
import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from src.agents.attraction_extraction.attraction_extraction import AttractionExtractionAgent
from src.models.trip import Attraction

load_dotenv()

async def test_attraction_extraction():
    """
    Test the attraction extraction functionality by loading the Copenhagen test report
    and extracting attractions from it.

    Returns:
        List[Attraction]: List of extracted and enriched attractions.
    """
    # Load the test report
    report_path = Path("cph_test_report.md")
    if not report_path.exists():
        raise FileNotFoundError(f"Test report not found at {report_path}")
    
    report_content = report_path.read_text(encoding="utf-8")
    destination_name = "Copenhagen"
    
    # Initialize the language model
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )
    
    # Initialize the search tool
    search_tool = TavilySearchResults(max_results=1)
    
    # Initialize the attraction extraction agent
    agent = AttractionExtractionAgent(
        llm=llm,
        tools=[search_tool]
    )
    
    # Process the report to extract attractions
    print(f"Extracting attractions from {destination_name} report...")
    attractions = await agent.process(
        report_content=report_content,
        destination_name=destination_name
    )
    
    # Print the results
    print(f"\nExtracted {len(attractions)} attractions:")
    for i, attraction in enumerate(attractions, 1):
        print(f"\n{i}. {attraction.name} ({attraction.category})")
        print(f"   Description: {attraction.description[:100]}..." if attraction.description else "   No description available")
        print(f"   Location: {attraction.location.name}")
        if attraction.opening_hours:
            print(f"   Opening Hours: {json.dumps(attraction.opening_hours, indent=2)}")
        if attraction.price is not None:
            print(f"   Price: {attraction.price}")
        if attraction.rating is not None:
            print(f"   Rating: {attraction.rating}/5")
        print(f"   Visit Duration: {attraction.visit_duration} minutes")
        
        # Print travel info if available
        if attraction.travel_info and len(attraction.travel_info) > 0:
            print(f"   Walking distances to other attractions:")
            for other_name, info in attraction.travel_info.items():
                print(f"     - To {other_name}: {info['distance']} meters, {info['time']} minutes")
    
    return attractions


if __name__ == "__main__":
    # Run the test
    attractions = asyncio.run(test_attraction_extraction())
    
    # Save results to a JSON file for further analysis
    output_path = Path("extracted_attractions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        # Convert attractions to dict for JSON serialization
        attractions_data = [attraction.model_dump() for attraction in attractions]
        json.dump(attractions_data, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")