#!/usr/bin/env python3
"""
Test script for the distance calculator functionality.

This script loads attractions from extracted_attractions.json,
sets the destination name to Copenhagen, and calculates walking
distances and travel times between attractions using Google Maps API.
"""

import os
import json
import asyncio
from dotenv import load_dotenv
from pprint import pprint

from src.models.trip import Attraction
from src.utils.distance_calculator import calculate_attraction_distances


async def main():
    """Main test function."""
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Check if GOOGLE_MAPS_API_KEY is set
    google_maps_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not google_maps_key:
        print("ERROR: GOOGLE_MAPS_API_KEY environment variable is not set.")
        print("Please set it in your environment or in a .env file.")
        return
    
    print(f"GOOGLE_MAPS_API_KEY found: {google_maps_key[:5]}...{google_maps_key[-5:]}")
    
    # Set destination name
    destination_name = "Copenhagen"
    print(f"Testing distance calculator for destination: {destination_name}")
    
    # Load attractions from JSON file
    try:
        with open("extracted_attractions.json", "r") as f:
            attractions_data = json.load(f)
        
        print(f"Loaded {len(attractions_data)} attractions from extracted_attractions.json")
    except Exception as e:
        print(f"Error loading attractions from JSON file: {e}")
        return
    
    # Convert JSON data to Attraction objects
    attractions = []
    for attraction_data in attractions_data:
        try:
            attraction = Attraction.model_validate(attraction_data)
            attractions.append(attraction)
        except Exception as e:
            print(f"Error parsing attraction data: {e}")
            print(f"Problematic data: {attraction_data}")
    
    print(f"Successfully parsed {len(attractions)} attractions")
    
    # Limit to a smaller number of attractions for testing if needed
    # Uncomment the following line to test with fewer attractions
    # attractions = attractions[:5]  # Start with just 5 attractions for testing
    
    # Calculate distances between attractions
    print("\nCalculating distances between attractions...")
    try:
        attractions_with_distances = await calculate_attraction_distances(
            destination_name, 
            attractions,
            google_maps_key
        )
        
        print(f"\nSuccessfully calculated distances for {len(attractions_with_distances)} attractions")
        
        # Print summary of results
        print("\nDistance Summary:")
        for attraction in attractions_with_distances:
            if not attraction.travel_info:
                print(f"- {attraction.name}: No travel info available")
                continue
                
            print(f"- {attraction.name}:")
            closest = None
            closest_distance = float('inf')
            farthest = None
            farthest_distance = 0
            
            for other_name, info in attraction.travel_info.items():
                distance = info['distance']
                if distance < closest_distance and distance > 0:
                    closest_distance = distance
                    closest = other_name
                if distance > farthest_distance:
                    farthest_distance = distance
                    farthest = other_name
            
            if closest:
                print(f"  - Closest: {closest} ({closest_distance}m, {attraction.travel_info[closest]['time']}min)")
            if farthest:
                print(f"  - Farthest: {farthest} ({farthest_distance}m, {attraction.travel_info[farthest]['time']}min)")
        
        # Save the results to a file
        output_file = "attractions_with_distances.json"
        print(f"\nSaving results to {output_file}...")
        
        # Convert to JSON-serializable format
        attractions_json = [attraction.model_dump() for attraction in attractions_with_distances]
        
        with open(output_file, "w") as f:
            json.dump(attractions_json, f, indent=2)
        
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error calculating distances: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
