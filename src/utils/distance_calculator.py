"""Utility for calculating distances and travel times between attractions.

This module provides functionality to calculate walking distances and travel times
between attractions using the Google Maps API.
"""

import os
from typing import Dict, Tuple, Optional, List
import aiohttp
import json
import urllib.parse

from src.models.trip import Attraction


async def calculate_walking_distance(
    origin_coords: Tuple[float, float], 
    destination_coords: Tuple[float, float],
    api_key: str
) -> Dict[str, float]:
    """Calculate walking distance and time between two coordinate points using Google Maps API.
    
    Args:
        origin_coords: Tuple of (longitude, latitude) for the origin point
        destination_coords: Tuple of (longitude, latitude) for the destination point
        api_key: Google Maps API key
        
    Returns:
        Dict containing distance (meters) and time (minutes) for walking between points
    """
    # Google Maps uses lat,lng format (opposite of Mapbox)
    origin = f"{origin_coords[1]},{origin_coords[0]}"
    destination = f"{destination_coords[1]},{destination_coords[0]}"
    
    # Google Maps Directions API endpoint for walking directions
    url = (
        f"https://maps.googleapis.com/maps/api/directions/json"
        f"?origin={origin}"
        f"&destination={destination}"
        f"&mode=walking"
        f"&key={api_key}"
    )
    
    print(f"[DEBUG] Requesting walking directions: {origin} -> {destination}")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                print(f"[ERROR] Directions API failed with status {response.status}")
                response_text = await response.text()
                print(f"[ERROR] Response: {response_text[:200]}...")
                # Return default values if API call fails
                return {"distance": 0, "time": 0}
            
            data = await response.json()
            print(f"[DEBUG] Directions API response received, status: {response.status}")
            
            # Extract distance (meters) and duration (seconds) from response
            if data.get("status") == "OK" and "routes" in data and data["routes"]:
                route = data["routes"][0]
                if "legs" in route and route["legs"]:
                    leg = route["legs"][0]
                    distance = leg.get("distance", {}).get("value", 0)  # meters
                    duration = leg.get("duration", {}).get("value", 0) / 60  # convert seconds to minutes
                    
                    result = {
                        "distance": round(distance),  # round to nearest meter
                        "time": round(duration, 1)  # round to 1 decimal place
                    }
                    print(f"[DEBUG] Calculated distance: {result['distance']}m, time: {result['time']}min")
                    return result
            else:
                print(f"[ERROR] No routes found in response: {json.dumps(data)[:200]}...")
            
            return {"distance": 0, "time": 0}


async def get_coordinates_from_address(address: str, api_key: str) -> Optional[Tuple[float, float]]:
    """Get coordinates (longitude, latitude) from an address using Google Maps Geocoding API.
    
    Args:
        address: Address string to geocode
        api_key: Google Maps API key
        
    Returns:
        Tuple of (longitude, latitude) if successful, None otherwise
    """
    # URL encode the address
    encoded_address = urllib.parse.quote(address)
    
    # Google Maps Geocoding API endpoint
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={encoded_address}&key={api_key}"
    
    print(f"[DEBUG] Geocoding address: '{address}'")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                print(f"[ERROR] Geocoding API failed with status {response.status}")
                response_text = await response.text()
                print(f"[ERROR] Response: {response_text[:200]}...")
                return None
            
            data = await response.json()
            print(f"[DEBUG] Geocoding API response received, status: {response.status}")
            
            # Extract coordinates from the first result
            if data.get("status") == "OK" and "results" in data and data["results"]:
                location = data["results"][0]["geometry"]["location"]
                # Google Maps returns lat/lng, but we want lng/lat to be consistent with previous code
                coordinates = (location["lng"], location["lat"])
                print(f"[DEBUG] Coordinates found: {coordinates}")
                return coordinates
            else:
                print(f"[ERROR] No results found for address '{address}'")
                print(f"[ERROR] Response: {json.dumps(data)[:200]}...")
            
            return None


async def calculate_attraction_distances(
    destination_name: str,
    attractions: List[Attraction], 
    api_key: Optional[str] = None
) -> List[Attraction]:
    """Calculate walking distances and times between all pairs of attractions.
    
    Args:
        destination_name: Name of the destination (city, region) for geocoding context
        attractions: List of attractions to calculate distances between
        api_key: Optional Google Maps API key (if not provided, tries to get from environment)
        
    Returns:
        List of attractions with travel_info field populated
    """
    print(f"[INFO] Calculating distances for {len(attractions)} attractions in {destination_name}")
    
    # Get API key from environment if not provided
    if not api_key:
        api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
        if not api_key:
            print("[ERROR] Google Maps API key not found in environment variables")
            raise ValueError(
                "Google Maps API key not provided. Set GOOGLE_MAPS_API_KEY environment variable "
                "or pass api_key parameter."
            )
        else:
            print("[INFO] Using Google Maps API key from environment variables")
    
    # Create a copy of attractions to avoid modifying the originals
    attractions_with_travel_info = []
    
    # Get coordinates for each attraction
    attraction_coords = {}
    
    # First, get coordinates for all attractions
    print("[INFO] Getting coordinates for all attractions...")
    for attraction in attractions:
        print(f"[INFO] Processing attraction: {attraction.name}")
        coords = None
        
        # Try to get coordinates from attraction name with destination context first
        search_query = f"{destination_name}, {attraction.name}"
        print(f"[DEBUG] Trying with destination context: {search_query}")
        coords = await get_coordinates_from_address(search_query, api_key)
        
        # If that fails, try with location address
        if not coords and attraction.location and attraction.location.address:
            print(f"[DEBUG] Trying address: {attraction.location.address}")
            coords = await get_coordinates_from_address(attraction.location.address, api_key)
            
        # If that fails too, try with just the attraction name
        if not coords:
            print(f"[DEBUG] Trying just attraction name: {attraction.name}")
            coords = await get_coordinates_from_address(attraction.name, api_key)

        if coords:
            print(f"[INFO] Found coordinates for {attraction.name}: {coords}")
            attraction_coords[attraction.name] = coords
        else:
            print(f"[WARNING] Could not find coordinates for {attraction.name}")
            
        # Create a copy with empty travel_info to be filled later
        attraction_copy = attraction.model_copy()
        attraction_copy.travel_info = {}
        attractions_with_travel_info.append(attraction_copy)
    
    print(f"[INFO] Found coordinates for {len(attraction_coords)} out of {len(attractions)} attractions")
    
    # Calculate distances between all pairs of attractions
    print("[INFO] Calculating distances between attractions...")
    for i, attraction in enumerate(attractions_with_travel_info):
        # Skip attractions without coordinates
        if attraction.name not in attraction_coords:
            print(f"[INFO] Skipping {attraction.name} - no coordinates available")
            continue
        
        origin_coords = attraction_coords[attraction.name]
        travel_info = {}
        
        # Calculate distance to all other attractions
        for other_attraction in attractions_with_travel_info:
            # Skip self or attractions without coordinates
            if other_attraction.name == attraction.name or other_attraction.name not in attraction_coords:
                continue
            
            destination_coords = attraction_coords[other_attraction.name]
            
            # Calculate distance and time
            print(f"[DEBUG] Calculating distance: {attraction.name} -> {other_attraction.name}")
            result = await calculate_walking_distance(origin_coords, destination_coords, api_key)
            
            # Add to travel_info dictionary
            travel_info[other_attraction.name] = result
        
        # Update the attraction's travel_info
        print(f"[INFO] Added travel info for {attraction.name} to {len(travel_info)} other attractions")
        attractions_with_travel_info[i].travel_info = travel_info
    
    return attractions_with_travel_info