"""
Tools for the ReAct-based Trip Planning Agent.

This module provides validation tools and helper functions for the
TripPlanningReactAgent to check opening hours, date availability,
and validate itineraries.
"""

import re
from datetime import datetime, time
from typing import Any, Dict, Optional


def is_attraction_open_at_time(attraction, date, time_str):
    """
    Check if an attraction is open at a specific time on a specific date.
    
    Args:
        attraction: The attraction to check
        date: The date to check
        time_str: The time to check in format "HH:MM"
        
    Returns:
        bool: True if the attraction is open, False otherwise
    """
    if not attraction.opening_hours:
        # If no opening hours are specified, assume it's open
        return True
    
    # Get the day of the week
    day_of_week = date.strftime("%A")
    
    # Check if the attraction has opening hours for this day
    if day_of_week not in attraction.opening_hours:
        # If no specific day is listed, look for a default entry
        if "default" in attraction.opening_hours:
            day_of_week = "default"
        else:
            # No opening hours for this day, assume closed
            return False
    
    opening_hours = attraction.opening_hours[day_of_week]
    
    # If explicitly marked as closed
    if opening_hours.lower() == "closed":
        return False
    
    # Parse the time string to a datetime.time object
    try:
        hour, minute = map(int, time_str.split(":"))
        check_time = time(hour, minute)
    except (ValueError, TypeError):
        # If we can't parse the time, assume it's open
        print(f"Warning: Could not parse time {time_str} for {attraction.name}")
        return True
    
    # Try to parse the opening hours
    try:
        # Handle multiple time ranges (e.g., "10:00-13:00, 14:00-18:00")
        time_ranges = opening_hours.split(",")
        
        for time_range in time_ranges:
            time_range = time_range.strip()
            
            # Handle 24-hour format (e.g., "09:00-17:00")
            if re.match(r'^\\d{1,2}:\\d{2}-\\d{1,2}:\\d{2}$', time_range):
                start_str, end_str = time_range.split("-")
                start_hour, start_minute = map(int, start_str.split(":"))
                end_hour, end_minute = map(int, end_str.split(":"))
                
                start_time = time(start_hour, start_minute)
                end_time = time(end_hour, end_minute)
                
                if start_time <= check_time <= end_time:
                    return True
            
            # Handle AM/PM format (e.g., "9:00 AM - 5:00 PM")
            elif re.match(r'^\\d{1,2}:\\d{2} [AP]M - \\d{1,2}:\\d{2} [AP]M$', time_range, re.IGNORECASE):
                start_str, end_str = time_range.split(" - ")
                
                # Parse start time
                start_time_parts = start_str.split(" ")
                start_hour, start_minute = map(int, start_time_parts[0].split(":"))
                if start_time_parts[1].upper() == "PM" and start_hour < 12:
                    start_hour += 12
                elif start_time_parts[1].upper() == "AM" and start_hour == 12:
                    start_hour = 0
                
                # Parse end time
                end_time_parts = end_str.split(" ")
                end_hour, end_minute = map(int, end_time_parts[0].split(":"))
                if end_time_parts[1].upper() == "PM" and end_hour < 12:
                    end_hour += 12
                elif end_time_parts[1].upper() == "AM" and end_hour == 12:
                    end_hour = 0
                
                start_time = time(start_hour, start_minute)
                end_time = time(end_hour, end_minute)
                
                if start_time <= check_time <= end_time:
                    return True
        
        # If we've checked all time ranges and none match, the attraction is closed
        return False
    
    except Exception as e:
        # If we can't parse the opening hours, assume it's open
        print(f"Warning: Could not parse opening hours {opening_hours} for {attraction.name}: {e}")
        return True


def is_attraction_available_on_date(attraction, date):
    """
    Check if an attraction is available on a specific date based on its date range.
    
    Args:
        attraction: The attraction to check
        date: The date to check
        
    Returns:
        bool: True if the attraction is available, False otherwise
    """
    if not attraction.date_range:
        # If no date range is specified, assume it's available
        return True
    
    date_range = attraction.date_range
    
    # Try to parse the date range
    try:
        # Handle ranges like "July 1-15, 2025"
        match = re.match(r'([A-Za-z]+)\\s+(\\d{1,2})\\s*-\\s*(\\d{1,2})\\s*,\\s*(\\d{4})', date_range)
        if match:
            month_name, start_day, end_day, year = match.groups()
            start_date = datetime.strptime(f"{month_name} {start_day} {year}", "%B %d %Y")
            end_date = datetime.strptime(f"{month_name} {end_day} {year}", "%B %d %Y")
            return start_date <= date <= end_date
        
        # Handle ranges like "July 1 - August 15, 2025"
        match = re.match(r'([A-Za-z]+)\\s+(\\d{1,2})\\s*-\\s*([A-Za-z]+)\\s+(\\d{1,2})\\s*,\\s*(\\d{4})', date_range)
        if match:
            start_month, start_day, end_month, end_day, year = match.groups()
            start_date = datetime.strptime(f"{start_month} {start_day} {year}", "%B %d %Y")
            end_date = datetime.strptime(f"{end_month} {end_day} {year}", "%B %d %Y")
            return start_date <= date <= end_date
        
        # Handle ranges like "July-August, 2025"
        match = re.match(r'([A-Za-z]+)\\s*-\\s*([A-Za-z]+)\\s*,\\s*(\\d{4})', date_range)
        if match:
            start_month, end_month, year = match.groups()
            start_date = datetime.strptime(f"{start_month} 1 {year}", "%B %d %Y")
            
            # Get the last day of the end month
            if end_month in ["January", "March", "May", "July", "August", "October", "December"]:
                last_day = 31
            elif end_month in ["April", "June", "September", "November"]:
                last_day = 30
            elif end_month == "February":
                # Simple leap year check
                year_num = int(year)
                if year_num % 4 == 0 and (year_num % 100 != 0 or year_num % 400 == 0):
                    last_day = 29
                else:
                    last_day = 28
            else:
                last_day = 30  # Default
            
            end_date = datetime.strptime(f"{end_month} {last_day} {year}", "%B %d %Y")
            return start_date <= date <= end_date
        
        # If we can't parse the date range, assume it's available
        print(f"Warning: Could not parse date range {date_range} for {attraction.name}")
        return True
    
    except Exception as e:
        # If we can't parse the date range, assume it's available
        print(f"Warning: Could not parse date range {date_range} for {attraction.name}: {e}")
        return True


class TripPlanningTools:
    """
    Tools for the ReAct-based Trip Planning Agent.
    
    This class provides validation tools for checking opening hours,
    date availability, and validating itineraries.
    """
    
    def __init__(self, state):
        """
        Initialize the tools with the current state.
        
        Args:
            state: Current trip planning state
        """
        self.state = state
    
    def check_opening_hours(self, attraction_name, date_str, time_str):
        """
        Tool function to check if an attraction is open at a specific time.
        
        Args:
            attraction_name: Name of the attraction
            date_str: Date in format "YYYY-MM-DD"
            time_str: Time in format "HH:MM"
            
        Returns:
            Dict with result of the check
        """
        # Find the attraction by name
        attraction = None
        for a in self.state["attractions"]:
            if a.name == attraction_name:
                attraction = a
                break
        
        if not attraction:
            return {
                "open": False,
                "error": f"Attraction '{attraction_name}' not found"
            }
        
        # Parse the date
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return {
                "open": False,
                "error": f"Invalid date format: {date_str}. Expected YYYY-MM-DD."
            }
        
        # Check if the attraction is open
        is_open = is_attraction_open_at_time(attraction, date, time_str)
        
        # Get the opening hours for the day
        day_of_week = date.strftime("%A")
        opening_hours = "Not specified"
        
        if attraction.opening_hours:
            if day_of_week in attraction.opening_hours:
                opening_hours = attraction.opening_hours[day_of_week]
            elif "default" in attraction.opening_hours:
                opening_hours = attraction.opening_hours["default"]
        
        return {
            "open": is_open,
            "attraction": attraction_name,
            "date": date_str,
            "time": time_str,
            "day_of_week": day_of_week,
            "opening_hours": opening_hours
        }
    
    def check_attraction_availability(self, attraction_name, date_str):
        """
        Tool function to check if an attraction is available on a specific date.
        
        Args:
            attraction_name: Name of the attraction
            date_str: Date in format "YYYY-MM-DD"
            
        Returns:
            Dict with result of the check
        """
        # Find the attraction by name
        attraction = None
        for a in self.state["attractions"]:
            if a.name == attraction_name:
                attraction = a
                break
        
        if not attraction:
            return {
                "available": False,
                "error": f"Attraction '{attraction_name}' not found"
            }
        
        # Parse the date
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return {
                "available": False,
                "error": f"Invalid date format: {date_str}. Expected YYYY-MM-DD."
            }
        
        # Check if the attraction is available
        is_available = is_attraction_available_on_date(attraction, date)
        
        return {
            "available": is_available,
            "attraction": attraction_name,
            "date": date_str,
            "date_range": attraction.date_range or "Not specified"
        }
    
    def validate_itinerary(self, itinerary):
        """
        Tool function to validate the entire itinerary for opening hours and other constraints.
        
        Args:
            itinerary: The itinerary to validate
            
        Returns:
            Dict with validation results
        """
        violations = []
        
        for day in itinerary.get("days", []):
            date_str = day.get("date")
            
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                violations.append({
                    "type": "invalid_date",
                    "message": f"Invalid date format: {date_str}. Expected YYYY-MM-DD."
                })
                continue
            
            # Check if date is within trip range
            if date < self.state["start_date"] or date > self.state["end_date"]:
                violations.append({
                    "type": "date_out_of_range",
                    "message": f"Date {date_str} is outside the trip date range ({self.state['start_date'].strftime('%Y-%m-%d')} to {self.state['end_date'].strftime('%Y-%m-%d')})."
                })
            
            for activity in day.get("activities", []):
                attraction_name = activity.get("attraction_name")
                start_time = activity.get("start_time")
                end_time = activity.get("end_time")
                
                # Find the attraction
                attraction = None
                for a in self.state["attractions"]:
                    if a.name == attraction_name:
                        attraction = a
                        break
                
                if not attraction:
                    violations.append({
                        "type": "unknown_attraction",
                        "date": date_str,
                        "message": f"Unknown attraction: {attraction_name}"
                    })
                    continue
                
                # Check opening hours
                if not is_attraction_open_at_time(attraction, date, start_time):
                    day_of_week = date.strftime("%A")
                    opening_hours = "Not specified"
                    
                    if attraction.opening_hours:
                        if day_of_week in attraction.opening_hours:
                            opening_hours = attraction.opening_hours[day_of_week]
                        elif "default" in attraction.opening_hours:
                            opening_hours = attraction.opening_hours["default"]
                    
                    violations.append({
                        "type": "opening_hours",
                        "date": date_str,
                        "attraction": attraction_name,
                        "scheduled_time": start_time,
                        "opening_hours": opening_hours,
                        "message": f"{attraction_name} may not be open at {start_time} on {date_str} ({day_of_week}). Opening hours: {opening_hours}"
                    })
                
                # Check date availability
                if not is_attraction_available_on_date(attraction, date):
                    violations.append({
                        "type": "date_range",
                        "date": date_str,
                        "attraction": attraction_name,
                        "date_range": attraction.date_range or "Not specified",
                        "message": f"{attraction_name} may not be available on {date_str}. Available dates: {attraction.date_range or 'Not specified'}"
                    })
                
                # Check time format
                if not re.match(r'^\\d{1,2}:\\d{2}$', start_time):
                    violations.append({
                        "type": "invalid_time_format",
                        "date": date_str,
                        "attraction": attraction_name,
                        "time": start_time,
                        "message": f"Invalid start time format: {start_time}. Expected HH:MM."
                    })
                
                if not re.match(r'^\\d{1,2}:\\d{2}$', end_time):
                    violations.append({
                        "type": "invalid_time_format",
                        "date": date_str,
                        "attraction": attraction_name,
                        "time": end_time,
                        "message": f"Invalid end time format: {end_time}. Expected HH:MM."
                    })
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "violation_count": len(violations)
        }
    
    def check_schedule_gaps(self, day_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for large gaps in the daily schedule during daytime hours.
        
        Args:
            day_plan: A day plan with activities
            
        Returns:
            Dictionary with validation results including any gaps found
        """
        # Extract activities and sort by start time
        activities = day_plan.get("activities", [])
        if not activities:
            return {
                "has_gaps": False,
                "message": "No activities scheduled for this day."
            }
        
        # Sort activities by start time
        sorted_activities = sorted(activities, key=lambda x: x["start_time"])
        
        # Check for gaps larger than 2 hours during daytime (9:00-19:00)
        gaps = []
        for i in range(len(sorted_activities) - 1):
            current_activity = sorted_activities[i]
            next_activity = sorted_activities[i + 1]
            
            # Parse end time of current activity and start time of next activity
            current_end_time = current_activity["end_time"]
            next_start_time = next_activity["start_time"]
            
            # Convert to hours for easy comparison
            current_end_hour, current_end_minute = map(int, current_end_time.split(":"))
            next_start_hour, next_start_minute = map(int, next_start_time.split(":"))
            
            current_end_decimal = current_end_hour + (current_end_minute / 60)
            next_start_decimal = next_start_hour + (next_start_minute / 60)
            
            # Calculate gap in hours
            gap_hours = next_start_decimal - current_end_decimal
            
            # Only flag gaps during daytime (9:00-19:00)
            if gap_hours > 1 and 9 <= current_end_hour < 19:
                gaps.append({
                    "after_activity": current_activity["attraction_name"],
                    "before_activity": next_activity["attraction_name"],
                    "gap_hours": round(gap_hours, 1),
                    "end_time": current_end_time,
                    "start_time": next_start_time,
                    "suggestion": f"Consider adding an activity between {current_end_time} and {next_start_time}, such as visiting a nearby attraction, shopping area, or scheduling a coffee break."
                })
        
        if gaps:
            return {
                "has_gaps": True,
                "gaps": gaps,
                "message": f"Found {len(gaps)} large gap(s) in the schedule during daytime hours.",
                "recommendation": "Please fill these gaps with additional activities to create a more balanced itinerary."
            }
        else:
            return {
                "has_gaps": False,
                "message": "No large gaps found in the schedule during daytime hours."
            }