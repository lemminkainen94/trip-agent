"""User preference models for the Trip Agent system."""

from typing import List, Optional, Any
from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    """User preferences for trip planning."""
    
    # Trip essentials
    destination: Optional[str] = Field(None, description="User's destination")
    start_date: Optional[str] = Field(None, description="Trip start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="Trip end date (YYYY-MM-DD)")
    
    # User information
    name: Optional[str] = Field(None, description="User's name")
    travel_style: Optional[str] = Field(
        None, description="User's travel style (luxury, budget, adventure, etc.)"
    )
    interests: List[str] = Field(
        default_factory=list,
        description="List of user interests (history, food, nature, etc.)"
    )
    activity_level: Optional[str] = Field(
        None, description="Preferred activity level (relaxed, moderate, active)"
    )
    accommodation_type: Optional[str] = Field(
        None, description="Preferred accommodation type (hotel, hostel, resort, etc.)"
    )
    budget_range: Optional[str] = Field(
        None, description="Budget range (budget, mid-range, luxury)"
    )
    dietary_restrictions: List[str] = Field(
        default_factory=list,
        description="List of dietary restrictions"
    )
    accessibility_needs: List[str] = Field(
        default_factory=list,
        description="List of accessibility requirements"
    )
    preferred_transportation: List[str] = Field(
        default_factory=list,
        description="Preferred modes of transportation"
    )
    excluded_categories: List[str] = Field(
        default_factory=list,
        description="Categories of attractions to exclude"
    )
    
    def __str__(self) -> str:
        """Return string representation of user preferences."""
        parts = []
        if self.destination:
            parts.append(f"Destination: {self.destination}")
        if self.start_date:
            parts.append(f"Start Date: {self.start_date}")
        if self.end_date:
            parts.append(f"End Date: {self.end_date}")
        if self.name:
            parts.append(f"Name: {self.name}")
        if self.travel_style:
            parts.append(f"Travel Style: {self.travel_style}")
        if self.interests:
            parts.append(f"Interests: {', '.join(self.interests)}")
        if self.activity_level:
            parts.append(f"Activity Level: {self.activity_level}")
        if self.accommodation_type:
            parts.append(f"Accommodation: {self.accommodation_type}")
        if self.budget_range:
            parts.append(f"Budget: {self.budget_range}")
        if self.excluded_categories:
            parts.append(f"Excluded Categories: {', '.join(self.excluded_categories)}")
        
        return ", ".join(parts)


class TripRequest(BaseModel):
    """A request for a trip plan."""
    
    destination: str = Field(description="Desired destination")
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")
    travelers: int = Field(description="Number of travelers")
    preferences: UserPreferences = Field(description="User preferences for the trip")
    additional_notes: Optional[str] = Field(
        None, description="Additional notes or special requests"
    )