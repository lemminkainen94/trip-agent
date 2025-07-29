"""Trip-related Pydantic models for the Trip Agent system."""

from datetime import datetime
from typing import Dict, List, Optional, Any, Set, TypedDict, Union
from pydantic import BaseModel, Field, validator


class Location(BaseModel):
    """A geographic location with name and coordinates."""
    
    name: str = Field(description="Name of the location")
    address: Optional[str] = Field(None, description="Full address of the location")

    def __str__(self) -> str:
        """Return string representation of the location."""
        return self.name


class Attraction(BaseModel):
    """A point of interest or attraction at a destination."""
    
    name: str = Field(description="Name of the attraction")
    description: str = Field(description="Description of the attraction")
    location: Location = Field(description="Location of the attraction")
    category: str = Field(description="Category of the attraction (museum, park, etc.)")
    visit_duration: str = Field(description="Estimated visit duration in minutes")
    opening_hours: Optional[Dict[str, str]] = Field(
        None, description="Opening hours by day of week"
    )
    price: Optional[str] = Field(None, description="Price of admission with currency if applicable")
    rating: Optional[float] = Field(None, description="Rating out of 5")
    date_range: Optional[str] = Field(
        None, 
        description="Date range when the attraction is available (e.g., 'July 1-15, 2025' for festivals). If None, assumed to be available year-round."
    )
    travel_info: Optional[Dict[str, Dict[str, float]]] = Field(
        None, 
        description="Dictionary mapping attraction names to their distance (meters) and travel time (minutes) by walking"
    )
    
    def __str__(self) -> str:
        """Return string representation of the attraction."""
        return f"{self.name} ({self.category})"


class TravelLeg(BaseModel):
    """A travel segment between two locations."""
    
    origin: Location = Field(description="Starting location")
    destination: Location = Field(description="Ending location")
    mode: str = Field(description="Mode of transportation (walk, drive, transit, etc.)")
    duration: int = Field(description="Estimated travel time in minutes")
    distance: float = Field(description="Distance in kilometers")
    
    def __str__(self) -> str:
        """Return string representation of the travel leg."""
        return f"{self.origin} to {self.destination} by {self.mode}"


class Activity(BaseModel):
    """An activity in the itinerary."""
    
    start_time: datetime = Field(description="Start time of the activity")
    end_time: datetime = Field(description="End time of the activity")
    attraction: Optional[Attraction] = Field(
        None, description="Attraction associated with this activity"
    )
    travel: Optional[TravelLeg] = Field(
        None, description="Travel leg associated with this activity"
    )
    description: str = Field(description="Description of the activity")
    
    def __str__(self) -> str:
        """Return string representation of the activity."""
        if self.attraction:
            return f"{self.start_time.strftime('%H:%M')} - {self.end_time.strftime('%H:%M')}: Visit {self.attraction.name}"
        elif self.travel:
            return f"{self.start_time.strftime('%H:%M')} - {self.end_time.strftime('%H:%M')}: Travel from {self.travel.origin.name} to {self.travel.destination.name}"
        else:
            return f"{self.start_time.strftime('%H:%M')} - {self.end_time.strftime('%H:%M')}: {self.description}"


class DayPlan(BaseModel):
    """A day's worth of activities in the itinerary."""
    
    date: datetime = Field(description="Date of this day plan")
    activities: List[Activity] = Field(description="List of activities for the day")
    
    def __str__(self) -> str:
        """Return string representation of the day plan."""
        return f"Plan for {self.date.strftime('%Y-%m-%d')}:\n{self.activities}"


class Trip(BaseModel):
    """A complete trip itinerary."""
    
    title: str = Field(description="Title of the trip")
    destination: Location = Field(description="Main destination of the trip")
    start_date: Union[datetime, str] = Field(description="Start date of the trip")
    end_date: Union[datetime, str] = Field(description="End date of the trip")
    days: List[DayPlan] = Field(description="Day-by-day itinerary")
    
    @validator('start_date', 'end_date', pre=True)
    def parse_date(cls, v):
        if isinstance(v, str):
            return datetime.strptime(v, '%Y-%m-%d')
        return v
    
    def __str__(self) -> str:
        """Return string representation of the trip."""
        return f"{self.title} ({self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')})"


class AttractionRanking:
    """
    Represents a ranked attraction with its score.
    
    Attributes:
        attraction: The attraction object
        score: Numerical score representing the attraction's rank (higher is better)
        reasoning: Optional reasoning for the ranking
    """
    
    def __init__(
        self, 
        attraction: Attraction, 
        score: float, 
        reasoning: Optional[str] = None
    ):
        """
        Initialize an AttractionRanking.
        
        Args:
            attraction: The attraction object
            score: Numerical score representing the attraction's rank
            reasoning: Optional reasoning for the ranking
        """
        self.attraction = attraction
        self.score = score
        self.reasoning = reasoning
    
    def __repr__(self) -> str:
        """Return string representation of the attraction ranking."""
        return f"AttractionRanking({self.attraction.name}, score={self.score})"


class CategoryRankings:
    """
    Represents ranked attractions within a specific category.
    
    Attributes:
        category: The category name
        attractions: List of ranked attractions in this category
    """
    
    def __init__(self, category: str, attractions: List[AttractionRanking]):
        """
        Initialize a CategoryRankings.
        
        Args:
            category: The category name
            attractions: List of ranked attractions in this category
        """
        self.category = category
        self.attractions = attractions
    
    def __repr__(self) -> str:
        """Return string representation of the category rankings."""
        return f"CategoryRankings({self.category}, {len(self.attractions)} attractions)"


class TripPlanningState(TypedDict, total=False):
    """
    Represents the state of the trip planning process.
    
    This is a TypedDict that can be accessed like a dictionary but provides
    type hints for better code completion and error checking.
    """
    
    # Input parameters
    destination_name: str
    attractions: List[Attraction]
    start_date: Union[str, datetime]
    end_date: Union[str, datetime]
    preferences: Dict[str, Any]
    excluded_categories: List[str]
    destination_report: Optional[str]
    
    # Intermediate state
    used_attractions: Set[str]
    ranked_categories: List[CategoryRankings]
    
    # Output state
    plan: Dict[str, Any]
    trip: Trip
    day_plans: List[DayPlan]