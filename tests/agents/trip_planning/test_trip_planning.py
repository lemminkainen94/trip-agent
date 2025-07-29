"""Unit tests for the Trip Planning Agent.

This module contains tests for the trip planning agent functionality,
including attraction ranking and itinerary creation.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.agents.trip_planning.trip_planning import TripPlanningAgent
from src.agents.trip_planning.models import (
    AttractionRanking,
    CategoryRankings,
    TripPlanningState
)
from src.models.trip import Attraction, Location, Activity, DayPlan, Trip


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content="""
    # Day 1 Itinerary for Copenhagen

    ## Morning
    9:00 AM - 10:30 AM: Visit Christiansborg Palace
    - Explore the Royal Reception Rooms and the ruins beneath the palace
    
    11:00 AM - 12:30 PM: Kongens Have (Rosenborg Castle Gardens)
    - Enjoy a stroll through Copenhagen's oldest public park
    
    ## Afternoon
    1:00 PM - 2:00 PM: Lunch at Torvehallerne
    - Sample local Danish cuisine at this popular food market
    
    2:30 PM - 5:00 PM: National Museum of Denmark
    - Explore Denmark's cultural history through extensive exhibits
    
    ## Evening
    5:30 PM - 7:00 PM: Nyhavn
    - Take in the iconic colorful buildings along the canal
    
    7:30 PM - 9:00 PM: Dinner at Restaurant Schønnemann
    - Enjoy traditional Danish smørrebrød
    """)
    return mock


@pytest.fixture
def sample_attractions():
    """Create sample attractions for testing."""
    return [
        Attraction(
            name="Christiansborg Palace",
            description="Historical palace housing the Danish Parliament",
            location=Location(name="Slotsholmen"),
            category="Historical landmark",
            visit_duration="120",
            opening_hours={"Monday": "10:00 AM - 5:00 PM"},
            price="215.0"
        ),
        Attraction(
            name="Kongens Have",
            description="Copenhagen's oldest public park",
            location=Location(name="Kongens Have"),
            category="Park",
            visit_duration="90",
            opening_hours={"Monday": "10:00 AM - 5:00 PM"}
        ),
        Attraction(
            name="National Museum of Denmark",
            description="Denmark's largest museum of cultural history",
            location=Location(name="Copenhagen"),
            category="Museum",
            visit_duration="150",
            opening_hours={"Monday": "10:00 AM - 5:00 PM"}
        ),
        Attraction(
            name="Torvehallerne",
            description="Popular food market with local products",
            location=Location(name="Copenhagen"),
            category="Food",
            visit_duration="60",
            opening_hours={"Monday": "10:00 AM - 7:00 PM"}
        ),
        Attraction(
            name="Nyhavn",
            description="Iconic waterfront district with colorful buildings",
            location=Location(name="Copenhagen"),
            category="Landmark",
            visit_duration="90",
            opening_hours={"Monday": "Always open"}
        ),
        Attraction(
            name="Restaurant Schønnemann",
            description="Traditional Danish restaurant serving smørrebrød",
            location=Location(name="Copenhagen"),
            category="Food",
            visit_duration="90",
            opening_hours={"Monday": "11:30 AM - 9:00 PM"}
        )
    ]


def test_trip_planning_agent_initialization(mock_llm):
    """Test that the trip planning agent initializes correctly."""
    agent = TripPlanningAgent(llm=mock_llm)
    
    assert agent.name == "Trip Planning Agent"
    assert agent.llm == mock_llm
    assert agent.graph is not None


def test_parse_rankings(mock_llm):
    """Test that the agent can parse attraction rankings from LLM responses."""
    agent = TripPlanningAgent(llm=mock_llm)
    
    # Sample attractions by category
    attractions_by_category = {
        "Historical landmark": [
            Attraction(
                name="Christiansborg Palace",
                description="Historical palace",
                location=Location(name="Slotsholmen"),
                category="Historical landmark",
                visit_duration="120"
            )
        ],
        "Park": [
            Attraction(
                name="Kongens Have",
                description="Public park",
                location=Location(name="Copenhagen"),
                category="Park",
                visit_duration="90"
            )
        ]
    }
    
    # Sample LLM response
    response = """
    Category: Historical landmark
    1. Christiansborg Palace - 9/10
       Reasoning: Significant historical importance and impressive architecture
    
    Category: Park
    1. Kongens Have - 8/10
       Reasoning: Beautiful and relaxing green space in the city center
    """
    
    rankings = agent._parse_rankings(response, attractions_by_category)
    
    assert len(rankings) == 2
    assert rankings[0].category == "Historical landmark"
    assert rankings[0].attractions[0].attraction.name == "Christiansborg Palace"
    assert rankings[0].attractions[0].score == 9.0
    assert "historical importance" in rankings[0].attractions[0].reasoning.lower()
    
    assert rankings[1].category == "Park"
    assert rankings[1].attractions[0].attraction.name == "Kongens Have"
    assert rankings[1].attractions[0].score == 8.0
    assert "beautiful" in rankings[1].attractions[0].reasoning.lower()


def test_parse_day_plan(mock_llm, sample_attractions):
    """Test that the agent can parse day plans from LLM responses."""
    agent = TripPlanningAgent(llm=mock_llm)
    
    # Create ranked categories
    ranked_categories = [
        CategoryRankings(
            category="Historical landmark",
            attractions=[
                AttractionRanking(
                    attraction=sample_attractions[0],
                    score=9.0,
                    reasoning="Important historical site"
                )
            ]
        ),
        CategoryRankings(
            category="Park",
            attractions=[
                AttractionRanking(
                    attraction=sample_attractions[1],
                    score=8.0,
                    reasoning="Beautiful park"
                )
            ]
        ),
        CategoryRankings(
            category="Museum",
            attractions=[
                AttractionRanking(
                    attraction=sample_attractions[2],
                    score=8.5,
                    reasoning="Excellent exhibits"
                )
            ]
        ),
        CategoryRankings(
            category="Food",
            attractions=[
                AttractionRanking(
                    attraction=sample_attractions[3],
                    score=7.5,
                    reasoning="Great food market"
                ),
                AttractionRanking(
                    attraction=sample_attractions[5],
                    score=8.0,
                    reasoning="Traditional cuisine"
                )
            ]
        ),
        CategoryRankings(
            category="Landmark",
            attractions=[
                AttractionRanking(
                    attraction=sample_attractions[4],
                    score=9.5,
                    reasoning="Iconic location"
                )
            ]
        )
    ]
    
    # Sample LLM response
    response = """
    # Day 1 Itinerary for Copenhagen

    ## Morning
    9:00 AM - 10:30 AM: Visit Christiansborg Palace
    - Explore the Royal Reception Rooms and the ruins beneath the palace
    
    11:00 AM - 12:30 PM: Kongens Have (Rosenborg Castle Gardens)
    - Enjoy a stroll through Copenhagen's oldest public park
    
    ## Afternoon
    1:00 PM - 2:00 PM: Lunch at Torvehallerne
    - Sample local Danish cuisine at this popular food market
    
    2:30 PM - 5:00 PM: National Museum of Denmark
    - Explore Denmark's cultural history through extensive exhibits
    
    ## Evening
    5:30 PM - 7:00 PM: Nyhavn
    - Take in the iconic colorful buildings along the canal
    
    7:30 PM - 9:00 PM: Dinner at Restaurant Schønnemann
    - Enjoy traditional Danish smørrebrød
    """
    
    date = datetime(2025, 7, 1)
    day_plan = agent._parse_day_plan(response, date, ranked_categories)
    
    assert day_plan.date == date
    assert len(day_plan.activities) == 6
    
    # Check first activity
    assert day_plan.activities[0].start_time.hour == 9
    assert day_plan.activities[0].start_time.minute == 0
    assert day_plan.activities[0].end_time.hour == 10
    assert day_plan.activities[0].end_time.minute == 30
    assert day_plan.activities[0].attraction.name == "Christiansborg Palace"
    
    # Check that activities are sorted by time
    for i in range(1, len(day_plan.activities)):
        assert day_plan.activities[i].start_time > day_plan.activities[i-1].start_time


@patch.object(TripPlanningAgent, '_parse_rankings')
@patch.object(TripPlanningAgent, '_parse_day_plan')
def test_plan_trip_expected_use(mock_parse_day_plan, mock_parse_rankings, mock_llm, sample_attractions):
    """Test the plan_trip method with expected inputs."""
    agent = TripPlanningAgent(llm=mock_llm)
    
    # Mock the parsing methods
    mock_parse_rankings.return_value = [
        CategoryRankings(
            category="Historical landmark",
            attractions=[
                AttractionRanking(
                    attraction=sample_attractions[0],
                    score=9.0,
                    reasoning="Important historical site"
                )
            ]
        )
    ]
    
    mock_parse_day_plan.return_value = DayPlan(
        date=datetime(2025, 7, 1),
        activities=[
            Activity(
                start_time=datetime(2025, 7, 1, 9, 0),
                end_time=datetime(2025, 7, 1, 10, 30),
                attraction=sample_attractions[0],
                description="Visit Christiansborg Palace"
            )
        ]
    )
    
    # Test the plan_trip method
    start_date = datetime(2025, 7, 1)
    end_date = datetime(2025, 7, 2)
    trip = agent.plan_trip(
        destination_name="Copenhagen",
        destination_report="Copenhagen is the capital of Denmark...",
        attractions=sample_attractions,
        start_date=start_date,
        end_date=end_date,
        start_location=Location(name="Copenhagen Airport"),
        end_location=Location(name="Copenhagen Airport")
    )
    
    # Verify the result
    assert trip.title == "Trip to Copenhagen"
    assert trip.destination.name == "Copenhagen"
    assert trip.start_date == start_date
    assert trip.end_date == end_date
    assert len(trip.days) == 2  # Two days of activities


def test_parse_time_edge_cases(mock_llm):
    """Test time parsing with various formats and edge cases."""
    agent = TripPlanningAgent(llm=mock_llm)
    date = datetime(2025, 7, 1)
    
    # Test various time formats
    assert agent._parse_time("9:00 AM", date).hour == 9
    assert agent._parse_time("9:00 AM", date).minute == 0
    
    assert agent._parse_time("12:00 PM", date).hour == 12
    assert agent._parse_time("12:00 AM", date).hour == 0
    
    assert agent._parse_time("1:30PM", date).hour == 13
    assert agent._parse_time("1:30PM", date).minute == 30
    
    assert agent._parse_time("23:45", date).hour == 23
    assert agent._parse_time("23:45", date).minute == 45
    
    # Test invalid formats
    assert agent._parse_time("not a time", date) is None
    assert agent._parse_time("25:00", date) is None


def test_plan_trip_failure_case(mock_llm):
    """Test the plan_trip method with invalid inputs."""
    agent = TripPlanningAgent(llm=mock_llm)
    
    # Test with end date before start date
    start_date = datetime(2025, 7, 2)
    end_date = datetime(2025, 7, 1)  # Before start date
    
    with pytest.raises(Exception):
        # This should fail because end_date is before start_date
        trip = agent.plan_trip(
            destination_name="Copenhagen",
            destination_report="Copenhagen is the capital of Denmark...",
            attractions=[],  # Empty attractions list
            start_date=start_date,
            end_date=end_date,
            start_location=Location(name="Copenhagen Airport"),
            end_location=Location(name="Copenhagen Airport")
        )