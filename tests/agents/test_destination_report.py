"""Tests for the Destination Report Agent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from src.agents.destination_research_assistant import DestinationReportAgent
from src.agents.destination_research_assistant.models import (
    Analyst, 
    ReportSection
)
from src.models.trip import Location, Attraction


@pytest.fixture
def mock_llm():
    """Create a mock language model."""
    mock = MagicMock(spec=BaseChatModel)
    mock.ainvoke = AsyncMock()
    return mock


@pytest.fixture
def report_agent(mock_llm):
    """Create a destination report agent with a mock language model."""
    return DestinationReportAgent(llm=mock_llm)


class TestDestinationReportAgent:
    """Test suite for the Destination Report Agent."""

    def test_initialization(self, mock_llm):
        """Test that the agent initializes correctly."""
        agent = DestinationReportAgent(llm=mock_llm)
        
        assert agent.name == "Destination Report Agent"
        assert "comprehensive reports" in agent.description
        assert agent.llm == mock_llm

    def test_system_prompt(self, report_agent):
        """Test that the system prompt is generated correctly."""
        system_prompt = report_agent.get_system_prompt()
        
        assert "Destination Report Agent" in system_prompt
        assert "comprehensive reports" in system_prompt
        assert "Structure the report in clear sections" in system_prompt

    @pytest.mark.asyncio
    async def test_extract_title_and_sources(self, report_agent):
        """Test the _extract_title_and_sources method."""
        # Test with a complete section
        section_content = """## Paris: City of Lights and History

### Summary
Paris, the capital of France, is known for its stunning architecture, art museums, and romantic atmosphere.
The city is home to iconic landmarks such as [1] the Eiffel Tower and [2] the Louvre Museum.

### Sources
[1] https://example.com/paris-landmarks
[2] https://example.com/paris-museums"""

        title, content, sources = report_agent._extract_title_and_sources(section_content)
        
        assert title == "Paris: City of Lights and History"
        assert "Summary" in content
        assert "Sources" not in content
        assert len(sources) == 2
        assert "https://example.com/paris-landmarks" in sources
        assert "https://example.com/paris-museums" in sources

    @pytest.mark.asyncio
    async def test_generate_executive_summary(self, report_agent, mock_llm):
        """Test the _generate_executive_summary method."""
        # Setup mock response
        mock_llm.ainvoke.return_value = MagicMock(content="Paris is a beautiful city with rich history and culture.")
        
        # Create test sections
        sections = [
            ReportSection(
                title="Paris History and Geography",
                content="Paris, the capital of France, has a rich history...",
                sources=["Wikipedia"],
                analyst_focus="history, geography, and demographics"
            ),
            ReportSection(
                title="Paris Landmarks",
                content="Paris is home to many famous landmarks...",
                sources=["Travel Guide"],
                analyst_focus="landmarks, important buildings, monuments, churches, and parks"
            )
        ]
        
        # Call the method
        result = await report_agent._generate_executive_summary("Paris", sections)
        
        # Verify the result
        assert result == "Paris is a beautiful city with rich history and culture."
        assert mock_llm.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_extract_location_info(self, report_agent, mock_llm):
        """Test the _extract_location_info method."""
        # Setup mock response
        mock_llm.ainvoke.return_value = MagicMock(content='{"name": "Paris", "latitude": 48.8566, "longitude": 2.3522}')
        
        # Create test sections
        sections = [
            ReportSection(
                title="Paris History and Geography",
                content="Paris is located at latitude 48.8566° N and longitude 2.3522° E.",
                sources=["Wikipedia"],
                analyst_focus="history, geography, and demographics"
            )
        ]
        
        # Call the method
        result = await report_agent._extract_location_info(sections, "Paris")
        
        # Verify the result
        assert result.name == "Paris"
        assert result.latitude == 48.8566
        assert result.longitude == 2.3522
        assert mock_llm.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_extract_attractions(self, report_agent, mock_llm):
        """Test the _extract_attractions method."""
        # Setup mock response
        mock_llm.ainvoke.return_value = MagicMock(content='''{"attractions": [
            {"name": "Eiffel Tower", "description": "Iconic iron tower", "category": "landmark", "visit_duration": 120},
            {"name": "Louvre Museum", "description": "World-famous art museum", "category": "museum", "visit_duration": 180}
        ]}''')
        
        # Create test sections
        sections = [
            ReportSection(
                title="Paris Landmarks",
                content="Paris is home to the Eiffel Tower, an iconic iron tower...",
                sources=["Travel Guide"],
                analyst_focus="landmarks, important buildings, monuments, churches, and parks"
            ),
            ReportSection(
                title="Paris Museums",
                content="The Louvre Museum is a world-famous art museum...",
                sources=["Museum Guide"],
                analyst_focus="museums and art galleries"
            )
        ]
        
        # Create location
        location = Location(name="Paris", latitude=48.8566, longitude=2.3522)
        
        # Call the method
        result = await report_agent._extract_attractions(sections, "Paris", location)
        
        # Verify the result
        assert len(result) == 2
        assert result[0].name == "Eiffel Tower"
        assert result[0].category == "landmark"
        assert result[0].visit_duration == 120
        assert result[1].name == "Louvre Museum"
        assert result[1].category == "museum"
        assert result[1].visit_duration == 180
        assert mock_llm.ainvoke.call_count == 1

    @pytest.mark.asyncio
    @patch('src.agents.destination_research_assistant.graph.create_research_graph')
    async def test_process_expected_use(self, mock_create_graph, report_agent, mock_llm):
        """Test the process method with expected inputs."""
        # Setup mock graph
        mock_graph = MagicMock()
        mock_graph.stream = MagicMock(return_value=[])
        mock_graph.get_state = MagicMock(return_value=MagicMock(
            values={
                'sections': ["## Paris: City of Lights\n\n### Summary\nParis is beautiful.\n\n### Sources\n[1] Wikipedia"]
            }
        ))
        
        # Setup mock analysts
        mock_analysts = [
            Analyst(
                name="Dr. Historical Context",
                focus="history, geography, and demographics",
                description="Historian",
                persona="You are a historian"
            )
        ]
        
        # Setup mock create_graph
        mock_create_graph.return_value = (mock_graph, mock_analysts)
        
        # Setup mock methods
        report_agent._extract_title_and_sources = MagicMock(return_value=(
            "Paris: City of Lights", 
            "Paris is beautiful.", 
            ["Wikipedia"]
        ))
        report_agent._generate_executive_summary = AsyncMock(return_value="Paris is a beautiful city.")
        report_agent._extract_location_info = AsyncMock(return_value=Location(
            name="Paris", 
            latitude=48.8566, 
            longitude=2.3522
        ))
        report_agent._extract_attractions = AsyncMock(return_value=[
            Attraction(
                name="Eiffel Tower",
                description="Iconic iron tower",
                location=Location(name="Paris", latitude=48.8566, longitude=2.3522),
                category="landmark",
                visit_duration=120
            )
        ])
        
        # Call the process method
        result = await report_agent.process("Paris")
        
        # Verify the result
        assert result["destination_name"] == "Paris"
        assert result["location"].name == "Paris"
        assert result["location"].latitude == 48.8566
        assert result["location"].longitude == 2.3522
        assert len(result["sections"]) == 1
        assert result["sections"][0].title == "Paris: City of Lights"
        assert result["summary"] == "Paris is a beautiful city."
        assert len(result["attractions"]) == 1
        assert result["attractions"][0].name == "Eiffel Tower"

    @pytest.mark.asyncio
    @patch('src.agents.destination_research_assistant.graph.create_research_graph')
    async def test_process_edge_case_empty_sections(self, mock_create_graph, report_agent, mock_llm):
        """Test the process method with empty sections (edge case)."""
        # Setup mock graph
        mock_graph = MagicMock()
        mock_graph.stream = MagicMock(return_value=[])
        mock_graph.get_state = MagicMock(return_value=MagicMock(values={'sections': []}))
        
        # Setup mock analysts
        mock_analysts = [
            Analyst(
                name="Dr. Historical Context",
                focus="history, geography, and demographics",
                description="Historian",
                persona="You are a historian"
            )
        ]
        
        # Setup mock create_graph
        mock_create_graph.return_value = (mock_graph, mock_analysts)
        
        # Setup mock methods
        report_agent._generate_executive_summary = AsyncMock(return_value="Default summary.")
        report_agent._extract_location_info = AsyncMock(return_value=Location(
            name="Unknown", 
            latitude=0.0, 
            longitude=0.0
        ))
        report_agent._extract_attractions = AsyncMock(return_value=[])
        
        # Call the process method
        result = await report_agent.process("Unknown")
        
        # Verify the result handles empty data gracefully
        assert result["destination_name"] == "Unknown"
        assert result["location"].name == "Unknown"
        assert result["location"].latitude == 0.0
        assert result["location"].longitude == 0.0
        assert len(result["sections"]) == 0
        assert result["summary"] == "Default summary."
        assert len(result["attractions"]) == 0

    @pytest.mark.asyncio
    @patch('src.agents.destination_research_assistant.graph.create_research_graph')
    async def test_process_failure_case(self, mock_create_graph, report_agent, mock_llm):
        """Test the process method with a failure case."""
        # Setup mock graph to raise an exception
        mock_graph = MagicMock()
        mock_graph.stream = MagicMock(side_effect=Exception("Test error"))
        
        # Setup mock analysts
        mock_analysts = [
            Analyst(
                name="Dr. Historical Context",
                focus="history, geography, and demographics",
                description="Historian",
                persona="You are a historian"
            )
        ]
        
        # Setup mock create_graph
        mock_create_graph.return_value = (mock_graph, mock_analysts)
        
        # Setup mock methods
        report_agent._generate_executive_summary = AsyncMock(return_value="Default summary.")
        report_agent._extract_location_info = AsyncMock(return_value=Location(
            name="Error", 
            latitude=0.0, 
            longitude=0.0
        ))
        report_agent._extract_attractions = AsyncMock(return_value=[])
        
        # Call the process method
        result = await report_agent.process("Error")
        
        # Verify the result handles errors gracefully
        assert result["destination_name"] == "Error"
        assert result["location"].name == "Error"
        assert len(result["sections"]) == 1
        assert "could not be generated" in result["sections"][0].content
        assert result["summary"] == "Default summary."