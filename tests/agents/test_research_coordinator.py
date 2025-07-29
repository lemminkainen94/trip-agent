"""Tests for the Research Coordinator component."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.language_models import BaseChatModel

from src.agents.destination_research_assistant.coordinator import ResearchCoordinator
from src.agents.destination_research_assistant.models import (
    Analyst, 
    Expert, 
    Interview, 
    InterviewMessage,
    ReportSection
)
from src.models.trip import Location


@pytest.fixture
def mock_llm():
    """Create a mock language model."""
    mock = MagicMock(spec=BaseChatModel)
    mock.ainvoke = AsyncMock()
    return mock


@pytest.fixture
def coordinator(mock_llm):
    """Create a research coordinator with a mock language model."""
    return ResearchCoordinator(llm=mock_llm)


@pytest.fixture
def mock_interview_manager():
    """Create a mock interview manager."""
    manager = MagicMock()
    manager.conduct_interview = AsyncMock()
    return manager


@pytest.fixture
def mock_report_writer():
    """Create a mock report writer."""
    writer = MagicMock()
    writer.write_section = AsyncMock()
    writer.write_executive_summary = AsyncMock()
    writer.extract_location_info = AsyncMock()
    return writer


class TestResearchCoordinator:
    """Test suite for the Research Coordinator."""

    def test_initialization(self, mock_llm):
        """Test that the coordinator initializes correctly."""
        coordinator = ResearchCoordinator(llm=mock_llm)
        
        assert coordinator.llm == mock_llm
        assert hasattr(coordinator, "interview_manager")
        assert hasattr(coordinator, "report_writer")

    def test_create_analysts(self, coordinator):
        """Test that analysts are created correctly."""
        analysts = coordinator._create_analysts("Paris")
        
        assert len(analysts) == 6
        assert any(a.focus == "history, geography, and demographics" for a in analysts)
        assert any(a.focus == "landmarks, important buildings, monuments, churches, and parks" for a in analysts)
        assert any(a.focus == "museums and art galleries" for a in analysts)
        assert any(a.focus == "food and coffee places" for a in analysts)
        assert any(a.focus == "pubs, bars, and clubs" for a in analysts)
        assert any(a.focus == "special events, festivals, and seasonal activities" for a in analysts)

    def test_create_experts(self, coordinator):
        """Test that experts are created correctly."""
        experts = coordinator._create_experts("Rome")
        
        assert len(experts) == 6
        assert any(e.specialty == "history, geography, and demographics" for e in experts)
        assert any(e.specialty == "landmarks, important buildings, monuments, churches, and parks" for e in experts)
        assert any(e.specialty == "museums and art galleries" for e in experts)
        assert any(e.specialty == "food and coffee places" for e in experts)
        assert any(e.specialty == "pubs, bars, and clubs" for e in experts)
        assert any(e.specialty == "special events, festivals, and seasonal activities" for e in experts)

    @pytest.mark.asyncio
    async def test_conduct_interviews(self, coordinator, mock_interview_manager):
        """Test the interview conducting process."""
        # Setup
        coordinator.interview_manager = mock_interview_manager
        analysts = coordinator._create_analysts("Tokyo")
        experts = coordinator._create_experts("Tokyo")
        
        # Create mock interview result
        mock_interview = Interview(
            analyst=analysts[0],
            expert=experts[0],
            messages=[
                InterviewMessage(role="analyst", content="Question 1?"),
                InterviewMessage(role="expert", content="Answer 1.")
            ],
            summary="Interview summary"
        )
        
        mock_interview_manager.conduct_interview.return_value = mock_interview
        
        # Call the method
        result = await coordinator._conduct_interviews(analysts[:1], experts[:1], "Tokyo")
        
        # Verify
        assert len(result) == 1
        assert result[0]["analyst"] == analysts[0]
        assert result[0]["expert"] == experts[0]
        assert result[0]["interview"] == mock_interview
        mock_interview_manager.conduct_interview.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_report_sections(self, coordinator, mock_report_writer):
        """Test the report section generation process."""
        # Setup
        coordinator.report_writer = mock_report_writer
        analyst = Analyst(
            name="Historical Analyst",
            focus="history, geography, and demographics",
            description="Researches historical aspects"
        )
        expert = Expert(
            name="History Expert",
            specialty="history, geography, and demographics",
            description="Expert in history"
        )
        interview = Interview(
            analyst=analyst,
            expert=expert,
            messages=[
                InterviewMessage(role="analyst", content="Question?"),
                InterviewMessage(role="expert", content="Answer.")
            ],
            summary="Summary"
        )
        
        interview_results = [{
            "analyst": analyst,
            "expert": expert,
            "interview": interview
        }]
        
        mock_section = ReportSection(
            title="History of Barcelona",
            content="Barcelona has a rich history...",
            sources=["Wikipedia"],
            analyst_focus="history, geography, and demographics"
        )
        
        mock_report_writer.write_section.return_value = mock_section
        
        # Call the method
        result = await coordinator._generate_report_sections(interview_results, "Barcelona")
        
        # Verify
        assert len(result) == 1
        assert result[0] == mock_section
        mock_report_writer.write_section.assert_called_once_with(
            analyst=analyst,
            interview=interview,
            destination_name="Barcelona"
        )

    @pytest.mark.asyncio
    async def test_compile_final_report(self, coordinator, mock_report_writer):
        """Test the final report compilation process."""
        # Setup
        coordinator.report_writer = mock_report_writer
        
        sections = [
            ReportSection(
                title="History of Vienna",
                content="Vienna has a rich history...",
                sources=["Wikipedia"],
                analyst_focus="history, geography, and demographics"
            ),
            ReportSection(
                title="Vienna's Museums",
                content="Vienna has many museums...",
                sources=["Museums.org"],
                analyst_focus="museums and art galleries"
            )
        ]
        
        mock_location = Location(name="Vienna", latitude=48.2082, longitude=16.3738)
        mock_summary = "Vienna is a beautiful city with rich history and culture."
        
        mock_report_writer.extract_location_info.return_value = mock_location
        mock_report_writer.write_executive_summary.return_value = mock_summary
        
        # Call the method
        result = await coordinator._compile_final_report("Vienna", sections)
        
        # Verify
        assert result["destination_name"] == "Vienna"
        assert result["location"] == mock_location
        assert result["sections"] == sections
        assert result["summary"] == mock_summary
        
        mock_report_writer.extract_location_info.assert_called_once()
        mock_report_writer.write_executive_summary.assert_called_once_with(
            "Vienna", 
            sections, 
            None
        )

    @pytest.mark.asyncio
    async def test_generate_report_integration(self, coordinator, mock_interview_manager, mock_report_writer):
        """Test the full report generation process (integration test)."""
        # Setup mocks
        coordinator.interview_manager = mock_interview_manager
        coordinator.report_writer = mock_report_writer
        
        # Mock interview
        mock_interview = MagicMock()
        mock_interview_manager.conduct_interview.return_value = mock_interview
        
        # Mock section
        mock_section = ReportSection(
            title="History of Berlin",
            content="Berlin has a fascinating history...",
            sources=["Wikipedia"],
            analyst_focus="history, geography, and demographics"
        )
        mock_report_writer.write_section.return_value = mock_section
        
        # Mock location and summary
        mock_location = Location(name="Berlin", latitude=52.5200, longitude=13.4050)
        mock_summary = "Berlin is a vibrant city with rich history."
        
        mock_report_writer.extract_location_info.return_value = mock_location
        mock_report_writer.write_executive_summary.return_value = mock_summary
        
        # Call the method
        result = await coordinator.generate_report("Berlin")
        
        # Verify
        assert result["destination_name"] == "Berlin"
        assert result["location"] == mock_location
        assert len(result["sections"]) > 0
        assert result["summary"] == mock_summary
        
        # Verify all the steps were called
        assert mock_interview_manager.conduct_interview.call_count > 0
        assert mock_report_writer.write_section.call_count > 0
        assert mock_report_writer.extract_location_info.call_count > 0
        assert mock_report_writer.write_executive_summary.call_count == 1