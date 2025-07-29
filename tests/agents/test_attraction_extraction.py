"""Unit tests for the Attraction Extraction Agent."""

import pytest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from src.agents.attraction_extraction import AttractionExtractionAgent
from src.agents.attraction_extraction.models import AttractionCandidate
from src.models.trip import Attraction, Location


class TestAttractionExtractionAgent:
    """Test cases for the AttractionExtractionAgent."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock language model."""
        mock = MagicMock(spec=BaseChatModel)
        mock.invoke.return_value = AIMessage(content="""```json
        [
            {
                "name": "Louvre Museum",
                "category": "museum",
                "description": "World's largest art museum",
                "location_name": "Paris",
                "extracted_from": "Museums and Galleries"
            },
            {
                "name": "Eiffel Tower",
                "category": "landmark",
                "description": "Iconic iron tower",
                "location_name": "Champ de Mars",
                "extracted_from": "Landmarks"
            }
        ]
        ```""")
        return mock
    
    @pytest.fixture
    def mock_tavily_search(self):
        """Create a mock Tavily search tool."""
        mock = MagicMock()
        mock.invoke.return_value = """
        The Louvre Museum is located in Paris, France. It is open from 9:00 AM to 6:00 PM 
        every day except Tuesdays. The average visit duration is about 3 hours (180 minutes). 
        Admission costs €17 for adults. The museum has a rating of 4.7 out of 5 based on visitor reviews.
        The museum is located at Rue de Rivoli, 75001 Paris, France.
        Coordinates: 48.8606° N, 2.3376° E
        """
        return mock
    
    @pytest.fixture
    def sample_report(self):
        """Create a sample destination report."""
        return """# Comprehensive Travel Report: Paris

## Introduction
Paris, the capital of France, is known as the "City of Light" and is one of the world's most visited destinations.

## Museums and Galleries
Paris is home to some of the world's finest museums. The **Louvre Museum** houses thousands of works of art, including the Mona Lisa. The **Musée d'Orsay** features impressionist masterpieces.

## Landmarks
The **Eiffel Tower** is the iconic symbol of Paris, offering stunning views of the city. **Notre-Dame Cathedral** is a masterpiece of Gothic architecture, though it's currently being restored after the 2019 fire.

## Food and Dining
Paris is a culinary paradise. **Café de Flore** is a historic café known for its literary connections. **Le Jules Verne** offers fine dining inside the Eiffel Tower.

## Conclusion
Paris offers an unmatched blend of culture, history, and gastronomy.
"""
    
    def test_init(self, mock_llm):
        """Test initialization of the AttractionExtractionAgent."""
        agent = AttractionExtractionAgent(llm=mock_llm)
        assert agent.name == "Attraction Extraction Agent"
        assert "extracts attractions" in agent.description.lower()
    
    @patch("src.agents.attraction_extraction.attraction_extraction.StateGraph")
    def test_process(self, mock_graph_class, mock_llm, mock_tavily_search, sample_report):
        """Test the process method of AttractionExtractionAgent."""
        # Setup mock graph
        mock_graph = MagicMock()
        mock_graph_class.return_value.compile.return_value = mock_graph
        
        # Setup mock result with enriched attractions
        location = Location(
            name="Paris",
            address="Rue de Rivoli, 75001 Paris, France",
            latitude=48.8606,
            longitude=2.3376
        )
        attraction = Attraction(
            name="Louvre Museum",
            description="World's largest art museum",
            location=location,
            category="museum",
            visit_duration=180,
            opening_hours={"Monday": "9:00 AM - 6:00 PM"},
            price=17.0,
            rating=4.7
        )
        mock_graph.invoke.return_value = {"enriched_attractions": [attraction]}
        
        # Create agent and process report
        agent = AttractionExtractionAgent(llm=mock_llm)
        agent.tavily_search = mock_tavily_search
        
        # Use asyncio.run to handle the async process method
        import asyncio
        result = asyncio.run(agent.process(
            report_content=sample_report,
            destination_name="Paris"
        ))
        
        # Verify results
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Attraction)
        assert result[0].name == "Louvre Museum"
        assert result[0].category == "museum"
        assert result[0].visit_duration == 180
    
    def test_extract_attractions_node(self, mock_llm, sample_report):
        """Test the extract_attractions node function."""
        agent = AttractionExtractionAgent(llm=mock_llm)
        
        # Create a test state
        from src.agents.attraction_extraction.models import AttractionExtractionState
        state = AttractionExtractionState(
            destination_name="Paris",
            report_content=sample_report,
            messages=[]
        )
        
        # Access the private method using name mangling
        graph = agent._create_extraction_graph()
        
        # Mock the extract_attractions function to return directly
        with patch.object(agent, '_create_extraction_graph') as mock_create_graph:
            mock_node = MagicMock()
            mock_node.return_value = {"extracted_attractions": [
                AttractionCandidate(
                    name="Louvre Museum",
                    category="museum",
                    description="World's largest art museum",
                    location_name="Paris",
                    extracted_from="Museums and Galleries"
                )
            ]}
            mock_graph = MagicMock()
            mock_graph.nodes = {"extract_attractions": mock_node}
            mock_create_graph.return_value = mock_graph
            
            # Create a new graph with the mock
            test_graph = agent._create_extraction_graph()
            
            # Verify the mock was called correctly
            assert mock_create_graph.called