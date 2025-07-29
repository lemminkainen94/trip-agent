"""Data models for the Attraction Extraction Agent.

This module defines the data models used by the attraction extraction agent
components, including state management and data structures.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import operator

from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

from src.models.trip import Attraction, Location


class AttractionCandidate(BaseModel):
    """Represents a candidate attraction extracted from the destination report."""
    
    name: str = Field(description="Name of the attraction")
    category: str = Field(description="Category of the attraction (museum, park, etc.)")
    description: Optional[str] = Field(None, description="Description of the attraction")
    location_name: Optional[str] = Field(None, description="Name of the location where the attraction is situated")
    date_range: Optional[str] = Field(None, description="Date range when the attraction is available (e.g., 'July 1-15, 2025' for festivals)")
    extracted_from: str = Field(description="Section of the report this attraction was extracted from")
    

class AttractionExtractionState(MessagesState):
    """State for the attraction extraction process."""
    
    destination_name: str = Field(..., description="Name of the destination being processed")
    report_content: str = Field(..., description="Content of the destination report")
    extracted_attractions: List[AttractionCandidate] = Field(
        default_factory=list, 
        description="List of attractions extracted from the report"
    )
    enriched_attractions: List[Attraction] = Field(
        default_factory=list, 
        description="List of attractions with enriched information from search"
    )
    current_attraction_index: int = Field(
        default=0,
        description="Index of the current attraction being processed"
    )
    search_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Search results for the current attraction"
    )