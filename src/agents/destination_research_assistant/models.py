"""Data models for the Destination Research Assistant.

This module defines the data models used by the destination research assistant
components, including analysts, experts, and report structures.
"""

from typing import List, Optional, Annotated
from typing_extensions import TypedDict
from datetime import datetime
import operator

from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

from src.models.preferences import UserPreferences


class Analyst(BaseModel):
    """Represents an analyst who researches a specific aspect of a destination."""
    
    name: str = Field(description="Name of the analyst")
    focus: str = Field(description="The analyst's area of focus (e.g., 'history and geography')")
    description: str = Field(description="Description of the analyst's expertise and research focus")
    persona: str = Field(description="The analyst's persona for interview purposes")


class Expert(BaseModel):
    """Represents an expert who provides information on a specific aspect of a destination."""
    
    name: str = Field(description="Name of the expert")
    specialty: str = Field(description="The expert's specialty area")
    description: str = Field(description="Description of the expert's background and knowledge")


class ReportSection(BaseModel):
    """Represents a section of the destination report."""
    
    title: str = Field(description="Title of the report section")
    content: str = Field(description="Content of the report section")
    sources: List[str] = Field(default_factory=list, description="Sources used in this section")
    analyst_focus: str = Field(description="The focus area of the analyst who created this section")


class ResearchReport(BaseModel):
    """Represents a comprehensive research report on a destination."""
    
    destination_name: str = Field(description="Name of the destination")
    generation_date: datetime = Field(default_factory=datetime.now, description="Date the report was generated")
    summary: str = Field(description="Executive summary of the report")
    sections: List[ReportSection] = Field(description="Sections of the report")


class SearchQuery(BaseModel):
    """Search query for retrieval."""
    
    search_query: str = Field(None, description="Search query for retrieval.")


class InterviewState(MessagesState):
    """State for the interview process."""

    max_num_turns: int = Field(default=3, description="Maximum number of turns in the conversation")
    context: Annotated[List[str], operator.add] = Field(default_factory=list, description="Source documents")
    analyst: Analyst = Field(..., description="Analyst asking questions")
    destination: str = Field(..., description="Destination being researched")
    interview: Optional[str] = Field(None, description="Interview transcript")
    sections: List[str] = Field(default_factory=list, description="Report sections")

class DestinationReportState(MessagesState):
    """State for the destination report generation process."""
    
    destination_name: str = Field(..., description="Name of the destination being researched")
    user_preferences: Optional[UserPreferences] = Field(None, description="User preferences to consider")
    analysts: List[Analyst] = Field(default_factory=list, description="Analysts researching the destination")
    sections: Annotated[List[str], operator.add] = Field(default_factory=list, description="Report sections collected")
    human_analyst_feedback: Optional[str] = Field(None, description="Human feedback on the analysts")
    introduction: Optional[str] = Field(None, description="Introduction section of the report")
    content: Optional[str] = Field(None, description="Main content of the report")
    conclusion: Optional[str] = Field(None, description="Conclusion section of the report")
    report: Optional[str] = Field(None, description="Final comprehensive report")

class GenerateAnalystsState(TypedDict):
    destination_name: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: str # Human feedback
    analysts: List[Analyst] # Analyst asking questions