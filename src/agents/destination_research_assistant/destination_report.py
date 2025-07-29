"""Destination Report Agent implementation for the Trip Agent system.

This agent researches information on destinations and generates comprehensive reports
using a team of analysts that interview experts on different topics related to the destination.
It uses LangGraph to orchestrate the research process and incorporates web search and
Wikipedia integration for information gathering.
"""

import asyncio
import operator
from typing import Any, Dict, List, Optional, Tuple, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send

from src.agents.base import BaseAgent
from src.agents.destination_research_assistant.graph import create_research_graph
from src.agents.destination_research_assistant.models import DestinationReportState, GenerateAnalystsState, Analyst
from src.models.preferences import UserPreferences


class DestinationReportAgent(BaseAgent):
    """Agent responsible for researching and providing comprehensive destination reports.
    
    This agent uses a team of specialized analysts to research different aspects of a destination
    and compile a comprehensive report. Each analyst focuses on a specific topic such as history,
    landmarks, museums, food, nightlife, and events. The research process is orchestrated using
    LangGraph and incorporates web search and Wikipedia integration.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[Any] = None,
        **data
    ):
        """Initialize the Destination Report Agent.
        
        Args:
            llm: Language model to use for this agent.
            tools: Optional list of tools available to this agent.
            memory: Optional memory system for this agent.
            **data: Additional data for the agent.
        """
        super().__init__(
            name="Destination Report Agent",
            description=(
                "an agent that researches and provides comprehensive reports about travel destinations. "
                "You coordinate a team of specialized analysts who research different aspects of a destination "
                "such as history, landmarks, museums, food, nightlife, and events. "
                "Each analyst interviews an expert on their topic and compiles their findings into a "
                "comprehensive report section."
            ),
            llm=llm,
            tools=tools or [],
            memory=memory,
            **data
        )
        
        # Store the LLM for later use in node functions
        self.llm = llm
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent.
        
        Returns:
            str: The system prompt for this agent.
        """
        base_prompt = super().get_system_prompt()
        additional_instructions = (
            "You are a destination research agent that produces comprehensive travel reports. "
            "Your goal is to provide detailed, accurate, and engaging information about destinations "
            "to help travelers make informed decisions.\n\n"
            "You coordinate a team of specialized analysts who each focus on different aspects of a destination. "
            "Each analyst interviews an expert on their topic and compiles a report section. "
            "You then combine these sections into a comprehensive destination report."
        )
        return f"{base_prompt}\n\n{additional_instructions}"
    
    def process(self, destination_name: str, user_preferences: Optional[UserPreferences] = None, callbacks: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Generate a comprehensive report for the specified destination.
        
        Args:
            destination_name: The name of the destination to research.
            user_preferences: Optional user preferences to consider.
            callbacks: Optional callbacks for progress tracking.
            
        Returns:
            Dict[str, Any]: A dictionary containing the comprehensive destination report.
        """
        # Create the research graph from graph.py to use as a subgraph
        research_graph = create_research_graph(llm=self.llm)
        
        # Create the main report graph
        report_graph = self._create_report_graph(research_graph)
        
        # Initialize the state
        initial_state = {
            "destination_name": destination_name,
            "user_preferences": user_preferences,
        }
        
        # Execute the report generation graph
        thread = {"configurable": {"thread_id": f"{destination_name}_report"}, "recursion_limit": 100}
        for _ in report_graph.stream(
            initial_state, 
            thread, 
            stream_mode="values"
        ):
            # Just let the graph run to completion
            pass
        
        # Get the final state with the comprehensive report
        final_state = report_graph.get_state(thread)
        
        # Return the comprehensive report
        return {
            "destination_name": destination_name,
            "sections": final_state.values.get('sections', []),
            "report": final_state.values.get('report', "Report could not be generated.")
        }
    
    def _create_report_graph(self, research_graph) -> StateGraph:
        """Create the LangGraph workflow for destination report generation.
        
        Args:
            research_graph: The research graph to use as a subgraph for interviews.
            
        Returns:
            StateGraph: The compiled graph for report generation.
        """
        builder = StateGraph(DestinationReportState)
        
        # Define node functions
        def create_analysts(state: GenerateAnalystsState) -> Dict[str, Any]:
            """Create a team of analysts for different aspects of the destination.
            
            Args:
                destination_name: The name of the destination to research.
                
            Returns:
                List[Analyst]: A list of analyst objects.
            """
            # Create analysts for different aspects of the destination
            destination_name = state["destination_name"]
            analysts = [
                Analyst(
                    name="Dr. Historical Context",
                    focus="history, geography, and demographics",
                    description=f"Specializes in researching the historical background, geographical features, and demographic information of {destination_name}.",
                    persona=f"You are Dr. Historical Context, a historian and geographer specializing in {destination_name}'s history, geography, and demographics. Your goal is to gather comprehensive information about {destination_name}'s historical development, geographical features, climate, and population demographics. Focus on key historical events, geographical significance, and demographic trends that would be valuable for travelers to understand."
                ),
                Analyst(
                    name="Architectural Guide",
                    focus="landmarks, important buildings, monuments, churches, and parks",
                    description=f"Focuses on identifying and describing key landmarks, monuments, architectural highlights, religious sites, and parks in {destination_name}.",
                    persona=f"You are an Architectural Guide specializing in {destination_name}'s landmarks, buildings, monuments, churches, and parks. Your goal is to gather detailed information about the most significant architectural sites, landmarks, religious buildings, monuments, and green spaces in {destination_name}. Focus on their historical significance, architectural styles, visiting information, and what makes them must-see attractions for travelers."
                ),
                Analyst(
                    name="Cultural Curator",
                    focus="museums and art galleries",
                    description=f"Researches museums, art galleries, and other cultural institutions in {destination_name}, including their collections, exhibitions, and significance.",
                    persona=f"You are a Cultural Curator specializing in {destination_name}'s museums and art galleries. Your goal is to gather comprehensive information about the most important museums, art galleries, and cultural institutions in {destination_name}. Focus on their collections, famous exhibits, historical significance, visiting information, and what makes them culturally significant for travelers interested in art and history."
                ),
                Analyst(
                    name="Culinary Explorer",
                    focus="food and coffee places",
                    description=f"Explores the food scene in {destination_name}, including restaurants, cafes, local cuisine, and specialty coffee shops.",
                    persona=f"You are a Culinary Explorer specializing in {destination_name}'s food scene and coffee culture. Your goal is to gather detailed information about local cuisine, traditional dishes, notable restaurants, food markets, cafes, and specialty coffee shops in {destination_name}. Focus on culinary traditions, must-try dishes, price ranges, best food neighborhoods, and insider tips for food-loving travelers."
                ),
                Analyst(
                    name="Nightlife Specialist",
                    focus="pubs, bars, and clubs",
                    description=f"Investigates the nightlife options in {destination_name}, including pubs, bars, clubs, and entertainment venues.",
                    persona=f"You are a Nightlife Specialist focusing on {destination_name}'s pubs, bars, clubs, and entertainment venues. Your goal is to gather comprehensive information about the best nightlife areas, popular bars, traditional pubs, dance clubs, live music venues, and other evening entertainment options in {destination_name}. Focus on atmosphere, music styles, price ranges, opening hours, and insider tips for travelers looking to experience the local nightlife."
                ),
                Analyst(
                    name="Events Coordinator",
                    focus="special events, festivals, and seasonal activities",
                    description=f"Researches special events, festivals, cultural celebrations, and seasonal activities in {destination_name}.",
                    persona=f"You are an Events Coordinator specializing in {destination_name}'s special events, festivals, and seasonal activities. Your goal is to gather detailed information about annual festivals, cultural celebrations, seasonal events, and special activities that travelers might want to experience in {destination_name}. Focus on event dates, locations, historical significance, what to expect, and how travelers can participate in these local experiences."
                )
            ]
        
            return {"analysts": analysts}

        def human_feedback(state: DestinationReportState) -> Dict[str, Any]:
            """Get human feedback on the interview results."""
            pass
        
        def initiate_all_interviews(state: DestinationReportState) -> List[Send]:
            """This is the "map" step where we run each interview sub-graph using Send API."""
            destination_name = state["destination_name"]
            return [
                Send("conduct_interview", {
                    "analyst": analyst,
                    "destination": destination_name,
                    "max_num_turns": 3
                }) for analyst in state["analysts"]
            ]
        
        def write_introduction(state: DestinationReportState) -> Dict[str, Any]:
            """Generate the report introduction."""
            destination_name = state["destination_name"]
            sections = state["sections"]
            
            # Create a summary of the sections for context
            section_summaries = "\n\n".join([
                f"- {section}..." 
                for section in sections
            ])
            
            prompt = f"""Write an engaging introduction for a comprehensive travel report about {destination_name}.
            
            The report will cover the following aspects:
            {section_summaries}
            
            Your introduction should:
            1. Provide a brief overview of {destination_name}
            2. Highlight what makes this destination special or unique
            3. Set the tone for the detailed sections that follow
            4. Be approximately 250-300 words
            
            Format the introduction with markdown, using ## for the main title.
            """
            
            introduction = self.llm.invoke([
                SystemMessage(content="You are a professional travel writer creating a comprehensive destination report."),
                HumanMessage(content=prompt)
            ])
            
            return {"introduction": introduction.content}
        
        def write_content(state: DestinationReportState) -> Dict[str, Any]:
            """Compile main content from collected sections."""
            sections = state["sections"]
            
            # Organize sections by analyst focus
            content_parts = []
            
            for section in sections:
                # Add the section to the content
                content_parts.append(f"## {section}")
            
            # Join all content parts
            content = "\n\n".join(content_parts)
            
            return {"content": content}
        
        def write_conclusion(state: DestinationReportState) -> Dict[str, Any]:
            """Generate the conclusion section."""
            destination_name = state["destination_name"]
            sections = state["sections"]
            user_preferences = state["user_preferences"]
            
            # Create a summary of the sections for context
            sections = "\n".join([f"- {section}" for section in sections])
            
            preferences_text = ""
            if user_preferences:
                preferences_text = f"""
                Consider these user preferences in your conclusion:
                - Budget: {user_preferences.budget if hasattr(user_preferences, 'budget') else 'Not specified'}
                - Interests: {', '.join(user_preferences.interests) if hasattr(user_preferences, 'interests') and user_preferences.interests else 'Not specified'}
                - Travel Style: {user_preferences.travel_style if hasattr(user_preferences, 'travel_style') else 'Not specified'}
                """
            
            prompt = f"""Write a conclusion for a comprehensive travel report about {destination_name}.
            
            The report covered these sections:
            {sections}
            
            {preferences_text}
            
            Your conclusion should:
            1. Summarize the key highlights of {destination_name}
            2. Provide final thoughts or recommendations
            3. End with an engaging closing statement
            4. Be approximately 200-250 words
            
            Format the conclusion with markdown, using ## for the title.
            """
            
            conclusion = self.llm.invoke([
                SystemMessage(content="You are a professional travel writer creating a comprehensive destination report."),
                HumanMessage(content=prompt)
            ])
            
            return {"conclusion": conclusion.content}
        
        def finalize_report(state: DestinationReportState) -> Dict[str, Any]:
            """Combine all parts into the final report."""
            introduction = state["introduction"] or ""
            content = state["content"] or ""
            conclusion = state["conclusion"] or ""
            
            # Combine all parts into a final report
            report = f"""# Comprehensive Travel Report: {state["destination_name"]}

{introduction}

{content}

{conclusion}
"""
            
            return {"report": report}
        
        # Add nodes to the graph
        builder.add_node("create_analysts", create_analysts)
        builder.add_node("human_feedback", human_feedback)
        builder.add_node("conduct_interview", research_graph.compile())
        builder.add_node("write_introduction", write_introduction)
        builder.add_node("write_content", write_content)
        builder.add_node("write_conclusion", write_conclusion)
        builder.add_node("finalize_report", finalize_report)
        
        # Define edges
        builder.add_edge(START, "create_analysts")
        builder.add_edge("create_analysts", "human_feedback")
        builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])
        builder.add_edge("conduct_interview", "write_introduction")
        builder.add_edge("conduct_interview", "write_content")
        builder.add_edge("conduct_interview", "write_conclusion")
        builder.add_edge(["write_introduction", "write_content", "write_conclusion"], "finalize_report")
        builder.add_edge("finalize_report", END)
        
        # Compile the graph
        memory = MemorySaver()
        return builder.compile(checkpointer=memory)