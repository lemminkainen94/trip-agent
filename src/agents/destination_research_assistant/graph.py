"""Graph module for the Destination Research Assistant.

This module implements the LangGraph workflow for the destination research assistant.
"""

import os
from typing import Dict, List, Any, Tuple, Optional, Literal
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable

from src.agents.destination_research_assistant.models import (
    Analyst, 
    Expert, 
    InterviewState,
    SearchQuery
)

load_dotenv()

# Configure LangSmith if API key is available
LANGSMITH_ENABLED = bool(os.getenv("LANGSMITH_API_KEY"))
if LANGSMITH_ENABLED:
    # Enable LangChain tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    # Configure LangSmith API endpoint
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    # Set LangChain API key
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    print("LangSmith tracing enabled")


@traceable(run_type="chain", name="DestinationResearchTest")
def create_research_graph(
    llm: BaseChatModel, 
    tavily_search: Optional[TavilySearchResults] = None
) -> StateGraph:
    """Create a LangGraph workflow for the destination research process.
    
    Args:
        llm: Language model to use for the research process.
        destination_name: The name of the destination to research.
        tavily_search: Optional Tavily search tool.
        
    Returns:
        StateGraph: The compiled graph.
    """
    
    # Create Tavily search if not provided
    if tavily_search is None:
        tavily_search = TavilySearchResults(max_results=3)
    
    # Define node functions
    def generate_question(state: InterviewState) -> Dict[str, Any]:
        """Generate a question from the analyst."""
        # Get state
        analyst = state["analyst"]
        messages = state["messages"]
        
        # Generate question
        question_instructions = f"""You are an analyst tasked with interviewing an expert to learn about a specific topic. 

Your goal is to boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.
    
2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {analyst.persona}
    
Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.
    
When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""
        
        question = llm.invoke([SystemMessage(content=question_instructions)] + messages)
        
        # Write messages to state
        return {"messages": [question]}
    
    def search_web(state: InterviewState) -> Dict[str, Any]:
        """Retrieve docs from web search."""
        # Search query
        search_instructions = SystemMessage(content=f"""You will be given a conversation between an analyst and an expert. 

Your goal is to generate a well-structured query for use in retrieval and/or web-search related to the conversation.
        
First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query that includes the destination name: {state["destination"]}""")
        
        structured_llm = llm.with_structured_output(SearchQuery)
        search_query = structured_llm.invoke([search_instructions] + state["messages"])
        
        # Search
        search_docs = tavily_search.invoke(search_query.search_query)
        
        # Format
        valid_search_docs = [doc for doc in search_docs if "url" in doc and "content" in doc]
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
                for doc in valid_search_docs
            ]
        )
        
        return {"context": [formatted_search_docs]}
    
    def search_wikipedia(state: InterviewState) -> Dict[str, Any]:
        """Retrieve docs from wikipedia."""
        # Search query
        search_instructions = SystemMessage(content=f"""You will be given a conversation between an analyst and an expert. 

Your goal is to generate a well-structured query for use in retrieval and/or web-search related to the conversation.
        
First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query that includes the destination name: {state["destination"]}""")
        
        structured_llm = llm.with_structured_output(SearchQuery)
        search_query = structured_llm.invoke([search_instructions] + state["messages"])
        
        # Search
        search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()
        
        # Format
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                for doc in search_docs
            ]
        )
        
        return {"context": [formatted_search_docs]}
    
    def generate_answer(state: InterviewState) -> Dict[str, Any]:
        """Generate an answer from the expert."""
        # Get state
        analyst = state["analyst"]
        messages = state["messages"]
        context = state["context"]
        
        # Answer question
        answer_instructions = f"""You are an expert being interviewed by an analyst.

Here is analyst area of focus: {analyst.persona}. 
        
You goal is to answer a question posed by the interviewer.

To answer question, use this context:
        
{context}

When answering questions, follow these guidelines:
        
1. Use only the information provided in the context. 
        
2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

3. The context contain sources at the topic of each individual document.

4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. 

5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc
        
6. If the source is: <Document source="wikipedia" page="7"/> then just list: 

[1] wikipedia, page 7 
        
And skip the addition of the brackets as well as the Document source preamble in your citation.

7. Be exhaustive and list some examples of attractions, events or landmarks from your area of expertise."""
        
        system_message = answer_instructions
        answer = llm.invoke([SystemMessage(content=system_message)] + messages)
        
        # Name the message as coming from the expert
        answer.name = "expert"
        
        # Update the state with the appended messages
        return {"messages": [answer]}
    
    def save_interview(state: InterviewState) -> Dict[str, Any]:
        """Save interviews."""
        # Get messages
        messages = state["messages"]
        
        # Convert interview to a string
        interview = get_buffer_string(messages)
        
        # Save to interviews key
        return {"interview": interview}
    
    def route_messages(state: InterviewState, name: str = "expert") -> Literal["ask_question", "save_interview"]:
        """Route between question and answer."""
        # Get messages
        messages = state["messages"]
        max_num_turns = state["max_num_turns"]
        
        # Check the number of expert answers
        num_responses = len(
            [m for m in messages if isinstance(m, AIMessage) and m.name == name]
        )
        # End if expert has answered more than the max turns
        if num_responses >= max_num_turns:
            return 'save_interview'
        
        # Force ending after a certain number of messages to prevent recursion limit
        # This ensures we don't get stuck in an infinite loop
        if len(messages) >= 6:  # Lower arbitrary limit to prevent recursion issues
            return 'save_interview'
        
        # This router is run after each question - answer pair
        # Check if any message contains a thank you, which signals the end of the interview
        for message in messages:
            if isinstance(message, HumanMessage) and any(phrase in message.content.lower() for phrase in ["thank you", "thanks"]):
                return 'save_interview'
        
        return "ask_question"
    
    def write_section(state: InterviewState) -> Dict[str, Any]:
        """Write a report section from the interview."""
        # Get state
        interview = state["interview"]
        context = state["context"]
        analyst = state["analyst"]
        
        # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
        section_writer_instructions = f"""You are an expert technical writer. 
            
Your task is to create a digestible yet comprehensive section of a report based on a set of source documents.
The report can be long if the source documents are long and there is a lot of information to convey.
Ideally, you should describe each attraction, event or landmark in detail, if it's described in the interview.

1. Analyze the content of the source documents: 
- The name of each source document is at the start of the document, with the <Document tag.
        
2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers
        
3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

4. Try to include a list of attractions, events or landmarks.

5. Make your title engaging based upon the focus area of the analyst: 
{analyst.focus}

6. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Always try to provide a numbered list of the most notable attractions and events mentioned in the interview
- Create a numbered list of source documents, as you use them
- Do not mention the names of interviewers or experts
- Aim for approximately 800
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
        
7. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

8. Be sure to combine sources. For example this is not correct:

[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

There should be no redundant sources. It should simply be:

[3] https://ai.meta.com/blog/meta-llama-3-1/
        
9. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""
        
        system_message = section_writer_instructions
        section = llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content=f"Use this source to write your section: {interview}")])
        # Append it to state
        return {"sections": [section.content]}
    
    # Build the graph
    interview_builder = StateGraph(InterviewState)
    interview_builder.add_node("ask_question", generate_question)
    interview_builder.add_node("search_web", search_web)
    interview_builder.add_node("search_wikipedia", search_wikipedia)
    interview_builder.add_node("answer_question", generate_answer)
    interview_builder.add_node("save_interview", save_interview)
    interview_builder.add_node("write_section", write_section)
    
    # Flow
    interview_builder.add_edge(START, "ask_question")
    interview_builder.add_edge("ask_question", "search_web")
    interview_builder.add_edge("ask_question", "search_wikipedia")
    interview_builder.add_edge("search_web", "answer_question")
    interview_builder.add_edge("search_wikipedia", "answer_question")
    interview_builder.add_conditional_edges("answer_question", route_messages, ['ask_question', 'save_interview'])
    interview_builder.add_edge("save_interview", "write_section")
    interview_builder.add_edge("write_section", END)
    
    return interview_builder