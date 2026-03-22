import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

class ResearcherState(TypedDict):
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int  # tracks search budget usage
    research_topic: str
    compressed_research: str
    final_report: str | None  # None until report generation node runs
    sources: list[str]  # List of cited sources in the format 'Title: URL'


class ResearcherOutputState(TypedDict):
    """
    Output state for the research agent, capturing the final outputs of the research process.
    """

    compressed_research: str
    raw_notes: Annotated[list[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]



class Summary(BaseModel):
    """Schema for webpage content summarization.
    
    """
    summary: str = Field(description="Concise summary of the webpage content")
    key_excerpts: str = Field(description="Important quotes and excerpts from the content")
    
class ResearchReport(BaseModel):
    """
    @param report: The comprehensive research report with inline citations
    @param sources: A list of all cited sources in the format 'Title: URL'
    """
    
    report: str = Field(description="Full research report with inline citations [1], [2]...")
    sources: list[str] = Field(description="All cited sources as 'Title: URL'")