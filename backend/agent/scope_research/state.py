from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing import Annotated, List, Optional, Sequence
from langgraph.graph.message import add_messages
import operator


class AgentInputState(MessagesState):
    """State of the agent input messages."""
    pass

class AgentState(MessagesState):

    '''
    Main State multi agent research system
    
    @param messages: The messages exchanged between the user and the agent
    @param research_brief: The research brief generated from the user's input
    @param supervisor_message: The messages exchanged between the agent and the supervisor
    @param notes: The notes collected during the research process
    @param final_report: The final report generated after the research process
    '''
    # Research brief from user conversation history 
    research_brief: Optional[str] = None
    # Messages exchanged between agents and supervisor
    supervisor_message: Annotated[Sequence[BaseMessage], add_messages]
    # Notes collected during the research process
    notes: Annotated[list[str], operator.add]
    # Final report generated after the research process
    final_report: Optional[str] = None
    
    
# Structured data models for specific aspects of the agent's state
class ClarifyWithUser(BaseModel):
    need_clarification: bool = Field(default=False, description="Indicates if clarification is needed from the user.")
    question: str = Field(default="", description="The question to ask the user for clarification.")
    verification: str = Field(default="", description="The statement to verify with the user.")
    
class ResearchQuestion(BaseModel):
    research_brief: str = Field(default="", description="The research brief derived from the user's input.")
    