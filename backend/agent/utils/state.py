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
    '''
    # Research brief from user conversation history 
    research_brief = Optional[str]
    # Messages exchanged between agents and supervisor
    supervisor_message = Annotated[Sequence[BaseMessage], add_messages]
    # Notes collected during the research process
    notes = Annotated[list[str], operator.add] = []
    # Final report generated after the research process
    final_report: str
    
    
# Structured data models for specific aspects of the agent's state
class ClarifyWithUser(BaseModel):
    need_clarification: bool = Field(default=False, description="Indicates if clarification is needed from the user.")
    question: str = Field(default="", description="The question to ask the user for clarification.")
    verification: str = Field(default="", description="The statement to verify with the user.")
    
class ResearchQuestion(BaseModel):
    research_brief: str = Field(default="", description="The research brief derived from the user's input.")
    