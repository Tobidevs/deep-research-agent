from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing import Annotated, List, Optional, Sequence
from langgraph.graph.message import add_messages
import operator


class AgentInputState(MessagesState):
    """State of the agent input messages."""
    pass

    
    
# Structured data models for specific aspects of the agent's state
class ClarifyWithUser(BaseModel):
    need_clarification: bool = Field(default=False, description="Indicates if clarification is needed from the user.")
    question: str = Field(default="", description="The question to ask the user for clarification.")
        
class ResearchQuestion(BaseModel):
    research_brief: str = Field(default="", description="The research brief derived from the user's input.")
    