import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class DeepResearchState(TypedDict):
    """Unified state shape for scope + deep research phases."""

    # Scope phase fields
    messages: Annotated[Sequence[BaseMessage], add_messages]
    research_brief: str | None
    supervisor_message: Annotated[Sequence[BaseMessage], add_messages]
    notes: Annotated[list[str], operator.add]

    # Research phase fields
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    compressed_research: str
    final_report: str | None
    sources: list[str]
