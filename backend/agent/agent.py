from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.types import Command
from typing import Annotated, Sequence, TypedDict
import asyncio
import operator

from .scope_research.nodes import clarify_with_user, write_research_brief
from .scope_research.state import AgentInputState, AgentState
from .research_agent.nodes import llm_call, should_continue, tool_node, final_report


class DeepResearchState(TypedDict):
    """Unified state shape for scope + deep research phases."""

    # Scope phase fields
    messages: Annotated[Sequence[BaseMessage], add_messages]
    research_brief: str | None
    supervisor_message: Annotated[Sequence[BaseMessage], add_messages]
    notes: Annotated[list[str], operator.add]

    # Research phase fields
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    research_topic: str
    tool_call_iterations: int
    compressed_research: str
    final_report: str | None
    sources: list[str]


deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)

deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", END)

scope_research = deep_researcher_builder.compile(checkpointer=InMemorySaver())


def prepare_research_input(state: DeepResearchState):
    """Map scoped brief output into the research agent's expected input fields."""
    research_brief = state.get("research_brief") or ""

    return {
        "research_topic": research_brief,
        "researcher_messages": [HumanMessage(content=research_brief)],
        "tool_call_iterations": state.get("tool_call_iterations", 0),
        "compressed_research": state.get("compressed_research", ""),
        "final_report": state.get("final_report", None),
        "sources": state.get("sources", []),
    }


deep_research_agent_builder = StateGraph(
    DeepResearchState, input_schema=AgentInputState
)

deep_research_agent_builder.add_node("clarify_with_user", clarify_with_user)
deep_research_agent_builder.add_node("write_research_brief", write_research_brief)
deep_research_agent_builder.add_node("prepare_research_input", prepare_research_input)
deep_research_agent_builder.add_node("llm_call", llm_call)
deep_research_agent_builder.add_node("tool_node", tool_node)
deep_research_agent_builder.add_node("final_report", final_report)

deep_research_agent_builder.add_edge(START, "clarify_with_user")
deep_research_agent_builder.add_edge("write_research_brief", "prepare_research_input")
deep_research_agent_builder.add_edge("prepare_research_input", "llm_call")
deep_research_agent_builder.add_edge("llm_call", "tool_node")
deep_research_agent_builder.add_conditional_edges(
    "tool_node",
    should_continue,
    {
        "llm_call": "llm_call",
        "final_report": "final_report",
    },
)
deep_research_agent_builder.add_edge("final_report", END)

deep_research_agent = deep_research_agent_builder.compile(checkpointer=InMemorySaver())


# Testing purposes only
async def test_deep_research_agent():
    thread_id = "deep-research-session-1"
    config = {"configurable": {"thread_id": thread_id}}

    result = await deep_research_agent.ainvoke(
        {"messages": [HumanMessage(content="explain god")]},
        config=config,
    )

    if "__interrupt__" in result:
        clarifying_question = result["__interrupt__"][0].value["question"]
        print(f"\nAgent: {clarifying_question}")
        user_input = input("You: ")

        result = await deep_research_agent.ainvoke(
            Command(resume=user_input), config=config
        )

    print("Final Research Report:")
    print(result["final_report"])
    print("Cited Sources:")
    for source in result["sources"]:
        print(f"- {source}")
        

asyncio.run(test_deep_research_agent())
