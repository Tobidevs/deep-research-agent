from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from .scope_research.nodes import clarify_with_user, write_research_brief
from .scope_research.state import AgentInputState, AgentState


deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)

deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", END)

scope_research = deep_researcher_builder.compile(checkpointer=InMemorySaver())

# Testing purposes only
thread_id = "scope-session-1"
config = {"configurable": {"thread_id": thread_id}}
initial = scope_research.invoke(
    {"messages": [HumanMessage(content="Help me pick a good PM tool")]},
    config=config,
)
# If clarify_with_user interrupted, resume with user clarification:
# config = {"configurable": {"thread_id": thread_id}}
# state = scope_research.get_state(config=config)

# print(state.tasks[0].interrupts[0].value)
# user_input = input("Enter your message: ")
# resumed = scope_research.invoke(Command(resume=user_input), config=config)
# print(resumed["research_brief"])
