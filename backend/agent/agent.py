from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from .utils.nodes import clarify_with_user, write_research_brief
from .utils.state import AgentInputState, AgentState


deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)

deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", END)

scope_research = deep_researcher_builder.compile()

# Testing purposes only
# user_input = input("Enter your message: ")
# result = scope_research.invoke({"messages": [HumanMessage(content=user_input)]})
# # AI response
# print(result["messages"][-1].content)
# user_input = input("Enter your message: ")
# result = scope_research.invoke({"messages": result["messages"] + [HumanMessage(content=user_input)]})
# print(result["messages"][-1].content)
# print(result["research_brief"])