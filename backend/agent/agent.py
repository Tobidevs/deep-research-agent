from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langgraph.types import Command
import asyncio
import logging
import uuid

from .scope_research.nodes import clarify_with_user, write_research_brief
from .scope_research.state import AgentInputState
from .research_agent.nodes import llm_call, should_continue, tool_node, final_report
from deepeval.integrations.langchain import CallbackHandler
from .errors import AgentWorkflowError
from .state import DeepResearchState


logger = logging.getLogger(__name__)


# scope_research_builder = StateGraph(AgentState, input_schema=AgentInputState)

# scope_research_builder.add_node("clarify_with_user", clarify_with_user)
# scope_research_builder.add_node("write_research_brief", write_research_brief)

# scope_research_builder.add_edge(START, "clarify_with_user")
# scope_research_builder.add_edge("write_research_brief", END)

# scope_research_agent = scope_research_builder.compile(checkpointer=InMemorySaver())


deep_research_agent_builder = StateGraph(
    DeepResearchState, input_schema=AgentInputState
)

deep_research_agent_builder.add_node("clarify_with_user", clarify_with_user)
deep_research_agent_builder.add_node("write_research_brief", write_research_brief)
deep_research_agent_builder.add_node("llm_call", llm_call)
deep_research_agent_builder.add_node("tool_node", tool_node)
deep_research_agent_builder.add_node("final_report", final_report)

deep_research_agent_builder.add_edge(START, "clarify_with_user")
deep_research_agent_builder.add_edge("write_research_brief", "llm_call")
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
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [CallbackHandler()],
    }

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
    # print("Cited Sources:")
    # for source in result["sources"]:
    #     print(f"- {source}")



# Local/manual testing only.
# asyncio.run(test_deep_research_agent())
