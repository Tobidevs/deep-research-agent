from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from .nodes import llm_call, should_continue, tool_node, final_report
from .state import ResearcherState
import asyncio


research_agent_builder = StateGraph(ResearcherState)

research_agent_builder.add_node("llm_call", llm_call)
research_agent_builder.add_node("tool_node", tool_node)
research_agent_builder.add_node("final_report", final_report)

research_agent_builder.add_edge(START, "llm_call")
research_agent_builder.add_edge("llm_call", "tool_node")
research_agent_builder.add_conditional_edges(
    "tool_node",
    should_continue,
    {
        "llm_call": "llm_call",  # Loop back for more research
        "final_report": "final_report",  # Provide final answer
    },
)
research_agent_builder.add_edge("final_report", END)  # End the graph

research_agent = research_agent_builder.compile()

# Testing purposes only

test_research_brief = """
I am searching for the right project management (PM) software for my company. We are a digital marketing agency with a team of 40, 
working fully remote. Currently, we do not use any dedicated PM tool and are managing projects through email and spreadsheets. Our budget 
for project management software is around $600 per month. A key requirement is strong client reporting features because we manage a lot of external stakeholders. 
The software must integrate with both Slack and Google Workspace. Time tracking functionality is desirable but not essential (nice-to-have, not a dealbreaker). 
For integration and other features beyond what I listed, I am open to options. No specific requirements were mentioned regarding platforms (web, desktop, mobile), methodology support 
(Agile, Waterfall), user interface, data residency, compliance, security standards, or onboarding/migration support—these are open and can be explored as part of the research scope. 
Please focus on official product sources and ensure the information is up-to-date and matches these explicit criteria.
"""


async def test_research_agent():
    result = await research_agent.ainvoke(
        {
            "researcher_messages": [HumanMessage(content=test_research_brief)],
            "research_topic": test_research_brief,
        }
    )
    print("Final Research Report:")
    print(result["final_report"])
    print("Cited Sources:")
    for source in result["sources"]:
        print(f"- {source}")


asyncio.run(test_research_agent())
