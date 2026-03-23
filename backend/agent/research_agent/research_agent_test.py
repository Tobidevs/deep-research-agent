from langchain_core.messages import HumanMessage
from langgraph.types import Command

from ..agent import deep_research_agent
import asyncio

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
    config = {"configurable": {"thread_id": "deep-research-test"}}
    result = await deep_research_agent.ainvoke(
        {"messages": [HumanMessage(content=test_research_brief)]},
        config=config,
    )

    if "__interrupt__" in result:
        print("Graph interrupted, resuming with test clarification...")
        result = await deep_research_agent.ainvoke(
            Command(
                resume=(
                    "Team size is 40, budget is around $600/month, and integrations "
                    "with Slack and Google Workspace are required."
                )
            ),
            config=config,
        )

    print("Final Research Report:")
    print(result["final_report"])
    print("Cited Sources:")
    for source in result["sources"]:
        print(f"- {source}")


asyncio.run(test_research_agent())
