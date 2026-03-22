import asyncio

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
)
from typing import Literal

from .prompts import (
    FINAL_REPORT_PROMPT,
    RESEARCH_AGENT_SYSTEM_PROMPT,
)

from .state import ResearcherState, ResearchReport
from .tools import tavily_search, think_tool

tools = [tavily_search, think_tool]
tools_dict = {tool.name: tool for tool in tools}


model = init_chat_model(
    model="openai:gpt-5.1"
)  # Development: openai:gpt-5.1, Production: anthropic:claude-sonnet-4-6

researcher_model = model.bind_tools(tools)
summarization_model = init_chat_model(model="openai:gpt-4o-mini")
final_report_model = init_chat_model(model="openai:gpt-5.1")

structured_final_report_model = final_report_model.with_structured_output(
    ResearchReport
)


def llm_call(state: ResearcherState):
    """Analyze current state and decide on next action.
    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more info
    2. Provide a final answer based on gathered info

    Returns updated state with the model's response.
    """
    return {
        "researcher_messages": [
            researcher_model.invoke(
                [
                    SystemMessage(content=RESEARCH_AGENT_SYSTEM_PROMPT),
                ]
                + state["researcher_messages"],
            )
        ]
    }


def should_continue(
    state: ResearcherState,
) -> Literal["llm_call", "final_report"]:
    """
    Determine whether to continue research or compress findings based on the last LLM response.

        Returns:
            "llm_call" to continue research loop, "final_report" to finalize findings
    """
    messages = state["researcher_messages"]
    last_message = messages[-1]

    # Check if last message is a tool message
    if not hasattr(last_message, "tool_name"):
        return "final_report"

    print(
        f"Last tool called: {last_message.tool_name} is_research_complete: {last_message.additional_kwargs.get('is_research_complete', 'N/A')}"
    )

    if last_message.tool_name == "tavily_search":
        return "llm_call"  # Loop back to LLM for more research
    if last_message.tool_name == "think_tool":
        if not last_message.additional_kwargs.get("is_research_complete", False):
            return "llm_call"  # Loop back to LLM for more research
    return "final_report"  # Move to finalizing findings


async def tool_node(state: ResearcherState):
    """Execute all tool calls from previous LLM response.
    Returns updated state with tool execution results.
    """
    tool_calls = state["researcher_messages"][-1].tool_calls

    # Build a coroutine for each tool call to execute them concurrently
    tasks = [
        tools_dict[tool_call["name"]].ainvoke(tool_call["args"])
        for tool_call in tool_calls
    ]

    # Run all tool calls concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    is_research_complete = (
        tool_calls[-1]["args"].get("is_research_complete", False)
        if tool_calls
        else False
    )
    
    tool_outputs = []
    for tool_call, result in zip(tool_calls, results):
        if isinstance(result, Exception):
            print(f"Tool call failed for '{tool_call['name']}': {result}")
            continue
        tool_outputs.append(
            ToolMessage(
                content=result,
                tool_name=tool_call["name"],
                tool_call_id=tool_call["id"],
                additional_kwargs={"is_research_complete": is_research_complete},
            )
        )

    return {"researcher_messages": tool_outputs}


def final_report(state: ResearcherState):
    """Compress research findings into a concise summary.

    Takes all the research messages and tool outputs and creates a compressed
    summary suitable for the supervisor's decision-making.
    """
    # Todo: Implement Supervisor evaluation
    # messages = (
    #     [SystemMessage(content=COMPRESS_RESEARCH_SYSTEM_PROMPT)]
    #     + state.get("researcher_messages", [])
    #     + [HumanMessage(content=COMPRESS_RESEARCH_REMINDER_PROMPT)]
    # )
    # response = compress_model.invoke(messages)

    # # Extract raw notes from tool and AI messages
    # raw_notes = [
    #     str(m.content)
    #     for m in filter_messages(
    #         state["researcher_messages"], include_types=["tool", "ai"]
    #     )
    # ]

    # return {
    #     "compressed_research": str(response.content),
    #     "raw_notes": ["\n".join(raw_notes)],
    # }

    response = structured_final_report_model.invoke(
        state["researcher_messages"]
        + [
            HumanMessage(
                content=FINAL_REPORT_PROMPT.format(
                    research_topic=state["research_topic"]
                )
            ),
        ]
    )

    return {"final_report": response.report, "sources": response.sources}
