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
    COMPRESS_RESEARCH_REMINDER_PROMPT,
    COMPRESS_RESEARCH_SYSTEM_PROMPT,
    RESEARCH_AGENT_SYSTEM_PROMPT,
)

from .state import ResearcherState, ResearcherOutputState, Summary
from .tools import tavily_search, think_tool

tools = [tavily_search, think_tool]
tools_dict = {tool.name: tool for tool in tools}


model = init_chat_model(model="anthropic:claude-sonnet-4-6")
researcher_model = model.bind_tools(tools)

summarization_model = init_chat_model(model="openai:gpt-4o-mini")
compress_model = init_chat_model(model="openai:gpt-4.1")


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
) -> Literal["tool_node", "compress_research"]:
    """
    Determines whether the agent should continue the research loop or provide
    a final answer based on whether the LLM made tool calls.

    Returns:
        "tool_node": Continue to tool execution
        "compress_research": Stop and compress research
    """
    messages = state["researcher_messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, continue to tool execution
    if last_message.tool_calls:
        return "tool_node"
    # Otherwise, proceed to compress research and provide final answer
    return "compress_research"


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
            )
        )

    return {"researcher_messages": tool_outputs}


def compress_research(state: ResearcherState):
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
    
    return {
        "compressed_research": "Compressed research summary placeholder.",
        "raw_notes": ["Raw notes placeholder."],
    }
