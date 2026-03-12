import os
from .state import AgentState, ClarifyWithUser, ResearchQuestion
from .prompts import (
    CLARIFY_WITH_USER_INSTRUCTIONS_PROMPT,
    TRANSFORM_MESSAGES_INTO_RESEARCH_TOPIC_PROMPT,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.graph import START, END
from langgraph.types import Command
from langsmith import traceable

from typing import Literal
from datetime import datetime

load_dotenv()

model = init_chat_model(
    model="openai:gpt-4.1",
)
# Structured output models
clarify_model = model.with_structured_output(ClarifyWithUser)
research_model = model.with_structured_output(ResearchQuestion)

def get_todays_date():
    """Return today's date as a string."""
    return datetime.now().strftime("%a %b %-d, %Y")

@traceable
def clarify_with_user(state: AgentState) -> Command[AgentState]:
    """Clarify the user's question if needed."""

    response = clarify_model.invoke(
        [
            HumanMessage(
                content=CLARIFY_WITH_USER_INSTRUCTIONS_PROMPT.format(
                    messages=state["messages"], date=get_todays_date()
                )
            )
        ]
    )

    if response.need_clarification:
        return Command(
        goto=END, update={"messages": [AIMessage(content=response.question)]}
    )
    else:
        return Command(
        goto="write_research_brief",
        update={"messages": [AIMessage(content=response.verification)]},
    )


@traceable
def write_research_brief(state: AgentState):
    """Write a research brief based on the user's input."""

    # Generate a research brief based on conversation history
    response = research_model.invoke(
        [
            HumanMessage(
                content=TRANSFORM_MESSAGES_INTO_RESEARCH_TOPIC_PROMPT.format(
                    messages=state["messages"], date=get_todays_date()
                )
            )
        ]
    )
    
    return {
        "research_brief": response.research_brief,
        "supervisor_message": [HumanMessage(content=f"{response.research_brief}.")],
    }
