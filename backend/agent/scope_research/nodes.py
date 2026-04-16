import os
import logging
from .state import ClarifyWithUser, ResearchQuestion
from .prompts import (
    CLARIFY_WITH_USER_INSTRUCTIONS_PROMPT,
    TRANSFORM_MESSAGES_INTO_RESEARCH_BRIEF_PROMPT,
)
from ..state import DeepResearchState
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.graph import START, END
from langgraph.types import Command, interrupt
from langsmith import traceable

from typing import Literal
from datetime import datetime
from ..errors import classify_exception


logger = logging.getLogger(__name__)

load_dotenv()

model = init_chat_model(
    model="openai:gpt-4.1",
    temperature=0.0,
)
# Structured output models
clarify_model = model.with_structured_output(ClarifyWithUser)
research_model = model.with_structured_output(ResearchQuestion)


def get_todays_date():
    """Return today's date as a string."""
    return datetime.now().strftime("%a %b %-d, %Y")


@traceable
def clarify_with_user(state: DeepResearchState) -> Command[DeepResearchState]:
    """Clarify the user's question if needed."""

    try:
        response = clarify_model.invoke(
            state["messages"]
            + [
                HumanMessage(
                    content=CLARIFY_WITH_USER_INSTRUCTIONS_PROMPT.format(
                        date=get_todays_date()
                    )
                )
            ]
        )
    except Exception as exc:
        logger.exception("Failed in clarify_with_user node")
        raise classify_exception(exc, "clarify_with_user") from exc

    if response.need_clarification:
        user_clarification = interrupt(
            {
                "kind": "clarification",
                "question": response.question,
            }
        )
        print(f"User clarification: {user_clarification}")

        return Command(
            goto="write_research_brief",
            update={
                "messages": [
                    AIMessage(content=response.question),
                    HumanMessage(content=user_clarification),
                ]
            },
        )
    else:
        return Command(
            goto="write_research_brief",
        )


@traceable
def write_research_brief(state: DeepResearchState):
    """Write a research brief based on the user's input."""

    # Generate a research brief based on conversation history
    try:
        response = research_model.invoke(
            state["messages"]
            + [
                HumanMessage(
                    content=TRANSFORM_MESSAGES_INTO_RESEARCH_BRIEF_PROMPT.format(
                        date=get_todays_date()
                    )
                )
            ]
        )
    except Exception as exc:
        logger.exception("Failed in write_research_brief node")
        raise classify_exception(exc, "write_research_brief") from exc

    return {
        "researcher_messages": [HumanMessage(content=response.research_brief)],
        "research_brief": response.research_brief,
        "supervisor_message": [HumanMessage(content=response.research_brief)],
    }
