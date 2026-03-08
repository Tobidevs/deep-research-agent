import os
from .state import AgentState, ClarifyWithUser, ResearchQuestion
from .prompts import CLARIFY_WITH_USER_INSTRUCTIONS_PROMPT
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.graph import Command, StateGraph, START, END
from typing import Literal
from datetime import datetime
load_dotenv()

model = init_chat_model(model="openai:gpt-4.1",)

def get_todays_date():
    """Return today's date as a string."""
    return datetime.now().strftime("%a %b %-d, %Y")


def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief"], "__end__"]:
    """Clarify the user's question if needed."""
    
    structured_output_model = model.with_structured_output(ClarifyWithUser)
    
    response = structured_output_model.invoke(
        HumanMessage(content=CLARIFY_WITH_USER_INSTRUCTIONS_PROMPT.format(
            messages=state.messages,
            date=get_todays_date()
        ))
    )
    
    if response.need_clarification:
        return Command(
            goto="write_research_brief",
            update={"messages": AIMessage(content=response.question)}
        )
    else:
        return Command(
            goto="__end__",
            update={"messages": AIMessage(content=response.verification)}
        )
    
    