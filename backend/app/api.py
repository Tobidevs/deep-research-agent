import uuid
import logging
import asyncio
import os
from typing import Any, Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from agent.agent import deep_research_agent
from agent.errors import (
    AgentError,
    InvalidRequestError,
    ThreadNotFoundError,
    classify_exception,
)

router = APIRouter()
logger = logging.getLogger(__name__)
AGENT_TIMEOUT_SECONDS = int(os.getenv("AGENT_TIMEOUT_SECONDS", "180"))


class StartRequest(BaseModel):
    message: str


class ResumeRequest(BaseModel):
    reply: str


class ChatResponse(BaseModel):
    thread_id: str
    status: Literal["awaiting_input", "complete"]
    question: str | None = None
    final_report: Any | None = None
    sources: Any | None = None


def _error_detail(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


def _to_http_exception(error: AgentError) -> HTTPException:
    return HTTPException(
        status_code=error.status_code,
        detail=_error_detail(error.error_code, error.message),
    )


def _validate_non_empty(value: str, field_name: str):
    if not isinstance(value, str) or not value.strip():
        raise InvalidRequestError(f"'{field_name}' must be a non-empty string.")


def _validate_thread_id(thread_id: str):
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise InvalidRequestError("'thread_id' must be a non-empty string.")


def _extract_interrupt(state: dict[str, Any]) -> str | None:
    """Pull the interrupt question out of graph state if paused."""

    print(
        f"State at interrupt: {state}"
    )  # Debugging statement to inspect state structure

    interrupts = state.get("__interrupt__")
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    interrupt_value = getattr(first_interrupt, "value", None)
    if not isinstance(interrupt_value, dict):
        return None

    question = interrupt_value["question"]
    if not question:
        return None

    return question


@router.post("/chat", response_model=ChatResponse)
async def start_run(body: StartRequest):
    try:
        _validate_non_empty(body.message, "message")
        user_message = body.message.strip()

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        state = await deep_research_agent.ainvoke(
                {"messages": [HumanMessage(content=user_message)]},
                config=config,
        )
    

    except AgentError as exc:
        logger.exception("start_run failed")
        raise _to_http_exception(exc) from exc
    except Exception as exc:
        logger.exception("start_run failed with unclassified exception")
        classified = classify_exception(exc, "start_run")
        raise _to_http_exception(classified) from exc

    question = _extract_interrupt(state)

    if question:
        # agent asked a clarifying question, return it to the user for input
        return {
            "thread_id": thread_id,
            "status": "awaiting_input",
            "question": question,
        }

    # No interrupt
    return {
        "thread_id": thread_id,
        "status": "complete",
        "final_report": state.get("final_report"),
        "sources": state.get("sources"),
    }


@router.post("/chat/{thread_id}/resume", response_model=ChatResponse)
async def resume_run(thread_id: str, body: ResumeRequest):
    try:
        _validate_thread_id(thread_id)
        _validate_non_empty(body.reply, "reply")
        user_reply = body.reply.strip()
        config = {"configurable": {"thread_id": thread_id}}

        # pull current state and ensure thread exists
        snapshot = await deep_research_agent.aget_state(config)
        if not snapshot:
            raise ThreadNotFoundError("Thread not found.")

        state = await deep_research_agent.ainvoke(
            Command(resume=user_reply),
            config=config,
        )

        print(
            f"State after resume: {state}"
        )  # Debugging statement to inspect state structure

    except AgentError as exc:
        logger.exception("resume_run failed")
        raise _to_http_exception(exc) from exc
    except Exception as exc:
        logger.exception("resume_run failed with unclassified exception")
        classified = classify_exception(exc, "resume_run")
        raise _to_http_exception(classified) from exc

    question = _extract_interrupt(state)

    if question:
        return {
            "thread_id": thread_id,
            "status": "awaiting_input",
            "question": question,
        }

    return {
        "thread_id": thread_id,
        "status": "complete",
        "final_report": state.get("final_report"),
        "sources": state.get("sources"),
    }
