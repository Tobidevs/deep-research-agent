import asyncio
from dataclasses import dataclass


@dataclass
class AgentError(Exception):
    message: str
    status_code: int = 500
    error_code: str = "agent_error"

    def __str__(self) -> str:
        return self.message


class InvalidRequestError(AgentError):
    def __init__(self, message: str):
        super().__init__(
            message=message,
            status_code=400,
            error_code="invalid_request",
        )


class ThreadNotFoundError(AgentError):
    def __init__(self, message: str = "Thread not found."):
        super().__init__(
            message=message,
            status_code=404,
            error_code="thread_not_found",
        )


class UpstreamServiceError(AgentError):
    def __init__(self, message: str = "Upstream AI service is unavailable."):
        super().__init__(
            message=message,
            status_code=503,
            error_code="upstream_unavailable",
        )


class UpstreamTimeoutError(AgentError):
    def __init__(self, message: str = "Upstream AI service timed out."):
        super().__init__(
            message=message,
            status_code=504,
            error_code="upstream_timeout",
        )


class AgentWorkflowError(AgentError):
    def __init__(self, message: str = "Agent workflow failed."):
        super().__init__(
            message=message,
            status_code=500,
            error_code="workflow_failure",
        )


def _class_name(exc: Exception) -> str:
    return exc.__class__.__name__.lower()


def classify_exception(exc: Exception, context: str = "agent") -> AgentError:
    if isinstance(exc, AgentError):
        return exc

    class_name = _class_name(exc)
    message = str(exc) if str(exc) else f"{context} failed"

    if isinstance(exc, (ValueError, TypeError)):
        return InvalidRequestError(message)

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or "timeout" in class_name:
        return UpstreamTimeoutError()

    if "ratelimit" in class_name or "rate_limit" in class_name or "quota" in class_name:
        return UpstreamServiceError("Upstream AI service is rate-limited.")

    if (
        "apierror" in class_name
        or "connection" in class_name
        or "service" in class_name
        or "http" in class_name
    ):
        return UpstreamServiceError()

    return AgentWorkflowError(f"{context} failed.")
