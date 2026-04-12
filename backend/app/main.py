import logging

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent.errors import AgentError, classify_exception
from .api import router as api_router

app = FastAPI(title="Deep Research API")
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["localhost", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.exception_handler(AgentError)
async def agent_error_handler(_: Request, exc: AgentError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.error_code, "message": exc.message}},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logger.exception("Unhandled exception in API")
    classified = classify_exception(exc, "api")
    return JSONResponse(
        status_code=classified.status_code,
        content={
            "error": {
                "code": classified.error_code,
                "message": classified.message,
            }
        },
    )