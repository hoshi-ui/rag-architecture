"""FastAPI entrypoint for the RAG backend service."""

import os
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import configure_api
from app.runtime import build_app_context, logger
from app.runtime.bootstrap import startup_app_context


def create_app(context: Any) -> FastAPI:
    fastapi_app = FastAPI(
        title="RAG Application",
        description="RAG backend service",
        version="1.0.0",
    )
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    configure_api(fastapi_app, context=context)

    @fastapi_app.on_event("startup")
    async def startup_event() -> None:
        await startup_app_context(context)

    web_dir = getattr(context, "WEB_DIR", None)
    runtime_logger = getattr(context, "logger", None)
    if web_dir and os.path.exists(web_dir):
        fastapi_app.mount("/", StaticFiles(directory=web_dir, html=True), name="static")
    elif runtime_logger:
        runtime_logger.warning(f"Web directory not found: {web_dir}")
    return fastapi_app


app = create_app(build_app_context())

__all__ = [
    "app",
    "create_app",
]


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting RAG Application...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
