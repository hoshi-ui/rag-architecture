import os
from datetime import datetime
from typing import Any

from fastapi import APIRouter
from fastapi import Response
from fastapi.responses import FileResponse


def create_router(context: Any) -> APIRouter:
    router = APIRouter()

    @router.get("/")
    async def root():
        return FileResponse(os.path.join(context.WEB_DIR, "index.html"))

    @router.get("/favicon.ico")
    async def favicon():
        return Response(status_code=204)

    @router.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "vector_db": "connected",
                "embedding": "connected",
                "rerank": "connected",
            },
        }

    return router
