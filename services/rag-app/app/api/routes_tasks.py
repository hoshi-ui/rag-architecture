from typing import Any

from fastapi import APIRouter


def create_router(context: Any) -> APIRouter:
    router = APIRouter()
    documents = context.document_service()

    @router.get("/tasks")
    async def list_tasks():
        return await documents.list_tasks()

    @router.get("/tasks/{task_id}")
    async def get_task(task_id: str):
        return await documents.get_task(task_id)

    return router
