from typing import Any

from fastapi import APIRouter, File, UploadFile

from app.schemas import DocumentRequest


def create_router(context: Any) -> APIRouter:
    router = APIRouter()
    documents = context.document_service()

    @router.post("/documents")
    async def upload_document(doc_req: DocumentRequest):
        return await documents.upload_document(doc_req)

    @router.post("/documents/upload")
    async def upload_document_file(file: UploadFile = File(...)):
        return await documents.upload_document_file(file)

    @router.get("/documents")
    async def list_documents():
        return await documents.list_documents()

    @router.delete("/documents/{filename}")
    async def delete_document(filename: str):
        return await documents.delete_document(filename)

    @router.post("/documents/{task_id}/retry")
    async def retry_task(task_id: str):
        return await documents.retry_task(task_id)

    @router.get("/documents/{filename}")
    async def get_document_detail(filename: str):
        return await documents.get_document_detail(filename)

    return router
