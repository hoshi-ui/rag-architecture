"""API route registration."""

from app.api.routes_documents import create_router as create_documents_router
from app.api.routes_health import create_router as create_health_router
from app.api.routes_query import create_router as create_query_router
from app.api.routes_tasks import create_router as create_tasks_router


def configure_api(app, context) -> None:
    app.include_router(create_health_router(context))
    app.include_router(create_query_router(context))
    app.include_router(create_documents_router(context))
    app.include_router(create_tasks_router(context))
