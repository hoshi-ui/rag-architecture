import os

import pytest
from fastapi.testclient import TestClient


os.environ.setdefault("APP_ENV", "test_local")
os.environ.setdefault("TEST_LEX_ONLY", "true")
os.environ.setdefault("EMBEDDING_SERVICE_URL", "http://127.0.0.1:8001")
os.environ.setdefault("RERANK_SERVICE_URL", "http://127.0.0.1:8002")
os.environ.setdefault("MILVUS_HOST", "127.0.0.1")
os.environ.setdefault("MILVUS_PORT", "19530")


@pytest.fixture(scope="session")
def app():
    from main import app as fastapi_app

    return fastapi_app


@pytest.fixture()
def client(app):
    return TestClient(app)


@pytest.fixture()
def query_handler():
    from app.runtime import runtime_context

    return runtime_context().query_core()


@pytest.fixture()
def query_runtime():
    from app.runtime import runtime_context

    return runtime_context().query_core()


@pytest.fixture()
def document_service():
    from app.runtime import runtime_context

    return runtime_context().document_service()
