import asyncio

from app.documents.common import split_text
from app.services import document_metadata
from app.services import document_service


def test_public_task_status_groups_lifecycle_states():
    assert document_metadata.public_task_status("vector_pending") == "indexing"
    assert document_metadata.public_task_status("delete_failed") == "failed"
    assert document_metadata.public_task_status("completed") == "completed"


def test_milvus_safe_metadata_trims_large_text_fields_and_summarizes_payload():
    metadata = document_metadata.milvus_safe_metadata(
        {
            "raw_text": "large",
            "text_normalized": "normalized",
            "fts_text": "fts",
            "keep": "value",
            "payload": {
                "ocr_role": "body",
                "probe": {"route": "pdf", "ignored": "x"},
                "layout": {"font_size": 12, "unknown": object()},
            },
        }
    )

    assert "raw_text" not in metadata
    assert "text_normalized" not in metadata
    assert "fts_text" not in metadata
    assert metadata["keep"] == "value"
    assert metadata["payload"]["ocr_role"] == "body"
    assert metadata["payload"]["probe"] == {"route": "pdf"}
    assert metadata["payload"]["layout"] == {"font_size": 12}


def test_document_service_lifecycle_facade_delegates(monkeypatch):
    calls = []

    async def fake_delete_document(filename, context):
        calls.append((context, filename))
        return {"filename": filename}

    monkeypatch.setattr(document_service.document_lifecycle_service, "delete_document", fake_delete_document)

    result = asyncio.run(document_service.delete_document("demo.docx", "ctx"))

    assert result == {"filename": "demo.docx"}
    assert calls == [("ctx", "demo.docx")]


def test_document_service_indexing_facade_delegates(monkeypatch):
    calls = []

    class FakeIndexing:
        def __init__(self, facade):
            self.facade = facade

        async def index_document(self, *args, **kwargs):
            calls.append((self.facade, args, kwargs))
            return 3

    monkeypatch.setattr(document_service, "DocumentIndexingService", FakeIndexing)
    facade = document_service.DocumentService.__new__(document_service.DocumentService)

    result = asyncio.run(facade.index_document(filename="demo.docx", text="hello"))

    assert result == 3
    assert calls == [(facade, (), {"filename": "demo.docx", "text": "hello"})]


def test_split_text_clamps_excessive_overlap_to_keep_windows_moving():
    text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    chunks = split_text(text, chunk_size=20, overlap=19)

    assert chunks[:3] == [
        "abcdefghijklmnopqrst",
        "klmnopqrstuvwxyzABCD",
        "uvwxyzABCDEFGHIJKLMN",
    ]
    assert len(chunks) == 6
