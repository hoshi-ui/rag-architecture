import asyncio
import importlib.util
import io
import json
import sys
import uuid
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "services" / "rag-app" / "main.py"


@pytest.fixture()
def rag_module(tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_FAULT_INJECT_STAGE", raising=False)
    monkeypatch.delenv("LEX_DB_CRASH_INJECT_STAGE", raising=False)

    module_name = f"rag_app_main_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    if getattr(module, "_LEX_DB", None) is not None:
        try:
            module._LEX_DB.close()
        except Exception:
            pass
    module._LEX_DB = None
    module.LEXICAL_DB_FILE = str(tmp_path / "lexical_index.db")
    module.TASKS_FILE = str(tmp_path / "tasks.json")
    module.TASKS.clear()
    module._SOURCE_LOCKS.clear()
    module._lex_db_init()
    yield module
    if getattr(module, "_LEX_DB", None) is not None:
        try:
            module._LEX_DB.close()
        except Exception:
            pass


def test_after_purge_fault(rag_module, monkeypatch):
    called = []

    def fake_delete(source, version):
        called.append((source, version))

    monkeypatch.setattr(rag_module, "_delete_milvus_source_version", fake_delete)
    monkeypatch.setenv("RAG_FAULT_INJECT_STAGE", "after_purge")

    with pytest.raises(RuntimeError, match="after_purge"):
        rag_module._purge_source_for_reindex("doc.txt", 2)

    assert called == [("doc.txt", 2)]


def test_after_meta_insert_fault(rag_module, monkeypatch):
    monkeypatch.setenv("RAG_FAULT_INJECT_STAGE", "after_meta_insert")

    with pytest.raises(RuntimeError, match="after_meta_insert"):
        rag_module._lex_db_add_chunk_sql("doc.txt", "hello", "", {}, 0)

    conn = rag_module._lex_db_connect()
    meta_count = conn.execute("SELECT COUNT(*) FROM chunks_meta").fetchone()[0]
    fts_count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    assert meta_count == 1
    assert fts_count == 0


def test_after_fts_insert_fault(rag_module, monkeypatch):
    monkeypatch.setenv("RAG_FAULT_INJECT_STAGE", "after_fts_insert")

    with pytest.raises(RuntimeError, match="after_fts_insert"):
        rag_module._lex_db_add_chunk_sql("doc.txt", "hello", "", {}, 0)

    conn = rag_module._lex_db_connect()
    meta_count = conn.execute("SELECT COUNT(*) FROM chunks_meta").fetchone()[0]
    fts_count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    assert meta_count == 1
    assert fts_count == 1


def test_after_milvus_insert_fault(rag_module, monkeypatch):
    calls = []

    class DummyClient:
        def insert(self, collection_name, data):
            calls.append((collection_name, data))

    service = rag_module.VectorDBService()
    service.client = DummyClient()
    monkeypatch.setenv("RAG_FAULT_INJECT_STAGE", "after_milvus_insert")

    with pytest.raises(RuntimeError, match="after_milvus_insert"):
        service.insert([{"source": "doc.txt", "text": "x", "embedding": [0.0]}])

    assert len(calls) == 1


def test_delete_milvus_fault(rag_module, monkeypatch):
    deleted_sqlite = []

    class DummyClient:
        def delete(self, **kwargs):
            raise AssertionError("Milvus delete should not execute when injected before call")

    class DummyVectorDB:
        def __init__(self):
            self.collection_name = "rag_documents"
            self.client = DummyClient()

        def connect(self):
            return None

    monkeypatch.setattr(rag_module, "VectorDBService", DummyVectorDB)
    monkeypatch.setattr(rag_module, "_lex_db_delete_source", lambda source: deleted_sqlite.append(source))
    monkeypatch.setenv("RAG_FAULT_INJECT_STAGE", "delete_milvus")

    response = asyncio.run(rag_module.delete_document("doc.txt"))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 202
    assert body["status"] == "pending_delete"
    assert body["stage"] == "queued_for_compensation"
    assert body["queued"] is True
    assert body["retryable"] is True
    assert deleted_sqlite == []
    row = rag_module._lex_db_connect().execute("SELECT source FROM pending_delete_queue WHERE source = ?", ("doc.txt",)).fetchone()
    assert row == ("doc.txt",)


def test_delete_sqlite_fault(rag_module, monkeypatch):
    class DummyClient:
        def query(self, **kwargs):
            return []

        def delete(self, **kwargs):
            return None

    class DummyVectorDB:
        def __init__(self):
            self.collection_name = "rag_documents"
            self.client = DummyClient()

        def connect(self):
            return None

    monkeypatch.setattr(rag_module, "VectorDBService", DummyVectorDB)
    monkeypatch.setenv("RAG_FAULT_INJECT_STAGE", "delete_sqlite")

    response = asyncio.run(rag_module.delete_document("doc.txt"))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 202
    assert body["status"] == "pending_delete"
    assert body["stage"] == "queued_for_compensation"
    assert body["queued"] is True
    assert body["retryable"] is True
    row = rag_module._lex_db_connect().execute("SELECT source FROM pending_delete_queue WHERE source = ?", ("doc.txt",)).fetchone()
    assert row == ("doc.txt",)


def test_delete_removes_all_local_records(rag_module, monkeypatch):
    class DummyClient:
        def query(self, **kwargs):
            return []

        def delete(self, **kwargs):
            return None

    class DummyVectorDB:
        def __init__(self):
            self.collection_name = "rag_documents"
            self.client = DummyClient()

        def connect(self):
            return None

    monkeypatch.setattr(rag_module, "VectorDBService", DummyVectorDB)

    source = "doc.txt"
    rag_module._doc_upsert(
        source,
        status="completed",
        active_version=3,
        canonical_title="doc",
        aliases="alias1,alias2",
        filename_stem="doc",
        doc_type="regulation",
        topic="topic-a",
    )
    rag_module._docfts_upsert(source, title="doc", aliases="alias1,alias2", doc_type="regulation", topic="topic-a")
    rag_module._lex_db_set_status(source, "completed")
    rag_module._lex_db_add_chunk_sql(source, "hello world", "sec-a", {"chunk_id": 0}, 0)

    response = asyncio.run(rag_module.delete_document(source))

    assert response["status"] == "completed"

    conn = rag_module._lex_db_connect()
    assert conn.execute("SELECT COUNT(*) FROM documents WHERE source = ?", (source,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM documents_fts WHERE filename = ?", (source,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM doc_status WHERE source = ?", (source,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks_meta WHERE source = ?", (source,)).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks_meta WHERE source = ?)",
        (source,),
    ).fetchone()[0] == 0


def test_delete_removes_uploaded_artifacts(rag_module, monkeypatch, tmp_path):
    class DummyClient:
        def query(self, **kwargs):
            return []

        def delete(self, **kwargs):
            return None

    class DummyVectorDB:
        def __init__(self):
            self.collection_name = "rag_documents"
            self.client = DummyClient()

        def connect(self):
            return None

    monkeypatch.setattr(rag_module, "VectorDBService", DummyVectorDB)

    source = "doc.txt"
    artifact_path = tmp_path / f"task__{source}"
    artifact_path.write_text("payload", encoding="utf-8")
    rag_module.TASKS["upload-task"] = {
        "filename": source,
        "path": str(artifact_path),
        "status": "completed",
        "created_at": "2026-04-20T00:00:00",
    }
    rag_module._doc_upsert(source, status="completed", active_version=1, filename_stem="doc")
    rag_module._lex_db_set_status(source, "completed")

    response = asyncio.run(rag_module.delete_document(source))

    assert response["status"] == "completed"
    assert str(artifact_path) in response["file_cleanup"]["removed"]
    assert not artifact_path.exists()


def test_delete_removes_milvus_container_residuals(rag_module, monkeypatch):
    delete_calls = []

    class DummyClient:
        def query(self, **kwargs):
            filter_expr = kwargs.get("filter") or ""
            if "text like" not in filter_expr:
                return []
            return [
                {"id": 101, "source": "laws_20250527.xlsx", "metadata": {"chunk_id": 50191}, "text": "地方性法规\\doc.txt"},
                {"id": 102, "source": "laws_20250527.xlsx", "metadata": {"chunk_id": 50192}, "text": "...doc.txt..."},
            ]

        def delete(self, **kwargs):
            delete_calls.append(kwargs)
            return None

    class DummyVectorDB:
        def __init__(self):
            self.collection_name = "rag_documents"
            self.client = DummyClient()

        def connect(self):
            return None

    monkeypatch.setattr(rag_module, "VectorDBService", DummyVectorDB)

    response = asyncio.run(rag_module.delete_document("doc.txt"))

    assert response["status"] == "completed"
    assert len(delete_calls) == 2
    assert delete_calls[0].get("filter") == 'source == "doc.txt"'
    assert delete_calls[1].get("ids") == [101, 102]
    assert response["vector_cleanup"]["residual_ids_deleted"] == 2
    assert response["vector_cleanup"]["residual_sources"] == ["laws_20250527.xlsx"]


def test_sqlite_helpers_preserve_outer_transaction(rag_module):
    source = "tx_doc.txt"

    rag_module._lex_tx_begin()
    try:
        rag_module._lex_db_set_status(source, "accepted")
        rag_module._doc_upsert(source, status="accepted", filename_stem="tx_doc")
        rag_module._docfts_upsert(source, title="tx_doc")
        rag_module._lex_db_add_chunk_sql(source, "hello world", "sec-a", {"chunk_id": 0}, 0)
        rag_module._lex_db_delete_source(source)
        rag_module._doc_upsert(source, status="vector_pending", pending_version=2, filename_stem="tx_doc")
        rag_module._lex_tx_commit()
    except Exception:
        rag_module._lex_tx_rollback()
        raise

    conn = rag_module._lex_db_connect()
    row = conn.execute("SELECT status, pending_version FROM documents WHERE source = ?", (source,)).fetchone()
    assert row == ("vector_pending", 2)


def test_source_state_keeps_old_active_visible_during_reindex(rag_module):
    rag_module._doc_upsert("doc.txt", status="reindexing", active_version=1, pending_version=2, filename_stem="doc")

    state = rag_module._source_state("doc.txt")
    hits = [
        {"entity": {"source": "doc.txt", "text": "v1", "metadata": {"doc_version": 1}}, "distance": 0.1},
        {"entity": {"source": "doc.txt", "text": "v2", "metadata": {"doc_version": 2}}, "distance": 0.2},
    ]
    filtered = rag_module._filter_hits_by_source_state(hits)

    assert state["visible"] is True
    assert state["active_version"] == 1
    assert state["pending_version"] == 2
    assert len(filtered["hits"]) == 1
    assert filtered["hits"][0]["entity"]["text"] == "v1"


def test_reindex_purge_preserves_control_plane_and_old_active(rag_module, monkeypatch):
    source = "doc.txt"
    rag_module._doc_upsert(source, status="reindexing", active_version=1, pending_version=2, filename_stem="doc")
    rag_module._docfts_upsert(source, title="doc")
    rag_module._lex_db_set_status(source, "reindexing")
    rag_module._lex_db_add_chunk_sql(source, "active chunk", "", {"chunk_id": 0, "doc_version": 1}, 0)
    rag_module._lex_db_add_chunk_sql(source, "pending chunk", "", {"chunk_id": 0, "doc_version": 2}, 0)
    rag_module._store_document_ir(source, rag_module._build_document_ir_from_text(source, "active body", doc_version=1))
    rag_module._store_document_ir(source, rag_module._build_document_ir_from_text(source, "pending body", doc_version=2))

    deleted_versions = []
    monkeypatch.setattr(rag_module, "_delete_milvus_source_version", lambda src, version: deleted_versions.append((src, version)))

    rag_module._purge_source_for_reindex(source, 2)

    conn = rag_module._lex_db_connect()
    doc_row = conn.execute("SELECT active_version, pending_version FROM documents WHERE source = ?", (source,)).fetchone()
    meta_rows = conn.execute("SELECT metadata FROM chunks_meta WHERE source = ? ORDER BY id ASC", (source,)).fetchall()
    ir_versions = conn.execute("SELECT doc_version FROM document_ir_meta WHERE source = ? ORDER BY doc_version", (source,)).fetchall()

    assert deleted_versions == [(source, 2)]
    assert doc_row == (1, 2)
    assert len(meta_rows) == 1
    assert json.loads(meta_rows[0][0])["doc_version"] == 1
    assert ir_versions == [(1,)]


def test_rebuild_vectors_for_source_uses_pending_version(rag_module, monkeypatch):
    source = "doc.txt"
    rag_module._doc_upsert(source, status="vector_pending", active_version=1, pending_version=2, filename_stem="doc")
    rag_module._lex_db_add_chunk_sql(source, "active chunk", "", {"chunk_id": 0, "doc_version": 1, "raw_text": "active chunk"}, 0)
    rag_module._lex_db_add_chunk_sql(
        source,
        "pending chunk",
        "",
        {"chunk_id": 0, "doc_version": 2, "raw_text": "pending chunk", "text_normalized": "pending chunk", "fts_text": "pending chunk"},
        0,
    )

    deleted_versions = []
    inserts = []

    class DummyEmbeddingService:
        async def embed_batched(self, texts, per_request=32, timeout=60, retries=2):
            return [[0.1, 0.2] for _ in texts]

    class DummyVectorDB:
        def __init__(self):
            self.collection_name = "rag_documents"

        def connect(self):
            return None

        def insert(self, docs):
            inserts.extend(docs)

    monkeypatch.setattr(rag_module, "EmbeddingService", DummyEmbeddingService)
    monkeypatch.setattr(rag_module, "VectorDBService", DummyVectorDB)
    monkeypatch.setattr(rag_module, "_delete_milvus_source_version", lambda src, version: deleted_versions.append((src, version)))

    ok = asyncio.run(rag_module._rebuild_vectors_for_source(source))

    assert ok is True
    assert deleted_versions == [(source, 2)]
    assert len(inserts) == 1
    assert inserts[0]["metadata"]["doc_version"] == 2
    assert "raw_text" not in inserts[0]["metadata"]
    assert "text_normalized" not in inserts[0]["metadata"]
    assert "fts_text" not in inserts[0]["metadata"]


def test_milvus_safe_metadata_removes_large_text_fields_and_summarizes_payload(rag_module):
    metadata = {
        "chunk_id": 3,
        "section": "appendix",
        "raw_text": "x" * 100,
        "text_normalized": "y" * 100,
        "fts_text": "z" * 100,
        "payload": {
            "ocr_role": "body",
            "ocr_meta": {
                "text": "long text" * 50,
                "confidence": 0.98,
                "bbox": [0, 0, 100, 20],
            },
            "layout": {
                "font_size": 18,
                "width_ratio": 0.5,
            },
        },
    }

    safe = rag_module._milvus_safe_metadata(metadata)

    assert safe["chunk_id"] == 3
    assert safe["section"] == "appendix"
    assert "raw_text" not in safe
    assert "text_normalized" not in safe
    assert "fts_text" not in safe
    assert safe["payload"]["ocr_role"] == "body"
    assert safe["payload"]["layout"]["font_size"] == 18
    assert "text" not in safe["payload"]["ocr_meta"]


def test_reconcile_sqlite_from_catalog_backfills_control_plane_and_lexical(rag_module):
    rag_module._doc_upsert("stale.txt", status="completed", filename_stem="stale")
    rag_module._docfts_upsert("stale.txt", title="stale")
    rag_module._lex_db_set_status("stale.txt", "completed")
    rag_module._lex_db_add_chunk_sql("stale.txt", "old chunk", "", {"chunk_id": 0}, 0)

    catalog = {
        "doc.txt": {
            "source": "doc.txt",
            "created_at": "2026-04-02T12:00:00",
            "chunks_indexed": 2,
            "active_version": 3,
            "doc_type": "regulation",
            "topics": ["topic-a"],
            "rows": [
                {
                    "text": "alpha",
                    "metadata": {"chunk_id": 0, "section": "A", "doc_version": 3, "doc_type": "regulation", "topics": ["topic-a"]},
                    "created_at": "2026-04-02T12:00:00",
                    "chunk_id": 0,
                },
                {
                    "text": "beta",
                    "metadata": {"chunk_id": 1, "section": "B", "doc_version": 3, "doc_type": "regulation", "topics": ["topic-a"]},
                    "created_at": "2026-04-02T12:00:01",
                    "chunk_id": 1,
                },
            ],
        }
    }

    report = rag_module._reconcile_sqlite_from_catalog(catalog, prune_sqlite_orphans=True)

    assert report["healthy"] is True
    assert report["upserted_sources"] == 1
    assert report["pruned_sources"] == ["stale.txt"]

    conn = rag_module._lex_db_connect()
    doc_row = conn.execute(
        "SELECT status, active_version, pending_version, filename_stem, doc_type, topic FROM documents WHERE source = ?",
        ("doc.txt",),
    ).fetchone()
    assert doc_row == ("completed", 3, None, "doc", "regulation", "topic-a")
    status_row = conn.execute("SELECT status FROM doc_status WHERE source = ?", ("doc.txt",)).fetchone()
    assert status_row == ("completed",)
    assert conn.execute("SELECT COUNT(*) FROM documents_fts WHERE filename = ?", ("doc.txt",)).fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM chunks_meta WHERE source = ?", ("doc.txt",)).fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM documents WHERE source = ?", ("stale.txt",)).fetchone()[0] == 0


def test_reconcile_defaults_legacy_source_version_to_one(rag_module):
    catalog = {
        "legacy.txt": {
            "source": "legacy.txt",
            "created_at": "2026-04-02T12:00:00",
            "chunks_indexed": 1,
            "active_version": 1,
            "doc_type": "regulation",
            "topics": [],
            "rows": [
                {
                    "text": "legacy body",
                    "metadata": {"chunk_id": 0, "section": ""},
                    "created_at": "2026-04-02T12:00:00",
                    "chunk_id": 0,
                }
            ],
        }
    }

    report = rag_module._reconcile_sqlite_from_catalog(catalog, prune_sqlite_orphans=False)

    assert report["healthy"] is True
    conn = rag_module._lex_db_connect()
    doc_row = conn.execute("SELECT active_version, status FROM documents WHERE source = ?", ("legacy.txt",)).fetchone()
    meta_raw = conn.execute("SELECT metadata FROM chunks_meta WHERE source = ?", ("legacy.txt",)).fetchone()[0]
    meta = json.loads(meta_raw)
    assert doc_row == (1, "completed")
    assert meta["doc_version"] == 1


def test_list_documents_ignores_terminal_task_only_entries(rag_module, monkeypatch):
    monkeypatch.setattr(rag_module, "_milvus_source_stats", lambda: {})
    rag_module.TASKS["failed-only"] = {
        "filename": "ghost.docx",
        "status": "failed",
        "created_at": "2026-04-02T12:00:00",
        "error": "boom",
    }
    rag_module.TASKS["active-only"] = {
        "filename": "live.docx",
        "status": "indexing",
        "created_at": "2026-04-02T12:00:01",
    }

    result = asyncio.run(rag_module.list_documents())
    names = [item["filename"] for item in result["documents"]]

    assert "ghost.docx" not in names
    assert "live.docx" in names


def test_doc_title_profile_strips_date_suffixes(rag_module):
    profile = rag_module._doc_title_profile("林芝市地方立法条例_2017-05-26_2017-05-26.pdf")

    assert profile["stem"] == "林芝市地方立法条例_2017-05-26_2017-05-26"
    assert profile["canonical_title"] == "林芝市地方立法条例"
    assert "林芝市地方立法条例" in profile["aliases"]


def test_doc_title_profile_generates_abbreviated_aliases(rag_module):
    profile = rag_module._doc_title_profile("陵水黎族自治县非物质文化遗产保护条例_2015-04-10_.docx")

    assert "非物质文化遗产保护条例" in profile["aliases"]
    assert "非遗保护条例" in profile["aliases"]


def test_filter_hits_by_source_state_drops_inactive_versions(rag_module):
    rag_module._doc_upsert("doc.txt", status="completed", active_version=2, filename_stem="doc")
    hits = [
        {"entity": {"source": "doc.txt", "text": "old", "metadata": {"doc_version": 1}}, "distance": 0.1},
        {"entity": {"source": "doc.txt", "text": "new", "metadata": {"doc_version": 2}}, "distance": 0.2},
    ]

    filtered = rag_module._filter_hits_by_source_state(hits)

    assert filtered["dropped"] == 1
    assert len(filtered["hits"]) == 1
    assert filtered["hits"][0]["entity"]["text"] == "new"


def test_filter_hits_by_source_state_hides_deleting_and_pending_delete(rag_module):
    rag_module._doc_upsert("deleting.docx", status="deleting", active_version=1, filename_stem="deleting")
    rag_module._doc_upsert("pending.docx", status="pending_delete", active_version=1, filename_stem="pending")

    hits = [
        {"entity": {"source": "deleting.docx", "text": "x", "metadata": {"doc_version": 1}}, "distance": 0.1},
        {"entity": {"source": "pending.docx", "text": "y", "metadata": {"doc_version": 1}}, "distance": 0.2},
    ]

    filtered = rag_module._filter_hits_by_source_state(hits)

    assert filtered["hits"] == []
    assert filtered["dropped"] == 2


def test_build_controlled_expansion_queries_for_weak_reference(rag_module):
    rag_module._doc_upsert(
        "林芝市地方立法条例.pdf",
        status="completed",
        canonical_title="林芝市地方立法条例",
        filename_stem="林芝市地方立法条例",
    )

    expansions = rag_module._build_controlled_expansion_queries(
        "地方立法条例的立法程序",
        ["林芝市地方立法条例.pdf"],
    )

    assert len(expansions) == 1
    assert expansions[0]["source"] == "林芝市地方立法条例.pdf"
    assert "林芝市地方立法条例" in expansions[0]["query"]


def test_contextualize_chunk_items_preserves_raw_text(rag_module):
    items = [
        {"chunk_id": 0, "section": "总则", "text": "第一条 文本A"},
        {"chunk_id": 1, "section": "总则", "text": "第二条 文本B"},
    ]

    contextualized = rag_module._contextualize_chunk_items("条例.docx", items)

    assert contextualized[0]["raw_text"] == "第一条 文本A"
    assert "文档标题：条例" in contextualized[0]["text"]
    assert "下文：第二条 文本B" in contextualized[0]["text"]


def test_process_pending_delete_queue_once_removes_document_and_artifacts(rag_module, monkeypatch, tmp_path):
    class DummyClient:
        def query(self, **kwargs):
            filter_expr = kwargs.get("filter") or ""
            if "text like" in filter_expr:
                return []
            return []

        def delete(self, **kwargs):
            return None

    class DummyVectorDB:
        def __init__(self):
            self.collection_name = "rag_documents"
            self.client = DummyClient()

        def connect(self):
            return None

    monkeypatch.setattr(rag_module, "VectorDBService", DummyVectorDB)

    source = "queued.docx"
    artifact_path = tmp_path / f"queued__{source}"
    artifact_path.write_text("payload", encoding="utf-8")
    rag_module.TASKS["queued-task"] = {
        "filename": source,
        "path": str(artifact_path),
        "status": "failed",
        "created_at": "2026-04-20T00:00:00",
    }
    rag_module._doc_upsert(source, status="pending_delete", active_version=2, filename_stem="queued")
    rag_module._docfts_upsert(source, title="queued")
    rag_module._lex_db_set_status(source, "pending_delete")
    rag_module._lex_db_add_chunk_sql(source, "hello world", "sec-a", {"chunk_id": 0}, 0)
    rag_module._enqueue_pending_delete(source, last_error="milvus unavailable", delete_files=True)

    asyncio.run(rag_module._process_pending_delete_queue_once(limit=10))

    conn = rag_module._lex_db_connect()
    assert conn.execute("SELECT COUNT(*) FROM pending_delete_queue WHERE source = ?", (source,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM documents WHERE source = ?", (source,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks_meta WHERE source = ?", (source,)).fetchone()[0] == 0
    assert not artifact_path.exists()


def test_prepare_structured_items_uses_chapter_clause_pipeline(rag_module):
    text = """第一章 总则
第一条 为了规范管理，制定本条例。
本条例适用于本市相关管理活动。

第二条 主管部门负责统筹协调。
第三条 相关单位应当建立协同机制。"""

    items = rag_module._prepare_structured_items("条例.txt", text, chunk_size=500, overlap=100)

    assert len(items) == 3
    assert items[0]["section"] == "第一章 总则"
    assert items[0]["clause_label"] == "第一条"
    assert "本条例适用于本市相关管理活动" in items[0]["raw_text"]
    assert items[0]["prev_chunk_id"] is None
    assert items[0]["next_chunk_id"] == 1
    assert items[1]["clause_label"] == "第二条"
    assert items[2]["clause_label"] == "第三条"


def test_document_ir_to_structured_items_keeps_clause_units_and_adjacency(rag_module):
    document_ir = {
        "elements": [
            {"element_id": "e1", "element_type": "heading", "text_raw": "第一章 总则", "text_normalized": "第一章 总则", "page_no": 1, "section_path": [], "reading_order": 1},
            {"element_id": "e2", "element_type": "paragraph", "text_raw": "第一条 为了规范管理，制定本条例。", "text_normalized": "第一条 为了规范管理，制定本条例。", "page_no": 1, "section_path": ["第一章 总则"], "reading_order": 2},
            {"element_id": "e3", "element_type": "paragraph", "text_raw": "本条例适用于本市相关管理活动。", "text_normalized": "本条例适用于本市相关管理活动。", "page_no": 1, "section_path": ["第一章 总则"], "reading_order": 3},
            {"element_id": "e4", "element_type": "paragraph", "text_raw": "第二条 主管部门负责统筹协调。", "text_normalized": "第二条 主管部门负责统筹协调。", "page_no": 1, "section_path": ["第一章 总则"], "reading_order": 4},
        ]
    }

    items = rag_module._document_ir_to_structured_items(document_ir, chunk_size=500, overlap=100)

    assert len(items) == 2
    assert items[0]["section"] == "第一章 总则"
    assert items[0]["clause_label"] == "第一条"
    assert items[0]["unit_kind"] == "clause"
    assert "本条例适用于本市相关管理活动" in items[0]["raw_text"]
    assert items[0]["next_chunk_id"] == 1
    assert items[0]["section_node_id"] == "section::第一章 总则"
    assert items[1]["clause_label"] == "第二条"
    assert items[1]["prev_chunk_id"] == 0


def test_document_ir_to_structured_items_respects_explicit_section_path_boundaries(rag_module):
    document_ir = {
        "elements": [
            {"element_id": "e1", "element_type": "paragraph", "text_raw": "第一条 总则内容。", "text_normalized": "第一条 总则内容。", "page_no": 1, "section_path": ["第一编 总纲", "第一章 总则"], "reading_order": 1},
            {"element_id": "e2", "element_type": "paragraph", "text_raw": "第二条 总则补充。", "text_normalized": "第二条 总则补充。", "page_no": 1, "section_path": ["第一编 总纲", "第一章 总则"], "reading_order": 2},
            {"element_id": "e3", "element_type": "paragraph", "text_raw": "第三条 罚则内容。", "text_normalized": "第三条 罚则内容。", "page_no": 2, "section_path": ["第一编 总纲", "第二章 罚则"], "reading_order": 3},
        ]
    }

    items = rag_module._document_ir_to_structured_items(document_ir, chunk_size=500, overlap=100)

    assert len(items) == 3
    assert items[0]["section"] == "第一编 总纲 / 第一章 总则"
    assert items[0]["section_node_id"] == "section::第一编 总纲 > 第一章 总则"
    assert items[0]["parent_section_id"] == "section::第一编 总纲"
    assert items[0]["parent_section_path"] == ["第一编 总纲"]
    assert items[1]["section_node_id"] == "section::第一编 总纲 > 第一章 总则"
    assert items[2]["section"] == "第一编 总纲 / 第二章 罚则"
    assert items[2]["section_node_id"] == "section::第一编 总纲 > 第二章 罚则"


def test_document_ir_to_structured_items_hydrates_section_context(rag_module):
    document_ir = {
        "source": "条例.docx",
        "elements": [
            {"element_id": "e1", "element_type": "paragraph", "text_raw": "第一条 为了规范管理，制定本条例。", "text_normalized": "第一条 为了规范管理，制定本条例。", "page_no": 2, "section_path": ["第一章 总则"], "reading_order": 1},
        ]
    }

    items = rag_module._document_ir_to_structured_items(document_ir, chunk_size=200, overlap=50)
    contextualized = rag_module._contextualize_chunk_items("条例.docx", items)

    assert len(contextualized) == 1
    assert "章节路径：第一章 总则" in contextualized[0]["text"]
    assert "条文锚点：第一条" in contextualized[0]["text"]
    assert "页码范围：2-2" in contextualized[0]["text"]


def test_document_ir_to_structured_items_merges_cross_page_clause(rag_module):
    document_ir = {
        "elements": [
            {"element_id": "e1", "element_type": "paragraph", "text_raw": "第一条 为了规范管理，制定本条例。", "text_normalized": "第一条 为了规范管理，制定本条例。", "page_no": 1, "section_path": ["第一章 总则"], "reading_order": 1},
            {"element_id": "pb1", "element_type": "page_break", "text_raw": "", "text_normalized": "", "page_no": 1, "section_path": ["第一章 总则"], "reading_order": 2},
            {"element_id": "e2", "element_type": "paragraph", "text_raw": "本条例适用于本市相关管理活动。", "text_normalized": "本条例适用于本市相关管理活动。", "page_no": 2, "section_path": ["第一章 总则"], "reading_order": 3},
        ]
    }

    items = rag_module._document_ir_to_structured_items(document_ir, chunk_size=500, overlap=100)

    assert len(items) == 1
    assert items[0]["clause_label"] == "第一条"
    assert "本条例适用于本市相关管理活动" in items[0]["raw_text"]
    assert items[0]["page_span"] == [1, 2]
    assert items[0]["semantic_unit_ids"]


def test_document_ir_to_structured_items_splits_table_with_payload(rag_module):
    document_ir = {
        "source": "表格.docx",
        "elements": [
            {
                "element_id": "t1",
                "element_type": "table",
                "text_raw": "",
                "text_normalized": "",
                "page_no": 3,
                "section_path": ["附表"],
                "reading_order": 1,
                "json_payload": {
                    "table_text": "表格",
                    "table_json": {
                        "headers": ["项目", "数值"],
                        "rows": [["A", "10"], ["B", "20"], ["C", "30"], ["D", "40"], ["E", "50"], ["F", "60"]],
                        "column_count": 2,
                        "row_count": 6,
                    },
                },
            }
        ]
    }

    items = rag_module._document_ir_to_structured_items(document_ir, chunk_size=80, overlap=10)

    assert len(items) >= 2
    assert all(item["chunk_role"] == "table" for item in items)
    assert all(item["payload"].get("rows") for item in items)
    assert items[0]["payload"]["row_start"] == 0
    assert items[-1]["payload"]["row_end"] == 6


def test_document_existence_matches_only_visible_sources(rag_module):
    rag_module._doc_upsert("visible.docx", status="completed", active_version=1, filename_stem="visible")
    rag_module._doc_upsert("hidden.docx", status="vector_failed", active_version=None, filename_stem="hidden")

    matches = rag_module._document_existence_matches(["visible.docx", "hidden.docx", "missing.docx"])

    assert matches == ["visible.docx"]


def test_document_existence_matches_falls_back_to_milvus(rag_module, monkeypatch):
    monkeypatch.setattr(rag_module, "_milvus_source_stats", lambda: {"milvus-only.docx": {"chunks_indexed": 2}})

    matches = rag_module._document_existence_matches(["milvus-only.docx", "missing.docx"])

    assert matches == ["milvus-only.docx"]


def test_negative_clean_refusal_reason_for_generic_regulation_query(rag_module):
    reason = rag_module._negative_clean_refusal_reason("条例核心条款", [], [])

    assert reason == "underspecified_query"


def test_negative_clean_refusal_reason_for_generic_reward_clause_query(rag_module):
    docs = [
        {
            "entity": {
                "source": "陵水黎族自治县非物质文化遗产保护条例_2015-04-10_.docx",
                "text": "对有贡献的单位和个人，可以给予表彰或者奖励。",
                "metadata": {"raw_text": "对有贡献的单位和个人，可以给予表彰或者奖励。"},
            },
            "score": 0.72,
        }
    ]

    reason = rag_module._negative_clean_refusal_reason("表彰奖励条款", docs, [])

    assert reason == "underspecified_query"


def test_negative_clean_refusal_reason_for_anchor_miss(rag_module):
    docs = [
        {
            "entity": {
                "source": "七台河市文明祭祀条例_2022-11-03_2022-12-01.docx",
                "text": "禁止在城市道路焚烧纸钱、抛撒祭品。",
                "metadata": {"raw_text": "禁止在城市道路焚烧纸钱、抛撒祭品。"},
            },
            "distance": 0.12,
        }
    ]

    reason = rag_module._negative_clean_refusal_reason("焚烧冥币是否被禁止", docs, [])

    assert reason == "query_anchor_miss"


def test_negative_clean_refusal_reason_allows_anchor_miss_when_dense_is_strong(rag_module):
    docs = [
        {
            "entity": {
                "source": "七台河市文明祭祀条例_2022-11-03_2022-12-01.docx",
                "text": "禁止在城市道路焚烧纸钱、抛撒祭品。",
                "metadata": {"raw_text": "禁止在城市道路焚烧纸钱、抛撒祭品。"},
            },
            "distance": 0.12,
        }
    ]

    reason = rag_module._negative_clean_refusal_reason(
        "焚烧冥币是否被禁止",
        docs,
        [],
        top_source_dense_score=0.69,
    )

    assert reason is None


def test_classify_query_route_distinguishes_control_and_retrieval_paths(rag_module):
    assert rag_module._classify_query_route("文件是否存在：visible.docx", ["visible.docx"]) == "existence"
    assert rag_module._classify_query_route("删除后是否不可见：文明祭祀条例核心条款", []) == "visibility_probe"
    assert rag_module._classify_query_route("版本切换后只读新版本：浏阳河条例 奖励与处罚", []) == "version_switch"
    assert rag_module._classify_query_route("地方立法条例的立法程序", []) == "weak_title_reference"
    assert rag_module._classify_query_route("总结一下这个系统架构", []) == "content_qa"


def test_classify_question_type_uses_policy_patterns(rag_module):
    assert rag_module._classify_question_type("总结一下这个系统架构") == "summary"
    assert rag_module._classify_question_type("系统设计有哪些模块") == "arch"
    assert rag_module._classify_question_type("difference between dense and lexical retrieval") == "compare"


def test_query_filters_uses_policy_rules(rag_module):
    assert rag_module._query_filters("地方性法规有哪些要求") == {"doc_type": "regulation", "topic": None}
    assert rag_module._query_filters("这份研究报告的结论是什么") == {"doc_type": "research_report", "topic": None}
    assert rag_module._query_filters("环保治理制度设计的实施路径") == {"doc_type": None, "topic": "环保治理制度设计"}
    assert rag_module._query_filters("人工智能成熟度如何评估") == {"doc_type": None, "topic": "AI成熟度研究"}


def test_classify_query_scope_distinguishes_anchored_and_open_questions(rag_module):
    assert rag_module._classify_query_scope("查询 demo.docx 的要求", ["demo.docx"], "content_qa") == "anchored_question"
    assert rag_module._classify_query_scope("地方立法条例的立法程序", [], "weak_title_reference") == "anchored_question"
    assert rag_module._classify_query_scope("总结一下这个系统架构", [], "content_qa") == "open_question"


def test_query_handler_retrieve_uses_light_control_plane_for_filename_query(rag_module, monkeypatch):
    handler = rag_module.QueryHandler()

    async def fake_embed(_texts):
        return [[0.0, 0.0, 0.0]]

    hit = {
        "entity": {
            "source": "visible.docx",
            "text": "visible.docx 第五章规定了奖励与处罚措施，包含整改要求、行政处罚、责令限期改正、表彰激励、适用条件和执行责任，能够直接回答奖励与处罚相关问题。",
            "metadata": {"raw_text": "visible.docx 第五章规定了奖励与处罚措施，包含整改要求、行政处罚、责令限期改正、表彰激励、适用条件和执行责任，能够直接回答奖励与处罚相关问题。", "section": "第五章"},
        },
        "score": 0.88,
    }

    monkeypatch.setattr(handler.embedding_service, "embed", fake_embed)
    monkeypatch.setattr(handler.vector_db, "search", lambda *args, **kwargs: [hit])
    monkeypatch.setattr(rag_module, "_collect_lexical_candidates", lambda *args, **kwargs: [])

    result = asyncio.run(handler.retrieve("visible.docx 的奖励与处罚是什么", user_id="tester"))

    assert result["metadata"]["query_route"] == "light_rag"
    assert result["metadata"]["control_plane"] == "light"
    assert [doc["source"] for doc in result["documents"]] == ["visible.docx"]


def test_query_handler_retrieve_refuses_when_light_evidence_is_missing(rag_module, monkeypatch):
    handler = rag_module.QueryHandler()

    async def fake_embed(_texts):
        return [[0.0, 0.0, 0.0]]

    monkeypatch.setattr(handler.embedding_service, "embed", fake_embed)
    monkeypatch.setattr(handler.vector_db, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(rag_module, "_collect_lexical_candidates", lambda *args, **kwargs: [])

    result = asyncio.run(handler.retrieve("删除后是否不可见：文明祭祀条例核心条款", user_id="tester"))

    assert result["documents"] == []
    assert result["metadata"]["control_plane"] == "light"
    assert result["metadata"]["refused"] == "no_relevant_evidence"


def test_doc_recall_indexed_falls_back_for_chinese_title_query(rag_module):
    source = "七台河市文明行为促进条例_2020-12-29_2021-03-05.docx"
    rag_module._doc_upsert(
        source,
        status="completed",
        active_version=1,
        canonical_title="七台河市文明行为促进条例",
        aliases="文明行为促进条例,七台河市文明行为促进条例",
        filename_stem="七台河市文明行为促进条例_2020-12-29_2021-03-05",
    )
    rag_module._docfts_upsert(source, title="七台河市文明行为促进条例", aliases="文明行为促进条例,七台河市文明行为促进条例")

    hits = rag_module._doc_recall_indexed("文明行为促进条例核心条款", limit=5)

    assert source in hits


def test_build_doc_recall_plan_combines_title_and_term_overlap(rag_module):
    source = "浏阳河管理条例__.docx"
    rag_module._doc_upsert(
        source,
        status="completed",
        active_version=1,
        canonical_title="浏阳河管理条例",
        aliases="浏阳河条例,浏阳河管理条例",
        filename_stem="浏阳河管理条例__",
    )
    rag_module._docfts_upsert(source, title="浏阳河管理条例", aliases="浏阳河条例,浏阳河管理条例")
    rag_module._lex_db_add_chunk_sql(
        source,
        "浏阳河河道管理范围内禁止违法建设和倾倒废弃物。",
        "管理范围",
        {"chunk_id": 0, "doc_version": 1, "raw_text": "浏阳河河道管理范围内禁止违法建设和倾倒废弃物。"},
        0,
    )

    plan = rag_module._build_doc_recall_plan("浏阳河条例 管理范围和禁止行为", limit=5)

    assert plan
    assert plan[0]["source"] == source
    assert "title_alias_substring" in plan[0]["reasons"]
    assert "doc_term_overlap" in plan[0]["reasons"]
    assert plan[0]["prior"] > 0


def test_lexical_recall_fallback_returns_chunk_hits_for_chinese_query(rag_module):
    source = "浏阳河管理条例__.docx"
    rag_module._doc_upsert(source, status="completed", active_version=1, filename_stem="浏阳河管理条例__")
    rag_module._lex_db_add_chunk_sql(
        source,
        "对浏阳河河道管理违法行为可以依法处罚。",
        "罚则",
        {"chunk_id": 0, "doc_version": 1, "raw_text": "对浏阳河河道管理违法行为可以依法处罚。"},
        0,
    )

    hits = rag_module._lexical_recall_fallback("河道管理处罚规定", limit=5)
    sources = [hit["entity"]["source"] for hit in hits]

    assert source in sources


def test_collect_lexical_candidates_includes_title_hit_for_allowed_doc(rag_module):
    source = "林芝市地方立法条例_2017-05-26_2017-05-26.pdf"
    rag_module._doc_upsert(
        source,
        status="completed",
        active_version=1,
        canonical_title="林芝市地方立法条例",
        aliases="地方立法条例,林芝市地方立法条例",
        filename_stem="林芝市地方立法条例_2017-05-26_2017-05-26",
    )

    hits = rag_module._collect_lexical_candidates(
        "地方立法条例的立法程序",
        safe_names=[],
        doc_recall_plan=[{"source": source, "prior": 0.8, "reasons": ["title_alias_substring"]}],
    )

    assert any(hit["entity"]["source"] == source for hit in hits)
    assert any((hit["entity"]["metadata"] or {}).get("title_hit") for hit in hits)


def test_resolve_query_target_sources_locks_unique_regulation_doc(rag_module):
    source = "林芝市地方立法条例_2017-05-26_2017-05-26.pdf"
    rag_module._doc_upsert(
        source,
        status="completed",
        active_version=1,
        canonical_title="林芝市地方立法条例",
        aliases="地方立法条例,林芝地方立法条例",
        filename_stem="林芝市地方立法条例_2017-05-26_2017-05-26",
    )

    resolution = rag_module._resolve_query_target_sources("林芝地方立法条例的立法程序", [])

    assert resolution["required"] is True
    assert resolution["resolved"] is True
    assert resolution["sources"] == [source]
    assert resolution["reason"] == "title_alias_unique"


def test_query_handler_process_requests_document_for_ambiguous_regulation_query(rag_module, monkeypatch):
    source_a = "林芝市地方立法条例_2017-05-26_2017-05-26.pdf"
    source_b = "攀枝花市地方立法条例_2016-09-01_2016-10-01.pdf"
    rag_module._doc_upsert(
        source_a,
        status="completed",
        active_version=1,
        canonical_title="林芝市地方立法条例",
        aliases="地方立法条例,林芝地方立法条例",
        filename_stem="林芝市地方立法条例_2017-05-26_2017-05-26",
    )
    rag_module._doc_upsert(
        source_b,
        status="completed",
        active_version=1,
        canonical_title="攀枝花市地方立法条例",
        aliases="地方立法条例,攀枝花地方立法条例",
        filename_stem="攀枝花市地方立法条例_2016-09-01_2016-10-01",
    )
    handler = rag_module.QueryHandler()

    async def fail_embed(_texts):
        raise AssertionError("embed should not run for ambiguous regulation clarification")

    monkeypatch.setattr(handler.embedding_service, "embed", fail_embed)

    result = asyncio.run(handler.process("地方立法条例的立法程序", user_id="tester"))

    assert result["sources"] == []
    assert result["metadata"]["query_route"] == "document_clarification"
    assert result["metadata"]["refused"] == "document_target_required"
    assert result["metadata"]["answer_mode"] == "clarification"
    assert len(result["metadata"]["candidate_sources"]) == 2
    assert "请先说明要查询哪一部法规文档" in result["answer"]


def test_query_handler_retrieve_requests_document_for_ambiguous_regulation_query(rag_module, monkeypatch):
    source_a = "林芝市地方立法条例_2017-05-26_2017-05-26.pdf"
    source_b = "攀枝花市地方立法条例_2016-09-01_2016-10-01.pdf"
    rag_module._doc_upsert(
        source_a,
        status="completed",
        active_version=1,
        canonical_title="林芝市地方立法条例",
        aliases="地方立法条例,林芝地方立法条例",
        filename_stem="林芝市地方立法条例_2017-05-26_2017-05-26",
    )
    rag_module._doc_upsert(
        source_b,
        status="completed",
        active_version=1,
        canonical_title="攀枝花市地方立法条例",
        aliases="地方立法条例,攀枝花地方立法条例",
        filename_stem="攀枝花市地方立法条例_2016-09-01_2016-10-01",
    )
    handler = rag_module.QueryHandler()

    async def fail_embed(_texts):
        raise AssertionError("embed should not run for ambiguous regulation clarification")

    monkeypatch.setattr(handler.embedding_service, "embed", fail_embed)

    result = asyncio.run(handler.retrieve("地方立法条例的立法程序", user_id="tester"))

    assert result["documents"] == []
    assert result["sources"] == []
    assert result["metadata"]["query_route"] == "document_clarification"
    assert result["metadata"]["refused"] == "document_target_required"
    assert len(result["metadata"]["candidate_sources"]) == 2
    assert "请先说明要查询哪一部法规文档" in result["metadata"]["clarification"]


def test_query_handler_retrieve_skips_doc_fallback_metadata(rag_module, monkeypatch):
    source = "陵水黎族自治县非物质文化遗产保护条例_2015-04-10_.docx"
    handler = rag_module.QueryHandler()

    async def fake_embed(_texts):
        return [[0.0, 0.0, 0.0]]

    monkeypatch.setattr(handler.embedding_service, "embed", fake_embed)
    monkeypatch.setattr(handler.vector_db, "search", lambda *args, **kwargs: [{
        "entity": {
            "source": source,
            "text": "第五章 奖励与处罚 对有贡献的单位和个人给予表彰或者奖励，并规定处罚措施、整改要求、适用条件、责任主体和执行程序。",
            "metadata": {"raw_text": "第五章 奖励与处罚 对有贡献的单位和个人给予表彰或者奖励，并规定处罚措施、整改要求、适用条件、责任主体和执行程序。", "section": "奖励与处罚"},
        },
        "score": 0.92,
    }])
    monkeypatch.setattr(rag_module, "_build_doc_recall_plan", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("doc fallback should not be used")))
    monkeypatch.setattr(rag_module, "_collect_lexical_candidates", lambda *args, **kwargs: [])

    result = asyncio.run(handler.retrieve("非遗保护条例 奖励与处罚", user_id="tester"))

    assert result["documents"]
    assert result["metadata"]["query_route"] == "light_rag"
    assert result["metadata"]["control_plane"] == "light"
    assert "doc_fallback_used" not in result["metadata"]


def test_query_handler_process_uses_rag_related_doc_when_target_doc_missing(rag_module, monkeypatch):
    handler = rag_module.QueryHandler()

    async def fake_embed(_texts):
        return [[0.0, 0.0, 0.0]]

    async def fake_chunk_rerank(_service, _query, hits, _top_k, _enable):
        return {"used": False, "hits": hits, "score_mode": "score"}

    async def fake_source_rerank(_service, _query, _docs, scores, _keep_n, _enable_rerank, **_kwargs):
        return {"used": False, "scores": scores}

    async def fake_generate_answer(query, evidence, qtype="other", max_tokens=None, answer_mode="target_hit"):
        return "其他文档中的相似要求如下"

    class DummyClient:
        def query(self, **kwargs):
            return []

    hit = {
        "entity": {
            "source": "b.docx",
            "text": "相同条款内容包括适用范围、责任主体、办理步骤、整改要求和处罚后果，能够作为当前问题的直接证据片段使用。",
            "metadata": {"raw_text": "相同条款内容包括适用范围、责任主体、办理步骤、整改要求和处罚后果，能够作为当前问题的直接证据片段使用。", "section": "第一章"},
        },
        "score": 0.88,
    }

    monkeypatch.setattr(handler.embedding_service, "embed", fake_embed)
    monkeypatch.setattr(handler.vector_db, "connect", lambda: None)
    handler.vector_db.client = DummyClient()
    monkeypatch.setattr(handler.vector_db, "search", lambda *args, **kwargs: [hit])
    monkeypatch.setattr(handler, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(rag_module, "_collect_lexical_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(rag_module, "_chunk_level_rerank", fake_chunk_rerank)
    monkeypatch.setattr(rag_module, "_merge_and_dedupe_hits", lambda hits, score_mode: hits)
    monkeypatch.setattr(rag_module, "_aggregate_doc_sections", lambda hits, score_mode: hits)
    monkeypatch.setattr(rag_module, "_apply_retrieval_filters", lambda docs, qfilters, fnames: docs)
    monkeypatch.setattr(rag_module, "_filter_low_relevance_sources", lambda hits, score_mode, query="", minimum_keep=None: hits)

    result = asyncio.run(handler.process("请根据 a.docx 说明相同条款内容", user_id="tester"))

    assert result["metadata"]["query_route"] == "light_rag"
    assert result["metadata"]["control_plane"] == "light"
    assert result["metadata"]["answer_mode"] == "rag_related_doc"
    assert result["metadata"]["target_sources"] == ["a.docx"]
    assert result["sources"][0]["source"] == "b.docx"
    assert result["answer"].startswith("未在当前可见知识库中命中目标文档")
    assert "根据相关文档证据" in result["answer"]
    assert "相同条款内容包括适用范围" in result["answer"]


def test_filter_display_sources_hides_stray_source_when_focus_is_single_doc(rag_module):
    docs = [
        {
            "entity": {
                "source": "a.docx",
                "text": "第一章 登记要求包括办理条件和时限。",
                "metadata": {"section": "登记要求", "raw_text": "第一章 登记要求包括办理条件和时限。"},
            },
            "score": 0.92,
        },
        {
            "entity": {
                "source": "a.docx",
                "text": "第二条 申请材料应当完整提交。",
                "metadata": {"section": "登记要求", "raw_text": "第二条 申请材料应当完整提交。"},
            },
            "score": 0.88,
        },
        {
            "entity": {
                "source": "b.docx",
                "text": "其他文档中有相似的概述性条款。",
                "metadata": {"section": "总则", "raw_text": "其他文档中有相似的概述性条款。"},
            },
            "score": 0.84,
        },
    ]

    result = rag_module._filter_display_sources(
        docs,
        score_mode="score",
        qfilters={},
        fnames=[],
        qtype="other",
        max_sources=3,
        observations={"intra_doc_focus_score": 0.82},
    )

    assert result
    assert all(item["entity"]["source"] == "a.docx" for item in result)


def test_evidence_observations_do_not_overreport_uncovered_for_regulation_wrapper_query(rag_module):
    docs = [
        {
            "entity": {
                "source": "heritage.pdf",
                "text": "第五章奖励与处罚明确了表彰奖励情形和多项处罚措施。",
                "metadata": {"section": "第五章奖励与处罚", "raw_text": "第五章奖励与处罚明确了表彰奖励情形和多项处罚措施。"},
            },
            "score": 0.91,
        }
    ]

    observations = rag_module._evidence_observations("保护条例的奖励与处罚", docs, qfilters={})

    assert observations["evidence_coverage_reason"] == "sufficient_evidence"
    assert observations["answer_scope"] == "full"
    assert "奖励与处罚" in observations["covered_aspects"]
    assert observations["uncovered_aspects"] == []


def test_evidence_observations_drop_wrapper_uncovered_when_section_hits(rag_module):
    docs = [
        {
            "entity": {
                "source": "dog.docx",
                "text": "第二十四条 禁止携犬进入医院、学校、商场等公共场所。",
                "metadata": {"section": "养犬行为规范", "raw_text": "第二十四条 禁止携犬进入医院、学校、商场等公共场所。"},
            },
            "score": 0.9,
        },
        {
            "entity": {
                "source": "dog.docx",
                "text": "第十二条 重点管理区内个人养犬的，每户限养一只。",
                "metadata": {"section": "养犬区划管理、免疫与登记", "raw_text": "第十二条 重点管理区内个人养犬的，每户限养一只。"},
            },
            "score": 0.88,
        },
    ]

    observations = rag_module._evidence_observations("对养犬有哪些限制", docs, qfilters={})

    assert observations["section_hit"] is True
    assert observations["evidence_coverage_reason"] == "sufficient_evidence"
    assert observations["answer_scope"] == "full"
    assert observations["uncovered_aspects"] == []


def test_evidence_observations_treat_document_title_as_covered_when_single_doc_hits(rag_module):
    source = "聊城市养犬管理条例_2020-06-15_2020-09-01.docx"
    rag_module._doc_upsert(
        source,
        status="completed",
        active_version=1,
        canonical_title="聊城市养犬管理条例",
        aliases="聊城市养犬管理条例,养犬管理条例",
        filename_stem="聊城市养犬管理条例_2020-06-15_2020-09-01",
    )
    docs = [
        {
            "entity": {
                "source": source,
                "text": "第二十四条 禁止携犬进入医院、学校、商场等公共场所。",
                "metadata": {"section": "养犬行为规范", "raw_text": "第二十四条 禁止携犬进入医院、学校、商场等公共场所。"},
            },
            "score": 0.9,
        },
        {
            "entity": {
                "source": source,
                "text": "第十二条 重点管理区内个人养犬的，每户限养一只。",
                "metadata": {"section": "养犬区划管理、免疫与登记", "raw_text": "第十二条 重点管理区内个人养犬的，每户限养一只。"},
            },
            "score": 0.88,
        },
    ]

    observations = rag_module._evidence_observations("聊城市养犬管理条例 对养犬有哪些限制", docs, qfilters={})

    assert observations["evidence_coverage_reason"] == "sufficient_evidence"
    assert observations["answer_scope"] == "full"
    assert observations["uncovered_aspects"] == []
    assert "聊城市养犬管理条例" in observations["covered_aspects"]


def test_query_handler_process_refuses_without_evidence_under_light_control_plane(rag_module, monkeypatch):
    handler = rag_module.QueryHandler()

    async def fake_embed(_texts):
        return [[0.0, 0.0, 0.0]]

    monkeypatch.setattr(handler.embedding_service, "embed", fake_embed)
    monkeypatch.setattr(handler.vector_db, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(rag_module, "_collect_lexical_candidates", lambda *args, **kwargs: [])

    result = asyncio.run(handler.process("这个系统怎么做缓存优化", user_id="tester"))

    assert result["metadata"]["control_plane"] == "light"
    assert result["metadata"]["answer_mode"] == "refusal"
    assert result["metadata"]["refused"] == "no_relevant_evidence"
    assert result["sources"] == []
    assert result["answer"] == "未检索到相关证据。"


def test_query_handler_process_returns_light_refusal_for_missing_evidence(rag_module, monkeypatch):
    handler = rag_module.QueryHandler()

    async def fake_embed(_texts):
        return [[0.0, 0.0, 0.0]]

    monkeypatch.setattr(handler.embedding_service, "embed", fake_embed)
    monkeypatch.setattr(handler.vector_db, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(rag_module, "_collect_lexical_candidates", lambda *args, **kwargs: [])

    result = asyncio.run(handler.process("条例核心条款", user_id="tester"))

    assert result["metadata"]["answer_mode"] == "refusal"
    assert result["metadata"]["refused"] == "no_relevant_evidence"
    assert result["answer"] == "未检索到相关证据。"


def test_query_handler_process_skips_heavy_source_control_layers(rag_module, monkeypatch):
    handler = rag_module.QueryHandler()

    async def fake_embed(_texts):
        return [[0.0, 0.0, 0.0]]

    async def fake_chunk_rerank(_service, _query, hits, _top_k, _enable):
        return {"used": False, "hits": hits, "score_mode": "score"}

    async def fake_generate_answer(query, evidence, qtype="other", max_tokens=None, answer_mode="target_hit"):
        assert answer_mode == "target_hit"
        return "可以确认相关条款内容[1]"

    hit = {
        "entity": {
            "source": "b.docx",
            "text": "奖励与处罚条款，明确了行政处罚、责令整改、警告、表彰奖励、适用条件、责任主体和执行要求，适用于相关违法行为。",
            "metadata": {"raw_text": "奖励与处罚条款，明确了行政处罚、责令整改、警告、表彰奖励、适用条件、责任主体和执行要求，适用于相关违法行为。", "section": "第六章"},
        },
        "score": 0.54,
    }

    monkeypatch.setattr(handler.embedding_service, "embed", fake_embed)
    monkeypatch.setattr(handler.vector_db, "search", lambda *args, **kwargs: [hit])
    monkeypatch.setattr(handler, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(rag_module, "_collect_lexical_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(rag_module, "_chunk_level_rerank", fake_chunk_rerank)
    monkeypatch.setattr(rag_module, "_source_level_rerank", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("source rerank should not be used")))
    monkeypatch.setattr(rag_module, "_final_admission_backstop", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("backstop should not be used")))
    monkeypatch.setattr(rag_module, "_merge_and_dedupe_hits", lambda hits, score_mode: hits)
    monkeypatch.setattr(rag_module, "_aggregate_doc_sections", lambda hits, score_mode: hits)
    monkeypatch.setattr(rag_module, "_apply_retrieval_filters", lambda docs, qfilters, fnames: docs)
    monkeypatch.setattr(rag_module, "_filter_low_relevance_sources", lambda hits, score_mode, query="", minimum_keep=None: hits)

    result = asyncio.run(handler.process("奖励与处罚有哪些规定", user_id="tester"))

    assert result["metadata"]["answer_mode"] == "target_hit"
    assert result["metadata"]["control_plane"] == "light"
    assert result["answer"] == "可以确认相关条款内容[1]"


def test_query_handler_process_keeps_low_score_hit_without_backstop(rag_module, monkeypatch):
    handler = rag_module.QueryHandler()

    async def fake_embed(_texts):
        return [[0.0, 0.0, 0.0]]

    async def fake_chunk_rerank(_service, _query, hits, _top_k, _enable):
        return {"used": False, "hits": hits, "score_mode": "score"}

    async def fake_generate_answer(query, evidence, qtype="other", max_tokens=None, answer_mode="target_hit"):
        assert qtype == "other"
        assert answer_mode == "target_hit"
        return "可以确认相关条款内容[1]"

    hit = {
        "entity": {
            "source": "b.docx",
            "text": "出租房安全管理条例中的法律责任部分明确了处罚主体、处罚方式、整改义务、罚款情形和追责要求。",
            "metadata": {"raw_text": "出租房安全管理条例中的法律责任部分明确了处罚主体、处罚方式、整改义务、罚款情形和追责要求。", "section": "第六章"},
        },
        "score": 0.61,
    }

    monkeypatch.setattr(handler.embedding_service, "embed", fake_embed)
    monkeypatch.setattr(handler.vector_db, "search", lambda *args, **kwargs: [hit])
    monkeypatch.setattr(handler, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(rag_module, "_collect_lexical_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(rag_module, "_chunk_level_rerank", fake_chunk_rerank)
    monkeypatch.setattr(rag_module, "_passes_relevance_cluster", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cluster gate should not be used")))
    monkeypatch.setattr(rag_module, "_merge_and_dedupe_hits", lambda hits, score_mode: hits)
    monkeypatch.setattr(rag_module, "_aggregate_doc_sections", lambda hits, score_mode: hits)
    monkeypatch.setattr(rag_module, "_apply_retrieval_filters", lambda docs, qfilters, fnames: docs)
    monkeypatch.setattr(rag_module, "_filter_low_relevance_sources", lambda hits, score_mode, query="", minimum_keep=None: hits)

    result = asyncio.run(handler.process("出租房安全管理条例 法律责任", user_id="tester"))

    assert result["metadata"]["answer_mode"] == "target_hit"
    assert result["metadata"]["control_plane"] == "light"
    assert result["answer"] == "可以确认相关条款内容[1]"


def test_query_handler_process_refuses_when_light_evidence_is_too_thin(rag_module, monkeypatch):
    handler = rag_module.QueryHandler()

    async def fake_embed(_texts):
        return [[0.0, 0.0, 0.0]]

    hit = {
        "entity": {
            "source": "b.docx",
            "text": "法律责任",
            "metadata": {"raw_text": "法律责任", "section": "第六章"},
        },
        "score": 0.61,
    }

    monkeypatch.setattr(handler.embedding_service, "embed", fake_embed)
    monkeypatch.setattr(handler.vector_db, "search", lambda *args, **kwargs: [hit])
    monkeypatch.setattr(rag_module, "_collect_lexical_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(rag_module, "_merge_and_dedupe_hits", lambda hits, score_mode: hits)
    monkeypatch.setattr(rag_module, "_aggregate_doc_sections", lambda hits, score_mode: hits)
    monkeypatch.setattr(rag_module, "_apply_retrieval_filters", lambda docs, qfilters, fnames: docs)
    monkeypatch.setattr(rag_module, "_filter_low_relevance_sources", lambda hits, score_mode, query="", minimum_keep=None: hits)

    result = asyncio.run(handler.process("出租房安全管理条例 法律责任", user_id="tester"))

    assert result["metadata"]["answer_mode"] == "refusal"
    assert result["metadata"]["refused"] == "insufficient_evidence"
    assert result["metadata"]["control_plane"] == "light"


def test_light_recall_preserves_visibility_filter_and_observability(rag_module, monkeypatch):
    handler = rag_module.QueryHandler()
    rag_module._doc_upsert("visible.docx", status="completed", active_version=1, filename_stem="visible")
    rag_module._doc_upsert("hidden.docx", status="vector_failed", active_version=None, filename_stem="hidden")

    async def fake_embed(_texts):
        return [[0.0, 0.0, 0.0]]

    hits = [
        {
            "entity": {
                "source": "hidden.docx",
                "text": "隐藏文档中的处罚条款不应被看到。",
                "metadata": {"raw_text": "隐藏文档中的处罚条款不应被看到。", "section": "第六章", "doc_version": 1},
            },
            "score": 0.95,
        },
        {
            "entity": {
                "source": "visible.docx",
                "text": "可见文档第六章明确了处罚主体、处罚方式、整改要求、适用条件和执行程序。",
                "metadata": {"raw_text": "可见文档第六章明确了处罚主体、处罚方式、整改要求、适用条件和执行程序。", "section": "第六章", "doc_version": 1},
            },
            "score": 0.88,
        },
    ]

    monkeypatch.setattr(handler.embedding_service, "embed", fake_embed)
    monkeypatch.setattr(handler.vector_db, "search", lambda *args, **kwargs: hits)
    monkeypatch.setattr(rag_module, "_collect_lexical_candidates", lambda *args, **kwargs: [])

    result = asyncio.run(handler.retrieve("处罚方式和整改要求是什么", user_id="tester"))

    assert [doc["source"] for doc in result["documents"]] == ["visible.docx"]
    assert result["metadata"]["visibility_enforced"] is True
    assert result["metadata"]["visibility_filtered"] == 1
    assert result["metadata"]["answer_scope"] == "full"


def test_light_retrieve_output_layer_keeps_explainable_top_chunks(rag_module, monkeypatch):
    handler = rag_module.QueryHandler()

    async def fake_embed(_texts):
        return [[0.0, 0.0, 0.0]]

    hits = []
    for idx in range(5):
        hits.append(
            {
                "entity": {
                    "source": "visible.docx",
                    "text": f"第六章 第{idx + 1}段说明处罚主体、处罚方式、整改要求和适用程序，第{idx + 1}段内容足够完整用于解释检索结果。",
                    "metadata": {"raw_text": f"第六章 第{idx + 1}段说明处罚主体、处罚方式、整改要求和适用程序，第{idx + 1}段内容足够完整用于解释检索结果。", "section": "第六章", "chunk_id": idx},
                },
                "score": 0.95 - idx * 0.01,
            }
        )

    monkeypatch.setattr(handler.embedding_service, "embed", fake_embed)
    monkeypatch.setattr(handler.vector_db, "search", lambda *args, **kwargs: hits)
    monkeypatch.setattr(rag_module, "_collect_lexical_candidates", lambda *args, **kwargs: [])

    result = asyncio.run(handler.retrieve("处罚方式和整改要求是什么", user_id="tester", top_k=4))

    assert len(result["documents"]) == 4
    assert result["metadata"]["answer_scope"] == "full"
    assert result["metadata"]["intra_doc_focus_score"] >= 0.8


def test_light_process_output_layer_prefers_section_focused_evidence(rag_module, monkeypatch):
    handler = rag_module.QueryHandler()

    async def fake_embed(_texts):
        return [[0.0, 0.0, 0.0]]

    captured = {}

    async def fake_generate_answer(query, evidence, qtype="other", max_tokens=None, answer_mode="target_hit"):
        captured["evidence"] = evidence
        return "第六章规定了处罚主体和整改要求[1]"

    hits = [
        {
            "entity": {
                "source": "reg.docx",
                "text": "总则部分介绍立法目的、适用范围和基本原则，这些内容不直接回答处罚主体。",
                "metadata": {"raw_text": "总则部分介绍立法目的、适用范围和基本原则，这些内容不直接回答处罚主体。", "section": "总则", "chunk_id": 0},
            },
            "score": 0.93,
        },
        {
            "entity": {
                "source": "reg.docx",
                "text": "第六章法律责任明确了处罚主体、处罚方式、整改要求、适用条件和执行程序。",
                "metadata": {"raw_text": "第六章法律责任明确了处罚主体、处罚方式、整改要求、适用条件和执行程序。", "section": "第六章 法律责任", "chunk_id": 1},
            },
            "score": 0.91,
        },
        {
            "entity": {
                "source": "other.docx",
                "text": "其他文档说明了背景介绍和一般原则，不涉及本题要求的处罚主体。",
                "metadata": {"raw_text": "其他文档说明了背景介绍和一般原则，不涉及本题要求的处罚主体。", "section": "背景", "chunk_id": 0},
            },
            "score": 0.89,
        },
    ]

    monkeypatch.setattr(handler.embedding_service, "embed", fake_embed)
    monkeypatch.setattr(handler.vector_db, "search", lambda *args, **kwargs: hits)
    monkeypatch.setattr(handler, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(rag_module, "_collect_lexical_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(rag_module, "_merge_and_dedupe_hits", lambda items, score_mode: items)
    monkeypatch.setattr(rag_module, "_aggregate_doc_sections", lambda items, score_mode: items)
    monkeypatch.setattr(rag_module, "_apply_retrieval_filters", lambda docs, qfilters, fnames: docs)
    monkeypatch.setattr(rag_module, "_filter_low_relevance_sources", lambda docs, score_mode, query="", minimum_keep=None: docs)

    result = asyncio.run(handler.process("第六章的处罚主体和整改要求是什么", user_id="tester"))

    assert "第六章 法律责任" in captured["evidence"]
    assert "总则" not in captured["evidence"]
    assert result["metadata"]["section_hit"] is True
    assert result["metadata"]["answer_scope"] == "full"


def test_light_process_partial_scope_explicitly_marks_uncovered_parts(rag_module, monkeypatch):
    handler = rag_module.QueryHandler()

    async def fake_embed(_texts):
        return [[0.0, 0.0, 0.0]]

    async def fake_generate_answer(query, evidence, qtype="other", max_tokens=None, answer_mode="target_hit"):
        return "证据显示已明确处罚主体和整改要求[1]"

    hit = {
        "entity": {
            "source": "reg.docx",
            "text": "第六章法律责任明确了处罚主体和整改要求，但未说明申诉流程或复议路径。",
            "metadata": {"raw_text": "第六章法律责任明确了处罚主体和整改要求，但未说明申诉流程或复议路径。", "section": "第六章 法律责任", "chunk_id": 1},
        },
        "score": 0.91,
    }

    monkeypatch.setattr(handler.embedding_service, "embed", fake_embed)
    monkeypatch.setattr(handler.vector_db, "search", lambda *args, **kwargs: [hit])
    monkeypatch.setattr(handler, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(rag_module, "_collect_lexical_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(rag_module, "_merge_and_dedupe_hits", lambda items, score_mode: items)
    monkeypatch.setattr(rag_module, "_aggregate_doc_sections", lambda items, score_mode: items)
    monkeypatch.setattr(rag_module, "_apply_retrieval_filters", lambda docs, qfilters, fnames: docs)
    monkeypatch.setattr(rag_module, "_filter_low_relevance_sources", lambda docs, score_mode, query="", minimum_keep=None: docs)

    result = asyncio.run(handler.process("第六章的处罚主体、整改要求和申诉流程是什么", user_id="tester"))

    assert result["metadata"]["answer_scope"] == "partial"
    assert result["metadata"]["evidence_coverage_reason"] == "partial_term_coverage"
    assert "已覆盖：" in result["answer"]
    assert "未覆盖：" in result["answer"]
    assert "申诉流程" in result["answer"]


def test_fusion_combined_sort_uses_dense_tiebreak_for_equal_scores(rag_module):
    dense_source_scores = {"weak.docx": 0.5942, "strong.docx": 0.5961}
    docs_all = [
        {"entity": {"source": "weak.docx", "text": "弱相关", "metadata": {}}, "score": 0.5942},
        {"entity": {"source": "strong.docx", "text": "强相关", "metadata": {}}, "score": 0.5961},
    ]
    combined = [(0.0138, 0), (0.0138, 1)]

    combined.sort(
        key=lambda x: (x[0], rag_module._source_dense_tiebreak_score(docs_all[x[1]], dense_source_scores)),
        reverse=True,
    )

    assert docs_all[combined[0][1]]["entity"]["source"] == "strong.docx"


def test_aggregate_doc_sections_keeps_chunks_independent(rag_module):
    hits = [
        {
            "entity": {
                "source": "doc.txt",
                "text": "第一段正文",
                "metadata": {"section": "第一章", "section_id": 1, "chunk_id_start": 1, "chunk_id_end": 1},
            },
            "score": 0.9,
        },
        {
            "entity": {
                "source": "doc.txt",
                "text": "第二段正文",
                "metadata": {"section": "第一章", "section_id": 1, "chunk_id_start": 2, "chunk_id_end": 2},
            },
            "score": 0.85,
        },
    ]

    result = rag_module._aggregate_doc_sections(hits, score_mode="score")

    assert len(result) == 2
    assert result[0]["entity"]["metadata"]["section"] == "第一章"
    assert result[0]["entity"]["metadata"]["section_title"] == "第一章"
    assert result[0]["entity"]["text"] == "第一段正文"
    assert result[1]["entity"]["text"] == "第二段正文"


def test_docs_for_query_context_prefers_section_metadata_version_for_most_queries(rag_module):
    merged_docs = [{"entity": {"source": "doc.txt", "text": f"chunk-{i}", "metadata": {}}, "score": 1.0} for i in range(9)]
    aggregated_docs = [{"entity": {"source": "doc.txt", "text": f"section-{i}", "metadata": {"section_title": "第一章"}}, "score": 1.0} for i in range(9)]

    result = rag_module._docs_for_query_context("other", merged_docs, aggregated_docs)

    assert result is aggregated_docs


def test_docs_for_query_context_keeps_chunks_for_short_or_locating_queries(rag_module):
    merged_docs = [{"entity": {"source": "doc.txt", "text": f"chunk-{i}", "metadata": {}}, "score": 1.0} for i in range(4)]
    aggregated_docs = [{"entity": {"source": "doc.txt", "text": f"section-{i}", "metadata": {"section_title": "第一章"}}, "score": 1.0} for i in range(4)]
    long_merged_docs = merged_docs * 3
    long_aggregated_docs = aggregated_docs * 3

    short_result = rag_module._docs_for_query_context("other", merged_docs, aggregated_docs)
    locating_result = rag_module._docs_for_query_context("single_doc_extract", long_merged_docs, long_aggregated_docs)

    assert short_result is merged_docs
    assert locating_result is long_merged_docs


def test_filter_low_relevance_sources_keeps_minimum_results_under_relative_threshold(rag_module, monkeypatch):
    monkeypatch.setattr(rag_module.config, "RECALL_RELATIVE_SCORE_RATIO", 0.9)
    monkeypatch.setattr(rag_module.config, "MIN_RELEVANCE_SCORE", 0.25)
    monkeypatch.setattr(rag_module.config, "RECALL_MIN_KEEP_N", 3)
    hits = [
        {"entity": {"source": "doc.txt", "text": "最强命中", "metadata": {"chunk_id": 0}}, "score": 0.90},
        {"entity": {"source": "doc.txt", "text": "次强命中", "metadata": {"chunk_id": 1}}, "score": 0.61},
        {"entity": {"source": "doc.txt", "text": "第三命中", "metadata": {"chunk_id": 2}}, "score": 0.34},
        {"entity": {"source": "doc.txt", "text": "尾部命中", "metadata": {"chunk_id": 3}}, "score": 0.28},
    ]

    result = rag_module._filter_low_relevance_sources(hits, score_mode="score", query="养犬登记")

    assert [item["entity"]["metadata"].get("chunk_id") for item in result] == [0, 1, 2]


def test_filter_low_relevance_sources_keeps_structural_hits_alive(rag_module, monkeypatch):
    monkeypatch.setattr(rag_module.config, "RECALL_RELATIVE_SCORE_RATIO", 0.9)
    monkeypatch.setattr(rag_module.config, "MIN_RELEVANCE_SCORE", 0.25)
    monkeypatch.setattr(rag_module.config, "RECALL_MIN_KEEP_N", 1)
    source = "legislation.docx"
    hits = [
        {
            "entity": {
                "source": source,
                "text": "本条例主要规定一般原则和适用范围。",
                "metadata": {"section": "总则", "chunk_id": 0, "raw_text": "本条例主要规定一般原则和适用范围。"},
            },
            "score": 0.95,
        },
        {
            "entity": {
                "source": source,
                "text": "立法程序包括立项、起草、审议和公布。",
                "metadata": {"section": "立法程序", "chunk_id": 1, "raw_text": "立法程序包括立项、起草、审议和公布。"},
            },
            "score": 0.64,
        },
        {
            "entity": {
                "source": source,
                "text": "第三条 起草部门应当提交审查材料并公开征求意见。",
                "metadata": {"section": "立法程序", "chunk_id": 2, "raw_text": "第三条 起草部门应当提交审查材料并公开征求意见。"},
            },
            "score": 0.58,
        },
        {
            "entity": {
                "source": source,
                "text": "监督管理部门负责日常巡查。",
                "metadata": {"section": "监督管理", "chunk_id": 3, "raw_text": "监督管理部门负责日常巡查。"},
            },
            "score": 0.57,
        },
    ]

    result = rag_module._filter_low_relevance_sources(hits, score_mode="score", query="立法程序需要提交哪些材料")

    assert [item["entity"]["metadata"].get("chunk_id") for item in result] == [0, 1, 2]


def test_intra_doc_chunk_rerank_prefers_section_and_keyword_matches(rag_module):
    source = "聊城市养犬管理条例_2020-06-15_2020-09-01.docx"
    rag_module._doc_upsert(
        source,
        status="completed",
        active_version=1,
        canonical_title="聊城市养犬管理条例",
        aliases="养犬管理条例",
        filename_stem="聊城市养犬管理条例_2020-06-15_2020-09-01",
    )
    hits = [
        {
            "entity": {
                "source": source,
                "text": "本章主要规定城市管理职责和一般要求。",
                "metadata": {"section": "总则", "raw_text": "本章主要规定城市管理职责和一般要求。", "chunk_id_start": 1, "chunk_id_end": 1},
            },
            "score": 0.95,
        },
        {
            "entity": {
                "source": source,
                "text": "单位和个人饲养犬只应当遵守免疫、登记、养犬证办理等规定。",
                "metadata": {"section": "养犬登记", "raw_text": "单位和个人饲养犬只应当遵守免疫、登记、养犬证办理等规定。", "chunk_id_start": 2, "chunk_id_end": 2},
            },
            "score": 0.9,
        },
    ]

    result = rag_module._intra_doc_chunk_rerank("养犬登记怎么办", hits, score_mode="score")

    assert result[0]["entity"]["metadata"]["section"] == "养犬登记"


def test_hybrid_structural_score_can_override_high_raw_score(rag_module):
    source = "xian_yangquan_tiaoli.docx"
    rag_module._doc_upsert(
        source,
        status="completed",
        active_version=1,
        canonical_title="西安市养犬管理条例",
        aliases="养犬管理条例",
        filename_stem="xian_yangquan_tiaoli",
    )
    hits = [
        {
            "entity": {
                "source": source,
                "text": "本章介绍立法背景和一般原则。",
                "metadata": {"section": "总则", "raw_text": "本章介绍立法背景和一般原则。"},
            },
            "score": 0.98,
        },
        {
            "entity": {
                "source": source,
                "text": "养犬登记应提交免疫证明并办理登记证。",
                "metadata": {"section": "养犬登记", "raw_text": "养犬登记应提交免疫证明并办理登记证。"},
            },
            "score": 0.72,
        },
    ]

    result = rag_module._intra_doc_chunk_rerank("养犬登记怎么办", hits, score_mode="score")

    assert result[0]["entity"]["metadata"]["section"] == "养犬登记"


def test_intra_doc_chunk_rerank_section_follow_bonus_promotes_same_section_body(rag_module):
    source = "legislation_process.docx"
    hits = [
        {
            "entity": {
                "source": source,
                "text": "立法程序包括立项、起草、审议和公布。",
                "metadata": {"section": "立法程序", "chunk_id_start": 10, "raw_text": "立法程序包括立项、起草、审议和公布。"},
            },
            "score": 0.72,
        },
        {
            "entity": {
                "source": source,
                "text": "第三条 起草部门应提交合法性审查材料并公开征求意见。",
                "metadata": {"section": "立法程序", "chunk_id_start": 11, "raw_text": "第三条 起草部门应提交合法性审查材料并公开征求意见。"},
            },
            "score": 0.68,
        },
        {
            "entity": {
                "source": source,
                "text": "本条例用于明确管理职责和基本原则。",
                "metadata": {"section": "总则", "chunk_id_start": 1, "raw_text": "本条例用于明确管理职责和基本原则。"},
            },
            "score": 0.9,
        },
    ]

    result = rag_module._intra_doc_chunk_rerank("立法程序", hits, score_mode="score", qtype="other")
    sections = [r["entity"].get("metadata", {}).get("section") for r in result[:2]]

    assert sections == ["立法程序", "立法程序"]


def test_intra_doc_chunk_rerank_penalizes_generic_sections_for_clause_lookup(rag_module):
    source = "dog_policy.docx"
    hits = [
        {
            "entity": {
                "source": source,
                "text": "本条例用于规范养犬管理，明确基本原则。",
                "metadata": {"section": "总则", "raw_text": "本条例用于规范养犬管理，明确基本原则。"},
            },
            "score": 0.96,
        },
        {
            "entity": {
                "source": source,
                "text": "违反规定未办理养犬登记的，处五百元罚款。",
                "metadata": {"section": "处罚", "raw_text": "违反规定未办理养犬登记的，处五百元罚款。"},
            },
            "score": 0.74,
        },
    ]

    result = rag_module._intra_doc_chunk_rerank("养犬处罚标准", hits, score_mode="score", qtype="other")

    assert result[0]["entity"]["metadata"]["section"] == "处罚"


def test_infer_rerank_profile_changes_with_query_intent(rag_module):
    assert rag_module._infer_rerank_profile("养犬处罚标准", "other") == "clause_lookup"
    assert rag_module._infer_rerank_profile("这部条例的主要内容", "summary") == "broad"
    assert rag_module._infer_rerank_profile("第五章奖惩有哪些规定", "other") == "section_lookup"
    assert rag_module._infer_rerank_profile("对养犬有哪些限制", "other") == "section_lookup"


def test_extract_section_query_targets_expands_business_chapters(rag_module):
    targets = rag_module._extract_section_query_targets("对养犬有哪些限制")

    assert "养犬行为规范" in targets
    assert "养犬区划管理、免疫与登记" in targets


def test_intra_doc_chunk_rerank_strengthens_section_lookup_queries(rag_module):
    source = "reg_rules.docx"
    hits = [
        {
            "entity": {
                "source": source,
                "text": "本条例规定了管理目的与基本原则。",
                "metadata": {"section": "总则", "chunk_id_start": 1, "raw_text": "本条例规定了管理目的与基本原则。"},
            },
            "score": 0.97,
        },
        {
            "entity": {
                "source": source,
                "text": "第五章 奖励与处罚：对违法行为予以罚款和其他处罚。",
                "metadata": {"section": "第五章 奖励与处罚", "chunk_id_start": 28, "raw_text": "第五章 奖励与处罚：对违法行为予以罚款和其他处罚。"},
            },
            "score": 0.75,
        },
        {
            "entity": {
                "source": source,
                "text": "对严重违法情形可以责令停业整顿。",
                "metadata": {"section": "第五章 奖励与处罚", "chunk_id_start": 29, "raw_text": "对严重违法情形可以责令停业整顿。"},
            },
            "score": 0.73,
        },
    ]

    result = rag_module._intra_doc_chunk_rerank("第五章奖惩有哪些规定", hits, score_mode="score", qtype="other")
    top_sections = [r["entity"].get("metadata", {}).get("section") for r in result[:2]]

    assert top_sections == ["第五章 奖励与处罚", "第五章 奖励与处罚"]


def test_intra_doc_chunk_rerank_preserves_source_group_order(rag_module):
    hits = [
        {"entity": {"source": "a.docx", "text": "第一段", "metadata": {"section": "总则"}}, "score": 0.8},
        {"entity": {"source": "b.docx", "text": "第二段", "metadata": {"section": "登记"}}, "score": 0.7},
        {"entity": {"source": "a.docx", "text": "养犬登记流程", "metadata": {"section": "登记"}}, "score": 0.79},
    ]

    result = rag_module._intra_doc_chunk_rerank("养犬登记", hits, score_mode="score")

    assert result[0]["entity"]["source"] == "a.docx"
    assert result[1]["entity"]["source"] == "a.docx"
    assert result[2]["entity"]["source"] == "b.docx"


def test_build_answer_prompt_other_target_hit_uses_relaxed_targeted_guidance(rag_module):
    prompt = rag_module._build_answer_prompt("养犬类的相关内容是什么", "[证据 1] 来源：doc | 章节：总则\n正文", "other", answer_mode="target_hit")

    assert "目标文档优先" in prompt
    assert "正文不足时，可结合章节标题、条款标题和文档标题归纳主题" in prompt
    assert "不要因为覆盖不全就直接拒答" in prompt


def test_build_answer_prompt_other_open_question_uses_graded_answering(rag_module):
    prompt = rag_module._build_answer_prompt("地方立法的相关内容是什么", "[证据 1] 来源：doc | 章节：总则\n正文", "other", answer_mode="target_hit")
    generic_prompt = rag_module._build_answer_prompt("地方立法的相关内容是什么", "[证据 1] 来源：doc | 章节：总则\n正文", "other", answer_mode="target_hit")
    open_prompt = rag_module._build_answer_prompt("地方立法的相关内容是什么", "[证据 1] 来源：doc | 章节：总则\n正文", "other", answer_mode="open_question")

    assert "只有当证据整体与问题明显无关时" in open_prompt
    assert "章节标题、条款标题和文档标题可以作为回答线索" in open_prompt
    assert generic_prompt != open_prompt


def test_chunk_level_rerank_skips_single_source_competition(rag_module):
    class DummyRerankService:
        async def rerank(self, query, documents, top_k):
            raise AssertionError("rerank should be skipped when all hits come from one source")

    hits = [
        {"entity": {"source": "doc.txt", "text": "第一段", "metadata": {}}, "distance": 0.1},
        {"entity": {"source": "doc.txt", "text": "第二段", "metadata": {}}, "distance": 0.2},
    ]

    result = asyncio.run(rag_module._chunk_level_rerank(DummyRerankService(), "问题", hits, top_k=2, enable_rerank=True))

    assert result["used"] is False
    assert result["hits"] == hits


def test_source_level_rerank_skips_single_source_score_map(rag_module):
    class DummyRerankService:
        async def rerank(self, query, documents, top_k):
            raise AssertionError("source rerank should be skipped for a single source")

    hits = [{"entity": {"source": "doc.txt", "text": "第一段", "metadata": {}}, "score": 0.8}]
    scores = {"doc.txt": 1.2}

    result = asyncio.run(rag_module._source_level_rerank(DummyRerankService(), "问题", hits, scores, keep_n=3, enable_rerank=True))

    assert result["used"] is False
    assert result["scores"] == scores


def test_doc_title_alias_hit_matches_filename_stem_substring(rag_module):
    source = "浏阳河管理条例__.docx"
    rag_module._doc_upsert(
        source,
        status="completed",
        active_version=1,
        canonical_title="浏阳河管理条例",
        aliases="浏阳河管理条例",
        filename_stem="浏阳河管理条例__",
    )

    assert rag_module._doc_title_alias_hit(source, "版本切换后只读新版本：浏阳河管理条例 奖励与处罚") is True
    assert rag_module._doc_title_alias_score(source, "浏阳河管理条例 奖励与处罚") > 0


def test_doc_title_alias_hit_matches_abbreviated_alias(rag_module):
    source = "陵水黎族自治县非物质文化遗产保护条例_2015-04-10_.docx"
    rag_module._doc_upsert(
        source,
        status="completed",
        active_version=1,
        canonical_title="陵水黎族自治县非物质文化遗产保护条例",
        aliases="非物质文化遗产保护条例",
        filename_stem="陵水黎族自治县非物质文化遗产保护条例_2015-04-10_",
    )

    assert rag_module._doc_title_alias_hit(source, "非遗保护条例 奖励与处罚") is True
    assert rag_module._doc_title_alias_score(source, "非遗保护条例 奖励与处罚") > 0


def test_lightweight_recall_uses_title_hit_as_source_narrowing(rag_module, monkeypatch):
    source = "聊城市养犬管理条例_2020-06-15_2020-09-01.docx"
    rag_module._doc_upsert(
        source,
        status="completed",
        active_version=1,
        canonical_title="聊城市养犬管理条例",
        aliases="聊城市养犬管理条例,养犬管理条例",
        filename_stem="聊城市养犬管理条例_2020-06-15_2020-09-01",
    )
    handler = rag_module.QueryHandler()
    captured = {}

    async def fake_embed(_texts):
        return [[0.0, 0.0, 0.0]]

    def fake_search(_embedding, top_k=0, filters=None):
        captured["filters"] = filters
        return []

    monkeypatch.setattr(handler.embedding_service, "embed", fake_embed)
    monkeypatch.setattr(handler.vector_db, "search", fake_search)
    monkeypatch.setattr(rag_module, "_collect_lexical_candidates", lambda *args, **kwargs: [])

    result = asyncio.run(
        handler._run_lightweight_recall(
            "聊城市养犬管理条例 对养犬有哪些限制",
            top_k=10,
            enable_rerank=True,
            filename_hints=[],
        )
    )

    assert result["retrieval_query"] == "对养犬有哪些限制"
    assert captured["filters"] == 'source == "聊城市养犬管理条例_2020-06-15_2020-09-01.docx"'


def test_source_level_rerank_skips_confident_title_anchor(rag_module):
    class DummyRerankService:
        async def rerank(self, query, documents, top_k):
            raise AssertionError("source rerank should be skipped for a confident title-anchored top source")

    hits = [
        {"entity": {"source": "a.docx", "text": "第一段", "metadata": {}}, "score": 0.62},
        {"entity": {"source": "b.docx", "text": "第二段", "metadata": {}}, "score": 0.60},
    ]
    scores = {"a.docx": 0.62, "b.docx": 0.60}
    dense_rank_map = {"a.docx": 0, "b.docx": 1}
    lex_rank_map = {"a.docx": 0, "b.docx": 1}
    source_signals = {"a.docx": {"title_hit": True}, "b.docx": {"lexical_hit": True}}

    result = asyncio.run(
        rag_module._source_level_rerank(
            DummyRerankService(),
            "问题",
            hits,
            scores,
            keep_n=3,
            enable_rerank=True,
            dense_rank_map=dense_rank_map,
            lex_rank_map=lex_rank_map,
            source_signals=source_signals,
        )
    )

    assert result["used"] is False
    assert result["scores"] == scores


def test_fusion_source_score_keeps_rule_bonus_from_reviving_zero_base_source(rag_module):
    query = "奖励与处罚"
    dense_rank_map = {"dense.docx": 0}
    lex_rank_map = {}
    source_count = {"dense.docx": 1}
    source_signals = {
        "dense.docx": {},
        "title_only.docx": {"title_hit": True, "doc_recall": True, "lexical_hit": True, "doc_prior": 1.0},
    }

    dense_score = rag_module._fusion_source_score(
        "dense.docx",
        query,
        dense_rank_map,
        lex_rank_map,
        source_count,
        source_signals,
        set(),
        set(),
        False,
    )
    title_only_score = rag_module._fusion_source_score(
        "title_only.docx",
        query,
        dense_rank_map,
        lex_rank_map,
        source_count,
        source_signals,
        set(),
        set(),
        False,
    )

    assert dense_score > 0
    assert title_only_score < dense_score
    assert title_only_score < 0.01


def test_build_document_ir_from_text_emits_structured_elements(rag_module):
    document_ir = rag_module._build_document_ir_from_text(
        "demo.md",
        "# 文档标题\n\n## 第一章\n正文段落\n- 清单项\n键: 值",
        metadata={"source_type": "unit"},
        parser_name="markdown",
        doc_version=2,
    )

    element_types = [element["element_type"] for element in document_ir["elements"]]

    assert document_ir["doc_version"] == 2
    assert element_types[:4] == ["title", "heading", "paragraph", "list_item"]
    assert element_types[-1] == "key_value"


def test_store_and_load_document_ir_roundtrip(rag_module):
    document_ir = rag_module._build_document_ir_from_text(
        "doc.txt",
        "第一段\n\n第二段",
        metadata={"parser_probe": {"route": "plain_text"}},
        parser_name="plain_text",
        doc_version=1,
    )

    rag_module._store_document_ir("doc.txt", document_ir)
    loaded = rag_module._load_document_ir("doc.txt", 1)

    assert loaded is not None
    assert loaded["source"] == "doc.txt"
    assert loaded["metadata"]["parser_probe"]["route"] == "plain_text"
    assert len(loaded["elements"]) == 2
    assert loaded["elements"][0]["text_raw"] == "第一段"


def test_document_ir_to_structured_items_carries_element_metadata(rag_module):
    document_ir = rag_module._new_document_ir("doc.txt", doc_version=3, parser_name="plain_text")
    rag_module._append_ir_element(
        document_ir,
        element_type="paragraph",
        text_raw="原文  文本",
        text_normalized="原文 文本",
        page_no=5,
        section_path=["第一章", "总则"],
    )

    items = rag_module._document_ir_to_structured_items(document_ir, chunk_size=50, overlap=10)

    assert len(items) == 1
    assert items[0]["element_type"] == "paragraph"
    assert items[0]["page_no"] == 5
    assert items[0]["section_path"] == ["第一章", "总则"]
    assert items[0]["section_node_id"] == "section::第一章 > 总则"
    assert items[0]["parent_section_id"] == "section::第一章"
    assert "原文 文本" in items[0]["fts_text"]


def test_index_document_stores_raw_text_in_sqlite_fts(rag_module, monkeypatch):
    inserted_docs = []

    class DummyEmbeddingService:
        async def embed_batched(self, texts, per_request=64, timeout=60, retries=2):
            return [[0.1, 0.2] for _ in texts]

    class DummyVectorDB:
        def connect(self):
            return None

        def insert(self, docs):
            inserted_docs.extend(docs)

    monkeypatch.setattr(rag_module, "EmbeddingService", DummyEmbeddingService)
    monkeypatch.setattr(rag_module, "VectorDBService", DummyVectorDB)
    monkeypatch.setattr(
        rag_module,
        "_contextualize_chunk_items",
        lambda filename, items: [{"chunk_id": 0, "section": "总则", "text": "文档标题：条例\n上文：甲\n原文：第一条 原文文本", "raw_text": "第一条 原文文本"}],
    )
    monkeypatch.setattr(rag_module, "_prepare_structured_items", lambda filename, text, chunk_size, overlap, document_ir=None: [{"text": text}])

    count = asyncio.run(rag_module.index_document("demo.docx", "ignored"))

    conn = rag_module._lex_db_connect()
    fts_text = conn.execute("SELECT text FROM chunks_fts").fetchone()[0]

    assert count == 1
    assert inserted_docs[0]["text"] == "文档标题：条例\n上文：甲\n原文：第一条 原文文本"
    assert inserted_docs[0]["metadata"]["raw_text"] == "第一条 原文文本"
    assert fts_text == "第一条 原文文本"


def test_reconcile_uses_raw_text_for_sqlite_fts(rag_module):
    catalog = {
        "demo.docx": {
            "source": "demo.docx",
            "created_at": "2026-04-08T10:00:00",
            "chunks_indexed": 1,
            "active_version": 1,
            "doc_type": "regulation",
            "topics": [],
            "rows": [
                {
                    "text": "文档标题：条例\n上文：甲\n原文：第一条 原文文本",
                    "metadata": {"chunk_id": 0, "section": "总则", "raw_text": "第一条 原文文本", "doc_version": 1},
                    "created_at": "2026-04-08T10:00:00",
                    "chunk_id": 0,
                }
            ],
        }
    }

    rag_module._reconcile_sqlite_from_catalog(catalog, prune_sqlite_orphans=False)

    conn = rag_module._lex_db_connect()
    fts_text = conn.execute("SELECT text FROM chunks_fts").fetchone()[0]

    assert fts_text == "第一条 原文文本"


def test_probe_file_for_parser_prefers_signature_over_extension(rag_module):
    probe = rag_module._probe_file_for_parser("report.txt", b"%PDF-1.7\n1 0 obj\n")

    assert probe["signature"] == "pdf"
    assert probe["detected_ext"] == ".pdf"
    assert probe["mime_type"] == "application/pdf"
    assert probe["route"] == "pdf_digital_fast"


def test_probe_file_for_parser_prefers_digital_fast_for_text_dense_image_majority_pdf(rag_module, monkeypatch):
    monkeypatch.setattr(
        rag_module,
        "_probe_pdf_document",
        lambda content: {
            "page_count": 8,
            "is_scanned_pdf": False,
            "image_page_majority": True,
            "avg_text_chars_per_page": 1258.875,
            "multi_column": True,
            "table_dense": False,
        },
    )
    monkeypatch.setattr(rag_module.config, "PDF_OCR_MAX_TEXT_CHARS_PER_PAGE", 300.0)

    probe = rag_module._probe_file_for_parser("dense.pdf", b"%PDF-1.7\n1 0 obj\n")

    assert probe["route"] == "pdf_digital_fast"


def test_probe_file_for_parser_keeps_ocr_for_low_text_image_majority_pdf(rag_module, monkeypatch):
    monkeypatch.setattr(
        rag_module,
        "_probe_pdf_document",
        lambda content: {
            "page_count": 6,
            "is_scanned_pdf": False,
            "image_page_majority": True,
            "avg_text_chars_per_page": 40.0,
            "multi_column": False,
            "table_dense": False,
        },
    )
    monkeypatch.setattr(rag_module.config, "PDF_OCR_MAX_TEXT_CHARS_PER_PAGE", 300.0)
    monkeypatch.setattr(rag_module.config, "OCR_SERVICE_URL", "http://ocr-host/api/ocr_text")

    probe = rag_module._probe_file_for_parser("scan-like.pdf", b"%PDF-1.7\n1 0 obj\n")

    assert probe["route"] == "pdf_ocr_layout"


def test_probe_file_for_parser_routes_cid_garbled_pdf_to_ocr(rag_module, monkeypatch):
    monkeypatch.setattr(
        rag_module,
        "_probe_pdf_document",
        lambda content: {
            "page_count": 5,
            "is_scanned_pdf": False,
            "image_page_majority": False,
            "garbled_text_pages": 3,
            "garbled_text_majority": True,
            "avg_text_chars_per_page": 820.0,
            "multi_column": False,
            "table_dense": False,
        },
    )
    monkeypatch.setattr(rag_module.config, "OCR_SERVICE_URL", "http://ocr-host/api/ocr_text")

    probe = rag_module._probe_file_for_parser("garbled.pdf", b"%PDF-1.7\n1 0 obj\n")

    assert probe["route"] == "pdf_ocr_layout"


def test_pdf_noise_text_filters_page_numbers_and_symbol_soup(rag_module):
    assert rag_module._is_pdf_noise_text("64", page_no=64) is True
    assert rag_module._is_pdf_noise_text("  63  ", page_no=64) is True
    assert rag_module._is_pdf_noise_text("一 ∴ ● _ _ ● △ ● _ ●", page_no=8) is True
    assert rag_module._is_pdf_noise_text("一\n一\ni\n一\n一\n一\n一\n亠\ni\n话\n扌\n氵\n芒\n艹\n扌", page_no=8) is True
    assert rag_module._is_pdf_noise_text("第六十三条 本条例自公布之日起施行。", page_no=8) is False


def test_document_ir_to_structured_items_skips_empty_figures_and_pdf_noise(rag_module):
    document_ir = rag_module._new_document_ir("sample.pdf", parser_name="pymupdf", doc_version=1)
    rag_module._append_ir_element(
        document_ir,
        element_type="figure",
        text_raw="",
        page_no=1,
        parser_name="pymupdf",
        json_payload={"kind": "image_block"},
    )
    rag_module._append_ir_element(
        document_ir,
        element_type="paragraph",
        text_raw="64",
        page_no=64,
        parser_name="pymupdf",
    )
    rag_module._append_ir_element(
        document_ir,
        element_type="paragraph",
        text_raw="第六十三条 本条例自公布之日起施行。",
        page_no=8,
        parser_name="pymupdf",
    )

    items = rag_module._document_ir_to_structured_items(document_ir, chunk_size=500, overlap=100)

    assert len(items) == 1
    assert "第六十三条 本条例自公布之日起施行。" in items[0]["text"]
    assert "image_block" not in items[0]["text"]
    assert "页码：64" not in items[0]["text"]


def test_extract_docx_document_ir_preserves_structures(rag_module):
    docx = pytest.importorskip("docx")
    Document = docx.Document

    document = Document()
    document.sections[0].header.paragraphs[0].text = "页眉信息"
    document.sections[0].footer.paragraphs[0].text = "页脚信息"
    document.add_heading("文档总标题", level=1)
    document.add_paragraph("正文第一段")
    document.add_page_break()
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "字段"
    table.cell(0, 1).text = "值"
    table.cell(1, 0).text = "名称"
    table.cell(1, 1).text = "条例"

    stream = io.BytesIO()
    document.save(stream)

    document_ir = rag_module.extract_document_ir_from_file("sample.docx", stream.getvalue())
    element_types = [element["element_type"] for element in document_ir["elements"]]

    assert document_ir["metadata"]["parser_probe"]["route"] == "docx_structured"
    assert "table" in element_types
    assert "page_break" in element_types
    assert any(element["text_raw"] == "页眉信息" and element["section_path"] == ["header_1"] for element in document_ir["elements"])
    assert any(element["text_raw"] == "页脚信息" and element["section_path"] == ["footer_1"] for element in document_ir["elements"])


def test_extract_docx_document_ir_infers_plain_heading_and_section_path(rag_module):
    docx = pytest.importorskip("docx")
    Document = docx.Document
    WD_ALIGN_PARAGRAPH = pytest.importorskip("docx.enum.text").WD_ALIGN_PARAGRAPH

    document = Document()
    heading = document.add_paragraph()
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = heading.add_run("总则")
    run.bold = True
    document.add_paragraph("第一条 为了规范管理，制定本条例。")

    stream = io.BytesIO()
    document.save(stream)

    document_ir = rag_module.extract_document_ir_from_file("plain-heading.docx", stream.getvalue())

    assert any(element["element_type"] == "heading" and element["text_raw"] == "总则" for element in document_ir["elements"])
    assert any(element["text_raw"].startswith("第一条") and element["section_path"] == ["总则"] for element in document_ir["elements"])


def test_extract_docx_document_ir_isolates_toc_entries(rag_module):
    docx = pytest.importorskip("docx")
    Document = docx.Document
    WD_ALIGN_PARAGRAPH = pytest.importorskip("docx.enum.text").WD_ALIGN_PARAGRAPH

    document = Document()
    document.add_paragraph("目 录")
    document.add_paragraph("总则")
    document.add_paragraph("法律责任")
    heading = document.add_paragraph()
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = heading.add_run("总则")
    run.bold = True
    document.add_paragraph("第一条 为了规范管理，制定本条例。")

    stream = io.BytesIO()
    document.save(stream)

    document_ir = rag_module.extract_document_ir_from_file("toc.docx", stream.getvalue())

    assert any(element["text_raw"] == "目 录" and element["section_path"] == ["toc"] for element in document_ir["elements"])
    assert any(element["text_raw"] == "法律责任" and element["section_path"] == ["toc"] for element in document_ir["elements"])
    assert any(element["element_type"] == "heading" and element["text_raw"] == "总则" and element["section_path"] == [] for element in document_ir["elements"])
    items = rag_module._document_ir_to_structured_items(document_ir, chunk_size=200, overlap=50)
    assert all(item["section"] != "toc" for item in items)


def test_extract_xlsx_document_ir_preserves_table_formula_and_hidden_state(rag_module):
    openpyxl = pytest.importorskip("openpyxl")
    Workbook = openpyxl.Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "预算表"
    sheet["A1"] = "项目"
    sheet["B1"] = "金额"
    sheet["C1"] = "备注"
    sheet["A2"] = "A项"
    sheet["B2"] = 10
    sheet["C2"] = "首行"
    sheet["A3"] = "B项"
    sheet["B3"] = "=SUM(B2,5)"
    sheet["C3"] = "公式行"
    sheet.row_dimensions[3].hidden = True
    sheet.merge_cells("D1:E1")
    sheet["D1"] = "合并头"

    stream = io.BytesIO()
    workbook.save(stream)

    document_ir = rag_module.extract_document_ir_from_file("budget.xlsx", stream.getvalue())
    table_elements = [element for element in document_ir["elements"] if element["element_type"] == "table"]

    assert document_ir["metadata"]["parser_probe"]["route"] == "xlsx_structured"
    assert table_elements
    payload = table_elements[0]["json_payload"]
    assert payload["table_json"]["hidden_policy"] == "preserve_with_visibility_flags"
    assert any(row["row_index"] == 3 and row["hidden"] for row in payload["table_json"]["rows"])
    assert any(formula["coord"] == "B3" and formula["formula"] == "=SUM(B2,5)" for formula in payload["table_json"]["formulas"])
    assert payload["table_json"]["range"]


def test_extract_image_document_ir_uses_external_ocr_service(rag_module, monkeypatch):
    ocr_calls = []

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"texts": ["图片第一行", "图片第二行"], "meta": {"build_id": "ocr-image"}}

    def fake_post(url, headers=None, json=None, timeout=None):
        ocr_calls.append({"url": url, "payload": json, "timeout": timeout})
        assert Path(json["image_path"]).exists()
        return DummyResponse()

    monkeypatch.setattr(rag_module.config, "OCR_SERVICE_URL", "http://ocr-host/api/ocr_text")
    monkeypatch.setattr("requests.post", fake_post)

    document_ir = rag_module.extract_document_ir_from_file("scan.png", b"\x89PNG\r\n\x1a\nmock-image")

    assert document_ir["metadata"]["parser_probe"]["route"] == "image_ocr_layout"
    assert document_ir["metadata"]["parser_probe"]["parser_backend"] == "external_http_ocr"
    assert [element["text_raw"] for element in document_ir["elements"]] == ["图片第一行", "图片第二行"]
    assert all(element["ocr_used"] for element in document_ir["elements"])
    assert ocr_calls[0]["url"] == "http://ocr-host/api/ocr_text"
    assert ocr_calls[0]["payload"]["mode"] == "general"
    assert ocr_calls[0]["payload"]["lang"] == "auto"
    assert not Path(ocr_calls[0]["payload"]["image_path"]).exists()


def test_extract_scanned_pdf_document_ir_uses_external_ocr_service(rag_module, monkeypatch, tmp_path):
    page_dir = tmp_path / "ocr-pages"
    page_dir.mkdir()
    page1 = page_dir / "page_0001.png"
    page2 = page_dir / "page_0002.png"
    page1.write_bytes(b"page-1")
    page2.write_bytes(b"page-2")

    def fake_probe(filename, content):
        return {
            "filename": filename,
            "extension": ".pdf",
            "detected_ext": ".pdf",
            "mime_type": "application/pdf",
            "signature": "pdf",
            "file_size": len(content),
            "page_count": 2,
            "is_scanned_pdf": True,
            "image_page_majority": True,
            "multi_column": False,
            "table_dense": False,
            "route": "pdf_ocr_layout",
            "parser_backend": "external_http_ocr",
            "backend_candidates": ["external_http_ocr"],
            "degraded": False,
        }

    class DummyResponse:
        def __init__(self, texts):
            self._texts = texts

        def raise_for_status(self):
            return None

        def json(self):
            return {"texts": self._texts, "meta": {"build_id": "ocr-pdf"}}

    def fake_post(url, headers=None, json=None, timeout=None):
        image_name = Path(json["image_path"]).name
        if image_name == "page_0001.png":
            return DummyResponse(["第一页内容"])
        return DummyResponse(["第二页内容"])

    monkeypatch.setattr(rag_module.config, "OCR_SERVICE_URL", "http://ocr-host/api/ocr_text")
    monkeypatch.setattr(rag_module, "_probe_file_for_parser", fake_probe)
    monkeypatch.setattr(
        rag_module,
        "_render_pdf_pages_for_ocr",
        lambda content: [
            {"page_no": 1, "image_path": str(page1)},
            {"page_no": 2, "image_path": str(page2)},
        ],
    )
    monkeypatch.setattr("requests.post", fake_post)

    document_ir = rag_module.extract_document_ir_from_file("scan.pdf", b"%PDF-1.4\nmock")
    element_types = [element["element_type"] for element in document_ir["elements"]]

    assert document_ir["metadata"]["parser_probe"]["route"] == "pdf_ocr_layout"
    assert document_ir["metadata"]["parser_probe"]["parser_backend"] == "external_http_ocr"
    assert element_types == ["paragraph", "page_break", "paragraph"]
    assert [element["text_raw"] for element in document_ir["elements"] if element["element_type"] == "paragraph"] == ["第一页内容", "第二页内容"]
    assert not page_dir.exists()


def test_parse_pdf_fast_document_ir_uses_visual_layout_for_toc_and_heading(rag_module, monkeypatch):
    def make_block(text, bbox, size, font="Songti-Bold"):
        return {
            "type": 0,
            "bbox": list(bbox),
            "lines": [
                {
                    "spans": [
                        {
                            "text": line,
                            "size": size,
                            "font": font,
                            "flags": 16 if "Bold" in font else 0,
                        }
                    ]
                }
                for line in text.split("\n")
            ],
        }

    class DummyRect:
        def __init__(self, width, height):
            self.width = width
            self.height = height

    class DummyPage:
        def __init__(self, blocks, width=1000, height=1600):
            self._blocks = blocks
            self.rect = DummyRect(width, height)

        def get_text(self, mode):
            assert mode == "dict"
            return {"blocks": self._blocks}

    class DummyDocument(list):
        @property
        def page_count(self):
            return len(self)

        def close(self):
            return None

    class DummyFitzModule:
        @staticmethod
        def open(stream=None, filetype=None):
            assert filetype == "pdf"
            return DummyDocument(
                [
                    DummyPage(
                        [
                            make_block("林芝市地方立法条例", (180, 80, 820, 150), 28),
                            make_block("目 录", (390, 200, 610, 245), 20),
                            make_block("第一章 总则 ........ 1", (260, 300, 740, 340), 16, font="Songti-Regular"),
                        ]
                    ),
                    DummyPage(
                        [
                            make_block("第一章 总则", (320, 120, 680, 170), 24),
                            make_block("第一条 为了规范管理，制定本条例。", (110, 260, 900, 320), 14, font="Songti-Regular"),
                        ]
                    ),
                ]
            )

    monkeypatch.setattr(rag_module, "_module_available", lambda name: name == "fitz")
    monkeypatch.setitem(sys.modules, "fitz", DummyFitzModule)

    document_ir = rag_module._parse_pdf_fast_document_ir("visual.pdf", b"%PDF-1.7", None, 1, backend="pymupdf")

    assert any(element["text_raw"] == "目 录" and element["section_path"] == ["toc"] for element in document_ir["elements"])
    assert any(element["text_raw"] == "第一章 总则 ........ 1" and element["section_path"] == ["toc"] for element in document_ir["elements"])
    assert any(element["element_type"] == "heading" and element["text_raw"] == "第一章 总则" and element["section_path"] == [] for element in document_ir["elements"])
    assert any(element["text_raw"].startswith("第一条") and element["section_path"] == ["第一章 总则"] for element in document_ir["elements"])


def test_build_ocr_document_ir_uses_structured_lines_for_heading_inference(rag_module):
    document_ir = rag_module._build_ocr_document_ir(
        "scan.pdf",
        metadata=None,
        doc_version=1,
        parser_name="external_http_ocr",
        probe={"route": "pdf_ocr_layout", "parser_backend": "external_http_ocr"},
        page_results=[
            {
                "page_no": 1,
                "meta": {"page_width": 1000, "page_height": 1600},
                "lines": [
                    {"text": "第一章 总则", "bbox": [260, 120, 740, 180], "font_size": 28, "confidence": 0.99},
                    {"text": "第一条 为了规范管理，制定本条例。", "bbox": [100, 250, 900, 310], "font_size": 16, "confidence": 0.96},
                ],
            }
        ],
        empty_notice="empty",
    )

    heading = next(element for element in document_ir["elements"] if element["element_type"] == "heading")
    article = next(element for element in document_ir["elements"] if element["element_type"] == "paragraph")

    assert heading["text_raw"] == "第一章 总则"
    assert heading["section_path"] == []
    assert article["section_path"] == ["第一章 总则"]
    assert article["bbox"] == {"x0": 100.0, "y0": 250.0, "x1": 900.0, "y1": 310.0}
    assert article["ocr_confidence"] == pytest.approx(0.96)


def test_visual_heading_heuristics_reject_sentence_like_lines(rag_module):
    layout = {"font_size": 26.0, "is_centered": True, "top_ratio": 0.18, "width_ratio": 0.48}
    profile = {"body_font_size": 14.0}

    assert rag_module._infer_visual_heading_level("结合本市实际，制定本条例。", layout, profile, "第一条 为了规范管理") is None
    assert rag_module._infer_visual_heading_level("结合本市实际袁制定本条例遥", layout, profile, "第一条 为了规范管理") is None
    assert rag_module._infer_visual_heading_level("这是一个超过三十个字符并且即使视觉很强也不应当被当作标题的测试短句", layout, profile, "第一条 为了规范管理") is None


def test_build_ocr_document_ir_truncates_legal_appendix_sections(rag_module):
    document_ir = rag_module._build_ocr_document_ir(
        "law.pdf",
        metadata=None,
        doc_version=1,
        parser_name="external_http_ocr",
        probe={"route": "pdf_ocr_layout", "parser_backend": "external_http_ocr"},
        page_results=[
            {
                "page_no": 1,
                "meta": {"page_width": 1000, "page_height": 1600},
                "lines": [
                    {"text": "第一章摇总摇摇则", "bbox": [260, 120, 740, 180], "font_size": 28},
                    {"text": "第一条摇为了加强管理，制定本条例。", "bbox": [100, 250, 900, 310], "font_size": 16},
                    {"text": "关于叶柳州市城市绿化条例曳的说明", "bbox": [180, 520, 820, 580], "font_size": 24},
                    {"text": "一、立法背景", "bbox": [110, 640, 500, 690], "font_size": 16},
                ],
            }
        ],
        empty_notice="empty",
    )

    assert any(element["element_type"] == "heading" and element["text_raw"] == "第一章 总 则" for element in document_ir["elements"])
    assert any(element["element_type"] == "heading" and element["text_raw"] == "关于《柳州市城市绿化条例》的说明" and element["section_path"] == ["appendix"] for element in document_ir["elements"])
    assert any(element["text_raw"] == "一、立法背景" and element["section_path"] == ["appendix"] for element in document_ir["elements"])
