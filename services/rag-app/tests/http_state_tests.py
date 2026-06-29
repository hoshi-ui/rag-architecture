import os
from datetime import datetime
from fastapi.testclient import TestClient

os.environ["TEST_LEX_ONLY"] = "true"
os.environ.setdefault("APP_ENV", "test_local")
os.environ.setdefault("EMBEDDING_SERVICE_URL", "http://127.0.0.1:8001")
os.environ.setdefault("RERANK_SERVICE_URL", "http://127.0.0.1:8002")
os.environ.setdefault("MILVUS_HOST", "127.0.0.1")
os.environ.setdefault("MILVUS_PORT", "19530")

from main import app
from app.runtime import runtime_context


def _assert_publish_and_delete(client, docs):
    fname = "版本发布测试.docx"
    r = client.post("/documents", json={"filename": fname, "content": "# Sheet: A\na,b\n# Sheet: B\nx,y", "metadata": {"doc_type": "regulation"}})
    assert r.status_code == 200
    src = docs.safe_filename(fname)
    doc = docs.doc_get(src)
    assert (doc.get("status") or "unknown") != "not_found"
    v_next = docs.doc_next_version(src)
    docs.doc_upsert(src, status="vector_pending", pending_version=v_next)
    doc2 = docs.doc_get(src)
    assert doc2.get("pending_version") == v_next
    # 发布：active_version==pending_version，清空 pending_version，状态 completed
    docs.doc_upsert(src, status="completed", active_version=v_next, pending_version=None, last_error=None)
    doc3 = docs.doc_get(src)
    assert doc3.get("active_version") == v_next
    assert doc3.get("pending_version") in (None, v_next if v_next is None else None)
    assert doc3.get("status") == "completed"
    # 删除：成功时应硬删除控制面/本地索引记录，失败时保留 delete_failed
    dr = client.delete(f"/documents/{fname}")
    assert dr.status_code in (200, 503, 500)
    doc4 = docs.doc_get(src)
    ds = docs.lex_db_get_status(src)
    conn = docs.lex_db_connect()
    doc_rows = conn.execute("SELECT COUNT(*) FROM documents WHERE source = ?", (src,)).fetchone()[0]
    doc_fts_rows = conn.execute("SELECT COUNT(*) FROM documents_fts WHERE filename = ?", (src,)).fetchone()[0]
    chunk_rows = conn.execute("SELECT COUNT(*) FROM chunks_meta WHERE source = ?", (src,)).fetchone()[0]
    if dr.status_code == 200:
        assert doc4.get("status") is None
        assert ds is None
        assert doc_rows == 0
        assert doc_fts_rows == 0
        assert chunk_rows == 0
    else:
        assert (doc4.get("status") == "delete_failed") or (ds == "delete_failed")
    print("HTTP control-plane state tests passed", datetime.now().isoformat())

def test_publish_and_delete(client, document_service):
    _assert_publish_and_delete(client, document_service)


if __name__ == "__main__":
    _assert_publish_and_delete(TestClient(app), runtime_context().document_service())
