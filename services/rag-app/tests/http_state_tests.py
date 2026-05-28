import os
import importlib.util
from datetime import datetime
from fastapi.testclient import TestClient
import uuid
import asyncio

os.environ["TEST_LEX_ONLY"] = "true"
os.environ.setdefault("APP_ENV", "test_local")
os.environ.setdefault("EMBEDDING_SERVICE_URL", "http://127.0.0.1:8001")
os.environ.setdefault("RERANK_SERVICE_URL", "http://127.0.0.1:8002")
os.environ.setdefault("MILVUS_HOST", "127.0.0.1")
os.environ.setdefault("MILVUS_PORT", "19530")
MAIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "main.py"))
spec = importlib.util.spec_from_file_location("rag_http_main", MAIN_PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
client = TestClient(m.app)

def test_publish_and_delete():
    fname = "版本发布测试.docx"
    r = client.post("/documents", json={"filename": fname, "content": "# Sheet: A\na,b\n# Sheet: B\nx,y", "metadata": {"doc_type": "regulation"}})
    assert r.status_code == 200
    src = m._safe_filename(fname)
    doc = m._doc_get(src)
    assert (doc.get("status") or "unknown") != "not_found"
    v_next = m._doc_next_version(src)
    m._doc_upsert(src, status="vector_pending", pending_version=v_next)
    doc2 = m._doc_get(src)
    assert doc2.get("pending_version") == v_next
    # 发布：active_version==pending_version，清空 pending_version，状态 completed
    m._doc_upsert(src, status="completed", active_version=v_next, pending_version=None, last_error=None)
    doc3 = m._doc_get(src)
    assert doc3.get("active_version") == v_next
    assert doc3.get("pending_version") in (None, v_next if v_next is None else None)
    assert doc3.get("status") == "completed"
    # 删除：成功时应硬删除控制面/本地索引记录，失败时保留 delete_failed
    dr = client.delete(f"/documents/{fname}")
    assert dr.status_code in (200, 503, 500)
    doc4 = m._doc_get(src)
    ds = m._lex_db_get_status(src)
    conn = m._lex_db_connect()
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

if __name__ == "__main__":
    test_publish_and_delete()
