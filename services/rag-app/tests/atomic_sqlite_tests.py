import os
import time
import sqlite3
import importlib.util
from datetime import datetime

from fastapi.testclient import TestClient

# dynamic import of main.py to avoid module name issues
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
MAIN_PATH = os.path.abspath(os.path.join(THIS_DIR, "..", "main.py"))
spec = importlib.util.spec_from_file_location("rag_app_main", MAIN_PATH)
rag_app_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_app_main)

app = rag_app_main.app
DB_CANDIDATES = ["/app/uploads/lexical_index.db",
                 os.path.abspath(os.path.join(THIS_DIR, "..", "uploads", "lexical_index.db")),
                 os.path.abspath(os.path.join(THIS_DIR, "..", "..", "rag-app", "uploads", "lexical_index.db"))]
def _connect_db():
    for p in DB_CANDIDATES:
        try:
            if os.path.exists(p):
                return sqlite3.connect(p)
        except Exception:
            pass
    # fallback: try first path
    return sqlite3.connect(DB_CANDIDATES[0])


def url_encode(name: str) -> str:
    import urllib.parse
    return urllib.parse.quote(name)


FNAME_BASE = "测试条例_示例"


def reset_fault():
    for k in ("RAG_FAULT_INJECT_STAGE", "LEX_DB_CRASH_INJECT_STAGE"):
        if k in os.environ:
            del os.environ[k]


def counts_for_source(src: str):
    conn = _connect_db()
    meta = conn.execute("SELECT COUNT(*) FROM chunks_meta WHERE source=?", (src,)).fetchone()[0]
    fts = conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks_meta WHERE source=?)", (src,)).fetchone()[0]
    return meta, fts


def test_fault(stage: str):
    reset_fault()
    os.environ["RAG_FAULT_INJECT_STAGE"] = stage
    client = TestClient(app)
    fname = f"{FNAME_BASE}_{stage}.docx"
    # pre-clean any leftovers
    r = client.post("/documents", json={"filename": fname, "content": "# Sheet: A\nabc", "metadata": {"doc_type": "regulation"}})
    assert r.status_code == 200, r.text
    # wait a bit for background task
    time.sleep(0.5)
    # no half-write
    meta, fts = counts_for_source(fname)
    assert meta == fts == 0, f"Expected rollback for stage={stage}, got meta={meta}, fts={fts}"
    conn = _connect_db()
    row = conn.execute("SELECT status FROM doc_status WHERE source = ?", (fname,)).fetchone()
    status = (row[0] if row else None)
    assert status in (None, "vector_failed", "accepted", "reindexing"), f"status unexpected: {status}"
    reset_fault()


def test_success_and_visibility():
    reset_fault()
    # Direct call to app internals to avoid HTTP/TestClient dependency variance
    rag_app_main._lex_tx_begin()
    fname = f"{FNAME_BASE}_success.docx"
    rag_app_main._purge_source_for_reindex(fname)
    # run incremental indexing with TEST_LEX_ONLY
    import uuid
    task_id = uuid.uuid4().hex
    import asyncio
    # prepare TASKS entry expected by app logic
    rag_app_main.TASKS[task_id] = {
        "status": "indexing",
        "stage": "embedding",
        "filename": fname,
        "created_at": datetime.now().isoformat()
    }
    asyncio.get_event_loop().run_until_complete(
        rag_app_main.index_document_incremental(task_id=task_id, filename=fname, text="# Sheet: A\nabc\n# Sheet: B\nxyz", metadata={"doc_type":"regulation"})
    )
    rag_app_main._lex_tx_commit()
    meta, fts = counts_for_source(fname)
    assert meta == fts and meta > 0, f"FTS integrity failed: meta={meta}, fts={fts}"
    # In direct call path, doc_status may be unset; core assertion is FTS integrity only


def run_all():
    # ensure db path is resolvable
    c = _connect_db()
    assert c is not None, "db connection failed"
    # BEGIN IMMEDIATE is used by writer (validated by fault stages without half-writes)
    for stage in ("before_purge", "after_purge", "after_meta_insert", "after_fts_insert", "before_commit"):
        test_fault(stage)
    test_success_and_visibility()
    print("All atomic SQLite tests passed at", datetime.now().isoformat())


if __name__ == "__main__":
    run_all()
    # Additional tests: old version protection & pre-commit visibility
    # 1) Old version protection: rollback to previous version on failure
    import uuid, asyncio
    fname = f"{FNAME_BASE}_oldver.docx"
    rag_app_main._lex_tx_begin()
    old_task = uuid.uuid4().hex
    rag_app_main.TASKS[old_task] = {"status":"indexing","stage":"embedding","filename":fname,"created_at":datetime.now().isoformat()}
    asyncio.get_event_loop().run_until_complete(
        rag_app_main.index_document_incremental(task_id=old_task, filename=fname, text="# Sheet: A\nabc\n# Sheet: B\nxyz", metadata={"doc_type":"regulation"})
    )
    rag_app_main._lex_tx_commit()
    meta_old, fts_old = counts_for_source(fname)
    assert meta_old == fts_old and meta_old > 0, f"prepare old version failed: meta={meta_old}, fts={fts_old}"
    # Inject failure on rebuild
    os.environ["RAG_FAULT_INJECT_STAGE"] = "after_meta_insert"
    rag_app_main._lex_tx_begin()
    try:
        rag_app_main._purge_source_for_reindex(fname)
        new_task = uuid.uuid4().hex
        rag_app_main.TASKS[new_task] = {"status":"indexing","stage":"embedding","filename":fname,"created_at":datetime.now().isoformat()}
        asyncio.get_event_loop().run_until_complete(
            rag_app_main.index_document_incremental(task_id=new_task, filename=fname, text="# Sheet: A\nnew\n# Sheet: B\ncontent", metadata={"doc_type":"regulation"})
        )
    except Exception:
        pass
    finally:
        rag_app_main._lex_tx_rollback()
        os.environ.pop("RAG_FAULT_INJECT_STAGE", None)
    meta_after, fts_after = counts_for_source(fname)
    assert meta_after == fts_after and meta_after == meta_old, f"old version protection failed: after meta={meta_after}, fts={fts_after}, old={meta_old}"
    print("Old version protection test passed")
    # 2) Pre-commit visibility: second connection cannot see uncommitted writes
    fname2 = f"{FNAME_BASE}_precommit.docx"
    rag_app_main._lex_tx_begin()
    prep_task = uuid.uuid4().hex
    rag_app_main.TASKS[prep_task] = {"status":"indexing","stage":"embedding","filename":fname2,"created_at":datetime.now().isoformat()}
    asyncio.get_event_loop().run_until_complete(
        rag_app_main.index_document_incremental(task_id=prep_task, filename=fname2, text="# Sheet: A\nold\n# Sheet: B\nversion", metadata={"doc_type":"regulation"})
    )
    rag_app_main._lex_tx_commit()
    base_meta, base_fts = counts_for_source(fname2)
    assert base_meta == base_fts and base_meta > 0, "prepare base version failed"
    # Begin new transaction and write new chunks without commit
    rag_app_main._lex_tx_begin()
    rag_app_main._purge_source_for_reindex(fname2)
    tmp_task = uuid.uuid4().hex
    rag_app_main.TASKS[tmp_task] = {"status":"indexing","stage":"embedding","filename":fname2,"created_at":datetime.now().isoformat()}
    try:
        asyncio.get_event_loop().run_until_complete(
            rag_app_main.index_document_incremental(task_id=tmp_task, filename=fname2, text="# Sheet: A\nnew\n# Sheet: B\nuncommitted", metadata={"doc_type":"regulation"})
        )
    except Exception:
        pass
    # Connection B observes counts (should still be base version)
    conn_b = _connect_db()
    obs_meta = conn_b.execute("SELECT COUNT(*) FROM chunks_meta WHERE source=?", (fname2,)).fetchone()[0]
    obs_fts  = conn_b.execute("SELECT COUNT(*) FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks_meta WHERE source=?)", (fname2,)).fetchone()[0]
    assert obs_meta == base_meta and obs_fts == base_fts, f"pre-commit visibility failed: obs ({obs_meta},{obs_fts}) vs base ({base_meta},{base_fts})"
    rag_app_main._lex_tx_rollback()
    print("Pre-commit visibility test passed")
