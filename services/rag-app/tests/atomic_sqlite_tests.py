import os
import sqlite3
from datetime import datetime

import pytest

# Keep this test fully local. Config reads these values at import time.
os.environ.setdefault("APP_ENV", "test_local")
os.environ.setdefault("TEST_LEX_ONLY", "true")

from app.runtime import LEXICAL_DB_FILE, runtime_context


def _docs():
    return runtime_context().document_service()


def _connect_db():
    return sqlite3.connect(LEXICAL_DB_FILE)


def url_encode(name: str) -> str:
    import urllib.parse
    return urllib.parse.quote(name)


FNAME_BASE = "测试条例_示例"
FAULT_STAGES = ("before_purge", "after_purge", "after_meta_insert", "after_fts_insert", "before_commit")


def reset_fault():
    for k in ("RAG_FAULT_INJECT_STAGE", "LEX_DB_CRASH_INJECT_STAGE"):
        if k in os.environ:
            del os.environ[k]


def counts_for_source(src: str):
    conn = _connect_db()
    meta = conn.execute("SELECT COUNT(*) FROM chunks_meta WHERE source=?", (src,)).fetchone()[0]
    fts = conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks_meta WHERE source=?)", (src,)).fetchone()[0]
    return meta, fts


def clean_source(src: str):
    docs = _docs()
    try:
        docs._lex_store.delete_source(src)
    except Exception:
        pass


async def run_incremental(docs, fname: str, text: str = "# Sheet: A\nabc\n# Sheet: B\nxyz") -> int:
    import uuid

    task_id = uuid.uuid4().hex
    runtime_context().tasks[task_id] = {
        "status": "indexing",
        "stage": "embedding",
        "filename": fname,
        "created_at": datetime.now().isoformat(),
    }
    return await docs.index_document_incremental(
        task_id=task_id,
        filename=fname,
        text=text,
        metadata={"doc_type": "regulation"},
    )


@pytest.mark.parametrize("stage", FAULT_STAGES)
def test_fault(stage: str):
    reset_fault()
    fname = f"{FNAME_BASE}_{stage}.docx"
    clean_source(fname)
    os.environ["RAG_FAULT_INJECT_STAGE"] = stage
    docs = _docs()
    import asyncio

    docs.lex_tx_begin()
    try:
        docs.purge_source_for_reindex(fname, None)
        asyncio.get_event_loop().run_until_complete(run_incremental(docs, fname, "# Sheet: A\nabc"))
        docs.crash_inject("before_commit")
        docs.lex_tx_commit()
    except RuntimeError as exc:
        assert f"Crash injection at stage: {stage}" in str(exc)
        docs.lex_tx_rollback()
    finally:
        reset_fault()
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
    fname = f"{FNAME_BASE}_success.docx"
    clean_source(fname)
    docs = _docs()
    # run incremental indexing with TEST_LEX_ONLY
    import asyncio

    asyncio.get_event_loop().run_until_complete(run_incremental(docs, fname))
    meta, fts = counts_for_source(fname)
    assert meta == fts and meta > 0, f"FTS integrity failed: meta={meta}, fts={fts}"
    # In direct call path, doc_status may be unset; core assertion is FTS integrity only


def run_all():
    # ensure db path is resolvable
    c = _connect_db()
    assert c is not None, "db connection failed"
    # BEGIN IMMEDIATE is used by writer (validated by fault stages without half-writes)
    for stage in FAULT_STAGES:
        test_fault(stage)
    test_success_and_visibility()
    print("All atomic SQLite tests passed at", datetime.now().isoformat())


if __name__ == "__main__":
    run_all()
    # Additional tests: old version protection & pre-commit visibility
    # 1) Old version protection: rollback to previous version on failure
    import asyncio
    fname = f"{FNAME_BASE}_oldver.docx"
    docs = _docs()
    clean_source(fname)
    docs.lex_tx_begin()
    asyncio.get_event_loop().run_until_complete(run_incremental(docs, fname))
    docs.lex_tx_commit()
    meta_old, fts_old = counts_for_source(fname)
    assert meta_old == fts_old and meta_old > 0, f"prepare old version failed: meta={meta_old}, fts={fts_old}"
    # Inject failure on rebuild
    os.environ["RAG_FAULT_INJECT_STAGE"] = "after_meta_insert"
    docs.lex_tx_begin()
    try:
        docs.purge_source_for_reindex(fname, None)
        asyncio.get_event_loop().run_until_complete(run_incremental(docs, fname, "# Sheet: A\nnew\n# Sheet: B\ncontent"))
    except Exception:
        pass
    finally:
        docs.lex_tx_rollback()
        os.environ.pop("RAG_FAULT_INJECT_STAGE", None)
    meta_after, fts_after = counts_for_source(fname)
    assert meta_after == fts_after and meta_after == meta_old, f"old version protection failed: after meta={meta_after}, fts={fts_after}, old={meta_old}"
    print("Old version protection test passed")
    # 2) Pre-commit visibility: second connection cannot see uncommitted writes
    fname2 = f"{FNAME_BASE}_precommit.docx"
    clean_source(fname2)
    docs.lex_tx_begin()
    asyncio.get_event_loop().run_until_complete(run_incremental(docs, fname2, "# Sheet: A\nold\n# Sheet: B\nversion"))
    docs.lex_tx_commit()
    base_meta, base_fts = counts_for_source(fname2)
    assert base_meta == base_fts and base_meta > 0, "prepare base version failed"
    # Begin new transaction and write new chunks without commit
    docs.lex_tx_begin()
    docs.purge_source_for_reindex(fname2, None)
    try:
        asyncio.get_event_loop().run_until_complete(run_incremental(docs, fname2, "# Sheet: A\nnew\n# Sheet: B\nuncommitted"))
    except Exception:
        pass
    # Connection B observes counts (should still be base version)
    conn_b = _connect_db()
    obs_meta = conn_b.execute("SELECT COUNT(*) FROM chunks_meta WHERE source=?", (fname2,)).fetchone()[0]
    obs_fts  = conn_b.execute("SELECT COUNT(*) FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks_meta WHERE source=?)", (fname2,)).fetchone()[0]
    assert obs_meta == base_meta and obs_fts == base_fts, f"pre-commit visibility failed: obs ({obs_meta},{obs_fts}) vs base ({base_meta},{base_fts})"
    docs.lex_tx_rollback()
    print("Pre-commit visibility test passed")
