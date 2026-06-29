import asyncio
import os
from datetime import datetime
from typing import Any, Dict

from fastapi import HTTPException


def _document_context(runtime: Any) -> Any:
    factory = getattr(runtime, "document_service", None)
    if callable(factory):
        return factory()
    return runtime


async def _run_serialized_upload_task(runtime: Any, work: Any) -> None:
    lock_factory = getattr(runtime, "upload_index_async_lock", None)
    lock = lock_factory() if callable(lock_factory) else None
    if lock is None:
        await work()
        return
    async with lock:
        await work()


async def list_documents(runtime: Any) -> Dict[str, Any]:
    runtime = _document_context(runtime)
    conn = runtime.lex_db_connect()
    rows = conn.execute(
        "SELECT source, status, active_version, pending_version, last_error, updated_at, canonical_title, same_title_group FROM documents"
    ).fetchall()
    docs_map: Dict[str, Dict[str, Any]] = {}
    for source, status, active_version, pending_version, last_error, updated_at, canonical_title, same_title_group in rows:
        item = {
            "filename": source,
            "created_at": updated_at,
            "status": status or "not_found",
            "document_status": status or "not_found",
            "task_status": None,
            "task_id": None,
            "chunks_indexed": None,
            "error": last_error,
            "searchable": runtime.doc_searchable_flag(source),
            "doc_type": None,
            "topics": [],
            "canonical_title": canonical_title or runtime.filename_stem(source),
            "canonical_doc_id": same_title_group or runtime.same_title_group(canonical_title or runtime.filename_stem(source)),
        }
        docs_map[source] = item

    for tid, task in runtime.tasks.items():
        fname = task.get("filename")
        if not fname:
            continue
        created_at = task.get("created_at") or ""
        status = runtime.public_task_status(task.get("status"))
        task_item = {
            "filename": fname,
            "created_at": created_at,
            "status": status,
            "document_status": (docs_map.get(fname) or {}).get("document_status") or status,
            "task_status": status,
            "task_id": tid,
            "chunks_indexed": task.get("chunks_indexed") if status == "completed" else None,
            "error": task.get("error") if status == "failed" else None,
            "searchable": runtime.doc_searchable_flag(fname),
            "doc_type": None,
            "topics": [],
            "canonical_title": (runtime.doc_get(fname).get("canonical_title") or runtime.filename_stem(fname)),
            "canonical_doc_id": runtime.canonical_doc_id_for_source(fname),
        }
        existing = docs_map.get(fname)
        if not existing:
            if status not in ("accepted", "indexing"):
                continue
            docs_map[fname] = task_item
            continue
        if (existing.get("status") or "") not in ("completed", "vector_pending"):
            docs_map[fname] = task_item
        else:
            newer = (created_at or "") > (existing.get("created_at") or "")
            if newer:
                existing["created_at"] = created_at
            existing["error"] = None

    milvus_stats = runtime.milvus_source_stats()
    for source, stats in milvus_stats.items():
        existing = docs_map.get(source)
        if existing is None:
            docs_map[source] = {
                "filename": source,
                "created_at": stats.get("created_at"),
                "status": "completed",
                "document_status": "completed",
                "task_status": None,
                "task_id": None,
                "chunks_indexed": stats.get("chunks_indexed"),
                "error": None,
                "searchable": True,
                "doc_type": None,
                "topics": [],
                "canonical_title": (runtime.doc_get(source).get("canonical_title") or runtime.filename_stem(source)),
                "canonical_doc_id": runtime.canonical_doc_id_for_source(source),
            }
            continue
        existing["chunks_indexed"] = stats.get("chunks_indexed")
        if stats.get("created_at") and (stats.get("created_at") > (existing.get("created_at") or "")):
            existing["created_at"] = stats.get("created_at")
        if int(stats.get("chunks_indexed") or 0) > 0 and (existing.get("status") or "") not in ("completed", "vector_pending"):
            existing["status"] = "completed"
            existing["document_status"] = "completed"
            existing["error"] = None
            existing["searchable"] = True

    documents = sorted(docs_map.values(), key=lambda x: x.get("created_at") or "", reverse=True)
    return {"documents": documents}


async def delete_document(filename: str, runtime: Any) -> Any:
    runtime = _document_context(runtime)
    safe_name = runtime.safe_filename(filename)
    delete_task_id = runtime.new_task_id()
    runtime.tasks[delete_task_id] = {
        "op": "delete",
        "status": "indexing",
        "stage": "deleting_milvus",
        "filename": safe_name,
        "created_at": datetime.now().isoformat(),
    }
    runtime.task_log(delete_task_id, "delete_started", {"filename": safe_name})
    runtime.lex_db_set_status(safe_name, "deleting")
    runtime.doc_upsert(safe_name, status="deleting", last_error=None)

    cancelled_tasks = await runtime.cancel_source_async_tasks(safe_name)
    if cancelled_tasks:
        runtime.task_log(delete_task_id, "cancelled_source_tasks", {"count": cancelled_tasks})

    lock = runtime.get_source_lock(safe_name)
    if not lock.acquire(timeout=30):
        runtime.tasks[delete_task_id]["status"] = "failed"
        runtime.tasks[delete_task_id]["stage"] = "deleting_locked"
        runtime.tasks[delete_task_id]["error"] = "document_locked"
        runtime.task_log(delete_task_id, "failed", {"error": runtime.tasks[delete_task_id]["error"]})
        raise HTTPException(status_code=429, detail="文档正在处理中，请稍后重试")

    try:
        try:
            vector_db = runtime.vector_db()
            vector_db.connect()
            runtime.crash_inject("delete_milvus")
            vector_delete_info = runtime.delete_milvus_document_object(vector_db, safe_name)
        except Exception as exc:
            msg = str(exc)
            runtime.tasks[delete_task_id]["status"] = "failed"
            runtime.tasks[delete_task_id]["stage"] = "deleting_milvus"
            runtime.tasks[delete_task_id]["error"] = msg
            runtime.lex_db_set_status(safe_name, "delete_failed")
            runtime.doc_upsert(safe_name, status="delete_failed", last_error=msg)
            runtime.task_log(delete_task_id, "failed", {"stage": "deleting_milvus", "error": msg})
            raise HTTPException(status_code=503, detail=f"delete_milvus_failed: {msg}")

        runtime.tasks[delete_task_id]["status"] = "indexing"
        runtime.tasks[delete_task_id]["stage"] = "deleting_lexical"
        runtime.task_log(delete_task_id, "deleting_lexical", vector_delete_info)
        try:
            runtime.lex_db_delete_source(safe_name)
        except Exception as exc:
            msg = str(exc)
            runtime.tasks[delete_task_id]["status"] = "failed"
            runtime.tasks[delete_task_id]["stage"] = "deleting_lexical"
            runtime.tasks[delete_task_id]["error"] = msg
            runtime.lex_db_set_status(safe_name, "delete_failed")
            runtime.doc_upsert(safe_name, status="delete_failed", last_error=msg)
            runtime.task_log(delete_task_id, "failed", {"stage": "deleting_lexical", "error": msg})
            raise HTTPException(status_code=503, detail=f"delete_sqlite_failed: {msg}")

        file_cleanup = runtime.delete_uploaded_artifacts(safe_name)
        if file_cleanup.get("failed"):
            msg = "artifact_cleanup_failed"
            runtime.tasks[delete_task_id]["status"] = "failed"
            runtime.tasks[delete_task_id]["stage"] = "deleting_artifacts"
            runtime.tasks[delete_task_id]["error"] = msg
            runtime.lex_db_set_status(safe_name, "delete_failed")
            runtime.doc_upsert(safe_name, status="delete_failed", last_error=msg)
            runtime.task_log(delete_task_id, "failed", {"stage": "deleting_artifacts", "failed": file_cleanup.get("failed")})
            raise HTTPException(status_code=503, detail=msg)

        runtime.tasks[delete_task_id]["status"] = "completed"
        runtime.tasks[delete_task_id]["stage"] = "completed"
        runtime.task_log(
            delete_task_id,
            "completed",
            {"action": "delete", "files_removed": len(file_cleanup.get("removed") or [])},
        )

        removed = []
        for tid in list(runtime.tasks.keys()):
            task = runtime.tasks.get(tid) or {}
            if task.get("filename") == safe_name:
                removed.append(tid)
                del runtime.tasks[tid]
        runtime.save_tasks()
        return {
            "filename": safe_name,
            "status": "completed",
            "task_id": delete_task_id,
            "tasks_removed": removed,
            "vector_cleanup": vector_delete_info,
            "file_cleanup": file_cleanup,
        }
    finally:
        try:
            lock.release()
        except Exception:
            pass


async def retry_task(task_id: str, runtime: Any) -> Dict[str, Any]:
    runtime = _document_context(runtime)
    task = runtime.tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    if (task.get("op") or "index") != "index":
        raise HTTPException(status_code=400, detail="仅索引任务支持重试")
    filename = task.get("filename") or "unknown"
    runtime.task_log(task_id, "retry", {"filename": filename})
    runtime.tasks[task_id]["status"] = "accepted"
    runtime.tasks[task_id]["stage"] = "accepted"
    runtime.tasks[task_id]["error"] = None
    runtime.tasks[task_id]["chunks_indexed"] = None
    runtime.save_tasks()

    async def _run() -> None:
        lock = runtime.get_source_lock(filename)
        try:
            if not lock.acquire(timeout=30):
                raise HTTPException(status_code=429, detail="文档正在处理中，请稍后重试")
            runtime.tasks[task_id]["status"] = "indexing"
            runtime.tasks[task_id]["stage"] = "parsing"
            runtime.task_log(task_id, "parsing")
            if task.get("path"):
                path = task.get("path")
                if not path or not os.path.exists(path):
                    raise HTTPException(status_code=404, detail="原始上传文件不存在，无法重试")
                with open(path, "rb") as f:
                    raw = f.read()
                document_ir = runtime.extract_document_ir_from_file(
                    filename,
                    raw,
                    metadata={"file_type": os.path.splitext(filename)[1].lstrip(".").lower(), "file_size": len(raw)},
                    doc_version=runtime.doc_get(filename).get("active_version") or 1,
                )
                text = runtime.document_ir_plain_text(document_ir, normalized=False)
                runtime.tasks[task_id]["status"] = "indexing"
                runtime.tasks[task_id]["stage"] = "embedding"
                runtime.task_log(task_id, "embedding")
                chunks = await runtime.index_document(
                    filename=filename,
                    text=text,
                    metadata={"file_type": os.path.splitext(filename)[1].lstrip(".").lower(), "file_size": len(raw)},
                    document_ir=document_ir,
                )
            else:
                payload = task.get("payload") or {}
                text = payload.get("text") or ""
                metadata = payload.get("metadata")
                document_ir = runtime.build_document_ir_from_text(
                    filename,
                    text,
                    metadata=metadata,
                    parser_name="direct_text",
                    doc_version=runtime.doc_get(filename).get("active_version") or 1,
                )
                runtime.tasks[task_id]["status"] = "indexing"
                runtime.tasks[task_id]["stage"] = "embedding"
                runtime.task_log(task_id, "embedding")
                chunks = await runtime.index_document(
                    filename=filename,
                    text=text,
                    metadata=metadata,
                    document_ir=document_ir,
                )
            runtime.tasks[task_id]["status"] = "completed"
            runtime.tasks[task_id]["stage"] = "completed"
            runtime.tasks[task_id]["chunks_indexed"] = chunks
            runtime.task_log(task_id, "completed", {"chunks_indexed": chunks})
        except asyncio.CancelledError:
            runtime.tasks[task_id]["status"] = "failed"
            runtime.tasks[task_id]["stage"] = "cancelled"
            runtime.tasks[task_id]["error"] = "cancelled_for_delete"
            runtime.doc_upsert(filename, status="delete_failed", last_error="cancelled_for_delete")
            runtime.task_log(task_id, "cancelled", {"reason": "delete_requested"})
            raise
        except Exception as exc:
            runtime.tasks[task_id]["status"] = "failed"
            runtime.tasks[task_id]["stage"] = "failed"
            runtime.tasks[task_id]["error"] = str(exc)
            runtime.task_log(task_id, "failed", {"error": str(exc)})
        finally:
            try:
                lock.release()
            except Exception:
                pass

    runtime.register_source_async_task(filename, asyncio.create_task(_run_serialized_upload_task(runtime, _run)))
    return {"task_id": task_id, "status": "accepted", "filename": filename}


async def get_document_detail(filename: str, runtime: Any) -> Any:
    runtime = _document_context(runtime)
    safe_name = runtime.safe_filename(filename)

    latest_task_id = None
    latest_task = None
    latest_created_at = ""
    for tid, task in runtime.tasks.items():
        if (task.get("filename") or "") != safe_name:
            continue
        created_at = task.get("created_at") or ""
        if created_at >= latest_created_at:
            latest_created_at = created_at
            latest_task_id = tid
            latest_task = task

    try:
        vector_db = None
        try:
            vector_db = runtime.vector_db()
            vector_db.connect()
        except Exception:
            doc = runtime.doc_get(safe_name)
            doc_status = runtime.lex_db_get_status(safe_name) or (doc.get("status") or "not_found")
            code = 200
            if doc_status in {"accepted", "indexing", "reindexing", "vector_pending"}:
                code = 202
            elif doc_status in {"failed", "vector_failed", "delete_failed"}:
                code = 409
            elif (doc.get("status") is None) and (doc_status == "not_found"):
                raise HTTPException(status_code=404, detail="文档不存在")
            return JSONResponse(
                status_code=code,
                content={
                    "filename": safe_name,
                    "status": doc_status,
                    "document_status": doc_status,
                    "task_status": runtime.public_task_status((latest_task or {}).get("status")),
                    "searchable": runtime.doc_searchable_flag(safe_name),
                    "task_id": latest_task_id,
                    "stage": (latest_task or {}).get("stage") or "",
                    "active_version": doc.get("active_version"),
                    "pending_version": doc.get("pending_version"),
                    "last_error": doc.get("last_error"),
                    "chunks": [],
                    "chunk_count": 0,
                },
            )

        response = runtime.milvus_query_source_chunks(vector_db, safe_name, limit=5000)

        active_version = runtime.get_active_version(safe_name)
        if active_version is not None:
            response = [
                row
                for row in (response or [])
                if ((row.get("metadata") or {}).get("doc_version") == active_version)
            ]

        if not response:
            doc_status = runtime.lex_db_get_status(safe_name) or "not_found"
            task_status = runtime.public_task_status((latest_task or {}).get("status"))
            task_stage = (latest_task or {}).get("stage") or ""
            processing_statuses = {"accepted", "indexing", "reindexing", "vector_pending"}

            if task_status in processing_statuses:
                doc = runtime.doc_get(safe_name)
                return JSONResponse(
                    status_code=202,
                    content={
                        "filename": safe_name,
                        "status": doc_status,
                        "task_id": latest_task_id,
                        "stage": task_stage or task_status,
                        "active_version": runtime.get_active_version(safe_name),
                        "pending_version": doc.get("pending_version"),
                        "last_error": doc.get("last_error"),
                        "chunks": [],
                        "chunk_count": 0,
                    },
                )

            if task_status == "failed":
                doc = runtime.doc_get(safe_name)
                return JSONResponse(
                    status_code=409,
                    content={
                        "filename": safe_name,
                        "status": doc_status or "vector_failed",
                        "task_id": latest_task_id,
                        "error": (latest_task or {}).get("error") or "索引失败",
                        "active_version": runtime.get_active_version(safe_name),
                        "pending_version": doc.get("pending_version"),
                        "last_error": doc.get("last_error"),
                        "chunks": [],
                        "chunk_count": 0,
                    },
                )

            raise HTTPException(status_code=404, detail="文档不存在")

        def _chunk_id(item: Dict[str, Any]) -> int:
            metadata = item.get("metadata") or {}
            try:
                return int(metadata.get("chunk_id", 0))
            except Exception:
                return 0

        chunks = sorted(response, key=_chunk_id)
        document_ir = runtime.ensure_document_ir(safe_name, active_version)
        if document_ir:
            full_text = runtime.document_detail_plain_text(document_ir, chunks)
        else:
            full_text = "\n\n".join(
                [
                    runtime.chunk_plain_display_text(
                        (chunk.get("metadata") or {}).get("raw_text") or chunk.get("text") or ""
                    )
                    for chunk in chunks
                ]
            ).strip()

        return {
            "filename": safe_name,
            "created_at": chunks[0].get("created_at"),
            "chunk_count": len(chunks),
            "status": runtime.lex_db_get_status(safe_name) or "completed",
            "content": full_text,
            "ir_available": bool(document_ir),
            "document_metadata": (document_ir or {}).get("metadata") or {},
            "elements": [
                {
                    "element_id": element.get("element_id"),
                    "page_no": element.get("page_no"),
                    "section_path": element.get("section_path") or [],
                    "element_type": element.get("element_type"),
                    "reading_order": element.get("reading_order"),
                    "text_raw": element.get("text_raw") or "",
                    "text_normalized": element.get("text_normalized") or "",
                    "ocr_used": bool(element.get("ocr_used")),
                    "ocr_confidence": element.get("ocr_confidence"),
                    "parser_name": element.get("parser_name"),
                    "parser_version": element.get("parser_version"),
                }
                for element in (document_ir or {}).get("elements") or []
            ],
            "chunks": [
                {
                    "chunk_id": _chunk_id(chunk),
                    "text": runtime.chunk_plain_display_text(
                        (chunk.get("metadata") or {}).get("raw_text") or chunk.get("text") or ""
                    ),
                    "metadata": chunk.get("metadata", {}),
                }
                for chunk in chunks
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        runtime.logger.exception(f"get_document_detail_error: filename={safe_name} err={exc}")
        doc = runtime.doc_get(safe_name)
        doc_status = runtime.lex_db_get_status(safe_name) or (doc.get("status") or "not_found")
        if (doc.get("status") is None) and (doc_status == "not_found"):
            raise HTTPException(status_code=404, detail="文档不存在")
        return JSONResponse(
            status_code=200,
            content={
                "filename": safe_name,
                "status": doc_status,
                "task_id": latest_task_id,
                "stage": (latest_task or {}).get("stage") or "",
                "active_version": doc.get("active_version"),
                "pending_version": doc.get("pending_version"),
                "last_error": doc.get("last_error"),
                "chunks": [],
                "chunk_count": 0,
            },
        )
