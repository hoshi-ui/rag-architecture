from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.documents import ir_store as document_ir_store
from app.config import Config
from app.runtime.container import AppRuntimeContext
from app.runtime.startup import run_startup
from app.utils.files import safe_filename


def _json_loads(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return fallback


def _all_document_sources(runtime: AppRuntimeContext) -> List[str]:
    rows = runtime._lex_store.read_connect().execute(
        "SELECT source FROM documents WHERE source IS NOT NULL AND TRIM(source) != '' ORDER BY source ASC"
    ).fetchall()
    return [safe_filename(row[0]) for row in rows if row and row[0]]


def _version_for_rebuild(doc: Dict[str, Any]) -> Optional[int]:
    for key in ("active_version", "pending_version"):
        value = doc.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def _probe_from_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mime_type": doc.get("mime_type"),
        "detected_ext": doc.get("detected_ext"),
        "file_size": doc.get("file_size"),
        "page_count": doc.get("page_count"),
        "route": doc.get("parser_route"),
        "parser_backend": doc.get("parser_backend"),
    }


async def rebuild_source(
    runtime: AppRuntimeContext,
    source: str,
    *,
    publish: bool = True,
    transactional: bool = True,
) -> Dict[str, Any]:
    started = time.perf_counter()
    service = runtime.document_service()
    safe = safe_filename(source)
    doc = service.doc_get(safe)
    if getattr(runtime, "skip_existing_pending", False) and doc.get("pending_version") is not None:
        return {
            "source": safe,
            "status": "skipped",
            "reason": "existing pending_version",
            "pending_version": doc.get("pending_version"),
        }
    old_version = _version_for_rebuild(doc)
    if old_version is None:
        return {"source": safe, "status": "skipped", "reason": "missing active_or_pending_version"}

    document_ir = service.ensure_document_ir(safe, old_version)
    if not document_ir or not (document_ir.get("elements") or []):
        return {"source": safe, "status": "skipped", "reason": f"missing document_ir for version {old_version}"}
    conn = runtime._lex_store.connect()
    if getattr(conn, "in_transaction", False):
        runtime._lex_store.commit()

    text = service.document_ir_plain_text(document_ir, normalized=False)
    if not (text or "").strip():
        return {"source": safe, "status": "skipped", "reason": "empty document text"}

    metadata = dict(document_ir.get("metadata") or {})
    content_sha256 = doc.get("content_sha256") or service.content_sha256_text(text)
    source_id = doc.get("source_id") or service.build_source_id(safe, content_sha256)
    quality = {
        "status": doc.get("parse_status") or "accepted",
        "score": float(doc.get("parse_quality_score") or 1.0),
        "flags": _json_loads(doc.get("quality_flags"), []),
    }
    probe = _probe_from_doc(doc)
    version_next = service.doc_next_version(safe)
    rebuilt_ir = dict(document_ir)
    rebuilt_ir["doc_version"] = version_next

    lock = service.get_source_lock(safe)
    if not lock.acquire(timeout=30):
        return {"source": safe, "status": "skipped", "reason": "source locked"}

    try:
        if transactional:
            service.lex_tx_begin()
        service.lex_db_set_status(safe, "reindexing")
        service.doc_upsert(
            safe,
            status="reindexing",
            pending_version=version_next,
            parse_status="parsing",
            searchable=service.doc_searchable_flag(safe),
        )
        profile = service.build_document_profile(
            safe,
            doc.get("original_filename") or safe,
            source_id,
            content_sha256,
            text,
            rebuilt_ir,
            probe,
            quality,
            metadata=metadata,
        )
        service.doc_upsert(
            safe,
            status="reindexing",
            pending_version=version_next,
            parse_status=quality["status"],
            parse_quality_score=quality["score"],
            quality_flags=service.json_dumps(quality["flags"]),
            canonical_title=profile.get("canonical_title") or doc.get("canonical_title"),
            title_tokens=" ".join(profile.get("title_aliases") or []),
            aliases=",".join((profile.get("title_aliases") or [])[1:]),
            filename_stem=service.filename_stem(safe),
            doc_type=profile.get("doc_type"),
            topic=",".join((profile.get("topic_terms") or [])[:8]),
            source_id=source_id,
            original_filename=doc.get("original_filename") or safe,
            content_sha256=content_sha256,
            mime_type=doc.get("mime_type"),
            detected_ext=doc.get("detected_ext"),
            file_size=doc.get("file_size"),
            page_count=doc.get("page_count"),
            parser_route=doc.get("parser_route"),
            parser_backend=doc.get("parser_backend"),
            searchable=service.doc_searchable_flag(safe),
        )
        service.purge_source_for_reindex(safe, version_next)
        task_id = f"rebuild_all::{safe}"
        runtime.tasks[task_id] = {"status": "indexing", "stage": "embedding", "filename": safe}
        chunks = await service.index_document_incremental(
            task_id=task_id,
            filename=safe,
            text=text,
            metadata=metadata,
            document_ir=rebuilt_ir,
        )
        service.persist_document_profile(safe, version_next, profile)
        service.lex_db_set_status(safe, "vector_pending")
        service.doc_upsert(
            safe,
            status="vector_pending",
            pending_version=version_next,
            last_error=None,
            parse_status=quality["status"],
            searchable=service.doc_searchable_flag(safe),
        )
        if transactional:
            service.lex_tx_commit()
        service.lex_db_checkpoint("PASSIVE")
    except Exception as exc:
        if transactional:
            service.lex_tx_rollback()
        service.doc_upsert(safe, status="vector_failed", last_error=str(exc), searchable=service.doc_searchable_flag(safe))
        raise
    finally:
        try:
            lock.release()
        except Exception:
            pass

    finalized = False
    if publish:
        finalized = service.finalize_pending_version_if_ready(safe)
    return {
        "source": safe,
        "status": "completed" if finalized else "vector_pending",
        "old_version": old_version,
        "new_version": version_next,
        "chunks": chunks,
        "published": finalized,
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }


async def _worker(
    *,
    name: str,
    queue: "asyncio.Queue[str]",
    results: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    runtime = AppRuntimeContext()
    runtime.document_service()._sqlite_write_lock = args.sqlite_write_lock
    runtime.skip_existing_pending = bool(args.skip_existing_pending)
    await run_startup(runtime)
    while True:
        try:
            source = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        try:
            result = await rebuild_source(
                runtime,
                source,
                publish=not args.no_publish,
                transactional=bool(args.workers <= 1),
            )
            result["worker"] = name
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as exc:
            result = {"source": source, "status": "failed", "error": str(exc), "worker": name}
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if not args.continue_on_error:
                while True:
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                return
        finally:
            queue.task_done()


async def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild all document indexes from stored Document IR.")
    parser.add_argument("--source", action="append", help="Rebuild one source. Can be provided multiple times.")
    parser.add_argument("--no-publish", action="store_true", help="Leave rebuilt versions in vector_pending.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue rebuilding other documents after a failure.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast rebuild mode: disable per-chunk LLM metadata enrichment.",
    )
    parser.add_argument(
        "--no-llm-metadata",
        action="store_true",
        help="Disable per-chunk LLM metadata enrichment for this rebuild.",
    )
    parser.add_argument(
        "--skip-existing-pending",
        action="store_true",
        help="Skip documents that already have a pending index version.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of document rebuild workers. Use 2-4 for full rebuilds with LLM metadata enabled.",
    )
    args = parser.parse_args()
    args.workers = max(1, int(args.workers or 1))
    args.sqlite_write_lock = threading.RLock()

    llm_metadata_enabled = bool(getattr(Config, "ENABLE_LLM_CHUNK_METADATA_ENRICHMENT", True))
    if args.fast or args.no_llm_metadata:
        Config.ENABLE_LLM_CHUNK_METADATA_ENRICHMENT = False
        llm_metadata_enabled = False

    runtime = AppRuntimeContext()
    runtime.document_service()._sqlite_write_lock = args.sqlite_write_lock
    runtime.skip_existing_pending = bool(args.skip_existing_pending)
    await run_startup(runtime)
    sources = [safe_filename(source) for source in args.source] if args.source else _all_document_sources(runtime)
    if not sources:
        print(json.dumps({"status": "skipped", "reason": "no documents found"}, ensure_ascii=False))
        return 0

    results: List[Dict[str, Any]] = []
    if args.workers <= 1:
        for source in sources:
            try:
                result = await rebuild_source(runtime, source, publish=not args.no_publish, transactional=True)
                result["worker"] = "main"
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
            except Exception as exc:
                result = {"source": source, "status": "failed", "error": str(exc), "worker": "main"}
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                if not args.continue_on_error:
                    break
    else:
        queue: "asyncio.Queue[str]" = asyncio.Queue()
        for source in sources:
            queue.put_nowait(source)
        workers = [
            asyncio.create_task(_worker(name=f"w{i + 1}", queue=queue, results=results, args=args))
            for i in range(min(args.workers, len(sources)))
        ]
        await asyncio.gather(*workers)

    failed = [item for item in results if item.get("status") == "failed"]
    skipped = [item for item in results if item.get("status") == "skipped"]
    completed = [item for item in results if item.get("status") in {"completed", "vector_pending"}]
    print(
        json.dumps(
            {
                "summary": {
                    "total": len(results),
                    "completed_or_pending": len(completed),
                    "skipped": len(skipped),
                    "failed": len(failed),
                    "llm_chunk_metadata_enrichment": llm_metadata_enabled,
                    "skip_existing_pending": bool(args.skip_existing_pending),
                    "workers": args.workers,
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
