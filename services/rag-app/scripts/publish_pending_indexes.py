from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.runtime.container import AppRuntimeContext
from app.runtime.startup import run_startup
from app.utils.files import safe_filename


def _pending_document_sources(runtime: AppRuntimeContext) -> List[str]:
    rows = runtime._lex_store.read_connect().execute(
        "SELECT source FROM documents WHERE pending_version IS NOT NULL ORDER BY source ASC"
    ).fetchall()
    return [safe_filename(row[0]) for row in rows if row and row[0]]


def _failed_gate_keys(gate: Dict[str, Any]) -> List[str]:
    return [
        key
        for key, value in gate.items()
        if key.endswith("_ok") and value is False
    ]


async def publish_source(runtime: AppRuntimeContext, source: str) -> Dict[str, Any]:
    service = runtime.document_service()
    safe = safe_filename(source)
    doc = service.doc_get(safe)
    pending_version = doc.get("pending_version")
    if pending_version is None:
        return {
            "source": safe,
            "status": "skipped",
            "reason": "missing pending_version",
            "active_version": doc.get("active_version"),
        }

    try:
        pending_version = int(pending_version)
    except Exception:
        return {
            "source": safe,
            "status": "skipped",
            "reason": "invalid pending_version",
            "pending_version": pending_version,
        }

    gate = service.build_publish_gate(safe, pending_version)
    published = service.finalize_pending_version_if_ready(safe)
    result = {
        "source": safe,
        "status": "completed" if published else "publish_failed",
        "published": published,
        "pending_version": pending_version,
        "publish_gate": gate,
    }
    if not published:
        result["failed_gates"] = _failed_gate_keys(gate)
    else:
        result["active_version"] = pending_version
    return result


async def main() -> int:
    parser = argparse.ArgumentParser(description="Publish pending document index versions if their publish gate is ready.")
    parser.add_argument("--source", action="append", help="Publish one source. Can be provided multiple times.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue publishing other documents after a failure.")
    args = parser.parse_args()

    runtime = AppRuntimeContext()
    await run_startup(runtime)
    sources = [safe_filename(source) for source in args.source] if args.source else _pending_document_sources(runtime)
    if not sources:
        print(json.dumps({"status": "skipped", "reason": "no pending documents found"}, ensure_ascii=False))
        return 0

    results: List[Dict[str, Any]] = []
    for source in sources:
        try:
            result = await publish_source(runtime, source)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as exc:
            result = {"source": source, "status": "failed", "error": str(exc)}
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if not args.continue_on_error:
                break

    print(
        json.dumps(
            {
                "summary": {
                    "total": len(results),
                    "published": len([item for item in results if item.get("published") is True]),
                    "publish_failed": len([item for item in results if item.get("status") == "publish_failed"]),
                    "skipped": len([item for item in results if item.get("status") == "skipped"]),
                    "failed": len([item for item in results if item.get("status") == "failed"]),
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 1 if any(item.get("status") in {"failed", "publish_failed"} for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
