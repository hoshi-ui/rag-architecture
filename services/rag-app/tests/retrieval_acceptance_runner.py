import argparse
import asyncio
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MAIN_PATH = os.path.join(BASE_DIR, "main.py")
DEFAULT_MANIFEST = os.path.join(BASE_DIR, "uploads", "training", "reranker_round1", "training_manifest.json")
DEFAULT_REPORT = os.path.join(BASE_DIR, "uploads", "training", "reranker_round1", "acceptance_report.json")
JSON_BEGIN = "===LEGAL_RAG_ACCEPTANCE_JSON_BEGIN==="
JSON_END = "===LEGAL_RAG_ACCEPTANCE_JSON_END==="


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval acceptance using source_hit/section_hit/body_hit metrics.")
    parser.add_argument("--inside-container", action="store_true")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


ARGS = _parse_args()


def _inside_container() -> bool:
    return os.path.exists("/.dockerenv")


if not _inside_container():
    os.environ.setdefault("APP_ENV", "test_local")
    os.environ.setdefault("EMBEDDING_SERVICE_URL", "http://127.0.0.1:8001")
    os.environ.setdefault("RERANK_SERVICE_URL", "http://127.0.0.1:8002")
    os.environ.setdefault("MILVUS_HOST", "127.0.0.1")
    os.environ.setdefault("MILVUS_PORT", "19530")


def _load_main_module():
    spec = importlib.util.spec_from_file_location("rag_acceptance_main", MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _extract_embedded_json(output: str) -> Dict[str, Any]:
    start = output.find(JSON_BEGIN)
    end = output.find(JSON_END)
    if start < 0 or end <= start:
        raise RuntimeError("acceptance runner did not emit embedded JSON")
    return json.loads(output[start + len(JSON_BEGIN):end].strip())


def _container_manifest_path(host_path: str) -> str:
    return os.path.join("/tmp/legal_rag_acceptance", os.path.basename(os.path.abspath(host_path)))


def _run_via_container(manifest_path: str, report_path: str) -> None:
    container_manifest = _container_manifest_path(manifest_path)
    subprocess.run(["docker", "cp", __file__, "rag-app:/app/tests/retrieval_acceptance_runner.py"], check=True)
    subprocess.run(["docker", "exec", "rag-app", "mkdir", "-p", os.path.dirname(container_manifest)], check=True)
    subprocess.run(["docker", "cp", manifest_path, f"rag-app:{container_manifest}"], check=True)
    container_report = "/app/uploads/training/reranker_round1/acceptance_report.json"
    cmd = [
        "docker", "exec", "rag-app", "python", "/app/tests/retrieval_acceptance_runner.py",
        "--inside-container", "--manifest", container_manifest, "--report", container_report, "--top-k", str(ARGS.top_k),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr or result.stdout)
        raise SystemExit(result.returncode)
    summary = _extract_embedded_json(result.stdout)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    subprocess.run(["docker", "cp", f"rag-app:{container_report}", report_path], check=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _load_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload.get("cases") or [])


def _rate(items: List[Dict[str, Any]], key: str) -> float:
    if not items:
        return 0.0
    return sum(1 for item in items if item.get(key)) / float(len(items))


async def _pre_gate_retrieve_docs(m: Any, handler: Any, query: str, top_k: int) -> List[Dict[str, Any]]:
    normalized_query = m._normalize_query(query)
    fnames = m._extract_filename_candidates(normalized_query)
    recall = await handler._run_lightweight_recall(
        normalized_query,
        top_k=top_k,
        enable_rerank=True,
        filename_hints=fnames,
    )
    return list(recall.get("retrieve_docs") or recall.get("post_filter_docs") or recall.get("selected_docs") or [])


async def _run_inside_container(manifest_path: str, report_path: str) -> Dict[str, Any]:
    m = _load_main_module()
    handler = m.QueryHandler()
    cases = _load_manifest(manifest_path)
    results: List[Dict[str, Any]] = []
    by_origin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for case in cases:
        query = str(case.get("query") or "").strip()
        expected_sources = [str(item).strip() for item in (case.get("expected_sources") or []) if str(item).strip()]
        positive_map = {
            (str(item.get("source") or ""), int(item.get("chunk_id") or 0)): item
            for item in (case.get("positive_chunks") or [])
            if str(item.get("source") or "") and int(item.get("chunk_id") or 0)
        }
        positive_sections = {str(item.get("section_title") or "").strip() for item in (case.get("positive_chunks") or []) if str(item.get("section_title") or "").strip()}
        result = await handler.retrieve(query=query, user_id="acceptance_runner", top_k=ARGS.top_k, enable_rerank=True)
        docs = await _pre_gate_retrieve_docs(m, handler, query, ARGS.top_k)
        returned_sources = [str(m._hit_entity_source(doc) or doc.get("source") or "") for doc in docs]
        metadata = result.get("metadata") or {}
        source_hit = bool(expected_sources) and any(src in expected_sources for src in returned_sources)
        wrong_source = bool(expected_sources) and bool(docs) and not source_hit
        section_hit = False
        body_hit = False
        for doc in docs:
            source = str(m._hit_entity_source(doc) or doc.get("source") or "")
            md = m._hit_metadata(doc) if hasattr(m, "_hit_metadata") else (doc.get("metadata") or {})
            section_title = str(md.get("section_title") or md.get("section") or "").strip()
            chunk_id = int(md.get("chunk_id") or 0)
            role = str(md.get("chunk_role") or "body").strip()
            if source in expected_sources and section_title and section_title in positive_sections:
                section_hit = True
            if source in expected_sources and chunk_id and (source, chunk_id) in positive_map and role not in {"title", "toc", "chapter_heading", "section_heading", "appendix_heading", "document_summary"}:
                body_hit = True
        negative_clean = (not expected_sources) and (not docs or bool(metadata.get("refused") or metadata.get("blocked")))
        row = {
            "case_id": case.get("case_id") or "",
            "origin": case.get("origin") or "",
            "category": case.get("category") or "",
            "query": query,
            "expected_sources": expected_sources,
            "returned_sources": returned_sources,
            "source_hit": source_hit,
            "section_hit": section_hit,
            "body_hit": body_hit,
            "wrong_source": wrong_source,
            "negative_clean": negative_clean,
            "refused": bool(metadata.get("refused") or metadata.get("blocked")),
            "metadata": metadata,
        }
        results.append(row)
        by_origin[row["origin"]].append(row)

    positives = [item for item in results if item.get("expected_sources")]
    negatives = [item for item in results if not item.get("expected_sources")]
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "manifest": manifest_path,
        "cases": len(results),
        "source_hit": _rate(positives, "source_hit"),
        "section_hit": _rate(positives, "section_hit"),
        "body_hit": _rate(positives, "body_hit"),
        "wrong_source": _rate(positives, "wrong_source"),
        "negative_clean_rate": _rate(negatives, "negative_clean"),
        "embedding_tune_candidates": [item["case_id"] for item in positives if not item.get("body_hit")],
        "by_origin": {
            origin: {
                "cases": len(items),
                "source_hit": _rate([item for item in items if item.get("expected_sources")], "source_hit"),
                "section_hit": _rate([item for item in items if item.get("expected_sources")], "section_hit"),
                "body_hit": _rate([item for item in items if item.get("expected_sources")], "body_hit"),
                "wrong_source": _rate([item for item in items if item.get("expected_sources")], "wrong_source"),
                "negative_clean_rate": _rate([item for item in items if not item.get("expected_sources")], "negative_clean"),
            }
            for origin, items in by_origin.items()
        },
    }
    report = {
        "summary": summary,
        "results": results,
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    manifest_path = os.path.abspath(ARGS.manifest)
    report_path = os.path.abspath(ARGS.report)
    if not ARGS.inside_container and not _inside_container():
        _run_via_container(manifest_path, report_path)
        return
    summary = asyncio.run(_run_inside_container(manifest_path, report_path))
    print(JSON_BEGIN)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(JSON_END)


if __name__ == "__main__":
    main()