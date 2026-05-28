import argparse
import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MAIN_PATH = os.path.join(BASE_DIR, "main.py")
DEFAULT_MANIFEST = os.path.join(BASE_DIR, "uploads", "training", "reranker_round1", "training_manifest.json")
DEFAULT_REPORT = os.path.join(BASE_DIR, "uploads", "training", "reranker_round1", "variant_guardrail_compare.json")
JSON_BEGIN = "===VARIANT_GUARDRAIL_COMPARE_JSON_BEGIN==="
JSON_END = "===VARIANT_GUARDRAIL_COMPARE_JSON_END==="
SPECIAL_ORIGIN = "legal_rag_special_training_cases_20260513.json"
HARD_ORIGIN = "hard_competitive_cases.json"
BODY_DENY_ROLES = {"title", "toc", "toc_heading", "chapter_heading", "section_heading", "appendix_heading", "document_summary"}
RESCUE_REASONS = {"section_not_hit", "partial_term_coverage", "low_evidence_relevance", "generic_only", "topic_not_hit"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare guardrail variants on special and hard-competitive suites.")
    parser.add_argument("--inside-container", action="store_true")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ft-rerank-url", default=os.getenv("FT_RERANK_URL", "http://rerank-service-ft:8000"))
    parser.add_argument("--strict-min-evidence-score", type=float, default=float(os.getenv("STRICT_MIN_EVIDENCE_SCORE", "0.75")))
    parser.add_argument("--strict-min-substantive-chunks", type=int, default=int(os.getenv("STRICT_MIN_SUBSTANTIVE_CHUNKS", "2")))
    parser.add_argument("--limit", type=int, default=0)
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


def _extract_embedded_json(output: str) -> Dict[str, Any]:
    start = output.find(JSON_BEGIN)
    end = output.find(JSON_END)
    if start < 0 or end <= start:
        raise RuntimeError("variant guardrail compare runner did not emit embedded JSON")
    return json.loads(output[start + len(JSON_BEGIN):end].strip())


def _container_manifest_path(host_path: str) -> str:
    return os.path.join("/tmp/legal_rag_variant_compare", os.path.basename(os.path.abspath(host_path)))


def _container_report_path() -> str:
    return "/app/uploads/training/reranker_round1/variant_guardrail_compare.json"


def _run_via_container(manifest_path: str, report_path: str) -> None:
    container_manifest = _container_manifest_path(manifest_path)
    subprocess.run(["docker", "cp", __file__, "rag-app:/app/tests/variant_guardrail_compare.py"], check=True)
    subprocess.run(["docker", "exec", "rag-app", "mkdir", "-p", os.path.dirname(container_manifest)], check=True)
    subprocess.run(["docker", "cp", manifest_path, f"rag-app:{container_manifest}"], check=True)
    cmd = [
        "docker", "exec", "rag-app", "python", "/app/tests/variant_guardrail_compare.py",
        "--inside-container",
        "--manifest", container_manifest,
        "--report", _container_report_path(),
        "--top-k", str(ARGS.top_k),
        "--ft-rerank-url", ARGS.ft_rerank_url,
        "--strict-min-evidence-score", str(ARGS.strict_min_evidence_score),
        "--strict-min-substantive-chunks", str(ARGS.strict_min_substantive_chunks),
    ]
    if ARGS.limit > 0:
        cmd.extend(["--limit", str(ARGS.limit)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr or result.stdout)
        raise SystemExit(result.returncode)
    payload = _extract_embedded_json(result.stdout)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    subprocess.run(["docker", "cp", f"rag-app:{_container_report_path()}", report_path], check=True)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


def _load_main_module():
    spec = importlib.util.spec_from_file_location("rag_variant_compare_main", MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    cases = list(payload.get("cases") or [])
    filtered = [case for case in cases if str(case.get("origin") or "") in {SPECIAL_ORIGIN, HARD_ORIGIN}]
    if ARGS.limit > 0:
        return filtered[:ARGS.limit]
    return filtered


def _case_suite(case: Dict[str, Any]) -> str:
    origin = str(case.get("origin") or "")
    if origin == SPECIAL_ORIGIN:
        return "special"
    if origin == HARD_ORIGIN:
        return "hard_competitive"
    return "other"


def _expected_answer_scope(case: Dict[str, Any]) -> str:
    value = str(case.get("answer_scope_expected") or "").strip()
    if value:
        return value
    return "refusal" if not (case.get("expected_sources") or []) else "full_candidate"


def _should_answer(case: Dict[str, Any]) -> bool:
    return bool(case.get("expected_sources") or []) and _expected_answer_scope(case) != "refusal"


def _should_refuse(case: Dict[str, Any]) -> bool:
    return not _should_answer(case)


def _positive_body_ids(case: Dict[str, Any]) -> set:
    out = set()
    for item in case.get("positive_chunks") or []:
        source = str(item.get("source") or "").strip()
        chunk_id = int(item.get("chunk_id") or 0)
        role = str(item.get("chunk_role") or "body").strip()
        if source and chunk_id and role not in BODY_DENY_ROLES:
            out.add((source, chunk_id))
    return out


def _body_article_hit_at(m: Any, docs: List[Any], positive_ids: set, k: int) -> bool:
    if not positive_ids:
        return False
    for doc in (docs or [])[:k]:
        md = m._hit_metadata(doc)
        source = str(m._hit_entity_source(doc) or "").strip()
        chunk_id = int(md.get("chunk_id") or 0)
        role = str(md.get("chunk_role") or "body").strip()
        if source and chunk_id and role not in BODY_DENY_ROLES and (source, chunk_id) in positive_ids:
            return True
    return False


def _doc_sources(m: Any, docs: List[Any]) -> List[str]:
    return [str(m._hit_entity_source(doc) or "").strip() for doc in (docs or [])]


@contextmanager
def _temp_config(m: Any, **updates: Any):
    original: Dict[str, Any] = {}
    for key, value in updates.items():
        original[key] = getattr(m.config, key)
        setattr(m.config, key, value)
    try:
        yield
    finally:
        for key, value in original.items():
            setattr(m.config, key, value)


async def _derive_process_docs_and_observations(m: Any, handler: Any, query: str, recall: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
    if recall.get("query_route") == "multi_doc_compare" and recall.get("compare_source_results"):
        compare_process_groups = []
        for item in recall.get("compare_source_results") or []:
            compare_process_groups.append({
                "source": item.get("source") or "",
                "evidence_query": item.get("evidence_query") or "",
                "docs": m._select_process_output_docs(
                    query,
                    item.get("post_filter_docs") or item.get("selected_docs") or [],
                    item.get("score_mode") or recall.get("score_mode") or "score",
                    recall.get("qfilters") or {},
                    max(2, int(recall.get("final_n") or 5)),
                ),
            })
        docs = m._merge_compare_source_doc_groups(compare_process_groups, per_source_limit=max(2, int(recall.get("final_n") or 5)))
        observations = m._compare_evidence_observations(query, compare_process_groups, qfilters=recall.get("qfilters") or {})
        return docs, observations
    docs = m._select_process_output_docs(
        query,
        recall.get("post_filter_docs") or recall.get("selected_docs") or [],
        recall.get("score_mode") or "score",
        recall.get("qfilters") or {},
        int(recall.get("final_n") or 5),
    )
    observations = m._evidence_observations(
        query,
        docs,
        qfilters=recall.get("qfilters") or {},
        candidate_docs=recall.get("post_filter_docs") or recall.get("selected_docs") or [],
        target_sources=recall.get("target_sources") or [],
        source_lock_resolved=bool(recall.get("resolved_source_lock")),
    )
    return docs, observations


def _rescue_query(observations: Dict[str, Any], query: str) -> Dict[str, str]:
    uncovered = [str(item).strip() for item in (observations.get("uncovered_aspects") or []) if str(item).strip()]
    uncovered = list(dict.fromkeys(uncovered))[:4]
    if not uncovered:
        return {"raw_text_query": query, "section_query": "", "doc_prior_query": query}
    expansion = " ".join(uncovered[:3]).strip()
    return {
        "raw_text_query": f"{query} {expansion}".strip(),
        "section_query": expansion,
        "doc_prior_query": query,
    }


async def _maybe_run_coverage_rescue(m: Any, handler: Any, query: str, recall: Dict[str, Any], observations: Dict[str, Any], top_k: int) -> Tuple[List[Any], Dict[str, Any]] | None:
    target_sources = [m._normalize_filename_for_match(src or "") for src in (recall.get("target_sources") or []) if m._normalize_filename_for_match(src or "")]
    if not target_sources:
        return None
    query_embedding = (await handler.embedding_service.embed([query]))[0]
    rescue_subquery = _rescue_query(observations, query)
    groups = []
    recall_k = min(max(top_k * 2, 20), int(getattr(m.config, "TOP_K", 50)))
    for source in target_sources[:3]:
        groups.append(
            await handler._run_target_scoped_recall(
                query=query,
                retrieval_query=query,
                query_embedding=query_embedding,
                qtype=recall.get("question_type") or m._classify_question_type(query),
                qfilters=recall.get("qfilters") or {},
                recall_k=recall_k,
                final_n=int(recall.get("final_n") or 5),
                pool_n=recall_k,
                enable_rerank=True,
                target_source=source,
                compare_subquery=rescue_subquery,
            )
        )
    groups = [group for group in groups if group.get("post_filter_docs") or group.get("selected_docs") or group.get("retrieve_docs")]
    if not groups:
        return None
    if len(groups) == 1:
        rescue_recall = {
            "post_filter_docs": groups[0].get("post_filter_docs") or groups[0].get("selected_docs") or [],
            "selected_docs": groups[0].get("selected_docs") or [],
            "score_mode": groups[0].get("score_mode") or recall.get("score_mode") or "score",
            "qfilters": recall.get("qfilters") or {},
            "final_n": recall.get("final_n") or 5,
            "query_route": recall.get("query_route") or "content_qa",
        }
        return await _derive_process_docs_and_observations(m, handler, query, rescue_recall)
    compare_groups = []
    for item in groups:
        compare_groups.append({
            "source": item.get("source") or "",
            "evidence_query": item.get("evidence_query") or query,
            "docs": m._select_process_output_docs(
                query,
                item.get("post_filter_docs") or item.get("selected_docs") or [],
                item.get("score_mode") or recall.get("score_mode") or "score",
                recall.get("qfilters") or {},
                max(2, int(recall.get("final_n") or 5)),
            ),
        })
    docs = m._merge_compare_source_doc_groups(compare_groups, per_source_limit=max(2, int(recall.get("final_n") or 5)))
    observations = m._compare_evidence_observations(query, compare_groups, qfilters=recall.get("qfilters") or {})
    return docs, observations


async def _run_case_variant(m: Any, case: Dict[str, Any], variant: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    query = m._normalize_query(str(case.get("query") or ""))
    expected_sources = [str(item).strip() for item in (case.get("expected_sources") or []) if str(item).strip()]
    positive_ids = _positive_body_ids(case)
    with _temp_config(
        m,
        RERANK_URL=str(variant["rerank_url"]),
        MIN_EVIDENCE_SCORE=float(variant["min_evidence_score"]),
        MIN_SUBSTANTIVE_CHUNKS=int(variant["min_substantive_chunks"]),
    ):
        handler = m.QueryHandler()
        fnames = m._extract_filename_candidates(query)
        recall = await handler._run_lightweight_recall(query, top_k=top_k, enable_rerank=True, filename_hints=fnames)
        if recall.get("source_lock_required") and not recall.get("resolved_source_lock"):
            reason = str(recall.get("source_lock_reason") or "document_target_required")
            return {
                "case_id": case.get("case_id") or case.get("id") or "",
                "suite": _case_suite(case),
                "query": query,
                "expected_sources": expected_sources,
                "expected_answer_scope": _expected_answer_scope(case),
                "returned_sources": [],
                "answer_scope": "refusal",
                "final_reason": reason,
                "generation_called": 0,
                "generation_called_on_partial": 0,
                "hard_answer": int(_should_refuse(case) and False),
                "wrong_source": 0,
                "mis_refusal": int(_should_answer(case)),
                "negative_clean": int(_should_refuse(case)),
                "body_article_hit_at3": 0,
                "body_article_hit_at5": 0,
                "rescue_applied": 0,
                "rescue_success": 0,
                "metadata": {
                    "query_route": recall.get("query_route") or "",
                    "rerank_used": recall.get("rerank_used"),
                    "source_lock_reason": reason,
                },
            }

        process_docs, observations = await _derive_process_docs_and_observations(m, handler, query, recall)
        refusal_reason = observations.get("evidence_coverage_reason") if observations.get("answer_scope") != "full" else None
        rescue_applied = 0
        rescue_success = 0
        if bool(variant.get("coverage_rescue")) and refusal_reason in RESCUE_REASONS:
            rescued = await _maybe_run_coverage_rescue(m, handler, query, recall, observations, top_k)
            if rescued is not None:
                rescue_applied = 1
                process_docs, observations = rescued
                refusal_reason = observations.get("evidence_coverage_reason") if observations.get("answer_scope") != "full" else None
                rescue_success = int(observations.get("answer_scope") == "full")

    answer_scope = str(observations.get("answer_scope") or ("refusal" if refusal_reason else "full"))
    generation_called = int(answer_scope == "full")
    returned_sources = _doc_sources(m, process_docs)
    wrong_source = int(_should_answer(case) and generation_called and bool(returned_sources) and not any(src in expected_sources for src in returned_sources))
    mis_refusal = int(_should_answer(case) and not generation_called)
    negative_clean = int(_should_refuse(case) and not generation_called)
    hard_answer = int(_should_refuse(case) and generation_called)
    return {
        "case_id": case.get("case_id") or case.get("id") or "",
        "suite": _case_suite(case),
        "query": query,
        "expected_sources": expected_sources,
        "expected_answer_scope": _expected_answer_scope(case),
        "returned_sources": returned_sources,
        "answer_scope": answer_scope,
        "final_reason": str(refusal_reason or observations.get("evidence_coverage_reason") or "sufficient_evidence"),
        "generation_called": generation_called,
        "generation_called_on_partial": int(generation_called and answer_scope != "full"),
        "hard_answer": hard_answer,
        "wrong_source": wrong_source,
        "mis_refusal": mis_refusal,
        "negative_clean": negative_clean,
        "body_article_hit_at3": int(_should_answer(case) and _body_article_hit_at(m, process_docs, positive_ids, 3)),
        "body_article_hit_at5": int(_should_answer(case) and _body_article_hit_at(m, process_docs, positive_ids, 5)),
        "rescue_applied": rescue_applied,
        "rescue_success": rescue_success,
        "metadata": {
            "query_route": recall.get("query_route") or "",
            "rerank_used": recall.get("rerank_used"),
            "docs_final": len(process_docs),
            "observations": observations,
        },
    }


def _suite_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    answerable = [item for item in items if item.get("expected_answer_scope") != "refusal" and item.get("expected_sources")]
    refusal_expected = [item for item in items if item not in answerable]
    denom_answerable = len(answerable)
    denom_refusal = len(refusal_expected)
    return {
        "cases": len(items),
        "answerable_cases": denom_answerable,
        "refusal_expected_cases": denom_refusal,
        "wrong_source_rate": (sum(item["wrong_source"] for item in answerable) / denom_answerable) if denom_answerable else 0.0,
        "mis_refusal_rate": (sum(item["mis_refusal"] for item in answerable) / denom_answerable) if denom_answerable else 0.0,
        "negative_clean_rate": (sum(item["negative_clean"] for item in refusal_expected) / denom_refusal) if denom_refusal else 0.0,
        "partial_term_coverage_count": sum(1 for item in items if item.get("final_reason") == "partial_term_coverage"),
        "section_not_hit_count": sum(1 for item in items if item.get("final_reason") == "section_not_hit"),
        "body_article_hit@3": (sum(item["body_article_hit_at3"] for item in answerable) / denom_answerable) if denom_answerable else 0.0,
        "body_article_hit@5": (sum(item["body_article_hit_at5"] for item in answerable) / denom_answerable) if denom_answerable else 0.0,
        "generation_called_on_partial": sum(item["generation_called_on_partial"] for item in items),
        "hard_answer_count": sum(item["hard_answer"] for item in items),
        "rescue_applied_count": sum(item["rescue_applied"] for item in items),
        "rescue_success_count": sum(item["rescue_success"] for item in items),
    }


async def _run_inside_container(manifest_path: str, report_path: str) -> Dict[str, Any]:
    m = _load_main_module()
    cases = _load_manifest(manifest_path)
    variants = {
        "A_baseline": {
            "rerank_url": "http://rerank-service:8000",
            "coverage_rescue": False,
            "min_evidence_score": float(getattr(m.config, "MIN_EVIDENCE_SCORE", 0.6)),
            "min_substantive_chunks": int(getattr(m.config, "MIN_SUBSTANTIVE_CHUNKS", 1)),
        },
        "B_new_reranker": {
            "rerank_url": ARGS.ft_rerank_url,
            "coverage_rescue": False,
            "min_evidence_score": float(getattr(m.config, "MIN_EVIDENCE_SCORE", 0.6)),
            "min_substantive_chunks": int(getattr(m.config, "MIN_SUBSTANTIVE_CHUNKS", 1)),
        },
        "C_new_reranker_coverage_rescue": {
            "rerank_url": ARGS.ft_rerank_url,
            "coverage_rescue": True,
            "min_evidence_score": float(getattr(m.config, "MIN_EVIDENCE_SCORE", 0.6)),
            "min_substantive_chunks": int(getattr(m.config, "MIN_SUBSTANTIVE_CHUNKS", 1)),
        },
        "D_new_reranker_strict_gate": {
            "rerank_url": ARGS.ft_rerank_url,
            "coverage_rescue": False,
            "min_evidence_score": float(ARGS.strict_min_evidence_score),
            "min_substantive_chunks": int(ARGS.strict_min_substantive_chunks),
        },
    }

    results: Dict[str, List[Dict[str, Any]]] = {name: [] for name in variants}
    for variant_name, variant in variants.items():
        for case in cases:
            results[variant_name].append(await _run_case_variant(m, case, variant, ARGS.top_k))

    summary: Dict[str, Any] = {}
    for variant_name, rows in results.items():
        special = [row for row in rows if row.get("suite") == "special"]
        hard = [row for row in rows if row.get("suite") == "hard_competitive"]
        summary[variant_name] = {
            "config": variants[variant_name],
            "special": _suite_summary(special),
            "hard_competitive": _suite_summary(hard),
            "combined": _suite_summary(rows),
        }

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "manifest": manifest_path,
        "strict_gate": {
            "min_evidence_score": float(ARGS.strict_min_evidence_score),
            "min_substantive_chunks": int(ARGS.strict_min_substantive_chunks),
        },
        "summary": summary,
        "results": results,
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    manifest_path = os.path.abspath(ARGS.manifest)
    report_path = os.path.abspath(ARGS.report)
    if not ARGS.inside_container and not _inside_container():
        _run_via_container(manifest_path, report_path)
        return
    report = asyncio.run(_run_inside_container(manifest_path, report_path))
    print(JSON_BEGIN)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(JSON_END)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()