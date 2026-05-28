import asyncio
import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MAIN_PATH = os.path.join(BASE_DIR, "main.py")
DEFAULT_CASES_PATH = os.path.join(THIS_DIR, "hard_competitive_cases.json")
REPORT_PATH = os.path.abspath(os.path.join(BASE_DIR, "uploads", "live_ablation_report.json"))
JSON_BEGIN = "===LIVE_ABLATION_JSON_BEGIN==="
JSON_END = "===LIVE_ABLATION_JSON_END==="


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live retrieval ablation against a case set.")
    parser.add_argument("--inside-container", action="store_true")
    parser.add_argument("--cases", default=os.getenv("LIVE_ABLATION_CASES_PATH", DEFAULT_CASES_PATH))
    parser.add_argument("--report", default=os.getenv("LIVE_ABLATION_REPORT_PATH", REPORT_PATH))
    return parser.parse_args()


ARGS = _parse_args()
CASES_PATH = os.path.abspath(ARGS.cases)
REPORT_PATH = os.path.abspath(ARGS.report)


def _inside_container() -> bool:
    return os.path.exists("/.dockerenv")

if not _inside_container():
    os.environ.setdefault("APP_ENV", "test_local")
    os.environ.setdefault("EMBEDDING_SERVICE_URL", "http://127.0.0.1:8001")
    os.environ.setdefault("RERANK_SERVICE_URL", "http://127.0.0.1:8002")
    os.environ.setdefault("MILVUS_HOST", "127.0.0.1")
    os.environ.setdefault("MILVUS_PORT", "19530")

spec = importlib.util.spec_from_file_location("rag_live_ablation_main", MAIN_PATH)
m = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(m)


RETRIEVAL_CATEGORIES = {"weak_reference_regulation", "version_switch"}
NEGATIVE_RETRIEVAL_IDS = {"wr_007", "wr_008"}


def _extract_embedded_json(output: str) -> Dict[str, Any]:
    start = output.find(JSON_BEGIN)
    end = output.find(JSON_END)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("live ablation runner did not emit embedded JSON payload")
    payload = output[start + len(JSON_BEGIN):end].strip()
    return json.loads(payload)


def _container_test_path(host_path: str) -> str:
    base = os.path.abspath(THIS_DIR)
    target = os.path.abspath(host_path)
    if target.startswith(base):
        rel = os.path.relpath(target, base)
        return os.path.join("/app/tests", rel)
    return target


def _run_via_container() -> None:
    cmd = [
        "docker",
        "exec",
        "rag-app",
        "python",
        "/app/tests/live_ablation_runner.py",
        "--inside-container",
        "--cases",
        _container_test_path(CASES_PATH),
        "--report",
        "/app/uploads/live_ablation_report.json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr or result.stdout)
        raise SystemExit(result.returncode)
    report = _extract_embedded_json(result.stdout)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def load_cases() -> List[Dict[str, Any]]:
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieval_cases(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if os.path.basename(CASES_PATH) == os.path.basename(DEFAULT_CASES_PATH):
        return list(cases)
    out = []
    for case in cases:
        if case.get("category") in RETRIEVAL_CATEGORIES:
            out.append(case)
            continue
        if case.get("id") in NEGATIVE_RETRIEVAL_IDS:
            out.append(case)
    return out


@contextmanager
def temp_config(**updates: Any):
    original: Dict[str, Any] = {}
    for key, value in updates.items():
        original[key] = getattr(m.config, key)
        setattr(m.config, key, value)
    try:
        yield
    finally:
        for key, value in original.items():
            setattr(m.config, key, value)


def _rank_map(hits: List[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for idx, hit in enumerate(hits):
        src = m._normalize_filename_for_match(m._hit_entity_source(hit) or "")
        if src and src not in out:
            out[src] = idx
    return out


def _first_source_order(hits: List[Any]) -> List[str]:
    out: List[str] = []
    for hit in hits:
        src = m._normalize_filename_for_match(m._hit_entity_source(hit) or "")
        if src and src not in out:
            out.append(src)
    return out


def _dedupe_in_order(hits: List[Any], limit: int) -> List[Any]:
    seen = set()
    out = []
    for hit in hits:
        key = (m._hit_entity_source(hit) or "unknown", (m._hit_entity_text(hit) or "")[:64])
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
        if len(out) >= limit:
            break
    return out


def _scores_for_fusion(
    query: str,
    docs_all: List[Any],
    dense_rank_map: Dict[str, int],
    lex_rank_map: Dict[str, int],
    source_count: Dict[str, int],
    source_signals: Dict[str, Dict[str, Any]],
    fname_set: set,
    allowed_set: set,
    weak_query: bool,
) -> List[Tuple[float, int]]:
    combined: List[Tuple[float, int]] = []
    for idx, hit in enumerate(docs_all):
        src = m._normalize_filename_for_match(m._hit_entity_source(hit) or "")
        score = m._fusion_source_score(src, query, dense_rank_map, lex_rank_map, source_count, source_signals, fname_set, allowed_set, weak_query)
        combined.append((score, idx))
    combined.sort(key=lambda x: x[0], reverse=True)
    return combined


async def _direct_intent_response(handler: Any, query: str, enable_rerank: bool) -> Optional[Dict[str, Any]]:
    if m._is_deleted_visibility_query(query) or m._is_doc_existence_query(query):
        return await handler.retrieve(query, user_id="live_ablation", top_k=10, enable_rerank=enable_rerank)
    return None


async def _run_mode(handler: Any, case: Dict[str, Any], mode: str, enable_rerank: bool) -> Dict[str, Any]:
    query = m._normalize_query(case["query"])
    direct = await _direct_intent_response(handler, query, enable_rerank)
    if direct is not None:
        return direct

    qtype = m._classify_question_type(query)
    fnames = m._extract_filename_candidates(query)
    weak_query = m._is_weak_reference_query(query)
    requested_k = 10
    recall_k = min(max(requested_k * 2, 20), min(m.config.TOP_K, int(getattr(m.config, "RETRIEVAL_CANDIDATE_K", m.config.RECALL_TOP_K))))
    final_n = min(max(m.config.FINAL_CONTEXT_N, 3), 5)
    pool_n = min(max(max(m.config.RERANK_KEEP_N, m.config.CHUNK_RERANK_KEEP_N), requested_k * 2), recall_k)

    doc_recall_plan: List[Dict[str, Any]] = []
    if m._should_use_doc_fallback(query, fnames):
        doc_recall_plan = m._build_doc_recall_plan(
            query,
            limit=max(10, int(getattr(m.config, "DOC_FALLBACK_SOURCE_LIMIT", 6))),
        )
    allowed_docs = [entry.get("source") for entry in doc_recall_plan if entry.get("source")]

    safe_names = [m._normalize_filename_for_match(x) for x in fnames] if fnames else []
    milvus_filter = None
    if len(safe_names) == 1:
        milvus_filter = f"source == {json.dumps(safe_names[0], ensure_ascii=False)}"

    dense_hits: List[Any] = []
    if mode != "lexical_only":
        query_embedding = (await handler.embedding_service.embed([query]))[0]
        dense_hits = handler.vector_db.search(query_embedding, top_k=recall_k, filters=milvus_filter)
        dense_hits = m._filter_hits_by_source_state(dense_hits)["hits"]

    lex_hits: List[Any] = []
    if mode != "dense_only":
        lex_hits = m._collect_lexical_candidates(query, safe_names, doc_recall_plan)
        lex_hits = m._filter_hits_by_source_state(lex_hits)["hits"]

    if mode == "dense_only":
        docs = _dedupe_in_order(dense_hits, recall_k)
    elif mode == "lexical_only":
        docs = _dedupe_in_order(lex_hits, recall_k)
    elif mode == "hybrid_concat":
        docs = _dedupe_in_order(dense_hits + lex_hits, recall_k)
    else:
        docs_all = dense_hits + lex_hits
        dense_rank_map = _rank_map(dense_hits)
        lex_rank_map = _rank_map(lex_hits)
        fname_set = set(safe_names)
        allowed_set = set([m._normalize_filename_for_match(x) for x in allowed_docs]) if allowed_docs else set()
        source_count: Dict[str, int] = {}
        for hit in docs_all:
            src = m._normalize_filename_for_match(m._hit_entity_source(hit) or "")
            if src:
                source_count[src] = source_count.get(src, 0) + 1
        source_signals = m._build_source_signal_map(query, lex_hits, doc_recall_plan)
        combined = _scores_for_fusion(query, docs_all, dense_rank_map, lex_rank_map, source_count, source_signals, fname_set, allowed_set, weak_query)
        docs = []
        seen = set()
        for fused_score, idx in combined:
            hit = docs_all[idx]
            key = (m._hit_entity_source(hit) or "unknown", (m._hit_entity_text(hit) or "")[:64])
            if key in seen:
                continue
            seen.add(key)
            docs.append(m._clone_hit_with_score(hit, fused_score))
            if len(docs) >= recall_k:
                break

    if not docs:
        return {
            "documents": [],
            "sources": [],
            "metadata": {"query": query, "refused": "no_relevant_evidence", "mode": mode, "rerank": enable_rerank}
        }

    pre_rerank_sources = _first_source_order(docs[:pool_n])
    dense_rank_map = _rank_map(dense_hits)
    lex_rank_map = _rank_map(lex_hits)
    source_signals = m._build_source_signal_map(query, lex_hits, doc_recall_plan)
    reranked = await m._chunk_level_rerank(
        handler.rerank_service,
        query,
        docs[:pool_n],
        pool_n,
        m._should_apply_chunk_rerank(docs[:pool_n], dense_rank_map, lex_rank_map, source_signals, enable_rerank),
    )
    docs = reranked["hits"]
    score_mode = reranked["score_mode"]
    thr = m._dynamic_thresholds(qtype, bool(fnames))
    if (not m._passes_relevance_cluster(docs, score_mode, thr, top_n=min(3, len(docs)))) and (not m._doc_cluster_accept(query, docs, fnames or [])):
        return {
            "documents": [],
            "sources": [],
            "metadata": {"query": query, "refused": "low_relevance_filtered", "mode": mode, "rerank": enable_rerank, "pre_rerank_sources": pre_rerank_sources}
        }

    merged_docs = m._merge_and_dedupe_hits(docs, score_mode=score_mode)
    aggregated_docs = m._aggregate_doc_sections(merged_docs, score_mode=score_mode)
    qfilters = m._query_filters(query)
    filtered_docs = m._apply_retrieval_filters(aggregated_docs, qfilters, fnames)
    if not filtered_docs:
        return {
            "documents": [],
            "sources": [],
            "metadata": {"query": query, "refused": "no_relevant_evidence", "mode": mode, "rerank": enable_rerank, "pre_rerank_sources": pre_rerank_sources}
        }

    if mode == "hybrid_fusion":
        fname_set = set(safe_names)
        allowed_set = set([m._normalize_filename_for_match(x) for x in allowed_docs]) if allowed_docs else set()
        source_count = {}
        for hit in dense_hits + lex_hits:
            src = m._normalize_filename_for_match(m._hit_entity_source(hit) or "")
            if src:
                source_count[src] = source_count.get(src, 0) + 1
        src_scores = m._summarize_source_scores(filtered_docs, dense_rank_map, lex_rank_map, source_count, source_signals, fname_set, allowed_set, weak_query, query)
        reranked_sources = await m._source_level_rerank(
            handler.rerank_service,
            query,
            filtered_docs,
            src_scores,
            max(final_n, int(getattr(m.config, "SOURCE_RERANK_KEEP_N", 6))),
            enable_rerank,
            dense_rank_map=dense_rank_map,
            lex_rank_map=lex_rank_map,
            source_signals=source_signals,
        )
        src_scores = reranked_sources["scores"]
        keep_sources = [s for s, _ in sorted(src_scores.items(), key=lambda x: x[1], reverse=True)[:min(len(src_scores), max(final_n, 5))]]
    else:
        keep_sources = _first_source_order(filtered_docs)[:min(len(_first_source_order(filtered_docs)), max(final_n, 5))]

    selected_docs = [d for d in filtered_docs if m._normalize_filename_for_match(m._hit_entity_source(d) or "") in keep_sources]
    selected_docs = m._filter_low_relevance_sources(selected_docs, score_mode=score_mode)
    selected_docs = selected_docs[:min(len(selected_docs), final_n)]
    refusal_reason = m._negative_clean_refusal_reason(query, selected_docs, fnames)
    if refusal_reason:
        return {
            "documents": [],
            "sources": [],
            "metadata": {
                "query": query,
                "refused": refusal_reason,
                "mode": mode,
                "rerank": enable_rerank,
                "pre_rerank_sources": pre_rerank_sources,
            },
        }

    display_docs = m._filter_display_sources(selected_docs, score_mode, qfilters, fnames, qtype, max_sources=3)
    documents = []
    for doc in selected_docs:
        documents.append({
            "source": m._hit_entity_source(doc) or "unknown",
            "score": m._hit_score(doc),
            "text": m._build_excerpt(m._hit_display_text(doc), query, 500),
            "metadata": m._hit_metadata(doc),
            "chunk_range": m._hit_chunk_range(doc),
        })
    return {
        "documents": documents,
        "sources": m._build_sources(display_docs if display_docs else selected_docs[:3], query, score_mode=score_mode),
        "metadata": {
            "query": query,
            "mode": mode,
            "rerank": enable_rerank,
            "pre_rerank_sources": pre_rerank_sources,
            "post_rerank_sources": _first_source_order(docs[:pool_n]),
            "final_sources": [doc["source"] for doc in documents],
            "docs_final": len(selected_docs),
            "rerank_used": bool(reranked["used"]),
            "dense_candidate_sources": _first_source_order(dense_hits[:10]),
            "lex_candidate_sources": _first_source_order(lex_hits[:10]),
            "doc_fallback_sources": [entry.get("source") for entry in doc_recall_plan[:5] if entry.get("source")],
        },
    }


def run_mode(case: Dict[str, Any], mode: str, enable_rerank: bool) -> Dict[str, Any]:
    handler = m.QueryHandler()
    t0 = time.perf_counter()
    with temp_config(ENABLE_RERANK=bool(enable_rerank)):
        data = asyncio.run(_run_mode(handler, case, mode, enable_rerank))
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {"data": data, "latency_ms": latency_ms}


def sources_from_result(data: Dict[str, Any]) -> List[str]:
    return [(doc.get("source") or "") for doc in (data.get("documents") or [])]


def is_refusal(data: Dict[str, Any]) -> bool:
    md = data.get("metadata") or {}
    return bool(md.get("refused") or md.get("blocked")) or not bool(data.get("documents") or [])


def hit_rate(expected: List[str], got: List[str], k: int) -> float:
    if not expected:
        return 0.0
    return 1.0 if any(src in got[:k] for src in expected) else 0.0


def eval_case(case: Dict[str, Any]) -> Dict[str, Any]:
    runs = {
        "dense_only": run_mode(case, "dense_only", enable_rerank=False),
        "lexical_only": run_mode(case, "lexical_only", enable_rerank=False),
        "hybrid_concat": run_mode(case, "hybrid_concat", enable_rerank=False),
        "hybrid_fusion": run_mode(case, "hybrid_fusion", enable_rerank=False),
        "hybrid_fusion_rerank": run_mode(case, "hybrid_fusion", enable_rerank=True),
    }
    expected = case.get("expected_sources") or []
    per_mode: Dict[str, Any] = {}
    for name, result in runs.items():
        data = result["data"]
        got = sources_from_result(data)
        per_mode[name] = {
            "returned_sources": got,
            "hit_top3": hit_rate(expected, got, 3),
            "hit_top5": hit_rate(expected, got, 5),
            "wrong_source": 1 if (expected and got and not any(src in got for src in expected)) else 0,
            "negative_clean": 1 if ((not expected) and is_refusal(data)) else 0,
            "mis_refusal": 1 if (expected and is_refusal(data)) else 0,
            "latency_ms": result["latency_ms"],
            "metadata": data.get("metadata") or {},
        }
    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "query": case.get("query"),
        "expected_sources": expected,
        "is_positive": bool(expected),
        "is_negative": not bool(expected),
        "modes": per_mode,
    }


def summarize_mode(results: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    items = [r["modes"][mode] for r in results]
    positives = [item for item, case in zip(items, results) if case["is_positive"]]
    negatives = [item for item, case in zip(items, results) if case["is_negative"]]
    latencies = [item["latency_ms"] for item in items]
    latencies_sorted = sorted(latencies)
    p95_idx = int(0.95 * (len(latencies_sorted) - 1)) if latencies_sorted else 0
    return {
        "cases": len(items),
        "positive_hit_rate_top3": (sum(item["hit_top3"] for item in positives) / len(positives)) if positives else 0.0,
        "positive_hit_rate_top5": (sum(item["hit_top5"] for item in positives) / len(positives)) if positives else 0.0,
        "wrong_source_rate": (sum(item["wrong_source"] for item in positives) / len(positives)) if positives else 0.0,
        "mis_refusal_rate": (sum(item["mis_refusal"] for item in positives) / len(positives)) if positives else 0.0,
        "negative_clean_rate": (sum(item["negative_clean"] for item in negatives) / len(negatives)) if negatives else 0.0,
        "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency_ms": latencies_sorted[p95_idx] if latencies_sorted else 0.0,
    }


def rerank_deltas(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    changed_top_source = []
    improved_cases = []
    worsened_cases = []
    for result in results:
        no_rerank = result["modes"]["hybrid_fusion"]
        rerank = result["modes"]["hybrid_fusion_rerank"]
        before = (no_rerank["returned_sources"] or [None])[0]
        after = (rerank["returned_sources"] or [None])[0]
        expected = result["expected_sources"]
        if before != after:
            changed_top_source.append({
                "id": result["id"],
                "query": result["query"],
                "before": before,
                "after": after,
                "expected": expected,
            })
        before_ok = bool(expected) and any(src in no_rerank["returned_sources"][:3] for src in expected)
        after_ok = bool(expected) and any(src in rerank["returned_sources"][:3] for src in expected)
        if (not before_ok) and after_ok:
            improved_cases.append(result["id"])
        if before_ok and (not after_ok):
            worsened_cases.append(result["id"])
    return {
        "top_source_changed_cases": changed_top_source,
        "improved_top3_cases": improved_cases,
        "worsened_top3_cases": worsened_cases,
    }


def main() -> None:
    if (not _inside_container()) and (not ARGS.inside_container):
        _run_via_container()
        return
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    cases = retrieval_cases(load_cases())
    results = [eval_case(case) for case in cases]
    summary = {
        "scope": {
            "included_case_ids": [case["id"] for case in cases],
            "case_file": CASES_PATH,
            "excluded_categories": [] if os.path.basename(CASES_PATH) == os.path.basename(DEFAULT_CASES_PATH) else ["document_existence", "deleted_invisible"],
            "note": "Default ablation runs on the hard competitive retrieval set; routed intent cases stay out of the default regression path.",
        },
        "fusion_ablation": {
            mode: summarize_mode(results, mode)
            for mode in ("dense_only", "lexical_only", "hybrid_concat", "hybrid_fusion")
        },
        "rerank_ablation": {
            mode: summarize_mode(results, mode)
            for mode in ("hybrid_fusion", "hybrid_fusion_rerank")
        },
        "rerank_deltas": rerank_deltas(results),
    }
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
        "summary": summary,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(JSON_BEGIN)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(JSON_END)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()