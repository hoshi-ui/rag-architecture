import argparse
import asyncio
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
MAIN_PATH = os.path.join(BASE_DIR, "main.py")
DEFAULT_CASES_PATH = os.path.join(THIS_DIR, "hard_competitive_cases.json")
DEFAULT_REPORT_PATH = os.path.abspath(os.path.join(BASE_DIR, "uploads", "dense_lexical_offline_eval_report.json"))
JSON_BEGIN = "===DENSE_LEXICAL_OFFLINE_EVAL_JSON_BEGIN==="
JSON_END = "===DENSE_LEXICAL_OFFLINE_EVAL_JSON_END==="


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pure dense-vs-lexical offline retrieval evaluation.")
    parser.add_argument("--inside-container", action="store_true")
    parser.add_argument("--cases", default=os.getenv("DENSE_LEXICAL_EVAL_CASES_PATH", DEFAULT_CASES_PATH))
    parser.add_argument("--report", default=os.getenv("DENSE_LEXICAL_EVAL_REPORT_PATH", DEFAULT_REPORT_PATH))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("DENSE_LEXICAL_EVAL_TOP_K", "20")))
    parser.add_argument("--top-sources", type=int, default=int(os.getenv("DENSE_LEXICAL_EVAL_TOP_SOURCES", "10")))
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


spec = importlib.util.spec_from_file_location("rag_dense_lexical_offline_eval_main", MAIN_PATH)
m = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(m)


def _extract_embedded_json(output: str) -> Dict[str, Any]:
    start = output.find(JSON_BEGIN)
    end = output.find(JSON_END)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("offline eval runner did not emit embedded JSON payload")
    payload = output[start + len(JSON_BEGIN):end].strip()
    return json.loads(payload)


def _container_test_path(host_path: str) -> str:
    base = os.path.abspath(THIS_DIR)
    target = os.path.abspath(host_path)
    if target.startswith(base):
        rel = os.path.relpath(target, base)
        return os.path.join("/app/tests", rel)
    return target


def _container_upload_path(host_path: str) -> str:
    base = os.path.abspath(os.path.join(BASE_DIR, "uploads"))
    target = os.path.abspath(host_path)
    if target.startswith(base):
        rel = os.path.relpath(target, base)
        return os.path.join("/app/uploads", rel)
    return "/app/uploads/dense_lexical_offline_eval_report.json"


def _run_via_container() -> None:
    cmd = [
        "docker",
        "exec",
        "rag-app",
        "python",
        "/app/tests/dense_lexical_offline_eval.py",
        "--inside-container",
        "--cases",
        _container_test_path(CASES_PATH),
        "--report",
        _container_upload_path(REPORT_PATH),
        "--top-k",
        str(int(ARGS.top_k)),
        "--top-sources",
        str(int(ARGS.top_sources)),
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


def _unique_sources(hits: List[Any], limit: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for hit in hits:
        src = m._normalize_filename_for_match(m._hit_entity_source(hit) or "")
        if not src or src in seen:
            continue
        seen.add(src)
        out.append(src)
        if len(out) >= limit:
            break
    return out


def _top_hit_payload(hits: List[Any], query: str) -> Optional[Dict[str, Any]]:
    if not hits:
        return None
    top_hit = hits[0]
    return {
        "source": m._normalize_filename_for_match(m._hit_entity_source(top_hit) or ""),
        "score": m._hit_score(top_hit),
        "section": (m._hit_metadata(top_hit).get("section") or "").strip(),
        "excerpt": m._build_excerpt(m._hit_display_text(top_hit), query, 220),
    }


def _hybrid_hits(query: str, dense_hits: List[Any], lexical_hits: List[Any], top_k: int) -> List[Any]:
    docs_all = list(dense_hits) + list(lexical_hits)
    if not docs_all:
        return []
    dense_rank_map: Dict[str, int] = {}
    for idx, hit in enumerate(dense_hits):
        src = m._normalize_filename_for_match(m._hit_entity_source(hit) or "")
        if src and src not in dense_rank_map:
            dense_rank_map[src] = idx
    lex_rank_map: Dict[str, int] = {}
    for idx, hit in enumerate(lexical_hits):
        src = m._normalize_filename_for_match(m._hit_entity_source(hit) or "")
        if src and src not in lex_rank_map:
            lex_rank_map[src] = idx
    source_count: Dict[str, int] = {}
    for hit in docs_all:
        src = m._normalize_filename_for_match(m._hit_entity_source(hit) or "")
        if src:
            source_count[src] = source_count.get(src, 0) + 1
    source_signals = m._build_source_signal_map(query, lexical_hits, [])
    combined = []
    fused_source_scores: Dict[str, float] = {}
    for idx, hit in enumerate(docs_all):
        src = m._normalize_filename_for_match(m._hit_entity_source(hit) or "")
        if src not in fused_source_scores:
            fused_source_scores[src] = m._fusion_source_score(src, query, dense_rank_map, lex_rank_map, source_count, source_signals, set(), set(), False)
        combined.append((fused_source_scores[src], idx))
    combined.sort(key=lambda item: item[0], reverse=True)
    seen = set()
    hits: List[Any] = []
    for fused_score, idx in combined:
        hit = docs_all[idx]
        key = (m._hit_entity_source(hit) or "unknown", (m._hit_entity_text(hit) or "")[:64])
        if key in seen:
            continue
        seen.add(key)
        hits.append(m._clone_hit_with_score(hit, fused_score))
        if len(hits) >= top_k:
            break
    return hits


def _reciprocal_rank(expected: List[str], got: List[str]) -> float:
    if not expected:
        return 0.0
    expected_set = {m._normalize_filename_for_match(src) for src in expected}
    for idx, src in enumerate(got, start=1):
        if m._normalize_filename_for_match(src) in expected_set:
            return 1.0 / float(idx)
    return 0.0


def _recall_at_k(expected: List[str], got: List[str], k: int) -> float:
    if not expected:
        return 0.0
    expected_set = {m._normalize_filename_for_match(src) for src in expected}
    return 1.0 if any(m._normalize_filename_for_match(src) in expected_set for src in got[:k]) else 0.0


async def _dense_hits(handler: Any, query: str, top_k: int) -> List[Any]:
    embedding = (await handler.embedding_service.embed([query]))[0]
    hits = handler.vector_db.search(embedding, top_k=top_k, filters=None)
    return m._filter_hits_by_source_state(hits)["hits"]


def _lexical_hits(query: str, top_k: int) -> List[Any]:
    hits = m._lexical_recall_indexed(query, limit=top_k, source_filter=None)
    return m._filter_hits_by_source_state(hits)["hits"]


def _positive_case(case: Dict[str, Any]) -> bool:
    return bool(case.get("expected_sources") or [])


def _evaluate_case(handler: Any, case: Dict[str, Any], top_k: int, top_sources: int) -> Dict[str, Any]:
    query = m._normalize_query(case["query"])
    expected = [m._normalize_filename_for_match(src) for src in (case.get("expected_sources") or [])]

    t0 = time.perf_counter()
    dense_hits = asyncio.run(_dense_hits(handler, query, top_k))
    dense_latency_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    lexical_hits = _lexical_hits(query, top_k)
    lexical_latency_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    hybrid_hits = _hybrid_hits(query, dense_hits, lexical_hits, top_k)
    hybrid_latency_ms = (time.perf_counter() - t2) * 1000.0

    dense_sources = _unique_sources(dense_hits, top_sources)
    lexical_sources = _unique_sources(lexical_hits, top_sources)
    hybrid_sources = _unique_sources(hybrid_hits, top_sources)

    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "query": query,
        "expected_sources": expected,
        "dense": {
            "sources": dense_sources,
            "recall_at_1": _recall_at_k(expected, dense_sources, 1),
            "recall_at_3": _recall_at_k(expected, dense_sources, 3),
            "recall_at_5": _recall_at_k(expected, dense_sources, 5),
            "mrr": _reciprocal_rank(expected, dense_sources),
            "latency_ms": dense_latency_ms,
            "top_hit": _top_hit_payload(dense_hits, query),
        },
        "lexical": {
            "sources": lexical_sources,
            "recall_at_1": _recall_at_k(expected, lexical_sources, 1),
            "recall_at_3": _recall_at_k(expected, lexical_sources, 3),
            "recall_at_5": _recall_at_k(expected, lexical_sources, 5),
            "mrr": _reciprocal_rank(expected, lexical_sources),
            "latency_ms": lexical_latency_ms,
            "top_hit": _top_hit_payload(lexical_hits, query),
        },
        "hybrid": {
            "sources": hybrid_sources,
            "recall_at_1": _recall_at_k(expected, hybrid_sources, 1),
            "recall_at_3": _recall_at_k(expected, hybrid_sources, 3),
            "recall_at_5": _recall_at_k(expected, hybrid_sources, 5),
            "mrr": _reciprocal_rank(expected, hybrid_sources),
            "latency_ms": dense_latency_ms + lexical_latency_ms + hybrid_latency_ms,
            "top_hit": _top_hit_payload(hybrid_hits, query),
        },
    }


def _summarize(results: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    positives = [result for result in results if result.get("expected_sources")]
    latencies = [result[key]["latency_ms"] for result in results]
    latencies_sorted = sorted(latencies)
    p95_idx = int(0.95 * (len(latencies_sorted) - 1)) if latencies_sorted else 0
    return {
        "cases": len(results),
        "positive_cases": len(positives),
        "recall_at_1": (sum(result[key]["recall_at_1"] for result in positives) / len(positives)) if positives else 0.0,
        "recall_at_3": (sum(result[key]["recall_at_3"] for result in positives) / len(positives)) if positives else 0.0,
        "recall_at_5": (sum(result[key]["recall_at_5"] for result in positives) / len(positives)) if positives else 0.0,
        "mrr": (sum(result[key]["mrr"] for result in positives) / len(positives)) if positives else 0.0,
        "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency_ms": latencies_sorted[p95_idx] if latencies_sorted else 0.0,
    }


def _delta_cases(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    dense_hit_lex_miss = []
    lex_hit_dense_miss = []
    dense_negative_retrieval = []
    hybrid_negative_scores = []

    for result in results:
        dense_hit = bool(result["dense"]["recall_at_5"])
        lexical_hit = bool(result["lexical"]["recall_at_5"])
        if result.get("expected_sources"):
            if dense_hit and not lexical_hit:
                dense_hit_lex_miss.append({
                    "id": result["id"],
                    "category": result["category"],
                    "query": result["query"],
                    "expected_sources": result["expected_sources"],
                    "dense_sources": result["dense"]["sources"][:5],
                    "lexical_sources": result["lexical"]["sources"][:5],
                    "dense_top_hit": result["dense"]["top_hit"],
                    "lexical_top_hit": result["lexical"]["top_hit"],
                })
            if lexical_hit and not dense_hit:
                lex_hit_dense_miss.append({
                    "id": result["id"],
                    "category": result["category"],
                    "query": result["query"],
                    "expected_sources": result["expected_sources"],
                    "dense_sources": result["dense"]["sources"][:5],
                    "lexical_sources": result["lexical"]["sources"][:5],
                    "dense_top_hit": result["dense"]["top_hit"],
                    "lexical_top_hit": result["lexical"]["top_hit"],
                })
        elif result["dense"]["sources"]:
            dense_negative_retrieval.append({
                "id": result["id"],
                "category": result["category"],
                "query": result["query"],
                "dense_sources": result["dense"]["sources"][:5],
                "lexical_sources": result["lexical"]["sources"][:5],
                "dense_top_hit": result["dense"]["top_hit"],
                "lexical_top_hit": result["lexical"]["top_hit"],
            })
        if not result.get("expected_sources"):
            hybrid_negative_scores.append({
                "id": result["id"],
                "category": result["category"],
                "query": result["query"],
                "hybrid_sources": result["hybrid"]["sources"][:5],
                "hybrid_top_hit": result["hybrid"]["top_hit"],
                "below_warning_line_0_25": bool((result["hybrid"]["top_hit"] or {}).get("score", 0.0) < 0.25),
            })

    return {
        "dense_hit_lex_miss_cases": dense_hit_lex_miss,
        "lex_hit_dense_miss_cases": lex_hit_dense_miss,
        "dense_negative_retrieval_cases": dense_negative_retrieval,
        "hybrid_negative_scores": hybrid_negative_scores,
    }


def main() -> None:
    if (not _inside_container()) and (not ARGS.inside_container):
        _run_via_container()
        return

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    cases = load_cases()
    handler = m.QueryHandler()
    results = [_evaluate_case(handler, case, int(ARGS.top_k), int(ARGS.top_sources)) for case in cases]
    summary = {
        "scope": {
            "cases_path": CASES_PATH,
            "top_k": int(ARGS.top_k),
            "top_sources": int(ARGS.top_sources),
            "evaluation_mode": "pure_retrieval_only",
            "notes": [
                "Dense path only runs normalize_query -> embed -> Milvus search -> visible-source filtering -> unique source ranking.",
                "Lexical path only runs normalize_query -> SQLite FTS indexed recall -> visible-source filtering -> unique source ranking.",
                "No fusion, no doc fallback, no title bonus, no rerank, no relevance threshold, no answer gate.",
            ],
        },
        "dense_only": _summarize(results, "dense"),
        "lexical_only": _summarize(results, "lexical"),
        "hybrid": _summarize(results, "hybrid"),
        "deltas": _delta_cases(results),
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