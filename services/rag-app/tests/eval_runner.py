import os
import json
import time
import statistics
import importlib.util
from typing import List, Dict, Any
from fastapi.testclient import TestClient

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAIN_PATH = os.path.join(BASE_DIR, "main.py")
CASES_PATH = os.path.join(os.path.dirname(__file__), "eval_cases.json")
REPORT_PATH = os.path.join(BASE_DIR, "uploads", "eval_report.json")

# local test profile
os.environ.setdefault("APP_ENV", "test_local")
os.environ.setdefault("EMBEDDING_SERVICE_URL", "http://127.0.0.1:8001")
os.environ.setdefault("RERANK_SERVICE_URL", "http://127.0.0.1:8002")
os.environ.setdefault("MILVUS_HOST", "127.0.0.1")
os.environ.setdefault("MILVUS_PORT", "19530")

spec = importlib.util.spec_from_file_location("rag_eval_main", MAIN_PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
client = TestClient(m.app)

def load_cases() -> List[Dict[str, Any]]:
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def run_retrieve(query: str, top_k: int, enable_rerank: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    r = client.post("/retrieve", json={"query": query, "user_id": "eval", "top_k": top_k, "enable_rerank": enable_rerank})
    t1 = time.perf_counter()
    latency = (t1 - t0) * 1000.0
    ok = r.status_code == 200
    data = r.json() if ok else {"documents": [], "sources": [], "metadata": {"error_status": r.status_code}}
    return {"data": data, "latency_ms": latency, "status_code": r.status_code}

def sources_from_result(data: Dict[str, Any]) -> List[str]:
    docs = data.get("documents") or []
    return [d.get("source") or "" for d in docs]

def hit_rate(expected: List[str], got: List[str], k: int) -> float:
    if not expected:
        return 0.0
    topk = got[:k]
    return 1.0 if any([(e in topk) for e in expected]) else 0.0

def is_refusal(data: Dict[str, Any]) -> bool:
    md = data.get("metadata") or {}
    return bool(md.get("refused") or md.get("blocked")) or (len(data.get("documents") or []) == 0)

def classify_failure(status_code: int, data: Dict[str, Any], expected: List[str], got: List[str]) -> str:
    md = data.get("metadata") or {}
    if status_code >= 500:
        return "service_unavailable"
    if md.get("blocked") == "query_too_short":
        return "empty_recall"
    if expected and is_refusal(data):
        return "mis_refusal"
    if is_refusal(data):
        return "empty_recall"
    if expected and got and not any([(e in got) for e in expected]):
        return "wrong_source"
    if md.get("error_status") == 504 or ("timeout" in (md.get("error") or "").lower()):
        return "timeout"
    return "ok"

def eval_case(case: Dict[str, Any]) -> Dict[str, Any]:
    q = case.get("query") or ""
    expected = case.get("expected_sources") or []
    m.config.LEXICAL_RECALL_LIMIT = 1000
    res_hybrid_no_rerank = run_retrieve(q, top_k=10, enable_rerank=False)
    res_hybrid_rerank = run_retrieve(q, top_k=10, enable_rerank=True)
    m.config.LEXICAL_RECALL_LIMIT = 0
    res_dense_no_rerank = run_retrieve(q, top_k=10, enable_rerank=False)
    res_dense_rerank = run_retrieve(q, top_k=10, enable_rerank=True)
    got_hybrid = sources_from_result(res_hybrid_rerank["data"])
    got_dense = sources_from_result(res_dense_rerank["data"])
    hr3_hybrid = hit_rate(expected, got_hybrid, 3)
    hr5_hybrid = hit_rate(expected, got_hybrid, 5)
    hr3_dense = hit_rate(expected, got_dense, 3)
    hr5_dense = hit_rate(expected, got_dense, 5)
    refusal = 1.0 if (expected and is_refusal(res_hybrid_rerank["data"])) else 0.0
    fail_reason = classify_failure(res_hybrid_rerank["status_code"], res_hybrid_rerank["data"], expected, got_hybrid)
    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "query": q,
        "expected": expected,
        "is_positive": 1 if expected else 0,
        "is_negative": 1 if not expected else 0,
        "hit_top3": 1 if (hr3_hybrid > 0) else 0,
        "hit_top5": 1 if (hr5_hybrid > 0) else 0,
        "negative_clean": 1 if (not expected and is_refusal(res_hybrid_rerank["data"])) else 0,
        "hybrid_top3": hr3_hybrid,
        "hybrid_top5": hr5_hybrid,
        "dense_top3": hr3_dense,
        "dense_top5": hr5_dense,
        "refusal": refusal,
        "failure_reason": fail_reason,
        "latency_ms": {
            "hybrid_no_rerank": res_hybrid_no_rerank["latency_ms"],
            "hybrid_rerank": res_hybrid_rerank["latency_ms"],
            "dense_no_rerank": res_dense_no_rerank["latency_ms"],
            "dense_rerank": res_dense_rerank["latency_ms"]
        }
    }

def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    def avg(key: str) -> float:
        return sum([r.get(key, 0.0) for r in results]) / max(n, 1)
    # positive / negative buckets
    positives = [r for r in results if r.get("is_positive")]
    negatives = [r for r in results if r.get("is_negative")]
    def rate(lst: List[Dict[str, Any]], key: str) -> float:
        if not lst:
            return 0.0
        return sum([1 for r in lst if r.get(key)]) / len(lst)
    lats = {
        "hybrid_no_rerank": [r["latency_ms"]["hybrid_no_rerank"] for r in results],
        "hybrid_rerank": [r["latency_ms"]["hybrid_rerank"] for r in results],
        "dense_no_rerank": [r["latency_ms"]["dense_no_rerank"] for r in results],
        "dense_rerank": [r["latency_ms"]["dense_rerank"] for r in results]
    }
    def p95(arr: List[float]) -> float:
        if not arr:
            return 0.0
        arr2 = sorted(arr)
        idx = int(0.95 * (len(arr2) - 1))
        return arr2[idx]
    reason_counts: Dict[str, int] = {}
    for r in results:
        reason = r.get("failure_reason") or "ok"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "cases": n,
        "metrics": {
            "positive_cases": len(positives),
            "negative_cases": len(negatives),
            "positive_hit_rate_top3": rate(positives, "hit_top3"),
            "positive_hit_rate_top5": rate(positives, "hit_top5"),
            "negative_clean_rate": rate(negatives, "negative_clean"),
            "hybrid_top3_hit_rate": avg("hybrid_top3"),
            "hybrid_top5_hit_rate": avg("hybrid_top5"),
            "dense_top3_hit_rate": avg("dense_top3"),
            "dense_top5_hit_rate": avg("dense_top5"),
            "mis_refusal_rate": avg("refusal"),
            "latency": {
                "hybrid_no_rerank_mean": statistics.mean(lats["hybrid_no_rerank"]) if lats["hybrid_no_rerank"] else 0.0,
                "hybrid_no_rerank_p95": p95(lats["hybrid_no_rerank"]),
                "hybrid_rerank_mean": statistics.mean(lats["hybrid_rerank"]) if lats["hybrid_rerank"] else 0.0,
                "hybrid_rerank_p95": p95(lats["hybrid_rerank"]),
                "dense_no_rerank_mean": statistics.mean(lats["dense_no_rerank"]) if lats["dense_no_rerank"] else 0.0,
                "dense_no_rerank_p95": p95(lats["dense_no_rerank"]),
                "dense_rerank_mean": statistics.mean(lats["dense_rerank"]) if lats["dense_rerank"] else 0.0,
                "dense_rerank_p95": p95(lats["dense_rerank"])
            },
            "failure_reasons": reason_counts
        }
    }

def main():
    os.makedirs(os.path.join(BASE_DIR, "uploads"), exist_ok=True)
    cases = load_cases()
    results = [eval_case(c) for c in cases]
    report = {"results": results, "summary": aggregate(results), "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report["summary"], ensure_ascii=False))

if __name__ == "__main__":
    main()
