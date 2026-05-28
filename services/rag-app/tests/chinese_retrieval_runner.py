import json
import os
import statistics
import time
from collections import defaultdict
from typing import Any, Dict, List

import requests

from source_equivalence import has_canonical_source_match, has_exact_source_match, load_document_registry


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
CASES_PATH = os.path.join(THIS_DIR, "chinese_retrieval_cases.json")
REPORT_PATH = os.path.abspath(os.path.join(THIS_DIR, "..", "uploads", "chinese_retrieval_report.json"))
BASE_URL = os.getenv("REAL_BASELINE_API", "http://127.0.0.1:8080")


def load_cases() -> List[Dict[str, Any]]:
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_retrieve(query: str, top_k: int = 10, enable_rerank: bool = True) -> Dict[str, Any]:
    t0 = time.perf_counter()
    response = requests.post(
        f"{BASE_URL}/retrieve",
        json={"query": query, "user_id": "chinese_retrieval_regression", "top_k": top_k, "enable_rerank": enable_rerank},
        timeout=180,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    return {"status_code": response.status_code, "latency_ms": latency_ms, "data": data}


def sources_from_result(data: Dict[str, Any]) -> List[str]:
    return [(doc.get("source") or "") for doc in (data.get("documents") or [])]


def is_refusal(data: Dict[str, Any]) -> bool:
    md = data.get("metadata") or {}
    return bool(md.get("refused") or md.get("blocked")) or not bool(data.get("documents") or [])


def classify_failure(status_code: int, data: Dict[str, Any], expected: List[str], got: List[str], source_to_canonical: Dict[str, str]) -> str:
    if status_code >= 500:
        return "service_unavailable"
    if expected and is_refusal(data):
        return "mis_refusal"
    if is_refusal(data):
        return "empty_recall"
    if expected and not has_exact_source_match(expected, got):
        if has_canonical_source_match(expected, got, source_to_canonical):
            return "physical_mismatch"
        return "wrong_source"
    if (not expected) and got:
        return "negative_dirty"
    return "ok"


def eval_case(case: Dict[str, Any], source_to_canonical: Dict[str, str]) -> Dict[str, Any]:
    result = run_retrieve(case["query"], top_k=10, enable_rerank=True)
    data = result["data"]
    expected = case.get("expected_sources") or []
    got = sources_from_result(data)
    failure_reason = classify_failure(result["status_code"], data, expected, got, source_to_canonical)
    return {
        "id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "expected_sources": expected,
        "returned_sources": got,
        "is_positive": bool(expected),
        "is_negative": not bool(expected),
        "mis_refusal": 1 if (expected and is_refusal(data)) else 0,
        "wrong_source": 1 if (expected and failure_reason == "wrong_source") else 0,
        "physical_mismatch": 1 if (expected and failure_reason == "physical_mismatch") else 0,
        "negative_clean": 1 if ((not expected) and is_refusal(data)) else 0,
        "negative_dirty": 1 if ((not expected) and bool(got)) else 0,
        "failure_reason": failure_reason,
        "latency_ms": result["latency_ms"],
        "metadata": data.get("metadata") or {},
    }


def bucket_metrics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {
            "cases": 0,
            "wrong_source_rate": 0.0,
            "mis_refusal_rate": 0.0,
            "negative_clean_rate": 0.0,
            "negative_dirty_rate": 0.0,
            "mean_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "failure_reasons": {},
        }
    positives = [item for item in items if item["is_positive"]]
    negatives = [item for item in items if item["is_negative"]]
    lats = sorted(item["latency_ms"] for item in items)
    p95_idx = int(0.95 * (len(lats) - 1)) if lats else 0
    failure_reasons: Dict[str, int] = {}
    for item in items:
        failure_reasons[item["failure_reason"]] = failure_reasons.get(item["failure_reason"], 0) + 1
    return {
        "cases": len(items),
        "wrong_source_rate": (sum(item["wrong_source"] for item in positives) / len(positives)) if positives else 0.0,
        "mis_refusal_rate": (sum(item["mis_refusal"] for item in positives) / len(positives)) if positives else 0.0,
        "negative_clean_rate": (sum(item["negative_clean"] for item in negatives) / len(negatives)) if negatives else 0.0,
        "negative_dirty_rate": (sum(item["negative_dirty"] for item in negatives) / len(negatives)) if negatives else 0.0,
        "mean_latency_ms": statistics.mean(item["latency_ms"] for item in items),
        "p95_latency_ms": lats[p95_idx] if lats else 0.0,
        "failure_reasons": failure_reasons,
    }


def main() -> None:
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    cases = load_cases()
    registry = load_document_registry(BASE_URL)
    source_to_canonical = registry["source_to_canonical"]
    results = [eval_case(case, source_to_canonical) for case in cases]
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in results:
        by_category[item["category"]].append(item)

    summary = {
        "focus_metrics": bucket_metrics(results),
        "by_category": {category: bucket_metrics(items) for category, items in by_category.items()},
        "focus_failures": {
            "wrong_source": [item for item in results if item["wrong_source"]],
            "physical_mismatch": [item for item in results if item["physical_mismatch"]],
            "mis_refusal": [item for item in results if item["mis_refusal"]],
            "negative_dirty": [item for item in results if item["negative_dirty"]],
        },
    }
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "api_base": BASE_URL,
        "cases": cases,
        "results": results,
        "summary": summary,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()