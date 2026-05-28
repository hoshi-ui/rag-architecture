import json
import os
import statistics
import time
from collections import defaultdict
from typing import Any, Dict, List

import requests

from source_equivalence import (
    has_canonical_source_match,
    has_exact_source_match,
    load_document_registry,
    normalize_source_name,
    source_is_visible_or_equivalent,
)


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
CASES_PATH = os.path.join(THIS_DIR, "real_regulation_cases.json")
REPORT_PATH = os.path.abspath(os.path.join(THIS_DIR, "..", "uploads", "real_regulation_baseline_report.json"))
BASE_URL = os.getenv("REAL_BASELINE_API", "http://127.0.0.1:8080")


def load_cases() -> List[Dict[str, Any]]:
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def run_retrieve(query: str, top_k: int = 10, enable_rerank: bool = True) -> Dict[str, Any]:
    t0 = time.perf_counter()
    response = requests.post(
        f"{BASE_URL}/retrieve",
        json={"query": query, "user_id": "real_reg_baseline", "top_k": top_k, "enable_rerank": enable_rerank},
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
    return "ok"


def eval_case(case: Dict[str, Any], registry: Dict[str, Any]) -> Dict[str, Any]:
    result = run_retrieve(case["query"], top_k=10, enable_rerank=True)
    data = result["data"]
    expected = case.get("expected_sources") or []
    normalized_expected = [normalize_source_name(src) for src in expected]
    visible_sources = registry["visible_sources"]
    visible_canonical_ids = registry["visible_canonical_ids"]
    source_to_canonical = registry["source_to_canonical"]
    missing_expected_sources = [
        src for src in normalized_expected
        if src and not source_is_visible_or_equivalent(src, visible_sources, visible_canonical_ids, source_to_canonical)
    ]
    got = sources_from_result(data)
    stale_positive = bool(normalized_expected) and bool(missing_expected_sources)
    failure_reason = "stale_positive_missing_source" if stale_positive else classify_failure(result["status_code"], data, expected, got, source_to_canonical)
    return {
        "id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "expected_sources": expected,
        "returned_sources": got,
        "is_positive": bool(expected),
        "is_negative": not bool(expected),
        "visible_expected_sources": [src for src in normalized_expected if src in visible_sources],
        "missing_expected_sources": missing_expected_sources,
        "stale_positive": 1 if stale_positive else 0,
        "mis_refusal": 1 if (expected and not stale_positive and is_refusal(data)) else 0,
        "wrong_source": 1 if (expected and failure_reason == "wrong_source") else 0,
        "physical_mismatch": 1 if (expected and failure_reason == "physical_mismatch") else 0,
        "negative_clean": 1 if ((not expected) and is_refusal(data)) else 0,
        "failure_reason": failure_reason,
        "latency_ms": result["latency_ms"],
        "metadata": data.get("metadata") or {},
    }


def bucket_metrics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {
            "cases": 0,
            "eval_cases": 0,
            "wrong_source_rate": 0.0,
            "mis_refusal_rate": 0.0,
            "negative_clean_rate": 0.0,
            "p95_latency_ms": 0.0,
            "failure_reasons": {},
            "stale_positive_cases": 0,
        }
    failure_reasons: Dict[str, int] = {}
    for item in items:
        reason = item.get("failure_reason") or "ok"
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    eval_items = [item for item in items if not item.get("stale_positive")]
    stale_positive_items = [item for item in items if item.get("stale_positive")]
    lats = sorted(item["latency_ms"] for item in eval_items)
    p95_idx = int(0.95 * (len(lats) - 1)) if lats else 0
    negatives = [item for item in eval_items if item["is_negative"]]
    denom = len(eval_items) or 1
    return {
        "cases": len(items),
        "eval_cases": len(eval_items),
        "wrong_source_rate": sum(item["wrong_source"] for item in eval_items) / denom,
        "mis_refusal_rate": sum(item["mis_refusal"] for item in eval_items) / denom,
        "negative_clean_rate": (sum(item["negative_clean"] for item in negatives) / len(negatives)) if negatives else 0.0,
        "mean_latency_ms": statistics.mean(item["latency_ms"] for item in eval_items) if eval_items else 0.0,
        "p95_latency_ms": lats[p95_idx] if lats else 0.0,
        "failure_reasons": failure_reasons,
        "stale_positive_cases": len(stale_positive_items),
    }


def main() -> None:
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    cases = load_cases()
    registry = load_document_registry(BASE_URL)
    visible_sources = registry["visible_sources"]
    results = [eval_case(case, registry) for case in cases]
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
            "stale_positive_missing_source": [item for item in results if item["stale_positive"]],
            "negative_dirty": [item for item in results if item["is_negative"] and not item["negative_clean"]],
        },
        "visible_sources": sorted(visible_sources),
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