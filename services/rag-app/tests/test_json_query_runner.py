import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import request


ROOT = Path(__file__).resolve().parents[3]
TEST_PATH = ROOT / "test.json"
DEFAULT_REPORT_PATH = ROOT / "services" / "rag-app" / "uploads" / "test_json_query_report.compare_degraded.v1.json"
BASE_URL = os.getenv("TEST_JSON_API", "http://127.0.0.1:8080")


def load_cases() -> List[Tuple[int, str]]:
    text = TEST_PATH.read_text(encoding="utf-8")
    cases: List[Tuple[int, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\.\s*(.+)$", line)
        if not m:
            continue
        idx = int(m.group(1))
        q = (m.group(2) or "").strip()
        if q:
            cases.append((idx, q))
    return cases


def call_query(query: str, user_id: str, top_k: int, enable_rerank: bool) -> Tuple[int, Dict[str, Any]]:
    payload = json.dumps(
        {"query": query, "user_id": user_id, "top_k": int(top_k), "enable_rerank": bool(enable_rerank)},
        ensure_ascii=False,
    ).encode("utf-8")
    req = request.Request(f"{BASE_URL}/query", data=payload, headers={"Content-Type": "application/json"})
    last_error: Optional[Exception] = None
    for timeout in (60, 75, 90):
        try:
            started = time.time()
            with request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return int(round((time.time() - started) * 1000)), data
        except Exception as exc:
            last_error = exc
            time.sleep(0.2)
    raise last_error or RuntimeError("query failed")


def percentile(arr: List[float], p: float) -> float:
    if not arr:
        return 0.0
    arr2 = sorted(arr)
    idx = int(round((p / 100.0) * (len(arr2) - 1)))
    idx = min(max(idx, 0), len(arr2) - 1)
    return float(arr2[idx])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--user-id", default="test_json_runner_compare_degraded_v1")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--enable-rerank", action="store_true", default=True)
    args = parser.parse_args()

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    cases = load_cases()
    results: List[Dict[str, Any]] = []
    latencies: List[float] = []
    final_channel_counts: Dict[str, int] = {}
    refusal_reason_counts: Dict[str, int] = {}
    evidence_coverage_reason_counts: Dict[str, int] = {}
    ok_count = 0

    for idx, query in cases:
        try:
            elapsed_ms, data = call_query(
                query=query,
                user_id=str(args.user_id),
                top_k=int(args.top_k),
                enable_rerank=bool(args.enable_rerank),
            )
            ok = True
            ok_count += 1
        except Exception as exc:
            elapsed_ms, data = 0, {"metadata": {"final_channel": "unknown", "error": f"{type(exc).__name__}: {exc}"}}
            ok = False

        md = data.get("metadata") or {}
        final_channel = str(md.get("final_channel") or "unknown")
        refusal_reason = str(md.get("refusal_reason") or "")
        evidence_coverage_reason = str(md.get("evidence_coverage_reason") or "unknown")
        answer_scope = str(md.get("answer_scope") or "unknown")
        answer = data.get("answer") or ""
        answer_preview = re.sub(r"\s+", " ", str(answer)).strip()
        if len(answer_preview) > 220:
            answer_preview = answer_preview[:220].rstrip() + "..."

        final_channel_counts[final_channel] = final_channel_counts.get(final_channel, 0) + 1
        if refusal_reason:
            refusal_reason_counts[refusal_reason] = refusal_reason_counts.get(refusal_reason, 0) + 1
        evidence_coverage_reason_counts[evidence_coverage_reason] = evidence_coverage_reason_counts.get(evidence_coverage_reason, 0) + 1
        latencies.append(float(elapsed_ms))

        results.append(
            {
                "id": idx,
                "query": query,
                "ok": bool(ok),
                "latency_ms": int(elapsed_ms),
                "final_channel": final_channel,
                "refusal_reason": refusal_reason,
                "evidence_coverage_reason": evidence_coverage_reason,
                "answer_scope": answer_scope,
                "answer_preview": answer_preview,
                "internal_route": md.get("internal_route") or "",
                "source_lock_reason": md.get("source_lock_reason") or "",
                "compare_degraded": bool(md.get("compare_degraded")),
                "compare_missing_targets": list(md.get("compare_missing_targets") or []),
                "is_comparison": bool((md.get("recall") or {}).get("is_comparison")),
                "candidate_sources": list(md.get("candidate_sources") or []),
                "target_text": md.get("target_text") or "",
                "target_sources": list(md.get("target_sources") or []),
                "covered_aspects": list(md.get("covered_aspects") or []),
                "uncovered_aspects": list(md.get("uncovered_aspects") or []),
                "qualified_substantive_chunks": md.get("qualified_substantive_chunks"),
                "intra_doc_focus_score": md.get("intra_doc_focus_score"),
                "generic_only": md.get("generic_only"),
                "heading_only": md.get("heading_only"),
                "rescue_attempted": md.get("rescue_attempted"),
                "rescue_success": md.get("rescue_success"),
                "source_resolution_trace": md.get("source_resolution_trace") or {},
            }
        )

    report = {
        "summary": {
            "total_cases": len(cases),
            "ok": ok_count,
            "final_channel_counts": final_channel_counts,
            "refusal_reason_counts": refusal_reason_counts,
            "evidence_coverage_reason_counts": evidence_coverage_reason_counts,
            "latency_ms_p50": int(round(percentile(latencies, 50))) if latencies else 0,
            "latency_ms_p95": int(round(percentile(latencies, 95))) if latencies else 0,
            "latency_ms_mean": int(round(statistics.mean(latencies))) if latencies else 0,
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "api_base": BASE_URL,
        "top_k": int(args.top_k),
        "enable_rerank": bool(args.enable_rerank),
        "results": results,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report_path": str(report_path),
                "total_cases": report["summary"]["total_cases"],
                "ok": report["summary"]["ok"],
                "final_channel_counts": report["summary"]["final_channel_counts"],
                "refusal_reason_counts": report["summary"]["refusal_reason_counts"],
                "latency_ms_p50": report["summary"]["latency_ms_p50"],
                "latency_ms_p95": report["summary"]["latency_ms_p95"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
