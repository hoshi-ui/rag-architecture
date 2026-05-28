import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Tuple


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

CHINESE_REPORT = os.path.join(UPLOAD_DIR, "chinese_retrieval_report.json")
REAL_BASELINE_REPORT = os.path.join(UPLOAD_DIR, "real_regulation_baseline_report.json")
HARD_COMPETITIVE_REPORT = os.path.join(UPLOAD_DIR, "hard_competitive_report.json")
OUTPUT_REPORT = os.path.join(UPLOAD_DIR, "guardrail_failure_diagnosis_report.json")


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _classify_failure_layer(item: Dict[str, Any]) -> Tuple[str, str]:
    md = item.get("metadata") or {}
    refusal_stage = (md.get("refusal_stage") or "").strip()
    refusal_reason = (md.get("refusal_reason") or item.get("failure_reason") or "").strip()
    source_lock_required = bool(md.get("source_lock_required"))
    source_lock_resolved = bool(md.get("source_lock_resolved"))
    docs_returned = _as_int(md.get("docs_returned"))
    if docs_returned <= 0:
        docs_returned = len(item.get("returned_sources") or [])
    qualified_chunks = _as_int(md.get("qualified_substantive_chunks"))

    if item.get("stale_positive"):
        return "stale_data", "expected_source_missing_from_library"
    if refusal_stage == "source_lock" or (source_lock_required and not source_lock_resolved):
        return "source_layer", refusal_reason or "source_lock_failed"
    if refusal_stage == "query_validation":
        return "source_layer", refusal_reason or "query_validation_blocked"
    if refusal_stage == "evidence":
        if docs_returned <= 0 or qualified_chunks <= 0:
            return "retrieval_chunk_layer", refusal_reason or "no_substantive_body_chunks"
        return "evidence_gate_layer", refusal_reason or "evidence_gate_refusal"
    if refusal_reason in {"query_anchor_miss", "no_relevant_evidence", "empty_evidence"}:
        return "retrieval_chunk_layer", refusal_reason
    return "unknown", refusal_reason or "unknown"


def _entry_view(item: Dict[str, Any], suite_name: str) -> Dict[str, Any]:
    md = item.get("metadata") or {}
    layer, detail = _classify_failure_layer(item)
    return {
        "suite": suite_name,
        "id": item.get("id") or "",
        "category": item.get("category") or "",
        "query": item.get("query") or "",
        "expected_sources": list(item.get("expected_sources") or []),
        "returned_sources": list(item.get("returned_sources") or []),
        "failure_reason": item.get("failure_reason") or "",
        "diagnosis_layer": layer,
        "diagnosis_detail": detail,
        "refusal_stage": md.get("refusal_stage"),
        "refusal_reason": md.get("refusal_reason"),
        "query_route": md.get("query_route"),
        "internal_route": md.get("internal_route"),
        "final_channel": md.get("final_channel"),
        "source_lock_required": bool(md.get("source_lock_required")),
        "source_lock_resolved": bool(md.get("source_lock_resolved")),
        "source_lock_reason": md.get("source_lock_reason") or "",
        "docs_returned": _as_int(md.get("docs_returned")),
        "qualified_substantive_chunks": _as_int(md.get("qualified_substantive_chunks")),
        "heading_only": bool(md.get("heading_only")),
        "answer_scope": md.get("answer_scope") or "",
        "target_sources": list(md.get("target_sources") or []),
        "candidate_sources": list(md.get("candidate_sources") or []),
        "missing_expected_sources": list(item.get("missing_expected_sources") or []),
    }


def _diagnose_suite(report: Dict[str, Any], suite_name: str) -> Dict[str, Any]:
    summary = report.get("summary") or {}
    failures = summary.get("focus_failures") or {}
    mis_refusal_entries = [_entry_view(item, suite_name) for item in (failures.get("mis_refusal") or [])]
    stale_entries = [_entry_view(item, suite_name) for item in (failures.get("stale_positive_missing_source") or [])]

    layer_counts = Counter(entry["diagnosis_layer"] for entry in mis_refusal_entries)
    detail_counts = Counter(entry["diagnosis_detail"] for entry in mis_refusal_entries)

    return {
        "focus_metrics": summary.get("focus_metrics") or {},
        "cleaned_eval_cases": (summary.get("focus_metrics") or {}).get("eval_cases"),
        "stale_positive_cases_removed": len(stale_entries),
        "mis_refusal_cases": len(mis_refusal_entries),
        "mis_refusal_layer_counts": dict(layer_counts),
        "mis_refusal_detail_counts": dict(detail_counts),
        "mis_refusal_entries": mis_refusal_entries,
        "stale_positive_entries": stale_entries,
    }


def _body_recall_status(entry: Dict[str, Any]) -> str:
    if not entry.get("source_lock_resolved"):
        return "source_not_locked"
    if (entry.get("docs_returned") or 0) <= 0:
        return "no_docs_returned"
    if (entry.get("qualified_substantive_chunks") or 0) <= 0:
        return "no_substantive_body_chunks"
    return "body_recalled"


def _build_audits(chinese_diag: Dict[str, Any], real_diag: Dict[str, Any], hard_diag: Dict[str, Any]) -> Dict[str, Any]:
    chinese_entries = list(chinese_diag.get("mis_refusal_entries") or [])
    real_entries = list(real_diag.get("mis_refusal_entries") or [])
    hard_entries = list(hard_diag.get("mis_refusal_entries") or [])
    all_live_entries = chinese_entries + real_entries + hard_entries

    locked_entries = [entry for entry in all_live_entries if entry.get("source_lock_resolved")]
    body_recall_counts = Counter(_body_recall_status(entry) for entry in locked_entries)
    evidence_gate_fix_candidates = [
        entry for entry in all_live_entries
        if entry.get("refusal_stage") == "evidence"
        and entry.get("source_lock_resolved")
        and (entry.get("qualified_substantive_chunks") or 0) > 0
    ]

    return {
        "stale_data_cleanup": {
            "removed_cases": len(real_diag.get("stale_positive_entries") or []),
            "removed_case_ids": [entry.get("id") or "" for entry in (real_diag.get("stale_positive_entries") or [])],
        },
        "chinese_source_lock_audit": {
            "mis_refusal_cases": len(chinese_entries),
            "all_blocked_before_source_lock_resolution": all(not entry.get("source_lock_resolved") for entry in chinese_entries),
            "all_at_source_layer": all(entry.get("diagnosis_layer") == "source_layer" for entry in chinese_entries),
            "detail_counts": dict(Counter(entry.get("diagnosis_detail") or "unknown" for entry in chinese_entries)),
            "case_ids": [entry.get("id") or "" for entry in chinese_entries],
        },
        "body_recall_audit": {
            "locked_mis_refusal_cases": len(locked_entries),
            "body_recall_status_counts": dict(body_recall_counts),
            "locked_case_ids": [entry.get("id") or "" for entry in locked_entries],
        },
        "evidence_gate_fix_scope": {
            "candidate_cases": len(evidence_gate_fix_candidates),
            "candidate_case_ids": [entry.get("id") or "" for entry in evidence_gate_fix_candidates],
            "candidate_reasons": dict(Counter(entry.get("refusal_reason") or "unknown" for entry in evidence_gate_fix_candidates)),
        },
    }


def main() -> None:
    chinese_report = _load_json(CHINESE_REPORT)
    real_baseline_report = _load_json(REAL_BASELINE_REPORT)
    hard_competitive_report = _load_json(HARD_COMPETITIVE_REPORT)

    chinese_diag = _diagnose_suite(chinese_report, "chinese_retrieval")
    real_diag = _diagnose_suite(real_baseline_report, "real_regulation_baseline")
    hard_diag = _diagnose_suite(hard_competitive_report, "hard_competitive")
    audits = _build_audits(chinese_diag, real_diag, hard_diag)

    combined_layer_counts = Counter()
    for suite_diag in (chinese_diag, real_diag, hard_diag):
        combined_layer_counts.update(suite_diag.get("mis_refusal_layer_counts") or {})

    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": {
            "chinese_retrieval_report": CHINESE_REPORT,
            "real_regulation_baseline_report": REAL_BASELINE_REPORT,
            "hard_competitive_report": HARD_COMPETITIVE_REPORT,
        },
        "summary": {
            "stale_positive_cases_removed": len(real_diag.get("stale_positive_entries") or []),
            "combined_mis_refusal_cases": len(chinese_diag.get("mis_refusal_entries") or []) + len(real_diag.get("mis_refusal_entries") or []) + len(hard_diag.get("mis_refusal_entries") or []),
            "combined_mis_refusal_layer_counts": dict(combined_layer_counts),
        },
        "audits": audits,
        "suites": {
            "chinese_retrieval": chinese_diag,
            "real_regulation_baseline": real_diag,
            "hard_competitive": hard_diag,
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps({"summary": output["summary"], "audits": output["audits"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()