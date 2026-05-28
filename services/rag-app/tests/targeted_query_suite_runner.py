import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib import request


ROOT = Path(__file__).resolve().parents[3]
TEST_PATH = ROOT / "test.json"
DEFAULT_REPORT_PATH = ROOT / "services" / "rag-app" / "uploads" / "targeted_query_suite_report.rerun.v6.json"
BASE_URL = os.getenv("TARGETED_SUITE_API", "http://127.0.0.1:8080")


def load_sections() -> Dict[str, List[str]]:
    text = TEST_PATH.read_text(encoding="utf-8")
    sections: Dict[str, List[str]] = {}
    current = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
            continue
        if current and not line.startswith("#"):
            sections[current].append(line)
    return sections


def call_query(query: str, user_id: str) -> Tuple[int, Dict[str, Any]]:
    payload = json.dumps(
        {"query": query, "user_id": user_id, "top_k": 10, "enable_rerank": True},
        ensure_ascii=False,
    ).encode("utf-8")
    req = request.Request(f"{BASE_URL}/query", data=payload, headers={"Content-Type": "application/json"})
    last_error = None
    for timeout in (45, 60, 75):
        try:
            started = time.time()
            with request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return round((time.time() - started) * 1000), data
        except Exception as exc:
            last_error = exc
            time.sleep(0.2)
    raise last_error


def is_answer_like(md: Dict[str, Any], data: Dict[str, Any]) -> bool:
    final_channel = md.get("final_channel")
    if final_channel in {"document_not_found", "document_ambiguous", "document_clarification", "refusal", "blocked"}:
        return False
    return bool(data.get("answer")) or (md.get("docs_returned") or 0) > 0


def has_multi_sources(data: Dict[str, Any], md: Dict[str, Any]) -> bool:
    sources = data.get("sources") or []
    target_sources = md.get("target_sources") or []
    unique = {src.get("source") for src in sources if isinstance(src, dict) and src.get("source")}
    unique.update(target_sources)
    return len(unique) >= 2 or len(sources) >= 2 or (md.get("docs_returned") or 0) >= 2


def _compare_statuses(md: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [item for item in (md.get("compare_source_statuses") or []) if isinstance(item, dict)]


def compare_answer_ok(data: Dict[str, Any], md: Dict[str, Any]) -> bool:
    statuses = _compare_statuses(md)
    answer = data.get("answer") or ""
    if md.get("internal_route") != "multi_doc_compare":
        return False
    if md.get("final_channel") != "light_rag":
        return False
    if not has_multi_sources(data, md):
        return False
    if statuses and not all((item.get("status") == "answerable") for item in statuses):
        return False
    return bool(answer.strip()) and (
        any(marker in answer for marker in ["对比", "相比", "区别", "共同", "分别", "两份文档"]) or bool(statuses)
    )


def compare_safe_refusal_ok(md: Dict[str, Any]) -> bool:
    statuses = _compare_statuses(md)
    if md.get("internal_route") != "multi_doc_compare":
        return False
    if md.get("final_channel") != "refusal":
        return False
    if md.get("refusal_reason") != "compare_evidence_insufficient":
        return False
    if len(statuses) < 2:
        return False
    return any((item.get("status") in {"evidence_insufficient", "not_found"}) for item in statuses)


def evaluate(section: str, data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    md = data.get("metadata") or {}
    final_channel = md.get("final_channel")
    route = md.get("internal_route")
    qroute = md.get("query_route")
    lock_required = bool(md.get("source_lock_required"))
    lock_resolved = bool(md.get("source_lock_resolved"))
    reason = md.get("source_lock_reason")
    refusal_reason = md.get("refusal_reason")

    if section.startswith("1. alias_title_reference"):
        ok = lock_required and lock_resolved and route in {"explicit_regulation_reference", "exact_title_reference", "alias_title_reference"} and is_answer_like(md, data)
    elif section.startswith("2. business_topic_qa"):
        ok = route == "business_topic_qa" and not lock_required and is_answer_like(md, data)
    elif section.startswith("3. topic_like_title_not_found"):
        ok = final_channel == "document_not_found" and lock_required and not lock_resolved
    elif section.startswith("4. topic_like_title_to_business_topic"):
        ok = route == "business_topic_qa" and not lock_required and is_answer_like(md, data)
    elif section.startswith("5. document_required"):
        ok = final_channel == "document_clarification" and lock_required and not lock_resolved
    elif section.startswith("6. document_ambiguous"):
        ok = final_channel == "document_ambiguous" and lock_required and not lock_resolved and reason in {"document_ambiguous", "section_anchor_ambiguous"}
    elif section.startswith("7. multi_doc_query"):
        ok = not lock_required and is_answer_like(md, data) and has_multi_sources(data, md)
    elif section.startswith("8. multi_doc_compare"):
        ok = compare_answer_ok(data, md) or compare_safe_refusal_ok(md)
    elif section.startswith("9. evidence_insufficient_locked_source"):
        ok = lock_required and lock_resolved and final_channel == "refusal" and qroute == "evidence_insufficient" and refusal_reason in {"section_not_hit", "evidence_insufficient", "target_not_covered", "no_relevant_evidence"}
    elif section.startswith("10. invalid_query / garbage_query"):
        ok = qroute in {"invalid_query", "garbage_query", "low_information_query", "out_of_domain_query"} and final_channel == "blocked"
    else:
        ok = False

    md_view = {
        key: md.get(key)
        for key in [
            "internal_route",
            "final_channel",
            "source_lock_required",
            "source_lock_resolved",
            "source_lock_reason",
            "target_status",
            "query_route",
            "query_quality",
            "refusal_reason",
            "docs_returned",
            "control_status",
            "compare_source_statuses",
        ]
    }
    return ok, md_view


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full targeted query suite against /query.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="Output report path")
    parser.add_argument("--user-id", default="targeted_suite_rerun_v6", help="User id used for /query requests")
    args = parser.parse_args()

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    sections = load_sections()
    summary: Dict[str, Dict[str, int]] = {}
    failures: List[Dict[str, Any]] = []

    for section, queries in sections.items():
        passed = 0
        for idx, query in enumerate(queries, start=1):
            try:
                elapsed_ms, data = call_query(query, user_id=args.user_id)
                ok, md_view = evaluate(section, data)
                if ok:
                    passed += 1
                else:
                    failures.append(
                        {
                            "section": section,
                            "index": idx,
                            "query": query,
                            "reason": f"internal_route={md_view['internal_route']}, lock={md_view['source_lock_required']}/{md_view['source_lock_resolved']}, final_channel={md_view['final_channel']}",
                            "metadata": md_view,
                            "elapsed_ms": elapsed_ms,
                        }
                    )
            except Exception as exc:
                failures.append(
                    {
                        "section": section,
                        "index": idx,
                        "query": query,
                        "reason": f"exception: {type(exc).__name__}: {exc}",
                        "metadata": {},
                        "elapsed_ms": None,
                    }
                )
        summary[section] = {"total": len(queries), "passed": passed, "failed": len(queries) - passed}

    total_cases = sum(item["total"] for item in summary.values())
    total_passed = sum(item["passed"] for item in summary.values())
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "api_base": BASE_URL,
        "grading_profile": "targeted_query_suite_v2_compare_safe_refusal",
        "summary": summary,
        "total_cases": total_cases,
        "total_passed": total_passed,
        "pass_rate": round(total_passed / total_cases, 4) if total_cases else 0.0,
        "failure_count": len(failures),
        "failures": failures[:150],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report_path": str(report_path),
                "total_cases": total_cases,
                "total_passed": total_passed,
                "pass_rate": report["pass_rate"],
                "summary": summary,
                "failure_count": len(failures),
                "grading_profile": report["grading_profile"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()