from typing import Any, Dict, List, Optional

from app.core import evidence as evidence_core
from app.core import intent_classifier
from app.core.legal_intent import classify_query_intent_fallback, legal_intent_from_payload, normalize_legal_intent


def _compare_source_status_prompt_lines(source_statuses: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in source_statuses or []:
        source = str((item or {}).get("source") or "").strip()
        status = str((item or {}).get("status") or "").strip()
        title = str((item or {}).get("title") or "").strip()
        evidence_query = str((item or {}).get("evidence_query") or "").strip()
        parts = [part for part in [title or source, status, evidence_query] if part]
        if parts:
            lines.append("- " + " | ".join(parts))
    return "\n".join(lines)

def _build_compare_status_matrix_answer(
    source_statuses: List[Dict[str, Any]],
    compare_refs: List[Dict[str, Any]],
    compare_plan: Optional[Dict[str, Any]] = None,
) -> str:
    focus = str((compare_plan or {}).get("focus") or (compare_plan or {}).get("subject_zone") or "").strip()
    lines = [f"围绕{focus or '对比目标'}，各目标文档的证据状态如下："]
    for idx, item in enumerate(source_statuses or [], start=1):
        title = str((item or {}).get("title") or (item or {}).get("source") or f"文档{idx}").strip()
        status = str((item or {}).get("status") or "UNKNOWN").strip()
        evidence_query = str((item or {}).get("evidence_query") or "").strip()
        ref = f" [{idx}]" if idx <= len(compare_refs or []) else ""
        suffix = f"；检索主题：{evidence_query}" if evidence_query else ""
        lines.append(f"- {title}: {status}{suffix}{ref}")
    if not source_statuses:
        lines.append("- 未形成可比较的文档证据集合。")
    return "\n".join(lines)

def _classifier_question_type(payload: Dict[str, Any]) -> str:
    return intent_classifier.question_type(payload)

def _classifier_route(payload: Dict[str, Any]) -> str:
    return intent_classifier.route(payload)

def _classifier_action(payload: Dict[str, Any]) -> str:
    return intent_classifier.action(payload)

def _classifier_is_comparison(payload: Dict[str, Any]) -> Optional[bool]:
    return intent_classifier.is_comparison(payload)


def _query_legal_intent(query: str, llm_parse: Dict[str, Any], intent_classification: Dict[str, Any]) -> str:
    for payload in (
        llm_parse,
        intent_classification,
        (intent_classification or {}).get("agentic_router") if isinstance(intent_classification, dict) else {},
    ):
        if isinstance(payload, dict):
            intent = legal_intent_from_payload(payload)
            if intent:
                return intent
    return normalize_legal_intent(classify_query_intent_fallback(query)) or "其他"


def _source_resolution_is_hard_locked(source_resolution: Dict[str, Any], active_fnames: Optional[List[str]] = None) -> bool:
    source_resolution = source_resolution or {}
    if source_resolution_status(source_resolution) == "locked":
        return True
    if not bool(source_resolution.get("resolved")):
        return False
    if str(source_resolution.get("lock_mode") or "") == "hard_lock":
        return True
    try:
        if float(source_resolution.get("lock_confidence") or 0.0) >= 1.0:
            return True
    except Exception:
        pass
    return bool(active_fnames) and bool(source_resolution.get("sources") or source_resolution.get("target_sources"))


def source_resolution_status(source_resolution: Dict[str, Any]) -> str:
    source_resolution = source_resolution or {}
    status = str(source_resolution.get("status") or "").strip()
    if status in {"locked", "ambiguous", "not_found", "global_fallback"}:
        return status
    if bool(source_resolution.get("resolved")) and list(
        source_resolution.get("target_fnames")
        or source_resolution.get("sources")
        or source_resolution.get("target_sources")
        or []
    ):
        return "locked"
    if bool(source_resolution.get("required")):
        return "ambiguous" if list(source_resolution.get("candidates") or []) else "not_found"
    return "global_fallback"


def source_resolution_target_fnames(source_resolution: Dict[str, Any]) -> List[str]:
    source_resolution = source_resolution or {}
    values = (
        source_resolution.get("target_fnames")
        or source_resolution.get("sources")
        or source_resolution.get("target_sources")
        or []
    )
    return [str(value or "").strip() for value in values if str(value or "").strip()]


def source_resolution_global_fallback(source_resolution: Dict[str, Any]) -> bool:
    return source_resolution_status(source_resolution) == "global_fallback"


ROUTER_TARGET_FAILURE_REASONS = {
    "compare_target_not_found",
    "compare_targets_not_found",
    "agentic_router_targets_not_found",
}

DELAYED_CLARIFICATION_REASONS = {
    "document_target_required",
    "missing_document",
    "missing_source",
    "missing_context",
    "weak_title_reference",
    "compare_target_incomplete",
    "compare_target_not_found",
    "compare_targets_not_found",
    "agentic_router_targets_not_found",
    "compare_source_set_incomplete",
}

HARD_SOURCE_NOT_FOUND_REASONS = {
    "document_not_found",
    "explicit_document_not_found",
    "explicit_filename_not_found",
}


def should_degrade_router_target_failure(source_resolution: Dict[str, Any]) -> bool:
    source_resolution = source_resolution or {}
    trace = dict(source_resolution.get("source_resolution_trace") or {})
    reason = str(source_resolution.get("reason") or "")
    if reason not in ROUTER_TARGET_FAILURE_REASONS:
        return False
    if source_resolution_target_fnames(source_resolution):
        return False
    if list(source_resolution.get("candidates") or []):
        return False
    source_kind = str(source_resolution.get("source_lock_kind") or "")
    route = str(source_resolution.get("route") or "")
    return (
        source_kind in {"compare_lock", "agentic_compare_lock"}
        or route in {"compare_target_not_found", "compare_targets_not_found"}
        or bool(trace.get("agentic_router"))
    )


def degrade_router_target_failure_to_global_fallback(source_resolution: Dict[str, Any]) -> Dict[str, Any]:
    source_resolution = dict(source_resolution or {})
    if not should_degrade_router_target_failure(source_resolution):
        return source_resolution
    trace = dict(source_resolution.get("source_resolution_trace") or {})
    reason = str(source_resolution.get("reason") or "")
    return {
        **source_resolution,
        "route": "content_qa",
        "required": False,
        "resolved": False,
        "status": "global_fallback",
        "scope_mode": "global",
        "fallback_allowed": True,
        "forced_retrieval_allowed": True,
        "sources": [],
        "target_fnames": [],
        "target_doc_ids": [],
        "candidates": [],
        "clarification": "",
        "lock_mode": "none",
        "lock_confidence": 0.0,
        "confidence": 0.0,
        "source_resolution_trace": {
            **trace,
            "router_target_failure_global_fallback": True,
            "original_route": trace.get("original_route") or source_resolution.get("route") or "",
            "original_reason": trace.get("original_reason") or reason,
            "global_fallback_reason": "router_target_resolution_failed",
        },
    }


def source_resolution_router_target_failure_fallback(source_resolution: Dict[str, Any]) -> bool:
    source_resolution = source_resolution or {}
    trace = dict(source_resolution.get("source_resolution_trace") or {})
    if bool(trace.get("router_target_failure_global_fallback")):
        return True
    return source_resolution_status(source_resolution) == "global_fallback" and should_degrade_router_target_failure(source_resolution)


def source_resolution_delayed_global_fallback(source_resolution: Dict[str, Any]) -> bool:
    source_resolution = source_resolution or {}
    trace = dict(source_resolution.get("source_resolution_trace") or {})
    return bool(
        trace.get("delayed_clarification_global_fallback")
        or trace.get("router_target_failure_global_fallback")
        or trace.get("compare_source_incomplete_global_fallback")
    )


def should_delay_source_clarification(source_resolution: Dict[str, Any]) -> bool:
    source_resolution = source_resolution or {}
    if _source_resolution_is_hard_locked(source_resolution):
        return False
    if source_resolution_target_fnames(source_resolution):
        return False
    if list(source_resolution.get("candidates") or []):
        return False
    reason = str(source_resolution.get("reason") or "")
    if reason in HARD_SOURCE_NOT_FOUND_REASONS:
        return False
    status = source_resolution_status(source_resolution)
    if status not in {"ambiguous", "not_found", "global_fallback"}:
        return False
    if should_degrade_router_target_failure(source_resolution):
        return True
    return bool(source_resolution.get("required")) and not bool(source_resolution.get("resolved")) and (
        reason in DELAYED_CLARIFICATION_REASONS
        or str(source_resolution.get("route") or "") in {"document_clarification", "compare_clarification"}
    )


def degrade_unresolved_source_to_global_fallback(
    source_resolution: Dict[str, Any],
    reason: str = "delayed_clarification",
) -> Dict[str, Any]:
    source_resolution = degrade_router_target_failure_to_global_fallback(source_resolution)
    if source_resolution_delayed_global_fallback(source_resolution):
        return source_resolution
    if not should_delay_source_clarification(source_resolution):
        return source_resolution
    trace = dict(source_resolution.get("source_resolution_trace") or {})
    original_reason = str(source_resolution.get("reason") or "")
    return {
        **dict(source_resolution or {}),
        "route": "content_qa",
        "required": False,
        "resolved": False,
        "status": "global_fallback",
        "scope_mode": "global",
        "fallback_allowed": True,
        "forced_retrieval_allowed": True,
        "sources": [],
        "target_sources": [],
        "target_fnames": [],
        "target_doc_ids": [],
        "candidates": [],
        "clarification": "",
        "lock_mode": "none",
        "lock_confidence": 0.0,
        "confidence": 0.0,
        "source_lock_kind": source_resolution.get("source_lock_kind") or "delayed_clarification",
        "source_resolution_trace": {
            **trace,
            "delayed_clarification_global_fallback": True,
            "delayed_clarification_reason": reason,
            "original_status": trace.get("original_status") or source_resolution_status(source_resolution),
            "original_route": trace.get("original_route") or source_resolution.get("route") or "",
            "original_reason": trace.get("original_reason") or original_reason,
            "global_fallback_reason": "delayed_source_resolution",
        },
    }


def _hit_similarity_score(runtime: Any, hit: Any, score_mode: str) -> float:
    try:
        raw_score = float(runtime.evidence.hit_score(hit) or 0.0)
    except Exception:
        raw_score = 0.0
    if score_mode == "distance":
        return 1.0 / (1.0 + max(raw_score, 0.0))
    return raw_score


def _best_similarity_score(runtime: Any, docs: List[Any], score_mode: str) -> float:
    if not docs:
        return 0.0
    return max(_hit_similarity_score(runtime, doc, score_mode) for doc in docs or [])


def source_resolution_scope_mode(source_resolution: Dict[str, Any], active_fnames: Optional[List[str]] = None) -> str:
    source_resolution = source_resolution or {}
    scope_mode = str(source_resolution.get("scope_mode") or "").strip()
    if scope_mode in {"doc_locked", "multi_doc_locked", "ambiguous", "not_found", "global"}:
        return scope_mode
    status = source_resolution_status(source_resolution)
    target_fnames = source_resolution_target_fnames(source_resolution) or list(active_fnames or [])
    if status == "locked":
        return "multi_doc_locked" if len(target_fnames) > 1 else "doc_locked"
    if status == "ambiguous":
        return "ambiguous"
    if status == "not_found":
        return "not_found"
    return "global"


def source_resolution_state_fields(source_resolution: Dict[str, Any], active_fnames: Optional[List[str]] = None) -> Dict[str, Any]:
    source_resolution = source_resolution or {}
    status = source_resolution_status(source_resolution)
    target_fnames = source_resolution_target_fnames(source_resolution) or list(active_fnames or [])
    return {
        "source_resolution_status": status,
        "target_doc_ids": list(source_resolution.get("target_doc_ids") or []),
        "target_fnames": target_fnames,
        "source_resolution_evidence": list(source_resolution.get("evidence") or []),
        "source_resolution_reason": str(source_resolution.get("reason") or ""),
        "scope_mode": source_resolution_scope_mode(source_resolution, target_fnames),
        "fallback_allowed": bool(source_resolution.get("fallback_allowed")) if "fallback_allowed" in source_resolution else status == "global_fallback",
        "forced_retrieval_allowed": bool(source_resolution.get("forced_retrieval_allowed")) if "forced_retrieval_allowed" in source_resolution else status == "global_fallback",
    }


def _empty_recall_result(
    runtime: Any,
    query: str,
    user_id: str,
    qtype: str,
    llm_parse: Dict[str, Any],
    intent_classification: Dict[str, Any],
    source_resolution: Dict[str, Any],
    query_route: str,
    classifier_compare: Optional[bool],
    reason: str,
    quality: str = "valid",
    blocked_reason: str = "",
    intent_tier: str = "",
    classifier_action: str = "",
) -> Dict[str, Any]:
    retrieval_query = runtime.common.normalize_query(str(llm_parse.get("retrieval_query") or "")) or query
    dense_query = (
        runtime.common.normalize_query(str(llm_parse.get("dense_query") or ""))
        or runtime.common.normalize_query(str(llm_parse.get("retrieval_query") or ""))
        or query
    )
    search_database_tool_used = bool((intent_classification or {}).get("search_database_tool_used"))
    resolution_status = source_resolution_status(source_resolution)
    allow_soft_clarification = resolution_status == "global_fallback"
    return {
        "query": query,
        "retrieval_query": retrieval_query,
        "retrieval_query_raw": query,
        "dense_query": dense_query,
        "llm_parse": llm_parse,
        "intent_classification": intent_classification,
        "is_comparison": bool(classifier_compare) if classifier_compare is not None else bool(llm_parse.get("is_comparison")),
        "question_type": qtype,
        "score_mode": "score",
        "docs": [],
        "selected_docs": [],
        "qfilters": runtime.routing.query_filters(query),
        "recall_k": 0,
        "final_n": 0,
        "rerank_used": False,
        "query_route": query_route,
        "weak_query": runtime.routing.is_weak_reference_query(query),
        "early_filtered": [],
        "visibility_filtered": [],
        "dense_source_scores": {},
        "post_filter_docs": [],
        "retrieve_docs": [],
        "source_lock_required": False,
        "resolved_source_lock": False,
        "target_sources": [],
        "source_lock_candidates": list(source_resolution.get("candidates") or source_resolution.get("sources") or []),
        "source_lock_reason": reason,
        "clarification": source_resolution.get("clarification") or "",
        "target_text": source_resolution.get("target_text") or "",
        "lock_mode": "none",
        "lock_confidence": 0.0,
        "lock_message_prefix": "",
        "source_lock_kind": "intent_classifier" if intent_tier == "classifier_control" else "",
        "source_resolution_trace": dict(source_resolution.get("source_resolution_trace") or {}),
        **source_resolution_state_fields(source_resolution),
        "inherited_from_context": bool(source_resolution.get("inherited_from_context")),
        "search_database_tool_used": search_database_tool_used,
        "search_database_tool_empty": search_database_tool_used,
        "compare_subjects": list(source_resolution.get("compare_subjects") or []),
        "compare_doc_like_subjects": list(source_resolution.get("compare_doc_like_subjects") or []),
        "compare_missing_targets": list(source_resolution.get("compare_missing_targets") or []),
        "compare_common_aspects": list(source_resolution.get("compare_common_aspects") or []),
        "compare_topic_pair": list(source_resolution.get("compare_topic_pair") or []),
        "compare_canonical_aspects": list(source_resolution.get("compare_canonical_aspects") or []),
        "compare_expanded_aspects": list(source_resolution.get("compare_expanded_aspects") or []),
        "compare_source_subqueries": dict(source_resolution.get("compare_source_subqueries") or {}),
        "compare_status": source_resolution.get("compare_status") or "not_compare",
        "compare_plan": dict(source_resolution.get("compare_plan") or {}),
        "compare_source_results": [],
        "blocked_reason": blocked_reason,
        "query_quality": quality,
        "intent_tier": intent_tier,
        "soft_clarification_required": allow_soft_clarification and (
            search_database_tool_used
            or (
                intent_tier == "classifier_control"
                and not (classifier_action == "refuse" or query_route == "refusal")
            )
        ),
        "soft_clarification_reason": (
            "search_database_empty"
            if allow_soft_clarification and search_database_tool_used
            else ("intent_classifier" if allow_soft_clarification and intent_tier == "classifier_control" else "")
        ),
    }

def _soft_clarification_result(
    runtime: Any,
    query: str,
    user_id: str,
    qtype: str,
    llm_parse: Dict[str, Any],
    intent_classification: Dict[str, Any],
    source_resolution: Dict[str, Any],
    query_route: str,
    classifier_compare: Optional[bool],
    query_quality: Dict[str, Any],
    intent_tier: str,
    candidates: List[str],
    compare_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = _empty_recall_result(
        runtime,
        query,
        user_id,
        qtype,
        llm_parse,
        intent_classification,
        source_resolution,
        query_route,
        classifier_compare,
        reason=source_resolution.get("reason") or "not_needed",
        quality=str(query_quality.get("quality") or "valid"),
        blocked_reason="",
        intent_tier=intent_tier,
    )
    result.update(
        {
            "source_lock_candidates": list(candidates or []),
            "compare_status": source_resolution.get("compare_status")
            or dict(compare_plan or {}).get("compare_status")
            or "not_compare",
            "compare_plan": dict(compare_plan or source_resolution.get("compare_plan") or {}),
            "soft_clarification_required": True,
            "soft_clarification_reason": (
                "tier2_soft_confirm" if intent_tier == "tier_2" else "tier3_summary_clarification"
            ),
        }
    )
    return result



from typing import Any, Dict, List


def _compare_min_required_per_doc(runtime: Any) -> int:
    for getter in (
        lambda: getattr(getattr(runtime, "config", None), "COMPARE_MIN_REQUIRED_PER_DOC"),
        lambda: runtime.common.policy_get("compare.min_required_per_doc", 1),
    ):
        try:
            value = int(getter() or 1)
            return max(1, value)
        except Exception:
            continue
    return 1


def _compare_doc_groups(compare_source_results: List[Dict[str, Any]], field: str) -> List[Dict[str, Any]]:
    return [
        {
            "source": item.get("source") or "",
            "evidence_query": item.get("evidence_query") or "",
            "docs": list(item.get(field) or []),
            "score_mode": item.get("score_mode") or "score",
        }
        for item in compare_source_results or []
    ]


def build_compare_coverage_trace(
    compare_sources: List[str],
    compare_source_results: List[Dict[str, Any]],
    min_required_per_doc: int,
    missing_targets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    min_required = max(1, int(min_required_per_doc or 1))
    by_source = {
        str(item.get("source") or ""): item
        for item in compare_source_results or []
        if str(item.get("source") or "")
    }
    target_docs: List[Dict[str, Any]] = []
    for source in compare_sources or []:
        item = by_source.get(source) or {}
        retrieved = len(item.get("retrieve_docs") or item.get("post_filter_docs") or item.get("selected_docs") or item.get("docs") or [])
        selected = len(item.get("selected_docs") or [])
        if retrieved <= 0:
            coverage = "missing"
        elif selected < min_required:
            coverage = "insufficient"
        else:
            coverage = "covered"
        target_docs.append(
            {
                "doc_id": source,
                "source": source,
                "retrieved": retrieved,
                "selected": selected,
                "coverage": coverage,
            }
        )
    for target in missing_targets or []:
        label = str(target or "").strip()
        if not label:
            continue
        target_docs.append(
            {
                "doc_id": label,
                "source": label,
                "retrieved": 0,
                "selected": 0,
                "coverage": "missing",
            }
        )
    return {
        "min_required_per_doc": min_required,
        "target_docs": target_docs,
        "any_doc_missing": any(item["coverage"] == "missing" for item in target_docs),
        "any_doc_insufficient": any(item["coverage"] in {"missing", "insufficient"} for item in target_docs),
    }


def build_multi_doc_compare_result(
    runtime: Any,
    query: str,
    retrieval_query: str,
    retrieval_query_raw: str,
    dense_query: str,
    qtype: str,
    qfilters: Dict[str, Any],
    llm_parse: Dict[str, Any],
    intent_classification: Dict[str, Any],
    is_comparison: bool,
    query_route: str,
    source_resolution: Dict[str, Any],
    compare_plan: Dict[str, Any],
    compare_source_set: Dict[str, Any],
    compare_sources: List[str],
    compare_subqueries: Dict[str, Any],
    compare_source_results: List[Dict[str, Any]],
    requested_k: int,
    recall_k: int,
    final_n: int,
) -> Dict[str, Any]:
    compare_final_n = max(final_n * max(1, len(compare_sources)), final_n)
    min_required_per_doc = _compare_min_required_per_doc(runtime)
    compare_coverage = build_compare_coverage_trace(
        compare_sources,
        compare_source_results,
        min_required_per_doc,
        missing_targets=list(source_resolution.get("compare_missing_targets") or compare_source_set.get("missing_targets") or []),
    )
    docs_groups = _compare_doc_groups(compare_source_results, "docs")
    selected_groups = _compare_doc_groups(compare_source_results, "selected_docs")
    post_filter_groups = _compare_doc_groups(compare_source_results, "post_filter_docs")
    retrieve_groups = _compare_doc_groups(compare_source_results, "retrieve_docs")
    compare_stage_trace = [
        {
            "source": item.get("source") or "",
            "evidence_query": item.get("evidence_query") or "",
            "stage_trace": dict(item.get("stage_trace") or {}),
        }
        for item in (compare_source_results or [])
    ]
    return {
        "query": query,
        "retrieval_query": retrieval_query,
        "retrieval_query_raw": retrieval_query_raw,
        "dense_query": dense_query,
        "llm_parse": llm_parse,
        "intent_classification": intent_classification,
        "is_comparison": bool(is_comparison),
        "question_type": qtype,
        "score_mode": next((item.get("score_mode") for item in compare_source_results if item.get("score_mode")), "score"),
        "docs": runtime.evidence.merge_compare_source_doc_groups(docs_groups, per_source_limit=max(min_required_per_doc, requested_k)),
        "selected_docs": runtime.evidence.merge_compare_source_doc_groups(selected_groups, per_source_limit=max(min_required_per_doc, final_n)),
        "qfilters": qfilters,
        "recall_k": recall_k,
        "final_n": compare_final_n,
        "rerank_used": any(bool(item.get("rerank_used")) for item in compare_source_results),
        "query_route": query_route,
        "weak_query": runtime.routing.is_weak_reference_query(query),
        "early_filtered": [entry for item in compare_source_results for entry in (item.get("early_filtered") or [])],
        "visibility_filtered": [entry for item in compare_source_results for entry in (item.get("visibility_filtered") or [])],
        "dense_source_scores": {
            key: value
            for item in compare_source_results
            for key, value in (item.get("dense_source_scores") or {}).items()
        },
        "post_filter_docs": runtime.evidence.merge_compare_source_doc_groups(post_filter_groups, per_source_limit=max(min_required_per_doc, final_n)),
        "retrieve_docs": runtime.evidence.merge_compare_source_doc_groups(retrieve_groups, per_source_limit=max(min_required_per_doc, requested_k)),
        "source_lock_required": False,
        "resolved_source_lock": True,
        "target_sources": compare_sources,
        "source_lock_candidates": compare_sources,
        "source_lock_reason": source_resolution.get("reason") or "",
        "clarification": source_resolution.get("clarification") or "",
        "target_text": source_resolution.get("target_text") or "",
        "compare_subjects": list(source_resolution.get("compare_subjects") or []),
        "compare_doc_like_subjects": list(source_resolution.get("compare_doc_like_subjects") or []),
        "compare_missing_targets": list(source_resolution.get("compare_missing_targets") or []),
        "compare_common_aspects": list(source_resolution.get("compare_common_aspects") or []),
        "compare_topic_pair": list(source_resolution.get("compare_topic_pair") or []),
        "compare_canonical_aspects": list(source_resolution.get("compare_canonical_aspects") or []),
        "compare_expanded_aspects": list(source_resolution.get("compare_expanded_aspects") or []),
        "compare_source_subqueries": compare_subqueries,
        "compare_status": source_resolution.get("compare_status") or compare_plan.get("compare_status") or "plan_ready",
        "compare_plan": compare_plan,
        "compare_coverage": compare_coverage,
        "compare_source_set": {
            **dict(compare_source_set),
            "complete": bool(compare_source_set.get("complete")),
            "sources": compare_sources,
        },
        "compare_source_results": compare_source_results,
        "lock_mode": source_resolution.get("lock_mode") or "hard_lock",
        "lock_confidence": float(source_resolution.get("lock_confidence") or 1.0),
        "lock_message_prefix": source_resolution.get("lock_message_prefix") or "",
        "source_lock_kind": source_resolution.get("source_lock_kind") or "",
        "source_resolution_trace": {
            **dict(source_resolution.get("source_resolution_trace") or {}),
            "compare_coverage": compare_coverage,
            "compare_retrieval_stage_trace": compare_stage_trace,
        },
        **source_resolution_state_fields(source_resolution, compare_sources),
        "inherited_from_context": bool(source_resolution.get("inherited_from_context")),
    }

def build_empty_search_result(
    runtime: Any,
    query: str,
    retrieval_query: str,
    retrieval_query_raw: str,
    dense_query: str,
    qtype: str,
    qfilters: Dict[str, Any],
    llm_parse: Dict[str, Any],
    intent_classification: Dict[str, Any],
    is_comparison: bool,
    query_route: str,
    docs: List[Any],
    lex_items: List[Any],
    visible_dense: Dict[str, Any],
    visible_lex: Dict[str, Any],
    dense_source_scores: Dict[str, float],
    recall_k: int,
    final_n: int,
    source_resolution: Dict[str, Any],
    active_fnames: List[str],
) -> Dict[str, Any]:
    dropped = int(visible_dense.get("dropped") or 0) + int(visible_lex.get("dropped") or 0)
    search_database_tool_used = bool((intent_classification or {}).get("search_database_tool_used"))
    suppress_empty_clarification = _source_resolution_is_hard_locked(source_resolution, active_fnames)
    delayed_global_fallback = source_resolution_delayed_global_fallback(source_resolution)
    source_trace = dict(source_resolution.get("source_resolution_trace") or {})
    if delayed_global_fallback:
        source_trace = {
            **source_trace,
            "global_fallback_best_similarity": 0.0,
            "global_fallback_min_similarity": float(
                runtime.common.policy_get("source_resolution.global_fallback_min_similarity", 0.3) or 0.3
            ),
            "global_fallback_low_score": True,
        }
    return {
        "query": query,
        "retrieval_query": retrieval_query,
        "retrieval_query_raw": retrieval_query_raw,
        "dense_query": dense_query,
        "llm_parse": llm_parse,
        "intent_classification": intent_classification,
        "is_comparison": bool(is_comparison),
        "evidence_query": retrieval_query,
        "question_type": qtype,
        "score_mode": "score",
        "docs": [],
        "dense_hits": docs,
        "lexical_hits": lex_items,
        "selected_docs": [],
        "qfilters": qfilters,
        "recall_k": recall_k,
        "final_n": final_n,
        "rerank_used": False,
        "query_route": query_route,
        "weak_query": runtime.routing.is_weak_reference_query(query),
        "early_filtered": dropped,
        "visibility_filtered": dropped,
        "dense_source_scores": dense_source_scores,
        "dense_visible_states": dict(visible_dense.get("states") or {}),
        "lexical_visible_states": dict(visible_lex.get("states") or {}),
        "post_filter_docs": [],
        "retrieve_docs": [],
        "source_lock_required": bool(source_resolution.get("required")),
        "resolved_source_lock": bool(source_resolution.get("resolved")),
        "target_sources": list(active_fnames),
        "source_lock_candidates": list(source_resolution.get("candidates") or []),
        "source_lock_reason": source_resolution.get("reason") or "",
        "clarification": source_resolution.get("clarification") or "",
        "target_text": source_resolution.get("target_text") or "",
        "lock_mode": source_resolution.get("lock_mode") or "none",
        "lock_confidence": float(source_resolution.get("lock_confidence") or 0.0),
        "lock_message_prefix": source_resolution.get("lock_message_prefix") or "",
        "source_lock_kind": source_resolution.get("source_lock_kind") or "",
        "source_resolution_trace": source_trace,
        **source_resolution_state_fields(source_resolution, active_fnames),
        "inherited_from_context": bool(source_resolution.get("inherited_from_context")),
        "search_database_tool_used": search_database_tool_used,
        "search_database_tool_empty": search_database_tool_used,
        "soft_clarification_required": (search_database_tool_used or delayed_global_fallback) and not suppress_empty_clarification,
        "soft_clarification_reason": (
            "global_fallback_low_similarity"
            if delayed_global_fallback and not suppress_empty_clarification
            else ("search_database_empty" if search_database_tool_used and not suppress_empty_clarification else "")
        ),
    }

def build_dynamic_lock_clarification_result(
    runtime: Any,
    query: str,
    retrieval_query: str,
    retrieval_query_raw: str,
    dense_query: str,
    qtype: str,
    qfilters: Dict[str, Any],
    llm_parse: Dict[str, Any],
    intent_classification: Dict[str, Any],
    is_comparison: bool,
    docs: List[Any],
    visible_dense: Dict[str, Any],
    visible_lex: Dict[str, Any],
    selected_docs: List[Any],
    post_filter_docs: List[Any],
    retrieve_docs: List[Any],
    dense_source_scores: Dict[str, float],
    score_mode: str,
    reranked_chunk: Dict[str, Any],
    recall_k: int,
    final_n: int,
    weak_query: bool,
    dynamic_lock: Dict[str, Any],
    source_resolution: Dict[str, Any],
    compare_plan: Dict[str, Any],
    intent_tier: str,
) -> Dict[str, Any]:
    dropped = int(visible_dense.get("dropped") or 0) + int(visible_lex.get("dropped") or 0)
    stage_trace = dict((reranked_chunk or {}).get("stage_trace") or {})
    return {
        "query": query,
        "retrieval_query": retrieval_query,
        "retrieval_query_raw": retrieval_query_raw,
        "dense_query": dense_query,
        "llm_parse": llm_parse,
        "intent_classification": intent_classification,
        "is_comparison": bool(is_comparison),
        "evidence_query": retrieval_query,
        "question_type": qtype,
        "score_mode": score_mode,
        "docs": docs,
        "dense_hits": visible_dense["hits"],
        "lexical_hits": visible_lex["hits"],
        "selected_docs": selected_docs,
        "qfilters": qfilters,
        "recall_k": recall_k,
        "final_n": final_n,
        "rerank_used": bool(reranked_chunk["used"]),
        "query_route": "open_topic_probe",
        "weak_query": weak_query,
        "early_filtered": dropped,
        "visibility_filtered": dropped,
        "dense_source_scores": dense_source_scores,
        "dense_visible_states": dict(visible_dense.get("states") or {}),
        "lexical_visible_states": dict(visible_lex.get("states") or {}),
        "post_filter_docs": post_filter_docs,
        "retrieve_docs": retrieve_docs,
        "source_lock_required": False,
        "resolved_source_lock": False,
        "target_sources": [],
        "source_lock_candidates": list(dynamic_lock.get("sources") or []),
        "source_lock_reason": "open_topic_multi_source",
        "clarification": runtime.source.clarification_prompt(list(dynamic_lock.get("sources") or [])),
        "target_text": "",
        "lock_mode": "none",
        "lock_confidence": 0.0,
        "lock_message_prefix": "",
        "source_lock_kind": "open_topic_multi_source",
        "source_resolution_trace": {
            "post_recall_dynamic_lock": dynamic_lock,
            **({"retrieval_stage_trace": stage_trace} if stage_trace else {}),
        },
        "inherited_from_context": False,
        "compare_subjects": list(source_resolution.get("compare_subjects") or []),
        "compare_doc_like_subjects": list(source_resolution.get("compare_doc_like_subjects") or []),
        "compare_missing_targets": list(source_resolution.get("compare_missing_targets") or []),
        "compare_common_aspects": list(source_resolution.get("compare_common_aspects") or []),
        "compare_topic_pair": list(source_resolution.get("compare_topic_pair") or []),
        "compare_canonical_aspects": list(source_resolution.get("compare_canonical_aspects") or []),
        "compare_expanded_aspects": list(source_resolution.get("compare_expanded_aspects") or []),
        "compare_source_subqueries": dict(source_resolution.get("compare_source_subqueries") or compare_plan.get("source_subqueries") or {}),
        "compare_status": source_resolution.get("compare_status") or compare_plan.get("compare_status") or "not_compare",
        "compare_plan": compare_plan,
        "compare_source_results": [],
        "intent_tier": intent_tier,
        "soft_clarification_required": True,
        "soft_clarification_reason": "open_topic_multi_source",
    }

def build_lightweight_recall_result(
    runtime: Any,
    query: str,
    retrieval_query: str,
    retrieval_query_raw: str,
    dense_query: str,
    qtype: str,
    qfilters: Dict[str, Any],
    llm_parse: Dict[str, Any],
    intent_classification: Dict[str, Any],
    is_comparison: bool,
    query_route: str,
    docs: List[Any],
    visible_dense: Dict[str, Any],
    visible_lex: Dict[str, Any],
    selected_docs: List[Any],
    post_filter_docs: List[Any],
    retrieve_docs: List[Any],
    dense_source_scores: Dict[str, float],
    score_mode: str,
    reranked_chunk: Dict[str, Any],
    recall_k: int,
    final_n: int,
    weak_query: bool,
    source_resolution: Dict[str, Any],
    active_fnames: List[str],
    topical_multi_doc_mode: bool,
    compare_plan: Dict[str, Any],
    intent_tier: str,
) -> Dict[str, Any]:
    dropped = int(visible_dense.get("dropped") or 0) + int(visible_lex.get("dropped") or 0)
    search_database_tool_used = bool((intent_classification or {}).get("search_database_tool_used"))
    source_trace = dict(source_resolution.get("source_resolution_trace") or {})
    stage_trace = dict((reranked_chunk or {}).get("stage_trace") or {})
    if stage_trace:
        source_trace = {
            **source_trace,
            "retrieval_stage_trace": stage_trace,
        }
    delayed_global_fallback = source_resolution_delayed_global_fallback(source_resolution)
    fallback_min_similarity = float(
        runtime.common.policy_get("source_resolution.global_fallback_min_similarity", 0.3) or 0.3
    )
    fallback_score_docs = selected_docs or post_filter_docs or retrieve_docs or docs
    fallback_best_similarity = _best_similarity_score(runtime, fallback_score_docs, score_mode)
    fallback_low_score = delayed_global_fallback and fallback_best_similarity < fallback_min_similarity
    if delayed_global_fallback:
        source_trace = {
            **source_trace,
            "global_fallback_best_similarity": round(fallback_best_similarity, 4),
            "global_fallback_min_similarity": fallback_min_similarity,
            "global_fallback_low_score": fallback_low_score,
        }
    return {
        "query": query,
        "retrieval_query": retrieval_query,
        "retrieval_query_raw": retrieval_query_raw,
        "dense_query": dense_query,
        "llm_parse": llm_parse,
        "intent_classification": intent_classification,
        "is_comparison": bool(is_comparison),
        "evidence_query": retrieval_query,
        "question_type": qtype,
        "score_mode": score_mode,
        "docs": docs,
        "dense_hits": visible_dense["hits"],
        "lexical_hits": visible_lex["hits"],
        "selected_docs": selected_docs,
        "qfilters": qfilters,
        "recall_k": recall_k,
        "final_n": final_n,
        "rerank_used": bool(reranked_chunk["used"]),
        "query_route": query_route,
        "weak_query": weak_query,
        "early_filtered": dropped,
        "visibility_filtered": dropped,
        "dense_source_scores": dense_source_scores,
        "dense_visible_states": dict(visible_dense.get("states") or {}),
        "lexical_visible_states": dict(visible_lex.get("states") or {}),
        "post_filter_docs": post_filter_docs,
        "retrieve_docs": retrieve_docs,
        "source_lock_required": bool(source_resolution.get("required")),
        "resolved_source_lock": bool(source_resolution.get("resolved") or (bool(active_fnames) and not topical_multi_doc_mode)),
        "target_sources": list(active_fnames),
        "source_lock_candidates": list(source_resolution.get("candidates") or []),
        "source_lock_reason": source_resolution.get("reason") or "",
        "clarification": source_resolution.get("clarification") or "",
        "target_text": source_resolution.get("target_text") or "",
        "lock_mode": source_resolution.get("lock_mode") or ("hard_lock" if active_fnames else "none"),
        "lock_confidence": float(source_resolution.get("lock_confidence") or (1.0 if active_fnames else 0.0)),
        "lock_message_prefix": source_resolution.get("lock_message_prefix") or "",
        "source_lock_kind": source_resolution.get("source_lock_kind") or "",
        "source_resolution_trace": source_trace,
        **source_resolution_state_fields(source_resolution, active_fnames),
        "inherited_from_context": bool(source_resolution.get("inherited_from_context")),
        "compare_subjects": list(source_resolution.get("compare_subjects") or []),
        "compare_doc_like_subjects": list(source_resolution.get("compare_doc_like_subjects") or []),
        "compare_missing_targets": list(source_resolution.get("compare_missing_targets") or []),
        "compare_common_aspects": list(source_resolution.get("compare_common_aspects") or []),
        "compare_topic_pair": list(source_resolution.get("compare_topic_pair") or []),
        "compare_canonical_aspects": list(source_resolution.get("compare_canonical_aspects") or []),
        "compare_expanded_aspects": list(source_resolution.get("compare_expanded_aspects") or []),
        "compare_source_subqueries": dict(source_resolution.get("compare_source_subqueries") or compare_plan.get("source_subqueries") or {}),
        "compare_status": source_resolution.get("compare_status") or compare_plan.get("compare_status") or "not_compare",
        "compare_plan": compare_plan,
        "compare_source_results": [],
        "intent_tier": intent_tier,
        "search_database_tool_used": search_database_tool_used,
        "soft_clarification_required": bool(fallback_low_score),
        "soft_clarification_reason": "global_fallback_low_similarity" if fallback_low_score else "",
    }

def apply_post_recall_dynamic_lock(
    source_resolution: Dict[str, Any],
    active_fnames: List[str],
    dynamic_lock: Dict[str, Any],
) -> Dict[str, Any]:
    if dynamic_lock.get("action") != "lock":
        return {
            "source_resolution": source_resolution,
            "active_fnames": active_fnames,
        }

    locked_source = str(dynamic_lock.get("source") or "")
    if not locked_source:
        return {
            "source_resolution": source_resolution,
            "active_fnames": active_fnames,
        }

    return {
        "active_fnames": [locked_source],
        "source_resolution": {
            **dict(source_resolution),
            "required": False,
            "resolved": True,
            "status": "locked",
            "sources": [locked_source],
            "target_fnames": [locked_source],
            "candidates": [locked_source],
            "reason": "post_recall_dominant_source",
            "lock_mode": "implicit_lock",
            "lock_confidence": float(dynamic_lock.get("share") or 0.0),
            "confidence": float(dynamic_lock.get("share") or 0.0),
            "evidence": [f"source:{locked_source}", "reason:post_recall_dominant_source"],
            "source_lock_kind": "post_recall_dominant_source",
        },
    }



import asyncio
from typing import Any, Dict, List, Optional, Tuple

from app.core import retrieval as retrieval_core


def has_forced_retrieval_signal(runtime: Any, query: str) -> bool:
    text = str(query or "").strip()
    if not text:
        return False
    if retrieval_core.target_article_ids({}, text):
        return True
    if runtime.routing.extract_filename_candidates(text):
        return True
    if runtime.routing.extract_explicit_regulation_mentions(text):
        return True
    if runtime.routing.has_contextual_doc_reference(text):
        return True
    if runtime.retrieval.seed_anchor_terms_for_probe(text):
        return True
    if runtime.routing.strong_topic_terms(text):
        return True
    return False


def carry_query_article_ids(qfilters: Dict[str, Any], *queries: str) -> Dict[str, Any]:
    article_ids = retrieval_core.target_article_ids(qfilters, " ".join(str(query or "") for query in queries))
    if article_ids:
        qfilters["article_ids"] = article_ids
    return qfilters


def drop_invalid_parsed_article_filter(qfilters: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
    values = [
        (parsed or {}).get("article_id"),
        (parsed or {}).get("article_no"),
        (parsed or {}).get("article_ids"),
        (parsed or {}).get("target_article"),
        (parsed or {}).get("target_articles"),
    ]
    if any(value not in (None, "", [], (), {}) for value in values):
        if not retrieval_core.configured_article_ids_are_valid(*values):
            qfilters.pop("article_id", None)
            qfilters.pop("article_ids", None)
            qfilters.pop("target_article", None)
            qfilters.pop("target_articles", None)
            qfilters["_skip_article_id_filter"] = True
    return qfilters


def force_retrieval_source_resolution(source_resolution: Dict[str, Any], reason: str) -> Dict[str, Any]:
    status = source_resolution_status(source_resolution)
    forced_status = "locked" if status == "locked" else "global_fallback"
    return {
        **dict(source_resolution or {}),
        "required": False,
        "resolved": bool((source_resolution or {}).get("resolved")),
        "status": forced_status,
        "reason": reason,
        "clarification": "",
        "source_resolution_trace": {
            **dict((source_resolution or {}).get("source_resolution_trace") or {}),
            "forced_retrieval_fallback": True,
            "forced_retrieval_reason": reason,
        },
    }


def is_agentic_compare_resolution(source_resolution: Dict[str, Any]) -> bool:
    trace = dict((source_resolution or {}).get("source_resolution_trace") or {})
    return bool(trace.get("agentic_router")) or str((source_resolution or {}).get("source_lock_kind") or "") == "agentic_compare_lock"

def prepare_recall_source_context(
    runtime: Any,
    query: str,
    qtype: str,
    llm_parse: Dict[str, Any],
    intent_classification: Dict[str, Any],
    source_resolution: Dict[str, Any],
    query_route: str,
    classifier_compare: Optional[bool],
    query_quality: Dict[str, Any],
    intent_tier: str,
    filename_hints: Optional[List[str]] = None,
    user_id: str = "anonymous",
) -> Dict[str, Any]:
    source_resolution = degrade_unresolved_source_to_global_fallback(source_resolution, "pre_recall_source_context")
    resolution_status = source_resolution_status(source_resolution)
    forced_retrieval = (
        has_forced_retrieval_signal(runtime, query)
        and source_resolution_global_fallback(source_resolution)
        and not is_agentic_compare_resolution(source_resolution)
    )
    compare_plan = dict(source_resolution.get("compare_plan") or {})
    is_comparison_hint = (
        bool(classifier_compare)
        if classifier_compare is not None
        else (bool(compare_plan.get("is_compare")) or runtime.compare.has_intent(query))
    )
    clarification_limit = max(1, int(runtime.common.policy_get("source_resolution.clarification_examples_limit", 3)))
    open_topic_without_context = (
        query_route in {"business_topic_qa", "open_regulation_qa", "content_qa"}
        and (
            runtime.retrieval.seed_anchor_terms_for_probe(query)
            or runtime.routing.has_strong_business_signal(query)
            or runtime.routing.strong_topic_terms(query)
        )
        and not runtime.routing.has_contextual_doc_reference(query)
        and not runtime.routing.extract_explicit_regulation_mentions(query)
        and not runtime.routing.extract_filename_candidates(query)
    )
    soft_clarification_eligible = (
        intent_tier in {"tier_2", "tier_3"}
        and not bool(source_resolution.get("resolved"))
        and resolution_status == "global_fallback"
        and not source_resolution_delayed_global_fallback(source_resolution)
        and not is_comparison_hint
        and not open_topic_without_context
        and (
            not bool(source_resolution.get("required"))
            or (
                (source_resolution.get("route") or "") == "weak_title_reference"
                and not list(source_resolution.get("candidates") or [])
                and (source_resolution.get("reason") or "") == "document_target_required"
            )
        )
    )
    if soft_clarification_eligible and not forced_retrieval:
        soft_candidates = runtime.retrieval.clarification_candidates(
            query,
            seed_sources=list(source_resolution.get("candidates") or source_resolution.get("sources") or []),
            limit=clarification_limit,
        )
        if soft_candidates:
            return {
                "early_return": _soft_clarification_result(
                    runtime,
                    query,
                    user_id,
                    qtype,
                    llm_parse,
                    intent_classification,
                    source_resolution,
                    query_route,
                    classifier_compare,
                    query_quality,
                    intent_tier,
                    soft_candidates,
                    compare_plan=compare_plan,
                )
            }
    elif soft_clarification_eligible and forced_retrieval:
        source_resolution = force_retrieval_source_resolution(
            source_resolution,
            "forced_retrieval_soft_clarification_bypass",
        )
        query_route = "content_qa" if query_route in {"document_clarification", "compare_clarification"} else query_route

    if resolution_status == "locked":
        fnames = list(source_resolution_target_fnames(source_resolution) or filename_hints or [])
    elif is_comparison_hint and query_route == "multi_doc_compare":
        fnames = list(source_resolution.get("sources") or source_resolution.get("target_sources") or [])
    elif query_route == "multi_doc_query" and not bool(source_resolution.get("required")):
        fnames = list(source_resolution.get("sources") or [])
    else:
        fnames = []
    active_fnames = list(fnames)

    if is_comparison_hint:
        compare_source_set = runtime.compare.source_set_completeness(compare_plan, active_fnames)
    else:
        compare_source_set = {
            "complete": False,
            "expected_target_count": 0,
            "resolved_source_count": 0,
            "sources": [],
            "missing_targets": [],
        }
    topical_multi_doc_mode = (
        query_route == "multi_doc_query"
        and bool(active_fnames)
        and not bool(source_resolution.get("required"))
    )
    return {
        "early_return": None,
        "compare_plan": compare_plan,
        "is_comparison_hint": is_comparison_hint,
        "clarification_limit": clarification_limit,
        "source_resolution": source_resolution,
        "query_route": query_route,
        "fnames": fnames,
        "active_fnames": active_fnames,
        "compare_source_set": compare_source_set,
        "topical_multi_doc_mode": topical_multi_doc_mode,
    }

def handle_required_source_lock(
    runtime: Any,
    query: str,
    retrieval_query: str,
    qtype: str,
    llm_parse: Dict[str, Any],
    intent_classification: Dict[str, Any],
    source_resolution: Dict[str, Any],
    query_route: str,
    classifier_compare: Optional[bool],
    is_comparison_hint: bool,
    query_quality: Dict[str, Any],
    intent_tier: str,
    compare_plan: Dict[str, Any],
    clarification_limit: int,
    fnames: List[str],
    active_fnames: List[str],
    topical_multi_doc_mode: bool,
    user_id: str = "anonymous",
) -> Dict[str, Any]:
    source_resolution = degrade_unresolved_source_to_global_fallback(source_resolution, "required_source_lock")
    if source_resolution_global_fallback(source_resolution):
        query_route = "content_qa" if query_route in {
            "document_clarification",
            "compare_clarification",
            "compare_target_not_found",
            "compare_targets_not_found",
        } else query_route

    if not (source_resolution.get("required") and not source_resolution.get("resolved")):
        if source_resolution_status(source_resolution) == "locked":
            locked_fnames = source_resolution_target_fnames(source_resolution)
            if locked_fnames:
                fnames = locked_fnames
                active_fnames = locked_fnames
        return {
            "early_return": None,
            "source_resolution": source_resolution,
            "query_route": query_route,
            "fnames": fnames,
            "active_fnames": active_fnames,
            "topical_multi_doc_mode": topical_multi_doc_mode,
        }

    resolution_status = source_resolution_status(source_resolution)
    if resolution_status in {"ambiguous", "not_found"}:
        source_resolution = {
            **dict(source_resolution),
            "resolved": False,
            "sources": [],
            "target_fnames": [],
            "target_doc_ids": [],
            "lock_mode": "none",
            "lock_confidence": 0.0,
            "lock_message_prefix": "",
            "source_lock_kind": source_resolution.get("source_lock_kind") or "source_resolution_state",
            "source_resolution_trace": {
                **dict(source_resolution.get("source_resolution_trace") or {}),
                "hard_source_resolution_state": resolution_status,
            },
        }
        fnames = []
        active_fnames = []
    elif (source_resolution.get("reason") or "") == "document_ambiguous":
        candidates = [
            runtime.common.normalize_filename(x or "")
            for x in (source_resolution.get("candidates") or [])
            if runtime.common.normalize_filename(x or "")
        ]
        candidates = list(dict.fromkeys(candidates))
        canonical_candidates = runtime.source.collapse_by_canonical(candidates, limit=max(1, len(candidates)))
        allow_ambiguous_soft_lock = bool(canonical_candidates) and len(canonical_candidates) == 1
        chosen = canonical_candidates[0] if allow_ambiguous_soft_lock else ""
        if chosen:
            title = runtime.source.display_title(chosen) or chosen
            if len(candidates) <= 1:
                prefix = f"我理解你查询的是《{title}》。\n"
            else:
                prefix = f"候选文档较相近，我先按《{title}》回答。\n"
            ambiguous_confidence, ambiguous_trace = runtime.retrieval.soft_lock_confidence(
                query,
                chosen,
                candidates,
                raw_title_score=6.2 if len(candidates) <= 1 else 5.6,
                top_competitors=[],
            )
            source_resolution = {
                **dict(source_resolution),
                "resolved": True,
                "sources": [chosen],
                "reason": "document_ambiguous_soft_lock",
                "lock_mode": "soft_lock",
                "lock_confidence": ambiguous_confidence,
                "lock_message_prefix": prefix,
                "source_lock_kind": "ambiguous_soft_lock",
                "source_resolution_trace": {
                    **dict(source_resolution.get("source_resolution_trace") or {}),
                    "original_reason": "document_ambiguous",
                    "ambiguous_soft_lock": True,
                    "ambiguous_candidates": candidates[:5],
                    "chosen_source": chosen,
                    **ambiguous_trace,
                },
            }
            fnames = [chosen]
            active_fnames = [chosen]
            query_route = source_resolution.get("route") or query_route or "content_qa"
            topical_multi_doc_mode = False
        else:
            source_resolution = {
                **dict(source_resolution),
                "resolved": False,
                "sources": [],
                "lock_mode": "none",
                "lock_confidence": 0.0,
                "lock_message_prefix": "",
                "source_lock_kind": "candidate_clarification",
                "source_resolution_trace": {
                    **dict(source_resolution.get("source_resolution_trace") or {}),
                    "ambiguous_soft_lock_blocked": True,
                    "ambiguous_candidates": candidates[:5],
                    "canonical_candidates": canonical_candidates[:5],
                    "blocked_reason": "multiple_distinct_candidates_require_clarification",
                },
            }

    if source_resolution.get("resolved"):
        return {
            "early_return": None,
            "source_resolution": source_resolution,
            "query_route": query_route,
            "fnames": fnames,
            "active_fnames": active_fnames,
            "topical_multi_doc_mode": topical_multi_doc_mode,
        }

    fallback_intent_tier = intent_tier or (
        "tier_2"
        if runtime.routing.has_strong_business_signal(query) or runtime.routing.strong_topic_terms(query)
        else ("tier_3" if runtime.routing.has_weak_business_signal(query) else "")
    )
    if (
        fallback_intent_tier in {"tier_2", "tier_3"}
        and source_resolution_status(source_resolution) == "global_fallback"
        and not list(source_resolution.get("candidates") or [])
        and (source_resolution.get("reason") or "") == "document_target_required"
    ):
        soft_candidates = runtime.retrieval.clarification_candidates(
            query,
            seed_sources=list(source_resolution.get("sources") or []),
            limit=clarification_limit,
        )
        if soft_candidates:
            return {
                "early_return": _soft_clarification_result(
                    runtime,
                    query,
                    user_id,
                    qtype,
                    llm_parse,
                    intent_classification,
                    source_resolution,
                    query_route,
                    bool(is_comparison_hint),
                    query_quality,
                    fallback_intent_tier,
                    soft_candidates,
                    compare_plan=compare_plan,
                )
            }

    early = _empty_recall_result(
        runtime,
        query,
        user_id,
        qtype,
        llm_parse,
        intent_classification,
        source_resolution,
        query_route,
        bool(is_comparison_hint),
        reason=source_resolution.get("reason") or "document_target_required",
        quality=str(query_quality.get("quality") or "valid"),
        intent_tier=fallback_intent_tier,
    )
    early.update(
        {
            "retrieval_query": retrieval_query,
            "source_lock_required": True,
            "source_lock_candidates": list(source_resolution.get("candidates") or []),
            "compare_status": source_resolution.get("compare_status")
            or compare_plan.get("compare_status")
            or "not_compare",
            "compare_plan": compare_plan,
            "lock_mode": source_resolution.get("lock_mode") or "none",
            "lock_confidence": float(source_resolution.get("lock_confidence") or 0.0),
            "lock_message_prefix": source_resolution.get("lock_message_prefix") or "",
            "source_lock_kind": source_resolution.get("source_lock_kind") or "",
            **source_resolution_state_fields(source_resolution, active_fnames),
        }
    )
    return {"early_return": early}

def build_compare_source_incomplete_result(
    runtime: Any,
    query: str,
    retrieval_query: str,
    retrieval_query_raw: str,
    dense_query: str,
    qtype: str,
    qfilters: Dict[str, Any],
    llm_parse: Dict[str, Any],
    intent_classification: Dict[str, Any],
    source_resolution: Dict[str, Any],
    compare_plan: Dict[str, Any],
    compare_source_set: Dict[str, Any],
    query_quality: Dict[str, Any],
    intent_tier: str,
) -> Dict[str, Any]:
    return {
        "query": query,
        "retrieval_query": retrieval_query,
        "retrieval_query_raw": retrieval_query_raw,
        "dense_query": dense_query,
        "llm_parse": llm_parse,
        "intent_classification": intent_classification,
        "is_comparison": True,
        "question_type": qtype,
        "score_mode": "score",
        "docs": [],
        "selected_docs": [],
        "qfilters": qfilters,
        "recall_k": 0,
        "final_n": 0,
        "rerank_used": False,
        "query_route": "compare_clarification",
        "weak_query": runtime.routing.is_weak_reference_query(query),
        "early_filtered": [],
        "visibility_filtered": [],
        "dense_source_scores": {},
        "post_filter_docs": [],
        "retrieve_docs": [],
        "source_lock_required": True,
        "resolved_source_lock": False,
        "target_sources": list(compare_source_set.get("sources") or []),
        "source_lock_candidates": list(compare_source_set.get("sources") or source_resolution.get("candidates") or []),
        "source_lock_reason": "compare_source_set_incomplete",
        "clarification": source_resolution.get("clarification") or runtime.compare.clarification_prompt(
            list(source_resolution.get("compare_subjects") or []),
            list(compare_source_set.get("sources") or []),
        ),
        "target_text": source_resolution.get("target_text") or "",
        "compare_subjects": list(source_resolution.get("compare_subjects") or []),
        "compare_doc_like_subjects": list(source_resolution.get("compare_doc_like_subjects") or []),
        "compare_missing_targets": list(source_resolution.get("compare_missing_targets") or compare_source_set.get("missing_targets") or []),
        "compare_common_aspects": list(source_resolution.get("compare_common_aspects") or []),
        "compare_topic_pair": list(source_resolution.get("compare_topic_pair") or []),
        "compare_canonical_aspects": list(source_resolution.get("compare_canonical_aspects") or []),
        "compare_expanded_aspects": list(source_resolution.get("compare_expanded_aspects") or []),
        "compare_source_subqueries": dict(source_resolution.get("compare_source_subqueries") or {}),
        "compare_status": "source_set_incomplete",
        "compare_plan": compare_plan,
        "compare_source_set": compare_source_set,
        "compare_source_results": [],
        "lock_mode": "none",
        "lock_confidence": 0.0,
        "lock_message_prefix": "",
        "source_lock_kind": "compare_lock",
        "source_resolution_trace": {
            **dict(source_resolution.get("source_resolution_trace") or {}),
            "compare_source_set_complete": False,
            "compare_source_set": compare_source_set,
        },
        "inherited_from_context": bool(source_resolution.get("inherited_from_context")),
        "query_quality": query_quality["quality"],
        "intent_tier": intent_tier,
    }

async def prepare_retrieval_query_context(
    runtime: Any,
    query: str,
    qtype: str,
    llm_parse: Dict[str, Any],
    intent_classification: Dict[str, Any],
    source_resolution: Dict[str, Any],
    query_route: str,
    classifier_compare: Optional[bool],
    is_comparison_hint: bool,
    query_quality: Dict[str, Any],
    intent_tier: str,
    compare_plan: Dict[str, Any],
    compare_source_set: Dict[str, Any],
    clarification_limit: int,
    fnames: List[str],
    active_fnames: List[str],
    topical_multi_doc_mode: bool,
    query_explicit_set: set,
) -> Dict[str, Any]:
    retrieval_query = query
    qfilters = runtime.routing.query_filters(query)
    retrieval_query_override = runtime.common.normalize_query(source_resolution.get("retrieval_query_override") or "")
    if retrieval_query_override:
        retrieval_query = retrieval_query_override
    elif source_resolution.get("strip_title_mentions") and active_fnames:
        if source_resolution.get("reason") == "explicit_filename_unique":
            stripped_query = runtime.retrieval.strip_filename_mentions(query, active_fnames)
        else:
            stripped_query = runtime.retrieval.strip_source_title_mentions(query, active_fnames)
        if stripped_query:
            retrieval_query = stripped_query
    elif fnames and query_explicit_set:
        mentioned_fnames = [
            runtime.common.normalize_filename(name or "")
            for name in fnames
            if runtime.common.normalize_filename(name or "") in query_explicit_set
        ]
        if mentioned_fnames:
            stripped_query = runtime.retrieval.strip_filename_mentions(query, mentioned_fnames)
            if stripped_query:
                retrieval_query = stripped_query
    elif not active_fnames:
        title_sources = runtime.source.extract_title_candidates(query)
        if title_sources:
            stripped_query = runtime.retrieval.strip_source_title_mentions(query, title_sources)
            if stripped_query:
                retrieval_query = stripped_query
            qfilters["_candidate_hint_sources"] = list(
                dict.fromkeys(
                    runtime.common.normalize_filename(source or "")
                    for source in title_sources
                    if runtime.common.normalize_filename(source or "")
                )
            )[:5]
            qfilters["_soft_source_scope"] = True
            trace = dict(source_resolution.get("source_resolution_trace") or {})
            source_resolution = {
                **dict(source_resolution),
                "candidates": list(dict.fromkeys(list(source_resolution.get("candidates") or []) + list(title_sources or [])))[:5],
                "source_resolution_trace": {
                    **trace,
                    "title_candidates_as_hints": True,
                    "soft_source_scope": True,
                    "candidate_hint_sources": list(title_sources or [])[:5],
                },
            }

    if is_comparison_hint and source_resolution.get("compare_missing_targets"):
        stripped_query = runtime.retrieval.strip_raw_text_mentions(
            retrieval_query,
            list(source_resolution.get("compare_missing_targets") or []),
        )
        if stripped_query:
            retrieval_query = stripped_query

    is_comparison = False
    if getattr(runtime.config, "ENABLE_COMPARE_INTENT_TAG", True):
        is_comparison = (
            bool(classifier_compare)
            if classifier_compare is not None
            else (bool(compare_plan.get("is_compare")) or runtime.compare.has_intent(query))
        )
    agentic_router_used = bool((source_resolution.get("source_resolution_trace") or {}).get("agentic_router"))
    if is_comparison and not agentic_router_used:
        cleaned = runtime.compare.strip_noise_terms(retrieval_query)
        if cleaned:
            retrieval_query = cleaned

    retrieval_query_raw = retrieval_query
    dense_query = retrieval_query
    qfilters["_legal_intent"] = _query_legal_intent(query, llm_parse, intent_classification)
    qfilters = drop_invalid_parsed_article_filter(qfilters, llm_parse)
    qfilters = carry_query_article_ids(qfilters, query, retrieval_query, retrieval_query_raw)
    forced_retrieval = (
        has_forced_retrieval_signal(runtime, query)
        and source_resolution_global_fallback(source_resolution)
        and not is_agentic_compare_resolution(source_resolution)
    )
    if (
        is_comparison_hint
        and query_route == "multi_doc_compare"
        and not bool(compare_source_set.get("complete"))
        and not bool(compare_source_set.get("sources"))
        and not forced_retrieval
    ):
        trace = dict(source_resolution.get("source_resolution_trace") or {})
        source_resolution = {
            **dict(source_resolution or {}),
            "route": "content_qa",
            "required": False,
            "resolved": False,
            "status": "global_fallback",
            "scope_mode": "global",
            "fallback_allowed": True,
            "forced_retrieval_allowed": True,
            "sources": [],
            "target_sources": [],
            "target_fnames": [],
            "target_doc_ids": [],
            "candidates": [],
            "clarification": "",
            "reason": "compare_source_set_incomplete",
            "lock_mode": "none",
            "lock_confidence": 0.0,
            "confidence": 0.0,
            "source_lock_kind": source_resolution.get("source_lock_kind") or "compare_lock",
            "source_resolution_trace": {
                **trace,
                "compare_source_incomplete_global_fallback": True,
                "delayed_clarification_global_fallback": True,
                "delayed_clarification_reason": "compare_source_set_incomplete",
                "original_status": trace.get("original_status") or source_resolution_status(source_resolution),
                "original_route": trace.get("original_route") or source_resolution.get("route") or "",
                "original_reason": trace.get("original_reason") or source_resolution.get("reason") or "",
                "global_fallback_reason": "delayed_source_resolution",
                "compare_source_set_complete": False,
                "compare_source_set": compare_source_set,
            },
        }
        query_route = "content_qa"
        fnames = []
        active_fnames = []
    if forced_retrieval and query_route in {"document_clarification", "compare_clarification"}:
        query_route = "content_qa"
        source_resolution = force_retrieval_source_resolution(
            source_resolution,
            "forced_retrieval_query_context_bypass",
        )

    open_topic_without_context = (
        not active_fnames
        and query_route in {"business_topic_qa", "open_regulation_qa", "content_qa"}
        and (
            runtime.retrieval.seed_anchor_terms_for_probe(query)
            or runtime.routing.has_strong_business_signal(query)
            or runtime.routing.strong_topic_terms(query)
        )
        and not runtime.routing.has_contextual_doc_reference(query)
        and not runtime.routing.extract_explicit_regulation_mentions(query)
        and not runtime.routing.extract_filename_candidates(query)
    )
    if open_topic_without_context:
        open_topic_hint_sources = runtime.retrieval.clarification_candidates(
            query,
            seed_sources=list(source_resolution.get("candidates") or source_resolution.get("sources") or []),
            limit=clarification_limit,
        )
        open_topic_hint_sources = [
            runtime.common.normalize_filename(source or "")
            for source in open_topic_hint_sources
            if runtime.common.normalize_filename(source or "")
        ]
        open_topic_hint_sources = list(dict.fromkeys(open_topic_hint_sources))
        if len(open_topic_hint_sources) == 1:
            locked_source = open_topic_hint_sources[0]
            probe_confidence, probe_trace = runtime.retrieval.soft_lock_confidence(
                query,
                locked_source,
                open_topic_hint_sources,
                raw_title_score=8.4,
                top_competitors=[],
            )
            fnames = [locked_source]
            active_fnames = [locked_source]
            canonical_doc_id_fn = getattr(runtime.source, "canonical_doc_id", None)
            locked_doc_id = canonical_doc_id_fn(locked_source) if callable(canonical_doc_id_fn) else locked_source
            source_resolution = {
                **dict(source_resolution),
                "required": False,
                "resolved": True,
                "status": "locked",
                "sources": [locked_source],
                "target_fnames": [locked_source],
                "target_doc_ids": [locked_doc_id or locked_source],
                "candidates": open_topic_hint_sources,
                "reason": "single_probe_candidate_lock",
                "lock_mode": "implicit_lock",
                "lock_confidence": probe_confidence,
                "confidence": probe_confidence,
                "evidence": [f"source:{locked_source}", "reason:single_probe_candidate_lock"],
                "source_lock_kind": "single_probe_candidate_lock",
                "source_resolution_trace": {
                    **dict(source_resolution.get("source_resolution_trace") or {}),
                    "single_probe_candidate_lock": True,
                    "candidate_sources": open_topic_hint_sources,
                    **probe_trace,
                },
            }
        elif open_topic_hint_sources:
            qfilters["_candidate_hint_sources"] = open_topic_hint_sources[:5]
            qfilters["_soft_source_scope"] = True

    locked_llm_parse: Dict[str, Any] = {}
    if (
        bool(getattr(runtime.config, "ENABLE_LLM_QUERY_PARSE", True))
        and len(active_fnames) == 1
        and bool(source_resolution.get("resolved") or (active_fnames and not topical_multi_doc_mode))
    ):
        locked_source = runtime.common.normalize_filename(active_fnames[0] or "")
        locked_title = runtime.source.display_title(locked_source)
        if locked_title:
            try:
                parsed = await asyncio.to_thread(runtime.llm.parse_query_cached, query, locked_title)
                if isinstance(parsed, dict) and parsed:
                    locked_llm_parse = parsed
                    merged_parse = dict(llm_parse or {})
                    for key, value in parsed.items():
                        if isinstance(value, list):
                            if value:
                                merged_parse[key] = value
                            continue
                        if isinstance(value, bool):
                            if value or key not in merged_parse:
                                merged_parse[key] = value
                            continue
                        normalized_value = runtime.common.normalize_query(str(value or ""))
                        if normalized_value:
                            merged_parse[key] = value
                    llm_parse = merged_parse
            except Exception:
                locked_llm_parse = {}

    if llm_parse:
        rq = runtime.common.normalize_query(str(llm_parse.get("retrieval_query") or ""))
        dq = runtime.common.normalize_query(str(llm_parse.get("dense_query") or "")) or rq
        locked_rq = runtime.common.normalize_query(str(locked_llm_parse.get("retrieval_query") or ""))
        prefer_locked_llm_query = bool(locked_rq)
        if rq and (prefer_locked_llm_query or not retrieval_query_override):
            retrieval_query = rq
        if dq and (prefer_locked_llm_query or not retrieval_query_override):
            dense_query = dq
        anchors = llm_parse.get("anchors")
        if isinstance(anchors, list) and anchors:
            qfilters["_llm_anchor_override"] = list(anchors)[: int(getattr(runtime.config, "QUERY_PARSE_MAX_ANCHORS", 1))]
        aspects = llm_parse.get("aspects")
        if isinstance(aspects, list) and aspects:
            qfilters["_llm_aspects_override"] = list(aspects)[: int(getattr(runtime.config, "QUERY_PARSE_MAX_ASPECTS", 4))]
        section_targets = llm_parse.get("section_targets")
        if isinstance(section_targets, list) and section_targets:
            qfilters["_llm_section_targets_override"] = list(section_targets)[: int(getattr(runtime.config, "QUERY_PARSE_MAX_SECTION_TARGETS", 4))]
        qfilters = drop_invalid_parsed_article_filter(qfilters, llm_parse)
        qfilters = carry_query_article_ids(
            qfilters,
            query,
            retrieval_query,
            retrieval_query_raw,
            dense_query,
            str(llm_parse.get("retrieval_query") or ""),
            str(llm_parse.get("evidence_query") or ""),
        )

    if len(active_fnames) == 1 and bool(source_resolution.get("resolved")):
        locked_source = runtime.common.normalize_filename(active_fnames[0] or "")
        locked_title = runtime.source.display_title(locked_source)
        locked_sources = [locked_source]
        purified_retrieval_query = runtime.retrieval.purify_locked_source_query(retrieval_query, locked_sources)
        if purified_retrieval_query:
            retrieval_query = purified_retrieval_query
        purified_dense_query = runtime.retrieval.purify_locked_source_query(dense_query, locked_sources)
        if purified_dense_query:
            dense_query = purified_dense_query
        if runtime.retrieval.has_doc_noise(
            retrieval_query,
            locked_title=locked_title,
            locked_sources=locked_sources,
        ):
            purified_seed = retrieval_query
            stripped_filename_query = runtime.retrieval.strip_filename_mentions(purified_seed, locked_sources)
            if stripped_filename_query:
                purified_seed = stripped_filename_query
            stripped_title_query = runtime.retrieval.strip_source_title_mentions(purified_seed, locked_sources)
            if stripped_title_query:
                purified_seed = stripped_title_query
            purified_seed = runtime.retrieval.purify_shallow(purified_seed) or purified_seed
            llm_purified_query = await asyncio.to_thread(
                runtime.llm.purify_retrieval_query,
                query,
                purified_seed,
                locked_title,
            )
            purified_query = runtime.retrieval.purify_shallow(llm_purified_query or purified_seed) or purified_seed
            if purified_query:
                retrieval_query = purified_query
                if not runtime.common.normalize_query(str(locked_llm_parse.get("dense_query") or "")):
                    dense_query = purified_query

    if query_route in {"business_topic_qa", "open_regulation_qa", "content_qa"}:
        expanded_retrieval_query, corpus_expanded_terms = runtime.retrieval.expand_from_corpus(query, retrieval_query)
        if corpus_expanded_terms:
            retrieval_query = expanded_retrieval_query
            dense_query, _ = runtime.retrieval.expand_from_corpus(query, dense_query)
            qfilters["_llm_anchor_extra"] = list(
                dict.fromkeys(list(qfilters.get("_llm_anchor_extra") or []) + corpus_expanded_terms[:4])
            )
            qfilters["_llm_aspects_extra"] = list(
                dict.fromkeys(list(qfilters.get("_llm_aspects_extra") or []) + corpus_expanded_terms[:8])
            )
            qfilters["_corpus_expanded_terms"] = corpus_expanded_terms

    if len(runtime.common.normalize_query(retrieval_query)) < max(2, int(getattr(runtime.config, "MIN_QUERY_CHARS", 2))):
        retrieval_query = retrieval_query_raw

    if is_comparison:
        cleaned_dense = runtime.compare.strip_noise_terms(dense_query)
        if cleaned_dense:
            dense_query = cleaned_dense

    return {
        "early_return": None,
        "retrieval_query": retrieval_query,
        "retrieval_query_raw": retrieval_query_raw,
        "dense_query": dense_query,
        "qfilters": qfilters,
        "llm_parse": llm_parse,
        "source_resolution": source_resolution,
        "fnames": fnames,
        "active_fnames": active_fnames,
        "is_comparison": is_comparison,
    }

def compute_recall_window(
    config: Any,
    top_k: int,
    enable_rerank: bool,
    active_fnames: List[str],
) -> Dict[str, int]:
    requested_k = int(top_k or 10)
    recall_k = min(
        max(requested_k * 2, 20),
        min(config.TOP_K, int(getattr(config, "RETRIEVAL_CANDIDATE_K", config.RECALL_TOP_K))),
    )
    final_n = min(
        max(config.FINAL_CONTEXT_N, 3),
        max(3, int(getattr(config, "FINAL_CONTEXT_N_MAX", 10))),
    )
    pool_n = min(max(max(config.RERANK_KEEP_N, config.CHUNK_RERANK_KEEP_N), requested_k * 2), recall_k)
    if (
        enable_rerank
        and bool(getattr(config, "ENABLE_CHUNK_RERANK", False))
        and bool(getattr(config, "ENABLE_RERANK", True))
    ):
        target_pool = int(getattr(config, "CHUNK_RERANK_POOL_N", 60))
        recall_k = min(
            max(recall_k, target_pool),
            min(int(config.TOP_K), int(getattr(config, "RETRIEVAL_CANDIDATE_K", recall_k))),
        )
        pool_n = min(max(pool_n, min(recall_k, target_pool)), recall_k)

    if len(active_fnames) == 1:
        recall_k = min(max(recall_k, int(getattr(config, "LOCKED_DOC_RECALL_K", 60))), int(config.TOP_K))
        pool_n = min(max(pool_n, min(recall_k, requested_k * 3)), recall_k)

    return {
        "requested_k": requested_k,
        "recall_k": recall_k,
        "final_n": final_n,
        "pool_n": pool_n,
    }



from typing import Any, Dict, List, Optional


def prepare_retrieve_query(runtime: Any, query: str, user_id: str) -> Dict[str, Any]:
    query = runtime.common.normalize_query(query)
    fnames = runtime.routing.extract_filename_candidates(query)
    if len(query) < runtime.config.MIN_QUERY_CHARS:
        return {
            "query": query,
            "fnames": fnames,
            "early_return": {
                "documents": [],
                "sources": [],
                "metadata": runtime.control.metadata(
                    query=query,
                    user_id=user_id,
                    query_route="query_too_short",
                    internal_route="query_too_short",
                    final_channel="blocked",
                    blocked="query_too_short",
                    query_quality="invalid",
                ),
            },
        }
    if len(query) > runtime.config.MAX_QUERY_CHARS:
        return {
            "query": query,
            "fnames": fnames,
            "early_return": {
                "documents": [],
                "sources": [],
                "metadata": runtime.control.metadata(
                    query=query,
                    user_id=user_id,
                    query_route="query_too_long",
                    internal_route="query_too_long",
                    final_channel="blocked",
                    blocked="query_too_long",
                    query_quality="invalid",
                ),
            },
        }
    blocked = runtime.guardrails.blocked_reason(query)
    if blocked:
        return {
            "query": query,
            "fnames": fnames,
            "early_return": {
                "documents": [],
                "sources": [],
                "metadata": runtime.control.metadata(
                    query=query,
                    user_id=user_id,
                    query_route=blocked,
                    internal_route=blocked,
                    final_channel="blocked",
                    blocked=blocked,
                    query_quality="invalid",
                ),
            },
        }
    query_quality = runtime.guardrails.static_quality_state(query)
    if query_quality["reason"]:
        return {
            "query": query,
            "fnames": fnames,
            "early_return": {
                "documents": [],
                "sources": [],
                "metadata": runtime.control.metadata(
                    query=query,
                    user_id=user_id,
                    query_route=query_quality["reason"],
                    internal_route=query_quality["reason"],
                    final_channel="blocked",
                    blocked=query_quality["reason"],
                    query_quality=query_quality["quality"],
                ),
            },
        }
    return {
        "query": query,
        "fnames": fnames,
        "early_return": None,
    }

def build_retrieve_recall_blocked_result(runtime: Any, query: str, user_id: str, recall: Dict[str, Any]) -> Dict[str, Any]:
    reason = recall.get("blocked_reason") or "low_information_query"
    return {
        "documents": [],
        "sources": [],
        "metadata": runtime.control.metadata(
            query=query,
            user_id=user_id,
            query_route=reason,
            internal_route=reason,
            final_channel="blocked",
            blocked=reason,
            query_quality=recall.get("query_quality") or "low_information",
            recall=recall,
        ),
    }

def build_retrieve_soft_clarification_result(
    runtime: Any,
    query: str,
    user_id: str,
    recall: Dict[str, Any],
    clarification: Dict[str, Any],
) -> Dict[str, Any]:
    reason = recall.get("soft_clarification_reason") or "document_clarification"
    return {
        "documents": [],
        "sources": [],
        "metadata": runtime.control.metadata(
            query=query,
            user_id=user_id,
            query_route="document_clarification",
            internal_route=recall.get("soft_clarification_reason") or recall.get("query_route") or "content_qa",
            final_channel="document_clarification",
            recall=recall,
            refusal_reason=reason,
            question_type=recall.get("question_type") or runtime.routing.classify_question_type(query),
            extra={
                "refused": reason,
                "clarification": clarification.get("message") or "",
                "candidate_sources": list(clarification.get("candidate_sources") or recall.get("source_lock_candidates") or []),
                "clarification_used_llm": bool(clarification.get("used_llm")),
                "intent_tier": recall.get("intent_tier") or "",
            },
        ),
    }

def build_retrieve_source_lock_result(runtime: Any, query: str, user_id: str, recall: Dict[str, Any]) -> Dict[str, Any]:
    source_lock_reason = recall.get("source_lock_reason") or "document_target_required"
    if recall.get("source_resolution_status") == "not_found" and source_lock_reason not in {
        "compare_target_not_found",
        "compare_targets_not_found",
    }:
        source_lock_reason = "document_not_found"
    if source_lock_reason in {"compare_target_not_found", "compare_targets_not_found", "compare_source_set_incomplete"}:
        is_incomplete = source_lock_reason == "compare_source_set_incomplete"
        message = (
            recall.get("clarification")
            or runtime.compare.clarification_prompt(
                list(recall.get("compare_subjects") or []),
                list(recall.get("source_lock_candidates") or recall.get("target_sources") or []),
            )
        ) if is_incomplete else runtime.compare.target_not_found_prompt(
            list(recall.get("compare_missing_targets") or []),
            list(recall.get("source_lock_candidates") or recall.get("target_sources") or []),
        )
        return {
            "documents": [],
            "sources": [],
            "metadata": runtime.control.metadata(
                query=query,
                user_id=user_id,
                query_route="compare_clarification" if is_incomplete else source_lock_reason,
                internal_route=recall.get("query_route") or source_lock_reason,
                final_channel="compare_clarification" if is_incomplete else "document_not_found",
                recall=recall,
                refusal_reason=source_lock_reason,
                question_type=recall.get("question_type") or runtime.routing.classify_question_type(query),
                extra={
                    "refused": source_lock_reason,
                    "target_text": recall.get("target_text") or "",
                    "message": message,
                },
            ),
        }

    if source_lock_reason == "document_not_found":
        return {
            "documents": [],
            "sources": [],
            "metadata": runtime.control.metadata(
                query=query,
                user_id=user_id,
                query_route="document_not_found",
                internal_route=recall.get("query_route") or "explicit_regulation_reference",
                final_channel="document_not_found",
                recall=recall,
                refusal_reason="document_not_found",
                question_type=recall.get("question_type") or runtime.routing.classify_question_type(query),
                extra={
                    "refused": "document_not_found",
                    "target_text": recall.get("target_text") or "",
                    "message": runtime.source.not_found_prompt(recall.get("target_text") or query),
                },
            ),
        }

    return {
        "documents": [],
        "sources": [],
        "metadata": runtime.control.metadata(
            query=query,
            user_id=user_id,
            query_route="document_ambiguous" if source_lock_reason == "document_ambiguous" else "document_clarification",
            internal_route=recall.get("query_route") or "weak_title_reference",
            final_channel="document_ambiguous" if source_lock_reason == "document_ambiguous" else "document_clarification",
            recall=recall,
            refusal_reason=source_lock_reason,
            question_type=recall.get("question_type") or runtime.routing.classify_question_type(query),
            extra={
                "refused": source_lock_reason,
                "clarification": recall.get("clarification") or runtime.source.clarification_prompt(recall.get("source_lock_candidates") or []),
                "candidate_sources": list(recall.get("source_lock_candidates") or []),
                "target_text": recall.get("target_text") or "",
            },
        ),
    }

async def prepare_retrieve_evidence_context(
    runtime: Any,
    query: str,
    recall: Dict[str, Any],
    fnames: List[str],
    top_k: int,
) -> Dict[str, Any]:
    def retrieve_output_docs(docs: List[Any]) -> List[Any]:
        non_heading = [
            doc
            for doc in (docs or [])
            if not evidence_core.is_heading_only_hit(doc, evidence_core.hit_metadata)
        ]
        return runtime.evidence.select_retrieve_docs(non_heading, top_k=top_k, default_n=recall["final_n"])

    resolved_targets = [
        runtime.common.normalize_filename(x)
        for x in (recall.get("target_sources") or fnames)
        if runtime.common.normalize_filename(x)
    ]
    if recall.get("query_route") == "multi_doc_compare" and recall.get("compare_source_results"):
        compare_retrieve_groups = []
        for item in recall.get("compare_source_results") or []:
            compare_retrieve_groups.append(
                {
                    "source": item.get("source") or "",
                    "evidence_query": item.get("evidence_query") or "",
                    "docs": retrieve_output_docs(
                        item.get("selected_docs") or item.get("post_filter_docs") or item.get("retrieve_docs") or [],
                    ),
                    "score_mode": item.get("score_mode") or recall["score_mode"],
                }
            )
        retrieve_docs = runtime.evidence.merge_compare_source_doc_groups(compare_retrieve_groups, per_source_limit=max(2, top_k))
        return {
            "resolved_targets": resolved_targets,
            "retrieve_docs": retrieve_docs,
            "observations": {
                "retrieve_gate_disabled": True,
                "retrieve_filter": "heading_only",
                "compare_coverage": dict(recall.get("compare_coverage") or {}),
            },
            "compare_retrieve_groups": compare_retrieve_groups,
        }

    candidate_docs = recall.get("selected_docs") or recall.get("post_filter_docs") or recall.get("retrieve_docs") or []
    retrieve_docs = retrieve_output_docs(candidate_docs)
    return {
        "resolved_targets": resolved_targets,
        "retrieve_docs": retrieve_docs,
        "observations": {
            "retrieve_gate_disabled": True,
            "retrieve_filter": "heading_only",
        },
        "compare_retrieve_groups": [],
    }

def retrieve_refusal_reason(
    recall: Dict[str, Any],
    observations: Dict[str, Any],
    retrieve_docs: Optional[List[Any]] = None,
) -> Optional[str]:
    if (
        str(observations.get("answer_scope") or "") == "partial"
        and bool(observations.get("source_lock_resolved") or recall.get("resolved_source_lock"))
        and recall.get("query_route") not in {"multi_doc_compare", "single_doc_compare"}
        and int(observations.get("qualified_substantive_chunks") or observations.get("evidence_docs") or len(retrieve_docs or [])) > 0
    ):
        return None
    if observations["answer_scope"] not in {"full", "guarded_full"} and not bool(observations.get("compare_degraded")):
        return observations["evidence_coverage_reason"]
    return None

def build_retrieve_evidence_refusal_result(
    runtime: Any,
    query: str,
    user_id: str,
    recall: Dict[str, Any],
    retrieve_docs: List[Any],
    observations: Dict[str, Any],
    refusal_reason: str,
) -> Dict[str, Any]:
    compare_route = recall.get("query_route") == "multi_doc_compare"
    return {
        "documents": [],
        "sources": [],
        "metadata": runtime.control.metadata(
            query=query,
            user_id=user_id,
            query_route="compare_rag" if compare_route else "evidence_insufficient",
            internal_route=recall.get("query_route") or "content_qa",
            final_channel="compare_rag" if compare_route else "refusal",
            recall=recall,
            refusal_reason=refusal_reason,
            question_type=recall["question_type"],
            docs_returned=len(retrieve_docs),
            extra={
                "refused": refusal_reason,
                "message": runtime.compare.evidence_failure_prompt(observations.get("compare_source_statuses") or [])
                if recall.get("query_route") == "multi_doc_compare"
                else runtime.guardrails.evidence_refusal_answer(query, refusal_reason, observations),
                "compare_status": observations.get("compare_status") or recall.get("compare_status") or "",
                "visibility_enforced": True,
                "visibility_filtered": recall["visibility_filtered"],
                **observations,
            },
        ),
    }

def build_retrieve_success_result(
    runtime: Any,
    query: str,
    user_id: str,
    recall: Dict[str, Any],
    retrieve_docs: List[Any],
    resolved_targets: List[str],
    observations: Dict[str, Any],
) -> Dict[str, Any]:
    display_docs = runtime.evidence.filter_display_sources(
        retrieve_docs,
        recall["score_mode"],
        recall["qfilters"],
        resolved_targets,
        recall["question_type"],
        max_sources=10,
        target_sources=resolved_targets,
        observations=observations,
    )
    sources = runtime.evidence.build_sources(display_docs if display_docs else retrieve_docs[:10], query, score_mode=recall["score_mode"])
    documents = []
    for d in retrieve_docs:
        documents.append(
            {
                "source": runtime.evidence.hit_entity_source(d) or "unknown",
                "score": runtime.evidence.hit_score(d),
                "text": runtime.evidence.build_excerpt(runtime.evidence.hit_display_text(d), query, 500),
                "content": runtime.evidence.build_excerpt(runtime.evidence.hit_display_text(d), query, 500),
                "metadata": runtime.evidence.hit_metadata(d),
                "chunk_range": runtime.evidence.hit_chunk_range(d),
            }
        )
    return {
        "documents": documents,
        "retrieved_contexts": documents,
        "sources": sources,
        "metadata": runtime.control.metadata(
            query=query,
            user_id=user_id,
            query_route="compare_rag" if recall.get("query_route") == "multi_doc_compare" else "light_rag",
            internal_route=recall.get("query_route") or "content_qa",
            final_channel="compare_rag" if recall.get("query_route") == "multi_doc_compare" else "light_rag",
            recall=recall,
            question_type=recall["question_type"],
            docs_returned=len(retrieve_docs),
            extra={
                "compare_status": observations.get("compare_status") or recall.get("compare_status") or "",
                "compare_matrix": [
                    {
                        "source": item.get("source") or "",
                        "title": item.get("title") or "",
                        "status": item.get("status") or "",
                        "presence_state": runtime.compare.matrix_presence_state(item.get("presence_state") or ""),
                        "evidence_query": item.get("evidence_query") or "",
                    }
                    for item in (observations.get("compare_source_statuses") or [])
                ],
                "compare_source_set": dict(recall.get("compare_source_set") or {}),
                "docs_recalled": recall["recall_k"],
                "docs_rerank_kept": len(recall["docs"]),
                "docs_final": len(retrieve_docs),
                "rerank_used": recall["rerank_used"],
                "weak_query_expansion": recall["weak_query"],
                "early_filtered": recall["early_filtered"],
                "visibility_enforced": True,
                "visibility_filtered": recall["visibility_filtered"],
                **observations,
            },
        ),
    }



STRONG_SOURCE_ROUTES = {
    "explicit_doc_reference",
    "explicit_regulation_reference",
    "exact_title_reference",
    "alias_title_reference",
    "version_switch",
}

OPEN_QA_ROUTES = {"content_qa", "business_topic_qa", "open_regulation_qa"}

