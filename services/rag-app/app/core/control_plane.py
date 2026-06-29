from typing import Any, Callable, Dict, List, Optional


STRONG_SOURCE_ROUTES = {
    "explicit_doc_reference",
    "explicit_regulation_reference",
    "exact_title_reference",
    "alias_title_reference",
    "weak_title_reference",
    "version_switch",
}


def should_allow_llm_fallback(
    query: str,
    query_route: str,
    refusal_reason: str,
    normalize_query: Callable[[str], str],
    query_has_lockable_doc_rescue: Callable[[str], bool],
) -> bool:
    q = normalize_query(query)
    if not q:
        return False
    if query_has_lockable_doc_rescue(q):
        return False
    if query_route in {
        "existence",
        "visibility_probe",
        "explicit_doc_reference",
        "explicit_regulation_reference",
        "exact_title_reference",
        "alias_title_reference",
    }:
        return False
    if refusal_reason:
        return False
    return True


def target_status(
    query_route: str,
    source_lock_required: bool,
    source_lock_resolved: bool,
    source_lock_reason: str,
    target_sources: List[str],
) -> str:
    if source_lock_required:
        if source_lock_resolved or target_sources:
            return "resolved"
        if source_lock_reason == "document_not_found":
            return "document_not_found"
        if source_lock_reason in {"document_ambiguous", "section_anchor_ambiguous"}:
            return "ambiguous"
        return "required_unresolved"
    if target_sources:
        return "resolved"
    if query_route in STRONG_SOURCE_ROUTES:
        return "document_required"
    return "open"


def control_status(
    final_channel: str,
    blocked: Optional[str],
    status: str,
    refusal_reason: Optional[str],
) -> str:
    if blocked:
        return blocked
    if final_channel == "document_not_found" or status == "document_not_found":
        return "document_not_found"
    if final_channel in {"document_ambiguous", "document_clarification"} or status in {"ambiguous", "required_unresolved"}:
        return "source_lock_failed"
    if refusal_reason:
        return "evidence_insufficient"
    if status == "resolved":
        return "source_locked"
    return "answerable"


def refusal_stage(
    blocked: Optional[str],
    source_lock_required: bool,
    source_lock_resolved: bool,
    refusal_reason: Optional[str],
) -> Optional[str]:
    if blocked:
        return "query_validation"
    if source_lock_required and not source_lock_resolved:
        return "source_lock"
    if refusal_reason:
        return "evidence"
    return None


def build_control_plane_metadata(
    *,
    query: str,
    user_id: str,
    final_channel: str,
    normalize_filename: Callable[[str], str],
    classify_query_scope: Callable[[str, List[str], Optional[str]], str],
    should_allow_llm_fallback_fn: Callable[[str, str, str], bool],
    should_use_doc_fallback: Callable[[str, List[str], Optional[str]], bool],
    query_route: Optional[str] = None,
    internal_route: Optional[str] = None,
    fnames: Optional[List[str]] = None,
    recall: Optional[Dict[str, Any]] = None,
    blocked: Optional[str] = None,
    refusal_reason: Optional[str] = None,
    query_quality: Optional[str] = None,
    docs_returned: Optional[int] = None,
    question_type: Optional[str] = None,
    answer_mode: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    recall = dict(recall or {})
    normalized_targets = [
        normalize_filename(item)
        for item in (recall.get("target_sources") or fnames or [])
        if normalize_filename(item)
    ]
    route_name = query_route or recall.get("query_route") or "content_qa"
    internal_name = internal_route or recall.get("query_route") or route_name
    source_lock_required = bool(recall.get("source_lock_required"))
    source_lock_resolved = bool(recall.get("resolved_source_lock") or (source_lock_required and normalized_targets))
    source_lock_reason = (recall.get("source_lock_reason") or "").strip()
    status = target_status(route_name, source_lock_required, source_lock_resolved, source_lock_reason, normalized_targets)
    effective_query_quality = query_quality or ("valid" if not blocked else "invalid")
    effective_refusal_reason = refusal_reason or blocked
    scope = classify_query_scope(query, normalized_targets or list(fnames or []), internal_name)
    llm_fallback_allowed = False
    if not blocked and not (source_lock_required and not source_lock_resolved):
        llm_fallback_allowed = should_allow_llm_fallback_fn(query, internal_name, refusal_reason or "")
    doc_fallback_enabled = False
    if not blocked and final_channel == "light_rag":
        doc_fallback_enabled = should_use_doc_fallback(query, normalized_targets, internal_name)

    metadata = {
        "query": query,
        "user_id": user_id,
        "query_route": route_name,
        "internal_route": internal_name,
        "final_channel": final_channel,
        "query_quality": effective_query_quality,
        "source_lock_required": source_lock_required,
        "source_lock_resolved": source_lock_resolved,
        "source_lock_reason": source_lock_reason,
        "target_status": status,
        "target_sources": normalized_targets,
        "doc_fallback_enabled": doc_fallback_enabled,
        "llm_fallback_allowed": llm_fallback_allowed,
        "scope": scope,
        "refusal_stage": refusal_stage(blocked, source_lock_required, source_lock_resolved, refusal_reason),
        "refusal_reason": effective_refusal_reason,
        "control_status": control_status(final_channel, blocked, status, refusal_reason),
        "control_plane": "light",
        "lock_mode": recall.get("lock_mode") or "",
        "lock_confidence": float(recall.get("lock_confidence") or 0.0),
        "lock_message_prefix": recall.get("lock_message_prefix") or "",
        "source_lock_kind": recall.get("source_lock_kind") or "",
        "source_resolution_trace": dict(recall.get("source_resolution_trace") or {}),
        "inherited_from_context": bool(recall.get("inherited_from_context")),
        "intent_classification": dict(recall.get("intent_classification") or {}),
    }
    if docs_returned is not None:
        metadata["docs_returned"] = docs_returned
    if question_type:
        metadata["question_type"] = question_type
    if answer_mode:
        metadata["answer_mode"] = answer_mode
    if extra:
        metadata.update(extra)
    return metadata
