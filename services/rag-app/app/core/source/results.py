from typing import Any, Callable, Dict, List


def explicit_regulation_unique_result(
    *,
    resolved_source: str,
    base_source: str,
    raw_candidates: List[str],
    prepared_sources: List[str],
    target_text: str,
    content_query: str,
    trace_label: str,
) -> Dict[str, Any]:
    return {
        "route": "explicit_regulation_reference",
        "required": True,
        "resolved": True,
        "sources": [resolved_source],
        "candidates": [resolved_source],
        "reason": "latest_effective_unique" if (len(raw_candidates) > 1 or resolved_source != base_source) else "explicit_regulation_unique",
        "strip_title_mentions": True,
        "clarification": "",
        "target_text": target_text,
        "retrieval_query_override": content_query,
        "source_resolution_trace": {
            "trace_label": trace_label,
            "raw_candidates": raw_candidates[:5],
            "prepared_candidates": prepared_sources[:5],
            "resolved_source": resolved_source,
        },
    }

def pseudo_singleton_ambiguous_result(
    *,
    resolved_source: str,
    raw_candidates: List[str],
    prepared_sources: List[str],
    target_text: str,
    content_query: str,
    trace_label: str,
    build_document_clarification_prompt: Callable[[List[str]], str],
) -> Dict[str, Any]:
    return {
        "route": "explicit_regulation_reference",
        "required": True,
        "resolved": False,
        "sources": [],
        "candidates": [resolved_source],
        "reason": "document_ambiguous",
        "strip_title_mentions": False,
        "clarification": build_document_clarification_prompt([resolved_source]),
        "target_text": target_text,
        "retrieval_query_override": content_query,
        "source_resolution_trace": {
            "trace_label": trace_label,
            "raw_candidates": raw_candidates[:5],
            "prepared_candidates": prepared_sources[:5],
            "resolved_source": resolved_source,
            "blocked_reason": "pseudo_singleton_region_mismatch",
        },
    }

def geo_context_locked_result(
    *,
    resolved_source: str,
    raw_candidates: List[str],
    prepared_sources: List[str],
    geo_filtered: List[str],
    target_text: str,
    content_query: str,
    trace_label: str,
    source_display_title: Callable[[str], str],
) -> Dict[str, Any]:
    return {
        "route": "explicit_regulation_reference",
        "required": True,
        "resolved": True,
        "sources": [resolved_source],
        "candidates": prepared_sources[:3],
        "reason": "geo_context_locked",
        "strip_title_mentions": True,
        "clarification": "",
        "target_text": target_text,
        "lock_mode": "soft_lock",
        "lock_confidence": 0.82,
        "lock_message_prefix": f"我理解你查询的是《{source_display_title(resolved_source)}》。\n",
        "source_lock_kind": "geo_context_locked",
        "source_resolution_trace": {
            "trace_label": trace_label,
            "raw_candidates": raw_candidates[:5],
            "prepared_candidates": prepared_sources[:5],
            "geo_filtered_candidates": geo_filtered[:5],
        },
        "retrieval_query_override": content_query,
    }

def soft_lock_unique_result(
    *,
    unique_weak: Dict[str, Any],
    raw_candidates: List[str],
    prepared_sources: List[str],
    target_text: str,
    content_query: str,
    trace_label: str,
) -> Dict[str, Any]:
    resolved_source = str(unique_weak.get("source") or "")
    return {
        "route": "explicit_regulation_reference",
        "required": True,
        "resolved": True,
        "sources": [resolved_source],
        "candidates": [resolved_source],
        "reason": str(unique_weak.get("reason") or "soft_lock_unique"),
        "strip_title_mentions": True,
        "clarification": "",
        "target_text": target_text,
        "lock_mode": "soft_lock",
        "lock_confidence": float(unique_weak.get("confidence") or 0.0),
        "lock_message_prefix": str(unique_weak.get("lock_message_prefix") or ""),
        "source_lock_kind": "soft_lock_unique",
        "source_resolution_trace": {
            "trace_label": trace_label,
            "raw_candidates": raw_candidates[:5],
            "prepared_candidates": prepared_sources[:5],
            **dict(unique_weak.get("trace") or {}),
        },
        "retrieval_query_override": content_query,
    }

def topical_suffix_multi_doc_result(
    *,
    topical_multi: Dict[str, Any],
    raw_candidates: List[str],
    prepared_sources: List[str],
    target_text: str,
    content_query: str,
    trace_label: str,
) -> Dict[str, Any]:
    return {
        "route": "multi_doc_query",
        "required": False,
        "resolved": False,
        "sources": list(topical_multi.get("sources") or []),
        "candidates": list(topical_multi.get("sources") or []),
        "reason": str(topical_multi.get("reason") or "topical_suffix_multi_doc"),
        "strip_title_mentions": False,
        "clarification": "",
        "target_text": target_text,
        "lock_mode": "none",
        "source_lock_kind": "topical_suffix_multi_doc",
        "source_resolution_trace": {
            "trace_label": trace_label,
            "raw_candidates": raw_candidates[:5],
            "prepared_candidates": prepared_sources[:5],
            **dict(topical_multi.get("trace") or {}),
        },
        "retrieval_query_override": content_query,
    }


TOPICAL_SUFFIX_TERMS = {
    "处罚",
    "罚款",
    "责任",
    "职责",
    "审批",
    "备案",
    "许可",
    "流程",
    "要求",
    "条件",
    "标准",
    "范围",
    "管理",
    "监管",
    "禁止",
    "义务",
    "权利",
    "措施",
    "程序",
}

GENERIC_DOC_INTENT_TERMS = {
    "条例",
    "办法",
    "规定",
    "规则",
    "规程",
    "细则",
    "通知",
    "意见",
    "决定",
    "方案",
    "标准",
    "法规",
    "文件",
}

def explicit_regulation_ambiguous_result(
    *,
    raw_candidates: List[str],
    prepared_sources: List[str],
    geo_filtered: List[str],
    target_text: str,
    content_query: str,
    trace_label: str,
    build_document_clarification_prompt: Callable[[List[str]], str],
) -> Dict[str, Any]:
    return {
        "route": "explicit_regulation_reference",
        "required": True,
        "resolved": False,
        "sources": [],
        "candidates": prepared_sources[:3],
        "reason": "document_ambiguous",
        "strip_title_mentions": False,
        "clarification": build_document_clarification_prompt(prepared_sources[:3]),
        "target_text": target_text,
        "retrieval_query_override": content_query,
        "source_resolution_trace": {
            "trace_label": trace_label,
            "raw_candidates": raw_candidates[:5],
            "prepared_candidates": prepared_sources[:5],
            "geo_filtered_candidates": geo_filtered[:5],
        },
    }

def document_not_found_result(target_text: str) -> Dict[str, Any]:
    return {
        "route": "explicit_regulation_reference",
        "required": True,
        "resolved": False,
        "sources": [],
        "candidates": [],
        "reason": "document_not_found",
        "strip_title_mentions": False,
        "clarification": "",
        "target_text": target_text,
    }
