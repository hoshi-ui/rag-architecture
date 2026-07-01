import asyncio
from typing import Any, Dict, List, Optional

from app.core.legal_intent import classify_query_intent_fallback, normalize_legal_intent
from app.core.query.recall_flow import (
    OPEN_QA_ROUTES,
    STRONG_SOURCE_ROUTES,
    _classifier_action,
    _classifier_is_comparison,
    _classifier_question_type,
    _classifier_route,
    _empty_recall_result,
    degrade_router_target_failure_to_global_fallback,
    force_retrieval_source_resolution,
    has_forced_retrieval_signal,
    source_resolution_global_fallback,
)
from app.core.retrieval.ranking import build_retrieval_stage_trace


def _compare_process_min_required(recall: Dict[str, Any]) -> int:
    coverage = recall.get("compare_coverage") if isinstance(recall.get("compare_coverage"), dict) else {}
    try:
        return max(1, int(coverage.get("min_required_per_doc") or 1))
    except Exception:
        return 1


def _process_doc_source(runtime: Any, doc: Any) -> str:
    try:
        source = runtime.evidence.hit_entity_source(doc) or ""
    except Exception:
        source = ""
    try:
        return runtime.common.normalize_filename(source)
    except Exception:
        return str(source or "").strip()


def _process_doc_identity(runtime: Any, doc: Any) -> tuple:
    source = _process_doc_source(runtime, doc)
    metadata: Dict[str, Any] = {}
    try:
        metadata = runtime.evidence.hit_metadata(doc) or {}
    except Exception:
        metadata = {}
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_id_start") or metadata.get("id")
    article_no = metadata.get("article_no") or metadata.get("article_id") or metadata.get("clause_id") or ""
    try:
        text = str(runtime.evidence.hit_display_text(doc) or "")[:160]
    except Exception:
        text = ""
    return source, str(chunk_id or ""), str(article_no or ""), text


def _mark_compare_pinned_doc(runtime: Any, doc: Any, source: str, rank: int) -> Any:
    if not isinstance(doc, dict):
        return doc
    entity = doc.get("entity")
    if not isinstance(entity, dict):
        return doc
    metadata = entity.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    metadata["is_pinned"] = True
    metadata["compare_pinned"] = True
    metadata["compare_pin_source"] = source
    metadata["compare_pin_rank"] = int(rank)
    entity = dict(entity)
    entity["metadata"] = metadata
    out = dict(doc)
    out["entity"] = entity
    return out


def _compare_pinned_docs(
    runtime: Any,
    item: Dict[str, Any],
    limit: int = 2,
    query: str = "",
    query_intent: str = "",
) -> List[Any]:
    try:
        source = runtime.common.normalize_filename(item.get("source") or "")
    except Exception:
        source = str(item.get("source") or "").strip()
    out: List[Any] = []
    seen = set()
    candidates = (
        list(item.get("selected_docs") or [])
        + list(item.get("post_filter_docs") or [])
        + list(item.get("retrieve_docs") or [])
        + list(item.get("docs") or [])
    )
    for doc in _prioritize_legal_intent_process_docs(runtime, query, candidates, query_intent=query_intent):
        if source and _process_doc_source(runtime, doc) != source:
            continue
        identity = _process_doc_identity(runtime, doc)
        if identity in seen:
            continue
        seen.add(identity)
        out.append(_mark_compare_pinned_doc(runtime, doc, source, len(out) + 1))
        if len(out) >= max(1, int(limit or 1)):
            break
    return out


_INTENT_FEATURES: Dict[str, Dict[str, Any]] = {
    "定义与范围": {
        "heading": ("适用范围", "定义", "范围", "总则"),
        "body": ("适用本条例", "适用于", "本条例适用", "不适用", "不包括", "除外", "另有规定", "包括", "所称"),
    },
    "职责与权限": {
        "heading": ("职责", "权限", "职权", "机构职责", "监督管理"),
        "body": ("负责", "主管", "职责", "职权", "权限", "监督管理", "协助", "会同", "分工"),
    },
    "程序与条件": {
        "heading": ("程序", "条件", "办理", "登记", "审查", "申请"),
        "body": ("程序", "流程", "申请", "审查", "办理", "登记", "期限", "条件", "材料", "提交", "发放", "不予"),
    },
    "法律责任": {
        "heading": ("法律责任", "罚则"),
        "body": ("法律责任", "罚则", "处罚", "罚款", "没收", "责令", "吊销", "违法", "逾期", "警告"),
    },
    "权利义务": {
        "heading": ("权利", "义务", "行为规范"),
        "body": ("权利", "义务", "应当", "不得", "禁止", "可以", "鼓励", "要求"),
    },
}


def _process_query_intent(query: str, query_intent: str = "") -> str:
    return normalize_legal_intent(query_intent) or normalize_legal_intent(classify_query_intent_fallback(query))


def _process_doc_text(runtime: Any, doc: Any) -> str:
    try:
        return str(runtime.evidence.hit_display_text(doc) or "")
    except Exception:
        entity = doc.get("entity") if isinstance(doc, dict) else {}
        return str((entity or {}).get("text") or doc.get("text") or "")


def _process_doc_legal_intent(heading: str, body: str, article_no: str = "") -> str:
    haystack = f"{heading}\n{body}"
    scores: Dict[str, int] = {}
    for intent, features in _INTENT_FEATURES.items():
        score = 0
        score += sum(3 for term in features.get("heading", ()) if term and term in heading)
        score += sum(2 for term in features.get("body", ()) if term and term in haystack)
        if intent == "定义与范围" and article_no in {"第二条", "第2条"}:
            score += 1
        if score:
            scores[intent] = score
    if not scores:
        return "其他"
    return max(scores.items(), key=lambda item: item[1])[0]


def _legal_intent_process_doc_priority(runtime: Any, query: str, doc: Any, query_intent: str = "") -> int:
    intent = _process_query_intent(query, query_intent)
    if not intent or intent == "其他":
        return 0
    try:
        metadata = runtime.evidence.hit_metadata(doc) or {}
    except Exception:
        metadata = {}
    clause_meta = metadata.get("clause_metadata") if isinstance(metadata.get("clause_metadata"), dict) else {}
    article_no = str(
        metadata.get("article_no")
        or metadata.get("article_id")
        or metadata.get("clause_id")
        or clause_meta.get("article_no")
        or ""
    )
    heading = " ".join(
        str(value or "")
        for value in (
            metadata.get("heading"),
            metadata.get("section"),
            metadata.get("section_title"),
            clause_meta.get("section_title"),
        )
    )
    body = _process_doc_text(runtime, doc)
    features = _INTENT_FEATURES.get(intent) or {}
    if not features:
        return 0
    score = 0
    score += sum(4 for term in features.get("heading", ()) if term and term in heading)
    score += sum(3 for term in features.get("body", ()) if term and term in body)
    if _process_doc_legal_intent(heading, body, article_no) == intent:
        score += 6
    if intent == "定义与范围":
        if article_no in {"第二条", "第2条"}:
            score += 2
        if "为了" in body[:120] and "制定本条例" in body[:180]:
            score -= 5
        if "原则" in body and score <= 3:
            score -= 2
    return score


def _prioritize_legal_intent_process_docs(
    runtime: Any,
    query: str,
    docs: List[Any],
    query_intent: str = "",
) -> List[Any]:
    if not docs or not _process_query_intent(query, query_intent):
        return list(docs or [])
    scored = [
        (_legal_intent_process_doc_priority(runtime, query, doc, query_intent=query_intent), index, doc)
        for index, doc in enumerate(docs)
    ]
    if max((score for score, _, _ in scored), default=0) <= 0:
        return list(docs)
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc for _, _, doc in scored]


def _recall_pinned_docs(runtime: Any, recall: Dict[str, Any], limit: int = 3) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for doc in list((recall or {}).get("retrieve_docs") or []):
        identity = _process_doc_identity(runtime, doc)
        if identity in seen:
            continue
        seen.add(identity)
        out.append(doc)
        if len(out) >= max(1, int(limit or 1)):
            break
    return out


def _prepend_unique_process_docs(runtime: Any, pinned_docs: List[Any], docs: List[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for doc in list(pinned_docs or []) + list(docs or []):
        identity = _process_doc_identity(runtime, doc)
        if identity in seen:
            continue
        seen.add(identity)
        out.append(doc)
    return out


def _ensure_compare_process_quota(
    runtime: Any,
    process_docs: List[Any],
    compare_process_groups: List[Dict[str, Any]],
    min_required_per_doc: int,
) -> List[Any]:
    min_required = max(1, int(min_required_per_doc or 1))
    out = list(process_docs or [])
    seen = {_process_doc_identity(runtime, doc) for doc in out}
    counts: Dict[str, int] = {}
    for doc in out:
        source = _process_doc_source(runtime, doc)
        if source:
            counts[source] = counts.get(source, 0) + 1

    for group in compare_process_groups or []:
        try:
            target_source = runtime.common.normalize_filename(group.get("source") or "")
        except Exception:
            target_source = str(group.get("source") or "").strip()
        if not target_source:
            continue
        current = counts.get(target_source, 0)
        if current >= min_required:
            continue
        for doc in list(group.get("pinned_docs") or []) + list(group.get("docs") or []) + list(group.get("display_docs") or []) + list(group.get("gate_docs") or []):
            if _process_doc_source(runtime, doc) != target_source:
                continue
            identity = _process_doc_identity(runtime, doc)
            if identity in seen:
                continue
            out.append(doc)
            seen.add(identity)
            current += 1
            counts[target_source] = current
            if current >= min_required:
                break
    return out


def _build_compare_process_coverage(
    runtime: Any,
    recall: Dict[str, Any],
    compare_process_groups: List[Dict[str, Any]],
    process_docs: List[Any],
) -> Dict[str, Any]:
    base = dict(recall.get("compare_coverage") or {})
    min_required = _compare_process_min_required(recall)
    selected_counts: Dict[str, int] = {}
    for doc in process_docs or []:
        source = _process_doc_source(runtime, doc)
        if source:
            selected_counts[source] = selected_counts.get(source, 0) + 1
    group_by_source: Dict[str, Dict[str, Any]] = {}
    for group in compare_process_groups or []:
        try:
            source = runtime.common.normalize_filename(group.get("source") or "")
        except Exception:
            source = str(group.get("source") or "").strip()
        if source:
            group_by_source[source] = group

    target_docs = []
    for item in list(base.get("target_docs") or []):
        source = str(item.get("source") or item.get("doc_id") or "").strip()
        try:
            source_key = runtime.common.normalize_filename(source)
        except Exception:
            source_key = source
        group = group_by_source.get(source_key) or {}
        retrieved = len(group.get("gate_docs") or group.get("display_docs") or group.get("docs") or [])
        if retrieved <= 0:
            retrieved = int(item.get("retrieved") or 0)
        selected = selected_counts.get(source_key, 0)
        if retrieved <= 0:
            coverage = "missing"
        elif selected < min_required:
            coverage = "insufficient"
        else:
            coverage = "covered"
        target_docs.append({**item, "retrieved": retrieved, "selected": selected, "coverage": coverage})

    return {
        **base,
        "min_required_per_doc": min_required,
        "target_docs": target_docs,
        "any_doc_missing": any(item.get("coverage") == "missing" for item in target_docs),
        "any_doc_insufficient": any(item.get("coverage") in {"missing", "insufficient"} for item in target_docs),
    }


def prepare_process_query(
    runtime: Any,
    query: str,
    user_id: str,
    forced_fnames: Optional[List[str]] = None,
    too_short_answer: str = "",
    blocked_answer: str = "",
) -> Dict[str, Any]:
    query = runtime.common.normalize_query(query)
    qtype = runtime.routing.classify_question_type(query)
    original_fnames = list(forced_fnames or runtime.routing.extract_filename_candidates(query))
    if len(query) < runtime.config.MIN_QUERY_CHARS:
        return {
            "query": query,
            "qtype": qtype,
            "original_fnames": original_fnames,
            "early_return": {
                "answer": too_short_answer,
                "sources": [],
                "metadata": runtime.control.metadata(
                    query=query,
                    user_id=user_id,
                    query_route="query_too_short",
                    internal_route="query_too_short",
                    final_channel="blocked",
                    blocked="query_too_short",
                    query_quality="invalid",
                    answer_mode="refusal",
                ),
            },
        }
    if len(query) > runtime.config.MAX_QUERY_CHARS:
        return {
            "query": query,
            "qtype": qtype,
            "original_fnames": original_fnames,
            "early_return": {
                "answer": f"????????? {runtime.config.MAX_QUERY_CHARS} ??????",
                "sources": [],
                "metadata": runtime.control.metadata(
                    query=query,
                    user_id=user_id,
                    query_route="query_too_long",
                    internal_route="query_too_long",
                    final_channel="blocked",
                    blocked="query_too_long",
                    query_quality="invalid",
                    answer_mode="refusal",
                ),
            },
        }
    blocked = runtime.guardrails.blocked_reason(query)
    if blocked:
        return {
            "query": query,
            "qtype": qtype,
            "original_fnames": original_fnames,
            "early_return": {
                "answer": blocked_answer,
                "sources": [],
                "metadata": runtime.control.metadata(
                    query=query,
                    user_id=user_id,
                    query_route=blocked,
                    internal_route=blocked,
                    final_channel="blocked",
                    blocked=blocked,
                    query_quality="invalid",
                    answer_mode="refusal",
                ),
            },
        }
    query_quality = runtime.guardrails.static_quality_state(query)
    if query_quality["reason"]:
        return {
            "query": query,
            "qtype": qtype,
            "original_fnames": original_fnames,
            "early_return": {
                "answer": runtime.guardrails.invalid_query_message(query_quality["reason"]),
                "sources": [],
                "metadata": runtime.control.metadata(
                    query=query,
                    user_id=user_id,
                    query_route=query_quality["reason"],
                    internal_route=query_quality["reason"],
                    final_channel="blocked",
                    blocked=query_quality["reason"],
                    query_quality=query_quality["quality"],
                    answer_mode="refusal",
                ),
            },
        }
    return {
        "query": query,
        "qtype": qtype,
        "original_fnames": original_fnames,
        "early_return": None,
    }

async def prepare_lightweight_recall_prelude(
    runtime: Any,
    query: str,
    filename_hints: Optional[List[str]] = None,
    user_id: str = "anonymous",
) -> Dict[str, Any]:
    from app.core.query import agentic_router
    from app.core.query.tool_router import route_with_search_database_tool

    intent_classification: Dict[str, Any] = {}
    tool_route = await route_with_search_database_tool(runtime, query)
    if tool_route.get("tool_called"):
        tool_args = dict(tool_route.get("arguments") or {})
        tool_query = str(tool_route.get("query") or tool_args.get("query") or query).strip() or query
        if not str(tool_args.get("query") or "").strip():
            tool_args["query"] = tool_query
        tool_reason = str(tool_route.get("reason") or tool_args.get("reason") or "").strip()
        intent_classification = {
            "search_database_tool_used": True,
            "search_database_tool_args": tool_args,
            "search_database_tool_query": tool_query,
            "search_database_tool_reason": tool_reason,
            "search_database_tool_route": {
                "tool_called": True,
                "tool_name": str(tool_route.get("tool_name") or "search_database"),
                "arguments": tool_args,
                "query": tool_query,
                "reason": tool_reason,
            },
        }

    qtype = runtime.routing.classify_question_type(query)
    llm_parse: Dict[str, Any] = {}
    if bool(getattr(runtime.config, "ENABLE_LLM_QUERY_PARSE", True)):
        try:
            parsed = await asyncio.to_thread(runtime.llm.parse_query_cached, query, "")
            if isinstance(parsed, dict):
                llm_parse = parsed
        except Exception:
            llm_parse = {}
    if tool_route.get("tool_called") and str((intent_classification or {}).get("search_database_tool_query") or tool_route.get("query") or "").strip():
        tool_query = runtime.common.normalize_query(
            str((intent_classification or {}).get("search_database_tool_query") or tool_route.get("query") or "")
        )
        if tool_query:
            llm_parse = {
                **dict(llm_parse or {}),
                "retrieval_query": tool_query,
                "dense_query": tool_query,
                "search_database_tool_query": tool_query,
            }

    query_explicit_fnames = [
        runtime.common.normalize_filename(name or "")
        for name in runtime.routing.extract_filename_candidates(query)
        if runtime.common.normalize_filename(name or "")
    ]
    source_resolution = await asyncio.to_thread(
        runtime.source.resolve_targets,
        query,
        list(filename_hints or []),
        user_id,
    )
    agentic_route: Dict[str, Any] = {}
    agentic_compare_resolution: Dict[str, Any] = {}
    if bool(getattr(runtime.config, "ENABLE_AGENTIC_ROUTER", True)):
        agentic_route = await agentic_router.route_query(runtime, query)
        agentic_compare_resolution = agentic_router.build_compare_resolution(runtime, query, agentic_route)
        if agentic_compare_resolution:
            original_source_resolution = dict(source_resolution or {})
            resolved = bool(agentic_compare_resolution.get("resolved"))
            agentic_sources = list(agentic_compare_resolution.get("sources") or [])
            final_status = "locked" if resolved and agentic_sources else "global_fallback"
            source_resolution = runtime.source.build_source_resolution_result(
                route=agentic_compare_resolution.get("route") or "multi_doc_compare",
                required=bool(agentic_compare_resolution.get("required")),
                resolved=resolved,
                sources=agentic_sources,
                candidates=agentic_sources,
                reason=agentic_compare_resolution.get("reason") or "agentic_router",
                strip_title_mentions=bool(agentic_compare_resolution.get("strip_title_mentions")),
                clarification=agentic_compare_resolution.get("clarification") or "",
                target_text=agentic_compare_resolution.get("target_text") or "",
                lock_mode="hard_lock" if resolved else "none",
                lock_confidence=float(agentic_route.get("confidence") or (1.0 if resolved else 0.0)),
                source_lock_kind="agentic_compare_lock",
                source_resolution_trace={
                    **dict(original_source_resolution.get("source_resolution_trace") or {}),
                    "agentic_router": {
                        "used": True,
                        "route": agentic_route.get("route") or "",
                        "query_intent": agentic_route.get("query_intent") or "",
                        "confidence": float(agentic_route.get("confidence") or 0.0),
                        "rationale": agentic_route.get("rationale") or "",
                        "sub_queries": list(agentic_route.get("sub_queries") or []),
                    },
                    "final_source_resolution": {
                        "selected": "agentic_router",
                        "route": agentic_compare_resolution.get("route") or "multi_doc_compare",
                        "status": final_status,
                        "source_lock_kind": "agentic_compare_lock",
                        "target_fnames": agentic_sources,
                        "compare_status": agentic_compare_resolution.get("compare_status") or "",
                        "reason": agentic_compare_resolution.get("reason") or "agentic_router",
                    },
                    "rule_compare_diagnostic": {
                        "ignored": True,
                        "route": original_source_resolution.get("route") or "",
                        "status": original_source_resolution.get("status") or "",
                        "reason": original_source_resolution.get("reason") or "",
                        "compare_status": original_source_resolution.get("compare_status") or "",
                        "compare_subjects": list(original_source_resolution.get("compare_subjects") or []),
                        "compare_missing_targets": list(original_source_resolution.get("compare_missing_targets") or []),
                    },
                },
                compare_subjects=list(agentic_compare_resolution.get("subjects") or []),
                compare_doc_like_subjects=list(agentic_compare_resolution.get("doc_like_subjects") or []),
                compare_missing_targets=list(agentic_compare_resolution.get("missing_doc_targets") or []),
                compare_common_aspects=list(agentic_compare_resolution.get("common_aspects") or []),
                compare_topic_pair=list(agentic_compare_resolution.get("topic_pair") or []),
                compare_canonical_aspects=list(agentic_compare_resolution.get("canonical_aspects") or []),
                compare_expanded_aspects=list(agentic_compare_resolution.get("expanded_aspects") or []),
                compare_source_subqueries=dict(agentic_compare_resolution.get("source_subqueries") or {}),
                compare_status=agentic_compare_resolution.get("compare_status") or "plan_ready",
                compare_plan=dict(agentic_compare_resolution.get("compare_plan") or agentic_compare_resolution),
            )
            source_resolution = degrade_router_target_failure_to_global_fallback(source_resolution)
            intent_classification = {
                **dict(intent_classification),
                "agentic_router_used": True,
                "agentic_router": {
                    "route": agentic_route.get("route") or "",
                    "query_intent": agentic_route.get("query_intent") or "",
                    "confidence": float(agentic_route.get("confidence") or 0.0),
                    "rationale": agentic_route.get("rationale") or "",
                    "sub_queries": list(agentic_route.get("sub_queries") or []),
                },
            }
        else:
            agentic_single_resolution = agentic_router.build_single_source_resolution(runtime, query, agentic_route)
            if agentic_single_resolution:
                original_source_resolution = dict(source_resolution or {})
                source_resolution = runtime.source.build_source_resolution_result(
                    route=agentic_single_resolution.get("route") or original_source_resolution.get("route") or "content_qa",
                    required=bool(agentic_single_resolution.get("required")),
                    resolved=True,
                    sources=list(agentic_single_resolution.get("sources") or []),
                    candidates=list(agentic_single_resolution.get("candidates") or agentic_single_resolution.get("sources") or []),
                    reason=agentic_single_resolution.get("reason") or "agentic_single_source_lock",
                    strip_title_mentions=bool(agentic_single_resolution.get("strip_title_mentions")),
                    clarification=agentic_single_resolution.get("clarification") or "",
                    target_text=agentic_single_resolution.get("target_text") or "",
                    lock_mode=agentic_single_resolution.get("lock_mode") or "implicit_lock",
                    lock_confidence=float(agentic_single_resolution.get("lock_confidence") or agentic_route.get("confidence") or 0.0),
                    source_lock_kind=agentic_single_resolution.get("source_lock_kind") or "agentic_single_source_lock",
                    source_resolution_trace={
                        **dict(original_source_resolution.get("source_resolution_trace") or {}),
                        **dict(agentic_single_resolution.get("source_resolution_trace") or {}),
                        "final_source_resolution": {
                            "selected": "agentic_single_source_lock",
                            "route": agentic_single_resolution.get("route") or "content_qa",
                            "status": "locked",
                            "source_lock_kind": agentic_single_resolution.get("source_lock_kind") or "agentic_single_source_lock",
                            "target_fnames": list(agentic_single_resolution.get("sources") or []),
                            "reason": agentic_single_resolution.get("reason") or "agentic_single_source_lock",
                        },
                    },
                )
                intent_classification = {
                    **dict(intent_classification),
                    "agentic_router_used": True,
                    "agentic_router": {
                        "route": agentic_route.get("route") or "",
                        "query_intent": agentic_route.get("query_intent") or "",
                        "confidence": float(agentic_route.get("confidence") or 0.0),
                        "rationale": agentic_route.get("rationale") or "",
                        "sub_queries": list(agentic_route.get("sub_queries") or []),
                        "accepted": True,
                        "accepted_as": "single_source_lock",
                    },
                }
        if (
            not agentic_compare_resolution
            and source_resolution.get("source_lock_kind") != "agentic_single_source_lock"
            and agentic_route.get("used")
        ):
            source_resolution = {
                **dict(source_resolution or {}),
                "source_resolution_trace": {
                    **dict((source_resolution or {}).get("source_resolution_trace") or {}),
                    "agentic_router_diagnostic": {
                        "used": True,
                        "accepted": False,
                        "route": agentic_route.get("route") or "",
                        "query_intent": agentic_route.get("query_intent") or "",
                        "confidence": float(agentic_route.get("confidence") or 0.0),
                        "reason": agentic_route.get("reason") or "",
                        "rationale": agentic_route.get("rationale") or "",
                        "sub_queries": list(agentic_route.get("sub_queries") or []),
                    },
                },
            }
            intent_classification = {
                **dict(intent_classification),
                "agentic_router_used": True,
                "agentic_router": {
                    "route": agentic_route.get("route") or "",
                    "query_intent": agentic_route.get("query_intent") or "",
                    "confidence": float(agentic_route.get("confidence") or 0.0),
                    "rationale": agentic_route.get("rationale") or "",
                    "sub_queries": list(agentic_route.get("sub_queries") or []),
                    "accepted": False,
                },
            }
    rule_query_route = (
        source_resolution.get("route")
        or runtime.routing.classify_query_route(query, list(filename_hints or []))
        or "content_qa"
    )
    classifier_route = ""
    source_route = str(source_resolution.get("route") or "")
    source_route_is_strong = source_route in STRONG_SOURCE_ROUTES
    query_route = rule_query_route

    classifier_compare = None
    classifier_action = ""
    forced_retrieval = has_forced_retrieval_signal(runtime, query)
    if (
        bool(intent_classification.get("search_database_tool_used"))
        and source_resolution_global_fallback(source_resolution)
        and query_route in {"document_clarification", "refusal", "compare_clarification"}
    ):
        query_route = rule_query_route if rule_query_route not in {"document_clarification", "refusal", "compare_clarification"} else "content_qa"
        source_resolution = force_retrieval_source_resolution(
            source_resolution,
            "search_database_tool_probe",
        )

    if forced_retrieval and source_resolution_global_fallback(source_resolution) and query_route == "document_clarification":
        query_route = rule_query_route if rule_query_route not in {"document_clarification", "refusal"} else "content_qa"
        source_resolution = force_retrieval_source_resolution(
            source_resolution,
            "forced_retrieval_explicit_signal",
        )
        intent_classification = {
            **dict(intent_classification),
            "forced_retrieval_fallback": True,
            "original_route": classifier_route,
        }

    query_quality = runtime.guardrails.deep_quality_state(query, llm_parse=llm_parse, source_resolution=source_resolution)
    intent_tier = str(query_quality.get("tier") or "")
    early_return = None
    if query_quality["reason"]:
        early_return = _empty_recall_result(
            runtime,
            query,
            user_id,
            qtype,
            llm_parse,
            intent_classification,
            source_resolution,
            query_quality["reason"],
            classifier_compare,
            reason=source_resolution.get("reason") or "",
            quality=query_quality["quality"],
            blocked_reason=query_quality["reason"],
            intent_tier=intent_tier,
        )
        early_return["source_lock_candidates"] = list(source_resolution.get("candidates") or [])

    return {
        "early_return": early_return,
        "intent_classification": intent_classification,
        "qtype": qtype,
        "llm_parse": llm_parse,
        "query_explicit_fnames": query_explicit_fnames,
        "query_explicit_set": set(query_explicit_fnames),
        "source_resolution": source_resolution,
        "query_route": query_route,
        "classifier_compare": classifier_compare,
        "classifier_action": classifier_action,
        "query_quality": query_quality,
        "intent_tier": intent_tier,
        "tool_route": tool_route,
    }

def build_process_retrieval_error_result(
    runtime: Any,
    query: str,
    user_id: str,
    qtype: str,
    error: Exception,
    answer: str,
) -> Dict[str, Any]:
    return {
        "answer": answer,
        "sources": [],
        "metadata": runtime.control.metadata(
            query=query,
            user_id=user_id,
            query_route="retrieval_error",
            internal_route="retrieval_error",
            final_channel="refusal",
            refusal_reason="retrieval_error",
            docs_returned=0,
            question_type=qtype,
            answer_mode="refusal",
            extra={
                "refused": "retrieval_error",
                "error": str(error),
                "error_type": type(error).__name__,
            },
        ),
    }

def build_process_recall_blocked_result(
    runtime: Any,
    query: str,
    user_id: str,
    qtype: str,
    recall: Dict[str, Any],
    clarification: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    reason = recall.get("blocked_reason") or "low_information_query"
    clarification = clarification or {}
    if clarification.get("candidate_sources"):
        return {
            "answer": clarification.get("message") or runtime.guardrails.invalid_query_message(reason),
            "sources": [],
            "metadata": runtime.control.metadata(
                query=query,
                user_id=user_id,
                query_route="document_clarification",
                internal_route=reason,
                final_channel="document_clarification",
                refusal_reason=reason,
                query_quality=recall.get("query_quality") or "low_information",
                answer_mode="clarification",
                docs_returned=0,
                question_type=qtype,
                recall=recall,
                extra={
                    "refused": reason,
                    "candidate_sources": list(clarification.get("candidate_sources") or []),
                    "clarification": clarification.get("message") or "",
                    "clarification_used_llm": bool(clarification.get("used_llm")),
                },
            ),
        }
    return {
        "answer": runtime.guardrails.invalid_query_message(reason),
        "sources": [],
        "metadata": runtime.control.metadata(
            query=query,
            user_id=user_id,
            query_route=reason,
            internal_route=reason,
            final_channel="blocked",
            blocked=reason,
            query_quality=recall.get("query_quality") or "low_information",
            answer_mode="refusal",
            recall=recall,
        ),
    }

def build_process_soft_clarification_result(
    runtime: Any,
    query: str,
    user_id: str,
    qtype: str,
    recall: Dict[str, Any],
    clarification: Dict[str, Any],
) -> Dict[str, Any]:
    reason = recall.get("soft_clarification_reason") or "document_clarification"
    return {
        "answer": clarification.get("message") or runtime.source.clarification_prompt(recall.get("source_lock_candidates") or []),
        "sources": [],
        "metadata": runtime.control.metadata(
            query=query,
            user_id=user_id,
            query_route="document_clarification",
            internal_route=recall.get("soft_clarification_reason") or recall.get("query_route") or "content_qa",
            final_channel="document_clarification",
            refusal_reason=reason,
            query_quality=recall.get("query_quality") or "valid",
            answer_mode="clarification",
            docs_returned=0,
            question_type=qtype,
            recall=recall,
            extra={
                "refused": reason,
                "candidate_sources": list(clarification.get("candidate_sources") or recall.get("source_lock_candidates") or []),
                "clarification": clarification.get("message") or "",
                "clarification_used_llm": bool(clarification.get("used_llm")),
                "intent_tier": recall.get("intent_tier") or "",
            },
        ),
    }

def build_process_source_lock_result(
    runtime: Any,
    query: str,
    user_id: str,
    qtype: str,
    recall: Dict[str, Any],
    clarification: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    source_lock_reason = recall.get("source_lock_reason") or "document_target_required"
    if recall.get("source_resolution_status") == "not_found" and source_lock_reason not in {
        "compare_target_not_found",
        "compare_targets_not_found",
    }:
        source_lock_reason = "document_not_found"
    if source_lock_reason in {"compare_target_not_found", "compare_targets_not_found", "compare_source_set_incomplete"}:
        compare_clarification = recall.get("clarification") or runtime.compare.clarification_prompt(
            list(recall.get("compare_subjects") or []),
            list(recall.get("source_lock_candidates") or recall.get("target_sources") or []),
        )
        is_incomplete = source_lock_reason == "compare_source_set_incomplete"
        return {
            "answer": compare_clarification if is_incomplete else runtime.compare.target_not_found_prompt(
                list(recall.get("compare_missing_targets") or []),
                list(recall.get("source_lock_candidates") or recall.get("target_sources") or []),
            ),
            "sources": [],
            "metadata": runtime.control.metadata(
                query=query,
                user_id=user_id,
                query_route="compare_clarification" if is_incomplete else source_lock_reason,
                internal_route=recall.get("query_route") or source_lock_reason,
                final_channel="compare_clarification" if is_incomplete else "document_not_found",
                recall=recall,
                refusal_reason=source_lock_reason,
                docs_returned=0,
                question_type=qtype,
                answer_mode="clarification" if is_incomplete else "refusal",
                extra={
                    "refused": source_lock_reason,
                    "target_text": recall.get("target_text") or "",
                    "clarification": compare_clarification if is_incomplete else "",
                },
            ),
        }
    if source_lock_reason == "document_not_found":
        return {
            "answer": runtime.source.not_found_prompt(recall.get("target_text") or query),
            "sources": [],
            "metadata": runtime.control.metadata(
                query=query,
                user_id=user_id,
                query_route="document_not_found",
                internal_route=recall.get("query_route") or "explicit_regulation_reference",
                final_channel="document_not_found",
                recall=recall,
                refusal_reason="document_not_found",
                docs_returned=0,
                question_type=qtype,
                answer_mode="refusal",
                extra={"refused": "document_not_found", "target_text": recall.get("target_text") or ""},
            ),
        }
    clarification = clarification or {}
    return {
        "answer": clarification.get("message") or recall.get("clarification") or runtime.source.clarification_prompt(recall.get("source_lock_candidates") or []),
        "sources": [],
        "metadata": runtime.control.metadata(
            query=query,
            user_id=user_id,
            query_route="document_ambiguous" if source_lock_reason == "document_ambiguous" else "document_clarification",
            internal_route=recall.get("query_route") or "weak_title_reference",
            final_channel="document_ambiguous" if source_lock_reason == "document_ambiguous" else "document_clarification",
            recall=recall,
            refusal_reason=source_lock_reason,
            docs_returned=0,
            question_type=qtype,
            answer_mode="clarification",
            extra={
                "refused": source_lock_reason,
                "candidate_sources": list(clarification.get("candidate_sources") or recall.get("source_lock_candidates") or []),
                "clarification": clarification.get("message") or recall.get("clarification") or "",
                "clarification_used_llm": bool(clarification.get("used_llm")),
                "target_text": recall.get("target_text") or "",
            },
        ),
    }

async def prepare_process_evidence_context(
    runtime: Any,
    query: str,
    recall: Dict[str, Any],
    original_fnames: List[str],
) -> Dict[str, Any]:
    resolved_targets = [
        runtime.common.normalize_filename(x)
        for x in (recall.get("target_sources") or original_fnames)
        if runtime.common.normalize_filename(x)
    ]
    process_input_docs = _prepend_unique_process_docs(
        runtime,
        _recall_pinned_docs(runtime, recall, limit=3),
        recall["selected_docs"],
    )

    process_seed_docs = runtime.evidence.select_process_docs(
        query,
        process_input_docs,
        recall["score_mode"],
        recall["qfilters"],
        recall["final_n"],
        recall.get("intent_classification") or {},
    )
    display_seed_docs = process_seed_docs[:]
    compare_process_groups: List[Dict[str, Any]] = []

    if recall.get("compare_source_results"):
        min_required_per_doc = _compare_process_min_required(recall)
        query_intent = str((recall.get("qfilters") or {}).get("_legal_intent") or "")
        for item in recall.get("compare_source_results") or []:
            pinned_docs = _compare_pinned_docs(
                runtime,
                item,
                limit=max(2, min_required_per_doc),
                query=query,
                query_intent=query_intent,
            )
            group_candidates = _prioritize_legal_intent_process_docs(
                runtime,
                query,
                list(pinned_docs)
                + list(item.get("post_filter_docs") or [])
                + list(item.get("selected_docs") or [])
                + list(item.get("retrieve_docs") or [])
                + list(item.get("docs") or []),
                query_intent=query_intent,
            )
            group_docs = runtime.evidence.select_process_docs(
                query,
                group_candidates,
                item.get("score_mode") or recall["score_mode"],
                recall["qfilters"],
                max(2, recall["final_n"]),
                recall.get("intent_classification") or {},
            )
            group_docs = _prepend_unique_process_docs(runtime, pinned_docs, group_docs)
            display_group_docs = group_docs[:]
            group_docs = runtime.evidence.expand_docs_with_full_article_chunks(group_docs)
            group_docs = _prepend_unique_process_docs(runtime, pinned_docs, group_docs)
            compare_process_groups.append(
                {
                    "source": item.get("source") or "",
                    "evidence_query": item.get("evidence_query") or "",
                    "pinned_docs": pinned_docs,
                    "docs": group_docs,
                    "display_docs": display_group_docs,
                    "gate_docs": display_group_docs,
                    "score_mode": item.get("score_mode") or recall["score_mode"],
                }
            )
        process_docs = runtime.evidence.merge_compare_source_doc_groups(compare_process_groups, per_source_limit=max(2, recall["final_n"]))
        process_docs = _ensure_compare_process_quota(runtime, process_docs, compare_process_groups, min_required_per_doc)
        display_seed_docs = runtime.evidence.merge_compare_source_doc_groups(
            [
                {
                    "source": item.get("source") or "",
                    "docs": item.get("display_docs") or item.get("docs") or [],
                }
                for item in compare_process_groups
            ],
            per_source_limit=max(2, recall["final_n"]),
        )
        observations = await runtime.evidence.compare_observations_async(query, compare_process_groups, qfilters=recall["qfilters"])
        compare_coverage = _build_compare_process_coverage(runtime, recall, compare_process_groups, process_docs)
        if compare_coverage:
            target_docs = list(compare_coverage.get("target_docs") or [])
            if compare_coverage.get("any_doc_missing"):
                observations = {
                    **dict(observations or {}),
                    "compare_status": "partial_sources_missing"
                    if any(item.get("coverage") == "covered" for item in target_docs)
                    else "all_sources_missing",
                    "answer_scope": "refusal",
                    "evidence_coverage_reason": "compare_source_missing",
                    "compare_coverage": compare_coverage,
                }
            elif compare_coverage.get("any_doc_insufficient"):
                observations = {
                    **dict(observations or {}),
                    "compare_status": "compare_coverage_insufficient",
                    "answer_scope": "refusal",
                    "evidence_coverage_reason": "compare_coverage_insufficient",
                    "compare_coverage": compare_coverage,
                }
            else:
                observations = {**dict(observations or {}), "compare_coverage": compare_coverage}
        return {
            "resolved_targets": resolved_targets,
            "process_input_docs": process_input_docs,
            "process_seed_docs": process_seed_docs,
            "display_seed_docs": display_seed_docs,
            "process_docs": process_docs,
            "compare_process_groups": compare_process_groups,
            "observations": observations,
            "process_stage_trace": build_retrieval_stage_trace(
                runtime,
                {
                    "process_input_docs": process_input_docs,
                    "process_seed_docs": process_seed_docs,
                    "process_docs": process_docs,
                    "display_seed_docs": display_seed_docs,
                },
                score_mode=recall["score_mode"],
            ),
        }

    process_docs = runtime.evidence.expand_docs_with_full_article_chunks(process_seed_docs)
    observations = await runtime.evidence.observations_async(
        recall.get("evidence_query") or recall.get("retrieval_query") or query,
        process_docs,
        qfilters=recall["qfilters"],
        candidate_docs=process_input_docs,
        target_sources=resolved_targets,
        source_lock_resolved=bool(recall.get("resolved_source_lock")),
        source_lock_reason=str(recall.get("source_lock_reason") or ""),
        is_comparison=bool(recall.get("is_comparison")),
        compare_missing_targets=list(recall.get("compare_missing_targets") or []),
        score_mode=recall["score_mode"],
        rerank_docs=process_seed_docs,
    )
    if bool(recall.get("resolved_source_lock")) and len(resolved_targets) == 1 and observations.get("uncovered_aspects"):
        aspect_rescue_docs = runtime.evidence.aspect_rescue_seed_docs(
            recall.get("evidence_query") or recall.get("retrieval_query") or query,
            process_seed_docs,
            list(observations.get("uncovered_aspects") or []),
            resolved_targets[0],
            qfilters=recall["qfilters"],
            top_n_per_aspect=2,
        )
        if aspect_rescue_docs:
            process_seed_docs = runtime.evidence.dedupe_docs(
                process_seed_docs + aspect_rescue_docs,
                max(len(process_seed_docs) + len(aspect_rescue_docs), recall["final_n"]),
            )
            display_seed_docs = process_seed_docs[:]
            process_docs = runtime.evidence.expand_docs_with_full_article_chunks(process_seed_docs)
            rescued_observations = await runtime.evidence.observations_async(
                recall.get("evidence_query") or recall.get("retrieval_query") or query,
                process_docs,
                qfilters=recall["qfilters"],
                candidate_docs=process_input_docs,
                target_sources=resolved_targets,
                source_lock_resolved=bool(recall.get("resolved_source_lock")),
                source_lock_reason=str(recall.get("source_lock_reason") or ""),
                is_comparison=bool(recall.get("is_comparison")),
                compare_missing_targets=list(recall.get("compare_missing_targets") or []),
                score_mode=recall["score_mode"],
                rerank_docs=process_seed_docs,
            )
            observations = {
                **rescued_observations,
                "aspect_rescue_attempted": True,
                "aspect_rescue_success": len(list(rescued_observations.get("uncovered_aspects") or [])) < len(list(observations.get("uncovered_aspects") or [])),
                "aspect_rescue_added_docs": len(aspect_rescue_docs),
                "aspect_rescue_aspects": list(dict.fromkeys([str(item).strip() for item in (observations.get("uncovered_aspects") or []) if str(item).strip()]))[:6],
            }
        else:
            observations = {
                **observations,
                "aspect_rescue_attempted": True,
                "aspect_rescue_success": False,
                "aspect_rescue_added_docs": 0,
                "aspect_rescue_aspects": list(dict.fromkeys([str(item).strip() for item in (observations.get("uncovered_aspects") or []) if str(item).strip()]))[:6],
            }

    return {
        "resolved_targets": resolved_targets,
        "process_input_docs": process_input_docs,
        "process_seed_docs": process_seed_docs,
        "display_seed_docs": display_seed_docs,
        "process_docs": process_docs,
        "compare_process_groups": compare_process_groups,
        "observations": observations,
        "process_stage_trace": build_retrieval_stage_trace(
            runtime,
            {
                "process_input_docs": process_input_docs,
                "process_seed_docs": process_seed_docs,
                "process_docs": process_docs,
                "display_seed_docs": display_seed_docs,
            },
            score_mode=recall["score_mode"],
        ),
    }

def _evidence_gate_warning(observations: Dict[str, Any]) -> str:
    reason = str((observations or {}).get("evidence_coverage_reason") or "").strip()
    uncovered = [
        str(item).strip()
        for item in ((observations or {}).get("uncovered_aspects") or [])
        if str(item).strip()
    ]
    parts = []
    if reason:
        parts.append(f"证据门控提示：当前证据被标记为 {reason}。")
    if uncovered:
        parts.append("未覆盖方面：" + "、".join(uncovered[:6]) + "。")
    parts.append("请优先基于已提供的 context 回答；context 未覆盖的部分明确说明“现有证据未覆盖”，不要编造。")
    return "".join(parts)


def downgrade_evidence_refusal_to_prompt_warning(
    observations: Dict[str, Any],
    process_docs: List[Any],
) -> Dict[str, Any]:
    observations = dict(observations or {})
    if not process_docs:
        return observations
    if str(observations.get("answer_scope") or "") in {"full", "guarded_full"}:
        return observations
    reason = str(observations.get("evidence_coverage_reason") or "").strip()
    return {
        **observations,
        "answer_scope": "guarded_full",
        "evidence_gate_degraded": True,
        "evidence_gate_original_scope": observations.get("answer_scope") or "",
        "evidence_gate_original_reason": reason,
        "evidence_gate_warning": observations.get("evidence_gate_warning") or _evidence_gate_warning(observations),
    }


def process_refusal_reason(
    recall: Dict[str, Any],
    resolved_targets: List[str],
    observations: Dict[str, Any],
    process_docs: Optional[List[Any]] = None,
) -> Optional[str]:
    if process_docs and str((observations or {}).get("answer_scope") or "") not in {"full", "guarded_full"}:
        return None
    allow_partial_answer = bool(
        recall.get("query_route") not in {"multi_doc_compare", "single_doc_compare"}
        and bool(recall.get("resolved_source_lock"))
        and len(resolved_targets) == 1
        and str(observations.get("answer_scope") or "") == "partial"
        and (
            (
                list(observations.get("covered_aspects") or [])
                and list(observations.get("uncovered_aspects") or [])
            )
            or (
                bool(observations.get("locked_doc_substantive_override"))
                and int(observations.get("qualified_substantive_chunks") or 0) >= 2
            )
        )
    )
    if (
        observations["answer_scope"] not in {"full", "guarded_full"}
        and not bool(observations.get("compare_degraded"))
        and not allow_partial_answer
    ):
        return observations["evidence_coverage_reason"]
    return None


def serialize_response_documents(
    runtime: Any,
    docs: List[Any],
    query: str,
    score_mode: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    doc_items = list(docs or [])[: max(1, int(limit or 20))]
    serialized = runtime.evidence.build_sources(doc_items, query, score_mode=score_mode)
    out: List[Dict[str, Any]] = []
    for index, item in enumerate(serialized or []):
        doc = doc_items[index] if index < len(doc_items) else None
        metadata = dict((item or {}).get("metadata") or {})
        text = ""
        if doc is not None:
            try:
                text = str(runtime.evidence.hit_display_text(doc) or "").strip()
            except Exception:
                text = ""
        if not text:
            text = str((item or {}).get("text") or "").strip()
        source = str((item or {}).get("source") or metadata.get("source_file") or "").strip()
        out.append(
            {
                "ref": (item or {}).get("ref"),
                "source": source,
                "score": (item or {}).get("score"),
                "text": text,
                "content": text,
                "metadata": metadata,
                "section": (item or {}).get("section") or metadata.get("section") or metadata.get("section_title") or "",
                "chunk_range": (item or {}).get("chunk_range") or "",
                "article_id": (item or {}).get("article_id") or metadata.get("article_id") or "",
                "article_no": (item or {}).get("article_no") or metadata.get("article_no") or "",
                "clause_id": (item or {}).get("clause_id") or metadata.get("clause_id") or "",
                "doc_id": (item or {}).get("doc_id") or metadata.get("doc_id") or source,
                "doc_title": metadata.get("doc_title") or source,
                "metadata_available": bool((item or {}).get("metadata_available") or metadata.get("metadata_available")),
            }
        )
    return out


def build_process_evidence_refusal_result(
    runtime: Any,
    query: str,
    user_id: str,
    qtype: str,
    recall: Dict[str, Any],
    process_docs: List[Any],
    observations: Dict[str, Any],
    refusal_reason: str,
) -> Dict[str, Any]:
    compare_route = recall.get("query_route") == "multi_doc_compare"
    response_documents = serialize_response_documents(
        runtime,
        process_docs,
        query,
        recall["score_mode"],
        limit=max(10, int(recall.get("final_n") or 10)),
    )
    return {
        "answer": runtime.compare.evidence_failure_prompt(observations.get("compare_source_statuses") or [])
        if recall.get("query_route") == "multi_doc_compare"
        else runtime.guardrails.evidence_refusal_answer(query, refusal_reason, observations),
        "sources": [],
        "documents": response_documents,
        "retrieved_contexts": response_documents,
        "metadata": runtime.control.metadata(
            query=query,
            user_id=user_id,
            query_route="compare_rag" if compare_route else "evidence_insufficient",
            internal_route=recall.get("query_route") or "content_qa",
            final_channel="compare_rag" if compare_route else "refusal",
            recall=recall,
            refusal_reason=refusal_reason,
            docs_returned=len(process_docs),
            question_type=qtype,
            answer_mode="refusal",
            extra={
                "refused": refusal_reason,
                "compare_status": observations.get("compare_status") or recall.get("compare_status") or "",
                "visibility_enforced": True,
                "visibility_filtered": recall["visibility_filtered"],
                "llm_query_parse_enabled": bool(getattr(runtime.config, "ENABLE_LLM_QUERY_PARSE", True)),
                "llm_parse": dict(recall.get("llm_parse") or {}),
                **observations,
            },
        ),
    }

def prepare_answer_generation_context(
    runtime: Any,
    query: str,
    qtype: str,
    recall: Dict[str, Any],
    resolved_targets: List[str],
    process_docs: List[Any],
    compare_process_groups: List[Dict[str, Any]],
    observations: Dict[str, Any],
) -> Dict[str, Any]:
    answer_mode = runtime.answer.mode_for_sources(resolved_targets, process_docs)
    if bool(observations.get("compare_degraded")) and recall.get("query_route") == "single_doc_compare":
        qtype = "compare_degraded"
        answer_mode = "compare_degraded"
    elif qtype in {"compare", "compare_degraded"} and observations.get("compare_status") == "compare_asymmetric":
        answer_mode = "compare_asymmetric"

    compare_refs: List[Dict[str, Any]] = []
    if qtype in {"compare", "compare_degraded"} and recall.get("query_route") == "multi_doc_compare" and compare_process_groups:
        evidence, compare_refs = runtime.evidence.format_compare(
            compare_process_groups,
            query,
            score_mode=recall["score_mode"],
            compare_plan=recall.get("compare_plan"),
            compare_source_statuses=observations.get("compare_source_statuses") or [],
        )
    elif qtype in {"compare", "compare_degraded"} and recall.get("query_route") == "single_doc_compare":
        evidence, compare_refs = runtime.evidence.format_single_doc_compare(
            process_docs,
            query,
            score_mode=recall["score_mode"],
            compare_plan=recall.get("compare_plan"),
        )
    else:
        evidence = runtime.evidence.format(process_docs, query, score_mode=recall["score_mode"])

    legal_clause_enumeration = runtime.answer.is_legal_clause_enumeration(query, evidence, answer_mode)
    recall["legal_clause_enumeration"] = legal_clause_enumeration
    aspect_plan = ""
    if qtype not in {"compare", "compare_degraded"} and not legal_clause_enumeration:
        aspect_plan = runtime.answer.aspect_plan(
            recall.get("evidence_query") or recall.get("retrieval_query") or query,
            process_docs,
            qfilters=recall["qfilters"],
            covered_aspects=list(observations.get("covered_aspects") or []),
            uncovered_aspects=list(observations.get("uncovered_aspects") or []),
        )

    limits = runtime.answer.limits(qtype)
    compare_source_statuses = list(observations.get("compare_source_statuses") or [])
    compare_target_count = len(
        [
            source
            for source in (resolved_targets or [])
            if runtime.common.normalize_filename(source or "")
        ]
    )
    compare_answer_refs = compare_refs if len(compare_refs) >= 2 else runtime.compare.fallback_refs_from_docs(process_docs)
    compare_status = str(observations.get("compare_status") or recall.get("compare_status") or "")
    compare_matrix_mode = bool(
        qtype in {"compare", "compare_degraded"}
        and recall.get("query_route") == "multi_doc_compare"
        and compare_source_statuses
        and compare_status != "compare_ready"
    )
    if compare_matrix_mode:
        answer_mode = "compare_matrix"

    return {
        "qtype": qtype,
        "answer_mode": answer_mode,
        "evidence": evidence,
        "aspect_plan": aspect_plan,
        "legal_clause_enumeration": legal_clause_enumeration,
        "limits": limits,
        "compare_refs": compare_refs,
        "compare_answer_refs": compare_answer_refs,
        "compare_source_statuses": compare_source_statuses,
        "compare_target_count": compare_target_count,
        "compare_matrix_mode": compare_matrix_mode,
    }

def finalize_generated_answer(
    runtime: Any,
    query: str,
    answer: str,
    qtype: str,
    answer_mode: str,
    evidence: str,
    aspect_plan: str,
    process_docs: List[Any],
    observations: Dict[str, Any],
    recall: Dict[str, Any],
    structured_answer: Optional[Dict[str, Any]],
    compare_matrix_mode: bool,
    compare_target_count: int,
    compare_answer_refs: List[Dict[str, Any]],
    compare_refs: List[Dict[str, Any]],
    refusal_answer: str,
) -> Dict[str, Any]:
    events: List[Dict[str, Any]] = []
    if (
        (not compare_matrix_mode)
        and (answer or "").strip() == refusal_answer
        and process_docs
        and str(observations.get("answer_scope") or "") in {"full", "guarded_full"}
    ):
        if qtype in {"compare", "compare_degraded"} and compare_target_count > 1:
            answer = runtime.compare.multi_doc_grounded_answer(compare_answer_refs, recall.get("compare_plan"))
        elif qtype in {"compare", "compare_degraded"}:
            answer = runtime.compare.single_doc_grounded_answer(compare_refs, recall.get("compare_plan"))
        else:
            answer = runtime.answer.related_doc_grounded_answer(process_docs)

    legal_clause_enumeration = runtime.answer.is_legal_clause_enumeration(query, evidence, answer_mode)
    if structured_answer:
        answer = runtime.answer.render_structured_markdown(structured_answer)
    elif aspect_plan and process_docs and qtype not in {"compare", "compare_degraded"} and not legal_clause_enumeration:
        answer = runtime.answer.ensure_aspect_coverage(answer, aspect_plan, process_docs)

    if answer_mode == "rag_related_doc" and evidence and "[" not in answer:
        answer = runtime.answer.related_doc_grounded_answer(process_docs)
    elif (not compare_matrix_mode) and qtype in {"compare", "compare_degraded"} and evidence and "[" not in answer and compare_target_count > 1:
        answer = runtime.compare.multi_doc_grounded_answer(compare_answer_refs, recall.get("compare_plan"))
    elif (not compare_matrix_mode) and qtype in {"compare", "compare_degraded"} and evidence and "[" not in answer:
        answer = runtime.compare.single_doc_grounded_answer(compare_refs, recall.get("compare_plan"))
    elif runtime.config.REQUIRE_EVIDENCE and evidence and "[" not in answer:
        events.append(
            {
                "event": "answer.force_refuse_no_citation",
                "query": query,
                "qtype": qtype,
                "answer_mode": answer_mode,
                "selected_docs": len(process_docs),
                "evidence_chars": len(evidence),
                "require_evidence": runtime.config.REQUIRE_EVIDENCE,
            }
        )
        if (not compare_matrix_mode) and qtype in {"compare", "compare_degraded"} and compare_target_count > 1:
            answer = runtime.compare.multi_doc_grounded_answer(compare_answer_refs, recall.get("compare_plan"))
        elif (not compare_matrix_mode) and qtype in {"compare", "compare_degraded"}:
            answer = runtime.compare.single_doc_grounded_answer(compare_refs, recall.get("compare_plan"))
        elif answer_mode in {"target_hit", "rag_related_doc"}:
            answer = runtime.answer.related_doc_grounded_answer(process_docs)
        else:
            answer = refusal_answer

    uncited_claim_lines = runtime.answer.uncited_claim_lines(answer)
    if evidence and runtime.guardrails.is_high_risk_claim_query(query) and uncited_claim_lines:
        events.append(
            {
                "event": "answer.force_refuse_high_risk_uncited_claim",
                "query": query,
                "qtype": qtype,
                "answer_mode": answer_mode,
                "uncited_claim_lines": uncited_claim_lines[:4],
            }
        )
        if (not compare_matrix_mode) and qtype in {"compare", "compare_degraded"} and compare_target_count > 1 and compare_answer_refs:
            answer = runtime.compare.multi_doc_grounded_answer(compare_answer_refs, recall.get("compare_plan"))
        elif (not compare_matrix_mode) and qtype in {"compare", "compare_degraded"} and compare_refs:
            answer = runtime.compare.single_doc_grounded_answer(compare_refs, recall.get("compare_plan"))
        elif answer_mode in {"target_hit", "rag_related_doc"} and process_docs:
            answer = runtime.answer.related_doc_grounded_answer(process_docs)
        else:
            answer = refusal_answer

    lock_prefix = str(recall.get("lock_message_prefix") or "")
    if lock_prefix and not answer.startswith(lock_prefix):
        answer = lock_prefix + answer

    return {
        "answer": answer,
        "events": events,
    }

def build_process_success_result(
    runtime: Any,
    query: str,
    user_id: str,
    answer: str,
    qtype: str,
    answer_mode: str,
    recall: Dict[str, Any],
    resolved_targets: List[str],
    process_docs: List[Any],
    display_seed_docs: List[Any],
    observations: Dict[str, Any],
    compare_source_statuses: List[Dict[str, Any]],
    structured_answer: Optional[Dict[str, Any]],
    structured_answer_origin: str,
    timings_ms: Dict[str, float],
) -> Dict[str, Any]:
    display_docs = runtime.evidence.filter_display_sources(
        display_seed_docs,
        recall["score_mode"],
        recall["qfilters"],
        resolved_targets,
        qtype,
        max_sources=10,
        target_sources=resolved_targets,
        observations=observations,
    )
    sources = runtime.answer.cited_sources(
        answer,
        process_docs,
        query,
        score_mode=recall["score_mode"],
        fallback_docs=(process_docs[:10] if process_docs else (display_docs if display_docs else display_seed_docs[:10])),
    )
    citation_protocol = runtime.answer.rewrite_citation_protocol(answer, sources, structured_answer)
    answer = citation_protocol["answer"]
    sources = citation_protocol["sources"]
    answer_refs = citation_protocol["answer_refs"]
    structured_refs = citation_protocol["structured_refs"]
    answer_maybe_truncated = runtime.answer.looks_truncated(answer)
    final_query_route = "compare_rag" if recall.get("query_route") == "multi_doc_compare" else "light_rag"
    final_channel = "compare_rag" if recall.get("query_route") == "multi_doc_compare" else "light_rag"
    response_documents = serialize_response_documents(
        runtime,
        process_docs,
        query,
        recall["score_mode"],
        limit=max(10, int(recall.get("final_n") or 10)),
    )
    return {
        "answer": answer,
        "sources": sources,
        "documents": response_documents,
        "retrieved_contexts": response_documents,
        "metadata": runtime.control.metadata(
            query=query,
            user_id=user_id,
            query_route=final_query_route,
            internal_route=recall.get("query_route") or "content_qa",
            final_channel=final_channel,
            recall={**recall, "target_sources": resolved_targets},
            docs_returned=len(process_docs),
            question_type=qtype,
            answer_mode=answer_mode,
            extra={
                "lock_mode": recall.get("lock_mode") or "",
                "lock_confidence": float(recall.get("lock_confidence") or 0.0),
                "lock_message_prefix": recall.get("lock_message_prefix") or "",
                "source_lock_kind": recall.get("source_lock_kind") or "",
                "source_resolution_trace": dict(recall.get("source_resolution_trace") or {}),
                "inherited_from_context": bool(recall.get("inherited_from_context")),
                "compare_status": observations.get("compare_status") or recall.get("compare_status") or "",
                "compare_matrix": [
                    {
                        "source": item.get("source") or "",
                        "title": item.get("title") or "",
                        "status": item.get("status") or "",
                        "presence_state": runtime.compare.matrix_presence_state(item.get("presence_state") or ""),
                        "evidence_query": item.get("evidence_query") or "",
                    }
                    for item in compare_source_statuses
                ],
                "compare_source_set": dict(recall.get("compare_source_set") or {}),
                "docs_recalled": recall["recall_k"],
                "docs_rerank_kept": len(recall["docs"]),
                "docs_final": len(process_docs),
                "rerank_used": recall["rerank_used"],
                "weak_query_expansion": recall["weak_query"],
                "early_filtered": recall["early_filtered"],
                "visibility_enforced": True,
                "visibility_filtered": recall["visibility_filtered"],
                "llm_query_parse_enabled": bool(getattr(runtime.config, "ENABLE_LLM_QUERY_PARSE", True)),
                "llm_parse": dict(recall.get("llm_parse") or {}),
                "answer_refs": answer_refs,
                "answer_citation_refs": answer_refs,
                "structured_refs": structured_refs,
                "citation_ref_map": citation_protocol.get("citation_ref_map") or {},
                "phantom_citation_refs": citation_protocol.get("phantom_citation_refs") or [],
                "answer_maybe_truncated": answer_maybe_truncated,
                "answer_format": "structured_json" if structured_answer else "markdown_text",
                "legal_clause_enumeration": bool(recall.get("legal_clause_enumeration")),
                "structured_answer_origin": structured_answer_origin,
                "structured_answer": structured_answer or {},
                "server_timing_ms": timings_ms,
                **observations,
            },
        ),
        "display_docs_count": len(display_docs) if display_docs else min(len(display_seed_docs), 10),
    }
