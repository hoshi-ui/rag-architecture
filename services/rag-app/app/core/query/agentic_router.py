import json
import re
from typing import Any, Dict, List, Optional

from app.core.legal_intent import classify_query_intent_fallback, legal_intent_from_payload


AGENTIC_COMPARE_ROUTES = {
    "multi_doc_compare",
    "single_doc_compare",
    "compare_clarification",
    "compare_target_not_found",
    "compare_targets_not_found",
}


ABSTRACT_ACTION_TERMS = {
    "\u4e49\u52a1": ["\u5e94\u5f53", "\u5fc5\u987b", "\u4e0d\u5f97", "\u7981\u6b62", "\u8d1f\u8d23"],
    "\u540e\u679c": ["\u8d23\u4ee4\u6539\u6b63", "\u7f5a\u6b3e", "\u6ca1\u6536", "\u5904\u7f5a"],
    "\u5904\u7406": ["\u8d23\u4ee4\u6539\u6b63", "\u7f5a\u6b3e", "\u6ca1\u6536", "\u5904\u7f5a"],
    "\u8d23\u4efb": ["\u8d23\u4ee4\u6539\u6b63", "\u7f5a\u6b3e", "\u5904\u7f5a", "\u6cd5\u5f8b\u8d23\u4efb"],
    "\u5904\u7f5a": ["\u8d23\u4ee4\u6539\u6b63", "\u7f5a\u6b3e", "\u6ca1\u6536", "\u540a\u9500"],
    "\u8fdd\u6cd5": ["\u8d23\u4ee4\u6539\u6b63", "\u7f5a\u6b3e", "\u6ca1\u6536", "\u5904\u7f5a"],
}


def _clean_text(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value or "").split()).strip()
    return text[:limit]


def _clean_string_list(values: Any, limit: int = 8) -> List[str]:
    if isinstance(values, str):
        raw_values = re.split(r"[,，;；、\n]+", values)
    elif isinstance(values, (list, tuple, set)):
        raw_values = list(values)
    else:
        return []
    out: List[str] = []
    for item in raw_values:
        text = _clean_text(item)
        if text and text not in out:
            out.append(text)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _expand_abstract_action_terms(query: str, limit: int = 8) -> str:
    text = _clean_text(query)
    if not text:
        return ""
    additions: List[str] = []
    for abstract, action_terms in ABSTRACT_ACTION_TERMS.items():
        if abstract not in text:
            continue
        for term in action_terms:
            if term not in text and term not in additions:
                additions.append(term)
                if len(additions) >= max(1, int(limit)):
                    return " ".join([text] + additions).strip()
    return " ".join([text] + additions).strip() if additions else text


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, flags=re.I)
    candidate = fenced.group(1) if fenced else ""
    if not candidate:
        match = re.search(r"\{[\s\S]*\}", raw)
        candidate = match.group(0) if match else ""
    if not candidate:
        return {}
    try:
        parsed = json.loads(candidate)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_subquery(item: Any) -> Dict[str, str]:
    if not isinstance(item, dict):
        return {}
    source = _clean_text(item.get("source") or item.get("document") or item.get("title"), limit=160)
    raw_query = _clean_text(
        item.get("raw_text_query")
        or item.get("retrieval_query")
        or item.get("query")
        or item.get("evidence_query"),
        limit=240,
    )
    section_query = _clean_text(item.get("section_query") or raw_query, limit=240)
    doc_prior_query = _clean_text(item.get("doc_prior_query") or source or raw_query, limit=240)
    raw_query = _expand_abstract_action_terms(raw_query)
    section_query = _expand_abstract_action_terms(section_query)
    if not source and not raw_query:
        return {}
    return {
        "source": source,
        "raw_text_query": raw_query or source,
        "section_query": section_query or raw_query or source,
        "doc_prior_query": doc_prior_query or source or raw_query,
    }


def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    try:
        confidence = float(payload.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    sub_queries = [
        item
        for item in (_normalize_subquery(entry) for entry in (payload.get("sub_queries") or []))
        if item
    ][:8]
    documents = _clean_string_list(payload.get("documents") or payload.get("document_titles"), limit=8)
    for item in sub_queries:
        source = item.get("source") or ""
        if source and source not in documents:
            documents.append(source)
    route = _clean_text(payload.get("route") or payload.get("query_route"), limit=64)
    is_comparison = bool(payload.get("is_comparison") or route in AGENTIC_COMPARE_ROUTES)
    is_multi_doc = bool(payload.get("is_multi_doc_compare") or route == "multi_doc_compare" or len(documents) >= 2)
    query_intent = legal_intent_from_payload(payload)
    return {
        "route": route,
        "query_intent": query_intent,
        "question_type": _clean_text(payload.get("question_type") or payload.get("qtype"), limit=64),
        "is_comparison": is_comparison,
        "is_multi_doc_compare": is_multi_doc,
        "documents": documents[:8],
        "common_aspects": _clean_string_list(payload.get("common_aspects") or payload.get("aspects"), limit=8),
        "sub_queries": sub_queries,
        "missing_targets": _clean_string_list(payload.get("missing_targets"), limit=8),
        "requires_clarification": bool(payload.get("requires_clarification")),
        "rationale": _clean_text(payload.get("rationale") or payload.get("reason"), limit=240),
        "confidence": max(0.0, min(confidence, 1.0)),
    }


async def route_query(runtime: Any, query: str) -> Dict[str, Any]:
    if not bool(getattr(runtime.config, "ENABLE_AGENTIC_ROUTER", True)):
        return {"used": False, "reason": "disabled"}
    client = getattr(runtime, "llm_client", None)
    if not client or not client.available():
        return {"used": False, "reason": "llm_unavailable"}
    user_query = str(query or "").strip()
    if not user_query:
        return {"used": False, "reason": "empty_query"}

    system_prompt = (
        "你是法规 RAG 的 Agentic Router，只做意图拆解和检索计划，不回答用户问题。"
        "只输出一个紧凑合法 JSON 对象，不要解释。\n"
        "当问题是多文档、多法规、版本差异、地区差异或对比问题时，输出并发子查询。"
        "不要通过删除“对比/比较/分别”等词做字符串清洗；要理解用户要比较的对象和共同维度。\n"
        "documents 只能写用户明确提到的法规/文件/地区目标，不要猜测库中不存在的文件。\n"
        "sub_queries 每项必须包含 source、raw_text_query、section_query、doc_prior_query。"
        "raw_text_query 写该 source 内要检索的实质问题；section_query 写条款/章节定位词；"
        "doc_prior_query 写用于锁定该文档的标题或主题。\n"
        "\u5982\u679c sub_queries \u4e2d\u5305\u542b\u62bd\u8c61\u6cd5\u5f8b\u6982\u5ff5\uff08\u5982\u4e49\u52a1\u3001\u540e\u679c\u3001\u5904\u7406\uff09\uff0c"
        "\u8bf7\u5728 raw_text_query \u548c section_query \u540e\u8865\u5145\u5bf9\u5e94\u7684\u5177\u4f53\u52a8\u4f5c\u8bcd"
        "\uff08\u5982\u5e94\u5f53\u3001\u5fc5\u987b\u3001\u7f5a\u6b3e\u3001\u8d23\u4ee4\uff09\uff0c\u4e0d\u8981\u628a\u8fd9\u4e9b\u52a8\u4f5c\u8bcd\u5199\u5165 source \u6216 doc_prior_query\u3002\n"
        "JSON schema: {"
        "\"route\":\"content_qa|multi_doc_compare|single_doc_compare|compare_clarification\","
        "\"query_intent\":\"定义与范围|职责与权限|程序与条件|法律责任|权利义务|其他\","
        "\"question_type\":\"other|compare|summary|howto|single_doc_extract\","
        "\"is_comparison\":true,"
        "\"is_multi_doc_compare\":true,"
        "\"documents\":[\"明确提到的文档或法规\"],"
        "\"common_aspects\":[\"对比维度\"],"
        "\"sub_queries\":[{\"source\":\"文档名\",\"raw_text_query\":\"检索问题\","
        "\"section_query\":\"章节定位词\",\"doc_prior_query\":\"文档锁定词\"}],"
        "\"missing_targets\":[],\"requires_clarification\":false,"
        "\"rationale\":\"简短原因\",\"confidence\":0.0}"
    )
    payload = client.build_payload(
        system_prompt,
        f"用户问题：{user_query}",
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(getattr(runtime.config, "AGENTIC_ROUTER_MAX_TOKENS", 520)),
        presence_penalty=0.0,
    )
    try:
        content = await client.chat_text(
            payload,
            timeout=max(1, int(getattr(runtime.config, "AGENTIC_ROUTER_TIMEOUT", 8))),
        )
    except Exception as exc:
        return {"used": False, "reason": "router_error", "error": type(exc).__name__}
    normalized = normalize_payload(_extract_json_object(content))
    if not normalized:
        return {"used": False, "reason": "invalid_json"}
    if not normalized.get("query_intent"):
        normalized["query_intent"] = classify_query_intent_fallback(user_query)
    return {"used": True, "reason": "llm_json", **normalized}


def build_compare_resolution(runtime: Any, query: str, route: Dict[str, Any]) -> Dict[str, Any]:
    if not route.get("used") or not route.get("is_comparison"):
        return {}
    min_confidence = float(getattr(runtime.config, "AGENTIC_ROUTER_MIN_CONFIDENCE", 0.62))
    if float(route.get("confidence") or 0.0) < min_confidence:
        return {}

    normalized_sources: List[str] = []
    subject_matches: List[Dict[str, Any]] = []
    def source_key(value: Any) -> str:
        normalized = runtime.common.normalize_query(value)
        return normalized.strip("《》<>\"'“”‘’")

    subqueries_by_text: Dict[str, Dict[str, str]] = {}
    for item in route.get("sub_queries") or []:
        source_text = item.get("source") or ""
        if not source_text:
            continue
        subqueries_by_text[source_text] = item
        subqueries_by_text[source_key(source_text)] = item
    target_texts = list(route.get("documents") or [])
    for item in route.get("sub_queries") or []:
        source_text = item.get("source") or ""
        if source_text and source_text not in target_texts:
            target_texts.append(source_text)

    source_subqueries: Dict[str, Dict[str, str]] = {}
    missing_targets: List[str] = []
    rejected_targets: List[Dict[str, Any]] = []

    def validation_query(target: str) -> str:
        subquery = subqueries_by_text.get(target) or subqueries_by_text.get(source_key(target)) or {}
        parts = [
            query,
            target,
            subquery.get("raw_text_query") or "",
            subquery.get("section_query") or "",
            subquery.get("doc_prior_query") or "",
        ]
        return runtime.common.normalize_query(" ".join(str(part or "") for part in parts if str(part or "").strip()))

    def validate_source(target: str, source: str, *, prior: float = 0.0, match_kind: str = "") -> Dict[str, Any]:
        validator = getattr(getattr(runtime, "source", None), "validate_source_lock_candidate", None)
        if not callable(validator):
            return {"accepted": True}
        try:
            return validator(
                validation_query(target),
                target,
                source,
                prior=prior,
                match_kind=match_kind,
            )
        except Exception:
            return {"accepted": True, "error": "validator_exception"}

    def resolve_source(target: str) -> Dict[str, Any]:
        match = runtime.compare.resolve_subject_source(target)
        source = runtime.common.normalize_filename(match.get("source") or "")
        if source:
            latest_source = ""
            latest_fn = getattr(getattr(runtime, "source", None), "latest_effective_equivalent_source", None)
            if callable(latest_fn):
                try:
                    latest_source = runtime.common.normalize_filename(latest_fn(source) or "")
                except Exception:
                    latest_source = ""
            if latest_source:
                match = {**match, "source": latest_source}
                source = latest_source
            validation = validate_source(
                target,
                source,
                prior=float(match.get("prior") or 0.0),
                match_kind=str(match.get("match_kind") or ""),
            )
            if not validation.get("accepted"):
                rejected_targets.append({"target": target, "source": source, "validation": validation})
                return {**match, "source": "", "source_lock_validation": validation}
            match = {**match, "source_lock_validation": validation}
            return match

        subquery = subqueries_by_text.get(target) or subqueries_by_text.get(source_key(target)) or {}
        probe_texts: List[str] = []
        for value in [
            target,
            subquery.get("source") or "",
            subquery.get("doc_prior_query") or "",
            f"{subquery.get('source') or ''} {subquery.get('doc_prior_query') or ''}",
            f"{subquery.get('source') or ''} {subquery.get('raw_text_query') or ''}",
            validation_query(target),
        ]:
            text = runtime.common.normalize_query(value)
            if text and text not in probe_texts:
                probe_texts.append(text)

        title_candidates: List[str] = []
        for probe in probe_texts:
            for candidate in list(runtime.source.extract_title_candidates(probe, limit=5) or []):
                normalized_candidate = runtime.common.normalize_filename(candidate or "")
                if normalized_candidate and normalized_candidate not in title_candidates:
                    title_candidates.append(normalized_candidate)
        prefer_latest = getattr(getattr(runtime, "source", None), "prefer_latest_effective_sources", None)
        if callable(prefer_latest):
            try:
                title_candidates = list(prefer_latest(title_candidates, limit=3) or title_candidates)
            except Exception:
                pass
        for candidate in title_candidates:
            normalized = runtime.common.normalize_filename(candidate or "")
            if normalized:
                validation = validate_source(target, normalized, prior=0.92, match_kind="agentic_title_candidate")
                if not validation.get("accepted"):
                    rejected_targets.append({"target": target, "source": normalized, "validation": validation})
                    continue
                return {
                    "subject": target,
                    "source": normalized,
                    "match_kind": "agentic_title_candidate",
                    "doc_like": True,
                    "prior": 0.92,
                    "source_lock_validation": validation,
                }
        strong_matches: List[Dict[str, Any]] = []
        for probe in probe_texts:
            for candidate in runtime.source.strong_title_source_matches(probe, limit=5) or []:
                source_value = runtime.common.normalize_filename((candidate or {}).get("source") or "")
                if source_value and not any(runtime.common.normalize_filename((item or {}).get("source") or "") == source_value for item in strong_matches):
                    strong_matches.append(candidate)
        for candidate in strong_matches:
            normalized = runtime.common.normalize_filename((candidate or {}).get("source") or "")
            if normalized:
                match_kind = (candidate or {}).get("match_kind") or "agentic_strong_title"
                prior = float((candidate or {}).get("score") or 0.88)
                validation = validate_source(target, normalized, prior=prior, match_kind=match_kind)
                if not validation.get("accepted"):
                    rejected_targets.append({"target": target, "source": normalized, "validation": validation})
                    continue
                return {
                    "subject": target,
                    "source": normalized,
                    "match_kind": match_kind,
                    "doc_like": True,
                    "prior": prior,
                    "source_lock_validation": validation,
                }
        return match

    for target in target_texts[:8]:
        match = resolve_source(target)
        source = runtime.common.normalize_filename(match.get("source") or "")
        subject_matches.append(
            {
                "subject": target,
                "source": source,
                "match_kind": match.get("match_kind") or "",
                "doc_like": bool(match.get("doc_like")),
                "prior": float(match.get("prior") or 0.0),
                "source_lock_validation": match.get("source_lock_validation") or {},
            }
        )
        if not source:
            missing_targets.append(target)
            continue
        if source not in normalized_sources:
            normalized_sources.append(source)
        subquery = subqueries_by_text.get(target) or subqueries_by_text.get(source_key(target)) or {}
        title = runtime.source.display_title(source) or target
        raw_text_query = runtime.common.normalize_query(subquery.get("raw_text_query") or " ".join([title] + list(route.get("common_aspects") or [])))
        section_query = runtime.common.normalize_query(subquery.get("section_query") or raw_text_query)
        doc_prior_query = runtime.common.normalize_query(subquery.get("doc_prior_query") or title)
        source_subqueries[source] = {
            "raw_text_query": raw_text_query or title,
            "section_query": section_query or raw_text_query or title,
            "doc_prior_query": doc_prior_query or title,
        }

    if route.get("is_multi_doc_compare") or len(target_texts) >= 2:
        if len(normalized_sources) >= 2:
            compare_route = "multi_doc_compare"
            resolved = not bool(missing_targets)
            required = False
            reason = "agentic_router_plan_ready" if resolved else "agentic_router_target_incomplete"
            compare_status = "plan_ready" if resolved else "target_incomplete"
        else:
            return {}
    elif len(normalized_sources) == 1:
        has_explicit_compare_intent = False
        has_intent_fn = getattr(getattr(runtime, "compare", None), "has_intent", None)
        if callable(has_intent_fn):
            try:
                has_explicit_compare_intent = bool(has_intent_fn(query))
            except Exception:
                has_explicit_compare_intent = False
        if not has_explicit_compare_intent:
            return {}
        compare_route = "single_doc_compare"
        resolved = True
        required = False
        reason = "agentic_router_single_doc_plan_ready"
        compare_status = "single_doc_ready"
    else:
        return {}

    common_aspects = list(route.get("common_aspects") or [])
    compare_plan = {
        "raw_query": query,
        "route": compare_route,
        "reason": reason,
        "required": required,
        "resolved": resolved,
        "is_compare": True,
        "subjects": target_texts[:8],
        "subject_matches": subject_matches,
        "sources": normalized_sources,
        "doc_like_subjects": target_texts[:8],
        "missing_doc_targets": missing_targets,
        "rejected_doc_targets": rejected_targets,
        "common_aspects": common_aspects,
        "topic_pair": [],
        "canonical_aspects": common_aspects[:4],
        "expanded_aspects": common_aspects[:8],
        "source_subqueries": source_subqueries,
        "target_text": "、".join(missing_targets[:3] or target_texts[:3]),
        "clarification": "",
        "strip_title_mentions": False,
        "compare_status": compare_status,
        "agentic_router": {
            "confidence": float(route.get("confidence") or 0.0),
            "rationale": route.get("rationale") or "",
            "query_intent": route.get("query_intent") or "",
            "sub_queries": list(route.get("sub_queries") or []),
        },
    }
    compare_plan["compare_plan"] = {
        "raw_query": query,
        "route": compare_route,
        "reason": reason,
        "required": required,
        "resolved": resolved,
        "subjects": [
            {
                "raw_text": item,
                "clean_text": item,
                "doc_like": True,
                "source": next((m["source"] for m in subject_matches if m["subject"] == item), ""),
            }
            for item in target_texts[:8]
        ],
        "matched_sources": normalized_sources,
        "missing_targets": missing_targets,
        "rejected_targets": rejected_targets,
        "doc_like_subjects": target_texts[:8],
        "common_aspects": common_aspects,
        "canonical_aspects": common_aspects[:4],
        "expanded_aspects": common_aspects[:8],
        "source_subqueries": source_subqueries,
        "compare_status": compare_status,
        "agentic_router": compare_plan["agentic_router"],
    }
    return compare_plan
