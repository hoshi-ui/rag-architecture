import json
import json
import re
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

from app.core.legal_subjects import normalize_subject_terms


INTENT_CLASSIFIER_ROUTES = {
    "content_qa",
    "business_topic_qa",
    "open_regulation_qa",
    "explicit_doc_reference",
    "explicit_regulation_reference",
    "weak_title_reference",
    "version_switch",
    "existence",
    "visibility_probe",
    "document_clarification",
    "refusal",
    "multi_doc_compare",
    "single_doc_compare",
    "compare_clarification",
}

INTENT_CLASSIFIER_QTYPES = {
    "definition",
    "summary",
    "howto",
    "compare",
    "screening",
    "single_doc_extract",
    "regulation_execution",
    "document_state",
    "existence",
    "other",
}

_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


def question_type(payload: Dict[str, Any]) -> str:
    value = str(payload.get("question_type") or payload.get("qtype") or payload.get("intent") or "").strip()
    return value if value in INTENT_CLASSIFIER_QTYPES else value


def route(payload: Dict[str, Any]) -> str:
    value = str(payload.get("route") or payload.get("query_route") or "").strip()
    return value if value in INTENT_CLASSIFIER_ROUTES else value


def action(payload: Dict[str, Any]) -> str:
    return str(payload.get("answer_action") or payload.get("action") or "").strip()


def is_comparison(payload: Dict[str, Any]) -> Optional[bool]:
    if "is_comparison" in payload:
        return bool(payload.get("is_comparison"))
    classifier_route = route(payload)
    if classifier_route:
        return classifier_route in {"multi_doc_compare", "single_doc_compare", "compare_clarification", "compare"}
    return None


def is_reliable(payload: Dict[str, Any], min_confidence: float = 0.72) -> bool:
    if not payload:
        return False
    if "reliable" in payload:
        return bool(payload.get("reliable"))
    if "confidence" in payload:
        try:
            return float(payload.get("confidence") or 0.0) >= float(min_confidence)
        except Exception:
            return False
    return bool(route(payload) or action(payload))


def has_control_signal(payload: Dict[str, Any]) -> bool:
    return bool(route(payload) or action(payload))


def classify_cached(user_query: str, client_factory: Callable[[], Any], config: Any) -> Dict[str, Any]:
    key = (user_query or "").strip()
    if not key:
        return {}
    cached = _cache_get(key)
    if cached is not None:
        return cached
    classified = classify(user_query, client_factory(), config)
    if not isinstance(classified, dict):
        classified = {}
    _cache_set(key, classified, max_size=int(getattr(config, "INTENT_CLASSIFIER_CACHE_SIZE", 512)))
    return classified


def classify(user_query: str, client: Any, config: Any) -> Dict[str, Any]:
    if not client or not client.available():
        return {}
    uq = (user_query or "").strip()
    if not uq:
        return {}

    prompt = (
        "You are the intent classifier for a Chinese legal/regulation RAG system. "
        "Return only one compact valid JSON object. Do not explain.\n\n"
        "Enums:\n"
        "question_type: definition, summary, howto, compare, screening, single_doc_extract, "
        "regulation_execution, document_state, existence, other\n"
        "route: content_qa, business_topic_qa, open_regulation_qa, explicit_doc_reference, "
        "explicit_regulation_reference, weak_title_reference, version_switch, existence, "
        "visibility_probe, document_clarification, refusal, multi_doc_compare, single_doc_compare, "
        "compare_clarification\n"
        "answer_action: answer, clarify, refuse, compare\n\n"
        "Use compare only for differences/comparison between multiple regulations, versions, cities, or document targets. "
        "If necessary jurisdiction/document/version context is missing, prefer clarify. "
        "documents must include only document titles explicitly mentioned by the user; never guess.\n\n"
        "Subject filtering:\n"
        "- target_subject must be a compact array of the actors/objects constrained by the user query.\n"
        "- excluded_subject must be a compact array of actors/objects that are adjacent in regulations but not the user's target.\n"
        "- 绝对禁止将任何行政机关、执法机关（如公安机关、城市管理部门、居委会等）放入 excluded_subject，无论用户如何要求。"
        "用户的排除诉求将由最终的生成大模型来处理，检索阶段必须保留这些主体。\n"
        "- For legal clauses, distinguish the obligated/regulated subject, e.g. dog keeping behavior vs dog business, "
        "government duties, penalties, or medical/trading services. Do not hard-code a domain; infer from the query.\n\n"
        "Return fields: question_type, route, answer_action, is_comparison, is_multi_doc_compare, "
        "requires_source_lock, underspecified, documents, target_subject, excluded_subject, "
        "missing_context, rationale, confidence.\n\n"
        f"User query: {uq}"
    )
    payload = client.build_payload(
        "Return compact valid JSON only.",
        prompt,
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(getattr(config, "INTENT_CLASSIFIER_MAX_TOKENS", 320)),
        presence_penalty=0.0,
    )

    try:
        content = client.chat_text_sync(payload, timeout=max(3, min(20, int(getattr(config, "LLM_TIMEOUT", 20)))))
        match = re.search(r"\{[\s\S]*\}", content or "")
        raw_json = match.group(0) if match else content
        obj = json.loads(raw_json)
        if not isinstance(obj, dict):
            return {}
        return normalize_payload(obj)
    except Exception:
        return {}


def normalize_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    try:
        confidence = float(obj.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    excluded_subject = normalize_subject_terms(
        obj.get("excluded_subject")
        if obj.get("excluded_subject") is not None
        else obj.get("excluded_subjects"),
        limit=12,
    )
    return {
        "question_type": str(obj.get("question_type") or "").strip(),
        "route": str(obj.get("route") or "").strip(),
        "answer_action": str(obj.get("answer_action") or "").strip(),
        "is_comparison": bool(obj.get("is_comparison")),
        "is_multi_doc_compare": bool(obj.get("is_multi_doc_compare")),
        "requires_source_lock": bool(obj.get("requires_source_lock")),
        "underspecified": bool(obj.get("underspecified")),
        "documents": _clean_string_list(obj.get("documents"), limit=6),
        "target_subject": normalize_subject_terms(
            obj.get("target_subject")
            if obj.get("target_subject") is not None
            else obj.get("target_subjects"),
            limit=12,
        ),
        "excluded_subject": _drop_enforcement_agency_subjects(excluded_subject),
        "missing_context": _clean_string_list(obj.get("missing_context"), limit=6),
        "rationale": str(obj.get("rationale") or "").strip()[:240],
        "confidence": max(0.0, min(confidence, 1.0)),
    }


def _drop_enforcement_agency_subjects(values: List[str]) -> List[str]:
    agency_markers = (
        "行政机关",
        "执法机关",
        "公安机关",
        "公安",
        "城市管理部门",
        "城市管理",
        "城管",
        "居委会",
        "居民委员会",
        "监管部门",
        "主管部门",
        "执法部门",
        "管理部门",
        "人民政府",
        "派出所",
        "街道办",
        "办事处",
        "市场监督",
        "住建",
    )
    agency_suffixes = ("机关", "部门", "委员会", "人民政府", "公安局", "管理局", "执法局", "监督局")
    out: List[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if not text:
            continue
        if any(marker in text for marker in agency_markers):
            continue
        if any(text.endswith(suffix) for suffix in agency_suffixes):
            continue
        out.append(text)
    return out


def _clean_string_list(values: Any, limit: int = 6) -> List[str]:
    if isinstance(values, str):
        raw_values = re.split(r"[,，;；、\n]+", values)
    elif isinstance(values, (list, tuple, set)):
        raw_values = list(values)
    else:
        return []
    out: List[str] = []
    for item in raw_values:
        text = str(item or "").strip()
        if not text:
            continue
        text = re.sub(r"\s+", " ", text)
        if text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    if key not in _CACHE:
        return None
    value = _CACHE.pop(key)
    _CACHE[key] = value
    return dict(value)


def _cache_set(key: str, value: Dict[str, Any], max_size: int) -> None:
    if not key:
        return
    _CACHE[key] = dict(value or {})
    while len(_CACHE) > max(1, int(max_size)):
        _CACHE.popitem(last=False)
