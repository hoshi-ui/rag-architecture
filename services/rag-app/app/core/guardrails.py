import re
from typing import Any, Dict, Iterable, List, Optional


PROMPT_INJECTION_TRIGGERS = [
    "ignore previous",
    "system prompt",
    "developer message",
    "reveal",
    "api key",
    "token",
    "密钥",
    "系统提示词",
    "开发者消息",
    "忽略之前",
    "越狱",
    "jailbreak",
]


LEGAL_ACTION_TERMS = [
    "应当",
    "不得",
    "可以",
    "必须",
    "禁止",
    "负责",
    "要求",
    "规定",
    "按照",
    "依据",
    "履行",
    "办理",
    "申请",
    "审批",
    "备案",
    "处罚",
    "整改",
    "监督",
    "管理",
    "shall",
    "must",
    "require",
    "prohibit",
]


def is_chinese_query(query: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", query or ""))


def blocked_reason(query: str) -> str:
    normalized = (query or "").lower()
    for trigger in PROMPT_INJECTION_TRIGGERS:
        if trigger in normalized:
            return "blocked_prompt_injection"
    return ""


def invalid_query_message(reason: str) -> str:
    messages = {
        "query_too_short": "问题太短，请补充更明确的法规、文件或事项。",
        "query_too_long": "问题过长，请拆成一个更具体的问题后重试。",
        "blocked_query": "该问题无法处理，请换一种合规、明确的问法。",
        "blocked_prompt_injection": "该问题包含不适合处理的指令内容。",
        "retrieval_error": "检索过程中出现异常，请稍后重试。",
        "evidence_insufficient": "当前证据不足，暂不能基于已检索资料回答。",
    }
    return messages.get(reason or "", "当前问题暂不能处理，请补充更明确的信息。")


def query_has_repeated_noise(query: str, normalize_query) -> bool:
    normalized = normalize_query(query)
    compact = re.sub(r"\s+", "", normalized)
    if not compact:
        return False
    if len(compact) >= 4 and len(set(compact)) == 1:
        return True
    if re.findall(r"(.)\1{3,}", compact):
        return True
    meaningful = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", compact)
    if meaningful and len(set(meaningful)) <= 2 and len(compact) >= 6:
        return True
    return False


def text_has_legal_action_signal(text: str, normalize_query) -> bool:
    content = normalize_query(text)
    if not content:
        return False
    if any(term in content for term in LEGAL_ACTION_TERMS):
        return True
    return bool(re.search(r"第[一二三四五六七八九十百千万0-9]+[条款项章]", content))


def build_evidence_refusal_answer(
    query: str,
    refusal_reason: str,
    observations: Optional[Dict[str, Any]] = None,
    normalize_query=None,
) -> str:
    obs = observations or {}
    normalizer = normalize_query or (lambda value: (value or "").strip())
    if refusal_reason == "core_entity_not_covered":
        terms: List[str] = [
            normalizer(str(term or ""))
            for term in (obs.get("core_entity_terms") or [])
            if normalizer(str(term or ""))
        ]
        if terms:
            return f"未在召回证据中找到与{'、'.join(terms[:3])}直接相关的内容，暂不能基于当前资料回答。"
        return "未在召回证据中找到问题核心实体的直接依据，暂不能基于当前资料回答。"
    if refusal_reason in {"wrong_source", "off_topic_in_document"}:
        return "当前召回内容与问题目标不匹配，暂不能基于当前资料回答。"
    return "当前证据不足，暂不能基于当前资料回答。"


def is_high_risk_claim_query(query: str, markers: Iterable[str], normalize_query) -> bool:
    normalized = normalize_query(query)
    return any(marker in normalized for marker in markers)


def query_static_quality_state(query: str, *, normalize_query, max_query_chars: int = 2000) -> Dict[str, str]:
    normalized = normalize_query(query)
    if len(normalized) > int(max_query_chars):
        return {"reason": "query_too_long", "quality": "invalid", "tier": "static"}
    return {"reason": "", "quality": "valid", "tier": "static"}


def query_deep_quality_state(
    query: str,
    *,
    normalize_query,
    llm_parse: Optional[Dict[str, Any]] = None,
    source_resolution: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    return {"reason": "", "quality": "valid", "tier": "deep"}
