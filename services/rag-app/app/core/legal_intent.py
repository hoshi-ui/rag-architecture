"""Coarse legal intent labels used by retrieval and reranking."""

from __future__ import annotations

import re
from typing import Any, Dict


LEGAL_INTENTS = {
    "定义与范围",
    "职责与权限",
    "程序与条件",
    "法律责任",
    "权利义务",
    "其他",
}


_ALIASES = {
    "definition_scope": "定义与范围",
    "definition": "定义与范围",
    "scope": "定义与范围",
    "定义": "定义与范围",
    "范围": "定义与范围",
    "适用范围": "定义与范围",
    "职责": "职责与权限",
    "权限": "职责与权限",
    "职责权限": "职责与权限",
    "职责与权限": "职责与权限",
    "duty_authority": "职责与权限",
    "authority": "职责与权限",
    "procedure_condition": "程序与条件",
    "procedure": "程序与条件",
    "condition": "程序与条件",
    "程序": "程序与条件",
    "条件": "程序与条件",
    "程序与条件": "程序与条件",
    "legal_responsibility": "法律责任",
    "responsibility": "法律责任",
    "penalty": "法律责任",
    "liability": "法律责任",
    "处罚": "法律责任",
    "罚则": "法律责任",
    "法律责任": "法律责任",
    "obligation": "权利义务",
    "rights_obligations": "权利义务",
    "义务": "权利义务",
    "权利义务": "权利义务",
    "其他": "其他",
    "other": "其他",
}


def normalize_legal_intent(value: Any) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    if text in LEGAL_INTENTS:
        return text
    folded = text.lower().replace("-", "_").replace(" ", "_")
    if folded in _ALIASES:
        return _ALIASES[folded]
    compact = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return _ALIASES.get(compact, "")


def classify_query_intent_fallback(query: str) -> str:
    text = str(query or "")
    if not text:
        return "其他"
    if any(term in text for term in ("法律责任", "罚则", "处罚", "罚款", "没收", "责令", "吊销", "违法后果")):
        return "法律责任"
    if any(term in text for term in ("职责", "权限", "职权", "负责", "主管", "监督管理", "分工")):
        return "职责与权限"
    if any(term in text for term in ("程序", "流程", "申请", "审查", "办理", "登记", "期限", "条件", "不符合")):
        return "程序与条件"
    if any(term in text for term in ("定义", "含义", "所称", "适用", "范围", "包括", "不包括")):
        return "定义与范围"
    if any(term in text for term in ("义务", "权利", "应当", "不得", "禁止", "可以")):
        return "权利义务"
    return "其他"


def legal_intent_from_payload(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("query_intent", "legal_intent", "intent", "macro_intent"):
        value = normalize_legal_intent(payload.get(key))
        if value:
            return value
    return ""
