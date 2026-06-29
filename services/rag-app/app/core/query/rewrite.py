from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


LEGAL_ABSTRACTION_TERMS: Dict[str, List[str]] = {
    "义务": ["应当", "必须", "不得", "禁止", "负责", "备案", "登记", "申报"],
    "职责": ["负责", "主管", "监督管理", "组织", "指导", "协同"],
    "责任": ["责令改正", "罚款", "没收", "吊销", "处罚", "法律责任"],
    "后果": ["责令改正", "罚款", "没收", "吊销", "处罚", "法律责任"],
    "处罚": ["责令改正", "罚款", "没收", "吊销", "行政处罚"],
    "处罚方式": ["责令改正", "罚款", "没收", "吊销", "行政处罚"],
    "罚款幅度": ["罚款", "以上", "以下", "处"],
    "适用范围": ["适用", "本条例所称", "不包括", "范围"],
    "管理活动": ["管理", "监督", "备案", "登记", "检查"],
    "主体": ["人民政府", "主管部门", "行政机关", "执法机关", "有关部门"],
}

COMMON_COMPARE_TERMS = ("比较", "对比", "区别", "差异", "异同", "分别", "各自", "三类", "多个")


def dedupe(values: List[str], limit: int = 12) -> List[str]:
    out: List[str] = []
    for value in values or []:
        item = " ".join(str(value or "").split()).strip()
        if item and item not in out:
            out.append(item)
        if len(out) >= max(1, int(limit)):
            break
    return out


def parse_json_object(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def unpack_legal_abstractions_fallback(query: str, limit: int = 12) -> List[str]:
    text = str(query or "")
    terms: List[str] = []
    for key, expansions in LEGAL_ABSTRACTION_TERMS.items():
        if key in text:
            terms.extend(expansions)
    if "违法" in text and not any(term in terms for term in ("责令改正", "罚款")):
        terms.extend(["责令改正", "罚款", "没收", "处罚"])
    if "办理" in text or "登记" in text:
        terms.extend(["登记", "备案", "补办", "逾期"])
    return dedupe(terms, limit=limit)


def expand_query_with_terms(query: str, terms: List[str], limit: int = 12) -> str:
    base = " ".join(str(query or "").split()).strip()
    normalized_base = base
    additions = [term for term in dedupe(terms, limit=limit) if term and term not in normalized_base]
    if not additions:
        return base
    return " ".join([base] + additions).strip()


def should_decompose_query(
    query: str,
    *,
    query_route: str = "",
    is_comparison: bool = False,
    is_comparison_hint: bool = False,
) -> bool:
    text = str(query or "")
    if query_route in {"compare", "multi_doc_compare"} or is_comparison or is_comparison_hint:
        return True
    if any(term in text for term in COMMON_COMPARE_TERMS) and ("、" in text or "和" in text or "与" in text):
        return True
    if re.search(r"[一二三四五六七八九十0-9]+类", text) and ("、" in text or "分别" in text):
        return True
    return False


def normalize_subquery_items(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    items = payload.get("sub_queries")
    if not isinstance(items, list):
        return []
    out: List[Dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        source = " ".join(str(item.get("source") or "").split()).strip()
        query = " ".join(str(item.get("query") or "").split()).strip()
        if source or query:
            out.append({"source": source, "query": query})
    return out[:8]


def _clean_subject_segment(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^(比较|对比|分别比较|请比较|请对比)", "", text).strip()
    text = re.sub(r"(三类|两类|多类)?违法行为的?(处罚方式|罚款幅度|处罚|后果|责任).*$", "", text).strip()
    text = re.sub(r"(中的|中与).*$", "", text).strip()
    return text.strip(" ，,、；;")


def decompose_query_fallback(query: str, common_terms: Optional[List[str]] = None) -> List[Dict[str, str]]:
    text = str(query or "").strip()
    if not text:
        return []
    common = list(common_terms or [])
    if "处罚" in text or "违法" in text or "罚款" in text:
        common.extend(["处罚方式", "罚款"])
    if "义务" in text:
        common.extend(["义务", "应当", "不得"])
    common_suffix = " ".join(dedupe(common, limit=5))

    body = re.sub(r"^(比较|对比|分别比较|请比较|请对比)", "", text).strip()
    parts = [_clean_subject_segment(part) for part in re.split(r"[、；;]", body)]
    parts = [part for part in parts if len(part) >= 3]
    if len(parts) < 2:
        return []
    out: List[Dict[str, str]] = []
    for part in parts[:6]:
        query_text = " ".join([part, common_suffix]).strip()
        out.append({"source": part, "query": query_text})
    return out
