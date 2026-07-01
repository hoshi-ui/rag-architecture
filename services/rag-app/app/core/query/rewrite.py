from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


LEGAL_ABSTRACTION_TERMS: Dict[str, List[str]] = {
    "义务": ["应当", "必须", "不得", "禁止", "负责", "备案", "登记", "申报"],
    "权利义务": ["应当", "必须", "不得", "禁止", "负责", "履行", "承担"],
    "职责": ["负责", "主管", "监督管理", "组织", "指导", "协同"],
    "分工": ["负责", "主管", "监督管理", "协助", "配合", "职责分工"],
    "协助": ["协助", "配合", "支持", "基层组织", "居民委员会", "村民委员会"],
    "责任": ["责令改正", "罚款", "没收", "吊销", "处罚", "法律责任"],
    "后果": ["责令改正", "罚款", "没收", "吊销", "处罚", "法律责任"],
    "处理结果": ["责令改正", "罚款", "没收", "吊销", "处罚", "依法处理"],
    "处罚": ["责令改正", "罚款", "没收", "吊销", "行政处罚"],
    "处罚方式": ["责令改正", "罚款", "没收", "吊销", "行政处罚"],
    "罚款幅度": ["罚款", "以上", "以下", "处"],
    "违法": ["违反规定", "责令改正", "罚款", "没收违法所得", "行政处罚"],
    "拒不改正": ["责令改正", "逾期不改正", "罚款", "吊销", "情节严重"],
    "违法所得": ["违法所得", "难以计算", "没收", "罚款", "货值金额"],
    "适用范围": ["适用", "本条例所称", "不包括", "范围"],
    "定义": ["本条例所称", "是指", "包括", "不包括", "适用"],
    "管理活动": ["管理", "监督", "备案", "登记", "检查"],
    "主体": ["人民政府", "主管部门", "行政机关", "执法机关", "有关部门"],
    "期限": ["期限", "时限", "自受理之日起", "日内", "办理机关", "主管部门"],
    "办理机关": ["办理机关", "受理", "主管部门", "公安机关", "行政机关"],
    "材料": ["申请材料", "提交材料", "证明材料", "受理", "审查"],
    "许可": ["申请", "许可", "审批", "批准", "审查", "条件"],
    "用途": ["用于", "用途", "事项", "记录", "公示", "共享", "信息平台"],
    "事项": ["用于", "事项", "记录", "登记", "备案", "信息平台", "电子档案"],
    "信息平台": ["信息平台", "电子信息平台", "记录", "公示", "共享", "查询"],
    "电子档案": ["电子档案", "信息记录", "登记信息", "管理信息", "共享"],
    "规划": ["规划", "保护规划", "编制", "组织实施", "主管部门", "人民政府"],
}

COMMON_COMPARE_TERMS = ("比较", "对比", "区别", "差异", "异同", "分别", "各自", "三类", "多个")
HYDE_TRIGGER_TERMS = (
    "职责",
    "分工",
    "协助",
    "配合",
    "权限",
    "处罚",
    "处罚方式",
    "罚款",
    "罚款幅度",
    "责任",
    "后果",
    "处理结果",
    "违法",
    "违规",
    "违反",
    "责令",
    "改正",
    "拒不改正",
    "逾期",
    "情节严重",
    "违法所得",
    "条件",
    "程序",
    "办理",
    "期限",
    "时限",
    "办理机关",
    "材料",
    "申请",
    "受理",
    "许可",
    "审批",
    "登记",
    "备案",
    "义务",
    "权利义务",
    "应当",
    "不得",
    "禁止",
    "范围",
    "定义",
    "适用",
    "标准",
    "要求",
    "用途",
    "事项",
    "信息平台",
    "电子平台",
    "电子信息平台",
    "电子档案",
    "记录",
    "支撑",
    "规划",
    "保护规划",
)


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


def has_abstract_legal_query(query: str) -> bool:
    text = str(query or "")
    if not text:
        return False
    return any(term in text for term in HYDE_TRIGGER_TERMS) or bool(unpack_legal_abstractions_fallback(text, limit=1))


def _intent_has(intent: str, *tokens: str) -> bool:
    text = str(intent or "")
    return any(token in text for token in tokens)


def build_hyde_fallback(query: str, terms: Optional[List[str]] = None, intent: str = "") -> str:
    base = " ".join(str(query or "").split()).strip()
    if not base:
        return ""
    term_set = set(terms or unpack_legal_abstractions_fallback(base, limit=12))
    fragments: List[str] = []
    if _intent_has(intent, "责任", "处罚") or any(term in term_set for term in ("责令改正", "罚款", "没收", "处罚", "行政处罚")):
        fragments.append("违反规定的，由有关主管部门责令改正、给予警告、处以罚款、没收违法所得或者依法处理。")
    if _intent_has(intent, "职责", "权限") or any(term in term_set for term in ("负责", "主管", "监督管理", "组织", "指导", "协同")):
        fragments.append("有关部门按照职责分工负责监督管理、组织协调、指导服务、检查执法和信息共享。")
    if _intent_has(intent, "程序", "条件") or any(term in term_set for term in ("登记", "备案", "补办", "逾期")):
        fragments.append("申请人应当提交材料并依法办理登记、备案、审查、变更或者补办手续，主管部门应当在规定期限内办结。")
    if _intent_has(intent, "定义", "范围") or any(term in term_set for term in ("适用", "本条例所称", "范围")):
        fragments.append("本条例适用于相关管理活动，明确适用范围、定义、包括情形和不适用情形。")
    if _intent_has(intent, "义务") or any(term in term_set for term in ("应当", "必须", "不得", "禁止")):
        fragments.append("相关单位和个人应当履行管理义务，不得违反禁止性规定，并接受监督检查。")
    if not fragments:
        fragments.append("相关条款通常规定主管部门、办理程序、适用条件、法律责任以及监督管理要求。")
    hyde = f"{base}。相关法条可能表述为：" + "".join(fragments)
    return hyde[:600].strip()


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
