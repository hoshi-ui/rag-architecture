import re
from typing import Any, Callable, Dict, List, Optional


FILENAME_PATTERN = re.compile(
    r"([A-Za-z0-9_\-\(\)\u4e00-\u9fa5\.]+?\.(?:pdf|docx|xlsx|txt|md|markdown|csv|json|log))(?![A-Za-z0-9_\-\(\)\u4e00-\u9fa5])",
    flags=re.IGNORECASE,
)

CONTEXTUAL_DOC_REFERENCE_MARKERS = [
    "这个",
    "这份",
    "本文",
    "该文档",
    "该文件",
    "上面",
    "刚才",
    "当前文档",
    "当前文件",
    "上一份",
    "这部",
    "该法规",
    "这条",
    "这个条例",
    "这个办法",
    "这个规定",
]

FOLLOWUP_TERMS = ["继续", "展开", "详细说明", "上一条", "下一条", "第几条", "相关规定"]


def extract_filename_candidates(query: str) -> List[str]:
    return list(dict.fromkeys(FILENAME_PATTERN.findall(query or "")))


def is_stats_intent(query: str) -> bool:
    normalized = (query or "").lower()
    keys = ["??", "??", "??", "??", "chunk", "???", "??", "??", "??", "sheet", "???", "??"]
    return any(key.lower() in normalized for key in keys)


def has_contextual_doc_reference(query: str, normalize_query: Callable[[str], str]) -> bool:
    normalized = normalize_query(query)
    if not normalized:
        return False
    return any(marker in normalized for marker in CONTEXTUAL_DOC_REFERENCE_MARKERS)


def classify_question_type(
    query: str,
    *,
    normalize_query: Callable[[str], str],
    question_type_patterns: List[Dict[str, Any]],
) -> str:
    normalized = normalize_query(query).lower()
    for pattern in question_type_patterns or []:
        qtype = str((pattern or {}).get("type") or "").strip()
        if not qtype:
            continue
        for keyword in (pattern or {}).get("keywords") or []:
            if str(keyword).lower() in normalized:
                return qtype
    return "other"


def strip_query_wrapper_terms(text: str, normalize_query: Callable[[str], str]) -> str:
    cleaned = normalize_query(text)
    if not cleaned:
        return ""
    for char in ['<', '>', '"', "'", '[', ']', '(', ')', ',', '.', ';', ':', '?', '!']:
        cleaned = cleaned.replace(char, " ")
    cleaned = re.sub(r"(please|about|regarding|query|search)", " ", cleaned, flags=re.I)
    return re.sub(r"\s+", " ", cleaned).strip()


def query_quality_strong_topic_terms(
    query: str,
    *,
    normalize_query: Callable[[str], str],
    local_fallback_anchor_terms: Callable[[str], List[str]],
    query_anchor_terms: Callable[[str], List[str]],
    domain_markers: List[str],
    generic_query_terms: set[str],
) -> List[str]:
    normalized = normalize_query(query)
    if not normalized:
        return []
    cleaned_terms: List[str] = []
    for term in local_fallback_anchor_terms(normalized):
        if any(marker in term for marker in domain_markers) and term not in cleaned_terms:
            cleaned_terms.append(term)
    for term in query_anchor_terms(normalized):
        token = strip_query_wrapper_terms(term, normalize_query)
        if len(token) < 2:
            continue
        if token in generic_query_terms:
            continue
        if any(marker in token for marker in domain_markers) or token not in cleaned_terms:
            cleaned_terms.append(token)
        if len(cleaned_terms) >= 8:
            break
    return cleaned_terms[:8]


def is_doc_existence_query(query: str, normalize_query: Callable[[str], str]) -> bool:
    normalized = normalize_query(query)
    if not normalized:
        return False
    return any(marker in normalized for marker in ("是否存在", "有没有", "是否有", "存在吗", "已上传", "上传了吗"))


def is_deleted_visibility_query(query: str, normalize_query: Callable[[str], str]) -> bool:
    normalized = normalize_query(query)
    if not normalized:
        return False
    return any(marker in normalized for marker in ("删除", "删掉", "不可见", "看不到", "是否还在", "还在吗"))


def is_business_topic_query(
    query: str,
    *,
    normalize_query: Callable[[str], str],
    extract_filename_candidates_fn: Callable[[str], List[str]],
    extract_explicit_regulation_mentions: Callable[[str], List[str]],
    is_doc_existence: Callable[[str], bool],
    is_deleted_visibility: Callable[[str], bool],
    is_version_switch: Callable[[str], bool],
    domain_markers: List[str],
    action_markers: List[str],
) -> bool:
    normalized = normalize_query(query)
    if not normalized or extract_filename_candidates_fn(normalized) or extract_explicit_regulation_mentions(normalized):
        return False
    if is_doc_existence(normalized) or is_deleted_visibility(normalized) or is_version_switch(normalized):
        return False
    return any(marker in normalized for marker in domain_markers) and any(marker in normalized for marker in action_markers)


def has_strong_business_signal(
    query: str,
    *,
    normalize_query: Callable[[str], str],
    is_business_topic: Callable[[str], bool],
    strong_topic_terms: Callable[[str], List[str]],
) -> bool:
    normalized = normalize_query(query)
    if not normalized:
        return False
    return bool(is_business_topic(normalized) or strong_topic_terms(normalized))


def has_weak_business_signal(
    query: str,
    *,
    normalize_query: Callable[[str], str],
    strong_topic_terms: Callable[[str], List[str]],
    domain_markers: List[str],
) -> bool:
    normalized = normalize_query(query)
    if not normalized:
        return False
    return any(marker in normalized for marker in domain_markers) or bool(strong_topic_terms(normalized))


def is_open_regulation_query(
    query: str,
    *,
    normalize_query: Callable[[str], str],
    extract_filename_candidates_fn: Callable[[str], List[str]],
    extract_explicit_regulation_mentions: Callable[[str], List[str]],
    is_doc_existence: Callable[[str], bool],
    is_deleted_visibility: Callable[[str], bool],
    strong_topic_terms: Callable[[str], List[str]],
) -> bool:
    normalized = normalize_query(query)
    if not normalized or extract_filename_candidates_fn(normalized) or extract_explicit_regulation_mentions(normalized):
        return False
    if is_doc_existence(normalized) or is_deleted_visibility(normalized):
        return False
    return bool(strong_topic_terms(normalized))


def classify_query_route(
    query: str,
    *,
    fnames: Optional[List[str]],
    normalize_query: Callable[[str], str],
    extract_filename_candidates_fn: Callable[[str], List[str]],
    analyze_compare_route: Callable[[str], Dict[str, Any]],
    resolve_explicit_reference_sources: Callable[[str, List[str]], Dict[str, Any]],
    classify_title_reference_route: Callable[[str, List[str]], str],
    policy_get: Callable[[str, Any], Any],
    is_doc_existence: Callable[[str], bool],
    is_deleted_visibility: Callable[[str], bool],
    is_version_switch: Callable[[str], bool],
    is_business_topic: Callable[[str], bool],
    is_open_regulation: Callable[[str], bool],
) -> str:
    normalized = normalize_query(query)
    names = fnames if fnames is not None else extract_filename_candidates_fn(normalized)
    compare_resolution = analyze_compare_route(normalized)
    if compare_resolution.get("is_compare"):
        return compare_resolution.get("route") or "open_topic_compare"
    explicit_resolution = resolve_explicit_reference_sources(normalized, list(names or []))
    title_route = classify_title_reference_route(normalized, list(names or []))
    strong_title_route = title_route if title_route in {"exact_title_reference", "alias_title_reference"} else ""
    weak_title_route = title_route if title_route in {"weak_title_reference", "topic_like_title"} else ""
    route_order = policy_get(
        "query_route.order",
        [
            "existence",
            "visibility_probe",
            "version_switch",
            "explicit_doc_reference",
            "explicit_regulation_reference",
            "weak_title_reference",
            "content_qa",
        ],
    )
    for route_name in route_order:
        if route_name == "existence" and is_doc_existence(normalized):
            return "existence"
        if route_name == "visibility_probe" and is_deleted_visibility(normalized):
            return "visibility_probe"
        if route_name == "version_switch" and is_version_switch(normalized):
            return "version_switch"
        if route_name == "explicit_doc_reference" and explicit_resolution.get("route") == "explicit_doc_reference":
            return "explicit_doc_reference"
        if route_name == "explicit_regulation_reference" and explicit_resolution.get("route") == "explicit_regulation_reference":
            return "explicit_regulation_reference"
        if route_name == "weak_title_reference":
            require_no_filenames = bool(policy_get("query_route.weak_title_reference.require_no_filenames", True))
            if ((not require_no_filenames) or (not names)) and strong_title_route:
                return strong_title_route
        if route_name == "business_topic_qa" and is_business_topic(normalized):
            return "business_topic_qa"
        if route_name == "open_regulation_qa" and is_open_regulation(normalized):
            return "open_regulation_qa"
        if route_name == "content_qa":
            if weak_title_route:
                return weak_title_route
            return "content_qa"
    return "content_qa"


def is_contextual_followup_query(
    query: str,
    normalize_query: Callable[[str], str],
    has_contextual_reference: Callable[[str], bool],
) -> bool:
    normalized = normalize_query(query)
    if not normalized:
        return False
    if has_contextual_reference(normalized):
        return True
    if re.search(r"第[一二三四五六七八九十百千万0-9]+[条款项章]", normalized):
        return True
    return any(term in normalized for term in FOLLOWUP_TERMS)


def is_pure_topic_question(
    query: str,
    route: str,
    normalize_query: Callable[[str], str],
    query_has_doc_identity_term: Callable[[str], bool],
    extract_explicit_regulation_mentions: Callable[[str], List[str]],
    extract_filename_candidates_fn: Callable[[str], List[str]],
    query_anchor_terms: Callable[[str], List[str]],
    has_contextual_reference: Callable[[str], bool],
) -> bool:
    normalized = normalize_query(query)
    if not normalized:
        return False
    if route not in {"business_topic_qa", "open_regulation_qa", "content_qa"}:
        return False
    if query_has_doc_identity_term(normalized):
        return False
    if extract_explicit_regulation_mentions(normalized) or extract_filename_candidates_fn(normalized):
        return False
    return bool(query_anchor_terms(normalized)) and not has_contextual_reference(normalized)


def classify_query_scope(
    query: str,
    fnames: List[str],
    query_route: Optional[str],
    classify_query_route: Callable[[str, List[str]], str],
    normalize_query: Callable[[str], str],
    has_contextual_reference: Callable[[str], bool],
    is_section_anchor_query: Callable[[str], bool],
) -> str:
    route = query_route or classify_query_route(query, fnames)
    normalized = normalize_query(query)
    anchored_markers = ["法规", "条例", "办法", "规定", "条款", "章节", "文档", "文件"]
    if fnames or route in {
        "version_switch",
        "explicit_doc_reference",
        "explicit_regulation_reference",
        "exact_title_reference",
        "alias_title_reference",
    }:
        return "anchored_question"
    if has_contextual_reference(normalized):
        return "anchored_question"
    if route not in {"business_topic_qa", "open_regulation_qa"} and is_section_anchor_query(normalized):
        return "anchored_question"
    if route in {"weak_title_reference", "exact_title_reference", "alias_title_reference"} and any(
        marker in normalized for marker in anchored_markers
    ):
        return "anchored_question"
    return "open_question"


def section_target_alignment(
    section: str,
    query: str,
    *,
    extract_section_query_targets: Callable[[str], List[str]],
) -> tuple:
    section_text = (section or "").strip().replace(" ", "")
    if not section_text:
        return 0.0, 0.0
    targets = extract_section_query_targets(query)
    if not targets:
        return 0.0, 0.0
    hits = 0.0
    exact = 0.0
    for target in targets:
        target_text = (target or "").strip().replace(" ", "")
        if not target_text:
            continue
        if target_text in section_text or section_text in target_text:
            hits += 1.0
            if target_text == section_text or (len(target_text) >= 3 and target_text in section_text):
                exact = 1.0
    return min(1.0, max(0.0, hits / max(1.0, float(len(targets))))), exact


def looks_like_section_target(target: str, normalize_query: Callable[[str], str]) -> bool:
    normalized = normalize_query(target)
    if len(normalized) < 2 or len(normalized) > 20:
        return False
    section_keywords = [
        "章",
        "节",
        "条",
        "款",
        "项",
        "目",
        "规定",
        "要求",
        "职责",
        "责任",
        "范围",
        "原则",
        "流程",
        "程序",
        "条件",
        "标准",
        "处罚",
        "监督",
        "管理",
        "备案",
        "审批",
    ]
    if any(keyword in normalized for keyword in section_keywords):
        return True
    if re.search(r"第[一二三四五六七八九十百千万0-9]+[章节条款项]", normalized):
        return True
    return False


def local_validate_section_targets(
    targets: List[str],
    *,
    normalize_query: Callable[[str], str],
    limit: int = 5,
) -> List[str]:
    out: List[str] = []
    generic_targets = {
        "内容",
        "规定",
        "要求",
        "条款",
        "章节",
        "说明",
        "依据",
        "范围",
        "职责",
        "管理",
        "原则",
        "general",
        "appendix",
        "scope",
        "responsibility",
        "overview",
        "introduction",
    }
    for raw in targets or []:
        target = normalize_query(raw)
        if not target:
            continue
        if len(target) < 2 or len(target) > 20:
            continue
        if target in generic_targets:
            continue
        if any(marker in target for marker in ["什么", "哪些", "怎么", "如何", "是否"]):
            continue
        if not looks_like_section_target(target, normalize_query):
            continue
        if target not in out:
            out.append(target)
        if len(out) >= max(1, int(limit)):
            break
    return out
