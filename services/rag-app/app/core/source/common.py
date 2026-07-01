import difflib
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.utils.text import sanitize_index_text


_GENERIC_STRUCTURE_TITLE_RE = re.compile(
    r"^\s*(?:"
    r"\u7b2c[\u4e00-\u9fff0-9\uff10-\uff19]+[\u7ae0\u8282\u7f16\u90e8]\s*"
    r"[\u4e00-\u9fff0-9\uff10-\uff19\u3001\uff08\uff09() ]{0,24}"
    r"|[\u4e00-\u9fff0-9\uff10-\uff19\u3001\uff08\uff09() ]{0,16}"
    r"(?:\u603b\u5219|\u9644\u5219|\u6cd5\u5f8b\u8d23\u4efb)"
    r")\s*$"
)


def _is_generic_structure_title(title: str) -> bool:
    text = sanitize_index_text(title or "").strip()
    return bool(text and _GENERIC_STRUCTURE_TITLE_RE.match(text))


def strip_leading_region_prefix(title: str) -> str:
    text = (title or "").strip()
    if not text:
        return ""
    suffixes = [
        "特别行政区",
        "自治区",
        "自治州",
        "自治县",
        "地区",
        "省",
        "市",
        "区",
        "县",
        "旗",
    ]
    current = text
    for _ in range(2):
        changed = False
        for suffix in suffixes:
            idx = current.find(suffix)
            if idx < 2 or idx > 12:
                continue
            candidate = current[idx + len(suffix):].strip(" _-")
            if len(candidate) < 4:
                continue
            current = candidate
            changed = True
            break
        if not changed:
            break
    return current if current != text else ""


def strip_region_admin_tokens(text: str) -> str:
    current = (text or "").strip()
    if not current:
        return ""
    removable_tokens = [
        "特别行政区",
        "自治区",
        "自治州",
        "自治县",
        "地区",
        "省",
        "市",
        "区",
        "县",
        "旗",
    ]
    changed = True
    while changed and current:
        changed = False
        for token in removable_tokens:
            if current.endswith(token):
                current = current[: -len(token)].strip()
                changed = True
                break
    return current


def extract_region_token(title: str) -> str:
    text = (title or "").strip().replace("_", " ")
    if not text:
        return ""
    remainder = strip_leading_region_prefix(text)
    if not remainder:
        return ""
    idx = text.find(remainder)
    if idx < 1:
        return ""
    return strip_region_admin_tokens(text[:idx].strip(" _-"))


def source_display_title(
    source: str,
    *,
    doc_get: Callable[[str], Dict[str, Any]],
    filename_stem: Callable[[str], str],
) -> str:
    info = doc_get(source)
    title = sanitize_index_text(info.get("canonical_title") or "")
    if (
        not title
        or "\n" in title
        or len(title) > 80
        or _is_generic_structure_title(title)
        or any(marker in title for marker in ("正文：", "下文：", "章节路径：", "元素类型：", "Content:", "Previous context:", "Next context:"))
    ):
        title = ""
    return title or filename_stem(source) or source


def build_document_clarification_prompt(
    candidate_sources: List[str],
    *,
    doc_get: Callable[[str], Dict[str, Any]],
    filename_stem: Callable[[str], str],
    examples_limit: int = 3,
) -> str:
    suggestions: List[str] = []
    limit = max(1, int(examples_limit or 3))
    for source in candidate_sources or []:
        title = source_display_title(source, doc_get=doc_get, filename_stem=filename_stem)
        if title and title not in suggestions:
            suggestions.append(title)
        if len(suggestions) >= limit:
            break
    if suggestions:
        joined = "、".join(suggestions)
        return f"我找到了多个可能相关的文档，请确认你要查询的是：{joined}。"
    return "我还不能确定你要查询哪份文档，请补充文档名称或更具体的法规名称。"


def build_retrieval_grounded_clarification_prompt(
    query: str,
    candidate_sources: List[str],
    *,
    display_title: Callable[[str], str],
    reason: str = "document_target_required",
) -> str:
    titles: List[str] = []
    for source in candidate_sources or []:
        title = display_title(source) if source else "未知文档"
        if title and title not in titles:
            titles.append(title)
    joined = "、".join([f"《{title}》" for title in titles[:3]])
    return (
        "我需要先确认你要查询的文档。\n\n"
        f"原始问题：{query}\n"
        f"可能相关文档：{joined or '暂无明确候选'}\n"
        f"原因：{reason}\n\n"
        "请补充或选择具体文档后，我再继续检索回答。"
    )


def build_document_not_found_prompt(
    target: str,
    *,
    extract_filename_candidates: Callable[[str], List[str]],
) -> str:
    title = (target or "").strip() or "目标文档"
    if not (title.startswith("《") and title.endswith("》")) and not extract_filename_candidates(title):
        title = f"《{title}》"
    return f"没有找到 {title} 对应的已索引文档。请确认文件名、法规全称，或先上传该文档。"


def collapse_sources_by_canonical(
    sources: List[str],
    *,
    normalize_filename: Callable[[str], str],
    canonical_doc_id_for_source: Callable[[str], str],
    limit: Optional[int] = None,
) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for source in sources or []:
        safe_source = normalize_filename(source)
        if not safe_source:
            continue
        canonical_id = canonical_doc_id_for_source(safe_source) or f"source:{safe_source}"
        if canonical_id in seen:
            continue
        seen.add(canonical_id)
        out.append(safe_source)
        if limit and len(out) >= max(1, int(limit)):
            break
    return out


def text_overlap_ratio(
    left: str,
    right: str,
    *,
    normalize_reference_text: Callable[[str], str],
) -> float:
    lnorm = normalize_reference_text(left)
    rnorm = normalize_reference_text(right)
    if not lnorm or not rnorm:
        return 0.0
    lset = set(lnorm)
    rset = set(rnorm)
    overlap = len(lset & rset)
    denom = max(1, min(len(lset), len(rset)))
    return float(overlap) / float(denom)


def edit_similarity_ratio(
    left: str,
    right: str,
    *,
    normalize_reference_text: Callable[[str], str],
) -> float:
    lnorm = normalize_reference_text(left)
    rnorm = normalize_reference_text(right)
    if not lnorm or not rnorm:
        return 0.0
    if lnorm == rnorm:
        return 1.0
    rows = len(lnorm) + 1
    cols = len(rnorm) + 1
    dp = list(range(cols))
    for i in range(1, rows):
        prev = dp[0]
        dp[0] = i
        for j in range(1, cols):
            current = dp[j]
            if lnorm[i - 1] == rnorm[j - 1]:
                dp[j] = prev
            else:
                dp[j] = min(prev, dp[j], dp[j - 1]) + 1
            prev = current
    distance = dp[-1]
    return 1.0 - float(distance) / float(max(len(lnorm), len(rnorm), 1))


def source_core_entities(
    source: str,
    *,
    doc_get: Callable[[str], Dict[str, Any]],
    filename_stem: Callable[[str], str],
    normalize_query: Callable[[str], str],
) -> List[str]:
    info = doc_get(source)
    title = normalize_query(info.get("canonical_title") or filename_stem(source) or source)
    if not title:
        return []
    variants: List[str] = []
    stripped = normalize_query(strip_leading_region_prefix(title))
    for candidate in [title, stripped]:
        value = normalize_query(candidate)
        if len(value) >= 3 and value not in variants:
            variants.append(value)
    return variants


def query_matches_source_region_or_landmark(
    query: str,
    source: str,
    *,
    normalize_query: Callable[[str], str],
    source_display_title: Callable[[str], str],
    source_profile_fields: Callable[[str], Dict[str, Any]],
    source_core_entities: Callable[[str], List[str]],
    generic_doc_intent_terms: set[str],
) -> bool:
    qnorm = normalize_query(query)
    if not qnorm:
        return False
    title = source_display_title(source) if source else ""
    region = normalize_query(source_profile_fields(source).get("region") or extract_region_token(title))
    compact_region = normalize_query(strip_region_admin_tokens(region))
    if region and (region in qnorm or (compact_region and compact_region in qnorm)):
        return True
    for entity in source_core_entities(source):
        normalized = normalize_query(entity)
        if not normalized or normalized in generic_doc_intent_terms:
            continue
        if len(normalized) >= 4 and normalized in qnorm:
            return True
    return False


def is_pseudo_singleton_soft_lock(
    query: str,
    source: str,
    *,
    normalize_filename: Callable[[str], str],
    normalize_query: Callable[[str], str],
    source_display_title: Callable[[str], str],
    source_profile_fields: Callable[[str], Dict[str, Any]],
    source_core_entities: Callable[[str], List[str]],
    generic_doc_intent_terms: set[str],
) -> bool:
    safe_source = normalize_filename(source or "")
    if not safe_source:
        return False
    title = source_display_title(safe_source)
    region = normalize_query(source_profile_fields(safe_source).get("region") or extract_region_token(title))
    if not region:
        return False
    return not query_matches_source_region_or_landmark(
        query,
        safe_source,
        normalize_query=normalize_query,
        source_display_title=source_display_title,
        source_profile_fields=source_profile_fields,
        source_core_entities=source_core_entities,
        generic_doc_intent_terms=generic_doc_intent_terms,
    )


def resolve_unique_weak_match_upgrade(
    query: str,
    candidate_sources: List[str],
    *,
    collapse_sources_by_canonical: Callable[[List[str], Optional[int]], List[str]],
    source_display_title: Callable[[str], str],
    normalize_reference_text: Callable[[str], str],
    find_same_title_candidates: Callable[[str, str], List[str]],
    visible_document_exists: Callable[[str], bool],
    is_pseudo_singleton_soft_lock: Callable[[str, str], bool],
    min_score: float = 0.70,
) -> Dict[str, Any]:
    candidates = collapse_sources_by_canonical(candidate_sources, 5)
    if len(candidates) != 1:
        return {"resolved": False}
    source = candidates[0]
    title = source_display_title(source) if source else ""
    overlap = text_overlap_ratio(query, title, normalize_reference_text=normalize_reference_text)
    edit_sim = edit_similarity_ratio(query, title, normalize_reference_text=normalize_reference_text)
    score = max(overlap, edit_sim)
    if score < float(min_score):
        return {"resolved": False}
    same_title_candidates = find_same_title_candidates(title, source)
    visible_competitors = [item for item in same_title_candidates if visible_document_exists(item)]
    if visible_competitors:
        return {"resolved": False, "competition": visible_competitors[:3]}
    if is_pseudo_singleton_soft_lock(query, source):
        return {
            "resolved": False,
            "blocked_reason": "pseudo_singleton_region_mismatch",
            "candidate": source,
        }
    return {
        "resolved": True,
        "source": source,
        "reason": "soft_lock_unique",
        "confidence": score,
        "lock_message_prefix": f"我理解你查询的是《{title}》。\n",
        "trace": {"overlap_ratio": overlap, "edit_similarity": edit_sim},
    }


def normalized_embedding_cosine(left: Tuple[float, ...], right: Tuple[float, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return float(sum(left_value * right_value for left_value, right_value in zip(left, right)))
