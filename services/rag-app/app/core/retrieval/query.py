import difflib
import re
from typing import Any, Callable, Dict, List, Optional

from app.utils import scoring as scoring_utils
from app.core.query import rewrite as query_rewrite


def retrieval_query_has_doc_noise(
    retrieval_query: str,
    *,
    locked_title: str = "",
    locked_sources: Optional[List[str]] = None,
    normalize_query,
) -> bool:
    normalized = normalize_query(retrieval_query)
    title = normalize_query(locked_title)
    if title and title in normalized:
        return True
    for source in locked_sources or []:
        normalized_source = normalize_query(source)
        if normalized_source and normalized_source in normalized:
            return True
    return False


def expand_retrieval_query_from_corpus(query: str, retrieval_query: str) -> tuple[str, List[str]]:
    base_query = retrieval_query or query
    expanded_terms = query_rewrite.unpack_legal_abstractions_fallback(query)
    expanded_query = query_rewrite.expand_query_with_terms(base_query, expanded_terms)
    return expanded_query or base_query, expanded_terms


def seed_anchor_terms_for_probe(query: str, query_anchor_terms) -> List[str]:
    return list(query_anchor_terms(query) or [])


def doc_recall_fallback(
    query: str,
    limit: int,
    *,
    document_fts_rows: Callable[[], List[Any]],
    normalize_filename: Callable[[str], str],
    source_state: Callable[[str], Dict[str, Any]],
    token_overlap_score: Callable[[str, str], float],
    doc_title_alias_score: Callable[[str, str], float],
    source_filter: Optional[str] = None,
) -> List[str]:
    ranked: List[tuple[float, str]] = []
    for filename, title, aliases, doc_type, topic, filename_stem in document_fts_rows():
        source = normalize_filename(filename or "")
        if not source:
            continue
        if source_filter and source != source_filter:
            continue
        state = source_state(source)
        if not state.get("visible"):
            continue
        title_text = "\n".join([title or "", aliases or "", filename_stem or "", source])
        score = token_overlap_score(query, title_text)
        score += doc_title_alias_score(source, query)
        if source_filter and source == source_filter:
            score += 2.0
        if score > 0:
            ranked.append((score, source))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    out: List[str] = []
    for _, source in ranked:
        if source not in out:
            out.append(source)
        if len(out) >= int(limit):
            break
    return out


def soft_lock_query_anchor_terms(
    query: str,
    source: str,
    *,
    source_title_aspect_terms: Callable[[List[str]], List[str]],
    query_semantic_aspects: Callable[..., Dict[str, Any]],
    query_content_anchor_terms: Callable[[str, Optional[Dict[str, Any]], List[str]], List[str]],
    extract_section_query_targets: Callable[[str], List[str]],
    local_validate_section_targets: Callable[..., List[str]],
    normalize_coverage_aspect: Callable[[str], str],
    normalize_query: Callable[[str], str],
) -> List[str]:
    title_terms = source_title_aspect_terms([source])
    qfilters: Dict[str, Any] = {}
    semantic = query_semantic_aspects(query, qfilters=qfilters)
    anchors = query_content_anchor_terms(query, qfilters, title_terms)
    out: List[str] = []
    section_targets = local_validate_section_targets(extract_section_query_targets(query), limit=4)
    for term in list(semantic.get("terms") or []) + list(anchors or []) + section_targets:
        normalized = normalize_coverage_aspect(term) or normalize_query(term)
        if len(normalized) >= 2 and normalized not in out:
            out.append(normalized)
    return out[:8]


def text_overlap_ratio(left: str, right: str, *, normalize_reference_text: Callable[[str], str]) -> float:
    lnorm = normalize_reference_text(left)
    rnorm = normalize_reference_text(right)
    if not lnorm or not rnorm:
        return 0.0
    lset = set(lnorm)
    rset = set(rnorm)
    overlap = len(lset & rset)
    return float(overlap) / float(max(1, min(len(lset), len(rset))))


def edit_similarity_ratio(left: str, right: str, *, normalize_reference_text: Callable[[str], str]) -> float:
    lnorm = normalize_reference_text(left)
    rnorm = normalize_reference_text(right)
    if not lnorm or not rnorm:
        return 0.0
    return float(difflib.SequenceMatcher(None, lnorm, rnorm).ratio())


def soft_lock_has_duplicate_formats(
    candidate_sources: List[str],
    *,
    normalize_filename: Callable[[str], str],
    collapse_sources_by_canonical: Callable[[List[str], Optional[int]], List[str]],
) -> bool:
    normalized = [
        normalize_filename(source or "")
        for source in (candidate_sources or [])
        if normalize_filename(source or "")
    ]
    collapsed = collapse_sources_by_canonical(normalized, max(1, len(normalized)))
    return len(normalized) > len(collapsed) and len(collapsed) == 1


def soft_lock_confidence(
    query: str,
    source: str,
    candidate_sources: List[str],
    *,
    raw_title_score: float = 0.0,
    top_competitors: Optional[List[Dict[str, Any]]] = None,
    normalize_filename: Callable[[str], str],
    normalize_reference_text: Callable[[str], str],
    collapse_sources_by_canonical: Callable[[List[str], Optional[int]], List[str]],
    source_display_title: Callable[[str], str],
    query_matches_source_region_or_landmark: Callable[[str, str], bool],
    geo_context_tokens: Callable[[str, str], List[str]],
    soft_lock_query_anchor_terms_fn: Callable[[str, str], List[str]],
    source_body_anchor_match_count: Callable[[str, List[str]], int],
    source_supports_doc_identity_term: Callable[[str, str], bool],
    rank_title_source_matches: Callable[..., List[Dict[str, Any]]],
    clamp01: Callable[[float], float],
) -> tuple[float, Dict[str, Any]]:
    safe_source = normalize_filename(source or "")
    normalized_candidates = [
        normalize_filename(item or "")
        for item in (candidate_sources or [])
        if normalize_filename(item or "")
    ]
    collapsed_candidates = collapse_sources_by_canonical(normalized_candidates, max(1, len(normalized_candidates)))
    title = source_display_title(safe_source)
    overlap = text_overlap_ratio(query, title, normalize_reference_text=normalize_reference_text)
    edit_sim = edit_similarity_ratio(query, title, normalize_reference_text=normalize_reference_text)
    title_strength = clamp01(max(raw_title_score / 10.0, overlap, edit_sim))
    geo_match = query_matches_source_region_or_landmark(query, safe_source)
    geo_strength = 1.0 if geo_match else (0.55 if not geo_context_tokens(query, "") else 0.15)
    anchor_terms = soft_lock_query_anchor_terms_fn(query, safe_source)
    topic_hits = source_body_anchor_match_count(safe_source, anchor_terms) if anchor_terms else 0
    topic_strength = clamp01(float(topic_hits) / float(max(1, len(anchor_terms)))) if anchor_terms else 0.55
    identity_strength = 1.0 if source_supports_doc_identity_term(safe_source, query) else 0.35
    canonical_candidate_count = len(collapsed_candidates)
    candidate_strength = 1.0 if canonical_candidate_count <= 1 else clamp01(1.0 - (0.22 * float(canonical_candidate_count - 1)))
    competitor_score = 0.0
    if top_competitors:
        competitor_score = float((top_competitors[0] or {}).get("score") or 0.0)
    elif raw_title_score > 0.0:
        for entry in rank_title_source_matches(query, limit=6, include_topic_like=True):
            competitor_source = normalize_filename((entry or {}).get("source") or "")
            if competitor_source and competitor_source != safe_source:
                competitor_score = max(competitor_score, float((entry or {}).get("score") or 0.0))
                break
    score_gap = max(0.0, float(raw_title_score) - float(competitor_score)) if raw_title_score > 0.0 else max(0.0, max(overlap, edit_sim) - 0.55)
    distribution_strength = clamp01(score_gap / 2.0)
    duplicate_bonus = 0.08 if soft_lock_has_duplicate_formats(
        normalized_candidates,
        normalize_filename=normalize_filename,
        collapse_sources_by_canonical=collapse_sources_by_canonical,
    ) else 0.0
    confidence = clamp01(
        (0.34 * title_strength)
        + (0.16 * geo_strength)
        + (0.18 * topic_strength)
        + (0.12 * identity_strength)
        + (0.12 * candidate_strength)
        + (0.12 * distribution_strength)
        + duplicate_bonus
    )
    return confidence, {
        "soft_lock_features": {
            "title_strength": round(title_strength, 4),
            "geo_strength": round(geo_strength, 4),
            "topic_strength": round(topic_strength, 4),
            "identity_strength": round(identity_strength, 4),
            "candidate_strength": round(candidate_strength, 4),
            "distribution_strength": round(distribution_strength, 4),
            "duplicate_format_bonus": round(duplicate_bonus, 4),
            "candidate_count": len(normalized_candidates),
            "canonical_candidate_count": canonical_candidate_count,
            "topic_hits": topic_hits,
            "topic_anchor_count": len(anchor_terms),
            "top_competitor_score": round(competitor_score, 4),
            "title_overlap": round(overlap, 4),
            "title_edit_similarity": round(edit_sim, 4),
        }
    }


def strip_compare_noise_terms(text: str, normalize_query: Callable[[str], str]) -> str:
    query = normalize_query(text)
    if not query:
        return ""
    query = re.sub(r"\b(vs|versus|compare)\b", " ", query, flags=re.IGNORECASE)
    query = re.sub(r"(对比|比较|差异|区别|不同点|相同点|异同|相比|之间)", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    return re.sub(r"[，,。.?？!！\s]+$", "", query).strip()


def strip_raw_text_mentions(query: str, terms: List[str], normalize_query: Callable[[str], str]) -> str:
    text = normalize_query(query)
    if not text:
        return ""
    for term in terms or []:
        token = normalize_query(term)
        if token:
            text = text.replace(token, " ")
    return normalize_query(re.sub(r"\s+", " ", text).strip())


def strip_filename_mentions(query: str, filenames: List[str], normalize_query: Callable[[str], str]) -> str:
    text = normalize_query(query)
    for name in filenames or []:
        token = normalize_query(name or "")
        if token:
            text = text.replace(token, " ")
        stem = re.sub(r"\.(docx?|pdf|xlsx?|txt|md)$", "", token, flags=re.IGNORECASE).strip()
        if stem and stem != token:
            text = text.replace(stem, " ")
    return normalize_query(text)


def strip_source_title_mentions(
    query: str,
    sources: List[str],
    *,
    normalize_query: Callable[[str], str],
    doc_title_alias_candidates: Callable[[str], List[str]],
) -> str:
    text = normalize_query(query)
    candidates: List[str] = []
    for source in sources or []:
        for candidate in doc_title_alias_candidates(source):
            token = (candidate or "").strip()
            if len(token) >= 4 and token not in candidates:
                candidates.append(token)
    for token in sorted(candidates, key=len, reverse=True):
        text = text.replace(token, " ")
    return normalize_query(text)


def purify_retrieval_query_shallow(query: str) -> str:
    text = (query or "").strip()
    if not text:
        return ""
    for char in ['<', '>', '"', "'", '[', ']', '(', ')', '《', '》', '“', '”', '‘', '’']:
        text = text.replace(char, " ")
    text = re.sub(r"(please|about|regarding)", " ", text, flags=re.I)
    text = re.sub(r"(what is|how to|whether|compare|difference)", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def purify_locked_source_query(
    query: str,
    sources: List[str],
    *,
    normalize_query: Callable[[str], str],
    doc_title_alias_candidates: Callable[[str], List[str]],
) -> str:
    text = normalize_query(query)
    if not text:
        return ""

    text = strip_filename_mentions(text, sources, normalize_query)
    text = strip_source_title_mentions(
        text,
        sources,
        normalize_query=normalize_query,
        doc_title_alias_candidates=doc_title_alias_candidates,
    )
    for source in sources or []:
        token = normalize_query(source or "")
        if not token:
            continue
        compact_token = re.sub(r"[\s_ -]+", "", token)
        compact_text = re.sub(r"[\s_ -]+", "", text)
        if compact_token and compact_token in compact_text:
            text = text.replace(token, " ")

    text = re.sub(r"\b\d{4}[-_/年]\d{1,2}[-_/月]\d{0,2}日?\b", " ", text)
    text = re.sub(r"\b\d{4}[-_/]\d{1,2}[-_/]?\b", " ", text)
    text = re.sub(r"\.(docx?|pdf|xlsx?|txt|md)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[_/\\]+", " ", text)
    text = re.sub(r"[《》“”‘’\"'<>【】\[\]()（）：:？?，,。；;]", " ", text)
    text = re.sub(r"(只依据|仅依据|依据|根据|基于|请|帮我|回答|说明|查询|文件|文档|中|里的|里面的)", " ", text)
    text = re.sub(r"(的规定是什么|规定是什么|有哪些规定|是什么|有哪些|是什么内容)", " ", text)
    return normalize_query(text)
