import re
from typing import Any, Callable, Dict, List, Optional, Tuple


from app.core.source.common import (
    strip_leading_region_prefix,
    strip_region_admin_tokens,
    extract_region_token,
    source_display_title,
    build_document_clarification_prompt,
    build_retrieval_grounded_clarification_prompt,
    build_document_not_found_prompt,
    collapse_sources_by_canonical,
    text_overlap_ratio,
    edit_similarity_ratio,
    source_core_entities,
    query_matches_source_region_or_landmark,
    is_pseudo_singleton_soft_lock,
    resolve_unique_weak_match_upgrade,
    normalized_embedding_cosine,
)


def dense_title_source_matches(
    text: str,
    limit: int,
    *,
    enabled: bool,
    normalize_query: Callable[[str], str],
    normalize_reference_text: Callable[[str], str],
    embed_text: Callable[[str], Tuple[float, ...]],
    dense_title_probe_entries: Callable[[], Tuple[Tuple[str, str, str], ...]],
    build_doc_recall_plan: Callable[[str, int], List[Dict[str, Any]]],
    normalize_filename: Callable[[str], str],
    source_display_title: Callable[[str], str],
    max_probe_chars: int = 160,
) -> List[Dict[str, Any]]:
    if not enabled:
        return []
    probe = normalize_reference_text(text) or normalize_query(text)
    if len(probe) < 2:
        return []
    strip_suffixes = [
        "管理条例",
        "实施办法",
        "暂行办法",
        "规定",
        "办法",
        "条例",
        "细则",
        "处罚",
        "责任",
        "要求",
        "职责",
        "程序",
        "范围",
    ]
    core_fragments: List[str] = []
    for fragment in re.findall(r"[\u4e00-\u9fff]{2,}", normalize_query(text)):
        cleaned = fragment
        for suffix in strip_suffixes:
            if cleaned.endswith(suffix) and len(cleaned) - len(suffix) >= 2:
                cleaned = cleaned[: -len(suffix)]
                break
        cleaned = normalize_query(cleaned)
        if len(cleaned) >= 2 and cleaned not in core_fragments:
            core_fragments.append(cleaned)

    max_chars = max(40, int(max_probe_chars or 160))
    query_embedding = embed_text(probe[:max_chars])
    ranked_map: Dict[str, Dict[str, Any]] = {}
    compact_probe = normalize_reference_text(text)
    if query_embedding:
        for source, display_title, probe_text in dense_title_probe_entries():
            title_embedding = embed_text(probe_text[:max_chars])
            if not title_embedding:
                continue
            score = normalized_embedding_cosine(query_embedding, title_embedding)
            compact_title = normalize_reference_text(display_title)
            if compact_probe and compact_title and (compact_probe in compact_title or compact_title in compact_probe):
                score = max(score, 0.86)
            if core_fragments and any(fragment in normalize_query(display_title) for fragment in core_fragments):
                score = max(score, 0.88)
            if score <= 0.0:
                continue
            ranked_map[source] = {
                "source": source,
                "title": display_title,
                "score": score,
            }

    for entry in build_doc_recall_plan(text, max(limit * 3, 8)):
        source = normalize_filename((entry or {}).get("source") or "")
        if not source:
            continue
        display_title = source_display_title(source) if source else ""
        if core_fragments and not any(fragment in normalize_query(display_title) for fragment in core_fragments):
            continue
        prior = float((entry or {}).get("prior") or 0.0)
        if prior <= 0.0:
            continue
        score = 0.82 + min(prior, 0.12)
        current = ranked_map.get(source)
        if current is None or score > float(current.get("score") or 0.0):
            ranked_map[source] = {
                "source": source,
                "title": display_title,
                "score": score,
            }

    ranked = list(ranked_map.values())
    ranked.sort(key=lambda item: (-float(item.get("score") or 0.0), item.get("source") or ""))
    return ranked[: max(1, int(limit))]


def canonical_doc_id_for_source(
    source: str,
    *,
    normalize_filename: Callable[[str], str],
    doc_get: Callable[[str], Dict[str, Any]],
    filename_stem: Callable[[str], str],
    same_title_group: Callable[[str], str],
    normalize_title_probe_text: Callable[[str], str],
) -> str:
    safe_source = normalize_filename(source)
    if not safe_source:
        return ""
    info = doc_get(safe_source)
    group = str(info.get("same_title_group") or "").strip()
    if group:
        return group
    canonical_title = str(info.get("canonical_title") or filename_stem(safe_source) or safe_source).strip()
    group = same_title_group(canonical_title)
    if group:
        return group
    return normalize_title_probe_text(canonical_title)

from app.core.source.explicit import (
    REGULATION_TITLE_SUFFIXES,
    TOPICAL_SUFFIX_TERMS,
    GENERIC_DOC_INTENT_TERMS,
    extract_explicit_regulation_mentions,
    regulation_identity_key,
    source_effective_rank,
    prefer_latest_effective_sources,
    strip_reference_text_from_query,
    explicit_content_query,
    geo_filtered_sources,
    prepare_explicit_regulation_candidates,
    resolve_explicit_filename_sources,
    explicit_regulation_unique_result,
    pseudo_singleton_ambiguous_result,
    geo_context_locked_result,
    soft_lock_unique_result,
    topical_suffix_multi_doc_result,
    topical_suffix_match,
    is_topical_suffix_query,
    query_doc_intent,
    query_has_specific_doc_entity,
    multi_doc_topical_downgrade_allowed,
    resolve_topical_suffix_multi_doc,
    explicit_regulation_ambiguous_result,
    resolve_prepared_regulation_candidates,
    collect_unique_match_entries,
    candidate_sources_from_entries,
    apply_resolved_reason_from_entries,
    collect_unique_sources,
    collect_related_title_sources,
    resolve_dense_title_unique,
    document_not_found_result,
    resolve_explicit_regulation_sources,
)

