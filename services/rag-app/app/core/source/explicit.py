import re
from typing import Any, Callable, Dict, List, Optional, Tuple


from app.core.source.results import (
    GENERIC_DOC_INTENT_TERMS,
    TOPICAL_SUFFIX_TERMS,
    explicit_regulation_unique_result,
    pseudo_singleton_ambiguous_result,
    geo_context_locked_result,
    soft_lock_unique_result,
    topical_suffix_multi_doc_result,
    explicit_regulation_ambiguous_result,
    document_not_found_result,
)

REGULATION_TITLE_SUFFIXES = (
    r"\u6761\u4f8b|\u529e\u6cd5|\u89c4\u5b9a|\u89c4\u5219|\u89c4\u7a0b|"
    r"\u7ec6\u5219|\u610f\u89c1|\u901a\u77e5|\u51b3\u5b9a|\u65b9\u6848|"
    r"\u6307\u5357|\u6807\u51c6"
)
TITLE_STRIP_CHARS = "\u300a\u300b\u201c\u201d\"'\uff0c\u3002\uff1b;: \uff1f?\uff01!\u3001 \t\r\n"
LEGAL_TITLE_SUFFIX_RE = re.compile(rf"(?:{REGULATION_TITLE_SUFFIXES})$")
TOPICAL_REFERENCE_SUFFIX_RE = re.compile(
    r"(\u89c4\u5219|\u6761\u4ef6|\u8981\u6c42|\u804c\u8d23|\u6743\u9650|\u6807\u51c6|"
    r"\u6d41\u7a0b|\u671f\u9650|\u673a\u5173|\u5904\u7406|\u5904\u7f5a|\u7f5a\u5219|"
    r"\u8d39\u7528|\u7528\u9014|\u5907\u6848|\u767b\u8bb0|\u5ba1\u6279|\u514d\u75ab|"
    r"\u82af\u7247|\u7981\u517b|\u7528\u706b|\u5165\u5c71)$"
)
QUERY_INTENT_PREFIX_RE = (
    r"^(?:\u8bf7\u95ee|\u67e5\u8be2|\u68c0\u7d22|\u8bf4\u660e|\u4ecb\u7ecd|"
    r"\u603b\u7ed3|\u6982\u62ec|\u5173\u4e8e|\u5e2e\u6211|\u5e2e\u5fd9|"
    r"\u770b\u4e00\u4e0b)"
)
DATE_TOKEN_RE = re.compile(r"\d{4}(?:(?:[-_./]|\u5e74)\d{1,2}(?:(?:[-_./]|\u6708)\d{1,2}\u65e5?)?)?")
SOURCE_DATE_TOKEN_RE = re.compile(r"(20\d{2}|19\d{2})[-_./](\d{1,2})[-_./](\d{1,2})")


def looks_like_legal_title_reference(text: str, *, normalize_query: Callable[[str], str]) -> bool:
    value = normalize_query(text).strip(TITLE_STRIP_CHARS)
    if not value:
        return False
    if re.search(r"^\u7b2c[\u4e00-\u9fff0-9]+\u6761.*\u89c4\u5b9a$", value):
        return False
    if re.search(r"(\u5982\u4f55|\u600e\u4e48|\u662f\u5426|\u5e94\u5f53|\u5e94\u5982\u4f55).*\u89c4\u5b9a$", value):
        return False
    if value.endswith("\u89c4\u5219") and TOPICAL_REFERENCE_SUFFIX_RE.search(value):
        has_title_context = bool(
            re.search(r"[\u7701\u5e02\u53bf\u533a\u5dde\u76df]", value)
            or re.search(r"(\u6761\u4f8b|\u529e\u6cd5|\u89c4\u5b9a|\u5b9e\u65bd|\u7ba1\u7406)", value)
        )
        if not has_title_context:
            return False
    if TOPICAL_REFERENCE_SUFFIX_RE.search(value) and not LEGAL_TITLE_SUFFIX_RE.search(value):
        return False
    if LEGAL_TITLE_SUFFIX_RE.search(value):
        return True
    if re.search(r"[\u300a\u300b]", text or "") and re.search(rf"(?:{REGULATION_TITLE_SUFFIXES})", value):
        return True
    if TOPICAL_REFERENCE_SUFFIX_RE.search(value):
        return False
    return False


def extract_explicit_regulation_mentions(
    query: str,
    *,
    normalize_query: Callable[[str], str],
    extract_filename_candidates: Callable[[str], List[str]],
    limit: int = 6,
) -> List[str]:
    q = normalize_query(query)
    if not q:
        return []
    pattern = re.compile(rf"([\u4e00-\u9fffA-Za-z0-9\u300a\u300b\uff08\uff09()\u3001\u00b7-]{{2,80}}?(?:{REGULATION_TITLE_SUFFIXES}))")
    out: List[str] = []
    filename_tokens = set(extract_filename_candidates(q))
    for bracketed in re.findall(r"\u300a([^\u300b]{2,80})\u300b", q):
        value = normalize_query(bracketed).strip(TITLE_STRIP_CHARS)
        if value and value not in filename_tokens and looks_like_legal_title_reference(value, normalize_query=normalize_query):
            out.append(value)
    for match in pattern.finditer(q):
        value = normalize_query(match.group(1))
        if not value or value in filename_tokens:
            continue
        value = value.strip(TITLE_STRIP_CHARS)
        value = re.sub(QUERY_INTENT_PREFIX_RE, "", value)
        value = value.strip(TITLE_STRIP_CHARS)
        if len(value) < 2 or not looks_like_legal_title_reference(value, normalize_query=normalize_query):
            continue
        if value not in out:
            out.append(value)
        if len(out) >= max(1, int(limit)):
            break
    return out[: max(1, int(limit))]


def regulation_identity_key(
    source: str,
    *,
    normalize_filename: Callable[[str], str],
    source_display_title: Callable[[str], str],
    normalize_reference_text: Callable[[str], str],
    source_profile_fields: Callable[[str], Dict[str, Any]],
    normalize_query: Callable[[str], str],
    extract_region_token: Callable[[str], str],
) -> str:
    safe_source = normalize_filename(source or "")
    if not safe_source:
        return ""
    title = source_display_title(safe_source)
    profile = source_profile_fields(safe_source)
    profile_title = str(profile.get("canonical_title") or "").strip()
    if profile_title:
        title = profile_title
    cleaned_title = DATE_TOKEN_RE.sub(" ", title)
    cleaned_title = re.sub(r"(\u73b0\u884c\u6709\u6548|\u6700\u65b0|\u4fee\u8ba2|\u8bd5\u884c|\u6682\u884c)", " ", cleaned_title)
    normalized_title = normalize_reference_text(cleaned_title)
    region = normalize_query(profile.get("region") or extract_region_token(title))
    return f"{region}|{normalized_title}".strip("|")


def _sortable_date_from_text(text: str) -> int:
    best = 0
    for year, month, day in SOURCE_DATE_TOKEN_RE.findall(str(text or "")):
        try:
            value = int(f"{int(year):04d}{int(month):02d}{int(day):02d}")
        except Exception:
            continue
        best = max(best, value)
    return best


def source_effective_rank(
    source: str,
    *,
    normalize_filename: Callable[[str], str],
    doc_get: Callable[[str], Dict[str, Any]],
    source_profile_fields: Callable[[str], Dict[str, Any]],
    source_display_title: Callable[[str], str],
    normalize_query: Callable[[str], str],
) -> Tuple[int, int, int, int, str]:
    safe_source = normalize_filename(source or "")
    if not safe_source:
        return (0, 0, 0, 0, "")
    info = doc_get(safe_source)
    profile = source_profile_fields(safe_source)
    display_title = source_display_title(safe_source)
    version_label = normalize_query(profile.get("doc_version_label") or "")
    effective_date = normalize_query(profile.get("effective_date") or "")
    publish_date = normalize_query(profile.get("publish_date") or "")
    current_marker = 1 if any(token in f"{display_title} {version_label}" for token in ["\u73b0\u884c\u6709\u6548", "\u6700\u65b0"]) else 0
    try:
        active_version = int(info.get("active_version") or 0)
    except Exception:
        active_version = 0
    effective_sort = int(re.sub(r"\D", "", effective_date) or "0")
    publish_sort = int(re.sub(r"\D", "", publish_date) or "0")
    filename_sort = _sortable_date_from_text(safe_source)
    if not effective_sort:
        effective_sort = filename_sort
    if not publish_sort:
        publish_sort = filename_sort
    return (current_marker, effective_sort, publish_sort, active_version, safe_source)


def prefer_latest_effective_sources(
    sources: List[str],
    *,
    normalize_filename: Callable[[str], str],
    regulation_identity_key: Callable[[str], str],
    source_effective_rank: Callable[[str], Tuple[int, int, int, int, str]],
    limit: Optional[int] = None,
) -> List[str]:
    grouped: Dict[str, str] = {}
    ordered_keys: List[str] = []
    for source in sources or []:
        safe_source = normalize_filename(source or "")
        if not safe_source:
            continue
        identity_key = regulation_identity_key(safe_source) or f"source:{safe_source}"
        if identity_key not in grouped:
            grouped[identity_key] = safe_source
            ordered_keys.append(identity_key)
            continue
        if source_effective_rank(safe_source) > source_effective_rank(grouped[identity_key]):
            grouped[identity_key] = safe_source

    out: List[str] = []
    for key in ordered_keys:
        source = grouped.get(key) or ""
        if source:
            out.append(source)
        if limit and len(out) >= max(1, int(limit)):
            break
    return out


def strip_reference_text_from_query(
    query: str,
    references: List[str],
    normalize_query: Callable[[str], str],
) -> str:
    text = normalize_query(query)
    for reference in references or []:
        normalized = normalize_query(reference)
        if normalized:
            text = text.replace(normalized, " ")
        raw = (reference or "").strip()
        if raw:
            text = text.replace(raw, " ")
    return normalize_query(re.sub(r"\s+", " ", text))


def explicit_content_query(
    query: str,
    regulation_mentions: List[str],
    *,
    normalize_query: Callable[[str], str],
    strip_reference_text: Callable[[str, List[str]], str],
) -> str:
    stripped = strip_reference_text(query, regulation_mentions)
    stripped = re.sub(
        QUERY_INTENT_PREFIX_RE,
        " ",
        stripped,
    )
    stripped = re.sub(r"\u300a\s*\u300b", " ", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped or normalize_query(query)


def geo_filtered_sources(
    query: str,
    user_id: str,
    candidate_sources: List[str],
    *,
    geo_context_tokens: Callable[[str, str], List[str]],
    normalize_filename: Callable[[str], str],
    source_profile_fields: Callable[[str], Dict[str, Any]],
    normalize_query: Callable[[str], str],
    extract_region_token: Callable[[str], str],
    source_display_title: Callable[[str], str],
) -> List[str]:
    geo_tokens = geo_context_tokens(query, user_id)
    if not geo_tokens:
        return []
    matched: List[str] = []
    for source in candidate_sources or []:
        safe_source = normalize_filename(source or "")
        if not safe_source:
            continue
        profile = source_profile_fields(safe_source)
        region = normalize_query(profile.get("region") or extract_region_token(source_display_title(safe_source)))
        if not region:
            continue
        if any(token == region or token in region or region in token for token in geo_tokens):
            if safe_source not in matched:
                matched.append(safe_source)
    return matched


def prepare_explicit_regulation_candidates(
    candidate_sources: List[str],
    *,
    normalize_filename: Callable[[str], str],
    prefer_latest_effective: Callable[[List[str], Optional[int]], List[str]],
    limit: int = 5,
) -> List[str]:
    unique_sources = [
        normalize_filename(source or "")
        for source in candidate_sources
        if normalize_filename(source or "")
    ]
    unique_sources = list(dict.fromkeys(unique_sources))
    return prefer_latest_effective(unique_sources, limit)


def resolve_explicit_filename_sources(
    fnames: List[str],
    *,
    normalize_filename: Callable[[str], str],
    collapse_sources_by_canonical: Callable[[List[str], Optional[int]], List[str]],
    document_existence_matches: Callable[[List[str]], List[str]],
    build_document_clarification_prompt: Callable[[List[str]], str],
) -> Optional[Dict[str, Any]]:
    explicit_sources = [
        normalize_filename(name or "")
        for name in (fnames or [])
        if normalize_filename(name or "")
    ]
    if not explicit_sources:
        return None

    unique_sources = collapse_sources_by_canonical(list(dict.fromkeys(explicit_sources)), 3)
    matched_sources = document_existence_matches(unique_sources)
    if len(unique_sources) == 1 and matched_sources:
        return {
            "route": "explicit_doc_reference",
            "required": True,
            "resolved": True,
            "sources": matched_sources,
            "candidates": matched_sources,
            "reason": "explicit_filename_unique",
            "strip_title_mentions": True,
            "clarification": "",
            "target_text": unique_sources[0],
        }
    if len(unique_sources) > 1:
        return {
            "route": "explicit_doc_reference",
            "required": True,
            "resolved": False,
            "sources": [],
            "candidates": unique_sources[:3],
            "reason": "document_ambiguous",
            "strip_title_mentions": False,
            "clarification": build_document_clarification_prompt(unique_sources[:3]),
            "target_text": "\u3001".join(unique_sources[:3]),
        }
    return {
        "route": "explicit_doc_reference",
        "required": True,
        "resolved": False,
        "sources": [],
        "candidates": [],
        "reason": "document_not_found",
        "strip_title_mentions": False,
        "clarification": "",
        "target_text": unique_sources[0] if unique_sources else "",
    }







def topical_suffix_match(query: str, *, normalize_query: Callable[[str], str]) -> str:
    normalized = normalize_query(query).replace(" ", "")
    if not normalized:
        return ""
    for term in TOPICAL_SUFFIX_TERMS:
        if normalize_query(term).replace(" ", "") in normalized:
            return term
    match = re.search(r"(澶勭綒|缃氭|璐ｄ换|瀹℃壒|澶囨|璁稿彲|娴佺▼|瑕佹眰|鏉′欢|鏍囧噯|鑼冨洿|绠＄悊)$", normalized)
    return match.group(1) if match else ""


def is_topical_suffix_query(query: str, *, normalize_query: Callable[[str], str]) -> bool:
    return bool(topical_suffix_match(query, normalize_query=normalize_query))


def query_doc_intent(
    query: str,
    *,
    normalize_query: Callable[[str], str],
    extract_explicit_regulation_mentions: Callable[[str], List[str]],
) -> str:
    mentions = extract_explicit_regulation_mentions(query)
    base_text = normalize_query(mentions[0] if mentions else query)
    suffix = topical_suffix_match(base_text, normalize_query=normalize_query)
    if suffix:
        base_text = base_text.replace(normalize_query(suffix), " ")
    base_text = re.sub(r"(鏄粈涔坾鏈夊摢浜泑鎬庝箞|濡備綍|鐩稿叧|鏈夊叧|涓昏|鍏蜂綋)$", " ", base_text)
    for term in GENERIC_DOC_INTENT_TERMS:
        base_text = re.sub(re.escape(normalize_query(term)) + r"$", " ", base_text)
    return normalize_query(base_text)


def query_has_specific_doc_entity(
    query: str,
    *,
    doc_intent: str = "",
    normalize_query: Callable[[str], str],
    extract_region_token: Callable[[str], str],
) -> bool:
    normalized = normalize_query(query)
    if not normalized:
        return False
    if extract_region_token(normalized):
        return True
    for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{4,}", normalized):
        value = normalize_query(token)
        if not value or value == normalize_query(doc_intent):
            continue
        if value in GENERIC_DOC_INTENT_TERMS:
            continue
        if topical_suffix_match(value, normalize_query=normalize_query):
            continue
        if value in normalized:
            return True
    return False


def multi_doc_topical_downgrade_allowed(
    query: str,
    candidate_sources: List[str],
    *,
    normalize_query: Callable[[str], str],
    extract_explicit_regulation_mentions: Callable[[str], List[str]],
    extract_region_token: Callable[[str], str],
) -> tuple[bool, Dict[str, Any]]:
    doc_intent = query_doc_intent(
        query,
        normalize_query=normalize_query,
        extract_explicit_regulation_mentions=extract_explicit_regulation_mentions,
    )
    if not doc_intent:
        return True, {"doc_intent": "", "blocked": False}
    generic_intent = doc_intent in {normalize_query(term) for term in GENERIC_DOC_INTENT_TERMS}
    specific_entity = query_has_specific_doc_entity(
        query,
        doc_intent=doc_intent,
        normalize_query=normalize_query,
        extract_region_token=extract_region_token,
    )
    blocked = generic_intent and not specific_entity
    return (not blocked), {
        "doc_intent": doc_intent,
        "generic_doc_intent": generic_intent,
        "specific_entity_present": specific_entity,
        "candidate_count": len(candidate_sources or []),
        "blocked": blocked,
    }


def resolve_topical_suffix_multi_doc(
    query: str,
    candidate_sources: List[str],
    *,
    collapse_sources_by_canonical: Callable[[List[str], Optional[int]], List[str]],
    normalize_query: Callable[[str], str],
    extract_explicit_regulation_mentions: Callable[[str], List[str]],
    extract_region_token: Callable[[str], str],
    source_profile_fields: Callable[[str], Dict[str, Any]],
    doc_get: Callable[[str], Dict[str, Any]],
) -> Dict[str, Any]:
    candidates = collapse_sources_by_canonical(candidate_sources, 3)
    if len(candidates) <= 1 or not is_topical_suffix_query(query, normalize_query=normalize_query):
        return {"resolved": False}
    allowed, gate_trace = multi_doc_topical_downgrade_allowed(
        query,
        candidates,
        normalize_query=normalize_query,
        extract_explicit_regulation_mentions=extract_explicit_regulation_mentions,
        extract_region_token=extract_region_token,
    )
    if not allowed:
        return {"resolved": False, "blocked_reason": "generic_doc_intent", "trace": gate_trace}
    doc_types = {
        normalize_query(source_profile_fields(source).get("doc_type") or doc_get(source).get("doc_type") or "")
        for source in candidates
    }
    doc_types = {item for item in doc_types if item}
    if len(doc_types) > 1:
        return {"resolved": False}
    return {
        "resolved": True,
        "sources": candidates[:3],
        "reason": "topical_suffix_multi_doc",
        "trace": {"doc_types": list(doc_types), "topical": True, **gate_trace},
    }



def resolve_prepared_regulation_candidates(
    raw_candidates: List[str],
    *,
    query: str,
    user_id: str,
    target_text: str,
    content_query: str,
    allow_soft_lock: bool,
    trace_label: str,
    prepare_candidates: Callable[[List[str], int], List[str]],
    latest_effective_equivalent_source: Callable[[str], str],
    is_pseudo_singleton_soft_lock: Callable[[str, str], bool],
    extract_region_token: Callable[[str], str],
    normalize_query: Callable[[str], str],
    geo_context_tokens: Callable[[str, str], List[str]],
    geo_filtered_sources_fn: Callable[[str, str, List[str]], List[str]],
    resolve_unique_weak_match_upgrade: Callable[[str, List[str]], Dict[str, Any]],
    resolve_topical_suffix_multi_doc: Callable[[str, List[str]], Dict[str, Any]],
    build_document_clarification_prompt: Callable[[List[str]], str],
    source_display_title: Callable[[str], str],
) -> Optional[Dict[str, Any]]:
    prepared_sources = prepare_candidates(raw_candidates, 5)
    if len(prepared_sources) == 1:
        base_source = prepared_sources[0]
        resolved_source = latest_effective_equivalent_source(base_source) or base_source
        if is_pseudo_singleton_soft_lock(query, resolved_source):
            if extract_region_token(normalize_query(query)) or geo_context_tokens(query, user_id):
                return pseudo_singleton_ambiguous_result(
                    resolved_source=resolved_source,
                    raw_candidates=raw_candidates,
                    prepared_sources=prepared_sources,
                    target_text=target_text,
                    content_query=content_query,
                    trace_label=trace_label,
                    build_document_clarification_prompt=build_document_clarification_prompt,
                )
        return explicit_regulation_unique_result(
            resolved_source=resolved_source,
            base_source=base_source,
            raw_candidates=raw_candidates,
            prepared_sources=prepared_sources,
            target_text=target_text,
            content_query=content_query,
            trace_label=trace_label,
        )

    geo_filtered = prepare_candidates(geo_filtered_sources_fn(query, user_id, prepared_sources), 5)
    if len(geo_filtered) == 1:
        resolved_source = geo_filtered[0]
        return geo_context_locked_result(
            resolved_source=resolved_source,
            raw_candidates=raw_candidates,
            prepared_sources=prepared_sources,
            geo_filtered=geo_filtered,
            target_text=target_text,
            content_query=content_query,
            trace_label=trace_label,
            source_display_title=source_display_title,
        )

    if allow_soft_lock:
        unique_weak = resolve_unique_weak_match_upgrade(query, prepared_sources)
        if unique_weak.get("resolved"):
            return soft_lock_unique_result(
                unique_weak=unique_weak,
                raw_candidates=raw_candidates,
                prepared_sources=prepared_sources,
                target_text=target_text,
                content_query=content_query,
                trace_label=trace_label,
            )

    if prepared_sources:
        topical_multi = resolve_topical_suffix_multi_doc(query, prepared_sources)
        if topical_multi.get("resolved"):
            return topical_suffix_multi_doc_result(
                topical_multi=topical_multi,
                raw_candidates=raw_candidates,
                prepared_sources=prepared_sources,
                target_text=target_text,
                content_query=content_query,
                trace_label=trace_label,
            )
        return explicit_regulation_ambiguous_result(
            raw_candidates=raw_candidates,
            prepared_sources=prepared_sources,
            geo_filtered=geo_filtered,
            target_text=target_text,
            content_query=content_query,
            trace_label=trace_label,
            build_document_clarification_prompt=build_document_clarification_prompt,
        )
    return None


def collect_unique_match_entries(
    mentions: List[str],
    *,
    matcher: Callable[[str, int], List[Dict[str, Any]]],
    normalize_filename: Callable[[str], str],
    default_reason: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for mention in mentions or []:
        for entry in matcher(mention, limit):
            source = normalize_filename(entry.get("source") or "")
            if source and not any(item.get("source") == source for item in matches):
                item = dict(entry)
                item["source"] = source
                item["reason"] = str(entry.get("reason") or default_reason)
                matches.append(item)
    return matches


def candidate_sources_from_entries(entries: List[Dict[str, Any]]) -> List[str]:
    return [str(entry.get("source") or "") for entry in entries if str(entry.get("source") or "")]


def strongest_unique_title_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(entries or []) <= 1:
        return list(entries or [])
    strong_entries = [
        entry
        for entry in entries
        if str(entry.get("match_kind") or "") in {"exact_title", "alias_title"}
    ]
    if not strong_entries:
        return list(entries or [])
    best_score = max(float(entry.get("score") or 0.0) for entry in strong_entries)
    best_entries = [
        entry
        for entry in strong_entries
        if abs(float(entry.get("score") or 0.0) - best_score) <= 1e-9
    ]
    return best_entries if len(best_entries) == 1 else list(entries or [])


def apply_resolved_reason_from_entries(
    resolution: Dict[str, Any],
    entries: List[Dict[str, Any]],
    *,
    default_reason: str,
) -> Dict[str, Any]:
    if not resolution:
        return resolution
    resolved_source = (resolution.get("sources") or [""])[0]
    resolved_reason = next(
        (
            str(entry.get("reason") or default_reason)
            for entry in entries
            if str(entry.get("source") or "") == resolved_source
        ),
        resolution.get("reason") or default_reason,
    )
    if resolution.get("resolved") and resolution.get("reason") != "latest_effective_unique":
        resolution["reason"] = resolved_reason
    return resolution


def collect_unique_sources(
    mentions: List[str],
    *,
    matcher: Callable[[str, int], List[str]],
    limit: int = 5,
) -> List[str]:
    sources: List[str] = []
    for mention in mentions or []:
        for source in matcher(mention, limit):
            if source and source not in sources:
                sources.append(source)
    return sources


def collect_related_title_sources(
    mentions: List[str],
    *,
    extract_title_source_candidates: Callable[[str, int], List[str]],
    related_marker: str,
    limit: int = 5,
) -> List[str]:
    sources: List[str] = []
    for mention in mentions or []:
        if related_marker not in mention:
            continue
        normalized_mention = mention.replace(related_marker, "").strip()
        for source in extract_title_source_candidates(normalized_mention, limit):
            if source and source not in sources:
                sources.append(source)
    return sources


def resolve_dense_title_unique(
    dense_title_matches: List[Dict[str, Any]],
    *,
    min_sim: float,
    min_margin: float,
    extra_margin: float = 0.05,
    resolve_prepared_candidates: Callable[[List[str], bool, str], Optional[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    if not dense_title_matches:
        return None
    top_match = dense_title_matches[0]
    top_score = float(top_match.get("score") or 0.0)
    second_score = float(dense_title_matches[1].get("score") or 0.0) if len(dense_title_matches) > 1 else 0.0
    if not (top_score >= min_sim and (top_score - second_score >= min_margin or top_score >= min_sim + float(extra_margin))):
        return None
    dense_resolution = resolve_prepared_candidates([str(top_match.get("source") or "")], True, "dense_title_match")
    if not dense_resolution:
        return None
    trace = dict(dense_resolution.get("source_resolution_trace") or {})
    trace["dense_title_scores"] = [
        {
            "source": item.get("source"),
            "title": item.get("title"),
            "score": item.get("score"),
        }
        for item in dense_title_matches[:3]
    ]
    dense_resolution["source_resolution_trace"] = trace
    if dense_resolution.get("resolved"):
        dense_resolution["reason"] = "dense_title_unique"
    return dense_resolution



def resolve_explicit_regulation_sources(
    regulation_mentions: List[str],
    *,
    resolve_prepared_candidates: Callable[[List[str], bool, str], Optional[Dict[str, Any]]],
    exact_title_or_alias_source_matches: Callable[[str, int], List[Dict[str, Any]]],
    exclusive_entity_source_matches: Callable[[str, int], List[Dict[str, Any]]],
    match_sources_for_explicit_title: Callable[[str, int], List[str]],
    extract_title_source_candidates: Callable[[str, int], List[str]],
    normalized_title_candidate_sources: Callable[[str, int], List[str]],
    dense_title_source_matches: Callable[[str, int], List[Dict[str, Any]]],
    normalize_filename: Callable[[str], str],
    dense_title_match_min_sim: float,
    dense_title_match_margin: float,
    dense_title_extra_margin: float = 0.05,
    related_marker: str = "鐩稿叧",
) -> Dict[str, Any]:
    target_text = regulation_mentions[0] if regulation_mentions else ""

    strong_matches = collect_unique_match_entries(
        regulation_mentions,
        matcher=exact_title_or_alias_source_matches,
        normalize_filename=normalize_filename,
        default_reason="explicit_regulation_unique",
    )
    strong_matches = strongest_unique_title_entries(strong_matches)
    strong_resolution = resolve_prepared_candidates(candidate_sources_from_entries(strong_matches), False, "strong_match")
    if strong_resolution:
        return apply_resolved_reason_from_entries(
            strong_resolution,
            strong_matches,
            default_reason="explicit_regulation_unique",
        )

    entity_matches = collect_unique_match_entries(
        regulation_mentions,
        matcher=exclusive_entity_source_matches,
        normalize_filename=normalize_filename,
        default_reason="exclusive_entity_unique",
    )
    entity_resolution = resolve_prepared_candidates(candidate_sources_from_entries(entity_matches), False, "entity_match")
    if entity_resolution:
        return apply_resolved_reason_from_entries(
            entity_resolution,
            entity_matches,
            default_reason="exclusive_entity_unique",
        )

    candidate_sources = collect_unique_sources(regulation_mentions, matcher=match_sources_for_explicit_title)
    prepared_candidate_resolution = resolve_prepared_candidates(candidate_sources, False, "explicit_title_match")
    if prepared_candidate_resolution:
        return prepared_candidate_resolution

    related_candidates = collect_related_title_sources(
        regulation_mentions,
        extract_title_source_candidates=extract_title_source_candidates,
        related_marker=related_marker,
    )
    related_resolution = resolve_prepared_candidates(related_candidates, False, "related_title_match")
    if related_resolution:
        return related_resolution

    normalized_title_candidates = normalized_title_candidate_sources(target_text, 5)
    if normalized_title_candidates:
        normalized_resolution = resolve_prepared_candidates(normalized_title_candidates, True, "normalized_title_match")
        if normalized_resolution:
            return normalized_resolution

    fallback_candidates = extract_title_source_candidates(target_text, 5)
    if fallback_candidates:
        fallback_resolution = resolve_prepared_candidates(fallback_candidates, True, "fallback_title_match")
        if fallback_resolution:
            return fallback_resolution

    dense_resolution = resolve_dense_title_unique(
        dense_title_source_matches(target_text, 5),
        min_sim=dense_title_match_min_sim,
        min_margin=dense_title_match_margin,
        extra_margin=dense_title_extra_margin,
        resolve_prepared_candidates=resolve_prepared_candidates,
    )
    if dense_resolution:
        return dense_resolution

    return document_not_found_result(target_text)

