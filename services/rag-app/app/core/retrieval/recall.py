import re
from typing import Any, Callable, Dict, List, Optional

from app.core.retrieval.query import doc_recall_fallback
from app.utils import scoring as scoring_utils


def rrf(rank: Optional[int], k: int = 60) -> float:
    if rank is None:
        return 0.0
    try:
        return 1.0 / (float(k) + float(rank) + 1.0)
    except Exception:
        return 0.0


def doc_term_overlap_recall(
    query: str,
    limit: int,
    *,
    query_match_terms: Callable[[str], List[str]],
    profile_source_recall: Callable[..., Dict[str, Dict[str, Any]]],
    scan_chunk_text_rows: Callable[..., List[Dict[str, Any]]],
    normalize_filename: Callable[[str], str],
    source_state: Callable[[str], Dict[str, Any]],
    source_filter: Optional[str] = None,
    chunk_scan_limit: int = 400,
) -> Dict[str, Dict[str, Any]]:
    terms = query_match_terms(query)
    profile_scores = profile_source_recall(
        query,
        limit=max(int(limit) * 2, 12),
        source_filter=source_filter,
    )
    if not terms:
        return profile_scores
    rows = scan_chunk_text_rows(
        terms,
        source_filter=source_filter,
        limit=max(int(limit) * 40, int(chunk_scan_limit)),
    )
    by_source: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        source = row.get("source")
        text = row.get("text")
        safe_source = normalize_filename(source or "")
        if not safe_source:
            continue
        state = source_state(safe_source)
        if not state.get("visible"):
            continue
        matched_terms = [term for term in terms if term in (text or "")]
        if not matched_terms:
            continue
        info = by_source.setdefault(safe_source, {"matched_terms": [], "hit_count": 0, "score": 0.0})
        info["hit_count"] += 1
        for term in matched_terms:
            if term not in info["matched_terms"]:
                info["matched_terms"].append(term)
                info["score"] += max(0.8, min(len(term), 8) / 3.0)
        info["score"] += min(len(matched_terms), 3) * 0.15

    ranked: Dict[str, Dict[str, Any]] = {}
    for source, info in by_source.items():
        coverage = float(len(info["matched_terms"])) / float(max(len(terms), 1))
        ranked[source] = {
            "score": float(info["score"] + coverage),
            "matched_terms": info["matched_terms"],
            "hit_count": int(info["hit_count"]),
            "coverage": coverage,
        }
    for source, info in profile_scores.items():
        current = ranked.setdefault(source, {"score": 0.0, "matched_terms": [], "hit_count": 0, "coverage": 0.0})
        current["score"] = float(current.get("score", 0.0)) + float(info.get("score", 0.0))
        current["hit_count"] = int(current.get("hit_count", 0)) + int(info.get("hit_count", 0))
        current["coverage"] = max(float(current.get("coverage", 0.0)), float(info.get("coverage", 0.0)))
        for term in info.get("matched_terms") or []:
            if term not in current["matched_terms"]:
                current["matched_terms"].append(term)
        if info.get("reasons"):
            current["reasons"] = list(dict.fromkeys(list(current.get("reasons") or []) + list(info.get("reasons") or [])))
    return {
        source: info
        for source, info in sorted(ranked.items(), key=lambda item: (-float(item[1].get("score", 0.0)), item[0]))[: int(limit)]
    }


def build_doc_recall_plan(
    query: str,
    limit: int,
    *,
    normalize_query: Callable[[str], str],
    normalize_filename: Callable[[str], str],
    document_fts_match_filenames: Callable[[str, int], List[str]],
    document_fts_rows: Callable[[], List[Dict[str, Any]]],
    source_state: Callable[[str], Dict[str, Any]],
    doc_title_alias_score: Callable[[str, str], float],
    query_match_terms: Callable[[str], List[str]],
    profile_source_recall: Callable[..., Dict[str, Dict[str, Any]]],
    scan_chunk_text_rows: Callable[..., List[Dict[str, Any]]],
    doc_fallback_min_prior: float = 0.18,
    chunk_scan_limit: int = 400,
    source_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    normalized_query = normalize_query(query)
    if not normalized_query:
        return []
    indexed_rank: Dict[str, int] = {}
    try:
        rows = document_fts_match_filenames(normalized_query, max(int(limit) * 3, 12))
        for idx, filename in enumerate(rows):
            source = normalize_filename(filename or "")
            if source and source not in indexed_rank:
                indexed_rank[source] = idx
    except Exception:
        pass

    overlap_scores = doc_term_overlap_recall(
        normalized_query,
        limit=max(int(limit) * 2, 12),
        query_match_terms=query_match_terms,
        profile_source_recall=profile_source_recall,
        scan_chunk_text_rows=scan_chunk_text_rows,
        normalize_filename=normalize_filename,
        source_state=source_state,
        source_filter=source_filter,
        chunk_scan_limit=chunk_scan_limit,
    )
    plan: List[Dict[str, Any]] = []
    for row in document_fts_rows():
        filename = row.get("filename")
        title = row.get("title")
        aliases = row.get("aliases")
        doc_type = row.get("doc_type")
        topic = row.get("topic")
        filename_stem = row.get("filename_stem")
        source = normalize_filename(filename or "")
        if not source:
            continue
        if source_filter and source != source_filter:
            continue
        state = source_state(source)
        if not state.get("visible"):
            continue
        title_text = "\n".join([title or "", aliases or "", filename_stem or "", doc_type or "", topic or "", source])
        title_score = doc_title_alias_score(source, normalized_query)
        label_overlap = scoring_utils.token_overlap_score(normalized_query, title_text, query_match_terms)
        indexed_score = rrf(indexed_rank.get(source), max(10, int(limit) * 6)) if source in indexed_rank else 0.0
        overlap_info = overlap_scores.get(source) or {}
        overlap_score = float(overlap_info.get("score", 0.0))
        reasons: List[str] = []
        if title_score > 0:
            reasons.append("title_alias_substring")
        if label_overlap > 0:
            reasons.append("doc_label_overlap")
        if source in indexed_rank:
            reasons.append("documents_fts")
        if overlap_info:
            reasons.append("doc_term_overlap")
        for reason in overlap_info.get("reasons") or []:
            if reason not in reasons:
                reasons.append(reason)
        raw_score = (
            min(title_score / 6.0, 1.0) * 1.25
            + min(label_overlap / 8.0, 1.0) * 0.55
            + min(overlap_score / 8.0, 1.0) * 1.10
            + indexed_score * 8.0
        )
        if source_filter and source == source_filter:
            raw_score += 0.2
        prior = min(raw_score / 2.8, 1.0)
        if prior < float(doc_fallback_min_prior):
            continue
        plan.append({
            "source": source,
            "prior": prior,
            "raw_score": raw_score,
            "reasons": reasons,
            "title_score": title_score,
            "label_overlap": label_overlap,
            "indexed_rank": indexed_rank.get(source),
            "matched_terms": overlap_info.get("matched_terms") or [],
            "term_overlap_score": overlap_score,
            "term_overlap_hits": int(overlap_info.get("hit_count", 0)),
        })
    plan.sort(key=lambda item: (-float(item.get("prior", 0.0)), -float(item.get("raw_score", 0.0)), item.get("source") or ""))
    return plan[: int(limit)]


def clarification_probe_terms(
    query: str,
    *,
    normalize_query: Callable[[str], str],
    query_anchor_terms: Callable[[str], List[str]],
    query_match_terms: Callable[[str], List[str]],
) -> List[str]:
    normalized = normalize_query(query)
    if not normalized:
        return []
    terms: List[str] = []

    def add(term: str) -> None:
        value = normalize_query(term)
        if len(value) >= 2 and value not in terms:
            terms.append(value)

    for candidate in list(query_anchor_terms(normalized) or []) + list(query_match_terms(normalized) or []) + [normalized]:
        text = normalize_query(candidate)
        if not text:
            continue
        add(text)
        trimmed = re.sub(r"(是什么|有哪些|怎么|如何|相关|内容|规定|要求)$", "", text)
        add(trimmed)
    return sorted(terms, key=len, reverse=True)[:8]


def clarification_chunk_candidate_sources(
    query: str,
    limit: int,
    *,
    source_hit_counts_by_like: Callable[[List[str], int], List[Dict[str, Any]]],
    normalize_filename: Callable[[str], str],
    source_state: Callable[[str], Dict[str, Any]],
    probe_terms: Callable[[str], List[str]],
) -> List[str]:
    terms = probe_terms(query)
    if not terms:
        return []
    scored: Dict[str, float] = {}
    for row in source_hit_counts_by_like(terms, max(10, int(limit) * 6)):
        term = str(row.get("term") or "")
        source = row.get("source")
        hit_count = row.get("hit_count")
        safe_source = normalize_filename(source or "")
        if not safe_source or not source_state(safe_source).get("visible"):
            continue
        scored[safe_source] = scored.get(safe_source, 0.0) + float(hit_count or 0) * max(1.0, min(float(len(term)), 6.0))
    ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    return [source for source, _ in ranked[: max(1, int(limit))]]


def retrieval_backed_clarification_candidates(
    query: str,
    *,
    seed_sources: Optional[List[str]],
    limit: int,
    normalize_filename: Callable[[str], str],
    source_state: Callable[[str], Dict[str, Any]],
    chunk_candidate_sources: Callable[[str, int], List[str]],
    build_doc_recall_plan: Callable[[str, int], List[Dict[str, Any]]],
) -> List[str]:
    out: List[str] = []
    max_limit = max(1, int(limit))
    for source in seed_sources or []:
        safe_source = normalize_filename(source or "")
        if safe_source and source_state(safe_source).get("visible") and safe_source not in out:
            out.append(safe_source)
    for source in chunk_candidate_sources(query, max(max_limit * 2, 6)):
        if source not in out:
            out.append(source)
        if len(out) >= max_limit:
            return out[:max_limit]
    for entry in build_doc_recall_plan(query, max(max_limit * 2, 6)):
        safe_source = normalize_filename((entry or {}).get("source") or "")
        if safe_source and safe_source not in out:
            out.append(safe_source)
        if len(out) >= max_limit:
            break
    return out[:max_limit]
