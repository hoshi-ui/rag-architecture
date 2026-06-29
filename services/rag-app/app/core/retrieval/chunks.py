import re
from typing import Any, Dict, List, Optional


ARTICLE_ANCHOR_RE = re.compile(r"第[一二三四五六七八九十百千万0-9]+[条款项章]")


def article_anchor_terms(query: str) -> List[str]:
    out: List[str] = []
    for match in ARTICLE_ANCHOR_RE.finditer(str(query or "")):
        value = match.group(0)
        if value not in out:
            out.append(value)
    return out[:6]


def hit_sort_value(runtime: Any, hit: Any, score_mode: str) -> float:
    score = float(runtime.hit_score(hit))
    if score_mode == "distance":
        return 1.0 / (1.0 + max(score, 0.0))
    return score


def strict_score_sort(runtime: Any, hits: List[Any], score_mode: str) -> List[Any]:
    return sorted(hits or [], key=lambda hit: hit_sort_value(runtime, hit, score_mode), reverse=True)


def metadata_article_key(metadata: Dict[str, Any]) -> str:
    clause_meta = metadata.get("clause_metadata") if isinstance(metadata.get("clause_metadata"), dict) else {}
    for key in ("article_id", "article_no", "clause_id", "clause_label", "clause"):
        value = metadata.get(key) or clause_meta.get(key)
        if str(value or "").strip():
            return str(value).strip()
    return ""


def fill_missing_metadata(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for key, value in (extra or {}).items():
        if value in (None, "", [], {}, ()):
            continue
        if key not in out or out.get(key) in (None, "", [], {}, ()):
            out[key] = value
    return out


def append_merged_window(
    runtime: Any,
    merged: List[Dict[str, Any]],
    cur: Optional[Dict[str, Any]],
    score_mode: str,
) -> None:
    if not cur:
        return
    texts = [str(item or "").strip() for item in cur.get("texts") or [] if str(item or "").strip()]
    if not texts:
        return
    merged.append(
        {
            "source": cur["source"],
            "section_id": cur.get("section_id"),
            "section": cur.get("section") or "",
            "start": cur.get("start"),
            "end": cur.get("end"),
            "text": "\n".join(texts).strip(),
            "rank": cur["rank"],
            "input_order": int(cur.get("input_order") or 0),
            "metadata": dict(cur.get("metadata") or {}),
        }
    )


def merge_and_dedupe_hits(runtime: Any, hits: List[Any], score_mode: str) -> List[Dict[str, Any]]:
    by_group: Dict[str, List[Any]] = {}
    order_by_identity: Dict[int, int] = {}
    for input_order, h in enumerate(hits):
        order_by_identity[id(h)] = input_order
        src = runtime.hit_entity_source(h) or "unknown"
        md = runtime.hit_metadata(h)
        section_id = md.get("section_node_id") or md.get("section_id")
        section = (md.get("section") or "").strip()
        key = f"{src}||{section_id}||{section}"
        by_group.setdefault(key, []).append(h)

    merged: List[Dict[str, Any]] = []
    for key, hs in by_group.items():
        src, section_id, section = (key.split("||", 2) + ["", "", ""])[:3]
        with_id = []
        without_id = []
        for h in hs:
            cid = runtime.hit_chunk_id(h)
            if cid is None:
                without_id.append(h)
            else:
                with_id.append((cid, h))

        with_id.sort(key=lambda x: x[0])
        max_merge_chunks = max(1, int(runtime.config_value("MAX_MERGED_CHUNKS_PER_EVIDENCE", 2)))
        max_merge_chars = max(200, int(runtime.config_value("MAX_MERGED_EVIDENCE_CHARS", 1800)))
        cur = None
        for cid, h in with_id:
            input_order = order_by_identity.get(id(h), 0)
            text = (runtime.hit_entity_text(h) or "").strip()
            if not text:
                continue
            score = runtime.hit_score(h)
            hit_metadata = dict(runtime.hit_metadata(h) or {})
            hit_article = metadata_article_key(hit_metadata)
            cur_article = metadata_article_key(dict((cur or {}).get("metadata") or {})) if cur else ""
            can_extend = bool(
                cur
                and cid == cur["end"] + 1
                and int(cur.get("chunk_count") or 0) < max_merge_chunks
                and len((cur.get("text") or "") + "\n" + text) <= max_merge_chars
                and (not cur_article or not hit_article or cur_article == hit_article)
            )
            if can_extend:
                cur["texts"].append(text)
                cur["text"] = ("\n".join(cur["texts"])).strip()
                if score_mode == "distance":
                    cur["rank"] = min(cur["rank"], score)
                else:
                    cur["rank"] = max(cur["rank"], score)
                cur["input_order"] = min(int(cur.get("input_order") or input_order), input_order)
                cur["end"] = cid
                cur["chunk_count"] = int(cur.get("chunk_count") or 0) + 1
                cur["metadata"] = fill_missing_metadata(dict(cur.get("metadata") or {}), hit_metadata)
            else:
                if cur:
                    append_merged_window(runtime, merged, cur, score_mode)
                cur = {
                    "source": src,
                    "section_id": section_id if section_id != "None" else None,
                    "section": section,
                    "start": cid,
                    "end": cid,
                    "text": text,
                    "texts": [text],
                    "rank": score,
                    "input_order": input_order,
                    "chunk_count": 1,
                    "metadata": hit_metadata,
                }
        if cur:
            append_merged_window(runtime, merged, cur, score_mode)

        for h in without_id:
            input_order = order_by_identity.get(id(h), 0)
            text = (runtime.hit_entity_text(h) or "").strip()
            if not text:
                continue
            merged.append(
                {
                    "source": src,
                    "section_id": section_id if section_id != "None" else None,
                    "section": section,
                    "start": None,
                    "end": None,
                    "text": text,
                    "rank": runtime.hit_score(h),
                    "input_order": input_order,
                    "metadata": dict(runtime.hit_metadata(h) or {}),
                }
            )

    best_by_text: Dict[str, Dict[str, Any]] = {}
    for m in merged:
        source_key = str(m.get("source") or "").strip().lower()
        text_key = "".join((m["text"] or "").split()).lower()
        key = f"{source_key}||{text_key}"
        if not key:
            continue
        prev = best_by_text.get(key)
        if not prev:
            best_by_text[key] = m
            continue
        if score_mode == "distance":
            if (m["rank"], int(m.get("input_order") or 0)) < (prev["rank"], int(prev.get("input_order") or 0)):
                best_by_text[key] = m
        else:
            if (m["rank"], -int(m.get("input_order") or 0)) > (prev["rank"], -int(prev.get("input_order") or 0)):
                best_by_text[key] = m

    uniq = list(best_by_text.values())
    if score_mode == "distance":
        uniq.sort(key=lambda x: (x["rank"], int(x.get("input_order") or 0)))
    else:
        uniq.sort(key=lambda x: (x["rank"], -int(x.get("input_order") or 0)), reverse=True)

    out: List[Dict[str, Any]] = []
    for m in uniq:
        md = dict(m.get("metadata") or {})
        if m["start"] is not None and m["end"] is not None:
            md["chunk_id_start"] = m["start"]
            md["chunk_id_end"] = m["end"]
            md.setdefault("chunk_id", m["start"])
        if m.get("section"):
            md["section"] = m.get("section")
            md.setdefault("section_title", m.get("section"))
        if m.get("section_id") is not None:
            md["section_id"] = m.get("section_id")
            if isinstance(m.get("section_id"), str) and str(m.get("section_id")).startswith("section::"):
                md["section_node_id"] = m.get("section_id")
        ent = {"source": m["source"], "text": m["text"], "metadata": md}
        item: Dict[str, Any] = {"entity": ent}
        if score_mode == "distance":
            item["distance"] = float(m["rank"])
        else:
            item["score"] = float(m["rank"])
        out.append(item)
    return out


def aggregate_doc_sections(runtime: Any, hits: List[Any], score_mode: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for hit in hits:
        src = runtime.hit_entity_source(hit) or "unknown"
        text = (runtime.hit_entity_text(hit) or "").strip()
        if not text:
            continue
        md = dict(runtime.hit_metadata(hit) or {})
        section = (md.get("section") or "").strip()
        if section:
            md["section"] = section
            md.setdefault("section_title", section)
        ent = {"source": src, "text": text, "metadata": md}
        item: Dict[str, Any] = {"entity": ent}
        if score_mode == "distance":
            item["distance"] = float(runtime.hit_score(hit))
        else:
            item["score"] = float(runtime.hit_score(hit))
        out.append(item)

    if score_mode == "distance":
        out.sort(key=lambda x: runtime.hit_score(x))
    else:
        out.sort(key=lambda x: runtime.hit_score(x), reverse=True)
    return out


def docs_for_query_context(qtype: str, merged_docs: List[Any], aggregated_docs: List[Any]) -> List[Any]:
    if len(merged_docs) <= 8:
        return merged_docs
    if qtype in {"single_doc_extract", "regulation_execution"}:
        return merged_docs
    return aggregated_docs


def chunk_base_relevance(runtime: Any, hit: Any, score_mode: str) -> float:
    score = float(runtime.hit_score(hit))
    if score_mode == "distance":
        return 1.0 / (1.0 + max(score, 0.0))
    return score


def chunk_query_signal(runtime: Any, query: str, hit: Any, score_mode: str) -> tuple:
    md = runtime.hit_metadata(hit)
    src = runtime.normalize_filename_for_match(runtime.hit_entity_source(hit) or "")
    text = runtime.hit_display_text(hit)
    section = (md.get("section_title") or md.get("section") or "").strip()
    title_signal = float(runtime.doc_title_alias_score(src, query))
    section_score = runtime.token_overlap_score(query, section)
    keyword_score = runtime.token_overlap_score(query, text)
    anchor_terms = runtime.query_anchor_terms(query)
    article_terms = article_anchor_terms(query)
    section_term_hits = sum(1 for term in anchor_terms if term and term in section)
    text_term_hits = sum(1 for term in anchor_terms if term and term in text)
    hay = f"{section}\n{text}"
    article_anchor_hits = sum(1 for term in article_terms if term and term in hay)
    if md.get("article_anchor_hit"):
        article_anchor_hits = max(article_anchor_hits, 1)
    title_hit = 1 if md.get("title_hit") or section == "document_title" or title_signal > 0 else 0
    return (
        float(section_term_hits),
        float(text_term_hits),
        float(article_anchor_hits),
        float(section_score),
        float(keyword_score),
        float(title_hit),
        float(title_signal),
        float(chunk_base_relevance(runtime, hit, score_mode)),
    )


def hybrid_structural_chunk_score(
    runtime: Any,
    query: str,
    hit: Any,
    score_mode: str,
    profile: str = "balanced",
    section_anchor_positions: Optional[Dict[str, List[int]]] = None,
    qfilters: Optional[Dict[str, Any]] = None,
) -> tuple:
    section_term_hits, text_term_hits, article_anchor_hits, section_score, keyword_score, title_hit, title_signal, base_rel = chunk_query_signal(runtime, query, hit, score_mode)
    md = runtime.hit_metadata(hit)
    section = (md.get("section_title") or md.get("section") or "").strip()
    text = runtime.hit_display_text(hit)
    pos = runtime.chunk_position_id(hit)
    anchor_terms = runtime.query_anchor_terms(query)
    anchor_cnt = max(1.0, float(len(anchor_terms)))

    section_term_norm = runtime.clip01(section_term_hits / anchor_cnt)
    text_term_norm = runtime.clip01(text_term_hits / anchor_cnt)
    section_overlap_norm = runtime.clip01(section_score / 12.0)
    keyword_overlap_norm = runtime.clip01(keyword_score / 18.0)
    title_norm = runtime.clip01((title_signal + 1.5 * title_hit) / 8.0)
    base_rel_norm = runtime.clip01(base_rel)

    weights = runtime.rerank_profile_weights(profile)
    hybrid_score = (
        weights["section_term"] * section_term_norm
        + weights["text_term"] * text_term_norm
        + weights["section_overlap"] * section_overlap_norm
        + weights["keyword"] * keyword_overlap_norm
        + weights["title"] * title_norm
        + weights["base"] * base_rel_norm
    )

    follow_bonus = runtime.section_follow_bonus(section, pos, section_anchor_positions or {}, profile)
    generic_penalty = runtime.generic_chunk_penalty(section, text, query, text_term_hits, section_term_hits, section_score, profile)
    section_align, section_exact = runtime.section_target_alignment(section, query)
    semantic_terms = list(dict.fromkeys((runtime.query_semantic_aspects(query, qfilters=qfilters or {}).get("terms") or [])))[:6]
    hay = f"{section}\n{text}"
    aspect_hits = [term for term in semantic_terms if term and term in hay]
    topic_bonus = 0.0
    if (qfilters or {}).get("topic") and (qfilters or {}).get("topic") in runtime.normalize_topics(md.get("topics")):
        topic_bonus = float(runtime.config_value("HYBRID_STRUCT_TOPIC_BONUS", 0.20))
    aspect_bonus = min(
        float(runtime.config_value("HYBRID_STRUCT_ASPECT_BONUS_CAP", 0.48)),
        float(runtime.config_value("HYBRID_STRUCT_ASPECT_BONUS", 0.18)) * len(aspect_hits),
    )
    article_anchor_bonus = min(
        float(runtime.config_value("HYBRID_STRUCT_ARTICLE_ANCHOR_BONUS_CAP", 0.75)),
        float(runtime.config_value("HYBRID_STRUCT_ARTICLE_ANCHOR_BONUS", 0.55)) * article_anchor_hits,
    )
    section_match_bonus = 0.0
    section_mismatch_penalty = 0.0
    if profile == "section_lookup":
        body_has_direct_signal = bool(text_term_hits > 0 or keyword_score > 0 or aspect_hits)
        base_match_bonus = float(runtime.config_value("HYBRID_STRUCT_SECTION_MATCH_BONUS", 0.22))
        section_match_bonus = (
            min(base_match_bonus, 0.08) * (0.4 + 0.6 * section_align + 0.2 * section_exact)
            if section_align > 0 and not body_has_direct_signal
            else 0.0
        )
        if runtime.extract_section_query_targets(query) and section_align <= 0 and not body_has_direct_signal:
            section_mismatch_penalty = float(runtime.config_value("HYBRID_STRUCT_SECTION_MISMATCH_PENALTY", 0.12))
    hybrid_score = hybrid_score + follow_bonus + section_match_bonus + topic_bonus + aspect_bonus + article_anchor_bonus - generic_penalty - section_mismatch_penalty

    tie_breaker = (
        float(topic_bonus),
        float(aspect_bonus),
        float(article_anchor_bonus),
        float(section_term_hits),
        float(text_term_hits),
        float(section_score),
        float(keyword_score),
        float(title_hit),
        float(title_signal),
        float(base_rel),
        float(follow_bonus),
        float(-generic_penalty),
        float(section_match_bonus),
        float(-section_mismatch_penalty),
    )
    return float(hybrid_score), tie_breaker


def intra_doc_chunk_rerank(
    runtime: Any,
    query: str,
    hits: List[Any],
    score_mode: str,
    qtype: str = "other",
    qfilters: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    if len(hits) <= 1:
        return hits
    if not bool(runtime.config_value("INTRA_DOC_CHUNK_RERANK_REORDER_ENABLED", False)):
        return hits
    raw_scores = [float(runtime.hit_score(hit) or 0.0) for hit in hits]
    if raw_scores and (max(raw_scores) - min(raw_scores)) <= 1e-6:
        return hits
    ordered_desc = score_mode != "distance"
    profile = runtime.infer_rerank_profile(query, qtype)
    section_anchor_positions_by_source: Dict[str, Dict[str, List[int]]] = {}
    for hit in hits:
        src = runtime.normalize_filename_for_match(runtime.hit_entity_source(hit) or "")
        md = runtime.hit_metadata(hit)
        section = (md.get("section_title") or md.get("section") or "").strip()
        if not section:
            continue
        section_term_hits, text_term_hits, article_anchor_hits, section_score, _, _, _, _ = chunk_query_signal(runtime, query, hit, score_mode)
        if article_anchor_hits > 0 or section_term_hits > 0 or (section_score >= 1.0 and text_term_hits > 0):
            pos = runtime.chunk_position_id(hit)
            if pos is not None:
                section_anchor_positions_by_source.setdefault(src, {}).setdefault(section, []).append(pos)

    decorated = []
    micro_weight = min(0.03, runtime.clip01(runtime.config_value("HYBRID_STRUCT_TIEBREAK_WEIGHT", 0.02)))
    for idx, hit in enumerate(hits):
        src = runtime.normalize_filename_for_match(runtime.hit_entity_source(hit) or "")
        hybrid_score, tie_breaker = hybrid_structural_chunk_score(
            runtime,
            query,
            hit,
            score_mode,
            profile=profile,
            section_anchor_positions=section_anchor_positions_by_source.get(src, {}),
            qfilters=qfilters,
        )
        base_rel = chunk_base_relevance(runtime, hit, score_mode)
        normalized_base = runtime.clip01(base_rel)
        final_score = normalized_base + (micro_weight * runtime.clip01(hybrid_score))
        if hasattr(runtime, "clone_hit_with_score"):
            scored_hit = runtime.clone_hit_with_score(hit, final_score)
            metadata = scored_hit.get("entity", {}).get("metadata", {}) if isinstance(scored_hit, dict) else {}
            if isinstance(metadata, dict):
                metadata["hybrid_struct_score"] = float(hybrid_score)
                metadata["base_relevance_score"] = float(base_rel)
        else:
            scored_hit = hit
        # Preserve upstream global ordering. Structural signals are a narrow
        # tie-breaker inside the same score bucket, never a source-level regroup.
        order_idx = -idx if ordered_desc else idx
        decorated.append((round(float(runtime.hit_score(hit) or 0.0), 6), tie_breaker, order_idx, scored_hit))
    decorated.sort(key=lambda item: item[:3], reverse=ordered_desc)
    return [hit for _, _, _, hit in decorated]


def should_keep_structural_chunk(runtime: Any, query: str, hit: Any, score_mode: str) -> bool:
    q = runtime.normalize_query(query)
    if not q:
        return False
    section = runtime.doc_section_name(hit)
    text = runtime.hit_display_text(hit)
    section_align, _ = runtime.section_target_alignment(section, q)
    section_term_hits, text_term_hits, article_anchor_hits, _, keyword_score, title_hit, title_signal, _ = chunk_query_signal(runtime, q, hit, score_mode)
    semantic_terms = runtime.query_semantic_aspects(q).get("terms") or []
    hay = f"{section}\n{text}"
    if article_anchor_hits > 0:
        return True
    if section_align > 0 or section_term_hits > 0:
        return True
    if text_term_hits > 0 or keyword_score > 0:
        return True
    if any(term and term in hay for term in semantic_terms):
        return True
    if title_hit > 0 and (title_signal > 0 or keyword_score > 0):
        return True
    return False


def filter_low_relevance_sources(
    runtime: Any,
    hits: List[Any],
    score_mode: str,
    query: str = "",
    minimum_keep: Optional[int] = None,
) -> List[Any]:
    if not hits:
        return []

    kept: List[Any] = []
    seen = set()

    def _hit_key(hit: Any) -> tuple:
        md = runtime.hit_metadata(hit)
        return (
            runtime.normalize_filename_for_match(runtime.hit_entity_source(hit) or ""),
            md.get("chunk_id_start"),
            md.get("chunk_id_end"),
            md.get("chunk_id"),
            (runtime.hit_entity_text(hit) or "")[:80],
        )

    def _append_unique(hit: Any):
        key = _hit_key(hit)
        if key in seen:
            return
        seen.add(key)
        kept.append(hit)

    for hit in hits:
        _append_unique(hit)

    return kept


def source_constraint_multiplier(
    runtime: Any,
    src: str,
    query: str,
    fname_set: set,
    allowed_set: set,
    weak_query: bool,
) -> float:
    multiplier = 1.0
    if src in fname_set:
        return max(multiplier, 1.2)
    title_hit = runtime.doc_title_alias_hit(src, query)
    if title_hit:
        multiplier = max(multiplier, float(runtime.config_value("TITLE_CONSTRAINT_BOOST", 1.08)))
    if allowed_set:
        if src in allowed_set:
            multiplier = max(multiplier, float(runtime.config_value("TITLE_CONSTRAINT_BOOST", 1.08)))
        elif weak_query or title_hit:
            multiplier *= float(runtime.config_value("TITLE_CONSTRAINT_PENALTY", 0.82))
    return multiplier
