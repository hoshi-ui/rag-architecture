import re
from typing import Any, Dict, List, Optional, Tuple

from app.core.evidence.context import _evidence_context
from app.core.evidence.hits import (
    _to_int,
    chunk_position_id,
    hit_entity_source,
    hit_entity_text,
    hit_metadata,
    hit_score_mode,
)
from app.documents import chunking as document_chunking


def _retrieve_output_key(doc: Any) -> Tuple[Any, ...]:
    metadata = hit_metadata(doc)
    source = str(hit_entity_source(doc) or "").strip()
    article_id = document_chunking.extract_article_id(
        metadata.get("article_id"),
        metadata.get("article_no"),
        metadata.get("clause_label"),
    )
    if article_id:
        return ("article", source, article_id)
    start = metadata.get("chunk_id_start")
    end = metadata.get("chunk_id_end")
    if start is not None or end is not None:
        return ("range", source, start, end)
    chunk_id = metadata.get("chunk_id")
    if chunk_id is not None:
        return ("chunk", source, chunk_id)
    text = re.sub(r"\s+", " ", hit_entity_text(doc) or "").strip()
    return ("text", source, text[:160])


def select_retrieve_output_docs(docs: List[Any], top_k: int, default_n: int) -> List[Any]:
    if not docs:
        return []
    keep_n = min(len(docs), min(max(int(top_k or default_n), 3), 8))
    seen = set()
    out: List[Any] = []
    for doc in docs:
        key = _retrieve_output_key(doc)
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
        if len(out) >= keep_n:
            break
    return out


def filter_display_sources(
    runtime: Any,
    docs: List[Any],
    score_mode: str,
    qfilters: Dict[str, Any],
    fnames: List[str],
    qtype: str,
    max_sources: int = 10,
    target_sources: Optional[List[str]] = None,
    observations: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    adapter = _evidence_context(runtime)
    if not docs:
        return []
    items = docs[:]
    effective_targets = list(target_sources or fnames or [])
    if effective_targets:
        source_set = {adapter.normalize_filename_for_match(item) for item in effective_targets}
        items = [
            doc
            for doc in items
            if adapter.normalize_filename_for_match(adapter.hit_entity_source(doc) or "") in source_set
        ]
    if qfilters.get("doc_type"):
        items = [doc for doc in items if (adapter.hit_metadata(doc).get("doc_type") or "") == qfilters["doc_type"]]
    if qfilters.get("topic"):
        items = [doc for doc in items if qfilters["topic"] in (adapter.hit_metadata(doc).get("topics") or [])]
    if not items:
        return []
    return items[: max(1, int(max_sources or 10))]


def select_process_output_docs(
    runtime: Any,
    query: str,
    docs: List[Any],
    score_mode: str,
    qfilters: Dict[str, Any],
    default_n: int,
    intent_classification: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    del query, score_mode, qfilters, intent_classification
    if not docs:
        return []
    keep_n = min(len(docs), max(0, int(default_n)))
    if keep_n <= 0:
        return []
    return dedupe_evidence_docs(runtime, docs, keep_n)


def _clean_subject_terms(values: Any, limit: int = 8) -> List[str]:
    if isinstance(values, str):
        raw_values = re.split(r"[,，;；、\n]+", values)
    elif isinstance(values, (list, tuple, set)):
        raw_values = list(values)
    else:
        return []
    out: List[str] = []
    for item in raw_values:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if not text or text in out:
            continue
        out.append(text)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _subject_terms(intent_classification: Optional[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    payload = intent_classification if isinstance(intent_classification, dict) else {}
    targets = _clean_subject_terms(
        payload.get("target_subject")
        if payload.get("target_subject") is not None
        else payload.get("target_subjects")
    )
    excluded = _clean_subject_terms(
        payload.get("excluded_subject")
        if payload.get("excluded_subject") is not None
        else payload.get("excluded_subjects")
    )
    return targets, excluded


def _subject_hit_count(text: str, terms: List[str]) -> int:
    if not text or not terms:
        return 0
    return sum(text.count(term) for term in terms if term)


def _clone_with_focus_score(doc: Any, score_mode: str, focus_score: float) -> Any:
    if not isinstance(doc, dict):
        return doc
    cloned = dict(doc)
    if score_mode == "distance" and "score" not in cloned:
        cloned["distance"] = max(0.0, float(focus_score))
    else:
        cloned["score"] = max(0.0, float(focus_score))
    return cloned


def _apply_subject_focus_filter(
    adapter: Any,
    docs: List[Any],
    score_mode: str,
    intent_classification: Optional[Dict[str, Any]],
) -> List[Any]:
    target_terms, excluded_terms = _subject_terms(intent_classification)
    if not target_terms and not excluded_terms:
        return docs

    scored: List[Tuple[float, int, Any]] = []
    for index, doc in enumerate(docs or []):
        haystack = "\n".join(
            [
                adapter.doc_section_name(doc) or "",
                adapter.hit_display_text(doc) or "",
            ]
        )
        target_hits = _subject_hit_count(haystack, target_terms)
        excluded_hits = _subject_hit_count(haystack, excluded_terms)

        if excluded_hits >= 2 and target_hits == 0:
            continue

        base_score = float(adapter.hit_score(doc) or 0.0)
        focus_score = base_score
        if target_hits:
            focus_score += min(adapter.subject_focus_target_bonus_cap, adapter.subject_focus_target_hit_bonus * float(target_hits))
        if excluded_hits:
            focus_score -= min(adapter.subject_focus_excluded_penalty_cap, adapter.subject_focus_excluded_hit_penalty * float(excluded_hits))
        if excluded_hits and not target_hits:
            focus_score -= adapter.subject_focus_unmatched_excluded_penalty

        scored.append((focus_score, -index, _clone_with_focus_score(doc, score_mode, focus_score)))

    if not scored:
        return docs
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored]


def dedupe_evidence_docs(runtime: Any, docs: List[Any], limit: int) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for doc in docs or []:
        metadata = _evidence_context(runtime).hit_metadata(doc)
        key = (
            _evidence_context(runtime).normalize_filename_for_match(_evidence_context(runtime).hit_entity_source(doc) or ""),
            int(metadata.get("chunk_id") or 0),
            (hit_entity_text(doc) or "")[:96],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
        if len(out) >= limit:
            break
    return out


def merge_compare_source_doc_groups(runtime: Any, source_groups: List[Dict[str, Any]], per_source_limit: int) -> List[Any]:
    merged: List[Any] = []
    seen_keys = set()
    limit = max(1, int(per_source_limit))
    grouped_docs = [list((group.get("docs") or [])[:limit]) for group in source_groups or []]
    max_group_len = max((len(items) for items in grouped_docs), default=0)
    for index in range(max_group_len):
        for docs in grouped_docs:
            if index >= len(docs):
                continue
            doc = docs[index]
            key = (
                _evidence_context(runtime).normalize_filename_for_match(_evidence_context(runtime).hit_entity_source(doc) or ""),
                (hit_entity_text(doc) or "")[:96],
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(doc)
    return merged


def _score_field_for_hit(hit: Any) -> Tuple[str, float]:
    mode = hit_score_mode(hit)
    if isinstance(hit, dict):
        if mode == "distance":
            return "distance", float(hit.get("distance") or 0.0)
        return "score", float(hit.get("score") or 0.0)
    return ("distance" if mode == "distance" else "score"), 0.0


def _chunk_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict((chunk or {}).get("metadata") or {})
    if (chunk or {}).get("section") and not metadata.get("section"):
        metadata["section"] = (chunk or {}).get("section")
    if (chunk or {}).get("chunk_id") is not None and metadata.get("chunk_id") is None:
        metadata["chunk_id"] = (chunk or {}).get("chunk_id")
    return metadata


def _chunk_id(chunk: Dict[str, Any]) -> int:
    metadata = _chunk_metadata(chunk)
    try:
        return int(metadata.get("chunk_id") or (chunk or {}).get("chunk_id") or 0)
    except Exception:
        return 0


def _chunk_reading_order(chunk: Dict[str, Any]) -> int:
    metadata = _chunk_metadata(chunk)
    for key in ("reading_order", "order", "idx", "chunk_id"):
        try:
            value = metadata.get(key)
            if value is not None:
                return int(value)
        except Exception:
            continue
    return _chunk_id(chunk)


def _article_id_from_chunk(chunk: Dict[str, Any]) -> str:
    metadata = _chunk_metadata(chunk)
    return document_chunking.extract_article_id(
        metadata.get("article_id"),
        metadata.get("article_no"),
        metadata.get("clause_label"),
        (chunk or {}).get("text"),
        (chunk or {}).get("raw_text"),
    )


def _article_id_from_hit(adapter: Any, doc: Any, source_chunks: List[Dict[str, Any]]) -> str:
    metadata = adapter.hit_metadata(doc)
    article_id = document_chunking.extract_article_id(
        metadata.get("article_id"),
        metadata.get("article_no"),
        metadata.get("clause_label"),
        adapter.hit_display_text(doc),
        hit_entity_text(doc),
    )
    if article_id:
        return article_id

    position = chunk_position_id(doc, adapter.hit_metadata)
    if position is None:
        return ""
    for chunk in source_chunks:
        if _chunk_id(chunk) == int(position):
            return _article_id_from_chunk(chunk)
    return ""


def _full_article_hit_from_chunks(
    adapter: Any,
    source: str,
    article_id: str,
    chunks: List[Dict[str, Any]],
    anchor_doc: Any,
) -> Any:
    article_chunks = [chunk for chunk in chunks if _article_id_from_chunk(chunk) == article_id]
    if not article_chunks:
        return anchor_doc

    article_chunks.sort(key=lambda chunk: (_chunk_reading_order(chunk), _chunk_id(chunk)))
    texts = [
        str((chunk or {}).get("raw_text") or (chunk or {}).get("text") or "").strip()
        for chunk in article_chunks
        if str((chunk or {}).get("raw_text") or (chunk or {}).get("text") or "").strip()
    ]
    if not texts:
        return anchor_doc

    first_md = _chunk_metadata(article_chunks[0])
    last_md = _chunk_metadata(article_chunks[-1])
    anchor_md = dict(adapter.hit_metadata(anchor_doc) or {})
    chunk_ids = [_chunk_id(chunk) for chunk in article_chunks if _chunk_id(chunk)]
    full_text = "\n".join(texts).strip()
    metadata = {
        **anchor_md,
        **first_md,
        "article_id": article_id,
        "article_no": article_id,
        "clause_label": article_id,
        "chunk_role": "article",
        "full_article_expanded": True,
        "full_article_chunk_count": len(article_chunks),
        "full_article_chunk_ids": chunk_ids,
        "full_article_text_chars": len(full_text),
        "content": full_text,
        "raw_text": full_text,
    }
    if chunk_ids:
        metadata["chunk_id"] = min(chunk_ids)
        metadata["chunk_id_start"] = min(chunk_ids)
        metadata["chunk_id_end"] = max(chunk_ids)
    reading_orders = [_chunk_reading_order(chunk) for chunk in article_chunks]
    if reading_orders:
        metadata["reading_order_start"] = min(reading_orders)
        metadata["reading_order_end"] = max(reading_orders)
    if last_md.get("reading_order") is not None:
        metadata["reading_order_end"] = last_md.get("reading_order")

    score_field, score = _score_field_for_hit(anchor_doc)
    return {
        "entity": {
            "source": source,
            "text": full_text,
            "metadata": metadata,
        },
        score_field: score,
    }


def expand_docs_with_full_article_chunks(runtime: Any, docs: List[Any]) -> List[Any]:
    adapter = _evidence_context(runtime)
    if not docs:
        return docs

    expanded: List[Any] = []
    seen_articles: set[Tuple[str, str]] = set()
    source_chunk_cache: Dict[Tuple[str, Optional[int]], List[Dict[str, Any]]] = {}

    for doc in docs:
        source = adapter.normalize_filename_for_match(adapter.hit_entity_source(doc) or "")
        if not source:
            expanded.append(doc)
            continue

        doc_version = _to_int(adapter.hit_metadata(doc).get("doc_version"))
        cache_key = (source, doc_version)
        if cache_key not in source_chunk_cache:
            source_chunk_cache[cache_key] = adapter.get_chunks_for_source(source, doc_version)
        source_chunks = source_chunk_cache.get(cache_key) or []

        article_id = _article_id_from_hit(adapter, doc, source_chunks)
        if not article_id:
            expanded.append(doc)
            continue

        article_key = (source, article_id)
        if article_key in seen_articles:
            continue
        seen_articles.add(article_key)
        expanded.append(_full_article_hit_from_chunks(adapter, source, article_id, source_chunks, doc))

    return expanded


def clone_context_expanded_hit(
    runtime: Any,
    base_hit: Any,
    source: str,
    chunk: Dict[str, Any],
    relation: str,
    anchor_chunk_id: Optional[int],
) -> Dict[str, Any]:
    metadata = dict(chunk.get("metadata") or {})
    metadata.update({
        "context_expanded": True,
        "context_relation": relation,
        "context_anchor_chunk_id": anchor_chunk_id,
    })
    entity = {
        "source": source,
        "text": chunk.get("text") or chunk.get("raw_text") or "",
        "metadata": metadata,
    }
    payload: Dict[str, Any] = {"entity": entity}
    if hit_score_mode(base_hit) == "distance":
        payload["distance"] = float(_evidence_context(runtime).hit_score(base_hit))
    else:
        payload["score"] = float(_evidence_context(runtime).hit_score(base_hit))
    return payload


def _section_identity(metadata: Dict[str, Any]) -> str:
    return str(metadata.get("section_node_id") or metadata.get("section_id") or "").strip()


def _parent_section_identity(metadata: Dict[str, Any]) -> str:
    parent_id = str(metadata.get("parent_section_id") or "").strip()
    if parent_id:
        return parent_id
    parent_title = str(metadata.get("parent_section_title") or "").strip()
    if parent_title:
        return parent_title
    path = metadata.get("parent_section_path") or []
    if isinstance(path, list) and path:
        first = path[0]
        if isinstance(first, dict):
            return str(first.get("id") or first.get("title") or "").strip()
        return str(first or "").strip()
    return ""


def _same_context_boundary(base_metadata: Dict[str, Any], neighbor_metadata: Dict[str, Any]) -> bool:
    base_section = _section_identity(base_metadata)
    neighbor_section = _section_identity(neighbor_metadata)
    if base_section and neighbor_section:
        return base_section == neighbor_section

    base_parent = _parent_section_identity(base_metadata)
    neighbor_parent = _parent_section_identity(neighbor_metadata)
    if base_parent and neighbor_parent:
        return base_parent == neighbor_parent

    base_section_name = str(base_metadata.get("section") or base_metadata.get("section_title") or "").strip()
    neighbor_section_name = str(neighbor_metadata.get("section") or neighbor_metadata.get("section_title") or "").strip()
    return bool(base_section_name and neighbor_section_name and base_section_name == neighbor_section_name)


def expand_docs_with_neighbor_chunks(runtime: Any, docs: List[Any]) -> List[Any]:
    adapter = _evidence_context(runtime)
    if not docs or not adapter.enable_parent_context_expansion:
        return docs

    backward = max(0, int(adapter.parent_context_backward_chunks))
    forward = max(0, int(adapter.parent_context_forward_chunks))
    max_extra = max(0, int(adapter.parent_context_max_extra))
    if backward == 0 and forward == 0:
        return docs

    expanded: List[Any] = []
    seen_keys: set[tuple[str, Optional[int]]] = set()
    chunk_cache: Dict[tuple[str, Optional[int]], Dict[int, Dict[str, Any]]] = {}
    extra_added = 0

    def _doc_key(source: str, chunk_id: Optional[int]) -> tuple[str, Optional[int]]:
        return (adapter.normalize_filename_for_match(source or ""), chunk_id)

    def _append_doc(doc: Any) -> None:
        source = adapter.normalize_filename_for_match(adapter.hit_entity_source(doc) or "")
        position = chunk_position_id(doc, adapter.hit_metadata)
        key = _doc_key(source, position)
        if key in seen_keys:
            return
        seen_keys.add(key)
        expanded.append(doc)

    for doc in docs:
        _append_doc(doc)
        if extra_added >= max_extra:
            continue
        source = adapter.normalize_filename_for_match(adapter.hit_entity_source(doc) or "")
        anchor_chunk_id = chunk_position_id(doc, adapter.hit_metadata)
        if not source or anchor_chunk_id is None:
            continue
        doc_version = _to_int(adapter.hit_metadata(doc).get("doc_version"))
        cache_key = (source, doc_version)
        if cache_key not in chunk_cache:
            chunk_cache[cache_key] = {
                int(item.get("chunk_id") or 0): item
                for item in adapter.get_chunks_for_source(source, doc_version)
            }
        source_chunks = chunk_cache.get(cache_key) or {}
        for offset in range(1, backward + 1):
            if extra_added >= max_extra:
                break
            neighbor = source_chunks.get(anchor_chunk_id - offset)
            if not neighbor:
                continue
            neighbor_key = _doc_key(source, anchor_chunk_id - offset)
            if neighbor_key in seen_keys:
                continue
            if not _same_context_boundary(adapter.hit_metadata(doc), dict(neighbor.get("metadata") or {})):
                continue
            seen_keys.add(neighbor_key)
            expanded.append(clone_context_expanded_hit(runtime, doc, source, neighbor, "previous", anchor_chunk_id))
            extra_added += 1
        for offset in range(1, forward + 1):
            if extra_added >= max_extra:
                break
            neighbor = source_chunks.get(anchor_chunk_id + offset)
            if not neighbor:
                continue
            neighbor_key = _doc_key(source, anchor_chunk_id + offset)
            if neighbor_key in seen_keys:
                continue
            if not _same_context_boundary(adapter.hit_metadata(doc), dict(neighbor.get("metadata") or {})):
                continue
            seen_keys.add(neighbor_key)
            expanded.append(clone_context_expanded_hit(runtime, doc, source, neighbor, "next", anchor_chunk_id))
            extra_added += 1

    return expanded
