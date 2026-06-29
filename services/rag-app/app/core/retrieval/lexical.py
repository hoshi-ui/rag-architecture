import re
from typing import Any, Dict, List, Optional

from app.core.retrieval.filters import normalize_configured_article_ids


ARTICLE_ANCHOR_RE = re.compile(r"第[一二三四五六七八九十百千万0-9]+[条款项章]")


def article_anchor_terms(query: str) -> List[str]:
    out: List[str] = []
    for match in ARTICLE_ANCHOR_RE.finditer(str(query or "")):
        value = match.group(0)
        if value not in out:
            out.append(value)
    return out[:6]


def _mark_article_anchor_hits(hits: List[Dict[str, Any]], anchor: str) -> List[Dict[str, Any]]:
    marked: List[Dict[str, Any]] = []
    for hit in hits or []:
        entity = dict((hit or {}).get("entity") or {})
        text = str(entity.get("text") or "")
        metadata = dict(entity.get("metadata") or {})
        section = str(metadata.get("section_title") or metadata.get("section") or "")
        out = dict(hit or {})
        if anchor in text or anchor in section:
            metadata["article_anchor_hit"] = True
            metadata["article_anchor"] = anchor
            metadata["lexical_signal"] = "article_anchor"
            entity["metadata"] = metadata
            out["entity"] = entity
            out["score"] = max(float(out.get("score") or 0.0), 8.0)
        marked.append(out)
    return marked


def collect_lexical_candidates(
    runtime: Any,
    query: str,
    safe_names: List[str],
    doc_recall_plan: List[Dict[str, Any]],
    article_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    normalized_article_ids = normalize_configured_article_ids(article_ids or [])
    source_filter = safe_names[0] if len(safe_names) == 1 else None
    allowed_docs = [entry.get("source") for entry in (doc_recall_plan or []) if (entry or {}).get("source")]
    doc_recall_map = {
        entry["source"]: {**entry, "rank": idx}
        for idx, entry in enumerate(doc_recall_plan or [])
        if entry.get("source")
    }
    try:
        items.extend(
            runtime.lexical_recall_indexed(
                query,
                runtime.config_value("LEXICAL_RECALL_LIMIT", 1000),
                source_filter=source_filter,
                article_ids=normalized_article_ids,
            )
        )
    except Exception:
        pass
    try:
        items.extend(
            runtime.lexical_recall_fallback(
                query,
                runtime.config_value("LEXICAL_RECALL_LIMIT", 1000),
                source_filter=source_filter,
                article_ids=normalized_article_ids,
            )
        )
    except Exception:
        pass
    for anchor in article_anchor_terms(query):
        anchor_limit = max(10, min(80, runtime.config_value("LEXICAL_RECALL_LIMIT", 1000) // 8))
        try:
            items.extend(
                _mark_article_anchor_hits(
                    runtime.lexical_recall_indexed(
                        anchor,
                        anchor_limit,
                        source_filter=source_filter,
                        article_ids=normalized_article_ids,
                    ),
                    anchor,
                )
            )
        except Exception:
            try:
                items.extend(
                    _mark_article_anchor_hits(
                        runtime.lexical_recall_fallback(
                            anchor,
                            anchor_limit,
                            source_filter=source_filter,
                            article_ids=normalized_article_ids,
                        ),
                        anchor,
                    )
                )
            except Exception:
                pass
    title_sources = [src for src in (allowed_docs or []) if src]
    if source_filter and source_filter not in title_sources:
        title_sources.insert(0, source_filter)
    for src in title_sources[: max(1, int(runtime.config_value("WEAK_QUERY_DOC_LIMIT", 6)))]:
        plan_entry = doc_recall_map.get(src) or {}
        items.append(
            runtime.synthetic_doc_title_hit(
                src,
                query,
                score=max(1.0, float(plan_entry.get("prior", 0.0)) + 1.0),
                metadata_updates={
                    "doc_recall_hit": True,
                    "doc_prior": float(plan_entry.get("prior", 0.0)),
                    "doc_recall_reasons": list(plan_entry.get("reasons") or []),
                    "doc_recall_rank": int(plan_entry.get("rank", 0)),
                },
            )
        )
    for expansion in runtime.build_controlled_expansion_queries(query, allowed_docs):
        expansion_limit = max(20, min(200, runtime.config_value("LEXICAL_RECALL_LIMIT", 1000) // 4))
        expansion_hits: List[Dict[str, Any]] = []
        try:
            expansion_hits.extend(
                runtime.lexical_recall_indexed(
                    expansion["query"],
                    expansion_limit,
                    source_filter=expansion.get("source"),
                    article_ids=normalized_article_ids,
                )
            )
        except Exception:
            pass
        try:
            expansion_hits.extend(
                runtime.lexical_recall_fallback(
                    expansion["query"],
                    expansion_limit,
                    source_filter=expansion.get("source"),
                    article_ids=normalized_article_ids,
                )
            )
        except Exception:
            pass
        items.extend(expansion_hits)
    allowed_set = {runtime.normalize_filename_for_match(src) for src in (allowed_docs or []) if src}
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        item = runtime.annotate_lexical_hit(query, item, allowed_set, doc_recall_map=doc_recall_map)
        src = runtime.normalize_filename_for_match(runtime.hit_entity_source(item) or "")
        chunk_id = runtime.hit_metadata(item).get("chunk_id")
        key = (src, chunk_id, (runtime.hit_entity_text(item) or "")[:80])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped

def synthetic_doc_title_hit(
    runtime: Any,
    source: str,
    query: str,
    score: float = 1.0,
    metadata_updates: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    info = runtime.doc_get(source)
    text = "\n".join([
        info.get("canonical_title") or "",
        info.get("aliases") or "",
        info.get("filename_stem") or runtime.filename_stem(source),
    ]).strip() or source
    entity = {
        "source": source,
        "text": text,
        "metadata": {
            "section": "document_title",
            "doc_type": info.get("doc_type") or "",
            "topic": info.get("topic") or "",
            "title_hit": True,
            "lexical_signal": "title_direct",
            "query": query,
        },
    }
    if metadata_updates:
        entity["metadata"].update(metadata_updates)
    return {"entity": entity, "score": float(score)}

def annotate_lexical_hit(
    runtime: Any,
    query: str,
    hit: Dict[str, Any],
    allowed_set: set,
    doc_recall_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    src = runtime.normalize_filename_for_match(runtime.hit_entity_source(hit) or "")
    if not src:
        return hit
    entity = dict((hit or {}).get("entity") or {})
    metadata = dict(entity.get("metadata") or {})
    title_score = runtime.doc_title_alias_score(src, query)
    if title_score > 0:
        metadata["title_match_score"] = title_score
    if title_score >= 2.0 or metadata.get("title_hit"):
        metadata["title_hit"] = True
    if src in allowed_set:
        metadata["doc_recall_hit"] = True
    plan_entry = (doc_recall_map or {}).get(src) or {}
    if plan_entry:
        metadata["doc_prior"] = float(plan_entry.get("prior", 0.0))
        metadata["doc_recall_reasons"] = list(plan_entry.get("reasons") or [])
        metadata["doc_recall_rank"] = int(plan_entry.get("rank", 0))
    if not metadata.get("lexical_signal"):
        if metadata.get("title_hit") or (metadata.get("section") or "") == "document_title":
            metadata["lexical_signal"] = "title_direct"
        else:
            metadata["lexical_signal"] = "indexed_fts"
    entity["metadata"] = metadata
    out = dict(hit)
    out["entity"] = entity
    return out

def build_controlled_expansion_queries(runtime: Any, query: str, allowed_docs: List[str]) -> List[Dict[str, str]]:
    if not runtime.is_weak_reference_query(query):
        return []
    expansions: List[Dict[str, str]] = []
    seen = {query}
    limit = max(0, int(runtime.config_value("WEAK_QUERY_EXPANSION_LIMIT", 3)))
    for source in allowed_docs[:limit]:
        info = runtime.doc_get(source)
        title = (info.get("canonical_title") or info.get("filename_stem") or runtime.filename_stem(source)).strip()
        if not title:
            continue
        expanded = f"{title} {query}".strip()
        if expanded in seen:
            continue
        seen.add(expanded)
        expansions.append({
            "query": expanded,
            "source": runtime.normalize_filename_for_match(source),
            "reason": "title_anchor",
        })
    return expansions

def distinct_hit_sources(runtime: Any, hits: List[Any]) -> List[str]:
    out: List[str] = []
    for hit in hits:
        src = runtime.normalize_filename_for_match(runtime.hit_entity_source(hit) or "")
        if src and src not in out:
            out.append(src)
    return out
