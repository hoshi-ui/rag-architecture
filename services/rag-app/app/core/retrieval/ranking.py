import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from app.core.legal_intent import classify_query_intent_fallback, normalize_legal_intent
from app.core.retrieval.chunks import (
    aggregate_doc_sections,
    filter_low_relevance_sources,
    intra_doc_chunk_rerank,
    merge_and_dedupe_hits,
    strict_score_sort,
)
from app.core.retrieval.filters import build_milvus_filter, target_article_ids


logger = logging.getLogger("rag-app")


def _safe_trace_limit(runtime: Any, default: int = 12) -> int:
    try:
        return max(1, int(runtime.config_value("RETRIEVAL_STAGE_TRACE_LIMIT", default) or default))
    except Exception:
        return default


def _trace_doc_id(metadata: Dict[str, Any]) -> str:
    clause_meta = metadata.get("clause_metadata") if isinstance(metadata.get("clause_metadata"), dict) else {}
    for key in ("doc_id", "document_id", "source_id"):
        value = metadata.get(key) or clause_meta.get(key)
        if value:
            return str(value)
    return ""


def _trace_doc_title(metadata: Dict[str, Any]) -> str:
    clause_meta = metadata.get("clause_metadata") if isinstance(metadata.get("clause_metadata"), dict) else {}
    for key in ("doc_title", "title", "source_file"):
        value = metadata.get(key) or clause_meta.get(key)
        if value:
            return str(value)
    return ""


def _normalize_trace_source(runtime: Any, source: Any) -> str:
    value = str(source or "")
    normalize = getattr(runtime, "normalize_filename_for_match", None)
    if callable(normalize):
        try:
            return normalize(value)
        except Exception:
            pass
    common = getattr(runtime, "common", None)
    normalize = getattr(common, "normalize_filename", None)
    if callable(normalize):
        try:
            return normalize(value)
        except Exception:
            pass
    normalize = getattr(common, "normalize_filename_for_match", None)
    if callable(normalize):
        try:
            return normalize(value)
        except Exception:
            pass
    return value


def _trace_hit_entity(doc: Any) -> Dict[str, Any]:
    entity = doc.get("entity") if isinstance(doc, dict) else getattr(doc, "entity", None)
    return entity if isinstance(entity, dict) else {}


def _trace_hit_metadata(runtime: Any, doc: Any) -> Dict[str, Any]:
    getter = getattr(runtime, "hit_metadata", None)
    if callable(getter):
        try:
            metadata = getter(doc)
            if isinstance(metadata, dict):
                return metadata
        except Exception:
            pass
    evidence = getattr(runtime, "evidence", None)
    getter = getattr(evidence, "hit_metadata", None)
    if callable(getter):
        try:
            metadata = getter(doc)
            if isinstance(metadata, dict):
                return metadata
        except Exception:
            pass
    metadata = _trace_hit_entity(doc).get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _trace_hit_source(runtime: Any, doc: Any) -> str:
    getter = getattr(runtime, "hit_entity_source", None)
    if callable(getter):
        try:
            return str(getter(doc) or "")
        except Exception:
            pass
    evidence = getattr(runtime, "evidence", None)
    getter = getattr(evidence, "hit_entity_source", None)
    if callable(getter):
        try:
            return str(getter(doc) or "")
        except Exception:
            pass
    return str(_trace_hit_entity(doc).get("source") or "")


def _trace_hit_text(runtime: Any, doc: Any) -> str:
    for name in ("hit_entity_text", "hit_display_text", "hit_llm_text"):
        getter = getattr(runtime, name, None)
        if callable(getter):
            try:
                text = getter(doc)
                if text:
                    return str(text)
            except Exception:
                pass
    evidence = getattr(runtime, "evidence", None)
    for name in ("hit_entity_text", "hit_display_text", "hit_llm_text"):
        getter = getattr(evidence, name, None)
        if callable(getter):
            try:
                text = getter(doc)
                if text:
                    return str(text)
            except Exception:
                pass
    entity = _trace_hit_entity(doc)
    return str(entity.get("text") or entity.get("content") or "")


def _trace_hit_score(runtime: Any, doc: Any) -> float:
    getter = getattr(runtime, "hit_score", None)
    if callable(getter):
        try:
            return float(getter(doc) or 0.0)
        except Exception:
            pass
    evidence = getattr(runtime, "evidence", None)
    getter = getattr(evidence, "hit_score", None)
    if callable(getter):
        try:
            return float(getter(doc) or 0.0)
        except Exception:
            pass
    if isinstance(doc, dict):
        try:
            return float(doc.get("score") if "score" in doc else doc.get("distance") or 0.0)
        except Exception:
            return 0.0
    for name in ("score", "distance"):
        if hasattr(doc, name):
            try:
                return float(getattr(doc, name) or 0.0)
            except Exception:
                return 0.0
    return 0.0


def _trace_hit_chunk_id(runtime: Any, doc: Any, metadata: Dict[str, Any]) -> Any:
    chunk_id_fn = getattr(runtime, "hit_chunk_id", None)
    if callable(chunk_id_fn):
        try:
            return chunk_id_fn(doc)
        except Exception:
            pass
    evidence = getattr(runtime, "evidence", None)
    chunk_id_fn = getattr(evidence, "hit_chunk_id", None)
    if callable(chunk_id_fn):
        try:
            return chunk_id_fn(doc)
        except Exception:
            pass
    return metadata.get("chunk_id")


def _trace_doc_summary(runtime: Any, doc: Any, index: int, score_mode: str = "score") -> Dict[str, Any]:
    metadata = _trace_hit_metadata(runtime, doc)
    clause_meta = metadata.get("clause_metadata") if isinstance(metadata.get("clause_metadata"), dict) else {}
    article_values = _metadata_article_values(metadata)
    chunk_id = _trace_hit_chunk_id(runtime, doc, metadata)
    text = _trace_hit_text(runtime, doc)
    heading = (
        metadata.get("heading")
        or metadata.get("title")
        or metadata.get("clause_label")
        or clause_meta.get("heading")
        or ""
    )
    return {
        "rank": index + 1,
        "source": _normalize_trace_source(runtime, _trace_hit_source(runtime, doc)),
        "doc_id": _trace_doc_id(metadata),
        "doc_title": _trace_doc_title(metadata),
        "article_no": article_values[0] if article_values else "",
        "article_values": article_values[:4],
        "chapter_title": metadata.get("chapter_title") or clause_meta.get("chapter_title") or "",
        "section_title": metadata.get("section_title") or clause_meta.get("section_title") or "",
        "heading": str(heading),
        "chunk_id": chunk_id,
        "score": round(_trace_hit_score(runtime, doc), 6),
        "score_mode": score_mode,
        "metadata_available": bool(metadata.get("metadata_available") or article_values or _trace_doc_id(metadata)),
        "text_preview": text[:120],
    }


def trace_doc_stage(runtime: Any, docs: List[Any], score_mode: str = "score", limit: Optional[int] = None) -> Dict[str, Any]:
    docs = list(docs or [])
    trace_limit = _safe_trace_limit(runtime) if limit is None else max(1, int(limit))
    articles: List[str] = []
    sources: List[str] = []
    for doc in docs:
        metadata = _trace_hit_metadata(runtime, doc)
        article_values = _metadata_article_values(metadata)
        article = article_values[0] if article_values else ""
        source = _normalize_trace_source(runtime, _trace_hit_source(runtime, doc))
        if article and article not in articles:
            articles.append(article)
        if source and source not in sources:
            sources.append(source)
    return {
        "count": len(docs),
        "sources": sources[:trace_limit],
        "articles": articles[:trace_limit],
        "items": [
            _trace_doc_summary(runtime, doc, index, score_mode=score_mode)
            for index, doc in enumerate(docs[:trace_limit])
        ],
    }


def build_retrieval_stage_trace(
    runtime: Any,
    stages: Dict[str, List[Any]],
    score_mode: str = "score",
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        name: trace_doc_stage(runtime, docs, score_mode=score_mode, limit=limit)
        for name, docs in (stages or {}).items()
    }


def _metadata_article_values(metadata: Dict[str, Any]) -> List[str]:
    clause_meta = metadata.get("clause_metadata") if isinstance(metadata.get("clause_metadata"), dict) else {}
    values = [
        metadata.get("article_no"),
        metadata.get("article_id"),
        metadata.get("clause_id"),
        metadata.get("clause"),
        clause_meta.get("article_no"),
        clause_meta.get("article_id"),
        clause_meta.get("clause_id"),
    ]
    return [str(value or "").strip() for value in values if str(value or "").strip()]


def _pinned_clause_ids(qfilters: Dict[str, Any], query: str) -> List[str]:
    values: List[str] = []
    for key in (
        "_pinned_article_ids",
        "_pinned_clause_ids",
        "article_ids",
        "article_id",
        "target_article",
        "target_articles",
    ):
        raw = (qfilters or {}).get(key)
        if isinstance(raw, (list, tuple, set)):
            values.extend(str(item or "").strip() for item in raw)
        elif raw:
            values.append(str(raw).strip())
    values.extend(target_article_ids(qfilters or {}, query) or [])
    out: List[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _hit_source(runtime: Any, doc: Any) -> str:
    return _normalize_trace_source(runtime, _trace_hit_source(runtime, doc))


def _source_effective_rank(runtime: Any, source: str) -> tuple:
    source_runtime = getattr(runtime, "source", None)
    rank_fn = getattr(source_runtime, "source_effective_rank", None)
    if callable(rank_fn):
        try:
            return tuple(rank_fn(source) or ())
        except Exception:
            pass
    return (0, 0, 0, 0, source or "")


def _source_identity_key(runtime: Any, source: str) -> str:
    source_runtime = getattr(runtime, "source", None)
    for name in ("regulation_identity_key", "canonical_doc_id"):
        fn = getattr(source_runtime, name, None)
        if callable(fn):
            try:
                value = str(fn(source) or "").strip()
            except Exception:
                value = ""
            if value:
                return value
    return f"source:{source}"


def _source_filename_family_key(source: str) -> str:
    value = str(source or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    value = re.sub(r"\.(docx?|pdf|txt|md)$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"(?:[_\-](?:20\d{2}|19\d{2})[-_./]\d{1,2}[-_./]\d{1,2}){1,2}$", "", value)
    value = re.sub(r"(现行有效|最新|修订|试行|暂行)", "", value)
    value = re.sub(r"[\s_\-（）()《》\"'“”]+", "", value)
    return value.strip().lower()


def _source_family_key(runtime: Any, source: str) -> str:
    identity = _source_identity_key(runtime, source)
    filename_key = _source_filename_family_key(source)
    if identity and not identity.startswith("source:"):
        return identity
    if identity.startswith("source:") or not identity:
        return filename_key or identity
    return filename_key or identity


def _latest_equivalent_source(runtime: Any, source: str) -> str:
    source_runtime = getattr(runtime, "source", None)
    fn = getattr(source_runtime, "latest_effective_equivalent_source", None)
    if callable(fn):
        try:
            latest = _normalize_trace_source(runtime, fn(source) or "")
        except Exception:
            latest = ""
        if latest:
            return latest
    return source


def _pinned_sources(runtime: Any, qfilters: Dict[str, Any], active_fnames: Optional[List[str]] = None) -> List[str]:
    raw_values: List[str] = []
    for key in (
        "_pinned_sources",
        "_pinned_source",
        "_locked_sources",
        "_locked_source",
        "_source_lock_sources",
        "_source_lock_source",
    ):
        raw = (qfilters or {}).get(key)
        if isinstance(raw, (list, tuple, set)):
            raw_values.extend(str(item or "").strip() for item in raw)
        elif raw:
            raw_values.append(str(raw).strip())
    raw_values.extend(str(item or "").strip() for item in (active_fnames or []) if str(item or "").strip())

    out: List[str] = []
    for value in raw_values:
        source = _normalize_trace_source(runtime, value)
        if not source:
            continue
        for candidate in (source, _latest_equivalent_source(runtime, source)):
            candidate = _normalize_trace_source(runtime, candidate)
            if candidate and candidate not in out:
                out.append(candidate)
    return out


def _select_with_pinned_clauses(
    runtime: Any,
    docs: List[Any],
    qfilters: Dict[str, Any],
    query: str,
    final_n: int,
    pinned_source_docs: Optional[List[Any]] = None,
    pinned_sources: Optional[List[str]] = None,
) -> List[Any]:
    limit = min(len(docs), max(0, int(final_n)))
    if pinned_source_docs:
        limit = min(len(docs or []) + len(pinned_source_docs or []), max(0, int(final_n)))
    if limit <= 0:
        return []
    pinned_ids = _pinned_clause_ids(qfilters, query)
    selected: List[Any] = []
    selected_keys = set()

    def key(doc: Any) -> tuple:
        md = _trace_hit_metadata(runtime, doc)
        chunk_id = _trace_hit_chunk_id(runtime, doc, md)
        return (
            _normalize_trace_source(runtime, _trace_hit_source(runtime, doc)),
            chunk_id,
            _trace_hit_text(runtime, doc)[:96],
        )

    pinned_pool = list(docs or [])
    pinned_pool_keys = {key(item) for item in pinned_pool}
    for doc in pinned_source_docs or []:
        doc_key = key(doc)
        if doc_key not in pinned_pool_keys:
            pinned_pool.append(doc)
            pinned_pool_keys.add(doc_key)

    for pinned_id in pinned_ids:
        for doc in pinned_pool:
            metadata = _trace_hit_metadata(runtime, doc)
            if pinned_id not in _metadata_article_values(metadata):
                continue
            doc_key = key(doc)
            if doc_key in selected_keys:
                continue
            selected.append(doc)
            selected_keys.add(doc_key)
            if len(selected) >= limit:
                return selected
    for pinned_source in pinned_sources or []:
        normalized_pin = _normalize_trace_source(runtime, pinned_source)
        latest_pin = _latest_equivalent_source(runtime, normalized_pin)
        allowed = {item for item in (normalized_pin, latest_pin) if item}
        for doc in pinned_pool:
            if _hit_source(runtime, doc) not in allowed:
                continue
            doc_key = key(doc)
            if doc_key in selected_keys:
                continue
            selected.append(doc)
            selected_keys.add(doc_key)
            break
        if len(selected) >= limit:
            return selected
    for doc in docs:
        doc_key = key(doc)
        if doc_key in selected_keys:
            continue
        selected.append(doc)
        selected_keys.add(doc_key)
        if len(selected) >= limit:
            break
    return selected


def _chunk_legal_intent(text: str, article_no: Any = "") -> str:
    value = str(text or "")
    if any(term in value for term in ("法律责任", "罚则", "罚款", "没收", "责令", "吊销", "处罚")):
        return "法律责任"
    if any(term in value for term in ("职责", "权限", "职权", "负责", "主管", "监督管理")):
        return "职责与权限"
    if any(term in value for term in ("程序", "流程", "申请", "审查", "办理", "登记", "期限", "条件")):
        return "程序与条件"
    if any(term in value for term in ("定义", "含义", "所称", "适用", "范围", "包括", "不包括")):
        return "定义与范围"
    if any(term in value for term in ("义务", "权利", "应当", "不得", "禁止", "可以")):
        return "权利义务"
    normalized_article = str(article_no or "")
    if re.search(r"第[一二三四五六七八九十0-9]+条", normalized_article) and any(term in value for term in ("总则", "目的")):
        return "定义与范围"
    return "其他"


def _hit_article_no(runtime: Any, hit: Any) -> str:
    metadata = runtime.hit_metadata(hit) or {}
    values = _metadata_article_values(metadata)
    return values[0] if values else ""


def _metadata_aware_rerank_text(runtime: Any, hit: Any, query_intent: str = "") -> str:
    metadata = runtime.hit_metadata(hit) or {}
    clause_meta = metadata.get("clause_metadata") if isinstance(metadata.get("clause_metadata"), dict) else {}
    article_no = _hit_article_no(runtime, hit)
    chapter_title = metadata.get("chapter_title") or clause_meta.get("chapter_title") or ""
    section_title = metadata.get("section_title") or metadata.get("section") or clause_meta.get("section_title") or ""
    heading = metadata.get("heading") or metadata.get("title") or metadata.get("clause_label") or ""
    doc_title = (
        metadata.get("doc_title")
        or clause_meta.get("doc_title")
        or metadata.get("source_file")
        or clause_meta.get("source_file")
        or runtime.hit_entity_source(hit)
        or ""
    )
    body = runtime.hit_entity_text(hit) or ""
    subject_parts = " ".join(str(part or "") for part in [chapter_title, section_title, heading, body])
    normalized_query_intent = normalize_legal_intent(query_intent)
    clause_intent = _chunk_legal_intent(subject_parts, article_no)
    topic = str(heading or section_title or chapter_title or clause_intent or "").strip()

    prefix = []
    if topic:
        prefix.append(f"[{topic}] {body}")
    elif body:
        prefix.append(body)
    if normalized_query_intent:
        prefix.append(f"查询意图：{normalized_query_intent}")
    if clause_intent and clause_intent != "其他":
        prefix.append(f"条款意图：{clause_intent}")
    if chapter_title:
        prefix.append(f"章节：{chapter_title}")
    if section_title and section_title != chapter_title:
        prefix.append(f"小节：{section_title}")
    if heading and heading != article_no:
        prefix.append(f"条款标题：{heading}")
    if body:
        prefix.append(f"内容：{body}")
    suffix = []
    if article_no:
        suffix.append(f"本条为{article_no}")
    if doc_title:
        suffix.append(f"文档：{doc_title}")
    if suffix:
        prefix.append(f"（注：{'；'.join(suffix)}）")
    return "\n".join(str(part) for part in prefix if str(part or "").strip())


def _article_match_reward(query: str, article_no: str) -> float:
    if not article_no:
        return 0.0
    mentioned = set(target_article_ids({}, query) or [])
    if not mentioned:
        return 0.0
    return 0.2 if article_no in mentioned else 0.0


def _version_freshness_rewards(runtime: Any, hits: List[Any]) -> Dict[str, float]:
    try:
        max_reward = float(runtime.config_value("RERANK_VERSION_FRESHNESS_REWARD", 0.04) or 0.0)
    except Exception:
        max_reward = 0.04
    if max_reward <= 0:
        return {}

    grouped: Dict[str, List[str]] = {}
    for hit in hits or []:
        source = _hit_source(runtime, hit)
        if not source:
            continue
        key = _source_family_key(runtime, source)
        grouped.setdefault(key, [])
        if source not in grouped[key]:
            grouped[key].append(source)

    rewards: Dict[str, float] = {}
    for sources in grouped.values():
        if len(sources) <= 1:
            continue
        ranked = sorted(sources, key=lambda item: _source_effective_rank(runtime, item))
        denom = max(1, len(ranked) - 1)
        for index, source in enumerate(ranked):
            reward = max_reward * (float(index) / float(denom))
            if reward > 0:
                rewards[source] = max(rewards.get(source, 0.0), reward)
    return rewards


async def chunk_level_rerank(
    runtime: Any,
    rerank_service: Any,
    query: str,
    hits: List[Any],
    top_k: int,
    enable_rerank: bool,
    query_intent: str = "",
) -> Dict[str, Any]:
    score_mode = runtime.hit_score_mode(hits[0]) if hits else "score"
    if (not hits) or (not enable_rerank) or (not runtime.config_value("ENABLE_RERANK", False)):
        kept = hits[:top_k] if top_k > 0 else hits[:]
        return {"hits": kept, "score_mode": score_mode, "used": False}
    rerank_n = min(len(hits), max(1, top_k))
    if rerank_n <= 1:
        kept = hits[:top_k] if top_k > 0 else hits[:]
        return {"hits": kept, "score_mode": score_mode, "used": False}
    try:
        reranked = await rerank_service.rerank(
            query=query,
            documents=[
                _metadata_aware_rerank_text(
                    runtime,
                    hit,
                    query_intent=normalize_legal_intent(query_intent) or classify_query_intent_fallback(query),
                )
                for hit in hits[:rerank_n]
            ],
            top_k=rerank_n,
        )
    except Exception:
        kept = hits[:top_k] if top_k > 0 else hits[:]
        return {"hits": kept, "score_mode": score_mode, "used": False}
    reranked_hits: List[Dict[str, Any]] = []
    freshness_rewards = _version_freshness_rewards(runtime, hits[:rerank_n])
    for item in reranked or []:
        idx = item.get("index") if isinstance(item, dict) else getattr(item, "index", None)
        score = item.get("score", 0.0) if isinstance(item, dict) else getattr(item, "score", 0.0)
        try:
            idx_i = int(idx)
        except Exception:
            continue
        if idx_i < 0 or idx_i >= rerank_n:
            continue
        base_hit = hits[idx_i]
        ent = base_hit.get("entity") if isinstance(base_hit, dict) else getattr(base_hit, "entity", None)
        source = _hit_source(runtime, base_hit)
        adjusted_score = (
            float(score)
            + _article_match_reward(query, _hit_article_no(runtime, base_hit))
            + float(freshness_rewards.get(source, 0.0))
        )
        reranked_hits.append({"entity": ent, "score": adjusted_score})
    if not reranked_hits:
        kept = hits[:top_k] if top_k > 0 else hits[:]
        return {"hits": kept, "score_mode": score_mode, "used": False}
    reranked_hits.sort(key=lambda hit: float(hit.get("score") or 0.0), reverse=True)
    return {"hits": reranked_hits, "score_mode": "score", "used": True}

async def source_level_rerank(
    runtime: Any,
    rerank_service: Any,
    query: str,
    hits: List[Any],
    src_scores: Dict[str, float],
    keep_n: int,
    enable_rerank: bool,
    dense_rank_map: Optional[Dict[str, int]] = None,
    lex_rank_map: Optional[Dict[str, int]] = None,
    source_signals: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if (not src_scores) or (not hits) or (not enable_rerank) or (not runtime.config_value("ENABLE_RERANK", False)):
        return {"scores": src_scores, "used": False}
    if len(src_scores) <= 1:
        return {"scores": src_scores, "used": False}
    if bool(runtime.config_value("RERANK_LOW_CONF_ONLY", True)):
        dense_top = runtime.top_ranked_source(dense_rank_map or {})
        lex_top = runtime.top_ranked_source(lex_rank_map or {})
        current_top = max(src_scores.items(), key=lambda item: item[1])[0]
        gap = runtime.source_score_gap(src_scores)
        anchored = bool((source_signals or {}).get(current_top, {}).get("title_hit"))
        if gap > float(runtime.config_value("RERANK_SOURCE_SCORE_GAP", 0.04)):
            return {"scores": src_scores, "used": False}
        if anchored and dense_top == current_top and (lex_top in (None, current_top)):
            return {"scores": src_scores, "used": False}
    doc_sources = list(src_scores.keys())
    by_src: Dict[str, List[str]] = {}
    for hit in hits:
        src = runtime.normalize_filename_for_match(runtime.hit_entity_source(hit) or "")
        by_src.setdefault(src, []).append(runtime.hit_entity_text(hit) or "")
    documents = []
    for src in doc_sources:
        snippets = [txt for txt in by_src.get(src, []) if txt]
        documents.append(("\n".join(snippets[:3])).strip() or src)
    try:
        reranked = await rerank_service.rerank(query=query, documents=documents, top_k=min(len(documents), max(1, keep_n)))
    except Exception:
        return {"scores": src_scores, "used": False}
    merged = dict(src_scores)
    weight = float(os.getenv("FUSION_W_RERANK_DOC", "0.3"))
    for item in reranked or []:
        idx = item.get("index") if isinstance(item, dict) else getattr(item, "index", None)
        score = item.get("score", 0.0) if isinstance(item, dict) else getattr(item, "score", 0.0)
        try:
            src = doc_sources[int(idx)]
        except Exception:
            continue
        merged[src] = merged.get(src, 0.0) + weight * float(score)
    return {"scores": merged, "used": True}

def apply_retrieval_filters(runtime: Any, docs: List[Any], qfilters: Dict[str, Any], fnames: List[str]) -> List[Any]:
    filtered_docs = docs[:]
    filtered_docs = [d for d in filtered_docs if not runtime.is_heading_only_hit(d)]
    if qfilters.get("doc_type"):
        filtered_docs = [d for d in filtered_docs if (runtime.hit_metadata(d).get("doc_type") or "") == qfilters["doc_type"]]
    if qfilters.get("topic"):
        filtered_docs = [d for d in filtered_docs if qfilters["topic"] in (runtime.hit_metadata(d).get("topics") or [])]
    if fnames:
        sset = set([runtime.normalize_filename_for_match(x) for x in fnames])
        filtered_docs = [d for d in filtered_docs if runtime.normalize_filename_for_match(runtime.hit_entity_source(d) or "") in sset]
    return filtered_docs


def _candidate_hint_sources(runtime: Any, qfilters: Optional[Dict[str, Any]]) -> List[str]:
    values = []
    for source in (qfilters or {}).get("_candidate_hint_sources") or []:
        normalized = runtime.normalize_filename_for_match(source or "")
        if normalized:
            values.append(normalized)
    return list(dict.fromkeys(values))

def summarize_source_scores(
    runtime: Any,
    docs: List[Any],
    dense_rank_map: Dict[str, int],
    lex_rank_map: Dict[str, int],
    source_count: Dict[str, int],
    source_signals: Dict[str, Dict[str, Any]],
    fname_set: set,
    allowed_set: set,
    weak_query: bool,
    query: str,
) -> Dict[str, float]:
    by_src: Dict[str, int] = {}
    for hit in docs:
        src = runtime.normalize_filename_for_match(runtime.hit_entity_source(hit) or "")
        by_src[src] = by_src.get(src, 0) + 1
    scores: Dict[str, float] = {}
    for src in by_src:
        scores[src] = runtime.fusion_source_score(src, query, dense_rank_map, lex_rank_map, source_count, source_signals, fname_set, allowed_set, weak_query)
    return scores

def _source_count_map(runtime: Any, docs: List[Any]) -> Dict[str, int]:
    source_count: Dict[str, int] = {}
    for hit in docs or []:
        src = runtime.normalize_filename_for_match(runtime.hit_entity_source(hit) or "")
        if src:
            source_count[src] = source_count.get(src, 0) + 1
    return source_count

def prune_hybrid_sources(
    runtime: Any,
    docs: List[Any],
    source_scores: Dict[str, float],
    fname_set: set,
    allowed_set: set,
) -> Dict[str, Any]:
    if not docs or not source_scores:
        return {"docs": docs, "kept_sources": set(source_scores), "pruned_sources": set(), "enabled": False}
    if not bool(runtime.config_value("HYBRID_SOURCE_PRUNE_ENABLED", True)):
        return {"docs": docs, "kept_sources": set(source_scores), "pruned_sources": set(), "enabled": False}

    keep = set(source_scores)
    return {
        "docs": docs,
        "kept_sources": keep,
        "pruned_sources": set(source_scores) - keep,
        "enabled": True,
    }

def doc_level_rerank(runtime: Any, hits: List[Any], score_mode: str) -> List[Dict[str, Any]]:
    by_src: Dict[str, List[Any]] = {}
    for h in hits:
        s = runtime.hit_entity_source(h) or "unknown"
        by_src.setdefault(s, []).append(h)
    ranked = []
    for s, items in by_src.items():
        scs = [runtime.hit_score(x) for x in items]
        score = min(scs) if score_mode == "distance" else max(scs)
        count = len(items)
        agg = (score if score_mode != "distance" else (1.0 / (1e-9 + score))) + 0.05 * count
        ranked.append({"source": s, "score": float(agg)})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked

def fuse_dense_lexical_hits(
    runtime: Any,
    dense_hits: List[Any],
    lexical_hits: List[Any],
    query: str,
    recall_k: int,
    dense_source_scores: Optional[Dict[str, float]] = None,
    fname_set: Optional[set] = None,
    allowed_set: Optional[set] = None,
    doc_recall_plan: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    docs_all = list(dense_hits or []) + list(lexical_hits or [])
    dense_rank_map: Dict[str, int] = {}
    for i, hit in enumerate(dense_hits or []):
        src = runtime.normalize_filename_for_match(runtime.hit_entity_source(hit) or "")
        if src and src not in dense_rank_map:
            dense_rank_map[src] = i

    lex_rank_map: Dict[str, int] = {}
    for i, hit in enumerate(lexical_hits or []):
        src = runtime.normalize_filename_for_match(runtime.hit_entity_source(hit) or "")
        if src and src not in lex_rank_map:
            lex_rank_map[src] = i

    weak_query = runtime.is_weak_reference_query(query)
    source_signals = runtime.build_source_signal_map(query, lexical_hits or [], doc_recall_plan or [])
    fname_set = fname_set or set()
    allowed_set = allowed_set or set()
    dense_source_scores = dense_source_scores or {}
    source_count = _source_count_map(runtime, docs_all)
    pre_prune_source_scores = summarize_source_scores(
        runtime,
        docs_all,
        dense_rank_map,
        lex_rank_map,
        source_count,
        source_signals,
        fname_set,
        allowed_set,
        weak_query,
        query,
    )
    prune_result = prune_hybrid_sources(runtime, docs_all, pre_prune_source_scores, fname_set, allowed_set)
    docs_all = prune_result["docs"]
    source_count = _source_count_map(runtime, docs_all)
    combined = [(0.0, i) for i in range(len(docs_all))]
    fused_source_scores: Dict[str, float] = {}

    for i, hit in enumerate(docs_all):
        src = runtime.normalize_filename_for_match(runtime.hit_entity_source(hit) or "")
        if src not in fused_source_scores:
            fused_source_scores[src] = runtime.fusion_source_score(
                src,
                query,
                dense_rank_map,
                lex_rank_map,
                source_count,
                source_signals,
                fname_set,
                allowed_set,
                weak_query,
            )
        combined[i] = (fused_source_scores[src], i)
    combined.sort(
        key=lambda item: (
            item[0],
            runtime.source_dense_tiebreak_score(
                runtime.normalize_filename_for_match(runtime.hit_entity_source(docs_all[item[1]]) or ""),
                dense_source_scores,
            ),
        ),
        reverse=True,
    )

    seen_keys = set()
    docs_fused = []
    for fused_score, idx in combined:
        hit = docs_all[idx]
        key = (runtime.hit_entity_source(hit) or "unknown", (runtime.hit_entity_text(hit) or "")[:64])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        docs_fused.append(runtime.clone_hit_with_score(hit, fused_score))
        if len(docs_fused) >= recall_k:
            break

    return {
        "docs": docs_fused,
        "docs_all": docs_all,
        "dense_rank_map": dense_rank_map,
        "lex_rank_map": lex_rank_map,
        "source_count": source_count,
        "source_signals": source_signals,
        "fused_source_scores": fused_source_scores,
        "source_prune": prune_result,
        "weak_query": weak_query,
    }

def run_lightweight_search_candidates(
    runtime: Any,
    handler: Any,
    query_embedding: List[float],
    retrieval_query: str,
    active_fnames: List[str],
    recall_k: int,
    qfilters: Optional[Dict[str, Any]] = None,
    query_sparse_embedding: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    safe_names: List[str] = []
    if active_fnames:
        safe_names = [runtime.normalize_filename_for_match(x) for x in active_fnames]
    hint_names = [x for x in _candidate_hint_sources(runtime, qfilters) if x not in set(safe_names)]
    article_ids = target_article_ids(qfilters, retrieval_query)
    milvus_filter = build_milvus_filter(sources=safe_names, article_ids=article_ids)
    if len(article_ids) > 1:
        logger.info("retrieval_filter_milvus: target_articles=%s filter=%s", article_ids, milvus_filter)

    docs = handler.vector_db.search(
        query_embedding,
        top_k=recall_k,
        filters=milvus_filter,
        query_sparse_embedding=query_sparse_embedding,
    )
    if hint_names:
        hint_filter = build_milvus_filter(sources=hint_names, article_ids=article_ids)
        hint_docs = handler.vector_db.search(
            query_embedding,
            top_k=recall_k,
            filters=hint_filter,
            query_sparse_embedding=query_sparse_embedding,
        )
        docs = list(docs or []) + list(hint_docs or [])
    visible_dense = runtime.filter_hits_by_source_state(docs)
    docs = visible_dense["hits"]
    dense_source_scores = runtime.dense_source_score_map(docs)

    lex_items = runtime.collect_lexical_candidates(retrieval_query, safe_names, [], article_ids=article_ids)
    if hint_names:
        hint_lex_items = runtime.collect_lexical_candidates(retrieval_query, hint_names, [], article_ids=article_ids)
        lex_items = list(lex_items or []) + list(hint_lex_items or [])
    visible_lex = runtime.filter_hits_by_source_state(lex_items)
    lex_items = visible_lex["hits"]

    return {
        "safe_names": safe_names,
        "hint_names": hint_names,
        "milvus_filter": milvus_filter,
        "docs": docs,
        "lex_items": lex_items,
        "docs_all": docs + lex_items,
        "visible_dense": visible_dense,
        "visible_lex": visible_lex,
        "dense_source_scores": dense_source_scores,
    }

def postprocess_recall_docs(
    runtime: Any,
    docs: List[Any],
    score_mode: str,
    query: str,
    qtype: str,
    qfilters: Dict[str, Any],
    active_fnames: List[str],
    final_n: int,
    pinned_source_docs: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    retrieve_docs = apply_retrieval_filters(runtime, docs, qfilters, active_fnames)

    merged_docs = merge_and_dedupe_hits(runtime, retrieve_docs, score_mode=score_mode)
    aggregated_docs = aggregate_doc_sections(runtime, merged_docs, score_mode=score_mode)
    context_docs = aggregated_docs
    rescue_pool = apply_retrieval_filters(runtime, list(pinned_source_docs or []), qfilters, active_fnames) if pinned_source_docs else []
    if not rescue_pool:
        rescue_pool = list(retrieve_docs or []) + list(merged_docs or []) + list(aggregated_docs or [])
    source_pins = _pinned_sources(runtime, qfilters, active_fnames)

    post_filter_docs = filter_low_relevance_sources(runtime, context_docs, score_mode=score_mode, query=query)
    post_filter_docs = intra_doc_chunk_rerank(runtime, query, post_filter_docs, score_mode=score_mode, qtype=qtype, qfilters=qfilters)
    selected_docs = _select_with_pinned_clauses(
        runtime,
        post_filter_docs,
        qfilters,
        query,
        final_n,
        pinned_source_docs=rescue_pool,
        pinned_sources=source_pins,
    )
    stage_trace = build_retrieval_stage_trace(
        runtime,
        {
            "retrieve_docs": retrieve_docs,
            "merged_docs": merged_docs,
            "aggregated_docs": aggregated_docs,
            "context_docs": context_docs,
            "pinned_rescue_pool": rescue_pool if _pinned_clause_ids(qfilters, query) else [],
            "source_pin_rescue_pool": rescue_pool if source_pins else [],
            "post_filter_docs": post_filter_docs,
            "selected_docs": selected_docs,
        },
        score_mode=score_mode,
    )
    return {
        "merged_docs": merged_docs,
        "aggregated_docs": aggregated_docs,
        "context_docs": context_docs,
        "retrieve_docs": retrieve_docs,
        "selected_docs": selected_docs,
        "post_filter_docs": post_filter_docs,
        "stage_trace": stage_trace,
    }

async def rerank_and_postprocess_lightweight_docs(
    runtime: Any,
    rerank_service: Any,
    docs: List[Any],
    lex_items: List[Any],
    retrieval_query: str,
    qtype: str,
    qfilters: Dict[str, Any],
    active_fnames: List[str],
    recall_k: int,
    final_n: int,
    pool_n: int,
    enable_rerank: bool,
    dense_source_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    soft_hint_set = set(_candidate_hint_sources(runtime, qfilters))
    fname_set = {runtime.normalize_filename_for_match(x) for x in active_fnames if x}
    fname_set.update(soft_hint_set)
    fusion = fuse_dense_lexical_hits(
        runtime,
        docs,
        lex_items,
        retrieval_query,
        recall_k,
        dense_source_scores=dense_source_scores,
        fname_set=fname_set,
        doc_recall_plan=[],
    )
    fused_docs = fusion["docs"]
    chunk_rerank_enabled = runtime.should_apply_chunk_rerank(
        fused_docs[:pool_n],
        fusion["dense_rank_map"],
        fusion["lex_rank_map"],
        fusion["source_signals"],
        enable_rerank,
    )
    reranked_chunk = await chunk_level_rerank(
        runtime,
        rerank_service,
        retrieval_query,
        fused_docs[:pool_n],
        pool_n,
        chunk_rerank_enabled,
        query_intent=str((qfilters or {}).get("_legal_intent") or ""),
    )
    reranked_docs = reranked_chunk["hits"]
    score_mode = reranked_chunk["score_mode"]
    postprocessed = postprocess_recall_docs(
        runtime,
        reranked_docs,
        score_mode=score_mode,
        query=retrieval_query,
        qtype=qtype,
        qfilters=qfilters,
        active_fnames=active_fnames,
        final_n=final_n,
        pinned_source_docs=list(fused_docs or []) + list(reranked_docs or []),
    )
    stage_trace = build_retrieval_stage_trace(
        runtime,
        {
            "dense_hits": docs,
            "lexical_hits": lex_items,
            "fused_docs": fused_docs,
            "reranked_docs": reranked_docs,
        },
        score_mode=score_mode,
    )
    stage_trace.update(dict(postprocessed.get("stage_trace") or {}))
    return {
        **fusion,
        "docs": reranked_docs,
        "reranked_chunk": {**dict(reranked_chunk), "stage_trace": stage_trace},
        "score_mode": score_mode,
        "retrieve_docs": postprocessed["retrieve_docs"],
        "selected_docs": postprocessed["selected_docs"],
        "post_filter_docs": postprocessed["post_filter_docs"],
        "postprocessed": postprocessed,
        "stage_trace": stage_trace,
    }
