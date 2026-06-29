from typing import Any, Dict, List

import re

from app.core.evidence.context import _evidence_context
from app.core.evidence.hits import estimate_token_count, evidence_relevance
from app.documents import clause_metadata as document_clause_metadata


LEGAL_TITLE_MARKERS = ("条例", "规定", "办法", "规则", "规程", "细则", "法律", "法规", "决定")


def _clean_document_title(raw: Any) -> str:
    title = str(raw or "").strip().strip("《》")
    if not title:
        return ""
    title = title.replace("\\", "/").split("/")[-1].strip()
    title = re.sub(r"\.(?:docx?|pdf|txt|md|html?)$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"[_\-\s]*\d{4}(?:[-_年\s]+)\d{1,2}(?:[-_月\s]+)\d{1,2}日?", "", title)
    title = re.sub(r"[_\-\s]*(?:现行有效|最新版本|有效|版本|v\d+)$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"[_\-\s]+$", "", title).strip()
    if not title or "\n" in title or len(title) > 100:
        return ""
    return title


def _looks_like_document_title(title: str) -> bool:
    compact = re.sub(r"\s+", "", str(title or ""))
    if not compact:
        return False
    if re.match(r"^第[一二三四五六七八九十百千万零〇0-9]+章", compact):
        return False
    if compact in {"总则", "附则", "第一章总则", "法律责任", "监督检查", "保护与管理", "职责", "范围"}:
        return False
    return any(marker in compact for marker in LEGAL_TITLE_MARKERS)


def _preferred_document_title(ctx: Any, source: str, metadata: Dict[str, Any]) -> str:
    safe_source = ctx.normalize_filename_for_match(source) if source else ""
    display_title = _clean_document_title(ctx.source_display_title(safe_source or source)) if source else ""
    source_title = _clean_document_title((metadata or {}).get("source_file") or source)
    source_title = source_title if _looks_like_document_title(source_title) else ""
    preferred_source_title = display_title if _looks_like_document_title(display_title) else source_title

    clause_meta = metadata.get("clause_metadata") if isinstance(metadata.get("clause_metadata"), dict) else {}
    for candidate in [
        metadata.get("doc_title"),
        clause_meta.get("doc_title"),
        metadata.get("canonical_title"),
    ]:
        title = _clean_document_title(candidate)
        if not _looks_like_document_title(title):
            continue
        if preferred_source_title and title in preferred_source_title and len(preferred_source_title) > len(title):
            return preferred_source_title
        return title
    return preferred_source_title or _clean_document_title(source) or source


def fit_evidence_block_to_budget(head: str, content: str, budget: int, model_name: str = "") -> str:
    """Return a non-empty evidence block that fits the remaining prompt budget."""
    safe_budget = max(64, int(budget or 0))
    clean_content = str(content or "").strip()
    block = f"{head}\n{clean_content}" if clean_content else head
    if estimate_token_count(block, model_name) + 2 <= safe_budget:
        return block

    head_tokens = estimate_token_count(head, model_name) + 2
    if head_tokens >= safe_budget:
        return head

    remaining = max(48, safe_budget - head_tokens - 2)
    approx_chars = max(160, remaining * 3)
    truncated = clean_content[:approx_chars].rstrip()
    while truncated and estimate_token_count(f"{head}\n{truncated}", model_name) + 2 > safe_budget:
        truncated = truncated[: max(0, int(len(truncated) * 0.8))].rstrip()
    return f"{head}\n{truncated}" if truncated else head


def format_evidence(
    runtime: Any,
    docs: List[Any],
    query: str,
    score_mode: str,
    *,
    token_budget: int = 6500,
    model_name: str = "",
) -> str:
    lines = []
    total_tokens = 0
    safe_budget = max(256, int(token_budget or 6500))
    best_score = _evidence_context(runtime).hit_score(docs[0]) if docs else 0.0
    for i, doc in enumerate(docs, start=1):
        source = _evidence_context(runtime).hit_entity_source(doc) or "unknown"
        content = (_evidence_context(runtime).hit_llm_text(doc) or "").strip()
        if not content:
            continue
        metadata = _evidence_context(runtime).hit_metadata(doc)
        section = (metadata.get("section") or metadata.get("section_title") or "").strip()
        chunk_range = _evidence_context(runtime).hit_chunk_range(doc)
        relevance = evidence_relevance(_evidence_context(runtime).hit_score(doc), score_mode, best_score)
        title = _preferred_document_title(_evidence_context(runtime), source, metadata)
        parts = [f"来源：{source}", f"标题：{title}", f"相关度：{relevance:.2f}"]
        if section:
            parts.append(f"章节：{section}")
        if chunk_range:
            parts.append(f"chunk：{chunk_range}")
        head = f"[{i}] " + " | ".join(parts)
        block = head + "\n" + content
        block_tokens = estimate_token_count(block, model_name) + 2
        if total_tokens + block_tokens > safe_budget:
            fitted = fit_evidence_block_to_budget(head, content, safe_budget - total_tokens, model_name)
            if fitted and estimate_token_count(fitted, model_name) + 2 <= safe_budget - total_tokens:
                lines.append(fitted)
            break
        lines.append(block)
        total_tokens += block_tokens
    return "\n\n".join(lines)


def build_sources(runtime: Any, final_docs: List[Any], query: str, score_mode: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    chunk_article_cache: Dict[str, Dict[int, Dict[str, str]]] = {}
    ctx = _evidence_context(runtime)

    def article_from_active_chunks(source: str, chunk_id: Any) -> Dict[str, str]:
        if chunk_id is None:
            return {}
        try:
            key = int(chunk_id)
        except Exception:
            return {}
        if source not in chunk_article_cache:
            lookup: Dict[int, Dict[str, str]] = {}
            try:
                for chunk in ctx.get_chunks_for_source(source, None) or []:
                    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
                    cid = metadata.get("chunk_id", chunk.get("chunk_id"))
                    try:
                        cid_int = int(cid)
                    except Exception:
                        continue
                    article_id = str(
                        metadata.get("article_id")
                        or metadata.get("article_no")
                        or chunk.get("article_id")
                        or chunk.get("article_no")
                        or ""
                    ).strip()
                    article_no = str(
                        metadata.get("article_no")
                        or metadata.get("article_id")
                        or chunk.get("article_no")
                        or chunk.get("article_id")
                        or ""
                    ).strip()
                    if article_id or article_no:
                        lookup.setdefault(cid_int, {"article_id": article_id or article_no, "article_no": article_no or article_id})
            except Exception:
                lookup = {}
            chunk_article_cache[source] = lookup
        return chunk_article_cache.get(source, {}).get(key, {})

    for i, doc in enumerate(final_docs, start=1):
        metadata = ctx.hit_metadata(doc)
        display_text = ctx.hit_display_text(doc)
        is_full_article = bool(metadata.get("full_article_expanded"))
        excerpt = str(display_text or "").strip() if is_full_article else ctx.build_excerpt(display_text, query, 200)
        section = (metadata.get("section") or "").strip()
        source = ctx.hit_entity_source(doc) or "unknown"
        article_id = str(metadata.get("article_id") or "").strip()
        article_no = str(metadata.get("article_no") or article_id or metadata.get("clause_label") or "").strip()
        if not article_id and not article_no:
            backfilled = article_from_active_chunks(source, metadata.get("chunk_id"))
            article_id = backfilled.get("article_id", "")
            article_no = backfilled.get("article_no", "")
        clause = str(metadata.get("clause") or article_no or article_id or "").strip()
        clause_meta = metadata.get("clause_metadata") if isinstance(metadata.get("clause_metadata"), dict) else {}
        if not clause_meta:
            clause_meta = document_clause_metadata.build_clause_metadata(
                source_file=source,
                doc_title=str(metadata.get("doc_title") or metadata.get("canonical_title") or source),
                item={
                    **metadata,
                    "article_id": article_id or article_no,
                    "article_no": article_no or article_id,
                    "section": section,
                },
                base_metadata=metadata,
                text=display_text,
            ).to_dict()
        source_metadata = {
            "article_id": article_id or article_no,
            "article_no": article_no or article_id,
            "clause": clause,
            "clause_id": clause,
            "clause_label": str(metadata.get("clause_label") or "").strip(),
            "doc_id": clause_meta.get("doc_id") or metadata.get("doc_id") or source,
            "doc_title": _preferred_document_title(ctx, source, metadata),
            "source_file": clause_meta.get("source_file") or source,
            "clause_metadata": clause_meta,
            "metadata_available": document_clause_metadata.metadata_available(
                {
                    **metadata,
                    "doc_id": clause_meta.get("doc_id") or metadata.get("doc_id") or source,
                    "article_no": article_no or article_id,
                    "article_id": article_id or article_no,
                    "clause_id": clause,
                    "clause_metadata": clause_meta,
                }
            ),
            "section": section,
            "section_title": str(metadata.get("section_title") or section).strip(),
            "chunk_id": metadata.get("chunk_id"),
            "chunk_id_start": metadata.get("chunk_id_start"),
            "chunk_id_end": metadata.get("chunk_id_end"),
            "chunk_count": metadata.get("chunk_count"),
            "full_article_expanded": metadata.get("full_article_expanded"),
            "full_article_chunk_count": metadata.get("full_article_chunk_count"),
            "full_article_chunk_ids": metadata.get("full_article_chunk_ids") or [],
            "full_article_text_chars": metadata.get("full_article_text_chars"),
            "source_text_chars": len(excerpt),
            "page_no": metadata.get("page_no"),
            "page_span": metadata.get("page_span") or [],
        }
        items.append({
            "ref": i,
            "source": source,
            "score": ctx.hit_score(doc),
            "section": section,
            "clause": clause,
            "article_id": source_metadata["article_id"],
            "article_no": source_metadata["article_no"],
            "clause_id": source_metadata["clause_id"],
            "doc_id": source_metadata["doc_id"],
            "metadata_available": source_metadata["metadata_available"],
            "chunk_range": ctx.hit_chunk_range(doc),
            "text": excerpt,
            "metadata": source_metadata,
        })
    return items
