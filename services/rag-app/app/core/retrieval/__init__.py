import difflib
import json
import logging
from typing import Any, Dict, List, Optional



from app.core.retrieval.query import (
    retrieval_query_has_doc_noise,
    expand_retrieval_query_from_corpus,
    seed_anchor_terms_for_probe,
    doc_recall_fallback,
    soft_lock_query_anchor_terms,
    text_overlap_ratio,
    edit_similarity_ratio,
    soft_lock_has_duplicate_formats,
    soft_lock_confidence,
    strip_compare_noise_terms,
    strip_raw_text_mentions,
    strip_filename_mentions,
    strip_source_title_mentions,
    purify_retrieval_query_shallow,
    purify_locked_source_query,
)
from app.core.retrieval.filters import (
    build_milvus_filter,
    configured_article_ids_are_valid,
    normalize_configured_article_ids,
    target_article_ids,
)


logger = logging.getLogger("rag-app")


from app.core.retrieval.recall import (
    rrf,
    doc_term_overlap_recall,
    build_doc_recall_plan,
    clarification_probe_terms,
    clarification_chunk_candidate_sources,
    retrieval_backed_clarification_candidates,
)


from app.core.retrieval.lexical import (
    collect_lexical_candidates,
    synthetic_doc_title_hit,
    annotate_lexical_hit,
    build_controlled_expansion_queries,
    distinct_hit_sources,
)

from app.core.retrieval.ranking import (
    chunk_level_rerank,
    source_level_rerank,
    apply_retrieval_filters,
    summarize_source_scores,
    doc_level_rerank,
    fuse_dense_lexical_hits,
    run_lightweight_search_candidates,
    postprocess_recall_docs,
    rerank_and_postprocess_lightweight_docs,
    build_retrieval_stage_trace,
)
from app.core.retrieval.clauses import (
    ClauseUnit,
    build_clause_units,
    clause_aware_text,
    clause_level_rerank,
)

from app.core.retrieval.chunks import (
    merge_and_dedupe_hits,
    aggregate_doc_sections,
    docs_for_query_context,
    chunk_base_relevance,
    chunk_query_signal,
    hybrid_structural_chunk_score,
    intra_doc_chunk_rerank,
    should_keep_structural_chunk,
    filter_low_relevance_sources,
    source_constraint_multiplier,
)
















async def run_target_scoped_recall(
    runtime: Any,
    handler: Any,
    query: str,
    retrieval_query: str,
    query_embedding: List[float],
    qtype: str,
    qfilters: Dict[str, Any],
    recall_k: int,
    final_n: int,
    pool_n: int,
    enable_rerank: bool,
    target_source: str,
    query_sparse_embedding: Optional[Dict[int, float]] = None,
    compare_subquery: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    safe_name = runtime.normalize_filename_for_match(target_source)
    effective_query = runtime.normalize_query((compare_subquery or {}).get("raw_text_query") or retrieval_query) or retrieval_query
    section_query = runtime.normalize_query((compare_subquery or {}).get("section_query") or "")
    doc_prior_query = runtime.normalize_query((compare_subquery or {}).get("doc_prior_query") or effective_query) or effective_query
    if runtime.query_has_compare_intent(query):
        cleaned = runtime.strip_compare_noise_terms(effective_query)
        if cleaned:
            effective_query = cleaned
        cleaned_prior = runtime.strip_compare_noise_terms(doc_prior_query)
        if cleaned_prior:
            doc_prior_query = cleaned_prior
    effective_embedding = query_embedding
    effective_sparse_embedding = query_sparse_embedding
    if effective_query != retrieval_query:
        embeddings, sparse_embeddings = await handler.embedding_service.embed_with_sparse(
            [effective_query],
            return_sparse=True,
        )
        effective_embedding = embeddings[0]
        effective_sparse_embedding = sparse_embeddings[0] if sparse_embeddings else None
    article_ids = target_article_ids(qfilters, effective_query)
    milvus_filter = build_milvus_filter(sources=[safe_name], article_ids=article_ids)
    if len(article_ids) > 1:
        logger.info("retrieval_filter_milvus: target_articles=%s filter=%s", article_ids, milvus_filter)
    docs = handler.vector_db.search(
        effective_embedding,
        top_k=recall_k,
        filters=milvus_filter,
        query_sparse_embedding=effective_sparse_embedding,
    )
    visible_dense = runtime.filter_hits_by_source_state(docs)
    docs = visible_dense["hits"]
    dense_source_scores = runtime.dense_source_score_map(docs)

    doc_recall_plan = runtime.build_doc_recall_plan(doc_prior_query, limit=3, source_filter=safe_name)
    lex_items = runtime.collect_lexical_candidates(effective_query, [safe_name], doc_recall_plan, article_ids=article_ids)
    if section_query and section_query != effective_query:
        try:
            lex_items.extend(
                runtime.lexical_recall_indexed(
                    section_query,
                    max(20, min(160, runtime.config_value("LEXICAL_RECALL_LIMIT", 1000) // 5)),
                    source_filter=safe_name,
                    article_ids=article_ids,
                )
            )
        except Exception:
            try:
                lex_items.extend(
                    runtime.lexical_recall_fallback(
                        section_query,
                        max(20, min(160, runtime.config_value("LEXICAL_RECALL_LIMIT", 1000) // 5)),
                        source_filter=safe_name,
                        article_ids=article_ids,
                    )
                )
            except Exception:
                pass
    visible_lex = runtime.filter_hits_by_source_state(lex_items)
    lex_items = visible_lex["hits"]
    docs_all = docs + lex_items
    if not docs_all:
        return {
            "source": safe_name,
            "evidence_query": effective_query,
            "section_query": section_query,
            "doc_prior_query": doc_prior_query,
            "score_mode": "score",
            "docs": [],
            "dense_hits": docs,
            "lexical_hits": lex_items,
            "selected_docs": [],
            "post_filter_docs": [],
            "retrieve_docs": [],
            "early_filtered": visible_dense["dropped"] + visible_lex["dropped"],
            "visibility_filtered": visible_dense["dropped"] + visible_lex["dropped"],
            "dense_source_scores": dense_source_scores,
            "dense_visible_states": dict(visible_dense.get("states") or {}),
            "lexical_visible_states": dict(visible_lex.get("states") or {}),
            "rerank_used": False,
            "stage_trace": build_retrieval_stage_trace(
                runtime,
                {
                    "dense_hits": visible_dense["hits"],
                    "lexical_hits": visible_lex["hits"],
                    "fused_docs": [],
                    "selected_docs": [],
                },
                score_mode="score",
            ),
            "doc_recall_plan": doc_recall_plan,
        }

    fusion = fuse_dense_lexical_hits(
        runtime,
        docs,
        lex_items,
        effective_query,
        recall_k,
        dense_source_scores=dense_source_scores,
        fname_set={safe_name},
        doc_recall_plan=doc_recall_plan,
    )
    docs = fusion["docs"]
    dense_rank_map = fusion["dense_rank_map"]
    lex_rank_map = fusion["lex_rank_map"]
    source_signals = fusion["source_signals"]
    docs = evidence_core.pin_article_hits(
        runtime.evidence_context(),
        " ".join([effective_query, section_query, retrieval_query, query]),
        docs,
        qfilters=qfilters,
        article_ids=article_ids,
        absolute_bonus=float(runtime.config_value("ARTICLE_ID_PIN_ABSOLUTE_BONUS", 1_000_000.0) or 1_000_000.0),
    )

    chunk_rerank_enabled = runtime.should_apply_chunk_rerank(docs[:pool_n], dense_rank_map, lex_rank_map, source_signals, enable_rerank)
    doc_title = ""
    try:
        doc_title = runtime.source_display_title(safe_name)
    except Exception:
        try:
            doc_title = runtime.display_title(safe_name)
        except Exception:
            doc_title = safe_name
    clause_rerank = await clause_level_rerank(
        runtime,
        handler.rerank_service,
        query,
        docs[:pool_n],
        pool_n,
        chunk_rerank_enabled,
        mentioned_articles=article_ids,
        doc_title=doc_title,
        query_intent=str((qfilters or {}).get("_legal_intent") or ""),
    )
    clause_docs = clause_rerank["hits"]
    try:
        pinned_article_limit = max(1, int(runtime.config_value("PINNED_CLAUSE_RESCUE_MAX_ARTICLES", 3) or 3))
    except Exception:
        pinned_article_limit = 3
    selected_clause_articles: List[str] = []
    for article in (clause_rerank.get("trace") or {}).get("selected_articles") or []:
        value = str(article or "").strip()
        if value and value not in selected_clause_articles:
            selected_clause_articles.append(value)
        if len(selected_clause_articles) >= pinned_article_limit:
            break
    reranked_chunk = await chunk_level_rerank(
        runtime,
        handler.rerank_service,
        query,
        clause_docs[:pool_n],
        pool_n,
        chunk_rerank_enabled,
        query_intent=str((qfilters or {}).get("_legal_intent") or ""),
    )
    reranked_docs = reranked_chunk["hits"]
    score_mode = reranked_chunk["score_mode"]
    heading_expansion_query = section_query or effective_query
    docs = runtime.expand_heading_hits_to_article_hits(heading_expansion_query, safe_name, reranked_docs, limit=max(final_n, 6))
    postprocess_qfilters = dict(qfilters or {})
    if selected_clause_articles:
        postprocess_qfilters["_pinned_article_ids"] = selected_clause_articles

    postprocessed = postprocess_recall_docs(
        runtime,
        docs,
        score_mode=score_mode,
        query=effective_query,
        qtype=qtype,
        qfilters=postprocess_qfilters,
        active_fnames=[safe_name],
        final_n=final_n,
        pinned_source_docs=list(clause_docs or []) + list(reranked_docs or []) + list(docs or []),
    )
    selected_docs = postprocessed["selected_docs"]
    post_filter_docs = postprocessed["post_filter_docs"]
    retrieve_docs = postprocessed["retrieve_docs"]
    stage_trace = build_retrieval_stage_trace(
        runtime,
        {
            "dense_hits": visible_dense["hits"],
            "lexical_hits": visible_lex["hits"],
            "fused_docs": fusion["docs"],
            "clause_docs": clause_docs,
            "reranked_docs": reranked_docs,
            "expanded_docs": docs,
        },
        score_mode=score_mode,
    )
    stage_trace.update(dict(postprocessed.get("stage_trace") or {}))
    if selected_clause_articles:
        stage_trace["pinned_article_ids"] = selected_clause_articles

    return {
        "source": safe_name,
        "evidence_query": effective_query,
        "section_query": section_query,
        "doc_prior_query": doc_prior_query,
        "score_mode": score_mode,
        "docs": docs,
        "dense_hits": visible_dense["hits"],
        "lexical_hits": visible_lex["hits"],
        "selected_docs": selected_docs,
        "post_filter_docs": post_filter_docs,
        "retrieve_docs": retrieve_docs,
        "early_filtered": visible_dense["dropped"] + visible_lex["dropped"],
        "visibility_filtered": visible_dense["dropped"] + visible_lex["dropped"],
        "dense_source_scores": dense_source_scores,
        "dense_visible_states": dict(visible_dense.get("states") or {}),
        "lexical_visible_states": dict(visible_lex.get("states") or {}),
        "rerank_used": bool(reranked_chunk["used"]),
        "clause_rerank": dict(clause_rerank.get("trace") or {}),
        "stage_trace": stage_trace,
        "doc_recall_plan": doc_recall_plan,
    }
