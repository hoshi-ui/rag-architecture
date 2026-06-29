"""Runtime adapter lexical helpers for retrieval operations."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from app.core import compare as compare_core
from app.core import evidence as evidence_core
from app.core import retrieval as retrieval_core
from app.core.query import routing as routing_core
from app.core.retrieval.filters import normalize_configured_article_ids, sqlite_article_filter_sql
from app.core.retrieval import rerank as rerank_core
from app.core.source import state as source_state_core
from app.documents import profile as document_profile
from app.utils import scoring as scoring_utils

class RetrievalLexicalMixin:
    def lexical_recall_indexed(
        self,
        query: str,
        limit: int,
        source_filter: Optional[str] = None,
        article_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        return self.runtime.source.lex_store.lexical_recall(
            query,
            limit,
            source_filter=source_filter,
            article_ids=article_ids,
            normalize_source=self.runtime.common.normalize_filename,
        )
    def lexical_recall_fallback(
        self,
        query: str,
        limit: int,
        source_filter: Optional[str] = None,
        article_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        terms = self.runtime.common.query_match_terms(query)
        if not terms:
            return []
        effective_limit = max(20, min(int(limit or 0), 160 if source_filter else 80))
        where_parts: List[str] = []
        params: List[Any] = []
        if source_filter:
            where_parts.append("m.source = ?")
            params.append(source_filter)
        article_clause, article_params = sqlite_article_filter_sql(article_ids)
        if article_clause:
            where_parts.append(article_clause)
            params.extend(article_params)
        like_parts = []
        for term in terms:
            like_parts.append("f.text LIKE ?")
            params.append(f"%{term}%")
        if like_parts:
            where_parts.append("(" + " OR ".join(like_parts) + ")")
        where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
        sql = (
            "SELECT m.id, f.text, m.source, m.section, m.metadata "
            "FROM chunks_meta m JOIN chunks_fts f ON f.rowid = m.id"
            f"{where_sql} LIMIT ?"
        )
        params.append(max(50, min(effective_limit * 4, 400)))
        normalized_articles = normalize_configured_article_ids(article_ids or [])
        if len(normalized_articles) > 1:
            import logging

            logging.getLogger("rag-app").info(
                "retrieval_filter_fts_fallback: target_articles=%s sql=%s params=%s",
                normalized_articles,
                sql,
                params,
            )
        rows = self.runtime.source.lex_store.connect().execute(sql, tuple(params)).fetchall()
        texts = [row[1] or "" for row in rows]
        bm25 = scoring_utils.bm25_scores(query, texts)
        ranked: List[Tuple[float, Dict[str, Any]]] = []
        for idx, row in enumerate(rows):
            _, text, source, section, metadata = row
            try:
                md = json.loads(metadata or "{}")
            except Exception:
                md = {}
            source_name = self.runtime.common.normalize_filename(source or "")
            if not source_name:
                continue
            hit = {
                "entity": {
                    "source": source_name,
                    "text": text or "",
                    "metadata": {
                        **md,
                        "section": section or "",
                        "lexical_signal": md.get("lexical_signal") or "chunk_fallback",
                    },
                },
                "score": 0.0,
            }
            if not source_state_core.hit_matches_source_state(
                hit,
                self.runtime.source.source_state(source_name),
                self.runtime.evidence.hit_metadata,
            ):
                continue
            score = self.token_overlap_score(query, text or "") + (bm25[idx] if idx < len(bm25) else 0.0)
            if score > 0:
                ranked.append((score, hit))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [hit for _, hit in ranked[:effective_limit]]
    def synthetic_doc_title_hit(
        self,
        source: str,
        query: str,
        score: float = 1.0,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return retrieval_core.synthetic_doc_title_hit(self, source, query, score=score, metadata_updates=metadata_updates)
    def annotate_lexical_hit(
        self,
        query: str,
        hit: Dict[str, Any],
        allowed_set: set[str],
        doc_recall_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return retrieval_core.annotate_lexical_hit(self, query, hit, allowed_set, doc_recall_map=doc_recall_map)
    def build_controlled_expansion_queries(self, query: str, allowed_docs: List[str]) -> List[Dict[str, str]]:
        return retrieval_core.build_controlled_expansion_queries(self, query, allowed_docs)
    def collect_lexical_candidates(
        self,
        query: str,
        safe_names: List[str],
        doc_recall_plan: List[Dict[str, Any]],
        article_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        return retrieval_core.collect_lexical_candidates(self, query, safe_names, doc_recall_plan, article_ids=article_ids)
    def filter_hits_by_source_state(self, hits: List[Any]) -> Dict[str, Any]:
        return source_state_core.filter_hits_by_source_state(
            hits,
            self.runtime.common.normalize_filename,
            self.runtime.evidence.hit_entity_source,
            self.runtime.evidence.hit_metadata,
            self.runtime.source.source_state,
        )
    def build_source_signal_map(
        self,
        query: str,
        lex_items: List[Dict[str, Any]],
        doc_recall_plan: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        signal_map: Dict[str, Dict[str, Any]] = {}
        for rank, entry in enumerate(doc_recall_plan or []):
            src = self.runtime.common.normalize_filename((entry or {}).get("source") or "")
            if not src:
                continue
            signals = signal_map.setdefault(src, {})
            signals["doc_recall"] = True
            signals["doc_prior"] = float((entry or {}).get("prior", 0.0))
            signals["doc_recall_rank"] = rank
            signals["doc_recall_reasons"] = list((entry or {}).get("reasons") or [])
            if "title_alias_substring" in signals["doc_recall_reasons"]:
                signals["title_hit"] = True
        for item in lex_items or []:
            src = self.runtime.common.normalize_filename(self.runtime.evidence.hit_entity_source(item) or "")
            if not src:
                continue
            signals = signal_map.setdefault(src, {})
            signals["lexical_hit"] = True
            md = self.runtime.evidence.hit_metadata(item)
            lexical_signal = (md.get("lexical_signal") or "").strip()
            if lexical_signal:
                signals[lexical_signal] = True
            if md.get("doc_recall_hit"):
                signals["doc_recall"] = True
            if md.get("doc_prior") is not None:
                signals["doc_prior"] = max(float(signals.get("doc_prior", 0.0)), float(md.get("doc_prior", 0.0)))
            if md.get("title_hit") or self.doc_title_alias_hit(src, query):
                signals["title_hit"] = True
        for src, signals in signal_map.items():
            if self.doc_title_alias_hit(src, query):
                signals["title_hit"] = True
        return signal_map
    def clone_hit_with_score(self, hit: Any, score: float) -> Dict[str, Any]:
        entity = dict((hit or {}).get("entity") or {}) if isinstance(hit, dict) else dict(getattr(hit, "entity", None) or {})
        metadata = dict(entity.get("metadata") or {})
        if "orig_score" not in metadata:
            metadata["orig_score"] = float(self.runtime.evidence.hit_score(hit) or 0.0)
            metadata["orig_score_mode"] = evidence_core.hit_score_mode(hit)
        metadata["fusion_score"] = float(score)
        entity["metadata"] = metadata
        return {"entity": entity, "score": float(score)}
    def fusion_source_score(
        self,
        src: str,
        query: str,
        dense_rank_map: Dict[str, int],
        lex_rank_map: Dict[str, int],
        source_count: Dict[str, int],
        source_signals: Dict[str, Dict[str, Any]],
        fname_set: set,
        allowed_set: set,
        weak_query: bool,
    ) -> float:
        k = int(getattr(self.runtime.config, "RRF_K", 60))
        dense_score = float(getattr(self.runtime.config, "FUSION_W_DENSE", 0.72)) * self.runtime.rrf(dense_rank_map.get(src), k)
        lex_score = float(getattr(self.runtime.config, "FUSION_W_LEX", 0.20)) * self.runtime.rrf(lex_rank_map.get(src), k)
        signals = source_signals.get(src, {})
        multiplier = 1.0
        if signals.get("title_hit"):
            multiplier *= max(1.0, float(getattr(self.runtime.config, "FUSION_M_TITLE", 1.35)))
        if signals.get("doc_recall"):
            multiplier *= max(1.0, float(getattr(self.runtime.config, "FUSION_M_DOC_RECALL", 1.2)))
        if signals.get("lexical_hit"):
            multiplier *= max(1.0, float(getattr(self.runtime.config, "FUSION_M_TERM", 1.08)))
        if dense_rank_map.get(src) is not None and (signals.get("lexical_hit") or signals.get("doc_recall")):
            multiplier *= max(1.0, float(getattr(self.runtime.config, "FUSION_M_AGREEMENT", 1.12)))
        doc_prior_bonus = min(max(float(signals.get("doc_prior", 0.0)), 0.0), 1.0) * float(getattr(self.runtime.config, "FUSION_W_DOC_PRIOR", 0.003))
        prior_bonus = min(source_count.get(src, 0) / 20.0, 1.0) * float(getattr(self.runtime.config, "FUSION_W_PRIOR", 0.002))
        return (dense_score + lex_score + doc_prior_bonus + prior_bonus) * multiplier * retrieval_core.source_constraint_multiplier(
            self,
            src,
            query,
            fname_set,
            allowed_set,
            weak_query,
        )
    def should_apply_chunk_rerank(
        self,
        hits: List[Any],
        dense_rank_map: Dict[str, int],
        lex_rank_map: Dict[str, int],
        source_signals: Dict[str, Dict[str, Any]],
        enable_rerank: bool,
    ) -> bool:
        return rerank_core.should_apply_chunk_rerank(
            self.runtime,
            hits,
            dense_rank_map,
            lex_rank_map,
            source_signals,
            enable_rerank,
        )
