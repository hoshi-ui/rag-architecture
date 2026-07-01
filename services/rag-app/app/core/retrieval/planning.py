"""Runtime adapter planning helpers for retrieval operations."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from app.core import compare as compare_core
from app.core import evidence as evidence_core
from app.core import retrieval as retrieval_core
from app.core.query import routing as routing_core
from app.core.retrieval import rerank as rerank_core
from app.core.source import state as source_state_core
from app.documents import profile as document_profile
from app.utils import scoring as scoring_utils

class RetrievalPlanningMixin:
    def build_doc_recall_plan(self, query: str, limit: int, source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        store = self.runtime.source.lex_store
        return retrieval_core.build_doc_recall_plan(
            query,
            limit,
            normalize_query=self.runtime.common.normalize_query,
            normalize_filename=self.runtime.common.normalize_filename,
            document_fts_match_filenames=store.document_fts_match_filenames,
            document_fts_rows=store.document_fts_rows,
            source_state=self.runtime.source.source_state,
            doc_title_alias_score=lambda source, text: document_profile.doc_title_alias_score(
                self.runtime.source.profile_query_context(),
                source,
                text,
            ),
            query_match_terms=self.runtime.common.query_match_terms,
            profile_source_recall=lambda text, limit, source_filter=None: document_profile.profile_source_recall(
                self.runtime.source.profile_query_context(),
                text,
                limit,
                source_filter=source_filter,
            ),
            scan_chunk_text_rows=store.scan_chunk_text_rows,
            doc_fallback_min_prior=float(getattr(self.runtime.config, "DOC_FALLBACK_MIN_PRIOR", 0.18)),
            chunk_scan_limit=int(getattr(self.runtime.config, "DOC_FALLBACK_CHUNK_SCAN_LIMIT", 400)),
            source_filter=source_filter,
        )
    def clarification_candidates(
        self,
        query: str,
        seed_sources: Optional[List[str]] = None,
        limit: int = 3,
    ) -> List[str]:
        return retrieval_core.retrieval_backed_clarification_candidates(
            query,
            seed_sources=seed_sources,
            limit=limit,
            normalize_filename=self.runtime.common.normalize_filename,
            source_state=self.runtime.source.source_state,
            chunk_candidate_sources=self.clarification_chunk_candidate_sources,
            build_doc_recall_plan=lambda text, plan_limit: self.build_doc_recall_plan(text, plan_limit),
        )
    def clarification_probe_terms(self, query: str) -> List[str]:
        return retrieval_core.clarification_probe_terms(
            query,
            normalize_query=self.runtime.common.normalize_query,
            query_anchor_terms=self.runtime.query_anchor_terms,
            query_match_terms=self.runtime.common.query_match_terms,
        )
    def clarification_chunk_candidate_sources(self, query: str, limit: int = 5) -> List[str]:
        return retrieval_core.clarification_chunk_candidate_sources(
            query,
            limit,
            source_hit_counts_by_like=self.runtime.source.lex_store.source_hit_counts_by_like,
            normalize_filename=self.runtime.common.normalize_filename,
            source_state=self.runtime.source.source_state,
            probe_terms=self.clarification_probe_terms,
        )
    def soft_lock_confidence(self, *args: Any, **kwargs: Any) -> Any:
        query = str(args[0] if len(args) > 0 else kwargs.pop("query", ""))
        source = str(args[1] if len(args) > 1 else kwargs.pop("source", ""))
        candidate_sources = list(args[2] if len(args) > 2 else kwargs.pop("candidate_sources", []) or [])
        raw_title_score = float(kwargs.pop("raw_title_score", 0.0) or 0.0)
        top_competitors = kwargs.pop("top_competitors", None)
        return retrieval_core.soft_lock_confidence(
            query,
            source,
            candidate_sources,
            raw_title_score=raw_title_score,
            top_competitors=top_competitors,
            normalize_filename=self.runtime.common.normalize_filename,
            normalize_reference_text=document_profile.normalize_reference_text,
            collapse_sources_by_canonical=self.runtime.source.collapse_by_canonical,
            source_display_title=self.runtime.source.display_title,
            query_matches_source_region_or_landmark=self.runtime.source.query_matches_source_region_or_landmark,
            geo_context_tokens=self.runtime.source.geo_context_tokens,
            soft_lock_query_anchor_terms_fn=lambda text, safe_source: retrieval_core.soft_lock_query_anchor_terms(
                text,
                safe_source,
                source_title_aspect_terms=self.runtime.source.source_title_aspect_terms,
                query_semantic_aspects=self.runtime.common.query_semantic_aspects,
                query_content_anchor_terms=self.query_content_anchor_terms,
                extract_section_query_targets=self.runtime.routing.extract_section_query_targets,
                local_validate_section_targets=self.runtime.routing.local_validate_section_targets,
                normalize_coverage_aspect=self.runtime.normalize_coverage_aspect,
                normalize_query=self.runtime.common.normalize_query,
            ),
            source_body_anchor_match_count=self.runtime.source.source_body_anchor_match_count,
            source_supports_doc_identity_term=self.runtime.source.source_supports_doc_identity_term,
            rank_title_source_matches=lambda text, limit=6, include_topic_like=True: document_profile.rank_title_source_matches(
                self.runtime.source.profile_query_context(),
                text,
                limit=limit,
                include_topic_like=include_topic_like,
            ),
            clamp01=lambda value: max(0.0, min(1.0, float(value))),
        )
    def has_doc_noise(
        self,
        retrieval_query: str,
        *,
        locked_title: str = "",
        locked_sources: Optional[List[str]] = None,
    ) -> bool:
        return retrieval_core.retrieval_query_has_doc_noise(
            retrieval_query,
            locked_title=locked_title,
            locked_sources=locked_sources,
            normalize_query=self.runtime.common.normalize_query,
        )
    def strip_filename_mentions(self, query: str, fnames: List[str]) -> str:
        return retrieval_core.strip_filename_mentions(
            query,
            fnames,
            self.runtime.common.normalize_query,
        )
    def strip_source_title_mentions(self, query: str, sources: List[str]) -> str:
        return retrieval_core.strip_source_title_mentions(
            query,
            sources,
            normalize_query=self.runtime.common.normalize_query,
            doc_title_alias_candidates=self.runtime.source.title_alias_candidates,
        )
    def strip_raw_text_mentions(self, query: str, values: List[str]) -> str:
        return retrieval_core.strip_raw_text_mentions(
            query,
            values,
            self.runtime.common.normalize_query,
        )
    def purify_shallow(self, query: str) -> str:
        return retrieval_core.purify_retrieval_query_shallow(query)
    def purify_locked_source_query(self, query: str, sources: List[str]) -> str:
        return retrieval_core.purify_locked_source_query(
            query,
            sources,
            normalize_query=self.runtime.common.normalize_query,
            doc_title_alias_candidates=self.runtime.source.title_alias_candidates,
        )
    def expand_from_corpus(self, query: str, retrieval_query: str) -> Tuple[str, List[str]]:
        return retrieval_core.expand_retrieval_query_from_corpus(query, retrieval_query)
