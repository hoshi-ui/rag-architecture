"""Runtime adapter base helpers for retrieval operations."""

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


def _normalize_topics(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item or "").strip() for item in value if str(item or "").strip()]
    text = str(value).strip()
    return [text] if text else []


class RetrievalBaseMixin:
    def config_value(self, name: str, default: Any = None) -> Any:
        return getattr(self.runtime.config, name, default)

    def normalize_query(self, query: str) -> str:
        return self.runtime.common.normalize_query(query)

    def normalize_filename_for_match(self, source: str) -> str:
        return self.runtime.common.normalize_filename(source)

    def hit_entity_source(self, hit: Any) -> str:
        return self.runtime.evidence.hit_entity_source(hit)

    def hit_entity_text(self, hit: Any) -> str:
        return evidence_core.hit_entity_text(hit)

    def hit_display_text(self, hit: Any) -> str:
        return evidence_core.hit_display_text(hit)

    def hit_metadata(self, hit: Any) -> Dict[str, Any]:
        return self.runtime.evidence.hit_metadata(hit)

    def hit_chunk_id(self, hit: Any) -> Optional[int]:
        return evidence_core.hit_chunk_id(hit, evidence_core.hit_metadata)

    def chunk_position_id(self, hit: Any) -> Optional[int]:
        return evidence_core.chunk_position_id(hit, evidence_core.hit_metadata)

    def hit_score(self, hit: Any) -> float:
        return self.runtime.evidence.hit_score(hit)

    def hit_score_mode(self, hit: Any) -> str:
        return evidence_core.hit_score_mode(hit)

    def is_heading_only_hit(self, hit: Any) -> bool:
        return evidence_core.is_heading_only_hit(hit, evidence_core.hit_metadata)

    def doc_section_name(self, hit: Any) -> str:
        return self.runtime.evidence.doc_section_name(hit)

    def query_semantic_aspects(self, query: str, qfilters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.runtime.common.query_semantic_aspects(query, qfilters=qfilters)

    def normalize_topics(self, topics: Any) -> List[str]:
        return _normalize_topics(topics)

    def section_target_alignment(self, section: str, query: str) -> tuple:
        return self.runtime.routing.section_target_alignment(section, query)

    def doc_get(self, source: str) -> Dict[str, Any]:
        return self.runtime.source.doc_get(source)

    def filename_stem(self, source: str) -> str:
        return self.runtime.source.filename_stem(source)

    def token_overlap_score(self, query: str, text: str) -> float:
        return scoring_utils.token_overlap_score(query, text, self.runtime.common.query_match_terms)

    def doc_title_alias_score(self, source: str, query: str) -> float:
        return document_profile.doc_title_alias_score(
            self.runtime.source.profile_query_context(),
            source,
            query,
        )

    def doc_title_alias_hit(self, source: str, query: str) -> bool:
        return self.doc_title_alias_score(source, query) > 0.0

    def query_has_compare_intent(self, query: str) -> bool:
        return compare_core.query_has_compare_intent(
            query,
            self.runtime.common.normalize_query,
            self.runtime.routing.classify_question_type,
        )

    def strip_compare_noise_terms(self, text: str) -> str:
        return retrieval_core.strip_compare_noise_terms(text, self.runtime.common.normalize_query)

    def is_weak_reference_query(self, query: str) -> bool:
        return self.runtime.routing.is_weak_reference_query(query)

    def query_anchor_terms(self, query: str) -> List[str]:
        return self.runtime.query_anchor_terms(query)

    def extract_section_query_targets(self, query: str) -> List[str]:
        return self.runtime.routing.extract_section_query_targets(query)

    def clip01(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def infer_rerank_profile(self, query: str, qtype: str = "", qfilters: Optional[Dict[str, Any]] = None) -> str:
        return rerank_core.infer_rerank_profile(self.runtime, query, qtype=qtype)

    def rerank_profile_weights(self, profile: str) -> Dict[str, float]:
        return rerank_core.rerank_profile_weights(self.runtime, profile)

    def section_follow_bonus(
        self,
        section: str = "",
        position: Optional[int] = None,
        section_anchor_positions: Optional[Dict[str, List[int]]] = None,
        profile: str = "",
    ) -> float:
        bonus = float(getattr(self.runtime.config, "HYBRID_STRUCT_FOLLOW_BONUS", 0.16))
        if not section or position is None:
            return bonus
        anchors = list((section_anchor_positions or {}).get(section) or [])
        if not anchors:
            return 0.0
        window = max(0, int(getattr(self.runtime.config, "HYBRID_STRUCT_FOLLOW_WINDOW", 3)))
        if any(0 < int(position) - int(anchor) <= window for anchor in anchors):
            return bonus
        return 0.0

    def generic_chunk_penalty(
        self,
        section: str = "",
        text: str = "",
        query: str = "",
        text_term_hits: float = 0.0,
        section_term_hits: float = 0.0,
        section_score: float = 0.0,
        profile: str = "",
    ) -> float:
        penalty = 0.0
        if section and evidence_core.is_generic_section_title(section):
            penalty += float(getattr(self.runtime.config, "HYBRID_STRUCT_GENERIC_SECTION_PENALTY", 0.08))
        weak_text_signal = float(text_term_hits or 0.0) <= 0.0 and float(section_term_hits or 0.0) <= 0.0 and float(section_score or 0.0) <= 0.0
        if weak_text_signal and len(str(text or "").strip()) < 120:
            penalty += float(getattr(self.runtime.config, "HYBRID_STRUCT_GENERIC_SHORT_PENALTY", 0.16))
        return penalty

    def vector_db(self) -> Any:
        factory = getattr(self.runtime, "create_vector_db", None)
        if callable(factory):
            return factory()
        vector_db = getattr(self.runtime, "vector_db", None)
        if callable(vector_db):
            return vector_db()
        return vector_db

    def dense_source_score_map(self, docs: List[Any], score_mode: str = "score") -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for hit in docs or []:
            source = self.runtime.common.normalize_filename(self.runtime.evidence.hit_entity_source(hit))
            if not source:
                continue
            raw_score = float(self.runtime.evidence.hit_score(hit))
            score = 1.0 / (1.0 + max(raw_score, 0.0)) if score_mode == "distance" else raw_score
            scores[source] = max(float(scores.get(source) or 0.0), score)
        return scores

    def top_ranked_source(self, docs: List[Any], score_mode: str = "score") -> str:
        scores = self.dense_source_score_map(docs, score_mode=score_mode)
        return max(scores.items(), key=lambda item: item[1])[0] if scores else ""

    def source_score_gap(self, docs: List[Any], score_mode: str = "score") -> float:
        scores = sorted(self.dense_source_score_map(docs, score_mode=score_mode).values(), reverse=True)
        return float(scores[0] - scores[1]) if len(scores) >= 2 else (float(scores[0]) if scores else 0.0)

    def source_dense_tiebreak_score(self, source: str, dense_source_scores: Dict[str, float]) -> float:
        safe_source = self.runtime.common.normalize_filename(source or "")
        return float((dense_source_scores or {}).get(safe_source) or 0.0)

    def expand_heading_hits_to_article_hits(self, *args: Any, limit: int = 5) -> List[Any]:
        query = str(args[0] or "") if len(args) >= 3 else ""
        source = self.runtime.common.normalize_filename(args[1] or "") if len(args) >= 3 else ""
        docs = args[2] if len(args) >= 3 else (args[0] if args else [])
        expanded = self.runtime.evidence.expand_docs_with_neighbor_chunks(list(docs or []))
        body_docs = [doc for doc in expanded if not self.is_heading_only_hit(doc)]
        if body_docs or not (query and source):
            return expanded[: max(0, int(limit))]
        rescued = self._rescue_source_body_hits(query, source, limit=max(1, int(limit)))
        if not rescued:
            return expanded[: max(0, int(limit))]
        return self.runtime.evidence.dedupe_docs(rescued + expanded, max(1, int(limit)))

    def _rescue_source_body_hits(self, query: str, source: str, limit: int = 5) -> List[Any]:
        safe_source = self.runtime.common.normalize_filename(source or "")
        if not safe_source:
            return []
        try:
            chunks = self.runtime.source.get_chunks_for_source(safe_source, None)
        except Exception:
            chunks = []
        if not chunks:
            return []
        query_terms = self.runtime.common.query_match_terms(query)
        semantic_terms = self.runtime.common.query_semantic_aspects(query).get("terms") or []
        section_terms = self.runtime.routing.extract_section_query_targets(query, limit=6)
        terms: List[str] = []
        for item in list(section_terms) + list(query_terms) + list(semantic_terms) + ["法律责任", "处罚", "罚款", "罚则", "责令"]:
            value = self.runtime.common.normalize_query(item)
            if len(value) >= 2 and value not in terms:
                terms.append(value)
        ranked: List[Tuple[float, int, Any]] = []
        for idx, chunk in enumerate(chunks, start=1):
            hit = self.runtime.source_chunk_to_hit(safe_source, chunk, score=0.0)
            if self.is_heading_only_hit(hit):
                continue
            text = self.runtime.common.normalize_query(
                " ".join(
                    [
                        str((chunk or {}).get("section") or ""),
                        str((chunk or {}).get("raw_text") or (chunk or {}).get("text") or ""),
                    ]
                )
            )
            if not text:
                continue
            matched = [term for term in terms if term and term in text]
            if not matched:
                continue
            section = self.runtime.common.normalize_query((chunk or {}).get("section") or "")
            score = float(len(matched))
            if any(term in section for term in ("法律责任", "罚则", "处罚")):
                score += float(getattr(self.runtime.config, "HEADING_RESCUE_LEGAL_SECTION_BONUS", 3.0))
            if any(term in text for term in ("处罚", "罚款", "责令", "没收", "法律责任")):
                score += float(getattr(self.runtime.config, "HEADING_RESCUE_LEGAL_BODY_BONUS", 1.5))
            hit["score"] = score
            ranked.append((score, -idx, hit))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [hit for _, _, hit in ranked[: max(1, int(limit))]]

    def seed_anchor_terms_for_probe(self, query: str) -> List[str]:
        return retrieval_core.seed_anchor_terms_for_probe(query, self.runtime.query_anchor_terms)

    def chunk_base_relevance(self, hit: Any, score_mode: str = "score") -> float:
        if score_mode == "distance":
            return 1.0 / (1.0 + max(float(self.runtime.evidence.hit_score(hit)), 0.0))
        return float(self.runtime.evidence.hit_score(hit))

    def query_content_anchor_terms(
        self,
        query: str,
        qfilters: Optional[Dict[str, Any]] = None,
        exclude_terms: Optional[List[str]] = None,
    ) -> List[str]:
        terms = self.runtime.common.query_match_terms(query)
        terms.extend(self.runtime.common.query_semantic_aspects(query, qfilters=qfilters or {}).get("terms") or [])
        excluded = {
            self.runtime.common.normalize_query(term)
            for term in (exclude_terms or [])
            if self.runtime.common.normalize_query(term)
        }
        out: List[str] = []
        for term in terms:
            normalized = self.runtime.common.normalize_query(term)
            if normalized and normalized not in excluded and normalized not in out:
                out.append(normalized)
        return out[:8]
