"""Runtime adapter profile helpers for source operations."""

import re
import math
from typing import Any, Dict, List, Optional, Tuple

from app.core import source as source_resolution_core
from app.core.source import state as source_state_core
from app.documents import chunking as document_chunking
from app.documents import common as document_common
from app.documents import ir as document_ir_helpers
from app.documents import ir_store as document_ir_store
from app.documents import profile as document_profile
from app.utils import files as file_utils

class SourceIdentityMixin:
    def regulation_identity_key(self, source: str) -> str:
        return source_resolution_core.regulation_identity_key(
            source,
            normalize_filename=self.runtime.common.normalize_filename,
            source_display_title=self.display_title,
            normalize_reference_text=document_profile.normalize_reference_text,
            source_profile_fields=self.source_profile_fields,
            normalize_query=self.runtime.common.normalize_query,
            extract_region_token=source_resolution_core.extract_region_token,
        )
    def source_core_entities(self, source: str) -> List[str]:
        return source_resolution_core.source_core_entities(
            source,
            doc_get=self.doc_get,
            filename_stem=self.filename_stem,
            normalize_query=self.runtime.common.normalize_query,
        )
    def query_matches_source_region_or_landmark(self, query: str, source: str) -> bool:
        return source_resolution_core.query_matches_source_region_or_landmark(
            query,
            source,
            normalize_query=self.runtime.common.normalize_query,
            source_display_title=self.display_title,
            source_profile_fields=self.source_profile_fields,
            source_core_entities=self.source_core_entities,
            generic_doc_intent_terms=self.generic_doc_intent_terms(),
        )
    def source_supports_doc_identity_term(self, source: str, query: str) -> bool:
        terms = [
            term
            for term in ["条例", "办法", "规定", "规则", "规程", "细则", "通知", "意见", "决定", "方案", "标准"]
            if term in self.runtime.common.normalize_query(query)
        ]
        if not terms:
            return False
        title_candidates = [self.doc_get(source).get("canonical_title") or ""] + self.title_alias_candidates(source)
        haystack = "\n".join(
            self.runtime.common.normalize_query(item)
            for item in title_candidates
            if self.runtime.common.normalize_query(item)
        )
        return bool(haystack and any(self.runtime.common.normalize_query(term) in haystack for term in terms))
    def source_body_anchor_match_count(self, source: str, anchors: List[str]) -> int:
        safe_source = self.runtime.common.normalize_filename(source or "")
        if not safe_source or not anchors or not self.source_state(safe_source).get("visible"):
            return 0
        matched = 0
        for anchor in anchors:
            hit = False
            variants = [
                variant
                for variant in self.runtime.coverage_aspect_variants(anchor)
                if len(self.runtime.common.normalize_query(variant)) >= 2
            ]
            for variant in variants[:6]:
                if self.lex_store.has_section_or_body_like(safe_source, f"%{variant}%"):
                    hit = True
                    break
            if hit:
                matched += 1
        return matched
    def validate_source_lock_candidate(
        self,
        query: str,
        target_text: str,
        source: str,
        *,
        prior: float = 0.0,
        match_kind: str = "",
    ) -> Dict[str, Any]:
        safe_source = self.runtime.common.normalize_filename(source or "")
        normalized_query = self.runtime.common.normalize_query(query)
        normalized_target = self.runtime.common.normalize_query(target_text)
        if not safe_source:
            return {"accepted": False, "score": 0.0, "reasons": ["empty_source"], "hard_negative": True}

        display_title = self.display_title(safe_source) or safe_source
        profile = self.source_profile_fields(safe_source)
        source_region = self.runtime.common.normalize_query(
            profile.get("region") or source_resolution_core.extract_region_token(display_title)
        )
        query_region = self.runtime.common.normalize_query(
            source_resolution_core.extract_region_token(normalized_target)
            or source_resolution_core.extract_region_token(normalized_query)
        )

        reasons: List[str] = []
        if query_region and source_region and query_region not in source_region and source_region not in query_region:
            return {
                "accepted": False,
                "score": 0.0,
                "reasons": [f"region_mismatch:{query_region}!={source_region}"],
                "hard_negative": True,
                "query_region": query_region,
                "source_region": source_region,
            }

        hay_title = "\n".join(
            self.runtime.common.normalize_query(item)
            for item in [display_title, *(self.title_alias_candidates(safe_source) or [])]
            if self.runtime.common.normalize_query(item)
        )
        title_probe = self.runtime.common.normalize_query(f"{normalized_target} {normalized_query}")
        title_match = bool(
            hay_title
            and title_probe
            and any(
                len(value) >= 4 and (value in title_probe or title_probe in value)
                for value in hay_title.splitlines()
            )
        )
        if title_match:
            reasons.append("title_or_alias_match")

        if query_region and (not source_region or query_region in source_region or source_region in query_region):
            reasons.append("region_match")

        generic_terms = {
            "条例", "办法", "规定", "规则", "法规", "文件", "文档", "管理",
            "相关", "分别", "什么", "哪些", "比较", "对比", "说明", "查询",
        }
        if query_region:
            generic_terms.add(query_region)
        raw_anchors = []
        for term in list(self.runtime.query_anchor_terms(f"{normalized_query} {normalized_target}") or []):
            value = self.runtime.common.normalize_query(term)
            if len(value) < 2 or value in generic_terms:
                continue
            if source_region and value == source_region:
                continue
            raw_anchors.append(value)
        anchors = list(dict.fromkeys(raw_anchors))[:6]
        anchor_hits = self.source_body_anchor_match_count(safe_source, anchors) if anchors else 0
        required_hits = 0
        if anchors:
            required_hits = max(1, min(2, int(math.ceil(float(len(anchors)) * 0.5))))
        if anchors and anchor_hits >= required_hits:
            reasons.append(f"anchor_hit:{anchor_hits}/{len(anchors)}")

        try:
            prior_value = max(0.0, min(1.0, float(prior or 0.0)))
        except Exception:
            prior_value = 0.0
        score = 0.0
        if title_match:
            score += float(getattr(self.runtime.config, "SOURCE_LOCK_TITLE_MATCH_BONUS", 0.35))
        if "region_match" in reasons:
            score += float(getattr(self.runtime.config, "SOURCE_LOCK_REGION_MATCH_BONUS", 0.25))
        if anchors and anchor_hits >= required_hits:
            score += float(getattr(self.runtime.config, "SOURCE_LOCK_ANCHOR_MATCH_BONUS", 0.30))
        if prior_value > 0:
            score += min(
                float(getattr(self.runtime.config, "SOURCE_LOCK_PRIOR_BONUS_CAP", 0.15)),
                prior_value * float(getattr(self.runtime.config, "SOURCE_LOCK_PRIOR_BONUS_WEIGHT", 0.15)),
            )

        exact_like = (match_kind or "").strip() in {"exact_title", "alias_title", "agentic_title_candidate", "agentic_strong_title"}
        accepted = bool(
            title_match
            or (anchors and anchor_hits >= required_hits)
            or score >= float(getattr(self.runtime.config, "SOURCE_LOCK_MIN_ACCEPT_SCORE", 0.55))
        )
        if exact_like and query_region and source_region and "region_match" in reasons:
            accepted = True
        if not accepted:
            reasons.append("insufficient_source_lock_evidence")
        return {
            "accepted": accepted,
            "score": round(float(score), 4),
            "reasons": reasons,
            "hard_negative": False,
            "query_region": query_region,
            "source_region": source_region,
            "anchors": anchors,
            "anchor_hits": anchor_hits,
            "required_anchor_hits": required_hits,
        }
    def generic_doc_intent_terms(self) -> set[str]:
        values = set(self.runtime.generic_query_terms or set())
        values.update(self.runtime.common.policy_keywords("weak_reference.generic_doc_markers"))
        return {self.runtime.common.normalize_query(value) for value in values if self.runtime.common.normalize_query(value)}
    def is_pseudo_singleton_soft_lock(self, query: str, source: str) -> bool:
        return source_resolution_core.is_pseudo_singleton_soft_lock(
            query,
            source,
            normalize_filename=self.runtime.common.normalize_filename,
            normalize_query=self.runtime.common.normalize_query,
            source_display_title=self.display_title,
            source_profile_fields=self.source_profile_fields,
            source_core_entities=self.source_core_entities,
            generic_doc_intent_terms=self.generic_doc_intent_terms(),
        )
    def resolve_unique_weak_match_upgrade(self, query: str, sources: List[str]) -> Dict[str, Any]:
        return source_resolution_core.resolve_unique_weak_match_upgrade(
            query,
            sources,
            collapse_sources_by_canonical=self.collapse_by_canonical,
            source_display_title=self.display_title,
            normalize_reference_text=document_profile.normalize_reference_text,
            find_same_title_candidates=lambda title, exclude_source="": document_profile.find_same_title_candidates(
                self.profile_store(),
                title,
                exclude_source=exclude_source,
            ),
            visible_document_exists=self.visible_document_exists,
            is_pseudo_singleton_soft_lock=self.is_pseudo_singleton_soft_lock,
            min_score=float(getattr(self.runtime.config, "SOURCE_WEAK_MATCH_UPGRADE_MIN_SCORE", 0.70)),
        )
    def source_effective_rank(self, source: str) -> Tuple[int, int, int, int, str]:
        return source_resolution_core.source_effective_rank(
            source,
            normalize_filename=self.runtime.common.normalize_filename,
            doc_get=self.doc_get,
            source_profile_fields=self.source_profile_fields,
            source_display_title=self.display_title,
            normalize_query=self.runtime.common.normalize_query,
        )
    def prefer_latest_effective_sources(self, sources: List[str], limit: Optional[int] = None) -> List[str]:
        return source_resolution_core.prefer_latest_effective_sources(
            sources,
            normalize_filename=self.runtime.common.normalize_filename,
            regulation_identity_key=self.regulation_identity_key,
            source_effective_rank=self.source_effective_rank,
            limit=limit,
        )
    def prepare_explicit_regulation_candidates(self, sources: List[str], limit: int = 5) -> List[str]:
        return source_resolution_core.prepare_explicit_regulation_candidates(
            sources,
            normalize_filename=self.runtime.common.normalize_filename,
            prefer_latest_effective=self.prefer_latest_effective_sources,
            limit=limit,
        )
    def latest_effective_equivalent_source(self, source: str) -> str:
        safe_source = self.runtime.common.normalize_filename(source or "")
        if not safe_source:
            return ""
        identity_key = self.regulation_identity_key(safe_source)
        sibling_sources = [safe_source]
        seen = {safe_source}

        def add_candidate(raw_source: str) -> None:
            candidate = self.runtime.common.normalize_filename(raw_source or "")
            if not candidate or candidate in seen or not self.visible_document_exists(candidate):
                return
            sibling_sources.append(candidate)
            seen.add(candidate)

        if identity_key:
            for raw_source in self.lex_store.document_sources():
                candidate = self.runtime.common.normalize_filename(raw_source or "")
                if not candidate or candidate == safe_source or not self.visible_document_exists(candidate):
                    continue
                if self.regulation_identity_key(candidate) == identity_key:
                    add_candidate(candidate)

        canonical_id = self.canonical_doc_id(safe_source)
        if canonical_id:
            for raw_source in self.lex_store.document_sources():
                candidate = self.runtime.common.normalize_filename(raw_source or "")
                if not candidate or candidate in seen or not self.visible_document_exists(candidate):
                    continue
                if self.canonical_doc_id(candidate) == canonical_id:
                    add_candidate(candidate)

        for title in [
            self.display_title(safe_source),
            (self.doc_get(safe_source) or {}).get("canonical_title") or "",
        ]:
            for candidate in document_profile.find_same_title_candidates(
                self.profile_store(),
                title,
                exclude_source=safe_source,
            ):
                add_candidate(candidate)

        latest = self.prefer_latest_effective_sources(sibling_sources, limit=max(1, len(sibling_sources)))
        if latest:
            return latest[0]
        prepared = sorted(
            sibling_sources,
            key=lambda item: self.source_effective_rank(item),
            reverse=True,
        )
        return prepared[0] if prepared else safe_source
    def strip_reference_text_from_query(self, query: str, references: List[str]) -> str:
        return source_resolution_core.strip_reference_text_from_query(
            query,
            references,
            self.runtime.common.normalize_query,
        )
    def explicit_content_query(self, query: str, regulation_mentions: List[str]) -> str:
        return source_resolution_core.explicit_content_query(
            query,
            regulation_mentions,
            normalize_query=self.runtime.common.normalize_query,
            strip_reference_text=self.strip_reference_text_from_query,
        )
    def geo_context_tokens(self, query: str, user_id: str = "anonymous") -> List[str]:
        out: List[str] = []
        normalized = self.runtime.common.normalize_query(query)
        for token in re.findall(r"[\u4e00-\u9fff]{2,12}(?:特别行政区|自治区|自治州|自治县|地区|省|市|区|县|旗)", normalized):
            value = self.runtime.common.normalize_query(token)
            if value and value not in out:
                out.append(value)
        if user_id:
            try:
                current = self.runtime.state_store.get_current_locked_document(user_id)
            except Exception:
                current = None
            current_source = self.runtime.common.normalize_filename((current or {}).get("source") or "")
            if current_source:
                profile = self.source_profile_fields(current_source)
                region = self.runtime.common.normalize_query(
                    profile.get("region") or source_resolution_core.extract_region_token(self.display_title(current_source))
                )
                if region and region not in out:
                    out.append(region)
        return out
    def geo_filtered_sources(self, query: str, user_id: str, sources: List[str]) -> List[str]:
        try:
            return source_resolution_core.geo_filtered_sources(
                query,
                user_id,
                sources,
                geo_context_tokens=self.geo_context_tokens,
                normalize_filename=self.runtime.common.normalize_filename,
                source_profile_fields=self.source_profile_fields,
                normalize_query=self.runtime.common.normalize_query,
                extract_region_token=source_resolution_core.extract_region_token,
                source_display_title=self.display_title,
            )
        except Exception:
            return list(sources or [])
class SourceTitleMixin:
    def display_title(self, source: str) -> str:
        return source_resolution_core.source_display_title(
            source,
            doc_get=self.doc_get,
            filename_stem=self.filename_stem,
        )
    def title_alias_candidates(self, source: str) -> List[str]:
        return document_profile.doc_title_alias_candidates(
            self.profile_store(),
            source,
        )
    def source_title_aspect_terms(self, sources: List[str]) -> List[str]:
        out: List[str] = []
        for source in sources or []:
            safe_source = self.runtime.common.normalize_filename(source or "")
            if not safe_source:
                continue
            for candidate in self.title_alias_candidates(safe_source):
                value = self.runtime.common.normalize_query(candidate)
                if len(value) >= 2 and value not in out:
                    out.append(value)
        return out
    def identity_terms_for_validation(self, sources: List[str]) -> List[str]:
        out: List[str] = []
        for source in sources or []:
            safe_source = self.runtime.common.normalize_filename(source or "")
            if not safe_source:
                continue
            for term in self.title_alias_candidates(safe_source):
                value = self.runtime.common.normalize_query(term)
                if len(value) >= 2 and value not in out:
                    out.append(value)
            for entity in self.source_core_entities(safe_source):
                value = self.runtime.common.normalize_query(entity)
                if len(value) >= 2 and value not in out:
                    out.append(value)
        return out
    def extract_explicit_regulation_mentions(self, query: str) -> List[str]:
        return source_resolution_core.extract_explicit_regulation_mentions(
            query,
            normalize_query=self.runtime.common.normalize_query,
            extract_filename_candidates=self.runtime.routing.extract_filename_candidates,
        )
    def strip_section_question_tail(self, text: str) -> str:
        return self.runtime.common.normalize_query(text)
    def profile_query_context(self) -> document_profile.ProfileQueryAdapter:
        return document_profile.ProfileQueryAdapter(
            store=self.profile_store(),
            query_match_terms=self.runtime.common.query_match_terms,
            local_validate_section_targets=self.runtime.routing.local_validate_section_targets,
            extract_section_query_targets=self.runtime.routing.extract_section_query_targets,
            strip_section_question_tail=self.strip_section_question_tail,
            extract_filename_candidates=self.runtime.routing.extract_filename_candidates,
            query_has_strong_business_signal=self.runtime.routing.has_strong_business_signal,
            query_quality_strong_topic_terms=self.runtime.routing.strong_topic_terms,
            is_weak_reference_query=self.runtime.routing.is_weak_reference_query,
        )
    def normalize_title_probe_text(self, text: str) -> str:
        try:
            return document_profile.normalize_title_probe_text(
                self.profile_query_context(),
                text,
            )
        except Exception:
            return document_profile.normalize_reference_text(self.runtime.common.normalize_query(text))
    def classify_title_reference_route(self, query: str, fnames: Optional[List[str]] = None) -> str:
        return document_profile.classify_title_reference_route(
            self.profile_query_context(),
            query,
            fnames=fnames,
        )
    def extract_title_candidates(self, query: str, limit: int = 5) -> List[str]:
        return document_profile.extract_title_source_candidates(
            self.profile_query_context(),
            query,
            limit=limit,
        )
    def strong_title_source_matches(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        return document_profile.extract_strong_title_source_matches(
            self.profile_query_context(),
            query,
            limit=limit,
        )
    def match_sources_for_explicit_title(self, query: str, limit: int = 5) -> List[str]:
        return self.extract_title_candidates(query, limit=limit)
    def embed_text_cached(self, text: str) -> Tuple[float, ...]:
        payload = self.runtime.common.normalize_query(text)
        if not payload:
            return tuple()
        cached = self._embedding_cache.get(payload)
        if cached is not None:
            return cached
        try:
            embedding_service = getattr(self.runtime, "embedding_service", None)
            if embedding_service is None:
                return tuple()
            embedding = embedding_service.embed_one_sync(payload, timeout=20)
            cached = tuple(float(value) for value in embedding)
        except Exception:
            cached = tuple()
        self._embedding_cache[payload] = cached
        if len(self._embedding_cache) > 2048:
            self._embedding_cache.pop(next(iter(self._embedding_cache)))
        return cached
    def dense_title_probe_entries(self) -> Tuple[Tuple[str, str, str], ...]:
        if self._dense_title_probe_cache is not None:
            return self._dense_title_probe_cache
        out: List[Tuple[str, str, str]] = []
        for row in self.lex_store.document_title_rows():
            source = row.get("source")
            canonical_title = row.get("canonical_title")
            aliases = row.get("aliases")
            filename_stem = row.get("filename_stem")
            safe_source = self.runtime.common.normalize_filename(source or "")
            if not safe_source or not self.visible_document_exists(safe_source):
                continue
            display_title = (canonical_title or "").strip() or (filename_stem or "").strip() or safe_source
            parts = [display_title, aliases or "", filename_stem or ""] + self.source_core_entities(safe_source)
            probe_text = "\n".join(str(part or "").strip() for part in parts if str(part or "").strip())
            if probe_text:
                out.append((safe_source, display_title, probe_text))
        self._dense_title_probe_cache = tuple(out)
        return self._dense_title_probe_cache
class SourceStateProfileMixin:
    def profile_store(self) -> document_profile.ProfileStoreAdapter:
        store = self.lex_store
        normalize = self.runtime.common.normalize_query
        return document_profile.ProfileStoreAdapter(
            doc_profile_upsert=store.upsert_profile,
            replace_aliases=lambda source, version, aliases: store.replace_aliases(
                source,
                int(version),
                aliases,
                normalize_text=normalize,
            ),
            replace_sections=lambda source, version, sections: store.replace_sections(
                source,
                int(version),
                sections,
                normalize_text=normalize,
            ),
            replace_topics=lambda source, version, topics: store.replace_topics(
                source,
                int(version),
                topics,
                normalize_text=normalize,
            ),
            load_document_ir=self.load_document_ir,
            source_by_content_hash=store.source_by_content_hash,
            doc_get=store.get_document,
            sources_by_same_title_group=store.sources_by_same_title_group,
            alias_rows=lambda source, version: store.aliases(source, int(version)),
            section_titles=lambda source, version: store.section_titles(source, int(version)),
            topic_terms=lambda source, version: store.topics(source, int(version)),
            doc_sources=store.document_sources,
            source_visible=lambda source: bool(self.source_state(source).get("visible")),
        )
    def public_task_status(self, status: Optional[str]) -> str:
        normalized = (status or "").strip().lower()
        if not normalized:
            return "unknown"
        if normalized == "uploaded":
            return "accepted"
        if normalized in {
            "validating",
            "parsing",
            "chunking",
            "embedding",
            "embedding_partial",
            "indexing",
            "indexing_sqlite",
            "indexing_vector",
            "profile_building",
            "publish_pending",
            "reindexing",
            "vector_pending",
        }:
            return "indexing"
        if normalized in {"indexed", "completed"}:
            return "completed"
        if normalized in {
            "failed",
            "parse_failed",
            "parse_empty",
            "parse_low_quality",
            "unsupported_or_corrupt",
            "encrypted_file",
            "suspicious_file_type",
            "index_failed",
            "profile_failed",
            "publish_failed",
            "vector_failed",
            "delete_failed",
        }:
            return "failed"
        return normalized
    def source_state(self, source: str) -> Dict[str, Any]:
        return source_state_core.build_source_state(
            source,
            self.runtime.common.normalize_filename,
            self.doc_get,
            self.public_task_status,
        )
    def source_profile_fields(self, source: str) -> Dict[str, Any]:
        safe_source = self.runtime.common.normalize_filename(source or "")
        if not safe_source:
            return {}
        info = self.doc_get(safe_source)
        try:
            version = int(info.get("active_version") or 0)
        except Exception:
            version = 0
        if version <= 0:
            return {}
        return self.lex_store.profile(safe_source, version)
    def visible_document_exists(self, source: str) -> bool:
        safe_source = self.runtime.common.normalize_filename(source or "")
        if not safe_source:
            return False
        try:
            return bool(self.source_state(safe_source).get("visible"))
        except Exception:
            return bool((self.doc_get(safe_source) or {}).get("status"))
    def document_existence_matches(self, sources: List[str]) -> List[str]:
        out: List[str] = []
        for source in sources or []:
            safe_source = self.runtime.common.normalize_filename(source or "")
            if safe_source and self.visible_document_exists(safe_source) and safe_source not in out:
                out.append(safe_source)
        return out
class SourceProfileMixin(
    SourceTitleMixin,
    SourceStateProfileMixin,
    SourceIdentityMixin,
):
    pass
