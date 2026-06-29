"""Runtime adapter resolution helpers for source operations."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from app.core import source as source_resolution_core
from app.core.source import state as source_state_core
from app.documents import chunking as document_chunking
from app.documents import common as document_common
from app.documents import ir as document_ir_helpers
from app.documents import ir_store as document_ir_store
from app.documents import profile as document_profile
from app.utils import files as file_utils


@dataclass
class SourceResolution:
    status: Literal["locked", "ambiguous", "not_found", "global_fallback"]
    target_doc_ids: List[str]
    target_fnames: List[str]
    confidence: float
    evidence: List[str]
    reason: str
    scope_mode: Literal["doc_locked", "multi_doc_locked", "ambiguous", "not_found", "global"]
    fallback_allowed: bool
    forced_retrieval_allowed: bool


class SourceLockStateMixin:
    def collapse_by_canonical(self, sources: List[str], limit: Optional[int] = None) -> List[str]:
        grouped: Dict[str, str] = {}
        ordered_keys: List[str] = []
        for source in sources or []:
            safe_source = self.runtime.common.normalize_filename(source or "")
            if not safe_source:
                continue
            canonical_id = self.canonical_doc_id(safe_source) or f"source:{safe_source}"
            if canonical_id not in grouped:
                grouped[canonical_id] = safe_source
                ordered_keys.append(canonical_id)
                continue
            try:
                if self.source_effective_rank(safe_source) > self.source_effective_rank(grouped[canonical_id]):
                    grouped[canonical_id] = safe_source
            except Exception:
                continue
        out = [grouped[key] for key in ordered_keys if grouped.get(key)]
        if limit:
            return out[: max(1, int(limit))]
        return out
    def canonical_doc_id(self, source: str) -> str:
        return source_resolution_core.canonical_doc_id_for_source(
            source,
            normalize_filename=self.runtime.common.normalize_filename,
            doc_get=self.doc_get,
            filename_stem=self.filename_stem,
            same_title_group=document_profile.same_title_group,
            normalize_title_probe_text=self.normalize_title_probe_text,
        )
    def set_current_locked_document(self, *args: Any, **kwargs: Any) -> Any:
        return self.runtime.state_store.set_current_locked_document(*args, **kwargs)
    def clear_current_locked_document(self, user_id: str) -> Any:
        return self.runtime.state_store.clear_current_locked_document(user_id)
    def clarification_prompt(self, candidate_sources: List[str]) -> str:
        return source_resolution_core.build_document_clarification_prompt(
            candidate_sources,
            doc_get=self.doc_get,
            filename_stem=self.filename_stem,
            examples_limit=int(self.runtime.common.policy_get("source_resolution.clarification_examples_limit", 3) or 3),
        )
    def retrieval_grounded_clarification_prompt(
        self,
        query: str,
        candidate_sources: List[str],
        reason: str = "document_target_required",
    ) -> str:
        return source_resolution_core.build_retrieval_grounded_clarification_prompt(
            query,
            candidate_sources,
            display_title=self.display_title,
            reason=reason,
        )
    def not_found_prompt(self, target: str) -> str:
        return source_resolution_core.build_document_not_found_prompt(
            target,
            extract_filename_candidates=self.runtime.routing.extract_filename_candidates,
        )
class SourceMatchMixin:
    def normalized_title_candidate_sources(self, text: str, limit: int = 5) -> List[str]:
        return self.extract_title_candidates(text, limit=limit)
    def dense_title_source_matches(self, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            return source_resolution_core.dense_title_source_matches(
                text,
                limit,
                enabled=bool(getattr(self.runtime.config, "ENABLE_DENSE_TITLE_FALLBACK", True)),
                normalize_query=self.runtime.common.normalize_query,
                normalize_reference_text=document_profile.normalize_reference_text,
                embed_text=self.embed_text_cached,
                dense_title_probe_entries=self.dense_title_probe_entries,
                build_doc_recall_plan=lambda query, plan_limit: self.runtime.retrieval.build_doc_recall_plan(query, plan_limit),
                normalize_filename=self.runtime.common.normalize_filename,
                source_display_title=self.display_title,
                max_probe_chars=int(getattr(self.runtime.config, "DENSE_TITLE_PROBE_MAX_CHARS", 160)),
            )
        except Exception:
            return []
    def resolve_explicit_reference_sources(
        self,
        query: str,
        fnames: Optional[List[str]] = None,
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        runtime = self.runtime
        filename_resolution = source_resolution_core.resolve_explicit_filename_sources(
            list(fnames or []),
            normalize_filename=runtime.common.normalize_filename,
            collapse_sources_by_canonical=self.collapse_by_canonical,
            document_existence_matches=self.document_existence_matches,
            build_document_clarification_prompt=self.clarification_prompt,
        )
        if filename_resolution:
            return filename_resolution

        regulation_mentions = self.extract_explicit_regulation_mentions(query)
        if not regulation_mentions:
            return {
                "route": "",
                "required": False,
                "resolved": False,
                "sources": [],
                "candidates": [],
                "reason": "not_explicit_reference",
                "strip_title_mentions": False,
                "clarification": "",
                "target_text": "",
            }

        content_query = self.explicit_content_query(query, regulation_mentions)

        def _resolve_prepared_candidates(
            raw_candidates: List[str],
            allow_soft_lock: bool = False,
            trace_label: str = "",
        ) -> Optional[Dict[str, Any]]:
            return source_resolution_core.resolve_prepared_regulation_candidates(
                raw_candidates,
                query=query,
                user_id=user_id,
                target_text=regulation_mentions[0],
                content_query=content_query,
                allow_soft_lock=allow_soft_lock,
                trace_label=trace_label,
                prepare_candidates=self.prepare_explicit_regulation_candidates,
                latest_effective_equivalent_source=self.latest_effective_equivalent_source,
                is_pseudo_singleton_soft_lock=self.is_pseudo_singleton_soft_lock,
                extract_region_token=source_resolution_core.extract_region_token,
                normalize_query=runtime.common.normalize_query,
                geo_context_tokens=self.geo_context_tokens,
                geo_filtered_sources_fn=self.geo_filtered_sources,
                resolve_unique_weak_match_upgrade=self.resolve_unique_weak_match_upgrade,
                resolve_topical_suffix_multi_doc=self.resolve_topical_suffix_multi_doc,
                build_document_clarification_prompt=self.clarification_prompt,
                source_display_title=self.display_title,
            )

        return source_resolution_core.resolve_explicit_regulation_sources(
            regulation_mentions,
            resolve_prepared_candidates=_resolve_prepared_candidates,
            exact_title_or_alias_source_matches=self.strong_title_source_matches,
            exclusive_entity_source_matches=lambda _text, _limit=5: [],
            match_sources_for_explicit_title=self.match_sources_for_explicit_title,
            extract_title_source_candidates=self.extract_title_candidates,
            normalized_title_candidate_sources=self.normalized_title_candidate_sources,
            dense_title_source_matches=self.dense_title_source_matches,
            normalize_filename=runtime.common.normalize_filename,
            dense_title_match_min_sim=float(getattr(runtime.config, "DENSE_TITLE_MATCH_MIN_SIM", 0.84)),
            dense_title_match_margin=float(getattr(runtime.config, "DENSE_TITLE_MATCH_MARGIN", 0.03)),
            related_marker="鐩稿叧",
        )
    def resolve_topical_suffix_multi_doc(self, query: str, sources: List[str]) -> Dict[str, Any]:
        return source_resolution_core.resolve_topical_suffix_multi_doc(
            query,
            sources,
            collapse_sources_by_canonical=self.collapse_by_canonical,
            normalize_query=self.runtime.common.normalize_query,
            extract_explicit_regulation_mentions=self.extract_explicit_regulation_mentions,
            extract_region_token=source_resolution_core.extract_region_token,
            source_profile_fields=self.source_profile_fields,
            doc_get=self.doc_get,
        )
class SourceTargetingMixin:
    def resolve_targets(
        self,
        query: str,
        fnames: Optional[List[str]] = None,
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        effective_fnames = list(fnames or [])
        if not effective_fnames:
            effective_fnames = self.runtime.routing.extract_filename_candidates(query)
        compare_resolution = self.runtime.compare.analyze_route(query)
        if compare_resolution.get("is_compare"):
            resolved = bool(compare_resolution.get("resolved"))
            return self.build_source_resolution_result(
                route=compare_resolution.get("route") or "open_topic_compare",
                required=bool(compare_resolution.get("required")),
                resolved=resolved,
                sources=list(compare_resolution.get("sources") or []),
                candidates=list(compare_resolution.get("sources") or []),
                reason=compare_resolution.get("reason") or "not_needed",
                strip_title_mentions=bool(compare_resolution.get("strip_title_mentions")),
                clarification=compare_resolution.get("clarification") or "",
                target_text=compare_resolution.get("target_text") or "",
                lock_mode="hard_lock" if resolved else "none",
                lock_confidence=1.0 if resolved else 0.0,
                source_lock_kind="compare_lock",
                compare_subjects=list(compare_resolution.get("subjects") or []),
                compare_doc_like_subjects=list(compare_resolution.get("doc_like_subjects") or []),
                compare_missing_targets=list(compare_resolution.get("missing_doc_targets") or []),
                compare_common_aspects=list(compare_resolution.get("common_aspects") or []),
                compare_topic_pair=list(compare_resolution.get("topic_pair") or []),
                compare_canonical_aspects=list(compare_resolution.get("canonical_aspects") or []),
                compare_expanded_aspects=list(compare_resolution.get("expanded_aspects") or []),
                compare_source_subqueries=dict(compare_resolution.get("source_subqueries") or {}),
                compare_status=compare_resolution.get("compare_status") or "not_compare",
                compare_plan=dict(compare_resolution.get("compare_plan") or {}),
            )

        explicit_resolution = self.resolve_explicit_reference_sources(query, effective_fnames, user_id=user_id)
        if explicit_resolution.get("route") in {"explicit_doc_reference", "explicit_regulation_reference"}:
            resolved = bool(explicit_resolution.get("resolved"))
            return self.build_source_resolution_result(
                route=explicit_resolution.get("route") or "",
                required=bool(explicit_resolution.get("required")),
                resolved=resolved,
                sources=list(explicit_resolution.get("sources") or []),
                candidates=list(explicit_resolution.get("candidates") or []),
                reason=explicit_resolution.get("reason") or "",
                strip_title_mentions=bool(explicit_resolution.get("strip_title_mentions")),
                clarification=explicit_resolution.get("clarification") or "",
                target_text=explicit_resolution.get("target_text") or "",
                lock_mode=explicit_resolution.get("lock_mode") or ("hard_lock" if resolved else "none"),
                lock_confidence=float(explicit_resolution.get("lock_confidence") or (1.0 if resolved else 0.0)),
                lock_message_prefix=explicit_resolution.get("lock_message_prefix") or "",
                source_lock_kind=explicit_resolution.get("source_lock_kind") or "explicit_reference",
                source_resolution_trace=dict(explicit_resolution.get("source_resolution_trace") or {}),
                retrieval_query_override=explicit_resolution.get("retrieval_query_override") or "",
            )
        if explicit_resolution.get("route") == "multi_doc_query":
            return self.build_source_resolution_result(
                route="multi_doc_query",
                required=False,
                resolved=False,
                sources=list(explicit_resolution.get("sources") or []),
                candidates=list(explicit_resolution.get("candidates") or []),
                reason=explicit_resolution.get("reason") or "topical_suffix_multi_doc",
                strip_title_mentions=bool(explicit_resolution.get("strip_title_mentions")),
                clarification=explicit_resolution.get("clarification") or "",
                target_text=explicit_resolution.get("target_text") or "",
                lock_mode="none",
                source_lock_kind=explicit_resolution.get("source_lock_kind") or "topical_suffix_multi_doc",
                source_resolution_trace=dict(explicit_resolution.get("source_resolution_trace") or {}),
                retrieval_query_override=explicit_resolution.get("retrieval_query_override") or "",
            )

        route = self.runtime.routing.classify_query_route(query, effective_fnames)
        if not self.source_lock_required(query, route):
            return self.build_source_resolution_result(
                route=route,
                required=False,
                resolved=False,
                sources=[],
                candidates=[],
                reason="not_needed",
                strip_title_mentions=False,
                clarification="",
                target_text="",
                lock_mode="none",
            )

        title_matches = self.strong_title_source_matches(
            query,
            limit=max(1, int(self.runtime.common.policy_get("source_resolution.title_candidate_limit", 5))),
        )
        title_sources = self.collapse_by_canonical(
            [entry.get("source") for entry in title_matches if entry.get("source")],
            limit=max(1, int(self.runtime.common.policy_get("source_resolution.title_candidate_limit", 5))),
        )
        if len(title_sources) == 1:
            match_kind = str((title_matches[0] or {}).get("match_kind") or "title").strip()
            matched_text = str((title_matches[0] or {}).get("matched_text") or "").strip()
            return self.build_source_resolution_result(
                route=route,
                required=True,
                resolved=True,
                sources=title_sources,
                candidates=title_sources,
                reason="exact_title_unique" if match_kind == "exact_title" else "title_alias_unique",
                strip_title_mentions=True,
                clarification="",
                target_text=matched_text or title_sources[0],
                lock_mode="hard_lock",
                lock_confidence=1.0,
                source_lock_kind="title_unique",
            )
        if len(title_sources) > 1:
            return self.build_source_resolution_result(
                route=route,
                required=True,
                resolved=False,
                sources=[],
                candidates=title_sources,
                reason="document_ambiguous",
                strip_title_mentions=False,
                clarification=self.clarification_prompt(title_sources),
                target_text="",
                lock_mode="none",
                source_lock_kind="candidate_hint",
            )

        fallback = self.runtime.retrieval.build_doc_recall_plan(
            query,
            max(1, int(self.runtime.common.policy_get("source_resolution.fallback_candidate_limit", 3))),
        )
        candidate_sources = self.collapse_by_canonical(
            [item.get("source") for item in fallback if item.get("source")],
            limit=3,
        )
        if len(candidate_sources) == 1 and route in {"exact_title_reference", "alias_title_reference", "weak_title_reference"}:
            return self.build_source_resolution_result(
                route=route,
                required=True,
                resolved=True,
                sources=candidate_sources,
                candidates=candidate_sources,
                reason="doc_recall_unique",
                strip_title_mentions=False,
                clarification="",
                target_text=candidate_sources[0],
                lock_mode="soft_lock",
                lock_confidence=0.72,
                source_lock_kind="doc_recall_unique",
            )
        return self.build_source_resolution_result(
            route=route,
            required=True,
            resolved=False,
            sources=[],
            candidates=candidate_sources,
            reason="document_target_required" if not candidate_sources else "document_ambiguous",
            strip_title_mentions=False,
            clarification=self.clarification_prompt(candidate_sources) if candidate_sources else "Please clarify which document or regulation you mean.",
            target_text="",
            lock_mode="none",
            source_lock_kind="candidate_hint" if candidate_sources else "none",
        )
    def source_lock_required(self, query: str, route: str) -> bool:
        normalized = self.runtime.common.normalize_query(query)
        if route in {
            "explicit_doc_reference",
            "explicit_regulation_reference",
            "exact_title_reference",
            "alias_title_reference",
            "weak_title_reference",
            "version_switch",
        }:
            return True
        if self.runtime.routing.has_contextual_doc_reference(normalized):
            return True
        if route in {"business_topic_qa", "open_regulation_qa", "content_qa"}:
            return False
        return False
    def build_source_resolution_result(
        self,
        *,
        route: str,
        required: bool,
        resolved: bool,
        sources: List[str],
        candidates: List[str],
        reason: str,
        strip_title_mentions: bool,
        clarification: str,
        target_text: str,
        lock_mode: str = "none",
        lock_confidence: float = 0.0,
        lock_message_prefix: str = "",
        source_lock_kind: str = "",
        source_resolution_trace: Optional[Dict[str, Any]] = None,
        retrieval_query_override: str = "",
        compare_subjects: Optional[List[str]] = None,
        compare_doc_like_subjects: Optional[List[str]] = None,
        compare_missing_targets: Optional[List[str]] = None,
        compare_common_aspects: Optional[List[str]] = None,
        compare_topic_pair: Optional[List[str]] = None,
        compare_canonical_aspects: Optional[List[str]] = None,
        compare_expanded_aspects: Optional[List[str]] = None,
        compare_source_subqueries: Optional[Dict[str, Any]] = None,
        compare_status: str = "not_compare",
        compare_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        safe_sources = [self.runtime.common.normalize_filename(source or "") for source in sources or []]
        safe_sources = [source for source in safe_sources if source]
        safe_candidates = [self.runtime.common.normalize_filename(source or "") for source in candidates or []]
        safe_candidates = [source for source in safe_candidates if source]
        status_reason = str(reason or "")
        target_doc_ids = [self.canonical_doc_id(source) or source for source in safe_sources]
        target_doc_ids = [doc_id for doc_id in target_doc_ids if doc_id]
        target_doc_ids = list(dict.fromkeys(target_doc_ids))
        target_fnames = list(dict.fromkeys(safe_sources))
        missing_reasons = {
            "document_not_found",
            "document_target_required",
            "compare_target_not_found",
            "compare_targets_not_found",
            "agentic_router_targets_not_found",
        }
        if bool(resolved) and target_fnames:
            status = "locked"
        elif bool(required) and status_reason in missing_reasons and not safe_candidates:
            status = "not_found"
        elif bool(required):
            status = "ambiguous" if safe_candidates else "not_found"
        else:
            status = "global_fallback"
        evidence: List[str] = []
        evidence.extend([f"source:{source}" for source in target_fnames])
        evidence.extend([f"candidate:{source}" for source in safe_candidates if source not in target_fnames])
        if status_reason:
            evidence.append(f"reason:{status_reason}")
        if status == "locked":
            scope_mode = "multi_doc_locked" if len(target_fnames) > 1 else "doc_locked"
        elif status == "ambiguous":
            scope_mode = "ambiguous"
        elif status == "not_found":
            scope_mode = "not_found"
        else:
            scope_mode = "global"
        fallback_allowed = status == "global_fallback"
        forced_retrieval_allowed = status == "global_fallback"
        source_state = SourceResolution(
            status=status,
            target_doc_ids=target_doc_ids,
            target_fnames=target_fnames,
            confidence=float(lock_confidence or 0.0),
            evidence=evidence,
            reason=status_reason,
            scope_mode=scope_mode,
            fallback_allowed=fallback_allowed,
            forced_retrieval_allowed=forced_retrieval_allowed,
        )
        return {
            "route": route,
            "required": bool(required),
            "resolved": bool(resolved),
            "status": source_state.status,
            "target_doc_ids": source_state.target_doc_ids,
            "target_fnames": source_state.target_fnames,
            "confidence": source_state.confidence,
            "evidence": source_state.evidence,
            "scope_mode": source_state.scope_mode,
            "fallback_allowed": source_state.fallback_allowed,
            "forced_retrieval_allowed": source_state.forced_retrieval_allowed,
            "sources": safe_sources,
            "candidates": safe_candidates,
            "reason": source_state.reason,
            "strip_title_mentions": bool(strip_title_mentions),
            "clarification": clarification,
            "target_text": target_text,
            "lock_mode": lock_mode,
            "lock_confidence": float(lock_confidence or 0.0),
            "lock_message_prefix": lock_message_prefix,
            "source_lock_kind": source_lock_kind,
            "source_resolution_trace": dict(source_resolution_trace or {}),
            "retrieval_query_override": retrieval_query_override,
            "compare_subjects": list(compare_subjects or []),
            "compare_doc_like_subjects": list(compare_doc_like_subjects or []),
            "compare_missing_targets": list(compare_missing_targets or []),
            "compare_common_aspects": list(compare_common_aspects or []),
            "compare_topic_pair": list(compare_topic_pair or []),
            "compare_canonical_aspects": list(compare_canonical_aspects or []),
            "compare_expanded_aspects": list(compare_expanded_aspects or []),
            "compare_source_subqueries": dict(compare_source_subqueries or {}),
            "compare_status": compare_status,
            "compare_plan": dict(compare_plan or {}),
        }
class SourceResolutionMixin(
    SourceMatchMixin,
    SourceTargetingMixin,
    SourceLockStateMixin,
):
    pass
