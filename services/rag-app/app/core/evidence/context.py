from typing import Any, Callable, Dict, List, Optional


class EvidenceAdapter:
    def __init__(
        self,
        *,
        normalize_filename_for_match: Callable[[str], str],
        normalize_query: Callable[[str], str],
        hit_display_text: Callable[[Any], str],
        hit_llm_text: Callable[[Any], str],
        hit_metadata: Callable[[Any], Dict[str, Any]],
        hit_entity_source: Callable[[Any], str],
        hit_score: Callable[[Any], float],
        hit_chunk_range: Callable[[Any], str],
        source_display_title: Callable[[str], str],
        build_excerpt: Callable[[str, str, int], str],
        get_chunks_for_source: Callable[[str, Optional[int]], List[Dict[str, Any]]],
        doc_section_name: Callable[[Any], str],
        normalize_coverage_aspect: Callable[[str], str],
        coverage_aspect_variants: Callable[[str], List[str]],
        chunk_plain_display_text: Callable[[str], str],
        section_target_alignment: Callable[[str, str], tuple],
        query_semantic_aspects: Callable[[str, Optional[Dict[str, Any]]], Dict[str, List[str]]],
        doc_semantic_aspect_hits: Callable[[Any, List[str]], List[str]],
        chunk_base_relevance: Callable[[Any, str], float],
        normalize_topics: Callable[[Any], List[str]],
        is_generic_section_title: Callable[[str], bool],
        rrf: Callable[[int, int], float],
        rrf_k: int = 60,
        max_relevance_distance: float = 0.8,
        min_relevance_score: float = 0.25,
        display_distance_margin: float = 0.02,
        display_score_ratio: float = 0.8,
        enable_parent_context_expansion: bool = True,
        parent_context_backward_chunks: int = 0,
        parent_context_forward_chunks: int = 1,
        parent_context_max_extra: int = 8,
        aspect_body_hit_weight: float = 1.45,
        aspect_inherited_hit_weight: float = 1.2,
        aspect_body_exact_weight: float = 0.95,
        aspect_inherited_exact_weight: float = 0.75,
        aspect_section_hit_weight: float = 0.5,
        aspect_section_exact_weight: float = 0.25,
        aspect_generic_section_penalty: float = 0.45,
        aspect_rank_bonus_base: float = 0.25,
        aspect_rank_bonus_decay: float = 0.01,
        aspect_clause_bonus: float = 0.35,
        aspect_substantive_bonus: float = 0.25,
        subject_focus_target_hit_bonus: float = 0.06,
        subject_focus_target_bonus_cap: float = 0.18,
        subject_focus_excluded_hit_penalty: float = 0.18,
        subject_focus_excluded_penalty_cap: float = 0.45,
        subject_focus_unmatched_excluded_penalty: float = 0.12,
    ) -> None:
        self.normalize_filename_for_match = normalize_filename_for_match
        self.normalize_query = normalize_query
        self.hit_display_text = hit_display_text
        self.hit_llm_text = hit_llm_text
        self.hit_metadata = hit_metadata
        self.hit_entity_source = hit_entity_source
        self.hit_score = hit_score
        self.hit_chunk_range = hit_chunk_range
        self.source_display_title = source_display_title
        self.build_excerpt = build_excerpt
        self.get_chunks_for_source = get_chunks_for_source
        self.doc_section_name = doc_section_name
        self.normalize_coverage_aspect = normalize_coverage_aspect
        self.coverage_aspect_variants = coverage_aspect_variants
        self.chunk_plain_display_text = chunk_plain_display_text
        self.section_target_alignment = section_target_alignment
        self.query_semantic_aspects = query_semantic_aspects
        self.doc_semantic_aspect_hits = doc_semantic_aspect_hits
        self.chunk_base_relevance = chunk_base_relevance
        self.normalize_topics = normalize_topics
        self.is_generic_section_title = is_generic_section_title
        self.rrf = rrf
        self.rrf_k = int(rrf_k)
        self.max_relevance_distance = float(max_relevance_distance)
        self.min_relevance_score = float(min_relevance_score)
        self.display_distance_margin = float(display_distance_margin)
        self.display_score_ratio = float(display_score_ratio)
        self.enable_parent_context_expansion = bool(enable_parent_context_expansion)
        self.parent_context_backward_chunks = int(parent_context_backward_chunks)
        self.parent_context_forward_chunks = int(parent_context_forward_chunks)
        self.parent_context_max_extra = int(parent_context_max_extra)
        self.aspect_body_hit_weight = float(aspect_body_hit_weight)
        self.aspect_inherited_hit_weight = float(aspect_inherited_hit_weight)
        self.aspect_body_exact_weight = float(aspect_body_exact_weight)
        self.aspect_inherited_exact_weight = float(aspect_inherited_exact_weight)
        self.aspect_section_hit_weight = float(aspect_section_hit_weight)
        self.aspect_section_exact_weight = float(aspect_section_exact_weight)
        self.aspect_generic_section_penalty = float(aspect_generic_section_penalty)
        self.aspect_rank_bonus_base = float(aspect_rank_bonus_base)
        self.aspect_rank_bonus_decay = float(aspect_rank_bonus_decay)
        self.aspect_clause_bonus = float(aspect_clause_bonus)
        self.aspect_substantive_bonus = float(aspect_substantive_bonus)
        self.subject_focus_target_hit_bonus = float(subject_focus_target_hit_bonus)
        self.subject_focus_target_bonus_cap = float(subject_focus_target_bonus_cap)
        self.subject_focus_excluded_hit_penalty = float(subject_focus_excluded_hit_penalty)
        self.subject_focus_excluded_penalty_cap = float(subject_focus_excluded_penalty_cap)
        self.subject_focus_unmatched_excluded_penalty = float(subject_focus_unmatched_excluded_penalty)


def _evidence_context(runtime: Any) -> EvidenceAdapter:
    if isinstance(runtime, EvidenceAdapter):
        return runtime
    raise TypeError("evidence operations require EvidenceAdapter")
