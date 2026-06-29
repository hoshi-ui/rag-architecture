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


def _evidence_context(runtime: Any) -> EvidenceAdapter:
    if isinstance(runtime, EvidenceAdapter):
        return runtime
    raise TypeError("evidence operations require EvidenceAdapter")
