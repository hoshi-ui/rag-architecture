from app.core.evidence.context import EvidenceAdapter, _evidence_context


from app.core.evidence.hits import (
    hit_entity,
    hit_metadata,
    hit_entity_source,
    hit_entity_text,
    hit_score,
    hit_score_mode,
    hit_display_text,
    hit_llm_text,
    doc_section_name,
    is_generic_section_title,
    hit_chunk_role,
    hit_is_context_expanded,
    is_heading_only_hit,
    text_has_legal_action_signal,
    is_substantive_short_legal_evidence,
    has_clause_like_body_evidence,
    aspect_requires_body_evidence,
    doc_aspect_evidence_features,
    doc_matches_semantic_aspect,
    aspect_doc_priority_score,
    doc_semantic_aspect_hits,
    hit_chunk_id,
    hit_chunk_range,
    _to_int,
    chunk_position_id,
    build_excerpt,
    _token_encoder,
    estimate_token_count,
    evidence_relevance,
)


from app.core.evidence.compare import (
    compare_answer_snippet,
    summarize_compare_source_blocks,
    format_compare_evidence,
    format_single_doc_compare_evidence,
    filter_identity_noise_aspects,
    compare_matrix_presence_state,
    compare_presence_state_for_observations,
    evidence_observations,
    finalize_compare_evidence_observations,
    compare_source_status_entry,
    compare_evidence_observations_async,
)

from app.core.evidence.selection import (
    select_retrieve_output_docs,
    select_process_output_docs,
    filter_display_sources,
    dedupe_evidence_docs,
    merge_compare_source_doc_groups,
    expand_docs_with_full_article_chunks,
    clone_context_expanded_hit,
    expand_docs_with_neighbor_chunks,
)

from app.core.evidence.format import (
    format_evidence,
    build_sources,
)

