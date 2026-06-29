from app.core.evidence.context import EvidenceAdapter
from app.core.evidence.hits import doc_aspect_evidence_features, is_heading_only_hit


def _metadata(hit):
    return hit.get("metadata") or {}


def test_document_title_section_is_heading_only_evidence():
    hit = {"metadata": {"section": "document_title"}}

    assert is_heading_only_hit(hit, _metadata) is True


def test_title_hit_metadata_is_heading_only_evidence():
    hit = {"metadata": {"title_hit": True}}

    assert is_heading_only_hit(hit, _metadata) is True


def _adapter(chunks=None):
    return EvidenceAdapter(
        normalize_filename_for_match=lambda value: str(value or ""),
        normalize_query=lambda value: str(value or ""),
        hit_display_text=lambda hit: (hit.get("entity") or {}).get("metadata", {}).get("raw_text") or "",
        hit_llm_text=lambda hit: (hit.get("entity") or {}).get("text") or "",
        hit_metadata=lambda hit: (hit.get("entity") or {}).get("metadata") or hit.get("metadata") or {},
        hit_entity_source=lambda hit: (hit.get("entity") or {}).get("source") or "",
        hit_score=lambda hit: float(hit.get("score") or 0.0),
        hit_chunk_range=lambda hit: "",
        source_display_title=lambda value: str(value or ""),
        build_excerpt=lambda text, query, limit: str(text or "")[:limit],
        get_chunks_for_source=lambda source, doc_version=None: list((chunks or {}).get((source, doc_version), [])),
        doc_section_name=lambda hit: str(((hit.get("entity") or {}).get("metadata") or {}).get("section") or ""),
        normalize_coverage_aspect=lambda value: str(value or ""),
        coverage_aspect_variants=lambda value: [str(value or "")],
        chunk_plain_display_text=lambda value: str(value or ""),
        section_target_alignment=lambda section, target: (False, ""),
        query_semantic_aspects=lambda query, qfilters=None: {},
        doc_semantic_aspect_hits=lambda hit, aspects: [],
        chunk_base_relevance=lambda hit, score_mode: 0.0,
        normalize_topics=lambda value: [],
        is_generic_section_title=lambda value: False,
        rrf=lambda rank, k: 0.0,
    )


def test_long_body_without_structural_semantics_does_not_qualify_by_length():
    hit = {
        "entity": {
            "source": "demo.pdf",
            "text": "x" * 80,
            "metadata": {"chunk_role": "body", "raw_text": "x" * 80, "section": "General", "doc_version": 1},
        },
        "score": 0.8,
    }

    features = doc_aspect_evidence_features(_adapter(), hit, "处罚")

    assert features["qualifies"] is False


def test_body_chunk_inherits_semantic_signal_from_parent_section():
    source = "demo.pdf"
    chunks = {
        (source, 1): [
            {
                "raw_text": "处罚规定",
                "section": "处罚规定",
                "chunk_id": 1,
                "metadata": {
                    "chunk_role": "section_heading",
                    "section": "处罚规定",
                    "section_node_id": "section::penalty",
                    "doc_version": 1,
                },
            },
            {
                "raw_text": "未登记犬只",
                "section": "处罚规定",
                "chunk_id": 2,
                "metadata": {
                    "chunk_role": "body",
                    "section": "处罚规定",
                    "section_node_id": "section::penalty",
                    "doc_version": 1,
                },
            },
        ]
    }
    hit = {
        "entity": {
            "source": source,
            "text": "未登记犬只",
            "metadata": {
                "chunk_role": "body",
                "raw_text": "未登记犬只",
                "section": "处罚规定",
                "section_node_id": "section::penalty",
                "doc_version": 1,
            },
        },
        "score": 0.8,
    }

    features = doc_aspect_evidence_features(_adapter(chunks), hit, "处罚")

    assert features["inherited_hits"] > 0
    assert features["inherited_action_signal"] is True
    assert features["qualifies"] is True
