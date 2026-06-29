from app.core.evidence.context import EvidenceAdapter
from app.core.evidence.format import format_evidence


def _adapter() -> EvidenceAdapter:
    return EvidenceAdapter(
        normalize_filename_for_match=lambda source: source,
        normalize_query=lambda text: text,
        hit_display_text=lambda hit: hit["text"],
        hit_llm_text=lambda hit: hit["text"],
        hit_metadata=lambda hit: hit["metadata"],
        hit_entity_source=lambda hit: hit["source"],
        hit_score=lambda hit: hit.get("score", 1.0),
        hit_chunk_range=lambda hit: hit.get("chunk_range", ""),
        source_display_title=lambda source: source,
        build_excerpt=lambda text, query, limit: text[:limit],
        get_chunks_for_source=lambda source, doc_version=None: [],
        doc_section_name=lambda hit: hit["metadata"].get("section", ""),
        normalize_coverage_aspect=lambda text: text,
        coverage_aspect_variants=lambda text: [text],
        chunk_plain_display_text=lambda text: text,
        section_target_alignment=lambda section, query: (0.0, []),
        query_semantic_aspects=lambda query, parsed=None: {},
        doc_semantic_aspect_hits=lambda hit, terms: [],
        chunk_base_relevance=lambda hit, score_mode: 1.0,
        normalize_topics=lambda value: [],
        is_generic_section_title=lambda section: False,
        rrf=lambda rank, k: 1.0 / (k + rank),
    )


def test_format_evidence_prefers_source_title_over_chapter_doc_title() -> None:
    evidence = format_evidence(
        _adapter(),
        [
            {
                "source": "绍兴市浙东唐诗之路文化资源保护和利用条例_2023-04-24_2023-05-01.pdf",
                "text": "第十条 市、县（市、区）文化旅游主管部门负责编制保护利用规划。",
                "metadata": {
                    "doc_title": "第一章 总则",
                    "source_file": "绍兴市浙东唐诗之路文化资源保护和利用条例_2023-04-24_2023-05-01.pdf",
                    "section": "第一章总则",
                    "article_no": "第十条",
                },
            }
        ],
        "规划职责",
        "score",
    )

    assert "标题：绍兴市浙东唐诗之路文化资源保护和利用条例" in evidence
    assert "标题：第一章 总则" not in evidence
