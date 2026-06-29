from app.core.evidence.context import EvidenceAdapter
from app.core.evidence.format import build_sources
from app.core.evidence.selection import expand_docs_with_full_article_chunks, merge_compare_source_doc_groups


def _adapter(chunks=None):
    return EvidenceAdapter(
        normalize_filename_for_match=lambda value: str(value or ""),
        normalize_query=lambda value: str(value or ""),
        hit_display_text=lambda hit: (hit.get("entity") or {}).get("metadata", {}).get("raw_text") or (hit.get("entity") or {}).get("text") or hit.get("text") or "",
        hit_llm_text=lambda hit: (hit.get("entity") or {}).get("text") or hit.get("text") or "",
        hit_metadata=lambda hit: (hit.get("entity") or {}).get("metadata") or hit.get("metadata") or {},
        hit_entity_source=lambda hit: (hit.get("entity") or {}).get("source") or hit.get("source") or "",
        hit_score=lambda hit: float(hit.get("score") or 0.0),
        hit_chunk_range=lambda hit: str(hit.get("chunk_range") or ""),
        source_display_title=lambda value: str(value or ""),
        build_excerpt=lambda text, query, limit: str(text or "")[:limit],
        get_chunks_for_source=lambda source, doc_version=None: list((chunks or {}).get((source, doc_version), [])),
        doc_section_name=lambda hit: str((hit.get("metadata") or {}).get("section") or ""),
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


def test_build_sources_exposes_article_metadata_for_evaluators():
    article = "\u7b2c\u5341\u4e03\u6761"
    docs = [
        {
            "source": "demo.docx",
            "score": 0.12,
            "chunk_range": "10",
            "text": "\u517b\u72ac\u4eba\u4e0d\u5f97\u8650\u5f85\u3001\u9057\u5f03\u9972\u517b\u7684\u72ac\u53ea\u3002",
            "metadata": {
                "article_id": article,
                "article_no": article,
                "section": "\u517b\u72ac\u884c\u4e3a\u89c4\u8303",
                "chunk_id": 10,
            },
        }
    ]

    source = build_sources(_adapter(), docs, "\u8650\u5f85\u72ac\u53ea", "distance")[0]

    assert source["article_id"] == article
    assert source["article_no"] == article
    assert source["clause_id"] == article
    assert source["doc_id"] == "demo.docx"
    assert source["metadata_available"] is True
    assert source["clause"] == article
    assert source["metadata"]["article_id"] == article
    assert source["metadata"]["article_no"] == article
    assert source["metadata"]["clause_id"] == article
    assert source["metadata"]["doc_id"] == "demo.docx"
    assert source["metadata"]["metadata_available"] is True
    assert source["metadata"]["clause_metadata"]["doc_id"] == "demo.docx"
    assert source["metadata"]["clause_metadata"]["article_no"] == article
    assert source["metadata"]["clause_metadata"]["source_file"] == "demo.docx"
    assert source["metadata"]["clause"] == article
    assert source["metadata"]["section"] == "\u517b\u72ac\u884c\u4e3a\u89c4\u8303"


def test_expand_docs_with_full_article_chunks_collapses_same_article_segments():
    source = "demo.pdf"
    article = "\u7b2c\u4e09\u5341\u516b\u6761"
    chunks = {
        (source, 1): [
            {
                "text": "\u7b2c\u4e09\u5341\u516b\u6761 \u7b2c\u4e00\u6bb5",
                "raw_text": "\u7b2c\u4e09\u5341\u516b\u6761 \u7b2c\u4e00\u6bb5",
                "chunk_id": 30,
                "metadata": {"chunk_id": 30, "doc_version": 1, "article_id": article, "section": "\u6cd5\u5f8b\u8d23\u4efb"},
            },
            {
                "text": "\u7b2c\u4e8c\u6bb5",
                "raw_text": "\u7b2c\u4e8c\u6bb5",
                "chunk_id": 31,
                "metadata": {"chunk_id": 31, "doc_version": 1, "article_id": article, "section": "\u6cd5\u5f8b\u8d23\u4efb"},
            },
            {
                "text": "\u7b2c\u4e09\u6bb5",
                "raw_text": "\u7b2c\u4e09\u6bb5",
                "chunk_id": 32,
                "metadata": {"chunk_id": 32, "doc_version": 1, "article_id": article, "section": "\u6cd5\u5f8b\u8d23\u4efb"},
            },
        ]
    }
    docs = [
        {
            "entity": {
                "source": source,
                "text": "\u7b2c\u4e8c\u6bb5",
                "metadata": {"chunk_id": 31, "doc_version": 1, "article_id": article},
            },
            "score": 0.9,
        },
        {
            "entity": {
                "source": source,
                "text": "\u7b2c\u4e09\u6bb5",
                "metadata": {"chunk_id": 32, "doc_version": 1, "article_id": article},
            },
            "score": 0.8,
        },
    ]

    expanded = expand_docs_with_full_article_chunks(_adapter(chunks), docs)

    assert len(expanded) == 1
    metadata = expanded[0]["entity"]["metadata"]
    assert metadata["article_id"] == article
    assert metadata["chunk_id_start"] == 30
    assert metadata["chunk_id_end"] == 32
    assert metadata["full_article_expanded"] is True
    assert metadata["full_article_chunk_count"] == 3
    assert expanded[0]["entity"]["text"] == "\u7b2c\u4e09\u5341\u516b\u6761 \u7b2c\u4e00\u6bb5\n\u7b2c\u4e8c\u6bb5\n\u7b2c\u4e09\u6bb5"

    source_item = build_sources(_adapter(chunks), expanded, "\u7b2c\u4e09\u5341\u516b\u6761", "distance")[0]
    assert source_item["metadata"]["chunk_id_start"] == 30
    assert source_item["metadata"]["chunk_id_end"] == 32
    assert source_item["metadata"]["full_article_expanded"] is True
    assert source_item["metadata"]["full_article_chunk_count"] == 3


def test_expand_docs_with_full_article_chunks_preserves_unarticled_docs():
    doc = {
        "entity": {
            "source": "demo.pdf",
            "text": "\u9644\u5219",
            "metadata": {"chunk_id": 9, "doc_version": 1, "chunk_role": "section_heading"},
        },
        "score": 0.5,
    }

    assert expand_docs_with_full_article_chunks(_adapter(), [doc]) == [doc]


def test_expand_docs_with_full_article_chunks_orders_by_reading_order():
    source = "demo.pdf"
    article = "\u7b2c\u5341\u6761"
    chunks = {
        (source, 1): [
            {
                "text": "\u7b2c\u4e8c\u6bb5",
                "raw_text": "\u7b2c\u4e8c\u6bb5",
                "chunk_id": 1,
                "metadata": {"chunk_id": 1, "reading_order": 20, "doc_version": 1, "article_id": article},
            },
            {
                "text": "\u7b2c\u4e00\u6bb5",
                "raw_text": "\u7b2c\u4e00\u6bb5",
                "chunk_id": 2,
                "metadata": {"chunk_id": 2, "reading_order": 10, "doc_version": 1, "article_id": article},
            },
        ]
    }
    docs = [
        {
            "entity": {
                "source": source,
                "text": "\u7b2c\u4e8c\u6bb5",
                "metadata": {"chunk_id": 1, "doc_version": 1, "article_id": article},
            },
            "score": 0.9,
        }
    ]

    expanded = expand_docs_with_full_article_chunks(_adapter(chunks), docs)

    assert expanded[0]["entity"]["text"] == "\u7b2c\u4e00\u6bb5\n\u7b2c\u4e8c\u6bb5"
    assert expanded[0]["entity"]["metadata"]["reading_order_start"] == 10
    assert expanded[0]["entity"]["metadata"]["reading_order_end"] == 20


def test_merge_compare_source_doc_groups_interleaves_sources():
    def hit(source, text):
        return {"entity": {"source": source, "text": text, "metadata": {}}, "score": 1.0}

    merged = merge_compare_source_doc_groups(
        _adapter(),
        [
            {"source": "a.pdf", "docs": [hit("a.pdf", "a1"), hit("a.pdf", "a2")]},
            {"source": "b.pdf", "docs": [hit("b.pdf", "b1"), hit("b.pdf", "b2")]},
        ],
        per_source_limit=2,
    )

    assert [(item["entity"]["source"], item["entity"]["text"]) for item in merged] == [
        ("a.pdf", "a1"),
        ("b.pdf", "b1"),
        ("a.pdf", "a2"),
        ("b.pdf", "b2"),
    ]
