from types import SimpleNamespace

from app.core.retrieval.base import RetrievalBaseMixin


def _dedupe_docs(docs, limit):
    return list(docs or [])[:limit]


class _Runtime:
    common = SimpleNamespace(
        normalize_filename=lambda value: str(value or "").strip(),
        normalize_query=lambda value: str(value or "").strip(),
        query_match_terms=lambda value: [term for term in str(value or "").split() if term],
        query_semantic_aspects=lambda value, qfilters=None: {"terms": []},
    )
    routing = SimpleNamespace(
        classify_question_type=lambda query: "compare" if "compare" in str(query or "") else "other",
        extract_section_query_targets=lambda query, limit=6: ["penalty"] if "penalty" in str(query or "") else [],
    )
    evidence = SimpleNamespace(
        hit_entity_source=lambda hit: hit["source"],
        hit_score=lambda hit: hit["score"],
        expand_docs_with_neighbor_chunks=lambda docs: docs,
        dedupe_docs=_dedupe_docs,
    )
    source = SimpleNamespace(
        chunks=[],
        get_chunks_for_source=lambda source, limit=None: _Runtime.source.chunks,
    )

    @staticmethod
    def source_chunk_to_hit(source, chunk, score=0.0):
        text = str((chunk or {}).get("raw_text") or (chunk or {}).get("text") or "")
        metadata = dict((chunk or {}).get("metadata") or {})
        metadata.setdefault("section", (chunk or {}).get("section") or "")
        metadata.setdefault("raw_text", text)
        if (chunk or {}).get("chunk_role"):
            metadata["chunk_role"] = (chunk or {}).get("chunk_role")
        return {"entity": {"source": source, "text": text, "metadata": metadata}, "score": score}


class _Adapter(RetrievalBaseMixin):
    runtime = _Runtime()


def test_dense_source_score_map_uses_best_score_per_source():
    scores = _Adapter().dense_source_score_map(
        [
            {"source": "a.pdf", "score": 0.2},
            {"source": "a.pdf", "score": 0.7},
            {"source": "b.pdf", "score": 0.5},
        ]
    )

    assert scores == {"a.pdf": 0.7, "b.pdf": 0.5}


def test_dense_source_score_map_converts_distance_to_similarity():
    scores = _Adapter().dense_source_score_map(
        [
            {"source": "a.pdf", "score": 3.0},
            {"source": "b.pdf", "score": 1.0},
        ],
        score_mode="distance",
    )

    assert scores == {"a.pdf": 0.25, "b.pdf": 0.5}


def test_query_has_compare_intent_passes_classifier_callback():
    assert _Adapter().query_has_compare_intent("compare two regulations") is True


def test_expand_heading_hits_accepts_target_scoped_signature():
    docs = [{"source": "a.pdf", "score": 0.5}]

    assert _Adapter().expand_heading_hits_to_article_hits("penalty", "a.pdf", docs, limit=1) == docs
    assert _Adapter().expand_heading_hits_to_article_hits(docs, limit=1) == docs


def test_expand_heading_hits_rescues_body_when_only_title_hit():
    _Runtime.source.chunks = [
        {"section": "document_title", "raw_text": "Title", "metadata": {"title_hit": True}},
        {"section": "Penalty", "raw_text": "The penalty includes a fine.", "metadata": {"chunk_id": 2}},
        {"section": "General", "raw_text": "Definitions only.", "metadata": {"chunk_id": 3}},
    ]
    title_hit = {
        "entity": {
            "source": "a.pdf",
            "text": "Title",
            "metadata": {"section": "document_title", "title_hit": True},
        },
        "score": 1.0,
    }

    docs = _Adapter().expand_heading_hits_to_article_hits("penalty rules", "a.pdf", [title_hit], limit=2)

    assert docs[0]["entity"]["metadata"]["section"] == "Penalty"
    assert docs[0]["score"] > 0
