from evals.runners.score_documents_metrics import _score_answer_relevance, _score_retrieval


def test_score_answer_relevance_matches_scope_labels_with_core_evidence():
    case = {
        "expected_aspects": [
            "\u9002\u7528\u5730\u57df",
            "\u9002\u7528\u6d3b\u52a8",
            "\u51fa\u79df\u623f\u5b9a\u4e49",
            "\u6392\u9664\u6c11\u5bbf\u548c\u65c5\u9986\u4e1a\u5ba2\u623f",
        ],
        "minimum_required_aspect_count": 3,
    }
    answer = (
        "\u7b2c\u4e8c\u6761\uff1a\u672c\u5e02\u884c\u653f\u533a\u57df\u5185\u51fa\u79df\u623f"
        "\u7684\u79df\u8d41\u3001\u6cbb\u5b89\u3001\u6d88\u9632\u7b49\u5b89\u5168\u7ba1"
        "\u7406\u53ca\u5176\u76d1\u7763\u6d3b\u52a8\uff0c\u9002\u7528\u672c\u6761\u4f8b"
        "\uff1b\u672c\u6761\u4f8b\u6240\u79f0\u51fa\u79df\u623f\uff0c\u662f\u6307\u51fa"
        "\u79df\u4eba\u5c06\u5176\u623f\u5c4b\u4f9d\u7167\u7ea6\u5b9a\u4ea4\u4ed8\u627f"
        "\u79df\u4eba\u4f7f\u7528\u3001\u6536\u76ca\uff1b\u4e0d\u5305\u62ec\u6c11\u5bbf"
        "\u3001\u65c5\u9986\u4e1a\u5ba2\u623f\u3002"
    )

    scored = _score_answer_relevance(case, {"answer": answer})

    assert scored["answer_relevance"] == 1.0
    assert scored["aspect_hit_count"] == 4
    assert scored["answer_relevance_pass"] is True


def test_score_retrieval_splits_source_clause_content_and_metadata_coverage():
    case = {
        "expected_evidence": [
            {
                "source": "demo.docx",
                "clause": "\u7b2c\u5341\u4e03\u6761",
                "text": "\u517b\u72ac\u4eba\u4e0d\u5f97\u8650\u5f85\u72ac\u53ea",
            }
        ]
    }
    result = {
        "retrieved_documents": {
            "hybrid_rerank": [
                {
                    "source": "demo.docx",
                    "text": "\u517b\u72ac\u4eba\u4e0d\u5f97\u8650\u5f85\u72ac\u53ea",
                    "metadata": {
                        "doc_id": "demo.docx",
                    },
                },
                {
                    "source": "other.docx",
                    "text": "\u65e0\u5173\u5185\u5bb9",
                    "metadata": {
                        "doc_id": "other.docx",
                        "article_no": "\u7b2c\u4e00\u6761",
                    },
                },
            ]
        }
    }

    scored = _score_retrieval(
        case,
        result,
        retrieval_key="hybrid_rerank",
        top_k=5,
        similarity_threshold=0.72,
    )

    assert scored["source_hit_rate"] == 1.0
    assert scored["content_hit_rate"] == 1.0
    assert scored["clause_id_hit_rate"] == 0.0
    assert scored["metadata_coverage_rate"] == 0.5
    assert scored["ref_diagnostics"][0]["source_hit"] is True
    assert scored["ref_diagnostics"][0]["clause_id_hit"] is False
