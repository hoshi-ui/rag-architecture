from app.core.retrieval.filters import (
    build_milvus_filter,
    configured_article_ids_are_valid,
    normalize_configured_article_ids,
    normalize_article_ids,
    sqlite_article_filter_sql,
    target_article_ids,
)


def test_target_article_ids_normalizes_arrays_and_query_mentions():
    assert target_article_ids(
        {"target_articles": ["\u4e8c\u5341\u4e8c\u6761", "\u7b2c\u4e09\u5341\u516b\u6761"]},
        "\u7b2c\u4e8c\u5341\u4e8c\u6761\u548c\u7b2c\u4e09\u5341\u516b\u6761\u5206\u522b\u89c4\u5b9a\u4ec0\u4e48\uff1f",
    ) == ["\u7b2c\u4e8c\u5341\u4e8c\u6761", "\u7b2c\u4e09\u5341\u516b\u6761"]


def test_target_article_ids_rejects_document_titles_and_long_query_spans():
    query = (
        "\u5bf9\u6bd4\u6df1\u5733\u5efa\u7b51\u5e02\u573a\u4e25\u91cd\u8fdd\u6cd5\u884c\u4e3a\u5904\u7406\u89c4\u5b9a\u3001"
        "\u6df1\u5733\u5efa\u8bbe\u5de5\u7a0b\u8d28\u91cf\u7ba1\u7406\u6761\u4f8b\u3001"
        "\u6df1\u5733\u5efa\u7b51\u5e9f\u5f03\u7269\u51cf\u6392\u4e0e\u5229\u7528\u6761\u4f8b"
        "\u4e2d\u4e0e\u5efa\u8bbe\u5355\u4f4d\u6216\u5de5\u7a0b\u53c2\u4e0e\u5355\u4f4d\u76f8\u5173\u7684\u4e49\u52a1"
    )

    assert target_article_ids(
        {
            "target_articles": [
                "\u5bf9\u6bd4\u6df1\u5733\u5efa\u7b51\u5e02\u573a\u4e25\u91cd\u8fdd\u6cd5\u884c\u4e3a\u5904\u7406\u89c4\u5b9a",
                "\u6df1\u5733\u5efa\u8bbe\u5de5\u7a0b\u8d28\u91cf\u7ba1\u7406\u6761\u4f8b",
                "\u6df1\u5733\u5efa\u7b51\u5e9f\u5f03\u7269\u51cf\u6392\u4e0e\u5229\u7528\u6761\u4f8b\u4e2d\u4e0e\u5efa\u8bbe\u5355\u4f4d\u76f8\u5173\u7684\u4e49\u52a1",
                "\u0531\u0576\u0585",
            ]
        },
        query,
    ) == []


def test_configured_article_ids_require_exact_short_article_labels():
    assert normalize_configured_article_ids("\u7b2c\u5341\u4e94\u6761") == ["\u7b2c\u5341\u4e94\u6761"]
    assert normalize_configured_article_ids("\u6df1\u5733\u5efa\u8bbe\u5de5\u7a0b\u8d28\u91cf\u7ba1\u7406\u6761\u4f8b\u7b2c\u5341\u4e94\u6761") == []
    assert configured_article_ids_are_valid("\u7b2c\u5341\u4e94\u6761")
    assert not configured_article_ids_are_valid("\u7b2c\u5341\u4e94\u6761\u548c\u7b2c\u5341\u4e03\u6761")


def test_skip_article_id_filter_uses_text_similarity_only():
    assert target_article_ids(
        {"_skip_article_id_filter": True, "article_id": "\u7b2c\u5341\u4e94\u6761"},
        "\u7b2c\u5341\u4e94\u6761\u7684\u4e49\u52a1",
    ) == []


def test_query_article_ids_extracts_only_explicit_article_numbers_from_long_text():
    assert target_article_ids(
        {},
        "\u5bf9\u6bd4\u6df1\u5733\u5efa\u8bbe\u5de5\u7a0b\u8d28\u91cf\u7ba1\u7406\u6761\u4f8b\u7b2c\u5341\u4e94\u6761\u548c\u7b2c\u5341\u4e03\u6761\u7684\u4e49\u52a1",
    ) == ["\u7b2c\u5341\u4e94\u6761", "\u7b2c\u5341\u4e03\u6761"]


def test_configured_article_string_can_contain_multiple_explicit_articles():
    assert normalize_article_ids("\u7b2c\u5341\u4e94\u6761\u548c\u7b2c\u5341\u4e03\u6761") == [
        "\u7b2c\u5341\u4e94\u6761",
        "\u7b2c\u5341\u4e03\u6761",
    ]


def test_build_milvus_filter_uses_in_for_multiple_articles():
    expr = build_milvus_filter(
        sources=["liaocheng.docx"],
        article_ids=["\u4e8c\u5341\u4e8c\u6761", "\u4e09\u5341\u516b\u6761"],
    )

    assert 'source == "liaocheng.docx"' in expr
    assert 'article_id in ["\u7b2c\u4e8c\u5341\u4e8c\u6761", "\u7b2c\u4e09\u5341\u516b\u6761"]' in expr
    assert "article_id ==" not in expr


def test_milvus_filter_omits_article_clause_for_invalid_article_values():
    expr = build_milvus_filter(
        sources=["demo.pdf"],
        article_ids=["\u6df1\u5733\u5efa\u8bbe\u5de5\u7a0b\u8d28\u91cf\u7ba1\u7406\u6761\u4f8b", "\u0531\u0576\u0585"],
    )

    assert expr == 'source == "demo.pdf"'


def test_milvus_filter_omits_article_clause_for_long_parsed_article_value():
    expr = build_milvus_filter(
        sources=["demo.pdf"],
        article_ids=["\u6df1\u5733\u5efa\u8bbe\u5de5\u7a0b\u8d28\u91cf\u7ba1\u7406\u6761\u4f8b\u7b2c\u5341\u4e94\u6761"],
    )

    assert expr == 'source == "demo.pdf"'


def test_sqlite_article_filter_uses_in_placeholders_for_arrays():
    clause, params = sqlite_article_filter_sql(["\u4e8c\u5341\u4e8c\u6761", "\u7b2c\u4e09\u5341\u516b\u6761"])

    assert "json_extract(m.metadata, '$.article_id') IN (?, ?)" in clause
    assert "json_extract(m.metadata, '$.article_no') IN (?, ?)" in clause
    assert params == [
        "\u7b2c\u4e8c\u5341\u4e8c\u6761",
        "\u7b2c\u4e09\u5341\u516b\u6761",
        "\u7b2c\u4e8c\u5341\u4e8c\u6761",
        "\u7b2c\u4e09\u5341\u516b\u6761",
    ]


def test_sqlite_article_filter_omits_invalid_article_values():
    clause, params = sqlite_article_filter_sql(["\u6df1\u5733\u5efa\u8bbe\u5de5\u7a0b\u8d28\u91cf\u7ba1\u7406\u6761\u4f8b"])

    assert clause == ""
    assert params == []


def test_sqlite_article_filter_omits_long_parsed_article_value():
    clause, params = sqlite_article_filter_sql([
        "\u6df1\u5733\u5efa\u8bbe\u5de5\u7a0b\u8d28\u91cf\u7ba1\u7406\u6761\u4f8b\u7b2c\u5341\u4e94\u6761"
    ])

    assert clause == ""
    assert params == []


def test_normalize_article_ids_accepts_comma_separated_values():
    assert normalize_article_ids("\u4e8c\u5341\u4e8c\u6761, \u4e09\u5341\u516b\u6761") == [
        "\u7b2c\u4e8c\u5341\u4e8c\u6761",
        "\u7b2c\u4e09\u5341\u516b\u6761",
    ]


def test_normalize_article_ids_repairs_gb18030_mojibake_query():
    article = "\u7b2c\u5341\u4e03\u6761"
    mojibake = article.encode("utf-8").decode("gb18030", errors="replace")

    assert normalize_article_ids(mojibake) == [article]
