from app.core.retrieval.lexical import _mark_article_anchor_hits, article_anchor_terms


def test_article_anchor_terms_extracts_unique_article_labels():
    assert article_anchor_terms("第二十二条禁止什么？第三十八条如何处罚？第二十二条还涉及什么？") == [
        "第二十二条",
        "第三十八条",
    ]


def test_article_anchor_terms_accepts_numeric_labels():
    assert article_anchor_terms("请解释第22条和第38条") == ["第22条", "第38条"]


def test_mark_article_anchor_hits_adds_metadata_and_score_boost():
    hits = [
        {
            "entity": {
                "source": "聊城市养犬管理条例.docx",
                "text": "第二十二条 养犬应当遵守下列规定",
                "metadata": {},
            },
            "score": 0.1,
        }
    ]

    marked = _mark_article_anchor_hits(hits, "第二十二条")

    assert marked[0]["score"] == 8.0
    assert marked[0]["entity"]["metadata"]["article_anchor_hit"] is True
    assert marked[0]["entity"]["metadata"]["lexical_signal"] == "article_anchor"
