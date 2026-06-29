from app.utils.scoring import normalize_core_aspect_term, query_core_aspect_terms, query_match_terms


def test_query_core_aspect_terms_keep_dl_f001_core_nouns_only():
    terms = query_core_aspect_terms(
        "\u300a\u6797\u829d\u5e02\u51fa\u79df\u623f\u5b89\u5168\u7ba1\u7406\u6761\u4f8b\u300b"
        "\u9002\u7528\u4e8e\u54ea\u4e9b\u51fa\u79df\u623f\u5b89\u5168\u7ba1\u7406\u6d3b\u52a8\uff1f",
        base_terms=[],
    )

    assert terms == ["\u51fa\u79df\u623f", "\u5b89\u5168\u7ba1\u7406"]
    assert "\u6797\u829d\u5e02" not in terms
    assert "\u54ea\u4e9b" not in terms


def test_normalize_core_aspect_term_drops_low_value_fragments():
    assert normalize_core_aspect_term("\u4e8e\u51fa") == ""
    assert normalize_core_aspect_term("\u51fa\u79df\u623f") == "\u51fa\u79df\u623f"


def test_query_match_terms_preserves_legal_chinese_phrases():
    terms = query_match_terms(
        "《聊城市养犬管理条例》中，养犬人干扰他人正常生活、"
        "放任犬只恐吓或者伤害他人时有哪些禁止性义务和可能处罚？"
    )

    assert "养犬人" in terms
    assert "正常生活" in terms
    assert "放任" in terms
    assert "犬只" in terms
    assert "恐吓" in terms
    assert "伤害" in terms
    assert "处罚" in terms


def test_query_match_terms_keeps_article_anchors_first():
    assert query_match_terms("请解释第二十二条和第38条的处罚", limit=4)[:2] == [
        "第二十二条",
        "第38条",
    ]
