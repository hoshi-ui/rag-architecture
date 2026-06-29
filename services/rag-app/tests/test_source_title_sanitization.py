from app.core.source.common import source_display_title


def test_source_display_title_falls_back_when_canonical_title_is_chunk_text():
    title = source_display_title(
        "聊城市养犬管理条例_2020-06-15_2020-09-01.docx",
        doc_get=lambda source: {
            "canonical_title": "??????66 块 正文：聊城市养犬管理条例 下文：章节路径：聊城市养犬管理条例",
        },
        filename_stem=lambda source: "聊城市养犬管理条例",
    )

    assert title == "聊城市养犬管理条例"


def test_source_display_title_falls_back_when_canonical_title_is_chapter_heading():
    title = source_display_title(
        "shaoxing_property.pdf",
        doc_get=lambda source: {"canonical_title": "\u7b2c\u4e00\u7ae0\u603b\u5219"},
        filename_stem=lambda source: "\u7ecd\u5174\u5e02\u7269\u4e1a\u7ba1\u7406\u6761\u4f8b",
    )

    assert title == "\u7ecd\u5174\u5e02\u7269\u4e1a\u7ba1\u7406\u6761\u4f8b"
