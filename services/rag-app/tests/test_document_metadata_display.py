from app.services.document_metadata import chunk_plain_display_text


def test_chunk_plain_display_text_prefers_content_over_previous_context_body_marker():
    text = (
        "Document: 聊城市养犬管理条例\n"
        "Previous context: 正文：第三十二条 公安机关应当组织对流浪犬只进行捕捉。\n"
        "Content: 正文：第二十二条 养犬人携犬出户应当遵守下列规定：\n"
        "（一）为犬只佩戴犬牌；\n"
        "Next context: 正文：第二十三条 其他内容"
    )

    display = chunk_plain_display_text(text)

    assert "第二十二条" in display
    assert "佩戴犬牌" in display
    assert "第三十二条" not in display
    assert "第二十三条" not in display
