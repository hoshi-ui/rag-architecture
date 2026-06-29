from app.core.evidence.hits import hit_display_text, hit_llm_text
from app.services.document_metadata import chunk_plain_display_text


def test_structured_chunk_display_uses_content_only():
    hit = {
        "entity": {
            "text": "Previous context: 第三十二条 公安机关应当组织捕捉。\n"
            "Content: 第二十二条 养犬人携犬出户应当遵守规定。\n"
            "Next context: 第二十三条 其他内容。",
            "metadata": {
                "previous_context": "第三十二条 公安机关应当组织捕捉。",
                "content": "第二十二条 养犬人携犬出户应当遵守规定。",
                "next_context": "第二十三条 其他内容。",
            },
        }
    }

    assert hit_display_text(hit) == "第二十二条 养犬人携犬出户应当遵守规定。"
    assert hit_llm_text(hit) == (
        "Previous context: 第三十二条 公安机关应当组织捕捉。\n"
        "Content: 第二十二条 养犬人携犬出户应当遵守规定。\n"
        "Next context: 第二十三条 其他内容。"
    )


def test_chunk_plain_display_text_prefers_structured_content_fallback():
    assert (
        chunk_plain_display_text(
            "Previous context: 正文：第三十二条 公安机关应当组织捕捉。\n"
            "Content: 正文：第二十二条 养犬人携犬出户应当遵守规定。\n"
            "Next context: 正文：第二十三条 其他内容。"
        )
        == "正文：第二十二条 养犬人携犬出户应当遵守规定。"
    )
