import json

from app.services.document_metadata import MILVUS_METADATA_JSON_LIMIT, milvus_safe_metadata


def test_milvus_safe_metadata_caps_late_serialized_chunk_fields():
    metadata = {
        "source": "demo.pdf",
        "doc_title": "demo",
        "section": "正文",
        "raw_text": "x" * 100000,
        "content": "第二十二条 " + ("养犬人应当遵守规定。" * 10000),
        "previous_context": "上文" * 10000,
        "next_context": "下文" * 10000,
        "payload": {"ocr_meta": {"huge": "y" * 100000}, "layout": {"line_count": 2}},
    }

    safe = milvus_safe_metadata(metadata)
    encoded = json.dumps(safe, ensure_ascii=False, default=str)

    assert "raw_text" not in safe
    assert len(encoded) <= MILVUS_METADATA_JSON_LIMIT
    assert safe["content"].startswith("第二十二条")
    assert len(safe["content"]) < len(metadata["content"])
