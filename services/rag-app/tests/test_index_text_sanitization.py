from app.services import document_service
from app.services.document_indexing_service import _plain_chunk_text
from app.storage.sqlite import fts_storage_text
from app.utils.text import sanitize_index_text


def test_sanitize_index_text_drops_dirty_block_label_lines():
    text = "第一条 正文\n??????66 块\n第二款 正文"

    cleaned = sanitize_index_text(text)

    assert "??????66" not in cleaned
    assert "第一条 正文" in cleaned
    assert "第二款 正文" in cleaned


def test_sanitize_index_text_drops_dirty_block_prefix():
    text = "??????66 块 正文：聊城市养犬管理条例"

    assert sanitize_index_text(text) == "正文：聊城市养犬管理条例"


def test_sanitize_index_text_keeps_legitimate_block_words():
    text = "建筑废弃物包括废弃砖瓦、混凝土块和建筑余土。"

    assert sanitize_index_text(text) == text


def test_vector_doc_entry_sanitizes_raw_and_vector_text():
    service = document_service.DocumentService.__new__(document_service.DocumentService)
    service.doc_title_profile = lambda filename: {"canonical_title": filename}

    doc = service.vector_doc_entry(
        filename="demo.docx",
        batch_item={
            "article_id": "第一条",
            "raw_text": "第一条 正文\n??????66 块",
            "text": "Document: demo\nContent: 第一条 正文\n??????66 块",
        },
        embedding_value=None,
        base_metadata={},
        chunk_id=0,
        chunk_count=1,
        created_at="2026-06-18T00:00:00",
    )

    assert "??????66" not in doc["text"]
    assert "Document:" not in doc["text"]
    assert "Content:" not in doc["text"]
    assert "第一条 正文" in doc["text"]


def test_vector_doc_entry_includes_sparse_embedding_when_available():
    service = document_service.DocumentService.__new__(document_service.DocumentService)
    service.doc_title_profile = lambda filename: {"canonical_title": filename}

    doc = service.vector_doc_entry(
        filename="demo.docx",
        batch_item={
            "raw_text": "第一条 正文",
            "text": "第一条 正文",
        },
        embedding_value=[0.1, 0.2],
        base_metadata={},
        chunk_id=0,
        chunk_count=1,
        created_at="2026-06-18T00:00:00",
        sparse_embedding_value={12: 0.5, 42: 1.25},
    )

    assert doc["sparse_embedding"] == {12: 0.5, 42: 1.25}


def test_vector_doc_entry_caps_applicable_subjects_for_milvus_array():
    service = document_service.DocumentService.__new__(document_service.DocumentService)
    service.doc_title_profile = lambda filename: {"canonical_title": filename}

    doc = service.vector_doc_entry(
        filename="demo.docx",
        batch_item={
            "raw_text": "第一条 正文",
            "text": "第一条 正文",
            "applicable_subjects": [f"主体{i}" for i in range(12)] + [""],
        },
        embedding_value=None,
        base_metadata={},
        chunk_id=0,
        chunk_count=1,
        created_at="2026-06-18T00:00:00",
    )

    assert doc["applicable_subjects"] == [f"主体{i}" for i in range(10)]
    assert doc["metadata"]["applicable_subjects"] == [f"主体{i}" for i in range(10)]


def test_vector_doc_entry_caps_applicable_subject_bytes_for_milvus_array():
    service = document_service.DocumentService.__new__(document_service.DocumentService)
    service.doc_title_profile = lambda filename: {"canonical_title": filename}
    long_subject = "超长适用主体" * 20

    doc = service.vector_doc_entry(
        filename="demo.docx",
        batch_item={
            "raw_text": "第一条 正文",
            "text": "第一条 正文",
            "applicable_subjects": [long_subject],
        },
        embedding_value=None,
        base_metadata={},
        chunk_id=0,
        chunk_count=1,
        created_at="2026-06-18T00:00:00",
    )

    assert len(doc["applicable_subjects"][0].encode("utf-8")) <= 128
    assert doc["applicable_subjects"] == doc["metadata"]["applicable_subjects"]


def test_fts_storage_text_sanitizes_raw_text_metadata():
    text = fts_storage_text(
        "Document: demo\nContent: 第一条 正文\n??????66 块",
        {"raw_text": "第一条 正文\n??????66 块", "doc_title": "demo"},
    )

    assert "??????66" not in text
    assert "Document:" not in text
    assert "Content:" not in text
    assert "文档：" not in text
    assert "章节：" not in text
    assert "正文：" not in text
    assert "正文:" not in text
    assert "第一条 正文" in text


def test_fts_storage_text_strips_parser_metadata_wrapper():
    text = fts_storage_text(
        "章节路径：第三章 管理规范\n页码：1\n元素类型：paragraph\n正文：第十七条 公安机关应当依法履行管理职责。",
        {},
    )

    assert text == "第十七条 公安机关应当依法履行管理职责。"


def test_plain_chunk_text_strips_parser_metadata_wrapper():
    text = _plain_chunk_text(
        {
            "raw_text": (
                "章节路径：第三章 管理规范\n"
                "页码：1\n"
                "元素类型：paragraph\n"
                "正文：第十七条 公安机关应当依法履行管理职责。"
            )
        }
    )

    assert text == "第十七条 公安机关应当依法履行管理职责。"
