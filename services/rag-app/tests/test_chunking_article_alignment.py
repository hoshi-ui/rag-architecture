from app.documents import chunking
from app.services import document_service


def _runtime():
    def serialize_ir_element(element):
        raw = element.get("text_raw") or element.get("text") or ""
        return {
            "serialized_text": raw,
            "raw_text": raw,
            "normalized_text": raw,
            "fts_text": raw,
        }

    return chunking.DocumentChunkingAdapter(
        normalize_ir_text=lambda text: str(text or ""),
        chapter_heading_title=lambda text: None,
        clause_heading_label=lambda text: chunking.extract_article_id(text),
        serialize_ir_element=serialize_ir_element,
        doc_title_profile=lambda filename: {"canonical_title": filename},
        filename_stem=lambda filename: filename.rsplit(".", 1)[0],
        split_text=lambda text, chunk_size, overlap: [text],
        chunk_prev_context_chars=80,
        chunk_next_context_chars=80,
    )


def test_contextualize_chunk_items_does_not_attach_cross_article_context():
    runtime = _runtime()
    items = [
        {
            "article_id": "第二十二条",
            "article_no": "第二十二条",
            "raw_text": "第二十二条 养犬人不得携犬进入禁止区域。",
            "text": "第二十二条 养犬人不得携犬进入禁止区域。",
        },
        {
            "article_id": "第二十三条",
            "article_no": "第二十三条",
            "raw_text": "第二十三条 公安机关应当建立管理档案。",
            "text": "第二十三条 公安机关应当建立管理档案。",
        },
    ]

    contextualized = chunking.contextualize_chunk_items(runtime, "demo.docx", items)

    assert "Next context:" not in contextualized[0]["text"]
    assert "Previous context:" not in contextualized[1]["text"]


def test_contextualize_chunk_items_fills_and_carries_article_id():
    runtime = _runtime()
    article = "\u7b2c\u5341\u4e03\u6761"
    items = [
        {
            "raw_text": f"{article} \u517b\u72ac\u4eba\u4e0d\u5f97\u8650\u5f85\u3001\u9057\u5f03\u9972\u517b\u7684\u72ac\u53ea\u3002",
            "text": f"{article} \u517b\u72ac\u4eba\u4e0d\u5f97\u8650\u5f85\u3001\u9057\u5f03\u9972\u517b\u7684\u72ac\u53ea\u3002",
        },
        {
            "raw_text": "\uff08\u516d\uff09\u8650\u5f85\u3001\u9057\u5f03\u72ac\u53ea\u7684\uff0c\u8d23\u4ee4\u6539\u6b63\u3002",
            "text": "\uff08\u516d\uff09\u8650\u5f85\u3001\u9057\u5f03\u72ac\u53ea\u7684\uff0c\u8d23\u4ee4\u6539\u6b63\u3002",
        },
    ]

    contextualized = chunking.contextualize_chunk_items(runtime, "demo.docx", items)

    assert contextualized[0]["article_id"] == article
    assert contextualized[0]["article_no"] == article
    assert contextualized[1]["article_id"] == article
    assert contextualized[1]["article_no"] == article


def test_apply_chunk_article_id_keeps_top_level_and_metadata_fields_aligned():
    article = "\u7b2c\u5341\u4e03\u6761"
    item = {
        "raw_text": "\uff08\u516d\uff09\u8650\u5f85\u3001\u9057\u5f03\u72ac\u53ea\u7684\uff0c\u8d23\u4ee4\u6539\u6b63\u3002",
        "metadata": {},
    }

    applied = chunking.apply_chunk_article_id(item, item["raw_text"], article)

    assert applied == article
    assert chunking.chunk_article_id(item) == article
    assert item["article_id"] == article
    assert item["article_no"] == article
    assert item["metadata"]["article_id"] == article
    assert item["metadata"]["article_no"] == article


def test_apply_chunk_article_id_falls_back_to_section_heading():
    article = "\u7b2c\u5341\u4e03\u6761"
    item = {
        "section": f"{article} \u517b\u72ac\u4eba\u4e0d\u5f97\u8650\u5f85\u9972\u517b\u7684\u72ac\u53ea",
        "raw_text": "\uff08\u516d\uff09\u8650\u5f85\u3001\u9057\u5f03\u9972\u517b\u7684\u72ac\u53ea\u3002",
        "metadata": {},
    }

    applied = chunking.apply_chunk_article_id(item, item["raw_text"])

    assert applied == article
    assert item["article_id"] == article
    assert item["article_no"] == article
    assert item["metadata"]["article_id"] == article
    assert item["metadata"]["article_no"] == article


def test_extract_article_id_repairs_gb18030_mojibake_heading():
    article = "\u7b2c\u5341\u4e03\u6761"
    mojibake = article.encode("utf-8").decode("gb18030", errors="replace")

    assert chunking.extract_article_id(mojibake) == article


def test_extract_article_id_normalizes_pdf_spaced_article_heading():
    assert chunking.extract_article_id("\u7b2c \u4e00 \u6761 \u4e3a\u4e86\u52a0\u5f3a\u7ba1\u7406") == "\u7b2c\u4e00\u6761"
    assert chunking.extract_article_id("\u7b2c\uff11\uff12\u6761 \u5e94\u5f53\u4f9d\u6cd5\u5c65\u804c") == "\u7b2c12\u6761"


def test_extract_leading_article_id_ignores_inline_article_references():
    text = "\u8fdd\u53cd\u672c\u6761\u4f8b\n\u7b2c\u5341\u4e00\u6761\u7b2c\u4e00\u6b3e\u89c4\u5b9a\u7684\uff0c\u8d23\u4ee4\u6539\u6b63\u3002"

    assert chunking.extract_article_id(text) == "\u7b2c\u5341\u4e00\u6761"
    assert chunking.extract_leading_article_id(text) == ""


def test_document_service_clause_heading_label_accepts_pdf_spaced_article_heading():
    service = document_service.DocumentService.__new__(document_service.DocumentService)

    assert service.clause_heading_label("\u7b2c \u4e00 \u6761 \u4e3a\u4e86\u52a0\u5f3a\u7ba1\u7406") == "\u7b2c\u4e00\u6761"


def test_vector_doc_entry_prefers_current_body_article_id():
    service = document_service.DocumentService.__new__(document_service.DocumentService)
    service.doc_title_profile = lambda filename: {"canonical_title": filename}

    doc = service.vector_doc_entry(
        filename="demo.docx",
        batch_item={
            "article_id": "第二十二条",
            "article_no": "第二十二条",
            "raw_text": "第二十三条 公安机关应当建立管理档案。",
            "text": "Content: 第二十三条 公安机关应当建立管理档案。",
        },
        embedding_value=None,
        base_metadata={},
        chunk_id=0,
        chunk_count=1,
        created_at="2026-06-18T00:00:00",
    )

    assert doc["article_id"] == "第二十三条"
    assert doc["metadata"]["article_id"] == "第二十三条"
    assert doc["metadata"]["article_no"] == "第二十三条"


def test_vector_doc_entry_does_not_promote_inline_reference_to_article_id():
    service = document_service.DocumentService.__new__(document_service.DocumentService)
    service.doc_title_profile = lambda filename: {"canonical_title": filename}

    doc = service.vector_doc_entry(
        filename="demo.pdf",
        batch_item={
            "section": "\u7b2c\u516d\u7ae0 \u6cd5\u5f8b\u8d23\u4efb",
            "raw_text": "\u8fdd\u53cd\u672c\u6761\u4f8b\n\u7b2c\u5341\u4e00\u6761\u7b2c\u4e00\u6b3e\u89c4\u5b9a\u7684\uff0c\u8d23\u4ee4\u6539\u6b63\u3002",
            "text": "\u8fdd\u53cd\u672c\u6761\u4f8b\n\u7b2c\u5341\u4e00\u6761\u7b2c\u4e00\u6b3e\u89c4\u5b9a\u7684\uff0c\u8d23\u4ee4\u6539\u6b63\u3002",
        },
        embedding_value=None,
        base_metadata={},
        chunk_id=0,
        chunk_count=1,
        created_at="2026-06-18T00:00:00",
    )

    assert doc["article_id"] == ""
    assert doc["metadata"]["article_id"] == ""
    assert doc["metadata"]["article_no"] == ""


def test_add_chunk_sql_fills_article_fields_before_sqlite_write():
    article = "\u7b2c\u5341\u4e03\u6761"
    captured = {}

    class FakeLexStore:
        def add_chunk(
            self,
            source,
            text,
            section,
            metadata,
            chunk_id,
            after_meta_insert=None,
            after_fts_insert=None,
        ):
            captured["metadata"] = metadata

    service = document_service.DocumentService.__new__(document_service.DocumentService)
    service._sqlite_write_lock = None
    service._lex_store = FakeLexStore()
    service.crash_inject = lambda name: None

    service.add_chunk_sql(
        "demo.docx",
        "\uff08\u516d\uff09\u8650\u5f85\u3001\u9057\u5f03\u72ac\u53ea\u7684\uff0c\u8d23\u4ee4\u6539\u6b63\u3002",
        f"{article} \u517b\u72ac\u4eba\u4e0d\u5f97\u8650\u5f85\u9972\u517b\u7684\u72ac\u53ea",
        {},
        17,
    )

    assert captured["metadata"]["article_id"] == article
    assert captured["metadata"]["article_no"] == article
    assert captured["metadata"]["chunk_id"] == 17


def test_add_chunk_sql_prefers_article_from_chunk_text():
    text_article = "\u7b2c\u5341\u4e03\u6761"
    stale_article = "\u7b2c\u4e09\u5341\u6761"
    captured = {}

    class FakeLexStore:
        def add_chunk(
            self,
            source,
            text,
            section,
            metadata,
            chunk_id,
            after_meta_insert=None,
            after_fts_insert=None,
        ):
            captured["metadata"] = metadata

    service = document_service.DocumentService.__new__(document_service.DocumentService)
    service._sqlite_write_lock = None
    service._lex_store = FakeLexStore()
    service.crash_inject = lambda name: None

    service.add_chunk_sql(
        "demo.docx",
        f"{text_article} \u517b\u72ac\u4eba\u4e0d\u5f97\u8650\u5f85\u9972\u517b\u7684\u72ac\u53ea\u3002",
        "\u517b\u72ac\u884c\u4e3a\u89c4\u8303",
        {"article_id": stale_article, "article_no": stale_article},
        17,
    )

    assert captured["metadata"]["article_id"] == text_article
    assert captured["metadata"]["article_no"] == text_article


def test_heading_style_article_carries_to_following_list_items():
    runtime = _runtime()
    article = "\u7b2c\u5341\u4e03\u6761"
    document_ir = {
        "source": "demo.docx",
        "elements": [
            {
                "element_id": "e1",
                "element_type": "heading",
                "text_raw": f"{article} \u517b\u72ac\u5e94\u5f53\u9075\u5b88\u4e0b\u5217\u89c4\u5b9a\uff1a",
                "section_path": ["\u517b\u72ac\u884c\u4e3a\u89c4\u8303"],
                "reading_order": 1,
            },
            {
                "element_id": "e2",
                "element_type": "list_item",
                "text_raw": "\uff08\u516d\uff09\u4e0d\u5f97\u8650\u5f85\u3001\u9057\u5f03\u9972\u517b\u7684\u72ac\u53ea\uff1b",
                "section_path": ["\u517b\u72ac\u884c\u4e3a\u89c4\u8303"],
                "reading_order": 2,
            },
        ],
    }

    items = chunking.document_ir_to_structured_items(runtime, document_ir, chunk_size=300, overlap=30)

    assert items
    assert items[0]["article_id"] == article
    assert items[0]["article_no"] == article
    assert "\u4e0d\u5f97\u8650\u5f85" in items[0]["raw_text"]


def test_chunk_plans_carry_article_id_to_following_body_chunk():
    runtime = _runtime()
    article = "\u7b2c\u5341\u4e03\u6761"
    plans = [
        {
            "section": "\u517b\u72ac\u884c\u4e3a\u89c4\u8303",
            "clause_label": article,
            "article_no": article,
            "article_id": article,
            "text": f"{article} \u517b\u72ac\u5e94\u5f53\u9075\u5b88\u4e0b\u5217\u89c4\u5b9a\uff1a",
            "raw_text": f"{article} \u517b\u72ac\u5e94\u5f53\u9075\u5b88\u4e0b\u5217\u89c4\u5b9a\uff1a",
            "chunk_role": "article",
        },
        {
            "section": "\u517b\u72ac\u884c\u4e3a\u89c4\u8303",
            "text": "\uff08\u516d\uff09\u4e0d\u5f97\u8650\u5f85\u3001\u9057\u5f03\u9972\u517b\u7684\u72ac\u53ea\uff1b",
            "raw_text": "\uff08\u516d\uff09\u4e0d\u5f97\u8650\u5f85\u3001\u9057\u5f03\u9972\u517b\u7684\u72ac\u53ea\uff1b",
            "chunk_role": "body",
        },
    ]

    items = chunking.chunk_plans_to_items(runtime, plans)

    assert items[1]["article_id"] == article
    assert items[1]["article_no"] == article


def test_chunk_plans_do_not_carry_article_id_to_section_heading():
    runtime = _runtime()
    article = "\u7b2c\u516b\u6761"
    plans = [
        {
            "section": "\u603b\u5219",
            "article_no": article,
            "article_id": article,
            "text": f"{article} \u4efb\u4f55\u5355\u4f4d\u548c\u4e2a\u4eba\u6709\u6743\u4e3e\u62a5\u8fdd\u53cd\u672c\u6761\u4f8b\u7684\u884c\u4e3a\u3002",
            "raw_text": f"{article} \u4efb\u4f55\u5355\u4f4d\u548c\u4e2a\u4eba\u6709\u6743\u4e3e\u62a5\u8fdd\u53cd\u672c\u6761\u4f8b\u7684\u884c\u4e3a\u3002",
            "chunk_role": "article",
        },
        {
            "section": "\u603b\u5219",
            "text": "\u517b\u72ac\u533a\u5212\u7ba1\u7406\u3001\u514d\u75ab\u4e0e\u767b\u8bb0",
            "raw_text": "\u517b\u72ac\u533a\u5212\u7ba1\u7406\u3001\u514d\u75ab\u4e0e\u767b\u8bb0",
            "chunk_role": "section_heading",
            "unit_kind": "section_heading",
            "element_type": "heading",
        },
        {
            "section": "\u517b\u72ac\u533a\u5212\u7ba1\u7406\u3001\u514d\u75ab\u4e0e\u767b\u8bb0",
            "text": "\u7b2c\u4e5d\u6761  \u517b\u72ac\u6309\u7167\u91cd\u70b9\u7ba1\u7406\u533a\u548c\u4e00\u822c\u7ba1\u7406\u533a\u5b9e\u884c\u5206\u533a\u57df\u7ba1\u7406\u3002",
            "raw_text": "\u7b2c\u4e5d\u6761  \u517b\u72ac\u6309\u7167\u91cd\u70b9\u7ba1\u7406\u533a\u548c\u4e00\u822c\u7ba1\u7406\u533a\u5b9e\u884c\u5206\u533a\u57df\u7ba1\u7406\u3002",
            "chunk_role": "article",
        },
    ]

    items = chunking.chunk_plans_to_items(runtime, plans)

    assert items[1].get("article_id") == ""
    assert items[1].get("article_no") == ""
    assert items[2]["article_id"] == "\u7b2c\u4e5d\u6761"


def test_contextualize_does_not_carry_article_id_to_section_heading():
    runtime = _runtime()
    article = "\u7b2c\u516b\u6761"
    items = [
        {
            "raw_text": f"{article} \u4efb\u4f55\u5355\u4f4d\u548c\u4e2a\u4eba\u6709\u6743\u4e3e\u62a5\u8fdd\u53cd\u672c\u6761\u4f8b\u7684\u884c\u4e3a\u3002",
            "text": f"{article} \u4efb\u4f55\u5355\u4f4d\u548c\u4e2a\u4eba\u6709\u6743\u4e3e\u62a5\u8fdd\u53cd\u672c\u6761\u4f8b\u7684\u884c\u4e3a\u3002",
            "chunk_role": "article",
        },
        {
            "raw_text": "\u517b\u72ac\u533a\u5212\u7ba1\u7406\u3001\u514d\u75ab\u4e0e\u767b\u8bb0",
            "text": "\u517b\u72ac\u533a\u5212\u7ba1\u7406\u3001\u514d\u75ab\u4e0e\u767b\u8bb0",
            "chunk_role": "section_heading",
            "unit_kind": "section_heading",
        },
    ]

    contextualized = chunking.contextualize_chunk_items(runtime, "demo.docx", items)

    assert contextualized[1].get("article_id") == ""
    assert contextualized[1].get("article_no") == ""


def test_semantic_plan_marks_paragraph_with_article_id_as_article_role():
    runtime = _runtime()
    article = "\u7b2c\u5341\u6761"
    units = [
        {
            "unit_id": "su_000001",
            "unit_type": "paragraph",
            "unit_kind": "paragraph",
            "section": "\u603b\u5219",
            "section_title": "\u603b\u5219",
            "section_path": ["\u603b\u5219"],
            "raw_text": f"{article}  \u5efa\u8bbe\u5355\u4f4d\u5e94\u5f53\u4f9d\u6cd5\u5c65\u884c\u7ba1\u7406\u4e49\u52a1\u3002",
            "text": f"{article}  \u5efa\u8bbe\u5355\u4f4d\u5e94\u5f53\u4f9d\u6cd5\u5c65\u884c\u7ba1\u7406\u4e49\u52a1\u3002",
            "article_id": article,
            "article_no": article,
        }
    ]

    plans = chunking.semantic_units_to_chunk_plans(runtime, "demo.pdf", units, chunk_size=300, overlap=30)

    assert plans[0]["chunk_role"] == "article"
    assert plans[0]["article_id"] == article


def test_semantic_plan_splits_pdf_page_text_with_multiple_articles():
    runtime = _runtime()
    body = (
        "\u7b2c\u4e00\u7ae0\u603b\u5219\n"
        "\u7b2c \u4e00 \u6761 \u4e3a\u4e86\u52a0\u5f3a\u7ba1\u7406\uff0c\u5236\u5b9a\u672c\u6761\u4f8b\u3002\n"
        "\u7b2c \u4e8c \u6761 \u672c\u6761\u4f8b\u9002\u7528\u4e8e\u672c\u884c\u653f\u533a\u57df\u3002\n"
        "\u7b2c\u4e09\u6761 \u6709\u5173\u90e8\u95e8\u5e94\u5f53\u4f9d\u6cd5\u5c65\u804c\u3002"
    )
    units = [
        {
            "unit_id": "su_000001",
            "unit_type": "paragraph",
            "unit_kind": "paragraph",
            "section": "\u603b\u5219",
            "section_title": "\u603b\u5219",
            "section_path": ["\u603b\u5219"],
            "raw_text": body,
            "text": body,
        }
    ]

    plans = chunking.semantic_units_to_chunk_plans(runtime, "demo.pdf", units, chunk_size=1000, overlap=30)

    assert [plan["article_id"] for plan in plans] == ["\u7b2c\u4e00\u6761", "\u7b2c\u4e8c\u6761", "\u7b2c\u4e09\u6761"]
    assert [plan["chunk_role"] for plan in plans] == ["article", "article", "article"]


def test_semantic_plan_does_not_split_inline_article_references():
    runtime = _runtime()
    body = (
        "\u7b2c\u5341\u6761 \u8fdd\u53cd\u672c\u6761\u4f8b\u7b2c\u4e09\u6761\u3001\u7b2c\u56db\u6761\u89c4\u5b9a\u7684\uff0c"
        "\u7531\u6709\u5173\u90e8\u95e8\u4f9d\u6cd5\u5904\u7406\u3002"
    )
    units = [
        {
            "unit_id": "su_000001",
            "unit_type": "paragraph",
            "unit_kind": "paragraph",
            "section": "\u6cd5\u5f8b\u8d23\u4efb",
            "section_title": "\u6cd5\u5f8b\u8d23\u4efb",
            "section_path": ["\u6cd5\u5f8b\u8d23\u4efb"],
            "raw_text": body,
            "text": body,
        }
    ]

    plans = chunking.semantic_units_to_chunk_plans(runtime, "demo.pdf", units, chunk_size=1000, overlap=30)

    assert len(plans) == 1
    assert plans[0]["article_id"] == "\u7b2c\u5341\u6761"
