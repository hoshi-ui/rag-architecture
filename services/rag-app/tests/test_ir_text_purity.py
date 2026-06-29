from app.documents import chunking
from app.documents import ir as document_ir


def _runtime():
    return chunking.DocumentChunkingAdapter(
        normalize_ir_text=document_ir.normalize_ir_text,
        chapter_heading_title=lambda text: None,
        clause_heading_label=lambda text: "",
        serialize_ir_element=lambda element: document_ir.serialize_ir_element(
            element,
            should_skip=lambda _element: False,
            normalize_section_path=chunking.normalize_section_path,
            section_path_label=chunking.section_path_label,
        ),
        doc_title_profile=lambda filename: {"canonical_title": filename},
        filename_stem=lambda filename: filename.rsplit(".", 1)[0],
        split_text=lambda text, chunk_size, overlap: [text],
    )


def test_append_ir_element_keeps_text_raw_pure_and_metadata_structured():
    doc = document_ir.new_document_ir(
        "demo.docx",
        safe_filename=lambda name: name,
        parser_name="unit-test",
    )

    document_ir.append_ir_element(
        doc,
        element_type="paragraph",
        text_raw="第十七条 公安机关应当依法履行管理职责。",
        page_no=3,
        section_path=["第三章 管理规范"],
    )

    element = doc["elements"][0]

    assert element["text_raw"] == "第十七条 公安机关应当依法履行管理职责。"
    assert element["metadata"]["page_no"] == 3
    assert element["metadata"]["section_path"] == ["第三章 管理规范"]
    assert element["metadata"]["element_type"] == "paragraph"


def test_serialize_ir_element_does_not_render_metadata_into_text():
    element = {
        "text_raw": "第十七条 公安机关应当依法履行管理职责。",
        "text_normalized": "第十七条 公安机关应当依法履行管理职责。",
        "page_no": 3,
        "section_path": ["第三章 管理规范"],
        "element_type": "paragraph",
    }

    serialized = document_ir.serialize_ir_element(
        element,
        should_skip=lambda _element: False,
        normalize_section_path=chunking.normalize_section_path,
        section_path_label=chunking.section_path_label,
    )

    assert serialized["serialized_text"] == "第十七条 公安机关应当依法履行管理职责。"
    assert "章节路径" not in serialized["serialized_text"]
    assert "页码" not in serialized["serialized_text"]
    assert "元素类型" not in serialized["serialized_text"]


def test_document_ir_to_structured_items_uses_raw_body_text_only():
    doc = document_ir.new_document_ir(
        "demo.docx",
        safe_filename=lambda name: name,
        parser_name="unit-test",
    )
    document_ir.append_ir_element(
        doc,
        element_type="paragraph",
        text_raw="第十七条 公安机关应当依法履行管理职责。",
        page_no=3,
        section_path=["第三章 管理规范"],
    )

    items = chunking.document_ir_to_structured_items(_runtime(), doc, chunk_size=500, overlap=100)

    assert len(items) == 1
    assert items[0]["text"] == "第十七条 公安机关应当依法履行管理职责。"
    assert items[0]["raw_text"] == "第十七条 公安机关应当依法履行管理职责。"
    assert "章节路径" not in items[0]["text"]
    assert "页码" not in items[0]["text"]
    assert "元素类型" not in items[0]["text"]
