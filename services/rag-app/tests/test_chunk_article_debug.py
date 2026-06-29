import json

from app.documents.chunking import contextualize_chunk_items


class _Runtime:
    normalize_ir_text = staticmethod(lambda text: text)
    chapter_heading_title = staticmethod(lambda text: None)
    clause_heading_label = staticmethod(lambda text: "")
    serialize_ir_element = staticmethod(lambda element: element)
    doc_title_profile = staticmethod(lambda filename: {"canonical_title": filename})
    filename_stem = staticmethod(lambda filename: filename)
    split_text = staticmethod(lambda text, chunk_size, overlap: [text])
    chunk_prev_context_chars = 220
    chunk_next_context_chars = 220


def test_chunk_article_debug_prints_text_and_article_id(monkeypatch, capsys):
    monkeypatch.setenv("DEBUG_CHUNK_ARTICLE_SPLIT", "true")
    monkeypatch.setenv("DEBUG_CHUNK_ARTICLE_MAX_CHARS", "0")

    contextualize_chunk_items(
        _Runtime(),
        "demo.docx",
        [{"raw_text": "第二十二条 养犬人携犬出户应当遵守规定。", "article_id": "第二十二条"}],
    )

    output = capsys.readouterr().out.strip()
    payload = json.loads(output)
    assert payload["event"] == "chunk_article_split"
    assert payload["article_id"] == "第二十二条"
    assert payload["text"] == "第二十二条 养犬人携犬出户应当遵守规定。"


def test_chunk_article_debug_writes_jsonl_file(monkeypatch, tmp_path, capsys):
    output_path = tmp_path / "chunk_article_debug.jsonl"
    monkeypatch.setenv("DEBUG_CHUNK_ARTICLE_SPLIT", "true")
    monkeypatch.setenv("DEBUG_CHUNK_ARTICLE_OUTPUT", str(output_path))

    contextualize_chunk_items(
        _Runtime(),
        "demo.docx",
        [{"raw_text": "第三十八条 违反规定的，由公安机关责令改正。", "article_id": "第三十八条"}],
    )

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert payload["article_id"] == "第三十八条"
    assert payload["text"] == "第三十八条 违反规定的，由公安机关责令改正。"
