from types import SimpleNamespace

from app.runtime.query import QueryCore


def test_query_core_exposes_source_display_title_adapter():
    core = QueryCore.__new__(QueryCore)
    core.source = SimpleNamespace(display_title=lambda source: f"title:{source}")

    assert core.source_display_title("a.pdf") == "title:a.pdf"
