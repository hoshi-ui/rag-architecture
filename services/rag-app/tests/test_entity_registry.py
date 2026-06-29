from app.documents import entity_registry
from app.documents import profile as document_profile
from app.core import compare as compare_core


def test_generate_entity_aliases_includes_compact_region_short_title():
    aliases = entity_registry.generate_entity_aliases(
        "林芝市出租房安全管理条例_2024-12-06_2025-01-01.docx",
        "林芝市出租房安全管理条例",
        region="林芝",
    )

    assert "林芝市出租房安全管理条例" in aliases
    assert "林芝出租房安全管理条例" in aliases
    assert "出租房安全管理条例" in aliases


def test_registered_entity_match_binds_query_to_source_by_alias():
    records = [
        entity_registry.EntityRecord(
            source="林芝市出租房安全管理条例_2024-12-06_2025-01-01.docx",
            canonical_title="林芝市出租房安全管理条例",
            region="林芝",
            filename_stem="林芝市出租房安全管理条例_2024-12-06_2025-01-01",
        ),
        entity_registry.EntityRecord(
            source="聊城市养犬管理条例_2020-06-15_2020-09-01.docx",
            canonical_title="聊城市养犬管理条例",
            region="聊城",
            filename_stem="聊城市养犬管理条例_2020-06-15_2020-09-01",
        ),
    ]

    matches = entity_registry.rank_registered_entity_records(
        "林芝出租房安全管理中公安机关负责什么？",
        records,
    )

    assert matches
    assert matches[0]["source"] == "林芝市出租房安全管理条例_2024-12-06_2025-01-01.docx"
    assert matches[0]["match_kind"] == "registered_entity"


def test_generate_entity_aliases_does_not_reparse_admin_suffix_inside_subject():
    aliases = entity_registry.generate_entity_aliases(
        "深圳市建筑市场严重违法行为特别处理规定_2007-08-15_.docx",
        "深圳市建筑市场严重违法行为特别处理规定",
        region="深圳",
    )

    assert "深圳建筑市场严重违法行为特别处理规定" in aliases
    assert not any(alias.startswith("场严重") for alias in aliases)
    assert not any("深圳场严重" in alias for alias in aliases)


def test_profile_prefers_complete_legal_title_from_filename_when_ir_title_is_truncated():
    title = document_profile._prefer_complete_legal_title(
        "深圳市建筑市场严重违法行为",
        "深圳市建筑市场严重违法行为特别处理规定_2007-08-15_.docx",
    )

    assert title == "深圳市建筑市场严重违法行为特别处理规定"


def test_fuzzy_id_match_accepts_region_and_partial_legal_entity():
    match = entity_registry.fuzzy_id_match(
        "深圳建筑市场中，工程勘察、设计、施工等单位存在事故隐患",
        ["深圳市建筑市场严重违法行为特别处理规定"],
        region="深圳",
    )

    assert match["accepted"] is True
    assert match["score"] >= 4.7


def test_fuzzy_id_match_accepts_region_and_domain_short_name():
    match = entity_registry.fuzzy_id_match(
        "聊城重点管理区个人养犬登记",
        ["聊城市养犬管理条例"],
        region="聊城",
    )

    assert match["accepted"] is True


def test_fuzzy_id_match_rejects_same_region_without_core_overlap():
    match = entity_registry.fuzzy_id_match(
        "深圳物业管理事项",
        ["深圳市建筑市场严重违法行为特别处理规定"],
        region="深圳",
    )

    assert match["accepted"] is False


def test_compare_subject_source_allows_region_fuzzy_probe_without_doc_suffix():
    result = compare_core.resolve_compare_subject_source(
        "聊城重点管理区个人养犬登记",
        normalize_query=lambda value: str(value or ""),
        normalize_filename=lambda value: str(value or ""),
        source_display_title=lambda _source: "聊城市养犬管理条例",
        looks_like_document_target=lambda _target: False,
        extract_strong_title_source_matches=lambda _target, limit=3: [
            {
                "source": "聊城市养犬管理条例_2020-06-15_2020-09-01.docx",
                "match_kind": "fuzzy_id",
                "score": 7.2,
            }
        ],
        rank_title_source_matches=lambda *_args, **_kwargs: [],
        build_doc_recall_plan=lambda *_args, **_kwargs: [],
    )

    assert result["source"] == "聊城市养犬管理条例_2020-06-15_2020-09-01.docx"
    assert result["match_kind"] == "fuzzy_id"
