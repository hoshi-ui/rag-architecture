from app.core.source.explicit import regulation_identity_key


def test_regulation_identity_key_strips_hyphenated_dates_without_regex_error():
    source = "聊城市养犬管理条例_2020-06-15_2020-09-01.docx"

    key = regulation_identity_key(
        source,
        normalize_filename=lambda value: value,
        source_display_title=lambda value: value.rsplit(".", 1)[0],
        normalize_reference_text=lambda value: "".join(str(value).split()).replace("_", ""),
        source_profile_fields=lambda value: {"region": "聊城市"},
        normalize_query=lambda value: "".join(str(value).split()),
        extract_region_token=lambda value: "",
    )

    assert key == "聊城市|聊城市养犬管理条例"
