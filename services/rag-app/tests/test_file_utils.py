from app.utils.files import filename_stem, normalize_filename_for_match, safe_filename


def test_normalize_filename_for_match_accepts_source_dict():
    assert normalize_filename_for_match({"source": "/tmp/聊城市养犬管理条例.docx"}) == "聊城市养犬管理条例.docx"


def test_filename_helpers_accept_filename_dict():
    payload = {"filename": r"C:\docs\聊城市养犬管理条例_2020-06-15.docx"}

    assert filename_stem(payload) == "聊城市养犬管理条例_2020-06-15"
    assert safe_filename(payload) == "聊城市养犬管理条例_2020-06-15.docx"
