from app.core.answer import rewrite_answer_citation_protocol


def test_rewrite_citation_protocol_injects_exact_legal_citation() -> None:
    result = rewrite_answer_citation_protocol(
        "公安机关负责出租房治安的统一监督管理[1]",
        [
            {
                "ref": 1,
                "source": "林芝市出租房安全管理条例_2024-12-06_2025-01-01.docx",
                "section": "第五条",
                "chunk_range": "1-1",
                "text": "公安机关负责出租房治安的统一监督管理。",
                "metadata": {
                    "doc_title": "林芝市出租房安全管理条例",
                    "article_no": "第五条",
                    "clause_metadata": {
                        "doc_title": "林芝市出租房安全管理条例",
                        "article_no": "第五条",
                    },
                },
            }
        ],
    )

    assert "《林芝市出租房安全管理条例》第五条[1]" in result["answer"]
    assert result["answer_refs"] == [1]


def test_rewrite_citation_protocol_strips_source_filename_version_for_citation_title() -> None:
    result = rewrite_answer_citation_protocol(
        "建设项目需要使用林地的，应当依法办理审批手续[1]",
        [
            {
                "ref": 1,
                "source": "深圳市林地保护管理办法_2024-01-01_2024-03-01.pdf",
                "section": "第二十七条",
                "chunk_range": "2-2",
                "text": "建设项目需要使用林地的，应当依法办理审批手续。",
                "metadata": {"article_no": "第二十七条"},
            }
        ],
    )

    assert "《深圳市林地保护管理办法》第二十七条[1]" in result["answer"]


def test_rewrite_citation_protocol_does_not_duplicate_exact_legal_citation() -> None:
    result = rewrite_answer_citation_protocol(
        "依据《林芝市出租房安全管理条例》第五条[1]，公安机关负责治安管理。",
        [
            {
                "ref": 1,
                "source": "林芝市出租房安全管理条例.docx",
                "section": "第五条",
                "chunk_range": "1-1",
                "text": "公安机关负责出租房治安的统一监督管理。",
                "metadata": {
                    "doc_title": "林芝市出租房安全管理条例",
                    "article_no": "第五条",
                },
            }
        ],
    )

    assert result["answer"].count("《林芝市出租房安全管理条例》第五条") == 1


def test_rewrite_citation_protocol_prefers_source_title_over_chapter_title() -> None:
    result = rewrite_answer_citation_protocol(
        "街道办事处、乡镇人民政府应当及时指导业主成立筹备组[1]",
        [
            {
                "ref": 1,
                "source": "绍兴市物业管理条例_2023-06-02_2023-06-02.pdf",
                "section": "第二章 业主大会和业主委员会",
                "chunk_range": "1-1",
                "text": "街道办事处、乡镇人民政府应当及时指导业主成立筹备组。",
                "metadata": {
                    "doc_title": "第一章总则",
                    "source_file": "绍兴市物业管理条例_2023-06-02_2023-06-02.pdf",
                    "article_no": "第十一条",
                    "clause_metadata": {
                        "doc_title": "第一章总则",
                        "article_no": "第十一条",
                    },
                },
            }
        ],
    )

    assert "《绍兴市物业管理条例》第十一条[1]" in result["answer"]
    assert "《第一章总则》第十一条" not in result["answer"]


def test_rewrite_citation_protocol_replaces_existing_wrong_exact_title() -> None:
    result = rewrite_answer_citation_protocol(
        "街道办事处、乡镇人民政府应当及时指导业主成立筹备组《第一章总则》第十一条[1]",
        [
            {
                "ref": 1,
                "source": "绍兴市物业管理条例_2023-06-02_2023-06-02.pdf",
                "section": "第二章 业主大会和业主委员会",
                "chunk_range": "1-1",
                "text": "街道办事处、乡镇人民政府应当及时指导业主成立筹备组。",
                "metadata": {
                    "doc_title": "第一章总则",
                    "source_file": "绍兴市物业管理条例_2023-06-02_2023-06-02.pdf",
                    "article_no": "第十一条",
                },
            }
        ],
    )

    assert "《绍兴市物业管理条例》第十一条[1]" in result["answer"]
    assert "《第一章总则》第十一条" not in result["answer"]


def test_rewrite_citation_protocol_prefers_complete_source_title_over_truncated_doc_title() -> None:
    result = rewrite_answer_citation_protocol(
        "逾期拒不整改或者整改不合格的，由市建设行政主管部门暂扣其资质证书[1]",
        [
            {
                "ref": 1,
                "source": "深圳市建筑市场严重违法行为特别处理规定_2007-08-15_.docx",
                "section": "特别处理规定",
                "chunk_range": "1-1",
                "text": "逾期拒不整改或者整改不合格的，由市建设行政主管部门暂扣其资质证书。",
                "metadata": {
                    "doc_title": "深圳市建筑市场严重违法行为",
                    "source_file": "深圳市建筑市场严重违法行为特别处理规定_2007-08-15_.docx",
                    "article_no": "第五条",
                },
            }
        ],
    )

    assert "《深圳市建筑市场严重违法行为特别处理规定》第五条[1]" in result["answer"]
    assert "《深圳市建筑市场严重违法行为》第五条" not in result["answer"]


def test_rewrite_citation_protocol_strips_space_separated_date_from_title() -> None:
    result = rewrite_answer_citation_protocol(
        "本条例适用于本市行政区域内建筑废弃物减排与利用活动[1]",
        [
            {
                "ref": 1,
                "source": "深圳市建筑废弃物减排与利用条例_2009-05-31_.docx",
                "section": "总则",
                "chunk_range": "1-1",
                "text": "本条例适用于本市行政区域内建筑废弃物减排与利用活动。",
                "metadata": {
                    "doc_title": "深圳市建筑废弃物减排与利用条例 2009 05 31",
                    "source_file": "深圳市建筑废弃物减排与利用条例_2009-05-31_.docx",
                    "article_no": "第二条",
                },
            }
        ],
    )

    assert "《深圳市建筑废弃物减排与利用条例》第二条[1]" in result["answer"]
    assert "2009 05 31" not in result["answer"]
