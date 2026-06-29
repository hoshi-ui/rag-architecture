from app.core.answer import AnswerAdapter, build_legal_clause_enumeration_result


def _adapter() -> AnswerAdapter:
    return AnswerAdapter(
        normalize_coverage_aspect=lambda text: text,
        normalize_query=lambda text: text,
        chunk_plain_display_text=lambda text: text,
        hit_display_text=lambda hit: hit["entity"]["metadata"].get("raw_text") or hit["entity"].get("text") or "",
        coverage_aspect_variants=lambda text: [text],
        build_sources=lambda docs, query, answer: [],
        answer_limits=lambda qtype: {},
        llm_temperature=0.0,
        llm_top_p=1.0,
        llm_max_tokens=1000,
        llm_presence_penalty=0.0,
        llm_timeout=30.0,
        final_fact_verify_max_tokens=1000,
        hit_metadata=lambda hit: hit["entity"].get("metadata") or {},
        hit_entity_source=lambda hit: hit["entity"].get("source") or "",
    )


def _doc(text: str):
    return {"entity": {"source": "聊城市养犬管理条例.docx", "text": text, "metadata": {"raw_text": text}}}


def test_legal_clause_subject_filter_expands_behavior_subject():
    answer, docs = build_legal_clause_enumeration_result(
        _adapter(),
        "聊城市养犬管理条例中，对养犬行为有哪些限制？",
        [
            _doc(
                "正文：第二十二条 养犬人携犬出户应当遵守下列规定：\n"
                "（一）为犬只佩戴犬牌；\n"
                "（二）用束犬绳牵领犬只，并主动避让他人；"
            )
        ],
        subject_filter={"target_subject": ["养犬行为"]},
    )

    assert docs
    assert "第二十二条" in answer
    assert "佩戴犬牌" in answer
    assert "束犬绳" in answer


def test_legal_clause_subject_filter_expands_excluded_responsibility_subject():
    answer, _ = build_legal_clause_enumeration_result(
        _adapter(),
        "聊城市养犬管理条例中，对养犬行为有哪些限制？",
        [
            _doc("正文：第三十二条 公安机关应当组织对流浪犬只进行捕捉，并送至犬只收容救助场所。")
        ],
        subject_filter={"target_subject": ["养犬行为"], "excluded_subject": ["公安机关职责"]},
    )

    assert answer == ""
