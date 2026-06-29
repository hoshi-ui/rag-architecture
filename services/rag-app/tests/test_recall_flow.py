from app.core.query.recall_flow import (
    build_lightweight_recall_result,
    build_multi_doc_compare_result,
    build_retrieve_source_lock_result,
    build_retrieve_success_result,
    drop_invalid_parsed_article_filter,
    handle_required_source_lock,
    has_forced_retrieval_signal,
    prepare_retrieve_evidence_context,
    prepare_retrieval_query_context,
    prepare_recall_source_context,
    source_resolution_delayed_global_fallback,
    source_resolution_router_target_failure_fallback,
)
from app.core.query.process import build_process_source_lock_result, prepare_process_evidence_context
from app.core import retrieval as retrieval_core
from app.core.compare import compare_source_set_completeness
from app.core.source.explicit import extract_explicit_regulation_mentions
from app.core.retrieval.clauses import clause_level_rerank
from app.core.retrieval.chunks import intra_doc_chunk_rerank, merge_and_dedupe_hits
from app.core.retrieval.ranking import _select_with_pinned_clauses, chunk_level_rerank
from app.core.evidence.selection import select_retrieve_output_docs


class _Routing:
    def extract_filename_candidates(self, query):
        return []

    def extract_explicit_regulation_mentions(self, query):
        return []

    def has_contextual_doc_reference(self, query):
        return False

    def strong_topic_terms(self, query):
        return []

    def has_strong_business_signal(self, query):
        return False

    def has_weak_business_signal(self, query):
        return False

    def query_filters(self, query):
        return {}

    def is_weak_reference_query(self, query):
        return False

    def classify_question_type(self, query):
        return "general"


class _Retrieval:
    def seed_anchor_terms_for_probe(self, query):
        return []

    def clarification_candidates(self, query, seed_sources=None, limit=3):
        return list(seed_sources or [])[:limit]


class _Common:
    @staticmethod
    def normalize_query(value):
        return str(value or "").strip()

    @staticmethod
    def normalize_filename(value):
        return str(value or "").strip()

    @staticmethod
    def policy_get(name, default=None):
        return default


def _hit(source, score=0.5, text="body", doc_id=None):
    return {
        "entity": {
            "source": source,
            "text": text,
            "metadata": {"doc_id": doc_id or source, "chunk_id": int(score * 100)},
        },
        "score": score,
    }


class _Compare:
    @staticmethod
    def has_intent(query):
        return False

    @staticmethod
    def matrix_presence_state(value):
        return str(value or "")

    @staticmethod
    def clarification_prompt(subjects, candidates):
        return "compare clarification"

    @staticmethod
    def target_not_found_prompt(missing, candidates):
        return "compare target not found"


class _Runtime:
    common = _Common()
    routing = _Routing()
    retrieval = _Retrieval()
    compare = _Compare()


class _Control:
    @staticmethod
    def metadata(**kwargs):
        return kwargs


class _SourceForResponses:
    @staticmethod
    def clarification_prompt(candidates):
        return "clarify: " + ",".join(candidates or [])

    @staticmethod
    def not_found_prompt(target):
        return "not found: " + str(target or "")


class _EvidenceForResponses:
    @staticmethod
    def filter_display_sources(docs, score_mode, qfilters, resolved_targets, qtype, max_sources, target_sources, observations):
        targets = set(resolved_targets or [])
        return [doc for doc in docs if not targets or doc["entity"]["source"] in targets][:max_sources]

    @staticmethod
    def build_sources(docs, query, score_mode="score"):
        return [{"source": doc["entity"]["source"], "metadata": doc["entity"].get("metadata") or {}} for doc in docs]

    @staticmethod
    def hit_entity_source(doc):
        return doc["entity"]["source"]

    @staticmethod
    def hit_score(doc):
        return doc.get("score", 0)

    @staticmethod
    def hit_display_text(doc):
        return doc["entity"].get("text") or ""

    @staticmethod
    def build_excerpt(text, query, limit):
        return str(text or "")[:limit]

    @staticmethod
    def hit_metadata(doc):
        return doc["entity"].get("metadata") or {}

    @staticmethod
    def hit_chunk_range(doc):
        return None

    @staticmethod
    def merge_compare_source_doc_groups(groups, per_source_limit):
        docs = []
        limit = max(1, int(per_source_limit))
        grouped = [list((group.get("docs") or [])[:limit]) for group in groups or []]
        max_len = max((len(items) for items in grouped), default=0)
        for idx in range(max_len):
            for items in grouped:
                if idx < len(items):
                    docs.append(items[idx])
        return docs


class _ResponseRuntime:
    common = _Common()
    routing = _Routing()
    compare = _Compare()
    control = _Control()
    source = _SourceForResponses()
    evidence = _EvidenceForResponses()


def test_forced_retrieval_signal_detects_article_number():
    assert has_forced_retrieval_signal(_Runtime(), "\u7b2c\u5341\u4e03\u6761\u5982\u4f55\u5904\u7406")


def test_explicit_regulation_extractor_ignores_topical_rule_phrases():
    mentions = extract_explicit_regulation_mentions(
        "聊城市重点管理区犬只出生满三个月后，免疫、芯片和禁养规则应如何依次判断？",
        normalize_query=_Common.normalize_query,
        extract_filename_candidates=lambda _query: [],
    )

    assert mentions == []


def test_explicit_regulation_extractor_keeps_real_legal_titles():
    mentions = extract_explicit_regulation_mentions(
        "《聊城市养犬管理条例》第十条如何规定？",
        normalize_query=_Common.normalize_query,
        extract_filename_candidates=lambda _query: [],
    )

    assert mentions == ["聊城市养犬管理条例"]


def test_required_source_lock_delays_clarification_when_article_is_explicit():
    result = handle_required_source_lock(
        _Runtime(),
        "\u7b2c\u5341\u4e03\u6761\u5982\u4f55\u5904\u7406",
        "\u7b2c\u5341\u4e03\u6761\u5982\u4f55\u5904\u7406",
        "general",
        {},
        {"search_database_tool_used": True},
        {
            "required": True,
            "resolved": False,
            "status": "ambiguous",
            "reason": "document_target_required",
            "candidates": [],
        },
        "document_clarification",
        None,
        False,
        {"quality": "valid"},
        "",
        {},
        3,
        [],
        [],
        False,
        user_id="u",
    )

    assert result["early_return"] is None
    assert result["query_route"] == "content_qa"
    assert result["source_resolution"]["required"] is False
    assert result["source_resolution"]["status"] == "global_fallback"
    assert source_resolution_delayed_global_fallback(result["source_resolution"]) is True
    assert result["source_resolution"]["source_resolution_trace"]["delayed_clarification_global_fallback"] is True


def test_locked_source_recall_does_not_cross_target_doc():
    class VectorDB:
        def __init__(self):
            self.filters = []

        def search(self, embedding, top_k, filters=None, query_sparse_embedding=None):
            self.filters.append(filters)
            if "A.pdf" not in str(filters):
                return [_hit("B.pdf", 0.99, "B stronger keyword", "B"), _hit("A.pdf", 0.40, "A weaker keyword", "A")]
            return [_hit("A.pdf", 0.40, "A locked weaker keyword", "A")]

    class ScopedRuntime:
        vector_db = VectorDB()

        @staticmethod
        def normalize_filename_for_match(value):
            return str(value or "").strip()

        @staticmethod
        def normalize_query(value):
            return str(value or "").strip()

        @staticmethod
        def query_has_compare_intent(query):
            return False

        @staticmethod
        def filter_hits_by_source_state(docs):
            return {"hits": list(docs or []), "dropped": 0, "states": {}}

        @staticmethod
        def dense_source_score_map(docs):
            return {doc["entity"]["source"]: doc.get("score", 0) for doc in docs or []}

        @staticmethod
        def build_doc_recall_plan(query, limit=3, source_filter=None):
            assert source_filter == "A.pdf"
            return []

        @staticmethod
        def collect_lexical_candidates(query, active_sources, doc_recall_plan, article_ids=None):
            assert active_sources == ["A.pdf"]
            return [_hit("A.pdf", 0.35, "A lexical weaker keyword", "A")]

        @staticmethod
        def should_apply_chunk_rerank(docs, dense_rank_map, lex_rank_map, source_signals, enable_rerank):
            return True

        @staticmethod
        def expand_heading_hits_to_article_hits(query, source, docs, limit):
            assert source == "A.pdf"
            return list(docs or [])[:limit]

        @staticmethod
        def config_value(name, default=None):
            if name == "MAX_MERGED_CHUNKS_PER_EVIDENCE":
                return 1
            return default

    class Handler:
        rerank_service = None
        embedding_service = None
        vector_db = ScopedRuntime.vector_db

    original_fuse = retrieval_core.fuse_dense_lexical_hits
    original_chunk = retrieval_core.chunk_level_rerank
    original_postprocess = retrieval_core.postprocess_recall_docs

    def fake_fuse(runtime, docs, lex_items, effective_query, recall_k, dense_source_scores=None, fname_set=None, doc_recall_plan=None):
        assert fname_set == {"A.pdf"}
        combined = list(docs or []) + list(lex_items or [])
        assert {doc["entity"]["source"] for doc in combined} == {"A.pdf"}
        return {"docs": combined, "dense_rank_map": {}, "lex_rank_map": {}, "source_signals": {}}

    async def fake_chunk(runtime, rerank_service, query, docs, pool_n, enabled, **kwargs):
        assert enabled is True
        assert {doc["entity"]["source"] for doc in docs} == {"A.pdf"}
        return {"hits": list(docs or [])[:pool_n], "score_mode": "score", "used": True}

    def fake_postprocess(runtime, docs, score_mode, query, qtype, qfilters, active_fnames, final_n):
        assert active_fnames == ["A.pdf"]
        assert {doc["entity"]["source"] for doc in docs} == {"A.pdf"}
        selected = list(docs or [])[:final_n]
        return {"selected_docs": selected, "post_filter_docs": selected, "retrieve_docs": selected}

    retrieval_core.fuse_dense_lexical_hits = fake_fuse
    retrieval_core.chunk_level_rerank = fake_chunk
    retrieval_core.postprocess_recall_docs = fake_postprocess
    try:
        import asyncio

        result = asyncio.run(
            retrieval_core.run_target_scoped_recall(
                ScopedRuntime(),
                Handler(),
                query="A file keyword",
                retrieval_query="keyword",
                query_embedding=[0.1],
                query_sparse_embedding=None,
                qtype="general",
                qfilters={},
                recall_k=10,
                final_n=5,
                pool_n=10,
                enable_rerank=True,
                target_source="A.pdf",
            )
        )
    finally:
        retrieval_core.fuse_dense_lexical_hits = original_fuse
        retrieval_core.chunk_level_rerank = original_chunk
        retrieval_core.postprocess_recall_docs = original_postprocess

    assert "A.pdf" in str(ScopedRuntime.vector_db.filters[0])
    assert {doc["entity"]["source"] for doc in result["selected_docs"]} == {"A.pdf"}
    assert {doc["entity"]["source"] for doc in result["retrieve_docs"]} == {"A.pdf"}
    assert "B.pdf" not in {doc["entity"]["source"] for doc in result["docs"]}


def test_clause_level_rerank_keeps_exact_article_over_neighbor():
    class RerankService:
        async def rerank(self, query, documents, top_k):
            return [
                {"index": 1, "score": 0.99},
                {"index": 0, "score": 0.20},
            ]

    class Runtime:
        @staticmethod
        def clone_hit_with_score(hit, score):
            return {"entity": hit["entity"], "score": score}

    hits = [
        {
            "entity": {
                "source": "A.pdf",
                "text": "第十二条 查封、扣押应当符合法定条件。",
                "metadata": {"article_id": "第十二条", "article_no": "第十二条", "chunk_id": 12, "heading": "查封、扣押条件"},
            },
            "score": 0.2,
        },
        {
            "entity": {
                "source": "A.pdf",
                "text": "第十三条 查封、扣押物品的处理程序包含关键词。",
                "metadata": {"article_id": "第十三条", "article_no": "第十三条", "chunk_id": 13, "heading": "处理程序"},
            },
            "score": 0.99,
        },
    ]

    import asyncio

    result = asyncio.run(
        clause_level_rerank(
            Runtime(),
            RerankService(),
            "第十二条查封扣押条件怎么规定",
            hits,
            top_k=5,
            enable_rerank=True,
            mentioned_articles=["第十二条"],
            doc_title="深圳经济特区无照经营查处若干规定",
        )
    )

    assert [doc["entity"]["metadata"]["article_id"] for doc in result["hits"]] == ["第十二条"]
    assert result["trace"]["selected_articles"] == ["第十二条"]
    assert "第十三条" not in result["trace"]["selected_articles"]


def test_clause_level_rerank_boosts_matching_macro_legal_intent():
    class Runtime:
        @staticmethod
        def clone_hit_with_score(hit, score):
            return {"entity": hit["entity"], "score": score}

    hits = [
        {
            "entity": {
                "source": "linzhi.docx",
                "text": "第五条 有关主管部门应当按照职责负责监督管理工作。",
                "metadata": {"article_id": "第五条", "article_no": "第五条", "chunk_id": 5, "heading": "职责权限"},
            },
            "score": 0.50,
        },
        {
            "entity": {
                "source": "linzhi.docx",
                "text": "第二十八条 违反本条例规定的，由有关部门依法处理。",
                "metadata": {"article_id": "第二十八条", "article_no": "第二十八条", "chunk_id": 28, "heading": "法律责任"},
            },
            "score": 0.60,
        },
    ]

    import asyncio

    result = asyncio.run(
        clause_level_rerank(
            Runtime(),
            None,
            "有关部门职责分工",
            hits,
            top_k=1,
            enable_rerank=False,
            doc_title="林芝市出租房安全管理条例",
            query_intent="职责与权限",
        )
    )

    assert result["hits"][0]["entity"]["metadata"]["article_id"] == "第五条"
    assert "legal_intent_match_bonus" in result["trace"]["adjustments"][0]["reasons"]


def test_clause_level_rerank_boosts_definition_scope_early_article():
    class Runtime:
        @staticmethod
        def clone_hit_with_score(hit, score):
            return {"entity": hit["entity"], "score": score}

    hits = [
        {
            "entity": {
                "source": "scope.pdf",
                "text": "第二条 本条例适用于本行政区域内相关管理活动。",
                "metadata": {"article_id": "第二条", "article_no": "第二条", "chunk_id": 2, "heading": "适用范围"},
            },
            "score": 0.45,
        },
        {
            "entity": {
                "source": "scope.pdf",
                "text": "第十二条 相关主体应当履行日常管理义务。",
                "metadata": {"article_id": "第十二条", "article_no": "第十二条", "chunk_id": 12, "heading": "管理义务"},
            },
            "score": 0.62,
        },
    ]

    import asyncio

    result = asyncio.run(
        clause_level_rerank(
            Runtime(),
            None,
            "这个条例的适用范围是什么？",
            hits,
            top_k=1,
            enable_rerank=False,
            doc_title="示例条例",
            query_intent="定义与范围",
        )
    )

    assert result["hits"][0]["entity"]["metadata"]["article_id"] == "第二条"
    assert "definition_scope_early_article_bonus" in result["trace"]["adjustments"][0]["reasons"]


def test_chunk_level_rerank_sends_metadata_aware_text_to_model():
    captured = {}

    class RerankService:
        async def rerank(self, query, documents, top_k):
            captured["documents"] = documents
            return [{"index": 0, "score": 0.9}, {"index": 1, "score": 0.1}]

    class Runtime:
        @staticmethod
        def hit_score_mode(hit):
            return "score"

        @staticmethod
        def config_value(name, default=None):
            return True if name == "ENABLE_RERANK" else default

        @staticmethod
        def hit_metadata(hit):
            return hit["entity"].get("metadata") or {}

        @staticmethod
        def hit_entity_text(hit):
            return hit["entity"].get("text") or ""

        @staticmethod
        def hit_entity_source(hit):
            return hit["entity"].get("source") or ""

    hit = {
        "entity": {
            "source": "linzhi.docx",
            "text": "公安机关负责出租房治安管理。",
            "metadata": {
                "doc_title": "林芝市出租房安全管理条例",
                "article_no": "第五条",
                "section_title": "职责权限",
                "heading": "公安机关职责",
            },
        },
        "score": 0.5,
    }
    other = {
        "entity": {
            "source": "linzhi.docx",
            "text": "违反本条例规定的，由有关部门依法处理。",
            "metadata": {"article_no": "第二十八条"},
        },
        "score": 0.4,
    }

    import asyncio

    result = asyncio.run(chunk_level_rerank(Runtime(), RerankService(), "主管部门职责", [hit, other], 2, True, query_intent="职责与权限"))

    assert result["used"] is True
    assert captured["documents"][0].startswith("[公安机关职责] 公安机关负责出租房治安管理。")
    assert "查询意图：职责与权限" in captured["documents"][0]
    assert "条款意图：职责与权限" in captured["documents"][0]
    assert "内容：公安机关负责出租房治安管理。" in captured["documents"][0]
    assert "（注：本条为第五条；文档：林芝市出租房安全管理条例）" in captured["documents"][0]


def test_chunk_level_rerank_rewards_explicit_article_match():
    captured = {}

    class RerankService:
        async def rerank(self, query, documents, top_k):
            captured["documents"] = documents
            return [{"index": 0, "score": 0.70}, {"index": 1, "score": 0.60}]

    class Runtime:
        @staticmethod
        def hit_score_mode(hit):
            return "score"

        @staticmethod
        def config_value(name, default=None):
            return True if name == "ENABLE_RERANK" else default

        @staticmethod
        def hit_metadata(hit):
            return hit["entity"].get("metadata") or {}

        @staticmethod
        def hit_entity_text(hit):
            return hit["entity"].get("text") or ""

        @staticmethod
        def hit_entity_source(hit):
            return hit["entity"].get("source") or ""

    neighbor = {
        "entity": {"source": "demo.docx", "text": "相邻条款文本", "metadata": {"article_no": "第八条"}},
        "score": 0.5,
    }
    exact = {
        "entity": {
            "source": "demo.docx",
            "text": "目标条款文本",
            "metadata": {"doc_title": "示例条例", "article_no": "第七条", "heading": "重新装修条件"},
        },
        "score": 0.5,
    }

    import asyncio

    result = asyncio.run(chunk_level_rerank(Runtime(), RerankService(), "第七条如何规定？", [neighbor, exact], 2, True))

    assert result["used"] is True
    assert captured["documents"][1].startswith("[")
    assert result["hits"][0]["entity"]["metadata"]["article_no"] == "第七条"
    assert result["hits"][0]["score"] == 0.8


def test_multi_doc_compare_selected_stage_preserves_per_target_quota():
    a_doc = _hit("A.pdf", 0.99, "A global top chunk", "A")
    b_doc = _hit("B.pdf", 0.95, "B global top chunk", "B")
    a_selected = _hit("A.pdf", 0.55, "A selected clause", "A")
    b_selected = _hit("B.pdf", 0.50, "B selected clause", "B")

    result = build_multi_doc_compare_result(
        _ResponseRuntime(),
        query="姣旇緝 A 鍜?B",
        retrieval_query="澶勭綒瑙勫畾",
        retrieval_query_raw="姣旇緝 A 鍜?B",
        dense_query="澶勭綒瑙勫畾",
        qtype="compare",
        qfilters={},
        llm_parse={},
        intent_classification={},
        is_comparison=True,
        query_route="multi_doc_compare",
        source_resolution={"reason": "compare_lock", "status": "locked"},
        compare_plan={},
        compare_source_set={"complete": True},
        compare_sources=["A.pdf", "B.pdf"],
        compare_subqueries={},
        compare_source_results=[
            {
                "source": "A.pdf",
                "docs": [a_doc],
                "selected_docs": [a_selected],
                "post_filter_docs": [a_selected],
                "retrieve_docs": [a_selected],
                "score_mode": "score",
            },
            {
                "source": "B.pdf",
                "docs": [b_doc],
                "selected_docs": [b_selected],
                "post_filter_docs": [b_selected],
                "retrieve_docs": [b_selected],
                "score_mode": "score",
            },
        ],
        requested_k=10,
        recall_k=20,
        final_n=1,
    )

    assert [doc["entity"]["text"] for doc in result["selected_docs"]] == ["A selected clause", "B selected clause"]
    assert [doc["doc_id"] for doc in result["compare_coverage"]["target_docs"]] == ["A.pdf", "B.pdf"]
    assert all(item["coverage"] == "covered" for item in result["compare_coverage"]["target_docs"])


def test_multi_doc_compare_missing_target_sets_coverage_warning_in_process_observations():
    class Evidence(_EvidenceForResponses):
        @staticmethod
        def select_process_docs(query, docs, score_mode, qfilters, final_n, intent_classification):
            return list(docs or [])[:final_n]

        @staticmethod
        def expand_docs_with_full_article_chunks(docs):
            return list(docs or [])

        async def compare_observations_async(self, query, groups, qfilters=None):
            return {
                "compare_status": "compare_ready",
                "answer_scope": "full",
                "evidence_coverage_reason": "sufficient_evidence",
                "compare_source_statuses": [
                    {"source": group.get("source") or "", "status": "answerable" if group.get("docs") else "not_found"}
                    for group in groups
                ],
            }

    class Runtime(_ResponseRuntime):
        evidence = Evidence()

    recall = {
        "query_route": "multi_doc_compare",
        "target_sources": ["A.pdf", "B.pdf"],
        "compare_source_results": [
            {"source": "A.pdf", "post_filter_docs": [_hit("A.pdf", 0.8, "A evidence")], "selected_docs": [_hit("A.pdf", 0.8, "A evidence")], "score_mode": "score"},
            {"source": "B.pdf", "post_filter_docs": [], "selected_docs": [], "score_mode": "score"},
        ],
        "selected_docs": [_hit("A.pdf", 0.8, "A evidence")],
        "score_mode": "score",
        "qfilters": {},
        "final_n": 2,
        "intent_classification": {},
        "compare_coverage": {
            "target_docs": [
                {"doc_id": "A.pdf", "retrieved": 1, "selected": 1, "coverage": "covered"},
                {"doc_id": "B.pdf", "retrieved": 0, "selected": 0, "coverage": "missing"},
            ],
            "any_doc_missing": True,
            "any_doc_insufficient": True,
        },
    }

    import asyncio

    result = asyncio.run(prepare_process_evidence_context(Runtime(), "姣旇緝 A 鍜?B", recall, []))

    assert result["observations"]["compare_status"] == "partial_sources_missing"
    assert result["observations"]["answer_scope"] == "refusal"
    assert result["observations"]["compare_coverage"]["target_docs"][1]["coverage"] == "missing"


def test_multi_doc_compare_process_quota_rescues_retrieve_only_target():
    class Evidence(_EvidenceForResponses):
        @staticmethod
        def select_process_docs(query, docs, score_mode, qfilters, final_n, intent_classification):
            return list(docs or [])[:final_n]

        @staticmethod
        def expand_docs_with_full_article_chunks(docs):
            return list(docs or [])

        @staticmethod
        def dedupe_docs(docs):
            return list(docs or [])

        async def compare_observations_async(self, query, groups, qfilters=None):
            return {
                "compare_status": "compare_ready",
                "answer_scope": "full",
                "evidence_coverage_reason": "sufficient_evidence",
                "compare_source_statuses": [
                    {"source": group.get("source") or "", "status": "answerable" if group.get("docs") else "not_found"}
                    for group in groups
                ],
            }

    class Runtime(_ResponseRuntime):
        evidence = Evidence()

    a_doc = _hit("A.pdf", 0.8, "A selected")
    b_doc = _hit("B.pdf", 0.7, "B retrieve only")
    recall = {
        "query_route": "multi_doc_compare",
        "target_sources": ["A.pdf", "B.pdf"],
        "compare_source_results": [
            {
                "source": "A.pdf",
                "post_filter_docs": [a_doc],
                "selected_docs": [a_doc],
                "retrieve_docs": [a_doc],
                "score_mode": "score",
            },
            {
                "source": "B.pdf",
                "post_filter_docs": [],
                "selected_docs": [],
                "retrieve_docs": [b_doc],
                "docs": [b_doc],
                "score_mode": "score",
            },
        ],
        "selected_docs": [a_doc],
        "score_mode": "score",
        "qfilters": {},
        "final_n": 1,
        "intent_classification": {},
        "compare_coverage": {
            "min_required_per_doc": 1,
            "target_docs": [
                {"doc_id": "A.pdf", "source": "A.pdf", "retrieved": 1, "selected": 1, "coverage": "covered"},
                {"doc_id": "B.pdf", "source": "B.pdf", "retrieved": 1, "selected": 0, "coverage": "insufficient"},
            ],
            "any_doc_missing": False,
            "any_doc_insufficient": True,
        },
    }

    import asyncio

    result = asyncio.run(prepare_process_evidence_context(Runtime(), "比较 A 和 B", recall, []))

    assert [doc["entity"]["source"] for doc in result["process_docs"]] == ["A.pdf", "B.pdf"]
    coverage = result["observations"]["compare_coverage"]["target_docs"]
    assert [item["coverage"] for item in coverage] == ["covered", "covered"]
    assert result["observations"]["answer_scope"] == "full"


def test_multi_doc_compare_process_keeps_pinned_selected_clause_before_rerank_choice():
    class Evidence(_EvidenceForResponses):
        @staticmethod
        def select_process_docs(query, docs, score_mode, qfilters, final_n, intent_classification):
            return [doc for doc in docs if doc["entity"].get("text") == "A process favorite"][:final_n]

        @staticmethod
        def expand_docs_with_full_article_chunks(docs):
            return list(docs or [])

        async def compare_observations_async(self, query, groups, qfilters=None):
            return {
                "compare_status": "compare_ready",
                "answer_scope": "full",
                "evidence_coverage_reason": "sufficient_evidence",
            }

    class Runtime(_ResponseRuntime):
        evidence = Evidence()

    pinned = _hit("A.pdf", 0.9, "A pinned clause")
    favorite = _hit("A.pdf", 0.8, "A process favorite")
    recall = {
        "query_route": "multi_doc_compare",
        "target_sources": ["A.pdf"],
        "compare_source_results": [
            {
                "source": "A.pdf",
                "post_filter_docs": [favorite],
                "selected_docs": [pinned],
                "retrieve_docs": [pinned, favorite],
                "score_mode": "score",
            }
        ],
        "selected_docs": [pinned],
        "score_mode": "score",
        "qfilters": {},
        "final_n": 1,
        "intent_classification": {},
        "compare_coverage": {
            "min_required_per_doc": 1,
            "target_docs": [{"doc_id": "A.pdf", "source": "A.pdf", "retrieved": 2, "selected": 1, "coverage": "covered"}],
        },
    }

    import asyncio

    result = asyncio.run(prepare_process_evidence_context(Runtime(), "比较 A", recall, []))

    assert result["process_docs"][0]["entity"]["text"] == "A pinned clause"
    assert result["process_docs"][0]["entity"]["metadata"]["is_pinned"] is True


def test_not_found_source_state_renders_document_not_found_on_retrieve_and_process_paths():
    recall = {
        "source_lock_required": True,
        "resolved_source_lock": False,
        "source_resolution_status": "not_found",
        "source_lock_reason": "document_target_required",
        "target_text": "涓嶅瓨鍦ㄦ枃浠?pdf",
        "query_route": "explicit_doc_reference",
        "question_type": "general",
        "source_resolution_trace": {},
    }

    retrieve = build_retrieve_source_lock_result(_ResponseRuntime(), "涓嶅瓨鍦ㄦ枃浠剁鍗佷簩鏉℃€庝箞瑙勫畾", "u", recall)
    process = build_process_source_lock_result(_ResponseRuntime(), "涓嶅瓨鍦ㄦ枃浠剁鍗佷簩鏉℃€庝箞瑙勫畾", "u", "general", recall)

    assert retrieve["metadata"]["query_route"] == "document_not_found"
    assert retrieve["metadata"]["final_channel"] == "document_not_found"
    assert retrieve["metadata"]["refusal_reason"] == "document_not_found"
    assert process["metadata"]["query_route"] == "document_not_found"
    assert process["metadata"]["final_channel"] == "document_not_found"
    assert process["metadata"]["refusal_reason"] == "document_not_found"
    assert "forced_retrieval_fallback" not in recall["source_resolution_trace"]


def test_global_fallback_is_the_only_state_that_allows_forced_retrieval():
    result = prepare_recall_source_context(
        _Runtime(),
        "\u7b2c\u5341\u4e8c\u6761\u600e\u4e48\u89c4\u5b9a",
        "general",
        {},
        {},
        {
            "required": False,
            "resolved": False,
            "status": "global_fallback",
            "reason": "not_needed",
            "sources": [],
            "candidates": [],
        },
        "document_clarification",
        None,
        {"quality": "valid"},
        "tier_2",
        filename_hints=[],
        user_id="u",
    )

    assert result["early_return"] is None
    assert result["query_route"] == "content_qa"
    assert result["active_fnames"] == []
    assert result["source_resolution"]["status"] == "global_fallback"
    assert result["source_resolution"]["source_resolution_trace"]["forced_retrieval_fallback"] is True


def test_single_doc_compare_does_not_preemptively_clarify_or_clear_sources():
    class Compare(_Compare):
        @staticmethod
        def has_intent(query):
            return True

        @staticmethod
        def source_set_completeness(compare_plan, active_fnames):
            return {
                "complete": False,
                "expected_target_count": 1,
                "resolved_source_count": len(active_fnames or []),
                "sources": list(active_fnames or []),
                "missing_targets": [],
            }

    class Runtime(_Runtime):
        compare = Compare()

    result = prepare_recall_source_context(
        Runtime(),
        "长春森林防火期和森林防火戒严期分别是什么时间？",
        "compare",
        {"is_comparison": True},
        {},
        {
            "route": "single_doc_compare",
            "required": False,
            "resolved": False,
            "status": "global_fallback",
            "reason": "agentic_router_single_doc_plan_ready",
            "sources": [],
            "candidates": [],
        },
        "single_doc_compare",
        True,
        {"quality": "valid"},
        "tier_2",
        filename_hints=[],
        user_id="u",
    )

    assert result["early_return"] is None
    assert result["query_route"] == "single_doc_compare"
    assert result["source_resolution"]["route"] == "single_doc_compare"
    assert result["active_fnames"] == []


def test_compare_source_set_requires_all_expected_targets():
    result = compare_source_set_completeness(
        {
            "doc_like_subjects": ["A", "B", "C"],
            "missing_doc_targets": ["C"],
        },
        ["a.docx", "b.docx"],
        lambda value: str(value or ""),
    )

    assert result["complete"] is False
    assert result["expected_target_count"] == 3
    assert result["resolved_source_count"] == 2
    assert result["missing_targets"] == ["C"]


def test_title_candidates_are_hints_not_hard_filters_for_open_topic():
    class Config:
        ENABLE_COMPARE_INTENT_TAG = False
        ENABLE_LLM_QUERY_PARSE = False
        MIN_QUERY_CHARS = 2

    class Source:
        @staticmethod
        def extract_title_candidates(query):
            return ["wrong-a.pdf", "wrong-b.pdf"]

    class Retrieval(_Retrieval):
        @staticmethod
        def strip_source_title_mentions(query, sources):
            return str(query or "")

        @staticmethod
        def expand_from_corpus(query, retrieval_query):
            return retrieval_query, []

    class Runtime(_Runtime):
        config = Config()
        source = Source()
        retrieval = Retrieval()

    import asyncio

    result = asyncio.run(
        prepare_retrieval_query_context(
            Runtime(),
            "10月20日在长春林区野外用火规则",
            "general",
            {},
            {},
            {
                "required": False,
                "resolved": False,
                "status": "global_fallback",
                "reason": "not_needed",
                "sources": [],
                "candidates": [],
            },
            "open_regulation_qa",
            None,
            False,
            {"quality": "valid"},
            "tier_2",
            {},
            {},
            3,
            [],
            [],
            False,
            set(),
        )
    )

    assert result["active_fnames"] == []
    assert result["qfilters"]["_candidate_hint_sources"] == ["wrong-a.pdf", "wrong-b.pdf"]
    assert result["qfilters"]["_soft_source_scope"] is True
    assert result["source_resolution"]["source_resolution_trace"]["title_candidates_as_hints"] is True
    assert result["source_resolution"]["source_resolution_trace"]["soft_source_scope"] is True


def test_candidate_hint_sources_add_supplemental_recall_without_hard_filter():
    class VectorDb:
        def __init__(self):
            self.filters = []

        def search(self, embedding, top_k, filters=None, query_sparse_embedding=None):
            self.filters.append(filters)
            if filters:
                return [_hit("wrong-a.pdf", 0.8, "hint dense")]
            return [_hit("global.pdf", 0.7, "global dense")]

    class Handler:
        vector_db = VectorDb()

    class Runtime:
        @staticmethod
        def normalize_filename_for_match(value):
            return str(value or "").strip()

        @staticmethod
        def filter_hits_by_source_state(docs):
            return {"hits": list(docs or []), "dropped": 0, "states": {}}

        @staticmethod
        def dense_source_score_map(docs):
            return {doc["entity"]["source"]: doc.get("score", 0) for doc in docs or []}

        @staticmethod
        def collect_lexical_candidates(query, safe_names, doc_recall_plan, article_ids=None):
            if safe_names:
                return [_hit(safe_names[0], 0.9, "hint lexical")]
            return [_hit("global.pdf", 0.6, "global lexical")]

    result = retrieval_core.run_lightweight_search_candidates(
        Runtime(),
        Handler(),
        [0.1, 0.2],
        "测试查询",
        [],
        5,
        qfilters={"_candidate_hint_sources": ["wrong-a.pdf"], "_soft_source_scope": True},
    )

    assert result["milvus_filter"] is None
    assert Handler.vector_db.filters[0] is None
    assert "wrong-a.pdf" in Handler.vector_db.filters[1]
    assert [doc["entity"]["source"] for doc in result["docs"]] == ["global.pdf", "wrong-a.pdf"]
    assert [doc["entity"]["source"] for doc in result["lex_items"]] == ["global.pdf", "wrong-a.pdf"]


def test_intra_doc_chunk_rerank_preserves_global_source_interleaving():
    class Runtime:
        @staticmethod
        def infer_rerank_profile(query, qtype):
            return "balanced"

        @staticmethod
        def normalize_filename_for_match(value):
            return str(value or "").strip()

        @staticmethod
        def hit_entity_source(hit):
            return hit["entity"]["source"]

        @staticmethod
        def hit_entity_text(hit):
            return hit["entity"].get("text") or ""

        @staticmethod
        def hit_display_text(hit):
            return hit["entity"].get("text") or ""

        @staticmethod
        def hit_metadata(hit):
            return hit["entity"].get("metadata") or {}

        @staticmethod
        def hit_score(hit):
            return hit.get("score", 0.0)

        @staticmethod
        def chunk_position_id(hit):
            return (hit["entity"].get("metadata") or {}).get("chunk_id")

        @staticmethod
        def query_anchor_terms(query):
            return []

        @staticmethod
        def doc_title_alias_score(src, query):
            return 0.0

        @staticmethod
        def token_overlap_score(query, text):
            return 0.0

        @staticmethod
        def clip01(value):
            return max(0.0, min(1.0, float(value or 0.0)))

        @staticmethod
        def rerank_profile_weights(profile):
            return {
                "section_term": 0.0,
                "text_term": 0.0,
                "section_overlap": 0.0,
                "keyword": 0.0,
                "title": 0.0,
                "base": 1.0,
            }

        @staticmethod
        def section_follow_bonus(section, pos, section_anchor_positions, profile):
            return 0.0

        @staticmethod
        def generic_chunk_penalty(section, text, query, text_term_hits, section_term_hits, section_score, profile):
            return 0.0

        @staticmethod
        def section_target_alignment(section, query):
            return (0.0, 0.0)

        @staticmethod
        def query_semantic_aspects(query, qfilters=None):
            return {"terms": []}

        @staticmethod
        def normalize_topics(value):
            return []

        @staticmethod
        def config_value(name, default=None):
            return default

        @staticmethod
        def clone_hit_with_score(hit, score):
            out = {
                "entity": {
                    "source": hit["entity"]["source"],
                    "text": hit["entity"].get("text") or "",
                    "metadata": dict(hit["entity"].get("metadata") or {}),
                },
                "score": score,
            }
            return out

    hits = [
        _hit("A.pdf", 0.9, "A first", doc_id="a1"),
        _hit("B.pdf", 0.8, "B target", doc_id="b1"),
        _hit("A.pdf", 0.7, "A second", doc_id="a2"),
    ]

    result = intra_doc_chunk_rerank(Runtime(), "职责", hits, "score", qtype="general", qfilters={})

    assert [doc["entity"]["source"] for doc in result] == ["A.pdf", "B.pdf", "A.pdf"]


def test_intra_doc_chunk_rerank_is_order_preserving_by_default():
    class Runtime:
        @staticmethod
        def config_value(name, default=None):
            return default

    hits = [
        _hit("demo.pdf", 0.1, "弱匹配但上游第一", doc_id="a1"),
        _hit("demo.pdf", 0.9, "职责 负责 管理 监督", doc_id="a2"),
    ]

    result = intra_doc_chunk_rerank(Runtime(), "职责", hits, "score", qtype="general", qfilters={})

    assert result is hits
    assert [doc["entity"]["text"] for doc in result] == ["弱匹配但上游第一", "职责 负责 管理 监督"]


def test_merge_and_dedupe_hits_preserves_upstream_order_when_scores_tie():
    class Runtime:
        @staticmethod
        def hit_entity_source(hit):
            return hit["entity"]["source"]

        @staticmethod
        def hit_metadata(hit):
            return hit["entity"].get("metadata") or {}

        @staticmethod
        def hit_chunk_id(hit):
            return (hit["entity"].get("metadata") or {}).get("chunk_id")

        @staticmethod
        def hit_entity_text(hit):
            return hit["entity"].get("text") or ""

        @staticmethod
        def hit_score(hit):
            return hit.get("score", 0.0)

        @staticmethod
        def config_value(name, default=None):
            return default

    def hit(article_no, chunk_id):
        return {
            "entity": {
                "source": "demo.pdf",
                "text": f"{article_no} 内容",
                "metadata": {
                    "chunk_id": chunk_id,
                    "article_no": article_no,
                    "section": "同一章节",
                },
            },
            "score": 0.5,
        }

    result = merge_and_dedupe_hits(
        Runtime(),
        [hit("第三条", 3), hit("第一条", 1), hit("第二条", 2)],
        "score",
    )

    assert [doc["entity"]["metadata"].get("article_no") for doc in result[:3]] == ["第三条", "第一条", "第二条"]


def test_merge_and_dedupe_hits_keeps_same_text_from_different_sources():
    class Runtime:
        @staticmethod
        def hit_entity_source(hit):
            return hit["entity"]["source"]

        @staticmethod
        def hit_metadata(hit):
            return hit["entity"].get("metadata") or {}

        @staticmethod
        def hit_chunk_id(hit):
            return (hit["entity"].get("metadata") or {}).get("chunk_id")

        @staticmethod
        def hit_entity_text(hit):
            return hit["entity"].get("text") or ""

        @staticmethod
        def hit_score(hit):
            return hit.get("score", 0.0)

        @staticmethod
        def config_value(name, default=None):
            if name == "MAX_MERGED_CHUNKS_PER_EVIDENCE":
                return 1
            return default

    def hit(source):
        return {
            "entity": {
                "source": source,
                "text": "第五条 同一条文内容",
                "metadata": {"chunk_id": 5, "article_no": "第五条", "section": "第五条"},
            },
            "score": 0.5,
        }

    result = merge_and_dedupe_hits(
        Runtime(),
        [hit("demo.docx"), hit("demo.pdf")],
        "score",
    )

    assert [doc["entity"]["source"] for doc in result] == ["demo.docx", "demo.pdf"]


def test_pinned_clause_selection_rescues_from_source_pool():
    class Runtime:
        common = _Common()

    ordinary = {
        "entity": {
            "source": "demo.docx",
            "text": "第三十六条 其他内容",
            "metadata": {"article_no": "第三十六条", "chunk_id": 36},
        },
        "score": 1.2,
    }
    rescued = {
        "entity": {
            "source": "demo.docx",
            "text": "第二条 本市建筑废弃物的减排与回收利用及其监督管理适用本条例。",
            "metadata": {"article_no": "第二条", "chunk_id": 2},
        },
        "score": 0.8,
    }

    selected = _select_with_pinned_clauses(
        Runtime(),
        [ordinary],
        {"_pinned_article_ids": ["第二条"]},
        "适用范围是什么？",
        final_n=2,
        pinned_source_docs=[rescued],
    )

    assert selected[0]["entity"]["metadata"]["article_no"] == "第二条"
    assert selected[1]["entity"]["metadata"]["article_no"] == "第三十六条"


def test_pinned_clause_rescue_does_not_push_existing_relevant_doc_out_of_top_five():
    class Runtime:
        common = _Common()

    docs = [
        {"entity": {"source": "demo.docx", "text": "第十条 免疫规则", "metadata": {"article_no": "第十条", "chunk_id": 10}}, "score": 1.0},
        {"entity": {"source": "demo.docx", "text": "第十一条 禁养规则", "metadata": {"article_no": "第十一条", "chunk_id": 11}}, "score": 0.99},
        {"entity": {"source": "demo.docx", "text": "第十二条 其他", "metadata": {"article_no": "第十二条", "chunk_id": 12}}, "score": 0.98},
        {"entity": {"source": "demo.docx", "text": "第十三条 其他", "metadata": {"article_no": "第十三条", "chunk_id": 13}}, "score": 0.97},
        {"entity": {"source": "demo.docx", "text": "第二十一条 其他", "metadata": {"article_no": "第二十一条", "chunk_id": 21}}, "score": 0.96},
    ]
    rescue = [
        {"entity": {"source": "demo.docx", "text": "第十四条 rescue", "metadata": {"article_no": "第十四条", "chunk_id": 14}}, "score": 1.3},
        {"entity": {"source": "demo.docx", "text": "第十五条 rescue", "metadata": {"article_no": "第十五条", "chunk_id": 15}}, "score": 1.2},
    ]

    selected = _select_with_pinned_clauses(
        Runtime(),
        docs,
        {"_pinned_article_ids": ["第十条", "第十四条", "第十五条"]},
        "免疫、芯片和禁养规则",
        final_n=5,
        pinned_source_docs=rescue,
    )

    top_five_articles = [item["entity"]["metadata"]["article_no"] for item in selected[:5]]
    assert "第十一条" in top_five_articles


def test_source_pin_rescues_latest_equivalent_source_from_pool():
    class Source:
        @staticmethod
        def latest_effective_equivalent_source(source):
            return "target_2024.docx" if source == "target_2004.pdf" else source

    class Runtime:
        common = _Common()
        source = Source()

    wrong = {"entity": {"source": "other.pdf", "text": "强关键词错误来源", "metadata": {"chunk_id": 1}}, "score": 1.3}
    target = {"entity": {"source": "target_2024.docx", "text": "目标法规新版内容", "metadata": {"chunk_id": 2}}, "score": 0.8}

    selected = _select_with_pinned_clauses(
        Runtime(),
        [wrong],
        {},
        "目标法规问题",
        final_n=2,
        pinned_source_docs=[target],
        pinned_sources=["target_2004.pdf"],
    )

    assert selected[0]["entity"]["source"] == "target_2024.docx"
    assert selected[1]["entity"]["source"] == "other.pdf"


def test_source_pin_ignores_plain_target_sources_candidates():
    class Runtime:
        common = _Common()

    wrong = {"entity": {"source": "wrong.pdf", "text": "错误候选", "metadata": {"chunk_id": 1}}, "score": 1.3}
    target = {"entity": {"source": "target.docx", "text": "目标内容", "metadata": {"chunk_id": 2}}, "score": 0.8}

    selected = _select_with_pinned_clauses(
        Runtime(),
        [wrong],
        {"target_sources": ["target.docx"]},
        "开放问题",
        final_n=2,
        pinned_source_docs=[target],
    )

    assert selected[0]["entity"]["source"] == "wrong.pdf"


def test_chunk_rerank_adds_freshness_reward_for_newer_equivalent_source():
    class Source:
        @staticmethod
        def regulation_identity_key(source):
            return "city|forest_rules"

        @staticmethod
        def source_effective_rank(source):
            if source == "forest_2024.docx":
                return (0, 20240402, 20240402, 1, source)
            return (0, 20040819, 20040819, 1, source)

    class Runtime:
        common = _Common()
        source = Source()

        @staticmethod
        def config_value(name, default=None):
            if name == "ENABLE_RERANK":
                return True
            if name == "RERANK_VERSION_FRESHNESS_REWARD":
                return 0.04
            return default

        @staticmethod
        def hit_score_mode(hit):
            return "score"

        @staticmethod
        def hit_metadata(hit):
            return hit["entity"].get("metadata") or {}

        @staticmethod
        def hit_entity_source(hit):
            return hit["entity"].get("source") or ""

        @staticmethod
        def hit_entity_text(hit):
            return hit["entity"].get("text") or ""

    class Reranker:
        async def rerank(self, query, documents, top_k):
            return [{"index": 0, "score": 0.99}, {"index": 1, "score": 0.99}]

    import asyncio

    old = {"entity": {"source": "forest_2004.pdf", "text": "第二十七条 使用林地审批", "metadata": {"article_no": "第二十七条"}}, "score": 0.8}
    new = {"entity": {"source": "forest_2024.docx", "text": "第二十七条 使用林地审批", "metadata": {"article_no": "第二十七条"}}, "score": 0.8}
    result = asyncio.run(chunk_level_rerank(Runtime(), Reranker(), "使用林地审批", [old, new], 2, True))

    assert result["hits"][0]["entity"]["source"] == "forest_2024.docx"


def test_chunk_rerank_freshness_reward_falls_back_to_filename_family():
    class Source:
        @staticmethod
        def regulation_identity_key(source):
            return f"source:{source}"

        @staticmethod
        def canonical_doc_id(source):
            return f"source:{source}"

        @staticmethod
        def source_effective_rank(source):
            if "2024-04-02" in source:
                return (0, 20240402, 20240402, 1, source)
            return (0, 20040819, 20040819, 1, source)

    class Runtime:
        common = _Common()
        source = Source()

        @staticmethod
        def config_value(name, default=None):
            if name == "ENABLE_RERANK":
                return True
            if name == "RERANK_VERSION_FRESHNESS_REWARD":
                return 0.04
            return default

        @staticmethod
        def hit_score_mode(hit):
            return "score"

        @staticmethod
        def hit_metadata(hit):
            return hit["entity"].get("metadata") or {}

        @staticmethod
        def hit_entity_source(hit):
            return hit["entity"].get("source") or ""

        @staticmethod
        def hit_entity_text(hit):
            return hit["entity"].get("text") or ""

    class Reranker:
        async def rerank(self, query, documents, top_k):
            return [{"index": 0, "score": 0.99}, {"index": 1, "score": 0.99}]

    import asyncio

    old = {"entity": {"source": "长春市森林资源管理条例_2004-08-19_2004-08-19.pdf", "text": "第二十七条 使用林地审批", "metadata": {"article_no": "第二十七条"}}, "score": 0.8}
    new = {"entity": {"source": "长春市森林资源管理条例_2024-04-02_2024-04-02.docx", "text": "第二十七条 使用林地审批", "metadata": {"article_no": "第二十七条"}}, "score": 0.8}
    result = asyncio.run(chunk_level_rerank(Runtime(), Reranker(), "使用林地审批", [old, new], 2, True))

    assert result["hits"][0]["entity"]["source"] == "长春市森林资源管理条例_2024-04-02_2024-04-02.docx"


def test_router_target_failure_global_fallback_does_not_require_source_lock():
    source_resolution = {
        "required": False,
        "resolved": False,
        "status": "global_fallback",
        "reason": "agentic_router_targets_not_found",
        "source_lock_kind": "agentic_compare_lock",
        "source_resolution_trace": {"router_target_failure_global_fallback": True},
    }

    result = handle_required_source_lock(
        _Runtime(),
        "林芝出租房公安机关职责是什么",
        "林芝出租房公安机关职责是什么",
        "general",
        {},
        {},
        source_resolution,
        "content_qa",
        None,
        True,
        {"quality": "valid"},
        "tier_2",
        {"is_compare": True},
        3,
        [],
        [],
        False,
        user_id="u",
    )

    assert result["early_return"] is None
    assert result["source_resolution"]["status"] == "global_fallback"
    assert source_resolution_router_target_failure_fallback(result["source_resolution"]) is True


def test_router_target_failure_required_state_is_degraded_before_source_lock_early_return():
    result = handle_required_source_lock(
        _Runtime(),
        "林芝出租房公安机关职责是什么",
        "林芝出租房公安机关职责是什么",
        "general",
        {},
        {},
        {
            "required": True,
            "resolved": False,
            "status": "not_found",
            "reason": "agentic_router_targets_not_found",
            "route": "compare_target_not_found",
            "source_lock_kind": "agentic_compare_lock",
            "source_resolution_trace": {"agentic_router": {"used": True}},
        },
        "document_clarification",
        None,
        True,
        {"quality": "valid"},
        "tier_2",
        {"is_compare": True},
        3,
        [],
        [],
        False,
        user_id="u",
    )

    assert result["early_return"] is None
    assert result["query_route"] == "content_qa"
    assert result["source_resolution"]["required"] is False
    assert result["source_resolution"]["status"] == "global_fallback"
    assert result["source_resolution"]["source_resolution_trace"]["router_target_failure_global_fallback"] is True


def test_router_target_failure_global_fallback_low_score_requests_clarification():
    source_resolution = {
        "required": False,
        "resolved": False,
        "status": "global_fallback",
        "reason": "agentic_router_targets_not_found",
        "source_lock_kind": "agentic_compare_lock",
        "source_resolution_trace": {"router_target_failure_global_fallback": True},
    }

    result = build_lightweight_recall_result(
        _ResponseRuntime(),
        "林芝出租房公安机关职责是什么",
        "林芝出租房公安机关职责是什么",
        "林芝出租房公安机关职责是什么",
        "林芝出租房公安机关职责是什么",
        "general",
        {},
        {},
        {},
        True,
        "content_qa",
        [_hit("A.docx", score=0.2)],
        {"hits": [_hit("A.docx", score=0.2)], "dropped": 0, "states": {}},
        {"hits": [], "dropped": 0, "states": {}},
        [_hit("A.docx", score=0.2)],
        [_hit("A.docx", score=0.2)],
        [_hit("A.docx", score=0.2)],
        {},
        "score",
        {"used": False},
        10,
        3,
        False,
        source_resolution,
        [],
        False,
        {"is_compare": True},
        "tier_2",
    )

    assert result["soft_clarification_required"] is True
    assert result["soft_clarification_reason"] == "global_fallback_low_similarity"
    assert result["source_resolution_trace"]["global_fallback_best_similarity"] == 0.2


def test_router_target_failure_global_fallback_high_score_keeps_retrieval():
    source_resolution = {
        "required": False,
        "resolved": False,
        "status": "global_fallback",
        "reason": "agentic_router_targets_not_found",
        "source_lock_kind": "agentic_compare_lock",
        "source_resolution_trace": {"router_target_failure_global_fallback": True},
    }

    result = build_lightweight_recall_result(
        _ResponseRuntime(),
        "林芝出租房公安机关职责是什么",
        "林芝出租房公安机关职责是什么",
        "林芝出租房公安机关职责是什么",
        "林芝出租房公安机关职责是什么",
        "general",
        {},
        {},
        {},
        True,
        "content_qa",
        [_hit("A.docx", score=0.8)],
        {"hits": [_hit("A.docx", score=0.8)], "dropped": 0, "states": {}},
        {"hits": [], "dropped": 0, "states": {}},
        [_hit("A.docx", score=0.8)],
        [_hit("A.docx", score=0.8)],
        [_hit("A.docx", score=0.8)],
        {},
        "score",
        {"used": False},
        10,
        3,
        False,
        source_resolution,
        [],
        False,
        {"is_compare": True},
        "tier_2",
    )

    assert result["soft_clarification_required"] is False
    assert result["source_resolution_trace"]["global_fallback_low_score"] is False


def test_delayed_source_global_fallback_low_score_requests_clarification():
    source_resolution = {
        "required": False,
        "resolved": False,
        "status": "global_fallback",
        "reason": "document_target_required",
        "source_resolution_trace": {"delayed_clarification_global_fallback": True},
    }

    result = build_lightweight_recall_result(
        _ResponseRuntime(),
        "article 12 requirements",
        "article 12 requirements",
        "article 12 requirements",
        "article 12 requirements",
        "general",
        {},
        {},
        {},
        False,
        "content_qa",
        [_hit("A.docx", score=0.1)],
        {"hits": [_hit("A.docx", score=0.1)], "dropped": 0, "states": {}},
        {"hits": [], "dropped": 0, "states": {}},
        [_hit("A.docx", score=0.1)],
        [_hit("A.docx", score=0.1)],
        [_hit("A.docx", score=0.1)],
        {},
        "score",
        {"used": False},
        10,
        3,
        False,
        source_resolution,
        [],
        False,
        {},
        "tier_2",
    )

    assert result["soft_clarification_required"] is True
    assert result["soft_clarification_reason"] == "global_fallback_low_similarity"
    assert result["source_resolution_trace"]["global_fallback_best_similarity"] == 0.1


def test_invalid_parsed_article_filter_is_dropped():
    qfilters = drop_invalid_parsed_article_filter(
        {"doc_type": None, "article_id": "\u7b2c\u5341\u4e94\u6761"},
        {"article_id": "\u6df1\u5733\u5efa\u8bbe\u5de5\u7a0b\u8d28\u91cf\u7ba1\u7406\u6761\u4f8b\u7b2c\u5341\u4e94\u6761"},
    )

    assert qfilters["_skip_article_id_filter"] is True
    assert "article_id" not in qfilters


def test_build_lightweight_recall_result_accepts_dropped_counts():
    result = build_lightweight_recall_result(
        runtime=None,
        query="q",
        retrieval_query="q",
        retrieval_query_raw="q",
        dense_query="q",
        qtype="general",
        qfilters={},
        llm_parse={},
        intent_classification={},
        is_comparison=False,
        query_route="normal",
        docs=[],
        visible_dense={"hits": [], "dropped": 2, "states": {}},
        visible_lex={"hits": [], "dropped": 3, "states": {}},
        selected_docs=[],
        post_filter_docs=[],
        retrieve_docs=[],
        dense_source_scores={},
        score_mode="score",
        reranked_chunk={"used": False},
        recall_k=10,
        final_n=5,
        weak_query=False,
        source_resolution={},
        active_fnames=[],
        topical_multi_doc_mode=False,
        compare_plan={},
        intent_tier="normal",
    )

    assert result["early_filtered"] == 5
    assert result["visibility_filtered"] == 5


def test_prepare_retrieve_context_filters_heading_without_evidence_gate():
    class Common:
        @staticmethod
        def normalize_filename(value):
            return str(value or "").strip()

    class Evidence:
        @staticmethod
        def select_retrieve_docs(docs, top_k, default_n):
            return list(docs or [])[:top_k]

        @staticmethod
        def merge_compare_source_doc_groups(groups, per_source_limit):
            docs = []
            for group in groups or []:
                docs.extend(list(group.get("docs") or [])[:per_source_limit])
            return docs

        async def observations_async(self, *args, **kwargs):
            raise AssertionError("retrieve must not run evidence gate")

    class Runtime:
        common = Common()
        evidence = Evidence()

    heading = {
        "entity": {
            "text": "第一章 总则",
            "metadata": {"chunk_role": "chapter_heading", "chunk_id": 1},
        }
    }
    body = {
        "entity": {
            "text": "违反规定的，由主管部门责令改正，并处罚款。",
            "metadata": {"chunk_role": "body", "chunk_id": 2},
        }
    }
    recall = {
        "target_sources": ["demo.pdf"],
        "query_route": "explicit_regulation_reference",
        "retrieve_docs": [heading, body],
        "post_filter_docs": [],
        "selected_docs": [],
        "final_n": 5,
        "qfilters": {},
    }

    import asyncio

    result = asyncio.run(prepare_retrieve_evidence_context(Runtime(), "处罚规定", recall, [], 10))

    assert result["retrieve_docs"] == [body]
    assert result["observations"]["retrieve_gate_disabled"] is True
    assert result["observations"]["retrieve_filter"] == "heading_only"


def test_select_retrieve_output_docs_dedupes_same_article():
    first = {
        "entity": {
            "source": "demo.pdf",
            "text": "第三十八条 第一款",
            "metadata": {"article_id": "第三十八条", "chunk_id": 10},
        },
        "score": 0.9,
    }
    duplicate = {
        "entity": {
            "source": "demo.pdf",
            "text": "第三十八条 第二款",
            "metadata": {"article_id": "第三十八条", "chunk_id": 11},
        },
        "score": 0.8,
    }
    other = {
        "entity": {
            "source": "demo.pdf",
            "text": "第三十九条",
            "metadata": {"article_id": "第三十九条", "chunk_id": 12},
        },
        "score": 0.7,
    }

    result = select_retrieve_output_docs([first, duplicate, other], top_k=10, default_n=10)

    assert result == [first, other]
