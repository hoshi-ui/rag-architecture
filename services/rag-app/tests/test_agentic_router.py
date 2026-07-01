import asyncio

from app.core.query import agentic_router
from app.core.query.process import prepare_lightweight_recall_prelude


class _Config:
    ENABLE_AGENTIC_ROUTER = True
    AGENTIC_ROUTER_MAX_TOKENS = 520
    AGENTIC_ROUTER_TIMEOUT = 8
    AGENTIC_ROUTER_MIN_CONFIDENCE = 0.62
    ENABLE_LLM_TOOL_ROUTER = False
    ENABLE_LLM_QUERY_PARSE = False


class _FakeClient:
    def __init__(self):
        self.payload = None
        self.timeout = None

    def available(self):
        return True

    def build_payload(self, system_prompt, user_prompt, **kwargs):
        self.payload = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            **kwargs,
        }
        return {"messages": []}

    async def chat_text(self, payload, *, timeout=None):
        self.timeout = timeout
        return """
        {
          "route": "multi_doc_compare",
          "question_type": "compare",
          "is_comparison": true,
          "is_multi_doc_compare": true,
          "documents": ["Dog Rules", "Property Rules"],
          "common_aspects": ["penalty", "liability"],
          "sub_queries": [
            {
              "source": "Dog Rules",
              "raw_text_query": "dog violations penalty fine",
              "section_query": "legal liability penalty",
              "doc_prior_query": "Dog Rules"
            },
            {
              "source": "Property Rules",
              "raw_text_query": "property management violations penalty fine",
              "section_query": "legal liability penalty",
              "doc_prior_query": "Property Rules"
            }
          ],
          "missing_targets": [],
          "requires_clarification": false,
          "rationale": "compare two regulations",
          "confidence": 0.91
        }
        """


class _Common:
    @staticmethod
    def normalize_query(value):
        return " ".join(str(value or "").split()).strip()

    @staticmethod
    def normalize_filename(value):
        return str(value or "").strip()


class _Compare:
    _source_map = {
        "Dog Rules": "dog_rules_2020.docx",
        "Property Rules": "property_rules_2023.pdf",
    }

    def resolve_subject_source(self, subject):
        source = self._source_map.get(subject, "")
        return {
            "subject": subject,
            "source": source,
            "match_kind": "test_exact" if source else "",
            "doc_like": bool(source),
            "prior": 1.0 if source else 0.0,
        }

    @staticmethod
    def has_intent(query):
        return "compare" in str(query or "").lower()


class _Source:
    _title_map = {
        "dog_rules_2020.docx": "Dog Rules",
        "property_rules_2023.pdf": "Property Rules",
    }

    def display_title(self, source):
        return self._title_map.get(source, source)

    def extract_title_candidates(self, target, limit=3):
        return []

    def strong_title_source_matches(self, target, limit=3):
        return []

    def validate_source_lock_candidate(self, query, target_text, source, *, prior=0.0, match_kind=""):
        return {"accepted": True, "score": 1.0, "reasons": ["test_accept"]}

    def source_core_entities(self, source):
        if source == "cement_rules_2018.pdf":
            return ["Changchun"]
        if source == "forest_rules_2024.docx":
            return ["Changchun"]
        return []

    def resolve_targets(self, query, fnames=None, user_id="anonymous"):
        return self.build_source_resolution_result(
            route="content_qa",
            required=False,
            resolved=False,
            sources=[],
            candidates=[],
            reason="fallback_rule_route",
        )

    @staticmethod
    def build_source_resolution_result(
        *,
        route,
        required,
        resolved,
        sources,
        candidates,
        reason,
        strip_title_mentions=False,
        clarification="",
        target_text="",
        lock_mode="none",
        lock_confidence=0.0,
        lock_message_prefix="",
        source_lock_kind="",
        source_resolution_trace=None,
        retrieval_query_override="",
        compare_subjects=None,
        compare_doc_like_subjects=None,
        compare_missing_targets=None,
        compare_common_aspects=None,
        compare_topic_pair=None,
        compare_canonical_aspects=None,
        compare_expanded_aspects=None,
        compare_source_subqueries=None,
        compare_status="not_compare",
        compare_plan=None,
    ):
        status = "locked" if resolved and sources else ("ambiguous" if required and candidates else ("not_found" if required else "global_fallback"))
        return {
            "route": route,
            "required": bool(required),
            "resolved": bool(resolved),
            "status": status,
            "sources": list(sources or []),
            "candidates": list(candidates or []),
            "reason": reason,
            "strip_title_mentions": bool(strip_title_mentions),
            "clarification": clarification,
            "target_text": target_text,
            "lock_mode": lock_mode,
            "lock_confidence": float(lock_confidence or 0.0),
            "lock_message_prefix": lock_message_prefix,
            "source_lock_kind": source_lock_kind,
            "source_resolution_trace": dict(source_resolution_trace or {}),
            "retrieval_query_override": retrieval_query_override,
            "compare_subjects": list(compare_subjects or []),
            "compare_doc_like_subjects": list(compare_doc_like_subjects or []),
            "compare_missing_targets": list(compare_missing_targets or []),
            "compare_common_aspects": list(compare_common_aspects or []),
            "compare_topic_pair": list(compare_topic_pair or []),
            "compare_canonical_aspects": list(compare_canonical_aspects or []),
            "compare_expanded_aspects": list(compare_expanded_aspects or []),
            "compare_source_subqueries": dict(compare_source_subqueries or {}),
            "compare_status": compare_status,
            "compare_plan": dict(compare_plan or {}),
        }


class _Routing:
    @staticmethod
    def classify_question_type(query):
        return "compare"

    @staticmethod
    def extract_filename_candidates(query):
        return []

    @staticmethod
    def extract_explicit_regulation_mentions(query):
        return []

    @staticmethod
    def has_contextual_doc_reference(query):
        return False

    @staticmethod
    def has_strong_business_signal(query):
        return False

    @staticmethod
    def strong_topic_terms(query):
        return []

    @staticmethod
    def classify_query_route(query, filename_hints):
        return "content_qa"


class _Guardrails:
    @staticmethod
    def deep_quality_state(query, llm_parse=None, source_resolution=None):
        return {"reason": "", "quality": "valid", "tier": "tier_2"}


class _Retrieval:
    @staticmethod
    def seed_anchor_terms_for_probe(query):
        return []


class _Runtime:
    config = _Config()
    common = _Common()
    compare = _Compare()
    source = _Source()
    routing = _Routing()
    guardrails = _Guardrails()
    retrieval = _Retrieval()

    def __init__(self):
        self.llm_client = _FakeClient()


class _MissingCompare(_Compare):
    def resolve_subject_source(self, subject):
        return {
            "subject": subject,
            "source": "",
            "match_kind": "",
            "doc_like": False,
            "prior": 0.0,
        }


class _MissingRuntime(_Runtime):
    compare = _MissingCompare()


def test_agentic_router_routes_compare_to_json_subqueries():
    runtime = _Runtime()

    result = asyncio.run(agentic_router.route_query(runtime, "compare Dog Rules and Property Rules penalties"))

    assert result["used"] is True
    assert result["route"] == "multi_doc_compare"
    assert result["is_comparison"] is True
    assert result["sub_queries"][0]["raw_text_query"] == "dog violations penalty fine"
    assert runtime.llm_client.payload["max_tokens"] == 520
    assert runtime.llm_client.timeout == 8


def test_agentic_router_prompt_expands_abstract_legal_terms():
    runtime = _Runtime()

    asyncio.run(agentic_router.route_query(runtime, "compare Dog Rules and Property Rules obligations"))

    prompt = runtime.llm_client.payload["system_prompt"]
    assert "\u62bd\u8c61\u6cd5\u5f8b\u6982\u5ff5" in prompt
    assert "\u4e49\u52a1" in prompt
    assert "raw_text_query" in prompt
    assert "doc_prior_query" in prompt


def test_agentic_subquery_expands_abstract_terms_without_polluting_doc_lock():
    payload = {
        "route": "multi_doc_compare",
        "is_comparison": True,
        "is_multi_doc_compare": True,
        "documents": ["Shenzhen Construction Rules"],
        "sub_queries": [
            {
                "source": "Shenzhen Construction Rules",
                "raw_text_query": "\u5efa\u8bbe\u5355\u4f4d\u76f8\u5173\u4e49\u52a1",
                "section_query": "\u4e49\u52a1",
                "doc_prior_query": "Shenzhen Construction Rules",
            }
        ],
        "confidence": 0.9,
    }

    result = agentic_router.normalize_payload(payload)
    subquery = result["sub_queries"][0]

    assert "\u5e94\u5f53" in subquery["raw_text_query"]
    assert "\u5fc5\u987b" in subquery["raw_text_query"]
    assert "\u4e0d\u5f97" in subquery["section_query"]
    assert subquery["source"] == "Shenzhen Construction Rules"
    assert subquery["doc_prior_query"] == "Shenzhen Construction Rules"


def test_agentic_payload_does_not_promote_string_false_or_content_subqueries():
    payload = {
        "route": "content_qa",
        "is_comparison": "false",
        "is_multi_doc_compare": "false",
        "documents": [],
        "sub_queries": [
            {"source": "Dog Rules", "query": "registration deadline"},
            {"source": "Property Rules", "query": "platform duties"},
        ],
        "confidence": 0.9,
    }

    result = agentic_router.normalize_payload(payload)

    assert result["is_comparison"] is False
    assert result["is_multi_doc_compare"] is False
    assert result["documents"] == []


def test_agentic_compare_resolution_maps_subqueries_to_sources():
    runtime = _Runtime()
    route = {
        "used": True,
        "route": "multi_doc_compare",
        "is_comparison": True,
        "is_multi_doc_compare": True,
        "documents": ["Dog Rules", "Property Rules"],
        "common_aspects": ["penalty", "liability"],
        "sub_queries": [
            {
                "source": "Dog Rules",
                "raw_text_query": "dog violations penalty fine",
                "section_query": "legal liability penalty",
                "doc_prior_query": "Dog Rules",
            },
            {
                "source": "Property Rules",
                "raw_text_query": "property management violations penalty fine",
                "section_query": "legal liability penalty",
                "doc_prior_query": "Property Rules",
            },
        ],
        "confidence": 0.9,
    }

    result = agentic_router.build_compare_resolution(runtime, "compare two regulations", route)

    assert result["route"] == "multi_doc_compare"
    assert result["resolved"] is True
    assert result["sources"] == ["dog_rules_2020.docx", "property_rules_2023.pdf"]
    assert result["source_subqueries"]["dog_rules_2020.docx"]["raw_text_query"] == "dog violations penalty fine"
    assert result["source_subqueries"]["property_rules_2023.pdf"]["section_query"] == "legal liability penalty"


def test_agentic_compare_resolution_uses_subqueries_over_combined_document_span():
    runtime = _Runtime()
    route = {
        "used": True,
        "route": "multi_doc_compare",
        "is_comparison": True,
        "is_multi_doc_compare": True,
        "documents": ["Dog Rules and Property Rules penalty comparison"],
        "common_aspects": ["penalty"],
        "sub_queries": [
            {
                "source": "Dog Rules",
                "raw_text_query": "dog violations penalty fine",
                "section_query": "legal liability penalty",
                "doc_prior_query": "Dog Rules",
            },
            {
                "source": "Property Rules",
                "raw_text_query": "property management violations penalty fine",
                "section_query": "legal liability penalty",
                "doc_prior_query": "Property Rules",
            },
        ],
        "confidence": 0.9,
    }

    result = agentic_router.build_compare_resolution(runtime, "compare two regulations", route)

    assert result["resolved"] is True
    assert result["subjects"] == ["Dog Rules", "Property Rules"]
    assert result["missing_doc_targets"] == []
    assert result["sources"] == ["dog_rules_2020.docx", "property_rules_2023.pdf"]


def test_agentic_compare_resolution_keeps_incomplete_multi_target_unresolved():
    runtime = _Runtime()
    route = {
        "used": True,
        "route": "multi_doc_compare",
        "is_comparison": True,
        "is_multi_doc_compare": True,
        "documents": ["Dog Rules", "Property Rules", "Missing Rules"],
        "common_aspects": ["duties"],
        "sub_queries": [
            {"source": "Dog Rules", "raw_text_query": "dog duties", "section_query": "duties", "doc_prior_query": "Dog Rules"},
            {"source": "Property Rules", "raw_text_query": "property duties", "section_query": "duties", "doc_prior_query": "Property Rules"},
            {"source": "Missing Rules", "raw_text_query": "missing duties", "section_query": "duties", "doc_prior_query": "Missing Rules"},
        ],
        "confidence": 0.9,
    }

    result = agentic_router.build_compare_resolution(runtime, "compare three regulations", route)

    assert result["route"] == "multi_doc_compare"
    assert result["resolved"] is False
    assert result["compare_status"] == "target_incomplete"
    assert result["sources"] == ["dog_rules_2020.docx", "property_rules_2023.pdf"]
    assert result["missing_doc_targets"] == ["Missing Rules"]


def test_agentic_compare_resolution_keeps_single_resolved_source_as_incomplete_multi_doc():
    runtime = _Runtime()
    route = {
        "used": True,
        "route": "multi_doc_compare",
        "is_comparison": True,
        "is_multi_doc_compare": True,
        "documents": ["Dog Rules", "Missing Rules"],
        "common_aspects": ["fees"],
        "sub_queries": [
            {"source": "Dog Rules", "raw_text_query": "dog fees", "section_query": "fees", "doc_prior_query": "Dog Rules"},
            {"source": "Missing Rules", "raw_text_query": "missing fees", "section_query": "fees", "doc_prior_query": "Missing Rules"},
        ],
        "confidence": 0.9,
    }

    result = agentic_router.build_compare_resolution(runtime, "compare Dog Rules and Missing Rules fees", route)

    assert result["route"] == "multi_doc_compare"
    assert result["resolved"] is False
    assert result["compare_status"] == "target_incomplete"
    assert result["sources"] == ["dog_rules_2020.docx"]
    assert result["missing_doc_targets"] == ["Missing Rules"]


def test_agentic_compare_resolution_supplements_sources_from_entity_scan():
    class Compare(_Compare):
        _source_map = {
            "Dog Rules": "dog_rules_2020.docx",
        }

    class Source(_Source):
        _title_map = {
            **_Source._title_map,
            "forest_rules_2024.docx": "Forest Resource Rules",
        }

        def extract_title_candidates(self, target, limit=3):
            if "forest" in str(target or "").lower():
                return ["forest_rules_2024.docx"]
            return []

    class Runtime(_Runtime):
        compare = Compare()
        source = Source()

    runtime = Runtime()
    route = {
        "used": True,
        "route": "multi_doc_compare",
        "is_comparison": True,
        "is_multi_doc_compare": True,
        "documents": ["Dog Rules", "Forest Rules"],
        "common_aspects": ["fees"],
        "sub_queries": [
            {"source": "Dog Rules", "raw_text_query": "dog fees", "section_query": "fees", "doc_prior_query": "Dog Rules"},
            {"source": "Forest Rules", "raw_text_query": "forest land fees", "section_query": "fees", "doc_prior_query": "Forest Rules"},
        ],
        "confidence": 0.9,
    }

    result = agentic_router.build_compare_resolution(runtime, "compare Dog Rules fees and forest land fees", route)

    assert result["route"] == "multi_doc_compare"
    assert result["resolved"] is True
    assert result["sources"] == ["dog_rules_2020.docx", "forest_rules_2024.docx"]
    assert result["source_subqueries"]["forest_rules_2024.docx"]["raw_text_query"]


def test_agentic_supplemental_scan_does_not_add_unrelated_broad_matches():
    class Compare(_Compare):
        _source_map = {
            "Dog Rules": "dog_rules_2020.docx",
            "Property Rules": "property_rules_2023.pdf",
        }

    class Source(_Source):
        _title_map = {
            **_Source._title_map,
            "cement_rules_2018.pdf": "Changchun Cement Rules",
        }

        def extract_title_candidates(self, target, limit=3):
            if "duties" in str(target or "").lower():
                return ["cement_rules_2018.pdf"]
            return []

    class Runtime(_Runtime):
        compare = Compare()
        source = Source()

    runtime = Runtime()
    route = {
        "used": True,
        "route": "multi_doc_compare",
        "is_comparison": True,
        "is_multi_doc_compare": True,
        "documents": ["Dog Rules", "Property Rules"],
        "common_aspects": ["duties"],
        "sub_queries": [
            {"source": "Dog Rules", "raw_text_query": "dog duties", "section_query": "duties", "doc_prior_query": "Dog Rules"},
            {"source": "Property Rules", "raw_text_query": "property duties", "section_query": "duties", "doc_prior_query": "Property Rules"},
        ],
        "confidence": 0.9,
    }

    result = agentic_router.build_compare_resolution(runtime, "compare Dog Rules and Property Rules duties", route)

    assert result["resolved"] is True
    assert result["sources"] == ["dog_rules_2020.docx", "property_rules_2023.pdf"]
    assert result["supplemented_targets"] == []


def test_agentic_strict_scan_filters_candidates_by_dynamic_region():
    class Compare(_Compare):
        _source_map = {
            "Shenzhen Market Rules": "shenzhen_market_rules.docx",
            "Shenzhen Quality Rules": "shenzhen_quality_rules.pdf",
        }

    class Source(_Source):
        _title_map = {
            **_Source._title_map,
            "shenzhen_market_rules.docx": "深圳市建筑市场规定",
            "shenzhen_quality_rules.pdf": "深圳市建设工程质量条例",
            "shaoxing_property_rules.pdf": "绍兴市物业管理条例",
        }

        def source_core_entities(self, source):
            if str(source or "").startswith("shenzhen"):
                return ["深圳市", "深圳"]
            if str(source or "").startswith("shaoxing"):
                return ["绍兴市", "绍兴"]
            return []

        def extract_title_candidates(self, target, limit=3):
            if "深圳" in str(target or ""):
                return ["shaoxing_property_rules.pdf", "shenzhen_quality_rules.pdf"]
            return []

    class Runtime(_Runtime):
        compare = Compare()
        source = Source()

    runtime = Runtime()
    route = {
        "used": True,
        "route": "multi_doc_compare",
        "is_comparison": True,
        "is_multi_doc_compare": True,
        "documents": ["Shenzhen Market Rules", "深圳建设工程质量条例"],
        "common_aspects": ["duties"],
        "sub_queries": [
            {"source": "Shenzhen Market Rules", "raw_text_query": "market duties", "section_query": "duties", "doc_prior_query": "Shenzhen Market Rules"},
            {"source": "深圳建设工程质量条例", "raw_text_query": "quality duties", "section_query": "duties", "doc_prior_query": "深圳建设工程质量条例"},
        ],
        "confidence": 0.9,
    }

    result = agentic_router.build_compare_resolution(runtime, "compare Shenzhen regulations", route)

    assert result["sources"] == ["shenzhen_market_rules.docx", "shenzhen_quality_rules.pdf"]
    assert "shaoxing_property_rules.pdf" not in result["sources"]


def test_agentic_source_validation_uses_target_query_not_whole_compare_query():
    class Source(_Source):
        def validate_source_lock_candidate(self, query, target_text, source, *, prior=0.0, match_kind=""):
            if "Other Rules" in query:
                return {"accepted": False, "hard_negative": True, "reasons": ["cross_target_pollution"]}
            return {"accepted": True, "score": 1.0, "reasons": ["target_only"]}

    class Runtime(_Runtime):
        source = Source()

    runtime = Runtime()
    route = {
        "used": True,
        "route": "multi_doc_compare",
        "is_comparison": True,
        "is_multi_doc_compare": True,
        "documents": ["Dog Rules", "Property Rules"],
        "common_aspects": ["duties"],
        "sub_queries": [
            {"source": "Dog Rules", "raw_text_query": "dog duties", "section_query": "duties", "doc_prior_query": "Dog Rules"},
            {"source": "Property Rules", "raw_text_query": "property duties", "section_query": "duties", "doc_prior_query": "Property Rules"},
        ],
        "confidence": 0.9,
    }

    result = agentic_router.build_compare_resolution(runtime, "compare Dog Rules and Other Rules duties", route)

    assert result["resolved"] is True
    assert result["sources"] == ["dog_rules_2020.docx", "property_rules_2023.pdf"]


def test_agentic_compare_resolution_rejects_failed_source_lock_validation():
    class Source(_Source):
        def validate_source_lock_candidate(self, query, target_text, source, *, prior=0.0, match_kind=""):
            if source == "property_rules_2023.pdf":
                return {
                    "accepted": False,
                    "score": 0.0,
                    "hard_negative": True,
                    "reasons": ["region_mismatch"],
                }
            return {"accepted": True, "score": 1.0, "reasons": ["test_accept"]}

    class Runtime(_Runtime):
        source = Source()

    runtime = Runtime()
    route = {
        "used": True,
        "route": "multi_doc_compare",
        "is_comparison": True,
        "is_multi_doc_compare": True,
        "documents": ["Dog Rules", "Property Rules"],
        "common_aspects": ["duties"],
        "sub_queries": [
            {"source": "Dog Rules", "raw_text_query": "dog duties", "section_query": "duties", "doc_prior_query": "Dog Rules"},
            {"source": "Property Rules", "raw_text_query": "property duties", "section_query": "duties", "doc_prior_query": "Property Rules"},
        ],
        "confidence": 0.9,
    }

    result = agentic_router.build_compare_resolution(runtime, "compare two regulations", route)

    assert result == {}


def test_agentic_compare_resolution_keeps_multi_doc_when_one_of_three_targets_rejected():
    class Source(_Source):
        def validate_source_lock_candidate(self, query, target_text, source, *, prior=0.0, match_kind=""):
            if source == "property_rules_2023.pdf":
                return {
                    "accepted": False,
                    "score": 0.0,
                    "hard_negative": True,
                    "reasons": ["region_mismatch"],
                }
            return {"accepted": True, "score": 1.0, "reasons": ["test_accept"]}

    class Compare(_Compare):
        _source_map = {
            **_Compare._source_map,
            "Market Rules": "market_rules_2007.docx",
        }

    class Runtime(_Runtime):
        compare = Compare()
        source = Source()

    runtime = Runtime()
    route = {
        "used": True,
        "route": "multi_doc_compare",
        "is_comparison": True,
        "is_multi_doc_compare": True,
        "documents": ["Dog Rules", "Property Rules", "Market Rules"],
        "common_aspects": ["duties"],
        "sub_queries": [
            {"source": "Dog Rules", "raw_text_query": "dog duties", "section_query": "duties", "doc_prior_query": "Dog Rules"},
            {"source": "Property Rules", "raw_text_query": "property duties", "section_query": "duties", "doc_prior_query": "Property Rules"},
            {"source": "Market Rules", "raw_text_query": "market duties", "section_query": "duties", "doc_prior_query": "Market Rules"},
        ],
        "confidence": 0.9,
    }

    result = agentic_router.build_compare_resolution(runtime, "compare three regulations", route)

    assert result["route"] == "multi_doc_compare"
    assert result["resolved"] is False
    assert result["compare_status"] == "target_incomplete"
    assert result["sources"] == ["dog_rules_2020.docx", "market_rules_2007.docx"]
    assert result["rejected_doc_targets"][0]["source"] == "property_rules_2023.pdf"


def test_prelude_prefers_agentic_compare_plan_over_rule_route():
    runtime = _Runtime()

    result = asyncio.run(
        prepare_lightweight_recall_prelude(
            runtime,
            "compare Dog Rules and Property Rules penalties",
            filename_hints=[],
            user_id="u1",
        )
    )

    assert result["early_return"] is None
    assert result["qtype"] == "compare"
    assert result["query_route"] == "multi_doc_compare"
    assert result["source_resolution"]["source_lock_kind"] == "agentic_compare_lock"
    assert result["source_resolution"]["sources"] == ["dog_rules_2020.docx", "property_rules_2023.pdf"]
    assert result["source_resolution"]["compare_source_subqueries"]["dog_rules_2020.docx"]["raw_text_query"] == "dog violations penalty fine"
    trace = result["source_resolution"]["source_resolution_trace"]
    assert trace["final_source_resolution"]["selected"] == "agentic_router"
    assert trace["final_source_resolution"]["status"] == "locked"
    assert trace["rule_compare_diagnostic"]["ignored"] is True
    assert "rule_source_resolution" not in trace


def test_prelude_ignores_unresolved_agentic_compare_and_keeps_rule_global_fallback():
    runtime = _MissingRuntime()

    result = asyncio.run(
        prepare_lightweight_recall_prelude(
            runtime,
            "compare Dog Rules and Property Rules penalties",
            filename_hints=[],
            user_id="u1",
        )
    )

    assert result["early_return"] is None
    assert result["query_route"] == "content_qa"
    assert result["source_resolution"]["status"] == "global_fallback"
    assert result["source_resolution"]["required"] is False
    assert result["source_resolution"]["reason"] == "fallback_rule_route"
    assert "router_target_failure_global_fallback" not in result["source_resolution"]["source_resolution_trace"]


def test_agentic_router_does_not_promote_single_doc_extract_without_compare_intent():
    runtime = _Runtime()
    route = {
        "used": True,
        "route": "single_doc_compare",
        "is_comparison": True,
        "is_multi_doc_compare": False,
        "documents": ["Dog Rules"],
        "common_aspects": ["penalty"],
        "sub_queries": [
            {
                "source": "Dog Rules",
                "raw_text_query": "penalty rules",
                "section_query": "legal liability penalty",
                "doc_prior_query": "Dog Rules",
            }
        ],
        "confidence": 0.9,
    }

    result = agentic_router.build_compare_resolution(runtime, "Dog Rules penalty rules", route)

    assert result == {}


def test_agentic_router_rejects_multi_doc_without_explicit_compare_intent():
    runtime = _Runtime()
    route = {
        "used": True,
        "reason": "llm_json",
        "route": "multi_doc_compare",
        "is_comparison": True,
        "is_multi_doc_compare": True,
        "documents": ["Dog Rules", "Property Rules"],
        "common_aspects": ["duties"],
        "sub_queries": [
            {
                "source": "Dog Rules",
                "raw_text_query": "alarm devices required",
                "section_query": "safety devices",
                "doc_prior_query": "Dog Rules",
            },
            {
                "source": "Property Rules",
                "raw_text_query": "alarm devices required",
                "section_query": "safety devices",
                "doc_prior_query": "Property Rules",
            },
        ],
        "confidence": 0.95,
    }

    result = agentic_router.build_compare_resolution(runtime, "Dog Rules alarm device requirements", route)

    assert result == {}
    assert route["reason"] == "multi_doc_without_explicit_intent"


def test_agentic_single_source_resolution_locks_unique_source():
    runtime = _Runtime()
    route = {
        "used": True,
        "route": "single_doc_compare",
        "is_comparison": True,
        "is_multi_doc_compare": False,
        "documents": ["Dog Rules"],
        "sub_queries": [
            {
                "source": "Dog Rules",
                "raw_text_query": "registration deadline",
                "section_query": "registration",
                "doc_prior_query": "Dog Rules",
            }
        ],
        "confidence": 0.95,
    }

    result = agentic_router.build_single_source_resolution(runtime, "Dog registration deadline", route)

    assert result["reason"] == "agentic_single_source_lock"
    assert result["sources"] == ["dog_rules_2020.docx"]
    assert result["source_resolution_trace"]["agentic_single_source_lock"] is True


def test_agentic_single_source_resolution_rejects_unresolved_source():
    runtime = _Runtime()
    route = {
        "used": True,
        "route": "single_doc_compare",
        "is_comparison": True,
        "is_multi_doc_compare": False,
        "documents": ["Missing Rules"],
        "sub_queries": [
            {
                "source": "Missing Rules",
                "raw_text_query": "registration deadline",
                "section_query": "registration",
                "doc_prior_query": "Missing Rules",
            }
        ],
        "confidence": 0.95,
    }

    result = agentic_router.build_single_source_resolution(runtime, "Missing registration deadline", route)

    assert result == {}
