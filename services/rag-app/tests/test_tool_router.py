import asyncio

from app.core.query.recall_flow import build_empty_search_result
from app.core.query.tool_router import route_with_search_database_tool


class _Config:
    ENABLE_LLM_TOOL_ROUTER = True
    LLM_TOOL_ROUTER_MAX_TOKENS = 120
    LLM_TOOL_ROUTER_TIMEOUT = 8


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
        return {"messages": [], **kwargs.get("extra", {})}

    async def chat_response(self, payload, *, timeout=None):
        self.timeout = timeout
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_database",
                                    "arguments": '{"query":"养犬处罚 公安机关","reason":"probe"}',
                                }
                            }
                        ]
                    }
                }
            ]
        }


class _Runtime:
    config = _Config()

    def __init__(self):
        self.llm_client = _FakeClient()


class _Routing:
    def is_weak_reference_query(self, query):
        return False


class _EmptyRuntime:
    routing = _Routing()


def test_tool_router_uses_search_database_tool_call():
    runtime = _Runtime()

    result = asyncio.run(route_with_search_database_tool(runtime, "养犬处罚是谁执法"))

    assert result["tool_called"] is True
    assert result["tool_name"] == "search_database"
    assert result["query"] == "养犬处罚 公安机关"
    assert result["arguments"]["query"] == "养犬处罚 公安机关"
    assert runtime.llm_client.payload["extra"]["tools"][0]["function"]["name"] == "search_database"
    assert runtime.llm_client.payload["extra"]["tool_choice"]["function"]["name"] == "search_database"
    assert runtime.llm_client.timeout == 8


def test_tool_router_fallback_preserves_user_query_when_model_omits_tool_call():
    runtime = _Runtime()

    async def no_tool_response(payload, *, timeout=None):
        return {"choices": [{"message": {"content": "我无法回答。"}}]}

    runtime.llm_client.chat_response = no_tool_response

    query = "《聊城市养犬管理条例》第三十八条怎么处罚？"
    result = asyncio.run(route_with_search_database_tool(runtime, query))

    assert result["tool_called"] is True
    assert result["tool_name"] == "search_database"
    assert result["query"] == query
    assert result["arguments"]["query"] == query
    assert result["reason"] == "router_no_tool_call_fallback"


def test_tool_router_backfills_missing_tool_arguments_query():
    runtime = _Runtime()

    async def empty_arguments_response(payload, *, timeout=None):
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_database",
                                    "arguments": "{}",
                                }
                            }
                        ]
                    }
                }
            ]
        }

    runtime.llm_client.chat_response = empty_arguments_response

    query = "《聊城市养犬管理条例》第三十八条怎么处罚？"
    result = asyncio.run(route_with_search_database_tool(runtime, query))

    assert result["tool_called"] is True
    assert result["query"] == query
    assert result["arguments"]["query"] == query


def test_empty_search_result_from_tool_requests_clarification():
    result = build_empty_search_result(
        _EmptyRuntime(),
        query="q",
        retrieval_query="q",
        retrieval_query_raw="q",
        dense_query="q",
        qtype="other",
        qfilters={},
        llm_parse={},
        intent_classification={"search_database_tool_used": True},
        is_comparison=False,
        query_route="content_qa",
        docs=[],
        lex_items=[],
        visible_dense={"dropped": 0, "states": {}},
        visible_lex={"dropped": 0, "states": {}},
        dense_source_scores={},
        recall_k=10,
        final_n=5,
        source_resolution={},
        active_fnames=[],
    )

    assert result["search_database_tool_empty"] is True
    assert result["soft_clarification_required"] is True
    assert result["soft_clarification_reason"] == "search_database_empty"


def test_empty_search_result_from_tool_does_not_clarify_after_hard_lock():
    result = build_empty_search_result(
        _EmptyRuntime(),
        query="q",
        retrieval_query="q",
        retrieval_query_raw="q",
        dense_query="q",
        qtype="other",
        qfilters={},
        llm_parse={},
        intent_classification={"search_database_tool_used": True},
        is_comparison=False,
        query_route="explicit_regulation_reference",
        docs=[],
        lex_items=[],
        visible_dense={"dropped": 0, "states": {}},
        visible_lex={"dropped": 0, "states": {}},
        dense_source_scores={},
        recall_k=10,
        final_n=5,
        source_resolution={
            "resolved": True,
            "sources": ["demo.docx"],
            "lock_mode": "hard_lock",
            "lock_confidence": 1.0,
        },
        active_fnames=["demo.docx"],
    )

    assert result["search_database_tool_empty"] is True
    assert result["resolved_source_lock"] is True
    assert result["soft_clarification_required"] is False
    assert result["soft_clarification_reason"] == ""
