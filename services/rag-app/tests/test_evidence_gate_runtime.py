import asyncio
import importlib.util
import sys
import types
from pathlib import Path


_APP_DIR = Path(__file__).resolve().parents[1]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

try:
    import fastapi  # noqa: F401
except ModuleNotFoundError:
    fastapi_stub = types.ModuleType("fastapi")
    fastapi_stub.HTTPException = Exception
    sys.modules["fastapi"] = fastapi_stub

_SPEC = importlib.util.spec_from_file_location("query_core_under_test", _APP_DIR / "app" / "runtime" / "query.py")
_QUERY_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_QUERY_MODULE)
QueryCore = _QUERY_MODULE.QueryCore


class _Config:
    ENABLE_LLM_EVIDENCE_CHECK = True
    LLM_EVIDENCE_CHECK_MAX_CHARS = 1800
    LLM_EVIDENCE_CHECK_MAX_TOKENS = 48
    LLM_EVIDENCE_CHECK_TIMEOUT = 4
    EVIDENCE_GATE_RERANK_MIN_SCORE = 0.6


class _FakeLlmClient:
    def __init__(self, content):
        self.content = content
        self.payload = None
        self.timeout = None

    def available(self):
        return True

    def build_payload(self, *args, **kwargs):
        self.payload = {"args": args, "kwargs": kwargs}
        return {"messages": [], "max_tokens": kwargs.get("max_tokens")}

    async def chat_text(self, payload, *, timeout=None):
        self.timeout = timeout
        return self.content


def _runtime(llm_content='{"scope":"partial","reason":"missing_core_aspect"}'):
    runtime = QueryCore.__new__(QueryCore)
    runtime.config = _Config
    runtime.llm_client = _FakeLlmClient(llm_content)
    return runtime


def _hit(score=0.9, text="证据正文应当直接支持问题。"):
    return {
        "score": score,
        "entity": {
            "source": "demo.docx",
            "text": text,
            "metadata": {"content": text},
        },
    }


def test_reranker_low_top_score_bypasses_evidence_gate_as_empty():
    runtime = _runtime()

    result = runtime._reranker_empty_gate_result([_hit(score=0.3)], score_mode="score")

    assert result["answer_scope"] == "refusal"
    assert result["evidence_coverage_reason"] == "empty_evidence"
    assert result["reranker_bypass_reason"] == "top_reranked_chunk_below_threshold"
    assert runtime._reranker_empty_gate_result([_hit(score=0.7)], score_mode="score") is None


def test_llm_evidence_gate_uses_small_output_and_downgrades_only():
    runtime = _runtime()
    observations = {
        "answer_scope": "full",
        "evidence_coverage_reason": "sufficient_evidence",
        "covered_aspects": [],
        "uncovered_aspects": [],
    }

    result = asyncio.run(runtime._apply_llm_evidence_gate("问题", [_hit()], observations))

    assert result["answer_scope"] == "partial"
    assert result["evidence_coverage_reason"] == "missing_core_aspect"
    assert runtime.llm_client.payload["kwargs"]["max_tokens"] == 48
    assert runtime.llm_client.timeout == 4
