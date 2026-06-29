import time
from typing import Any, Dict

from fastapi import APIRouter

from app.core import clarification
from app.schemas import QueryRequest, QueryResponse


def _now_perf() -> float:
    return time.perf_counter()


def _attach_total_request_timing(result: Dict[str, Any], request_started: float) -> Dict[str, Any]:
    meta = dict(result.get("metadata") or {})
    server_timing = dict(meta.get("server_timing_ms") or {})
    server_timing["total_request"] = round((time.perf_counter() - request_started) * 1000, 1)
    meta["server_timing_ms"] = server_timing
    result["metadata"] = meta
    return result


def _log_query_request(query_req: QueryRequest) -> None:
    print(f"[QUERY] user_id={query_req.user_id!r} query={query_req.query!r}")


def create_router(context: Any) -> APIRouter:
    router = APIRouter()
    query_core = context.query_core()

    @router.post("/query", response_model=QueryResponse)
    async def query(query_req: QueryRequest):
        request_started = _now_perf()
        _log_query_request(query_req)

        resolved = await clarification.try_resolve_pending(query_req, context, query_core)
        if resolved is not None:
            return QueryResponse(**_attach_total_request_timing(resolved, request_started))

        result = await query_core.process(
            query=query_req.query,
            user_id=query_req.user_id,
            top_k=query_req.top_k,
            enable_rerank=bool(query_req.enable_rerank),
        )
        clarification.remember_if_needed(query_req, context, result)
        return QueryResponse(**_attach_total_request_timing(result, request_started))

    @router.post("/retrieve")
    async def retrieve(query_req: QueryRequest):
        request_started = _now_perf()
        result = await query_core.retrieve(
            query=query_req.query,
            user_id=query_req.user_id,
            top_k=query_req.top_k,
            enable_rerank=bool(query_req.enable_rerank),
        )
        return _attach_total_request_timing(result, request_started)

    return router
