import os
import re
import time
from typing import Any, Dict, List, Optional

from app.schemas import QueryRequest
from app.utils import files as file_utils
from app.utils import text as text_utils


class _MemoryClarificationStore:
    def __init__(self) -> None:
        self._pending: Dict[str, Dict[str, Any]] = {}

    def get_pending_clarification(self, user_id: str) -> Optional[Dict[str, Any]]:
        item = self._pending.get(user_id)
        if not item:
            return None
        expires_at = float(item.get("expires_at") or 0.0)
        if expires_at and expires_at < time.time():
            self._pending.pop(user_id, None)
            return None
        return dict(item)

    def set_pending_clarification(
        self,
        user_id: str,
        query: str,
        candidates: List[str],
        reason: str,
        ttl_sec: int,
    ) -> None:
        self._pending[user_id] = {
            "query": query,
            "candidates": list(candidates or []),
            "reason": reason,
            "expires_at": time.time() + max(1, int(ttl_sec)),
        }

    def clear_pending_clarification(self, user_id: str) -> None:
        self._pending.pop(user_id, None)


def _zh_number_to_int(token: str) -> Optional[int]:
    mapping = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    value = (token or "").strip()
    if not value:
        return None
    if value in mapping:
        return mapping[value]
    if value.startswith("十") and len(value) == 2 and value[1] in mapping:
        return 10 + int(mapping[value[1]])
    if len(value) == 2 and value[0] in mapping and value[1] == "十":
        return int(mapping[value[0]]) * 10
    if len(value) == 3 and value[0] in mapping and value[1] == "十" and value[2] in mapping:
        return int(mapping[value[0]]) * 10 + int(mapping[value[2]])
    return None


def parse_pending_candidate_selection(query: str, max_options: int) -> Optional[int]:
    normalized = text_utils.normalize_query(query)
    if not normalized or max_options <= 0 or len(normalized) > 20:
        return None
    number: Optional[int] = None
    if re.fullmatch(r"\d{1,2}", normalized):
        number = int(normalized)
    else:
        match = re.fullmatch(r"(?:选择第|选第|选择|选|第)?\s*(\d{1,2})\s*(?:个|项|条|号|部|份)?", normalized)
        if match:
            number = int(match.group(1))
        else:
            zh_match = re.fullmatch(r"(?:选择第|选第|选择|选|第)?\s*([一二两三四五六七八九十]{1,3})\s*(?:个|项|条|号|部|份)?", normalized)
            if zh_match:
                number = _zh_number_to_int(zh_match.group(1))
    if number is None:
        return None
    if 1 <= number <= int(max_options):
        return number - 1
    return None


class ClarificationService:
    def __init__(self, backend: Any = None):
        self._state = (
            getattr(backend, "state_store", None)
            or getattr(backend, "_lex_store", None)
            or getattr(backend, "lex_store", None)
            or _MemoryClarificationStore()
        )

    def get_pending(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self._state.get_pending_clarification(user_id)

    def set_pending(self, user_id: str, query: str, candidates: List[str], reason: str) -> None:
        safe_candidates = [self.normalize_filename(item) for item in candidates or []]
        ttl = max(60, int(os.getenv("CLARIFICATION_PENDING_TTL_SEC", "900")))
        self._state.set_pending_clarification(
            user_id,
            self.normalize_query(query),
            [item for item in safe_candidates if item],
            reason=reason,
            ttl_sec=ttl,
        )

    def clear_pending(self, user_id: str) -> None:
        self._state.clear_pending_clarification(user_id)

    def parse_candidate_selection(self, query: str, max_options: int) -> Optional[int]:
        return parse_pending_candidate_selection(query, max_options)

    def normalize_filename(self, source: str) -> str:
        return file_utils.normalize_filename_for_match(source)

    def normalize_query(self, query: str) -> str:
        return text_utils.normalize_query(query)


def _clarification_service(runtime: Any) -> ClarificationService:
    if isinstance(runtime, ClarificationService):
        return runtime
    factory = getattr(runtime, "clarification_service", None)
    if callable(factory):
        return factory()
    return ClarificationService(runtime)


async def try_resolve_pending(
    query_req: QueryRequest,
    runtime: Any,
    handler: Any,
) -> Optional[Dict[str, Any]]:
    context = _clarification_service(runtime)
    pending = context.get_pending(query_req.user_id)
    if not pending:
        return None

    candidates = list((pending or {}).get("candidates") or [])
    picked_idx = context.parse_candidate_selection(query_req.query, len(candidates))
    if picked_idx is None:
        return None

    locked_source = context.normalize_filename(
        candidates[picked_idx] if picked_idx < len(candidates) else ""
    )
    base_query = context.normalize_query((pending or {}).get("query") or "")
    if not locked_source or not base_query:
        return None

    result = await handler.process(
        query=base_query,
        user_id=query_req.user_id,
        top_k=query_req.top_k,
        enable_rerank=bool(query_req.enable_rerank),
        forced_fnames=[locked_source],
    )
    context.clear_pending(query_req.user_id)
    meta = dict(result.get("metadata") or {})
    meta["clarification_resolved"] = True
    meta["clarification_selected_source"] = locked_source
    meta["clarification_original_query"] = base_query
    result["metadata"] = meta
    return result


def remember_if_needed(query_req: QueryRequest, runtime: Any, result: Dict[str, Any]) -> None:
    context = _clarification_service(runtime)
    meta = dict(result.get("metadata") or {})
    candidates = list(meta.get("candidate_sources") or [])
    if meta.get("answer_mode") != "clarification" or not candidates:
        return
    context.set_pending(
        user_id=query_req.user_id,
        query=meta.get("query") or query_req.query,
        candidates=candidates,
        reason=(meta.get("refused") or meta.get("blocked") or "document_clarification"),
    )
