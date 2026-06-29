from collections import OrderedDict
import threading
from typing import Any, Callable, Dict, List, Optional

from app.utils.text import sanitize_index_text


class QueryParseCache:
    def __init__(self, max_size: int = 256) -> None:
        self.max_size = max(16, int(max_size))
        self._items: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        cache_key = (key or "").strip()
        if not cache_key:
            return None
        with self._lock:
            if cache_key not in self._items:
                return None
            value = self._items.pop(cache_key)
            self._items[cache_key] = value
            return dict(value or {})

    def set(self, key: str, value: Dict[str, Any]) -> None:
        cache_key = (key or "").strip()
        if not cache_key:
            return
        with self._lock:
            self._items[cache_key] = dict(value or {})
            while len(self._items) > self.max_size:
                self._items.popitem(last=False)


def parse_query_fallback(
    user_query: str,
    *,
    locked_title: str = "",
    normalize_query: Callable[[str], str],
    extract_filename_candidates: Callable[[str], List[str]],
    extract_explicit_regulation_mentions: Callable[[str], List[str]],
    query_anchor_terms: Callable[[str], List[str]],
    query_has_compare_intent: Callable[[str], bool],
    build_compare_plan: Callable[[str], Dict[str, Any]],
    extract_compare_common_aspects: Callable[[Dict[str, Any]], List[str]],
    query_semantic_aspects: Callable[[str], Dict[str, Any]],
) -> Dict[str, Any]:
    query = normalize_query(user_query)
    locked = normalize_query(sanitize_index_text(locked_title))
    documents = extract_filename_candidates(user_query) + extract_explicit_regulation_mentions(user_query)
    anchors = query_anchor_terms(user_query)[:8]
    aspects: List[str] = []
    if query_has_compare_intent(user_query):
        try:
            aspects = extract_compare_common_aspects(build_compare_plan(user_query))
        except Exception:
            aspects = []
    if not aspects:
        semantic = query_semantic_aspects(user_query)
        aspects = (semantic.get("terms") or [])[:6] if isinstance(semantic, dict) else []
    intent = "compare" if query_has_compare_intent(user_query) else "qa"
    route = "compare" if intent == "compare" else "content_qa"
    return {
        "route": route,
        "intent": intent,
        "question_type": intent,
        "action": "answer",
        "documents": list(dict.fromkeys([item for item in documents if item])),
        "anchors": list(dict.fromkeys([item for item in anchors if item])),
        "aspects": list(dict.fromkeys([item for item in aspects if item])),
        "evidence_query": query,
        "locked_title": locked,
        "confidence": 0.35,
        "used_llm": False,
    }


def parse_query_cached(
    cache: QueryParseCache,
    user_query: str,
    *,
    locked_title: str = "",
    parse_query: Callable[[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    key = f"{(locked_title or '').strip()}||{(user_query or '').strip()}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    parsed = parse_query(user_query, locked_title)
    if isinstance(parsed, dict):
        cache.set(key, parsed)
        cache.set((user_query or "").strip(), parsed)
        return parsed
    cache.set(key, {})
    return {}


def extract_section_query_targets(
    cache: QueryParseCache,
    query: str,
    *,
    normalize_query: Callable[[str], str],
    limit: int = 8,
) -> List[str]:
    parsed = cache.get((query or "").strip()) or {}
    targets = parsed.get("section_targets")
    if not isinstance(targets, list):
        return []
    out: List[str] = []
    for item in targets:
        value = normalize_query(str(item or ""))
        if not value:
            continue
        if value not in out:
            out.append(value)
        if len(out) >= max(1, int(limit)):
            break
    return out
