"""Filter helpers shared by dense and lexical retrieval."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, List, Optional, Sequence, Tuple


_ARTICLE_NUMERAL_CHARS = "\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\u4e07\u96f6\u3007\u4e24\u58f9\u8d30\u53c1\u8086\u4f0d\u9646\u67d2\u634c\u73960-9\uff10-\uff19"
ARTICLE_RE = re.compile(rf"(?:\u7b2c\s*)?[{_ARTICLE_NUMERAL_CHARS}\s]+\u6761")
ARTICLE_EXACT_RE = re.compile(rf"^(?:\u7b2c)?[{_ARTICLE_NUMERAL_CHARS}]+\u6761$")
MAX_ARTICLE_ID_CHARS = 16


def _article_text_candidates(value: Any) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    candidates = [text]
    try:
        repaired = text.encode("gb18030", errors="replace").decode("utf-8", errors="replace").strip()
        if repaired and repaired != text:
            candidates.append(repaired)
    except Exception:
        pass
    return candidates


def _flatten_values(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        for key in ("article_id", "article_no", "value", "name", "text"):
            if value.get(key):
                return _flatten_values(value.get(key))
        return []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            out.extend(_flatten_values(item))
        return out
    text = str(value or "").strip()
    if not text:
        return []
    parts = re.split(r"[,，、；;\s]+", text)
    return [part for part in parts if part.strip()] or [text]


def _normalize_article_text(text: str) -> str:
    compact = re.sub(r"\s+", "", str(text or "").strip())
    if not compact:
        return ""
    if len(compact) > MAX_ARTICLE_ID_CHARS:
        return ""
    compact = compact.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if compact and not compact.startswith("\u7b2c") and compact.endswith("\u6761"):
        compact = "\u7b2c" + compact
    return compact if ARTICLE_EXACT_RE.match(compact) else ""


def normalize_exact_article_id(value: Any) -> str:
    """Return an article id only when the whole value is a valid article label."""
    raw_text = str(value or "").strip()
    if not raw_text:
        return ""
    for candidate in _article_text_candidates(raw_text):
        normalized = _normalize_article_text(candidate)
        if normalized:
            return normalized
    return ""


def normalize_article_id(value: Any) -> str:
    """Return a strict article id like ``第十五条`` or ``""``.

    The retrieval filter must never treat document titles, long query spans, or
    entity strings as article ids. If no explicit article-number pattern is
    present, we intentionally return an empty string.
    """
    raw_text = str(value or "").strip()
    if not raw_text:
        return ""
    for candidate in _article_text_candidates(raw_text):
        match = ARTICLE_RE.search(candidate)
        if match:
            normalized = _normalize_article_text(match.group(0))
            if normalized:
                return normalized
    return ""


def _article_ids_from_text(value: Any) -> List[str]:
    out: List[str] = []
    for candidate in _article_text_candidates(value):
        for match in ARTICLE_RE.finditer(candidate):
            normalized = _normalize_article_text(match.group(0))
            if normalized and normalized not in out:
                out.append(normalized)
    return out


def normalize_article_ids(*values: Any) -> List[str]:
    out: List[str] = []
    for value in values:
        for item in _flatten_values(value):
            for normalized in _article_ids_from_text(item):
                if normalized and normalized not in out:
                    out.append(normalized)
    return out


def normalize_configured_article_ids(*values: Any) -> List[str]:
    out: List[str] = []
    for value in values:
        for item in _flatten_values(value):
            normalized = normalize_exact_article_id(item)
            if normalized and normalized not in out:
                out.append(normalized)
    return out


def configured_article_ids_are_valid(*values: Any) -> bool:
    raw_items: List[str] = []
    for value in values:
        raw_items.extend(list(_flatten_values(value)))
    if not raw_items:
        return True
    return all(bool(normalize_exact_article_id(item)) for item in raw_items)


def article_ids_from_query(query: str) -> List[str]:
    return _article_ids_from_text(query)


def target_article_ids(qfilters: Optional[dict], query: str = "") -> List[str]:
    qfilters = qfilters or {}
    if qfilters.get("_skip_article_id_filter"):
        return []
    configured = normalize_configured_article_ids(
        qfilters.get("target_articles"),
        qfilters.get("target_article"),
        qfilters.get("article_ids"),
        qfilters.get("article_id"),
    )
    query_ids = article_ids_from_query(query)
    return normalize_article_ids(configured, query_ids)


def _in_expr(field: str, values: Sequence[str]) -> Optional[str]:
    normalized = normalize_configured_article_ids(values)
    if not normalized:
        return None
    if len(normalized) == 1:
        return f"{field} == {json.dumps(normalized[0], ensure_ascii=False)}"
    return f"{field} in {json.dumps(normalized, ensure_ascii=False)}"


def build_milvus_filter(
    *,
    sources: Optional[Sequence[str]] = None,
    article_ids: Optional[Sequence[str]] = None,
) -> Optional[str]:
    parts: List[str] = []
    source_values = [str(source or "").strip() for source in (sources or []) if str(source or "").strip()]
    if len(source_values) == 1:
        parts.append(f"source == {json.dumps(source_values[0], ensure_ascii=False)}")
    elif len(source_values) > 1:
        parts.append(f"source in {json.dumps(source_values, ensure_ascii=False)}")
    article_expr = _in_expr("article_id", article_ids or [])
    if article_expr:
        parts.append(article_expr)
    return " and ".join(parts) if parts else None


def sqlite_article_filter_sql(article_ids: Optional[Sequence[str]]) -> Tuple[str, List[Any]]:
    normalized = normalize_configured_article_ids(article_ids or [])
    if not normalized:
        return "", []
    placeholders = ", ".join(["?"] * len(normalized))
    clause = (
        "("
        f"json_extract(m.metadata, '$.article_id') IN ({placeholders}) "
        f"OR json_extract(m.metadata, '$.article_no') IN ({placeholders})"
        ")"
    )
    return clause, list(normalized) + list(normalized)
