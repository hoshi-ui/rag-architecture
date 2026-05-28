import os
import re
from typing import Any, Dict, Iterable, Set

import requests


BASE_URL = os.getenv("REAL_BASELINE_API", "http://127.0.0.1:8080")


def normalize_source_name(name: str) -> str:
    value = (name or "").strip().replace("\\", "/")
    return value.split("/")[-1]


def _filename_stem(name: str) -> str:
    source = normalize_source_name(name)
    if "." not in source:
        return source
    return ".".join(source.split(".")[:-1])


def _normalize_title_key(text: str) -> str:
    value = (text or "").strip().lower()
    value = re.sub(r"(?:[_\-]\d{4}[-_]\d{2}[-_]\d{2}){1,2}$", "", value)
    value = re.sub(r"[\s_\-./]+", "", value)
    return value


def _fallback_canonical_doc_id(source: str) -> str:
    stem = _filename_stem(source)
    return _normalize_title_key(stem)


def canonical_doc_id_from_document(item: Dict[str, Any]) -> str:
    canonical_title = str(item.get("canonical_title") or "").strip()
    if canonical_title:
        return _normalize_title_key(canonical_title)
    canonical_doc_id = str(item.get("canonical_doc_id") or item.get("same_title_group") or "").strip()
    if canonical_doc_id:
        return canonical_doc_id
    return _fallback_canonical_doc_id(item.get("filename") or "")


def canonical_doc_id_for_source(source: str, source_to_canonical: Dict[str, str]) -> str:
    normalized = normalize_source_name(source)
    if not normalized:
        return ""
    return source_to_canonical.get(normalized) or _fallback_canonical_doc_id(normalized)


def load_document_registry(base_url: str = BASE_URL) -> Dict[str, Any]:
    response = requests.get(f"{base_url}/documents", timeout=30)
    response.raise_for_status()
    payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    documents = payload.get("documents") if isinstance(payload, dict) else payload
    visible_sources: Set[str] = set()
    visible_canonical_ids: Set[str] = set()
    source_to_canonical: Dict[str, str] = {}
    for item in documents or []:
        source = normalize_source_name(item.get("filename") or "")
        if not source:
            continue
        canonical_doc_id = canonical_doc_id_from_document(item)
        source_to_canonical[source] = canonical_doc_id
        status = item.get("status") or ""
        chunks_indexed = int(item.get("chunks_indexed") or 0)
        if status in ("completed", "vector_pending") or chunks_indexed > 0:
            visible_sources.add(source)
            if canonical_doc_id:
                visible_canonical_ids.add(canonical_doc_id)
    return {
        "visible_sources": visible_sources,
        "visible_canonical_ids": visible_canonical_ids,
        "source_to_canonical": source_to_canonical,
    }


def has_exact_source_match(expected: Iterable[str], got: Iterable[str]) -> bool:
    expected_set = {normalize_source_name(source) for source in expected if normalize_source_name(source)}
    got_set = {normalize_source_name(source) for source in got if normalize_source_name(source)}
    return bool(expected_set & got_set)


def has_canonical_source_match(expected: Iterable[str], got: Iterable[str], source_to_canonical: Dict[str, str]) -> bool:
    expected_ids = {
        canonical_doc_id_for_source(source, source_to_canonical)
        for source in expected
        if canonical_doc_id_for_source(source, source_to_canonical)
    }
    got_ids = {
        canonical_doc_id_for_source(source, source_to_canonical)
        for source in got
        if canonical_doc_id_for_source(source, source_to_canonical)
    }
    return bool(expected_ids & got_ids)


def source_is_visible_or_equivalent(
    source: str,
    visible_sources: Set[str],
    visible_canonical_ids: Set[str],
    source_to_canonical: Dict[str, str],
) -> bool:
    normalized = normalize_source_name(source)
    if not normalized:
        return False
    if normalized in visible_sources:
        return True
    canonical_doc_id = canonical_doc_id_for_source(normalized, source_to_canonical)
    return bool(canonical_doc_id and canonical_doc_id in visible_canonical_ids)