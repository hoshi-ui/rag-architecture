import json
from typing import Any, Dict, Optional

from app.utils.text import sanitize_index_text

from app.documents import ir as document_ir_helpers


def public_task_status(status: Optional[str]) -> str:
    normalized = (status or "").strip().lower()
    if not normalized:
        return "unknown"
    if normalized == "uploaded":
        return "accepted"
    if normalized in {
        "validating",
        "parsing",
        "chunking",
        "embedding",
        "embedding_partial",
        "indexing",
        "indexing_sqlite",
        "indexing_vector",
        "profile_building",
        "publish_pending",
        "reindexing",
        "vector_pending",
    }:
        return "indexing"
    if normalized in {"indexed", "completed"}:
        return "completed"
    if normalized in {
        "failed",
        "parse_failed",
        "parse_empty",
        "parse_low_quality",
        "unsupported_or_corrupt",
        "encrypted_file",
        "suspicious_file_type",
        "index_failed",
        "profile_failed",
        "publish_failed",
        "vector_failed",
        "delete_failed",
    }:
        return "failed"
    if normalized == "deleting":
        return "deleting"
    if normalized == "deleted":
        return "deleted"
    return normalized


def chunk_plain_display_text(text: str) -> str:
    value = sanitize_index_text(str(text or ""))
    if not value:
        return ""
    drop_prefixes = (
        "Document:",
        "Document:",
        "Section:",
        "section:",
        "Page:",
        "Metadata:",
        "文件:",
        "文档:",
        "章节路径:",
        "页码:",
        "元素类型:",
    )

    def drop_wrapper_lines(raw_value: str) -> str:
        lines = []
        for line in raw_value.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if any(stripped.startswith(prefix) for prefix in drop_prefixes):
                continue
            lines.append(stripped)
        return "\n".join(lines).strip() or raw_value.strip()

    if "Content:" in value:
        tail = value.split("Content:", 1)[1]
        for end_marker in ("\nNext context:", "\nPrevious context:", "\nMetadata:", "\n元数据：", "\n元数据:"):
            if end_marker in tail:
                tail = tail.split(end_marker, 1)[0]
        return drop_wrapper_lines(tail)

    for marker in ("正文:", "正文：", "Text:", "Body:"):
        if marker in value:
            tail = value.split(marker, 1)[1]
            for end_marker in ("\n元数据：", "\n元数据:", "\nMetadata:"):
                if end_marker in tail:
                    tail = tail.split(end_marker, 1)[0]
            return drop_wrapper_lines(tail)
    return drop_wrapper_lines(value)


def document_detail_plain_text(document_ir: Optional[Dict[str, Any]], chunks: Any) -> str:
    if document_ir and document_ir.get("parser_name") != "legacy_chunk_backfill":
        return document_ir_helpers.document_ir_plain_text(document_ir, normalized=False)

    blocks = []
    for chunk in chunks or []:
        metadata = chunk.get("metadata") or {}
        candidate = chunk_plain_display_text(metadata.get("raw_text") or chunk.get("text") or "")
        if not candidate:
            continue
        candidate_norm = document_ir_helpers.normalize_ir_text(candidate)
        if not candidate_norm:
            continue
        if blocks:
            previous_norm = document_ir_helpers.normalize_ir_text(blocks[-1])
            if candidate_norm == previous_norm or candidate_norm in previous_norm:
                continue
            if previous_norm and previous_norm in candidate_norm:
                blocks[-1] = candidate
                continue
        blocks.append(candidate)
    return "\n\n".join(blocks).strip()


def summarize_chunk_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    if any(key in payload for key in ("ocr_meta", "ocr_line_meta", "ocr_role", "probe")):
        layout = payload.get("layout") if isinstance(payload.get("layout"), dict) else {}
        layout_summary = {}
        for key in (
            "font_size",
            "font_size_max",
            "font_size_median",
            "is_centered",
            "width_ratio",
            "top_ratio",
            "left_ratio",
            "center_offset_ratio",
            "line_count",
            "bold_ratio",
        ):
            value = layout.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                layout_summary[key] = value
        probe = payload.get("probe") if isinstance(payload.get("probe"), dict) else {}
        summarized = {
            "probe": {key: probe.get(key) for key in ("route", "parser_backend") if probe.get(key) is not None},
            "ocr_role": payload.get("ocr_role"),
            "pdf_role": payload.get("pdf_role"),
            "appendix_label": payload.get("appendix_label"),
            "heading_level": payload.get("heading_level"),
            "ocr_line_index": payload.get("ocr_line_index"),
            "ocr_meta": payload.get("ocr_meta") if isinstance(payload.get("ocr_meta"), dict) else {},
            "ocr_line_meta": payload.get("ocr_line_meta") if isinstance(payload.get("ocr_line_meta"), dict) else {},
            "layout": layout_summary,
        }
        return {key: value for key, value in summarized.items() if value not in (None, {}, [])}
    return payload


MILVUS_METADATA_JSON_LIMIT = 60000
MILVUS_METADATA_STRING_LIMITS = {
    "content": 6000,
    "previous_context": 1200,
    "next_context": 1200,
    "doc_title": 512,
    "canonical_title": 512,
    "section": 512,
    "section_title": 512,
}


def _trim_metadata_string(value: Any, limit: int) -> str:
    text = sanitize_index_text(str(value or ""))
    safe_limit = max(32, int(limit or 0))
    if len(text) <= safe_limit:
        return text
    return text[:safe_limit].rstrip() + "..."


def _compact_metadata_value(key: str, value: Any) -> Any:
    if isinstance(value, str):
        return _trim_metadata_string(value, MILVUS_METADATA_STRING_LIMITS.get(key, 2048))
    if isinstance(value, list):
        compact = []
        for item in value[:32]:
            if isinstance(item, str):
                compact.append(_trim_metadata_string(item, 512))
            elif isinstance(item, (int, float, bool)) or item is None:
                compact.append(item)
            else:
                compact.append(_trim_metadata_string(json.dumps(item, ensure_ascii=False, default=str), 512))
        return compact
    if isinstance(value, dict):
        encoded = json.dumps(value, ensure_ascii=False, default=str)
        if len(encoded) <= 4096:
            return value
        return {"_truncated": True, "preview": _trim_metadata_string(encoded, 4096)}
    return value


def _metadata_json_len(value: Dict[str, Any]) -> int:
    try:
        return len(json.dumps(value or {}, ensure_ascii=False, default=str))
    except Exception:
        return MILVUS_METADATA_JSON_LIMIT + 1


def _fit_milvus_metadata_budget(metadata: Dict[str, Any]) -> Dict[str, Any]:
    if _metadata_json_len(metadata) <= MILVUS_METADATA_JSON_LIMIT:
        return metadata
    fitted = dict(metadata)
    for key, limit in (("content", 2000), ("previous_context", 400), ("next_context", 400)):
        if key in fitted:
            fitted[key] = _trim_metadata_string(fitted.get(key), limit)
        if _metadata_json_len(fitted) <= MILVUS_METADATA_JSON_LIMIT:
            return fitted
    for key in ("payload", "source_resolution_fields", "topics", "aliases"):
        if key in fitted:
            fitted.pop(key, None)
        if _metadata_json_len(fitted) <= MILVUS_METADATA_JSON_LIMIT:
            return fitted
    return {
        key: _compact_metadata_value(key, value)
        for key, value in fitted.items()
        if key not in {"content", "previous_context", "next_context"}
    }


def milvus_safe_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    trimmed = {}
    for key, value in metadata.items():
        if key in {"raw_text", "text_normalized", "fts_text"}:
            continue
        if key == "payload":
            value = summarize_chunk_payload(value)
        trimmed[key] = _compact_metadata_value(key, value)
    return _fit_milvus_metadata_budget(trimmed)
