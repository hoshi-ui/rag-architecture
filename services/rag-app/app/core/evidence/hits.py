import re
import math
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

from app.core.evidence.context import _evidence_context
from app.utils import scoring as scoring_utils


def hit_entity(hit: Any) -> Dict[str, Any]:
    entity = hit.get("entity") if isinstance(hit, dict) else getattr(hit, "entity", None)
    return entity if isinstance(entity, dict) else {}


def hit_metadata(hit: Any) -> Dict[str, Any]:
    metadata = hit_entity(hit).get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def hit_entity_source(hit: Any) -> str:
    return str(hit_entity(hit).get("source") or "")


def hit_entity_text(hit: Any) -> str:
    return str(hit_entity(hit).get("text") or "")


def hit_score(hit: Any) -> float:
    if isinstance(hit, dict):
        if "score" in hit:
            return float(hit.get("score") or 0.0)
        if "distance" in hit:
            return float(hit.get("distance") or 0.0)
    if hasattr(hit, "score"):
        return float(getattr(hit, "score") or 0.0)
    if hasattr(hit, "distance"):
        return float(getattr(hit, "distance") or 0.0)
    return 0.0


def hit_score_mode(hit: Any) -> str:
    if isinstance(hit, dict) and "distance" in hit and "score" not in hit:
        return "distance"
    if hasattr(hit, "distance") and not hasattr(hit, "score"):
        return "distance"
    return "score"


def hit_display_text(hit: Any) -> str:
    metadata = hit_metadata(hit)
    return str(metadata.get("content") or metadata.get("raw_text") or hit_entity_text(hit) or "")


def hit_llm_text(hit: Any) -> str:
    metadata = hit_metadata(hit)
    content = str(metadata.get("content") or "").strip()
    if not content:
        return hit_display_text(hit)
    parts = []
    previous_context = str(metadata.get("previous_context") or "").strip()
    next_context = str(metadata.get("next_context") or "").strip()
    if previous_context:
        parts.append(f"Previous context: {previous_context}")
    parts.append(f"Content: {content}")
    if next_context:
        parts.append(f"Next context: {next_context}")
    return "\n".join(parts)


def doc_section_name(hit: Any, hit_metadata: Callable[[Any], Dict[str, Any]]) -> str:
    metadata = hit_metadata(hit)
    return (metadata.get("section_title") or metadata.get("section") or "").strip()


def is_generic_section_title(section: str) -> bool:
    value = (section or "").strip().lower()
    if not value:
        return False
    generic_keys = [
        "总则",
        "附则",
        "范围",
        "职责",
        "概述",
        "简介",
        "说明",
        "要求",
        "管理",
        "general",
        "appendix",
        "scope",
        "responsibility",
        "overview",
        "introduction",
    ]
    return any(key in value for key in generic_keys)


HEADING_CHUNK_ROLES = {"title", "chapter_heading", "section_heading", "toc", "toc_heading", "appendix_heading"}
STRICT_BODY_ASPECT_MARKERS = {
    "处罚",
    "责任",
    "义务",
    "禁止",
    "不得",
    "应当",
    "必须",
    "程序",
    "期限",
    "标准",
    "条件",
}
STRUCTURAL_ACTION_MARKERS = STRICT_BODY_ASPECT_MARKERS | {
    "法律责任",
    "罚则",
    "罚款",
    "责令",
    "没收",
    "行政处罚",
    "治安管理处罚",
}


def hit_chunk_role(hit: Any, hit_metadata: Callable[[Any], Dict[str, Any]]) -> str:
    return str(hit_metadata(hit).get("chunk_role") or "").strip()


def hit_is_context_expanded(hit: Any, hit_metadata: Callable[[Any], Dict[str, Any]]) -> bool:
    return bool(hit_metadata(hit).get("context_expanded"))


def is_heading_only_hit(hit: Any, hit_metadata: Callable[[Any], Dict[str, Any]]) -> bool:
    metadata = hit_metadata(hit)
    chunk_role = hit_chunk_role(hit, hit_metadata)
    if chunk_role in HEADING_CHUNK_ROLES:
        return True
    section = str(metadata.get("section") or metadata.get("section_title") or "").strip()
    return section == "document_title" or bool(metadata.get("title_hit"))


def text_has_legal_action_signal(text: str) -> bool:
    return bool(re.search(r"(应当|不得|禁止|可以|必须|责令|罚款|处罚|承担|依法|申请|备案|审批|登记)", text or ""))


def is_substantive_short_legal_evidence(hit: Any, hit_metadata: Callable[[Any], Dict[str, Any]], hit_display_text: Callable[[Any], str]) -> bool:
    text = (hit_display_text(hit) or "").strip()
    if not text:
        return False
    metadata = hit_metadata(hit)
    chunk_role = hit_chunk_role(hit, hit_metadata)
    if chunk_role in HEADING_CHUNK_ROLES:
        return False
    if chunk_role in {"article", "clause"}:
        return True
    if str(metadata.get("article_no") or metadata.get("clause_label") or "").strip():
        return True
    return False


def has_clause_like_body_evidence(hit: Any, hit_metadata: Callable[[Any], Dict[str, Any]], hit_display_text: Callable[[Any], str]) -> bool:
    if is_heading_only_hit(hit, hit_metadata):
        return False
    metadata = hit_metadata(hit)
    chunk_role = hit_chunk_role(hit, hit_metadata)
    if chunk_role in {"article", "clause"}:
        return True
    if str(metadata.get("article_no") or metadata.get("clause_label") or "").strip():
        return True
    text = hit_display_text(hit) or ""
    return bool(re.search(r"第[一二三四五六七八九十百千万0-9]+[条款项]", text))


def aspect_requires_body_evidence(runtime: Any, term: str) -> bool:
    adapter = _evidence_context(runtime)
    normalized = adapter.normalize_coverage_aspect(term) or adapter.normalize_query(term)
    if not normalized:
        return False
    return any(marker in normalized for marker in STRICT_BODY_ASPECT_MARKERS)


def _metadata_text(metadata: Dict[str, Any], *keys: str) -> str:
    parts: List[str] = []
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, list):
            parts.extend(str(item or "").strip() for item in value if str(item or "").strip())
        elif isinstance(value, dict):
            parts.extend(str(item or "").strip() for item in value.values() if str(item or "").strip())
        elif str(value or "").strip():
            parts.append(str(value or "").strip())
    return " ".join(parts)


def _chunk_metadata(chunk: Any) -> Dict[str, Any]:
    metadata = dict((chunk or {}).get("metadata") or {})
    for key in (
        "section",
        "section_title",
        "section_node_id",
        "section_path",
        "parent_section_id",
        "parent_section_path",
        "parent_section_title",
        "article_id",
        "article_no",
        "clause_label",
        "chunk_role",
        "reading_order",
    ):
        if (chunk or {}).get(key) is not None and metadata.get(key) is None:
            metadata[key] = (chunk or {}).get(key)
    return metadata


def _article_key(metadata: Dict[str, Any]) -> str:
    return str(metadata.get("article_id") or metadata.get("article_no") or metadata.get("clause_label") or "").strip()


def _related_structural_text(runtime: Any, doc: Any) -> str:
    adapter = _evidence_context(runtime)
    metadata = adapter.hit_metadata(doc)
    parts = [
        _metadata_text(
            metadata,
            "section",
            "section_title",
            "section_path",
            "parent_section_title",
            "parent_section_path",
            "article_id",
            "article_no",
            "clause_label",
        )
    ]
    source = adapter.normalize_filename_for_match(adapter.hit_entity_source(doc) or "")
    if not source:
        return adapter.normalize_query(" ".join(parts))
    doc_version = _to_int(metadata.get("doc_version"))
    try:
        chunks = adapter.get_chunks_for_source(source, doc_version)
    except Exception:
        chunks = []

    article = _article_key(metadata)
    section_node_id = str(metadata.get("section_node_id") or "").strip()
    parent_section_id = str(metadata.get("parent_section_id") or "").strip()
    section_title = str(metadata.get("section_title") or metadata.get("section") or "").strip()
    for chunk in chunks or []:
        chunk_metadata = _chunk_metadata(chunk)
        chunk_article = _article_key(chunk_metadata)
        same_article = bool(article and chunk_article and article == chunk_article)
        same_section = bool(
            (section_node_id and str(chunk_metadata.get("section_node_id") or "").strip() == section_node_id)
            or (
                parent_section_id
                and parent_section_id
                in {
                    str(chunk_metadata.get("parent_section_id") or "").strip(),
                    str(chunk_metadata.get("section_node_id") or "").strip(),
                }
            )
            or (
                section_title
                and section_title
                == str(chunk_metadata.get("section_title") or chunk_metadata.get("section") or "").strip()
            )
        )
        if not (same_article or same_section):
            continue
        chunk_role = str(chunk_metadata.get("chunk_role") or "").strip()
        parts.append(
            _metadata_text(
                chunk_metadata,
                "section",
                "section_title",
                "section_path",
                "parent_section_title",
                "parent_section_path",
                "article_id",
                "article_no",
                "clause_label",
            )
        )
        if same_article or chunk_role in HEADING_CHUNK_ROLES:
            parts.append(str(chunk_metadata.get("raw_text") or (chunk or {}).get("raw_text") or (chunk or {}).get("text") or ""))
    return adapter.normalize_query(" ".join(part for part in parts if part))


def _aspect_variants(adapter: Any, normalized_aspect: str) -> List[str]:
    variants = _aspect_variants(adapter, normalized_aspect)
    expansion_map = {
        "处罚": ["法律责任", "罚则", "罚款", "责令", "没收", "行政处罚"],
        "责任": ["法律责任", "罚则", "责令", "承担"],
        "禁止": ["不得", "禁止"],
    }
    for item in expansion_map.get(normalized_aspect, []):
        value = adapter.normalize_query(item)
        if value and value not in variants:
            variants.append(value)
    return variants


def doc_aspect_evidence_features(runtime: Any, doc: Any, aspect: str) -> Dict[str, Any]:
    adapter = _evidence_context(runtime)
    normalized_aspect = adapter.normalize_coverage_aspect(aspect) or adapter.normalize_query(aspect)
    if not normalized_aspect:
        return {
            "normalized_aspect": "",
            "body_hits": 0,
            "section_hits": 0,
            "body_exact": 0.0,
            "section_exact": 0.0,
            "context_expanded": False,
            "heading_only": False,
            "qualifies": False,
        }
    section_text = adapter.normalize_query(adapter.doc_section_name(doc))
    body_text = adapter.normalize_query(adapter.chunk_plain_display_text(adapter.hit_display_text(doc)))
    if not body_text:
        body_text = adapter.normalize_query(adapter.hit_display_text(doc))
    variants = adapter.coverage_aspect_variants(normalized_aspect) or [normalized_aspect]
    if len(normalized_aspect) >= 4:
        for tail_len in (2, 3, 4):
            if len(normalized_aspect) > tail_len:
                tail = normalized_aspect[-tail_len:]
                if len(tail) >= 2 and tail not in variants:
                    variants.append(tail)
    chunk_role = hit_chunk_role(doc, adapter.hit_metadata)
    section_path = adapter.hit_metadata(doc).get("section_path") or []
    in_toc = bool(section_path and str(section_path[0] or "").strip().lower() == "toc")
    heading_only = is_heading_only_hit(doc, adapter.hit_metadata)
    title_like = bool(adapter.hit_metadata(doc).get("title_hit")) or chunk_role == "title" or adapter.doc_section_name(doc) == "document_title"
    toc_like = chunk_role in {"toc", "toc_heading"} or in_toc
    context_expanded = hit_is_context_expanded(doc, adapter.hit_metadata)
    inherited_text = _related_structural_text(adapter, doc)
    body_hits = sum(1 for variant in variants if variant and variant in body_text)
    inherited_hits = sum(1 for variant in variants if variant and variant in inherited_text)
    section_hits = 0 if title_like or toc_like else sum(1 for variant in variants if variant and variant in section_text)
    body_exact = 1.0 if normalized_aspect and normalized_aspect in body_text else 0.0
    inherited_exact = 1.0 if normalized_aspect and normalized_aspect in inherited_text else 0.0
    section_exact = 1.0 if normalized_aspect and normalized_aspect in section_text else 0.0
    requires_body = aspect_requires_body_evidence(adapter, normalized_aspect)
    inherited_action_signal = any(adapter.normalize_query(marker) in inherited_text for marker in STRUCTURAL_ACTION_MARKERS)
    substantive = bool(
        is_substantive_short_legal_evidence(doc, adapter.hit_metadata, adapter.hit_display_text)
        or (chunk_role == "body" and inherited_action_signal)
    )
    semantic_match = bool(body_hits > 0 or inherited_hits > 0 or (not requires_body and section_hits > 0))
    qualifies = bool(
        not context_expanded
        and not title_like
        and not toc_like
        and not heading_only
        and substantive
        and semantic_match
    )
    return {
        "normalized_aspect": normalized_aspect,
        "body_hits": body_hits,
        "inherited_hits": inherited_hits,
        "section_hits": section_hits,
        "body_exact": body_exact,
        "inherited_exact": inherited_exact,
        "section_exact": section_exact,
        "context_expanded": context_expanded,
        "heading_only": heading_only,
        "inherited_action_signal": inherited_action_signal,
        "qualifies": qualifies,
    }


def doc_matches_semantic_aspect(runtime: Any, doc: Any, aspect: str) -> bool:
    return bool(doc_aspect_evidence_features(runtime, doc, aspect).get("qualifies"))


def aspect_doc_priority_score(runtime: Any, doc: Any, aspect: str, rank_index: int) -> float:
    adapter = _evidence_context(runtime)
    features = doc_aspect_evidence_features(adapter, doc, aspect)
    normalized_aspect = str(features.get("normalized_aspect") or "")
    if not normalized_aspect or not bool(features.get("qualifies")):
        return -1.0
    section_hits = int(features.get("section_hits") or 0)
    body_hits = int(features.get("body_hits") or 0)
    inherited_hits = int(features.get("inherited_hits") or 0)
    section_exact = float(features.get("section_exact") or 0.0)
    body_exact = float(features.get("body_exact") or 0.0)
    inherited_exact = float(features.get("inherited_exact") or 0.0)
    generic_penalty = adapter.aspect_generic_section_penalty if is_generic_section_title(adapter.doc_section_name(doc)) else 0.0
    rank_bonus = max(0.0, adapter.aspect_rank_bonus_base - (adapter.aspect_rank_bonus_decay * max(0, rank_index - 1)))
    clause_bonus = adapter.aspect_clause_bonus if has_clause_like_body_evidence(doc, adapter.hit_metadata, adapter.hit_display_text) else 0.0
    substantive_bonus = adapter.aspect_substantive_bonus if is_substantive_short_legal_evidence(doc, adapter.hit_metadata, adapter.hit_display_text) else 0.0
    return (
        (adapter.aspect_body_hit_weight * body_hits)
        + (adapter.aspect_inherited_hit_weight * inherited_hits)
        + (adapter.aspect_body_exact_weight * body_exact)
        + (adapter.aspect_inherited_exact_weight * inherited_exact)
        + (adapter.aspect_section_hit_weight * section_hits)
        + (adapter.aspect_section_exact_weight * section_exact)
        + clause_bonus
        + substantive_bonus
        + rank_bonus
        - generic_penalty
    )


def doc_semantic_aspect_hits(runtime: Any, doc: Any, aspect_terms: List[str]) -> List[str]:
    if not aspect_terms:
        return []
    matched: List[str] = []
    for term in aspect_terms:
        features = doc_aspect_evidence_features(runtime, doc, term)
        normalized_term = scoring_utils.normalize_core_aspect_term(str(features.get("normalized_aspect") or ""))
        if not normalized_term or normalized_term in matched:
            continue
        if bool(features.get("qualifies")):
            matched.append(normalized_term)
    return matched


def hit_chunk_id(hit: Any, hit_metadata: Callable[[Any], Dict[str, Any]]) -> Optional[int]:
    metadata = hit_metadata(hit)
    value = metadata.get("chunk_id")
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def hit_chunk_range(hit: Any, hit_metadata: Callable[[Any], Dict[str, Any]]) -> str:
    metadata = hit_metadata(hit)
    start = metadata.get("chunk_id_start")
    end = metadata.get("chunk_id_end")
    if start is not None and end is not None:
        try:
            start_i = int(start)
            end_i = int(end)
            return f"{start_i}-{end_i}" if start_i != end_i else f"{start_i}"
        except Exception:
            pass
    chunk_id = hit_chunk_id(hit, hit_metadata)
    return f"{chunk_id}" if chunk_id is not None else ""


def _to_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def chunk_position_id(hit: Any, hit_metadata: Callable[[Any], Dict[str, Any]]) -> Optional[int]:
    metadata = hit_metadata(hit)
    for key in ("chunk_id_start", "chunk_id", "reading_order", "order", "idx"):
        value = _to_int(metadata.get(key))
        if value is not None:
            return value
    value = _to_int(metadata.get("chunk_id_end"))
    if value is not None:
        return value
    return None


def build_excerpt(text: str, query: str, max_chars: int, normalize_query: Callable[[str], str]) -> str:
    body = (text or "").strip()
    if not body:
        return ""
    normalized_query = normalize_query(query)
    terms = [word for word in re.sub(r"[?,，。；;:：、\s]+", " ", normalized_query).split() if len(word) >= 2]
    terms = list(dict.fromkeys(terms))[:8]
    if not terms or len(body) <= max_chars:
        return body[:max_chars]
    lowered = body.lower()
    best_index = -1
    for term in terms:
        index = lowered.find(term.lower())
        if index != -1:
            best_index = index
            break
    if best_index == -1:
        return body[:max_chars]
    half = max_chars // 2
    start = max(0, best_index - half)
    end = min(len(body), start + max_chars)
    start = max(0, end - max_chars)
    snippet = body[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(body):
        snippet = snippet + "..."
    return snippet


@lru_cache(maxsize=8)
def _token_encoder(model_name: str) -> Any:
    try:
        import tiktoken
    except Exception:
        return None
    try:
        return tiktoken.encoding_for_model(model_name) if model_name else tiktoken.get_encoding("cl100k_base")
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def estimate_token_count(text: str, model_name: str = "") -> int:
    if not text:
        return 0
    encoder = _token_encoder((model_name or "").strip())
    if encoder is None:
        return max(1, math.ceil(len(text) / 4))
    try:
        return len(encoder.encode(text))
    except Exception:
        return max(1, math.ceil(len(text) / 4))


def evidence_relevance(score: float, score_mode: str, best_score: float) -> float:
    if score_mode == "distance":
        best_sim = 1.0 / (1.0 + max(best_score, 0.0))
        sim = 1.0 / (1.0 + max(score, 0.0))
        return min(max(sim / max(best_sim, 1e-9), 0.0), 1.0)
    if best_score <= 0:
        return 0.0
    return min(max(score / best_score, 0.0), 1.0)
