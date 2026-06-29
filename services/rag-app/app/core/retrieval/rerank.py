"""Rerank strategy helpers.

This module is the stable home for rerank policy.  The retrieval modules still
own broader fusion/search orchestration, while this file exposes the rerank
decision and service-call steps as the target structure expects.
"""

from typing import Any, Dict, List, Optional

from app.core.retrieval.ranking import (
    chunk_level_rerank,
    doc_level_rerank,
    source_level_rerank,
)
from app.core.retrieval.lexical import distinct_hit_sources


def infer_rerank_profile(runtime: Any, query: str, qtype: str) -> str:
    if qtype in {"single_doc_extract", "regulation_execution"}:
        return "section_lookup" if runtime.routing.extract_section_query_targets(query) else "content_lookup"
    if qtype in {"compare", "arch"}:
        return "balanced"
    return "balanced"


def rerank_profile_weights(runtime: Any, profile: str) -> Dict[str, float]:
    defaults = {
        "balanced": {
            "section_term": 0.08,
            "text_term": 0.34,
            "section_overlap": 0.06,
            "keyword": 0.24,
            "title": 0.04,
            "base": 0.24,
        },
        "section_lookup": {
            "section_term": 0.12,
            "text_term": 0.32,
            "section_overlap": 0.08,
            "keyword": 0.24,
            "title": 0.04,
            "base": 0.20,
        },
        "content_lookup": {
            "section_term": 0.04,
            "text_term": 0.38,
            "section_overlap": 0.04,
            "keyword": 0.28,
            "title": 0.02,
            "base": 0.24,
        },
    }
    weights = defaults.get(profile) or defaults["balanced"]
    configured = getattr(runtime.config, "HYBRID_STRUCT_PROFILE_WEIGHTS", None)
    if isinstance(configured, dict) and isinstance(configured.get(profile), dict):
        merged = dict(weights)
        for key, value in configured[profile].items():
            if key not in merged:
                continue
            try:
                merged[key] = float(value)
            except Exception:
                pass
        return merged
    return dict(weights)


def should_apply_chunk_rerank(
    runtime: Any,
    hits: List[Any],
    dense_rank_map: Dict[str, int],
    lex_rank_map: Dict[str, int],
    source_signals: Dict[str, Dict[str, Any]],
    enable_rerank: bool,
) -> bool:
    if (not hits) or (not enable_rerank) or (not runtime.config.ENABLE_RERANK):
        return False
    if not bool(getattr(runtime.config, "ENABLE_CHUNK_RERANK", False)):
        return False
    if len(distinct_hit_sources(runtime.retrieval, hits)) <= 1:
        return True
    if not bool(getattr(runtime.config, "RERANK_LOW_CONF_ONLY", True)):
        return True
    top_dense: Optional[str] = min(dense_rank_map.items(), key=lambda item: item[1])[0] if dense_rank_map else None
    top_lex: Optional[str] = min(lex_rank_map.items(), key=lambda item: item[1])[0] if lex_rank_map else None
    if top_dense and top_lex and top_dense != top_lex:
        return True
    top_src = top_dense or top_lex
    if top_src and source_signals.get(top_src, {}).get("title_hit"):
        return False
    return False
