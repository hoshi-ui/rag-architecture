"""Clause-level reranking helpers for source-locked retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.documents import chunking as document_chunking
from app.core.legal_intent import classify_query_intent_fallback, normalize_legal_intent


@dataclass
class ClauseUnit:
    doc_id: str
    article_no: Optional[str]
    clause_no: Optional[str]
    item_no: Optional[str]
    chapter_title: Optional[str]
    section_title: Optional[str]
    heading: Optional[str]
    text: str
    prev_article_no: Optional[str]
    next_article_no: Optional[str]


def _hit_entity(hit: Any) -> Dict[str, Any]:
    if isinstance(hit, dict):
        entity = hit.get("entity")
        return entity if isinstance(entity, dict) else {}
    entity = getattr(hit, "entity", None)
    return entity if isinstance(entity, dict) else {}


def _hit_metadata(hit: Any) -> Dict[str, Any]:
    entity = _hit_entity(hit)
    metadata = entity.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _hit_text(hit: Any) -> str:
    entity = _hit_entity(hit)
    return str(entity.get("text") or entity.get("raw_text") or "").strip()


def _hit_source(hit: Any) -> str:
    entity = _hit_entity(hit)
    return str(entity.get("source") or "").strip()


def _hit_score(hit: Any) -> float:
    if isinstance(hit, dict):
        try:
            return float(hit.get("score") or 0.0)
        except Exception:
            return 0.0
    try:
        return float(getattr(hit, "score", 0.0) or 0.0)
    except Exception:
        return 0.0


def _clone_hit_with_score(runtime: Any, hit: Any, score: float) -> Any:
    clone = getattr(runtime, "clone_hit_with_score", None)
    if callable(clone):
        return clone(hit, score)
    entity = _hit_entity(hit)
    return {"entity": entity, "score": float(score)}


def _metadata_article(metadata: Dict[str, Any], text: str = "") -> str:
    return document_chunking.extract_article_id(
        metadata.get("article_id"),
        metadata.get("article_no"),
        metadata.get("clause_label"),
        text,
    )


def _reading_order(hit: Any) -> Tuple[int, int]:
    metadata = _hit_metadata(hit)
    for key in ("reading_order", "chunk_id", "page", "page_no"):
        try:
            return (0, int(metadata.get(key)))
        except Exception:
            continue
    return (1, 0)


def _norm_terms(text: str) -> List[str]:
    value = re.sub(r"[^\w\u4e00-\u9fff]+", " ", str(text or "").lower())
    terms = [part for part in value.split() if len(part) >= 2]
    if terms:
        return terms
    return re.findall(r"[\u4e00-\u9fff]{2,}", str(text or ""))


def _overlap(query: str, text: str) -> bool:
    query_terms = set(_norm_terms(query))
    if not query_terms:
        return False
    text_value = str(text or "").lower()
    return any(term in text_value for term in query_terms)


def _clause_structural_text(unit: ClauseUnit) -> str:
    return " ".join(
        part
        for part in [
            unit.chapter_title,
            unit.section_title,
            unit.heading,
            unit.article_no,
            unit.text,
        ]
        if part
    )


def _article_number_value(article_no: Optional[str]) -> Optional[int]:
    normalized = document_chunking.normalize_article_id(article_no or "")
    if not normalized:
        return None
    digits = re.findall(r"\d+", normalized)
    if digits:
        try:
            return int(digits[0])
        except Exception:
            return None
    numerals = "零〇一二三四五六七八九十"
    body = normalized.removeprefix("第").removesuffix("条")
    if not body or any(ch not in numerals for ch in body):
        return None
    if body == "十":
        return 10
    if "十" in body:
        left, _, right = body.partition("十")
        tens = 1 if not left else numerals.index(left)
        ones = 0 if not right else numerals.index(right)
        return tens * 10 + ones
    try:
        return numerals.index(body)
    except Exception:
        return None


def _classify_clause_intent(unit: ClauseUnit) -> str:
    text = _clause_structural_text(unit)
    lower = text.lower()
    if any(term in text for term in ("法律责任", "罚则", "罚款", "没收", "责令", "吊销", "处罚")):
        return "法律责任"
    if any(term in text for term in ("职责", "权限", "职权", "负责", "主管", "监督管理")):
        return "职责与权限"
    if any(term in text for term in ("程序", "流程", "申请", "审查", "办理", "登记", "期限", "条件")):
        return "程序与条件"
    if any(term in text for term in ("定义", "含义", "所称", "适用", "范围", "包括", "不包括")):
        return "定义与范围"
    if any(term in text for term in ("义务", "权利", "应当", "不得", "禁止", "可以")):
        return "权利义务"
    if any(term in lower for term in ("liability", "penalty", "fine")):
        return "法律责任"
    return "其他"


def _clause_key(hit: Any) -> Tuple[str, str]:
    metadata = _hit_metadata(hit)
    source = _hit_source(hit)
    article = _metadata_article(metadata, _hit_text(hit))
    if article:
        return (source, article)
    chunk_id = str(metadata.get("chunk_id") or metadata.get("reading_order") or _hit_text(hit)[:32])
    return (source, f"chunk::{chunk_id}")


def build_clause_units(hits: List[Any]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Any]] = {}
    for hit in hits or []:
        groups.setdefault(_clause_key(hit), []).append(hit)

    ordered: List[Dict[str, Any]] = []
    for key, group_hits in groups.items():
        group_hits = sorted(group_hits, key=_reading_order)
        first = group_hits[0]
        metadata = _hit_metadata(first)
        texts = [_hit_text(hit) for hit in group_hits if _hit_text(hit)]
        article = _metadata_article(metadata, texts[0] if texts else "")
        source = _hit_source(first)
        unit = ClauseUnit(
            doc_id=str(metadata.get("doc_id") or metadata.get("document_id") or source or key[0]),
            article_no=article or None,
            clause_no=str(metadata.get("clause_no") or "").strip() or None,
            item_no=str(metadata.get("item_no") or "").strip() or None,
            chapter_title=str(metadata.get("chapter_title") or metadata.get("chapter") or "").strip() or None,
            section_title=str(metadata.get("section_title") or metadata.get("section") or "").strip() or None,
            heading=str(metadata.get("heading") or metadata.get("title") or metadata.get("clause_label") or article or "").strip() or None,
            text="\n".join(texts)[:3000],
            prev_article_no=None,
            next_article_no=None,
        )
        ordered.append(
            {
                "key": key,
                "unit": unit,
                "hits": group_hits,
                "base_score": max(_hit_score(hit) for hit in group_hits),
                "order": _reading_order(first),
            }
        )

    ordered.sort(key=lambda item: item["order"])
    article_positions = [item for item in ordered if item["unit"].article_no]
    for idx, item in enumerate(article_positions):
        item["unit"].prev_article_no = article_positions[idx - 1]["unit"].article_no if idx > 0 else None
        item["unit"].next_article_no = article_positions[idx + 1]["unit"].article_no if idx + 1 < len(article_positions) else None
    return ordered


def clause_aware_text(unit: ClauseUnit, doc_title: str = "", query_intent: str = "") -> str:
    parts = []
    clause_intent = _classify_clause_intent(unit)
    normalized_query_intent = normalize_legal_intent(query_intent)
    if normalized_query_intent:
        parts.append(f"查询意图：{normalized_query_intent}")
    if clause_intent and clause_intent != "其他":
        parts.append(f"条款意图：{clause_intent}")
    if doc_title:
        parts.append(f"文档：《{doc_title}》")
    if unit.chapter_title:
        parts.append(f"章节：{unit.chapter_title}")
    if unit.section_title and unit.section_title != unit.chapter_title:
        parts.append(f"小节：{unit.section_title}")
    if unit.article_no:
        parts.append(f"条款：{unit.article_no}")
    if unit.heading:
        parts.append(f"条款主题：{unit.heading}")
    parts.append(f"正文：{unit.text}")
    return "\n".join(parts)


def _adjust_clause_score(
    query: str,
    unit: ClauseUnit,
    score: float,
    mentioned_articles: List[str],
    query_intent: str = "",
    config: Any = None,
) -> Tuple[float, Dict[str, Any]]:
    heading_text = " ".join(part for part in [unit.heading, unit.chapter_title, unit.section_title] if part)
    heading_matches = _overlap(query, heading_text)
    topic_matches = _overlap(query, unit.text[:300])
    exact_article_mentioned = bool(mentioned_articles)
    normalized_query_intent = normalize_legal_intent(query_intent) or classify_query_intent_fallback(query)
    clause_intent = _classify_clause_intent(unit)
    article_number = _article_number_value(unit.article_no)
    adjusted = float(score)
    reasons: List[str] = []
    if exact_article_mentioned and unit.article_no and unit.article_no not in mentioned_articles:
        adjusted -= float(getattr(config, "CLAUSE_RERANK_NON_EXACT_ARTICLE_PENALTY", 0.25))
        reasons.append("non_exact_article_penalty")
    if heading_matches:
        adjusted += float(getattr(config, "CLAUSE_RERANK_HEADING_MATCH_BONUS", 0.15))
        reasons.append("heading_match_bonus")
    if topic_matches:
        adjusted += float(getattr(config, "CLAUSE_RERANK_TOPIC_MATCH_BONUS", 0.15))
        reasons.append("clause_topic_match_bonus")
    if normalized_query_intent and clause_intent == normalized_query_intent and clause_intent != "其他":
        adjusted += float(getattr(config, "CLAUSE_RERANK_INTENT_MATCH_BONUS", 0.18))
        reasons.append("legal_intent_match_bonus")
    if normalized_query_intent == "法律责任" and any(term in heading_text for term in ("法律责任", "罚则")):
        adjusted += float(getattr(config, "CLAUSE_RERANK_INTENT_HEADING_BONUS", 0.12))
        reasons.append("legal_responsibility_heading_bonus")
    if normalized_query_intent == "定义与范围" and article_number is not None and 1 <= article_number <= 3:
        adjusted += float(getattr(config, "CLAUSE_RERANK_INTENT_HEADING_BONUS", 0.12))
        reasons.append("definition_scope_early_article_bonus")
    if normalized_query_intent == "职责与权限" and any(term in heading_text for term in ("职责", "权限", "职权", "机构职责")):
        adjusted += float(getattr(config, "CLAUSE_RERANK_INTENT_HEADING_BONUS", 0.12))
        reasons.append("duty_authority_heading_bonus")
    if normalized_query_intent == "程序与条件" and any(term in heading_text for term in ("程序", "条件", "办理", "登记", "审查")):
        adjusted += float(getattr(config, "CLAUSE_RERANK_INTENT_HEADING_BONUS", 0.12))
        reasons.append("procedure_condition_heading_bonus")
    if (
        exact_article_mentioned
        and unit.article_no
        and unit.article_no not in mentioned_articles
        and (unit.prev_article_no in mentioned_articles or unit.next_article_no in mentioned_articles)
        and not heading_matches
    ):
        adjusted -= float(getattr(config, "CLAUSE_RERANK_NEIGHBOR_ARTICLE_PENALTY", 0.10))
        reasons.append("neighbor_without_heading_penalty")
    return adjusted, {
        "article_no": unit.article_no or "",
        "query_intent": normalized_query_intent,
        "clause_intent": clause_intent,
        "base_score": float(score),
        "adjusted_score": adjusted,
        "reasons": reasons,
    }


async def clause_level_rerank(
    runtime: Any,
    rerank_service: Any,
    query: str,
    hits: List[Any],
    top_k: int,
    enable_rerank: bool,
    mentioned_articles: Optional[List[str]] = None,
    doc_title: str = "",
    query_intent: str = "",
) -> Dict[str, Any]:
    clauses = build_clause_units(hits)
    if not clauses:
        return {"hits": hits[:top_k], "used": False, "score_mode": "score", "trace": {"clause_units": 0}}

    mentioned_articles = [
        document_chunking.normalize_article_id(article)
        for article in (mentioned_articles or [])
        if document_chunking.normalize_article_id(article)
    ]
    if mentioned_articles and any(item["unit"].article_no in mentioned_articles for item in clauses):
        clauses = [item for item in clauses if item["unit"].article_no in mentioned_articles]

    scores = [float(item["base_score"]) for item in clauses]
    used = False
    if enable_rerank and rerank_service is not None and len(clauses) > 1:
        try:
            reranked = await rerank_service.rerank(
                query=query,
                documents=[clause_aware_text(item["unit"], doc_title=doc_title, query_intent=query_intent) for item in clauses],
                top_k=len(clauses),
            )
            rerank_scores = [0.0 for _ in clauses]
            for item in reranked or []:
                idx = item.get("index") if isinstance(item, dict) else getattr(item, "index", None)
                score = item.get("score", 0.0) if isinstance(item, dict) else getattr(item, "score", 0.0)
                idx_i = int(idx)
                if 0 <= idx_i < len(rerank_scores):
                    rerank_scores[idx_i] = float(score)
            if any(score != 0.0 for score in rerank_scores):
                scores = rerank_scores
                used = True
        except Exception:
            used = False

    ranked: List[Tuple[float, int, Dict[str, Any]]] = []
    adjustments: List[Dict[str, Any]] = []
    for idx, item in enumerate(clauses):
        adjusted, trace = _adjust_clause_score(
            query,
            item["unit"],
            scores[idx],
            mentioned_articles,
            query_intent=query_intent,
            config=getattr(runtime, "config", None),
        )
        adjustments.append(trace)
        ranked.append((adjusted, idx, trace))
    ranked.sort(key=lambda item: item[0], reverse=True)

    selected_hits: List[Any] = []
    selected_articles: List[str] = []
    for adjusted, idx, _trace in ranked:
        unit = clauses[idx]["unit"]
        if unit.article_no:
            selected_articles.append(unit.article_no)
        for hit in clauses[idx]["hits"]:
            selected_hits.append(_clone_hit_with_score(runtime, hit, adjusted))
            if len(selected_hits) >= top_k:
                break
        if len(selected_hits) >= top_k:
            break

    return {
        "hits": selected_hits,
        "used": used or bool(mentioned_articles),
        "score_mode": "score",
        "trace": {
            "clause_units": len(clauses),
            "query_intent": normalize_legal_intent(query_intent) or classify_query_intent_fallback(query),
            "mentioned_articles": mentioned_articles,
            "selected_articles": selected_articles[: max(1, min(5, top_k))],
            "adjustments": adjustments[:10],
            "rerank_used": used,
        },
    }
