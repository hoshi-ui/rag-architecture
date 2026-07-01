from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_load_env_file(ROOT / "config" / "app.env")

DEFAULT_CASES = ROOT / "evals" / "cases" / "documents_leveled_query_dataset.json"
DEFAULT_REPORT = ROOT / "reports" / "evals" / "documents_leveled_eval_report.json"
DEFAULT_OUT = ROOT / "reports" / "evals" / "documents_leveled_metrics_report.json"

CITATION_EXACT_RE = re.compile(
    r"《(?P<title>[^》\n\r]+)》\s*(?P<clause>第[一二三四五六七八九十百千万零〇0-9]+条)"
)
PUNCT_RE = re.compile(r"[\s\[\]\(\),.;:!?，。；、：（）《》“”‘’]+")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _cases_by_id(payload: Any) -> dict[str, dict[str, Any]]:
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ValueError("cases file must be a list or an object with a cases list")
    return {str(case.get("id")): case for case in cases if case.get("id")}


def _result_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    if isinstance(payload, list):
        return payload
    raise ValueError("eval report must be a list or an object with a results list")


def _compact(value: Any) -> str:
    return " ".join(str(value or "").split())


def _normalize(value: Any) -> str:
    return PUNCT_RE.sub("", str(value or "")).lower()


def _contains_all(haystack: str, terms: list[str]) -> bool:
    normalized_terms = [_normalize(term) for term in terms if _normalize(term)]
    return bool(normalized_terms) and all(term in haystack for term in normalized_terms)


def _contains_any(haystack: str, terms: list[str]) -> bool:
    return any(_normalize(term) in haystack for term in terms if _normalize(term))


def _aspect_covered_by_label(aspect: str, answer_n: str) -> bool:
    aspect_n = _normalize(aspect)
    if not aspect_n:
        return False
    if aspect_n in answer_n:
        return True

    if "适用地域" in aspect or "适用范围" in aspect or aspect in {"适用主体"}:
        return _contains_any(answer_n, ["行政区域", "本市", "本行政区域", "区域内", "范围内", "单位和个人"])

    if "适用活动" in aspect:
        activity_terms = ["租赁", "治安", "消防", "安全管理", "监督活动", "管理活动"]
        return sum(1 for term in activity_terms if _normalize(term) in answer_n) >= 2

    if "定义" in aspect:
        subject = re.sub(r"(定义|所称|所谓)", "", aspect).strip()
        return bool(subject and _normalize(subject) in answer_n and _contains_any(answer_n, ["是指", "所称", "指"]))

    if aspect.startswith("排除") or "排除" in aspect or "不包括" in aspect:
        tail = re.sub(r"(排除|不包括|不含|除外)", "", aspect)
        parts = [part for part in re.split(r"[和及与、，,]", tail) if len(part.strip()) >= 2]
        if parts and _contains_all(answer_n, parts):
            return _contains_any(answer_n, ["不包括", "不含", "除外", "除"])

    return False


def _safe_div(num: float, den: float) -> float:
    return round(float(num) / float(den), 4) if den else 0.0


NEGATIVE_EXPECTED_BEHAVIORS = {
    "document_not_found",
    "ask_clarification",
    "source_ambiguous",
    "refuse",
    "evidence_insufficient",
    "out_of_scope",
}


def _is_negative_case(case: dict[str, Any]) -> bool:
    behavior = str(case.get("expected_behavior") or "").strip()
    policy = str(case.get("expected_source_policy") or "").strip()
    return bool(
        behavior in NEGATIVE_EXPECTED_BEHAVIORS
        or policy in {"not_found", "document_not_found", "ambiguous", "source_ambiguous", "evidence_insufficient"}
        or case.get("negative_control")
    )


def _mean(values: list[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def _mean_or_none(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def _text_similarity(left: str, right: str) -> float:
    left_n = _normalize(left)
    right_n = _normalize(right)
    if not left_n or not right_n:
        return 0.0
    if left_n in right_n or right_n in left_n:
        return 1.0
    return SequenceMatcher(None, left_n, right_n).ratio()


def _doc_source(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    return str(doc.get("source") or metadata.get("source") or "").strip()


def _canonical_source_key(source: str) -> str:
    value = str(source or "").strip()
    value = re.sub(r"\.[A-Za-z0-9]{1,8}$", "", value)
    value = value.replace("_", " ")
    for _ in range(2):
        current = value.rstrip(" _-./")
        match = re.search(
            r"(?:^|[\s_\-])((?:19|20)\d{2}(?:[-_./年]\d{1,2}(?:[-_./月]\d{1,2}日?)?)?)$",
            current,
        )
        if not match:
            break
        value = current[: match.start()].rstrip(" _-./")
    return re.sub(r"[\s_\-./]+", "", _normalize(value))


def _source_matches(doc: dict[str, Any], ref: dict[str, str]) -> bool:
    left = _doc_source(doc)
    right = ref["source"]
    if left == right:
        return True
    left_key = _canonical_source_key(left)
    right_key = _canonical_source_key(right)
    return bool(left_key and right_key and left_key == right_key)


def _doc_clause(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    clause_meta = metadata.get("clause_metadata") if isinstance(metadata.get("clause_metadata"), dict) else {}
    return str(
        doc.get("clause")
        or doc.get("clause_id")
        or doc.get("article_no")
        or doc.get("article_id")
        or metadata.get("clause")
        or metadata.get("clause_id")
        or metadata.get("article_no")
        or metadata.get("article_id")
        or clause_meta.get("article_no")
        or ""
    ).strip()


def _doc_id(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    clause_meta = metadata.get("clause_metadata") if isinstance(metadata.get("clause_metadata"), dict) else {}
    return str(doc.get("doc_id") or metadata.get("doc_id") or clause_meta.get("doc_id") or _doc_source(doc)).strip()


def _doc_metadata_available(doc: dict[str, Any]) -> bool:
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    clause_meta = metadata.get("clause_metadata") if isinstance(metadata.get("clause_metadata"), dict) else {}
    doc_id = _doc_id(doc)
    article = _doc_clause(doc) or str(clause_meta.get("article_no") or "").strip()
    return bool(doc_id and article)


def _doc_text(doc: dict[str, Any]) -> str:
    return _compact(
        doc.get("text")
        or doc.get("content")
        or doc.get("page_content")
        or doc.get("snippet")
        or doc.get("preview")
        or doc.get("text_preview")
        or ""
    )


def _expected_refs(case: dict[str, Any]) -> list[dict[str, str]]:
    refs = []
    for item in case.get("expected_evidence") or []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()
        clause = str(item.get("clause") or "").strip()
        if source and clause:
            refs.append(
                {
                    "source": source,
                    "title": str(item.get("title") or "").strip(),
                    "clause": clause,
                    "text": str(item.get("text") or ""),
                }
            )
    return refs


def _matches_ref(doc: dict[str, Any], ref: dict[str, str], *, similarity_threshold: float) -> bool:
    source_match = _source_matches(doc, ref)
    if not source_match:
        return False
    clause = _doc_clause(doc)
    text = _doc_text(doc)
    if clause and clause == ref["clause"]:
        return True
    if ref["clause"] and ref["clause"] in text:
        return True
    return _text_similarity(text, ref.get("text") or "") >= similarity_threshold


def _hit_breakdown(doc: dict[str, Any], ref: dict[str, str], *, similarity_threshold: float) -> dict[str, bool]:
    source_hit = _source_matches(doc, ref)
    clause = _doc_clause(doc)
    text = _doc_text(doc)
    clause_id_hit = bool(clause and clause == ref["clause"])
    content_hit = bool(ref["clause"] and ref["clause"] in text) or _text_similarity(text, ref.get("text") or "") >= similarity_threshold
    return {
        "source_hit": source_hit,
        "clause_id_hit": source_hit and clause_id_hit,
        "content_hit": source_hit and content_hit,
        "metadata_available": source_hit and _doc_metadata_available(doc),
    }


def _doc_score(doc: dict[str, Any]) -> float | None:
    for key in ("score", "rerank_score", "relevance_score"):
        if key in doc:
            try:
                return float(doc.get(key) or 0.0)
            except Exception:
                return None
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    for key in ("score", "rerank_score", "hybrid_struct_score", "base_relevance_score"):
        if key in metadata:
            try:
                return float(metadata.get(key) or 0.0)
            except Exception:
                return None
    return None


def _dynamic_elbow_docs(docs: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    safe_top_k = max(1, int(top_k or 1))
    if len(docs) <= safe_top_k:
        return docs
    scores = [_doc_score(doc) for doc in docs]
    if any(score is None for score in scores[: safe_top_k + 1]):
        return docs[:safe_top_k]
    delta_threshold = float(os.getenv("EVAL_DYNAMIC_ELBOW_SCORE_DELTA", "0.025"))
    max_extra = max(0, int(os.getenv("EVAL_DYNAMIC_ELBOW_MAX_EXTRA", "5")))
    hard_limit = min(len(docs), safe_top_k + max_extra)
    limit = safe_top_k
    for idx in range(safe_top_k, hard_limit):
        previous = float(scores[idx - 1] or 0.0)
        current = float(scores[idx] or 0.0)
        if abs(previous - current) > delta_threshold:
            break
        limit = idx + 1
    return docs[:limit]


def _retrieved_docs(result: dict[str, Any], key: str, top_k: int) -> list[dict[str, Any]]:
    retrieved = result.get("retrieved_documents") if isinstance(result.get("retrieved_documents"), dict) else {}
    docs = retrieved.get(key) or retrieved.get("hybrid_rerank") or result.get("documents") or []
    dict_docs = [doc for doc in docs if isinstance(doc, dict)]
    return _dynamic_elbow_docs(dict_docs, top_k)


def _answer_docs(result: dict[str, Any]) -> list[dict[str, Any]]:
    docs = result.get("answer_documents") or []
    return [doc for doc in docs if isinstance(doc, dict)]


def _score_retrieval(
    case: dict[str, Any],
    result: dict[str, Any],
    *,
    retrieval_key: str,
    top_k: int,
    similarity_threshold: float,
) -> dict[str, Any]:
    refs = _expected_refs(case)
    docs = _retrieved_docs(result, retrieval_key, top_k)
    metadata_available_count = sum(1 for doc in docs if _doc_metadata_available(doc))
    relevant_flags = [
        any(_matches_ref(doc, ref, similarity_threshold=similarity_threshold) for ref in refs)
        for doc in docs
    ]
    covered = []
    ref_diagnostics = []
    for ref in refs:
        if any(_matches_ref(doc, ref, similarity_threshold=similarity_threshold) for doc in docs):
            covered.append({"source": ref["source"], "clause": ref["clause"]})
        breakdowns = [_hit_breakdown(doc, ref, similarity_threshold=similarity_threshold) for doc in docs]
        ref_diagnostics.append(
            {
                "source": ref["source"],
                "clause": ref["clause"],
                "source_hit": any(item["source_hit"] for item in breakdowns),
                "clause_id_hit": any(item["clause_id_hit"] for item in breakdowns),
                "content_hit": any(item["content_hit"] for item in breakdowns),
                "metadata_available": any(item["metadata_available"] for item in breakdowns),
            }
        )

    first_rank = None
    for index, flag in enumerate(relevant_flags, start=1):
        if flag:
            first_rank = index
            break

    dcg = 0.0
    for index, flag in enumerate(relevant_flags, start=1):
        if flag:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(refs), top_k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))

    return {
        "context_precision": _safe_div(sum(1 for flag in relevant_flags if flag), min(top_k, len(docs)) or top_k),
        "context_recall": _safe_div(len(covered), len(refs)),
        "mrr": round(1.0 / first_rank, 4) if first_rank else 0.0,
        "ndcg": _safe_div(dcg, idcg),
        "first_correct_rank": first_rank,
        "covered_refs": covered,
        "ref_diagnostics": ref_diagnostics,
        "source_hit_rate": _safe_div(sum(1 for item in ref_diagnostics if item["source_hit"]), len(refs)),
        "clause_id_hit_rate": _safe_div(sum(1 for item in ref_diagnostics if item["clause_id_hit"]), len(refs)),
        "content_hit_rate": _safe_div(sum(1 for item in ref_diagnostics if item["content_hit"]), len(refs)),
        "metadata_available_rate": _safe_div(sum(1 for item in ref_diagnostics if item["metadata_available"]), len(refs)),
        "metadata_coverage_rate": _safe_div(metadata_available_count, len(docs)),
        "metadata_available_count": metadata_available_count,
        "expected_ref_count": len(refs),
        "retrieved_count": len(docs),
    }


def _split_claims(answer: str) -> list[str]:
    claims = []
    for raw in re.split(r"[\n\r]+|(?<=[。！？；;])", answer or ""):
        claim = re.sub(r"^[\-*0-9.、\s]+", "", _compact(raw))
        if len(_normalize(claim)) >= 6:
            claims.append(claim)
    return claims


def _supported_by_any_text(claim: str, support_texts: list[str], threshold: float) -> bool:
    claim_n = _normalize(CITATION_EXACT_RE.sub("", claim))
    if not claim_n:
        return True
    for text in support_texts:
        text_n = _normalize(text)
        if claim_n and claim_n in text_n:
            return True
        fragments = [
            _normalize(part)
            for part in re.split(r"[，。；、：（）(),.;:]", CITATION_EXACT_RE.sub("", claim))
            if len(_normalize(part)) >= 8
        ]
        if fragments and all(fragment in text_n for fragment in fragments[:3]):
            return True
        if SequenceMatcher(None, claim_n, text_n).ratio() >= threshold:
            return True
    return False


def _score_faithfulness(
    case: dict[str, Any],
    result: dict[str, Any],
    *,
    claim_similarity_threshold: float,
) -> dict[str, Any]:
    answer = str(result.get("actual_answer") or result.get("answer") or "")
    claims = _split_claims(answer)
    if _env_flag("DISABLE_STATIC_FAITHFULNESS"):
        return {
            "faithfulness": None,
            "faithfulness_method": "disabled_static",
            "claim_count": len(claims),
            "unsupported_claim_count": None,
            "unsupported_claim_rate": None,
            "unsupported_claims": [],
        }
    evidence_texts = [_doc_text(doc) for doc in _answer_docs(result)]
    if not evidence_texts:
        evidence_texts = [str(item.get("text") or "") for item in case.get("expected_evidence") or [] if isinstance(item, dict)]
    unsupported = [claim for claim in claims if not _supported_by_any_text(claim, evidence_texts, claim_similarity_threshold)]
    return {
        "faithfulness": 1.0 if claims and not unsupported else 0.0,
        "claim_count": len(claims),
        "unsupported_claim_count": len(unsupported),
        "unsupported_claim_rate": _safe_div(len(unsupported), len(claims)),
        "unsupported_claims": unsupported[:5],
    }


def _score_answer_relevance(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    answer = str(result.get("actual_answer") or result.get("answer") or "")
    answer_n = _normalize(answer)
    aspects = [str(item).strip() for item in case.get("expected_aspects") or [] if str(item).strip()]
    hits = [aspect for aspect in aspects if _aspect_covered_by_label(aspect, answer_n)]
    min_required = int(case.get("minimum_required_aspect_count") or max(1, len(aspects)))
    score = _safe_div(len(hits), len(aspects)) if aspects else 0.0
    return {
        "answer_relevance": score,
        "aspect_hit_count": len(hits),
        "aspect_total": len(aspects),
        "minimum_required_aspect_count": min_required,
        "answer_relevance_pass": len(hits) >= min_required if aspects else False,
        "aspect_hits": hits,
    }


def _expected_citations(case: dict[str, Any]) -> set[tuple[str, str]]:
    expected = set()
    for item in case.get("expected_evidence") or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        clause = str(item.get("clause") or "").strip()
        if title and clause:
            expected.add((title, clause))
    return expected


def _extract_citations(answer: str) -> set[tuple[str, str]]:
    return {
        (match.group("title").strip(), match.group("clause").strip())
        for match in CITATION_EXACT_RE.finditer(answer or "")
    }


def _score_citation_exact_match(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    answer = str(result.get("actual_answer") or result.get("answer") or "")
    expected = _expected_citations(case)
    extracted = _extract_citations(answer)
    hits = expected & extracted
    missing = sorted(expected - extracted)
    extra = sorted(extracted - expected)
    recall = _safe_div(len(hits), len(expected)) if expected else 0.0
    precision = _safe_div(len(hits), len(extracted)) if extracted else 0.0
    return {
        "citation_exact_match": 1.0 if expected and extracted == expected else 0.0,
        "citation_recall": recall,
        "citation_precision": precision,
        "citation_hit_count": len(hits),
        "citation_expected_count": len(expected),
        "citation_extracted_count": len(extracted),
        "missing_citations": [{"title": title, "clause": clause} for title, clause in missing],
        "extra_citations": [{"title": title, "clause": clause} for title, clause in extra],
    }


def _metadata_signals(result: dict[str, Any]) -> set[str]:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    signals = set()
    for key in (
        "query_route",
        "internal_route",
        "final_channel",
        "refusal_reason",
        "blocked",
        "refused",
        "source_resolution_status",
        "source_resolution_reason",
    ):
        value = str(metadata.get(key) or "").strip()
        if value:
            signals.add(value)
    trace = metadata.get("source_resolution_trace")
    if isinstance(trace, dict):
        for key in ("reason", "status", "scope_mode", "delayed_clarification_reason"):
            value = str(trace.get(key) or "").strip()
            if value:
                signals.add(value)
    return signals


def _default_expected_signals(case: dict[str, Any]) -> set[str]:
    behavior = str(case.get("expected_behavior") or "").strip()
    policy = str(case.get("expected_source_policy") or "").strip()
    if behavior == "document_not_found" or policy in {"not_found", "document_not_found"}:
        return {"document_not_found", "not_found"}
    if behavior in {"ask_clarification", "source_ambiguous"} or policy in {"ambiguous", "source_ambiguous"}:
        return {"document_clarification", "document_ambiguous", "compare_clarification", "ambiguous"}
    if behavior in {"refuse", "evidence_insufficient", "out_of_scope"} or policy == "evidence_insufficient":
        return {"refusal", "evidence_insufficient", "blocked", "out_of_scope"}
    return set()


def _score_negative_control(case: dict[str, Any], result: dict[str, Any], *, top_k: int, retrieval_key: str) -> dict[str, Any]:
    docs = _retrieved_docs(result, retrieval_key, top_k)
    answer_docs = _answer_docs(result)
    all_docs = docs + [doc for doc in answer_docs if doc not in docs]
    sources = {_doc_source(doc) for doc in all_docs if _doc_source(doc)}
    expected_signals = set(str(item or "").strip() for item in case.get("expected_signals") or [] if str(item or "").strip())
    expected_signals.update(_default_expected_signals(case))
    actual_signals = _metadata_signals(result)
    route_pass = bool(expected_signals & actual_signals) if expected_signals else False

    behavior = str(case.get("expected_behavior") or "").strip()
    policy = str(case.get("expected_source_policy") or "").strip()
    no_retrieval_expected = bool(
        case.get("expected_no_retrieval", behavior in {"document_not_found", "ask_clarification", "source_ambiguous"} or policy in {"not_found", "document_not_found", "ambiguous", "source_ambiguous"})
    )
    no_retrieval_pass = (len(all_docs) == 0) if no_retrieval_expected else True
    forbidden_sources = set(str(item or "").strip() for item in case.get("must_not_use_sources") or [] if str(item or "").strip())
    no_wrong_source_pass = not bool(forbidden_sources & sources)
    answer = str(result.get("actual_answer") or result.get("answer") or "")
    answer_n = _normalize(answer)
    fallback_terms = ["找不到", "未找到", "不存在", "无法", "不能确定", "请明确", "请提供", "未检索到", "证据不足", "不在已上传"]
    answer_signal_pass = _contains_any(answer_n, fallback_terms)
    pass_value = bool(route_pass and no_retrieval_pass and no_wrong_source_pass)
    return {
        "negative_control": True,
        "negative_control_pass": pass_value,
        "negative_control_score": 1.0 if pass_value else 0.0,
        "route_pass": route_pass,
        "no_retrieval_pass": no_retrieval_pass,
        "no_wrong_source_pass": no_wrong_source_pass,
        "answer_signal_pass": answer_signal_pass,
        "expected_signals": sorted(expected_signals),
        "actual_signals": sorted(actual_signals),
        "retrieved_count": len(all_docs),
        "retrieved_sources": sorted(sources),
        "forbidden_sources": sorted(forbidden_sources),
    }


def _score_case(
    case: dict[str, Any],
    result: dict[str, Any],
    *,
    retrieval_key: str,
    top_k: int,
    evidence_similarity_threshold: float,
    claim_similarity_threshold: float,
) -> dict[str, Any]:
    if _is_negative_case(case):
        negative = _score_negative_control(case, result, top_k=top_k, retrieval_key=retrieval_key)
        return {
            "id": case.get("id"),
            "difficulty_category": case.get("difficulty_category") or case.get("category"),
            "difficulty_level": case.get("difficulty_level"),
            "subtype": case.get("subtype"),
            "query": case.get("query"),
            "retrieval": {
                "context_precision": None,
                "context_recall": None,
                "mrr": None,
                "ndcg": None,
                "source_hit_rate": None,
                "clause_id_hit_rate": None,
                "content_hit_rate": None,
                "metadata_available_rate": None,
                "metadata_coverage_rate": None,
                "expected_ref_count": 0,
                "retrieved_count": negative["retrieved_count"],
            },
            "generation": {
                "faithfulness": None,
                "unsupported_claim_rate": None,
                "answer_relevance": negative["negative_control_score"],
                "answer_relevance_pass": negative["negative_control_pass"],
            },
            "rule_based": {
                "citation_exact_match": None,
                "citation_recall": None,
                "citation_precision": None,
                "citation_hit_count": 0,
                "citation_expected_count": 0,
                "citation_extracted_count": 0,
            },
            "negative_control": negative,
        }
    retrieval = _score_retrieval(
        case,
        result,
        retrieval_key=retrieval_key,
        top_k=top_k,
        similarity_threshold=evidence_similarity_threshold,
    )
    faithfulness = _score_faithfulness(case, result, claim_similarity_threshold=claim_similarity_threshold)
    relevance = _score_answer_relevance(case, result)
    citations = _score_citation_exact_match(case, result)
    return {
        "id": case.get("id"),
        "difficulty_category": case.get("difficulty_category") or case.get("category"),
        "difficulty_level": case.get("difficulty_level"),
        "subtype": case.get("subtype"),
        "query": case.get("query"),
        "retrieval": retrieval,
        "generation": {**faithfulness, **relevance},
        "rule_based": citations,
    }


def _bucket_key(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    return str(value or "unknown")


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(path: tuple[str, ...], source_rows: list[dict[str, Any]] | None = None) -> list[float]:
        out = []
        for row in (rows if source_rows is None else source_rows):
            cur: Any = row
            for part in path:
                cur = cur.get(part) if isinstance(cur, dict) else None
            if isinstance(cur, (int, float)):
                out.append(float(cur))
        return out

    positive_rows = [row for row in rows if not row.get("negative_control")]
    negative_rows = [row for row in rows if row.get("negative_control")]

    return {
        "cases": len(rows),
        "positive_cases": len(positive_rows),
        "negative_cases": len(negative_rows),
        "retrieval": {
            "context_precision": _mean(values(("retrieval", "context_precision"), positive_rows)),
            "context_recall": _mean(values(("retrieval", "context_recall"), positive_rows)),
            "mrr": _mean(values(("retrieval", "mrr"), positive_rows)),
            "ndcg": _mean(values(("retrieval", "ndcg"), positive_rows)),
            "source_hit_rate": _mean(values(("retrieval", "source_hit_rate"), positive_rows)),
            "clause_id_hit_rate": _mean(values(("retrieval", "clause_id_hit_rate"), positive_rows)),
            "content_hit_rate": _mean(values(("retrieval", "content_hit_rate"), positive_rows)),
            "metadata_available_rate": _mean(values(("retrieval", "metadata_available_rate"), positive_rows)),
            "metadata_coverage_rate": _mean(values(("retrieval", "metadata_coverage_rate"), positive_rows)),
        },
        "generation": {
            "faithfulness": _mean_or_none(values(("generation", "faithfulness"), positive_rows)),
            "unsupported_claim_rate": _mean_or_none(values(("generation", "unsupported_claim_rate"), positive_rows)),
            "answer_relevance": _mean(values(("generation", "answer_relevance"), positive_rows)),
            "answer_relevance_pass_rate": _mean([
                1.0 if row.get("generation", {}).get("answer_relevance_pass") else 0.0 for row in rows
                if not row.get("negative_control")
            ]),
        },
        "rule_based": {
            "citation_exact_match": _mean(values(("rule_based", "citation_exact_match"), positive_rows)),
            "citation_recall": _mean(values(("rule_based", "citation_recall"), positive_rows)),
            "citation_precision": _mean(values(("rule_based", "citation_precision"), positive_rows)),
        },
        "negative_control": {
            "cases": len(negative_rows),
            "pass_rate": _mean(values(("negative_control", "negative_control_score"))),
            "route_pass_rate": _mean([1.0 if (row.get("negative_control") or {}).get("route_pass") else 0.0 for row in negative_rows]),
            "no_retrieval_pass_rate": _mean([1.0 if (row.get("negative_control") or {}).get("no_retrieval_pass") else 0.0 for row in negative_rows]),
            "no_wrong_source_pass_rate": _mean([1.0 if (row.get("negative_control") or {}).get("no_wrong_source_pass") else 0.0 for row in negative_rows]),
            "answer_signal_pass_rate": _mean([1.0 if (row.get("negative_control") or {}).get("answer_signal_pass") else 0.0 for row in negative_rows]),
        },
    }


def _aggregate_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(_bucket_key(row, key), []).append(row)
    return {bucket: _aggregate(items) for bucket, items in sorted(buckets.items())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score legal RAG retrieval, generation, and citation metrics.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="Golden cases JSON.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Eval runner report JSON.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Metrics report output JSON.")
    parser.add_argument("--retrieval-key", default="hybrid_rerank", help="retrieved_documents bucket to score.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--evidence-similarity-threshold", type=float, default=0.72)
    parser.add_argument("--claim-similarity-threshold", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases_by_id = _cases_by_id(_load_json(Path(args.cases)))
    report = _load_json(Path(args.report))
    rows = []
    for result in _result_items(report):
        case_id = str(result.get("id") or "")
        case = cases_by_id.get(case_id)
        if not case:
            continue
        rows.append(
            _score_case(
                case,
                result,
                retrieval_key=str(args.retrieval_key),
                top_k=max(1, int(args.top_k)),
                evidence_similarity_threshold=float(args.evidence_similarity_threshold),
                claim_similarity_threshold=float(args.claim_similarity_threshold),
            )
        )

    output = {
        "cases_path": str(Path(args.cases).resolve()),
        "eval_report_path": str(Path(args.report).resolve()),
        "top_k": int(args.top_k),
        "retrieval_key": str(args.retrieval_key),
        "summary": {
            "overall": _aggregate(rows),
            "by_difficulty_category": _aggregate_by(rows, "difficulty_category"),
            "by_difficulty_level": _aggregate_by(rows, "difficulty_level"),
            "by_subtype": _aggregate_by(rows, "subtype"),
        },
        "results": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"]["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
