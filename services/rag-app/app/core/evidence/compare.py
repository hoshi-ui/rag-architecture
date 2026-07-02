import asyncio
import re
from typing import Any, Callable, Dict, List, Optional

from app.core.evidence.context import _evidence_context
from app.core.evidence.format import fit_evidence_block_to_budget
from app.core.evidence.hits import (
    doc_semantic_aspect_hits,
    estimate_token_count,
    evidence_relevance,
    is_generic_section_title,
    is_heading_only_hit,
    is_substantive_short_legal_evidence,
)

DOC_NAMESPACE_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _doc_namespace(group_no: int) -> str:
    index = max(1, int(group_no)) - 1
    if index < len(DOC_NAMESPACE_ALPHABET):
        return DOC_NAMESPACE_ALPHABET[index]
    return f"D{group_no}"


def compare_answer_snippet(runtime: Any, doc: Any, limit: int = 120) -> str:
    snippet = re.sub(r"\s+", " ", _evidence_context(runtime).hit_display_text(doc) or "").strip()
    if len(snippet) > limit:
        snippet = snippet[:limit].rstrip() + "..."
    return snippet


def summarize_compare_source_blocks(
    *,
    title: str,
    evidence_query: str,
    focus_text: str,
    blocks: List[str],
    status: str = "",
) -> str:
    status_suffix = f" | 状态：{status}" if status else ""
    heading = f"来源小结：{title} | 检索主题：{evidence_query}{status_suffix}"
    if focus_text:
        heading += f" | 对比焦点：{focus_text}"
    if not blocks:
        return heading + "\n未找到可用于该来源的直接证据。"
    return heading + "\n" + "\n\n".join(blocks)


def format_compare_evidence(
    runtime: Any,
    source_groups: List[Dict[str, Any]],
    query: str,
    score_mode: str,
    *,
    compare_focus_text: Callable[[Optional[Dict[str, Any]]], str],
    token_budget: int,
    model_name: str = "",
    compare_plan: Optional[Dict[str, Any]] = None,
    compare_source_statuses: Optional[List[Dict[str, Any]]] = None,
) -> tuple[str, List[Dict[str, Any]]]:
    adapter = _evidence_context(runtime)
    lines: List[str] = []
    refs: List[Dict[str, Any]] = []
    total_tokens = 0
    max_tokens = max(256, int(token_budget))
    focus_text = compare_focus_text(compare_plan)
    status_map = {
        adapter.normalize_filename_for_match((item or {}).get("source") or ""): str((item or {}).get("status") or "")
        for item in (compare_source_statuses or [])
        if adapter.normalize_filename_for_match((item or {}).get("source") or "")
    }
    header = f"对比证据焦点：{focus_text}\n引用规则：每部法规都有独立编号空间，回答时必须使用对应法规下的编号，例如 [A-1]、[B-1]；禁止用其他法规编号支撑本法规结论。"
    header_tokens = estimate_token_count(header, model_name) + 2
    if header_tokens <= max_tokens:
        lines.append(header)
        total_tokens += header_tokens
    flattened_docs = [doc for group in source_groups or [] for doc in (group.get("docs") or [])]
    best_score = adapter.hit_score(flattened_docs[0]) if flattened_docs else 0.0
    evidence_index = 1
    for group_no, group in enumerate(source_groups or [], start=1):
        doc_namespace = _doc_namespace(group_no)
        source = adapter.normalize_filename_for_match((group or {}).get("source") or "")
        title = adapter.source_display_title(source) if source else "未知来源"
        evidence_query = adapter.normalize_query((group or {}).get("evidence_query") or query) or query
        group_head = f"【《{title}》】\n文档命名空间：{doc_namespace} | 检索问题：{evidence_query} | 本法规证据编号只能使用 [{doc_namespace}-1]、[{doc_namespace}-2] ..."
        group_head_tokens = estimate_token_count(group_head, model_name) + 2
        if total_tokens + group_head_tokens <= max_tokens:
            lines.append(group_head)
            total_tokens += group_head_tokens
        first_ref_recorded = False
        group_blocks: List[str] = []
        for local_index, doc in enumerate(group.get("docs") or [], start=1):
            src = adapter.hit_entity_source(doc) or source or "unknown"
            content = (adapter.hit_display_text(doc) or "").strip()
            if not content:
                evidence_index += 1
                continue
            metadata = adapter.hit_metadata(doc)
            section = adapter.doc_section_name(doc)
            article_no = str(
                metadata.get("article_no")
                or metadata.get("article_id")
                or metadata.get("clause_id")
                or metadata.get("clause_label")
                or ""
            ).strip()
            chunk_range = adapter.hit_chunk_range(doc)
            relevance = evidence_relevance(adapter.hit_score(doc), score_mode, best_score)
            parts = [f"来源：{src}", f"标题：{title}", f"相关度：{relevance:.2f}"]
            if article_no:
                parts.append(f"条款：{article_no}")
            if section:
                parts.append(f"章节：{section}")
            if chunk_range:
                parts.append(f"chunk：{chunk_range}")
            local_ref = f"{doc_namespace}-{local_index}"
            block = f"[{local_ref}] " + " | ".join(parts) + "\n" + content
            block_tokens = estimate_token_count(block, model_name) + 2
            if total_tokens + block_tokens > max_tokens:
                fitted = fit_evidence_block_to_budget(
                    f"[{local_ref}] " + " | ".join(parts),
                    content,
                    max_tokens - total_tokens,
                    model_name,
                )
                if fitted and estimate_token_count(fitted, model_name) + 2 <= max_tokens - total_tokens:
                    group_blocks.append(fitted)
                    if not first_ref_recorded:
                        refs.append({
                            "index": evidence_index,
                            "source": source,
                            "title": title,
                            "section": section,
                            "article_no": article_no,
                            "chunk_range": chunk_range,
                            "status": status_map.get(source, ""),
                            "snippet": compare_answer_snippet(runtime, doc),
                            "evidence_query": evidence_query,
                            "doc_namespace": doc_namespace,
                            "local_ref": local_ref,
                        })
                    lines.extend(group_blocks)
                return "\n\n".join(lines), refs
            group_blocks.append(block)
            if not first_ref_recorded:
                refs.append({
                    "index": evidence_index,
                    "source": source,
                    "title": title,
                    "section": section,
                    "article_no": article_no,
                    "snippet": compare_answer_snippet(runtime, doc),
                    "evidence_query": evidence_query,
                    "doc_namespace": doc_namespace,
                    "local_ref": local_ref,
                })
                first_ref_recorded = True
            evidence_index += 1
        group_summary = summarize_compare_source_blocks(
            title=title,
            evidence_query=evidence_query,
            focus_text=focus_text,
            blocks=group_blocks,
            status=status_map.get(source, ""),
        )
        summary_tokens = estimate_token_count(group_summary, model_name) + 2
        if total_tokens + summary_tokens > max_tokens:
            return "\n\n".join(lines), refs
        lines.append(group_summary)
        total_tokens += summary_tokens
    return "\n\n".join(lines), refs


def format_single_doc_compare_evidence(
    runtime: Any,
    docs: List[Any],
    query: str,
    score_mode: str,
    *,
    compare_focus_text: Callable[[Optional[Dict[str, Any]]], str],
    token_budget: int,
    model_name: str = "",
    compare_plan: Optional[Dict[str, Any]] = None,
) -> tuple[str, List[Dict[str, Any]]]:
    adapter = _evidence_context(runtime)
    lines: List[str] = []
    refs: List[Dict[str, Any]] = []
    total_tokens = 0
    max_tokens = max(256, int(token_budget))
    plan = dict(compare_plan or {})
    topic_pair = [str(item).strip() for item in (plan.get("topic_pair") or []) if str(item).strip()]
    focus_text = " vs ".join(topic_pair[:2]) or compare_focus_text(plan)
    header = f"单文档对比证据焦点：{focus_text}"
    header_tokens = estimate_token_count(header, model_name) + 2
    if header_tokens <= max_tokens:
        lines.append(header)
        total_tokens += header_tokens
    best_score = adapter.hit_score(docs[0]) if docs else 0.0
    for index, doc in enumerate(docs, start=1):
        src = adapter.hit_entity_source(doc) or "unknown"
        title = adapter.source_display_title(adapter.normalize_filename_for_match(src) or src)
        content = (adapter.hit_display_text(doc) or "").strip()
        if not content:
            continue
        section = adapter.doc_section_name(doc)
        chunk_range = adapter.hit_chunk_range(doc)
        relevance = evidence_relevance(adapter.hit_score(doc), score_mode, best_score)
        label = topic_pair[min(len(topic_pair) - 1, len(refs))] if topic_pair else (section or f"片段{index}")
        parts = [f"来源：{src}", f"标题：{title}", f"相关度：{relevance:.2f}"]
        if section:
            parts.append(f"章节：{section}")
        if chunk_range:
            parts.append(f"chunk：{chunk_range}")
        block = f"[{index}] " + " | ".join(parts) + "\n" + content
        block_tokens = estimate_token_count(block, model_name) + 2
        if total_tokens + block_tokens > max_tokens:
            fitted = fit_evidence_block_to_budget(
                f"[{index}] " + " | ".join(parts),
                content,
                max_tokens - total_tokens,
                model_name,
            )
            if fitted and estimate_token_count(fitted, model_name) + 2 <= max_tokens - total_tokens:
                lines.append(fitted)
                if len(refs) < 2:
                    refs.append({
                        "index": index,
                        "title": title,
                        "source": src,
                        "label": label,
                        "section": section,
                        "chunk_range": chunk_range,
                        "snippet": compare_answer_snippet(runtime, doc),
                    })
            break
        lines.append(block)
        total_tokens += block_tokens
        if len(refs) < 2:
            refs.append({
                "index": index,
                "title": title,
                "section": section,
                "snippet": compare_answer_snippet(runtime, doc),
                "label": label,
            })
    return "\n\n".join(lines), refs


def filter_identity_noise_aspects(aspects: List[str], identity_terms: List[str], normalize_query: Callable[[str], str]) -> List[str]:
    normalized_identity = [normalize_query(term) for term in identity_terms or [] if normalize_query(term)]
    out: List[str] = []
    for aspect in aspects or []:
        value = normalize_query(aspect)
        if not value:
            continue
        if any(identity and (value in identity or identity in value) for identity in normalized_identity):
            continue
        if value not in out:
            out.append(value)
    return out


def compare_matrix_presence_state(value: str) -> str:
    state = str(value or "").strip().upper()
    if state in {"PRESENT", "ABSENT_CONFIRMED", "UNKNOWN"}:
        return state
    if state in {"ANSWERABLE", "GUARDED_FULL", "COMPARABLE_PARTIAL"}:
        return "PRESENT"
    if state in {"NOT_FOUND", "MISSING", "ABSENT"}:
        return "ABSENT_CONFIRMED"
    return "UNKNOWN"


def compare_presence_state_for_observations(observations: Dict[str, Any], docs_count: int) -> str:
    if docs_count <= 0:
        return "UNKNOWN"
    scope = str((observations or {}).get("answer_scope") or "")
    if scope in {"full", "guarded_full"}:
        return "PRESENT"
    reason = str((observations or {}).get("evidence_coverage_reason") or "")
    if reason in {"absent_confirmed", "not_applicable", "compare_absent_confirmed"}:
        return "ABSENT_CONFIRMED"
    return "UNKNOWN"


def evidence_observations(
    runtime: Any,
    query: str,
    docs: List[Any],
    *,
    qfilters: Optional[Dict[str, Any]] = None,
    candidate_docs: Optional[List[Any]] = None,
    target_sources: Optional[List[str]] = None,
    source_lock_resolved: bool = False,
    source_lock_reason: str = "",
    is_comparison: bool = False,
    compare_missing_targets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    del candidate_docs, source_lock_reason
    adapter = _evidence_context(runtime)
    all_docs = list(docs or [])
    evidence_docs = [doc for doc in all_docs if not is_heading_only_hit(doc, adapter.hit_metadata)]
    substantive_docs = [
        doc
        for doc in evidence_docs
        if is_substantive_short_legal_evidence(doc, adapter.hit_metadata, adapter.hit_display_text)
    ]
    semantic = adapter.query_semantic_aspects(query, qfilters=qfilters)
    aspect_terms = list(dict.fromkeys([str(term).strip() for term in (semantic.get("terms") or []) if str(term).strip()]))[:8]

    covered: List[str] = []
    for doc in substantive_docs or evidence_docs:
        for term in doc_semantic_aspect_hits(runtime, doc, aspect_terms):
            if term not in covered:
                covered.append(term)
    uncovered = [
        term
        for term in aspect_terms
        if term not in covered and not any(term in item or item in term for item in covered)
    ]

    target_set = [
        adapter.normalize_filename_for_match(source or "")
        for source in (target_sources or [])
        if adapter.normalize_filename_for_match(source or "")
    ]
    wrong_source = False
    if target_set and evidence_docs:
        doc_sources = [
            adapter.normalize_filename_for_match(adapter.hit_entity_source(doc) or "")
            for doc in evidence_docs
            if adapter.normalize_filename_for_match(adapter.hit_entity_source(doc) or "")
        ]
        wrong_source = bool(doc_sources and not any(source in target_set for source in doc_sources))

    if is_comparison and compare_missing_targets:
        reason = "compare_targets_not_found"
        scope = "refusal"
    elif not all_docs:
        reason = "empty_evidence"
        scope = "refusal"
    elif not evidence_docs:
        reason = "heading_only_evidence"
        scope = "refusal"
    elif wrong_source:
        reason = "wrong_source"
        scope = "refusal"
    elif not substantive_docs:
        reason = "low_evidence_relevance"
        scope = "refusal"
    elif aspect_terms and uncovered:
        reason = "partial_term_coverage"
        scope = "partial" if source_lock_resolved and covered else "guarded_full"
    else:
        reason = "sufficient_evidence"
        scope = "full"

    if scope == "full" and any(is_generic_section_title(adapter.doc_section_name(doc)) for doc in substantive_docs):
        scope = "guarded_full"

    section_names = [adapter.doc_section_name(doc) for doc in evidence_docs if adapter.doc_section_name(doc)]
    qualified_count = len(substantive_docs)
    return {
        "evidence_coverage_reason": reason,
        "answer_scope": scope,
        "covered_aspects": covered[:8],
        "uncovered_aspects": uncovered[:8],
        "qualified_substantive_chunks": qualified_count,
        "qualified_evidence_chunks": qualified_count,
        "evidence_docs": len(evidence_docs),
        "heading_only_chunks": max(0, len(all_docs) - len(evidence_docs)),
        "source_lock_resolved": bool(source_lock_resolved),
        "target_sources": target_set,
        "wrong_source": wrong_source,
        "section_names": list(dict.fromkeys(section_names))[:8],
        "compare_missing_targets": list(compare_missing_targets or []),
    }


def finalize_compare_evidence_observations(source_statuses: List[Dict[str, Any]]) -> Dict[str, Any]:
    covered_aspects: List[str] = []
    uncovered_aspects: List[str] = []
    strong_success_statuses = {"answerable", "guarded_full", "comparable_partial"}
    asymmetric_success_statuses = strong_success_statuses | {"absent_confirmed", "not_applicable"}
    for item in source_statuses:
        observations = dict(item.get("observations") or {})
        for aspect in observations.get("covered_aspects") or []:
            if aspect and aspect not in covered_aspects:
                covered_aspects.append(aspect)
        for aspect in observations.get("uncovered_aspects") or []:
            if aspect and aspect not in uncovered_aspects:
                uncovered_aspects.append(aspect)
    if source_statuses and all(item["status"] in strong_success_statuses for item in source_statuses):
        return {
            "evidence_coverage_reason": "sufficient_evidence",
            "answer_scope": "guarded_full" if any(item["status"] in {"guarded_full", "comparable_partial"} for item in source_statuses) else "full",
            "compare_status": "compare_ready",
            "compare_source_statuses": source_statuses,
            "covered_aspects": covered_aspects[:8],
            "uncovered_aspects": uncovered_aspects[:8],
        }
    if (
        source_statuses
        and all(item["status"] in asymmetric_success_statuses for item in source_statuses)
        and any(item["status"] in {"absent_confirmed", "not_applicable"} for item in source_statuses)
    ):
        return {
            "evidence_coverage_reason": "compare_asymmetric_supported",
            "answer_scope": "guarded_full",
            "compare_status": "compare_asymmetric",
            "compare_degraded": True,
            "compare_source_statuses": source_statuses,
            "covered_aspects": covered_aspects[:8],
            "uncovered_aspects": uncovered_aspects[:8],
        }
    if source_statuses and len(source_statuses) >= 2:
        return {
            "evidence_coverage_reason": "sufficient_evidence",
            "answer_scope": "guarded_full",
            "compare_status": "compare_asymmetric",
            "compare_degraded": any(item["status"] in {"not_found", "evidence_insufficient"} for item in source_statuses),
            "compare_source_statuses": source_statuses,
            "covered_aspects": covered_aspects[:8],
            "uncovered_aspects": uncovered_aspects[:8],
        }
    if source_statuses and all(item["status"] == "not_found" for item in source_statuses):
        reason = "compare_targets_not_found"
        compare_status = "all_sources_missing"
    elif any(item["status"] == "not_found" for item in source_statuses):
        reason = "compare_source_missing"
        compare_status = "partial_sources_missing"
    else:
        reason = "compare_evidence_insufficient"
        compare_status = "evidence_insufficient"
    return {
        "evidence_coverage_reason": reason,
        "answer_scope": "refusal",
        "compare_status": compare_status,
        "compare_source_statuses": source_statuses,
        "covered_aspects": covered_aspects[:8],
        "uncovered_aspects": uncovered_aspects[:8],
    }


def compare_source_status_entry(
    runtime: Any,
    query: str,
    group: Dict[str, Any],
    qfilters: Optional[Dict[str, Any]],
    compare_identity_terms: List[str],
    min_substantive_chunks_for_compare_partial: int,
    observations_fn: Callable[..., Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    adapter = _evidence_context(runtime)
    source = adapter.normalize_filename_for_match((group or {}).get("source") or "")
    docs = list((group or {}).get("docs") or [])
    evidence_query = adapter.normalize_query((group or {}).get("evidence_query") or query) or query
    if not source:
        return None
    if not docs:
        observations = {
            "evidence_coverage_reason": "compare_source_not_found",
            "answer_scope": "refusal",
            "covered_aspects": [],
            "uncovered_aspects": [],
        }
        status = "not_found"
        presence_state = "UNKNOWN"
    else:
        observations = observations_fn(evidence_query, docs, qfilters=qfilters)
        presence_state = compare_presence_state_for_observations(observations, len(docs))
        if observations.get("answer_scope") == "full":
            status = "answerable"
        elif observations.get("answer_scope") == "guarded_full":
            status = "guarded_full"
        else:
            status = "evidence_insufficient"
    filtered_covered = filter_identity_noise_aspects(observations.get("covered_aspects") or [], compare_identity_terms, adapter.normalize_query)
    filtered_uncovered = filter_identity_noise_aspects(observations.get("uncovered_aspects") or [], compare_identity_terms, adapter.normalize_query)
    if status == "evidence_insufficient" and docs and filtered_covered:
        status = "comparable_partial"
        presence_state = "PRESENT"
    elif (
        status == "evidence_insufficient"
        and docs
        and not filtered_covered
        and int(observations.get("qualified_substantive_chunks") or 0) >= min_substantive_chunks_for_compare_partial
    ):
        status = "comparable_partial"
        presence_state = "PRESENT"
    elif status == "evidence_insufficient" and presence_state == "ABSENT_CONFIRMED":
        status = "absent_confirmed"
    return {
        "source": source,
        "title": adapter.source_display_title(source),
        "evidence_query": evidence_query,
        "status": status,
        "presence_state": compare_matrix_presence_state(presence_state),
        "docs_count": len(docs),
        "observations": {
            **observations,
            "covered_aspects": filtered_covered,
            "uncovered_aspects": filtered_uncovered,
        },
    }


async def compare_evidence_observations_async(
    runtime: Any,
    query: str,
    source_groups: List[Dict[str, Any]],
    qfilters: Optional[Dict[str, Any]],
    *,
    source_identity_terms_for_validation: Callable[[List[str]], List[str]],
    observations_fn: Callable[..., Dict[str, Any]],
    min_substantive_chunks_for_compare_partial: int,
) -> Dict[str, Any]:
    adapter = _evidence_context(runtime)
    compare_identity_terms = source_identity_terms_for_validation([
        adapter.normalize_filename_for_match((group or {}).get("source") or "")
        for group in (source_groups or [])
        if adapter.normalize_filename_for_match((group or {}).get("source") or "")
    ])
    tasks = [
        asyncio.to_thread(
            compare_source_status_entry,
            runtime,
            query,
            group,
            qfilters,
            compare_identity_terms,
            min_substantive_chunks_for_compare_partial,
            observations_fn,
        )
        for group in (source_groups or [])
    ]
    source_statuses = [item for item in await asyncio.gather(*tasks) if item]
    return finalize_compare_evidence_observations(source_statuses)
