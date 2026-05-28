import asyncio
import argparse
import json
import re
from typing import Any, Dict, List, Tuple

import main as m


OFF_IDS = {
    15,
    25,
    26,
    33,
    34,
    41,
    52,
    60,
    77,
    99,
    103,
    109,
    133,
    146,
    155,
    156,
    157,
    161,
    163,
    168,
    170,
    171,
    173,
    175,
    177,
}


def _parse_ids(text: str) -> List[int]:
    raw = (text or "").strip()
    if not raw:
        return []
    out: List[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            continue
    return out


def _load_cases(path: str, ids: List[int]) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    id_set = set(ids or [])
    for raw in open(path, "r", encoding="utf-8"):
        line = (raw or "").strip()
        if not line:
            continue
        mm = re.match(r"^(\d+)\.\s*(.+)$", line)
        if not mm:
            continue
        cid = int(mm.group(1))
        q = mm.group(2).strip()
        if cid in id_set:
            out.append((cid, q))
    return out


def _load_cases_json(path: str) -> List[Tuple[int, str]]:
    payload = json.loads(open(path, "r", encoding="utf-8").read())
    items = payload.get("items") if isinstance(payload, dict) else payload
    out: List[Tuple[int, str]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        try:
            cid = int(item.get("id"))
        except Exception:
            continue
        q = str(item.get("query") or "").strip()
        if q:
            out.append((cid, q))
    return out


def _doc_preview(doc: Any, limit: int = 220) -> str:
    section = (m._doc_section_name(doc) or "").strip()
    text = (m._hit_display_text(doc) or "").strip()
    blob = (section + "\n" + text).strip()
    blob = re.sub(r"\s+", " ", blob).strip()
    return blob[:limit]


def _hit_scores(doc: Any) -> Dict[str, Any]:
    md = m._hit_metadata(doc)
    return {
        "orig_score_mode": md.get("orig_score_mode"),
        "orig_score": md.get("orig_score"),
        "fusion_score": md.get("fusion_score"),
        "score_mode": m._hit_score_mode(doc),
        "score": m._hit_score(doc),
    }


async def _run_one(handler: m.QueryHandler, cid: int, query: str) -> Dict[str, Any]:
    normalized_query = m._normalize_query(query)
    fnames = m._extract_filename_candidates(normalized_query)
    recall = await handler._run_lightweight_recall(
        normalized_query,
        top_k=10,
        enable_rerank=True,
        filename_hints=fnames,
        user_id=f"off_topic_diag_{cid}",
    )
    resolved_targets = [
        m._normalize_filename_for_match(x)
        for x in (recall.get("target_sources") or fnames)
        if m._normalize_filename_for_match(x)
    ]
    process_docs = m._select_process_output_docs(
        normalized_query,
        recall.get("post_filter_docs") or recall.get("selected_docs") or [],
        recall.get("score_mode") or "fusion_score",
        recall.get("qfilters") or {},
        int(recall.get("final_n") or 10),
    )
    evidence_query = recall.get("evidence_query") or recall.get("retrieval_query") or normalized_query
    anchors = m._query_content_anchor_terms(
        evidence_query,
        qfilters=recall.get("qfilters") or {},
        source_title_terms=m._source_title_aspect_terms(resolved_targets),
    )
    observations = m._evidence_observations(
        evidence_query,
        process_docs,
        qfilters=recall.get("qfilters") or {},
        candidate_docs=recall.get("post_filter_docs") or recall.get("selected_docs") or [],
        target_sources=resolved_targets,
        source_lock_resolved=bool(recall.get("resolved_source_lock")),
        source_lock_reason=str(recall.get("source_lock_reason") or ""),
        is_comparison=bool(recall.get("is_comparison")),
        compare_missing_targets=list(recall.get("compare_missing_targets") or []),
    )
    would_off = m._is_off_topic_locked_document_query(
        evidence_query,
        process_docs,
        qfilters=recall.get("qfilters") or {},
        target_sources=resolved_targets,
        source_lock_resolved=bool(recall.get("resolved_source_lock")),
    )
    best_dense_rel = m._best_dense_relevance_for_locked_source(recall.get("docs") or [], resolved_targets)
    llm_window = m._locked_source_evidence_window(recall.get("docs") or [], resolved_targets, max(3, int(getattr(m.config, "PRIMARY_EVIDENCE_TOPK", 5))))
    llm_hit = m._llm_evidence_core_concept_hit(evidence_query, llm_window) if llm_window else False
    top = []
    for d in process_docs[:5]:
        top.append(
            {
                "source": m._hit_entity_source(d),
                "section": m._doc_section_name(d),
                "preview": _doc_preview(d),
                "scores": _hit_scores(d),
            }
        )
    return {
        "id": cid,
        "query": query,
        "route": recall.get("query_route"),
        "lock_mode": recall.get("lock_mode"),
        "source_lock_reason": recall.get("source_lock_reason"),
        "target_sources": resolved_targets,
        "retrieval_query": recall.get("retrieval_query") or "",
        "dense_query": recall.get("dense_query"),
        "evidence_query": evidence_query,
        "anchors": anchors[:10],
        "best_dense_rel": float(best_dense_rel),
        "would_off_topic_lexical": bool(would_off),
        "llm_window_n": len(llm_window),
        "llm_core_concept_hit": bool(llm_hit),
        "final_channel": "refusal" if observations.get("answer_scope") not in {"full", "guarded_full"} else "light_rag",
        "refusal_reason": observations.get("evidence_coverage_reason") if observations.get("answer_scope") not in {"full", "guarded_full"} else "",
        "final_evidence_coverage_reason": observations.get("evidence_coverage_reason") or "",
        "answer_scope": observations.get("answer_scope") or "",
        "covered_aspects": observations.get("covered_aspects") or [],
        "uncovered_aspects": observations.get("uncovered_aspects") or [],
        "top_docs": top,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", default="/app/test.json")
    parser.add_argument("--report", default="/app/uploads/off_topic_semantic_whitebox.json")
    parser.add_argument("--cases-json", default="")
    parser.add_argument("--ids", default="")
    args = parser.parse_args()

    cases_json = (args.cases_json or "").strip()
    if cases_json:
        cases = _load_cases_json(cases_json)
        ids = [cid for cid, _ in cases]
    else:
        ids = _parse_ids(args.ids) or sorted(list(OFF_IDS))
        cases = _load_cases(str(args.test_path), ids)
    handler = m.QueryHandler()
    out: List[Dict[str, Any]] = []
    for cid, q in cases:
        out.append(await _run_one(handler, cid, q))
    out.sort(key=lambda x: x["id"])
    payload = {"count": len(out), "ids": ids, "items": out}
    open(str(args.report), "w", encoding="utf-8").write(
        json.dumps(payload, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    asyncio.run(main())
