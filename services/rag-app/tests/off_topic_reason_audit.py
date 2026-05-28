import argparse
import asyncio
import json
import re
from typing import Any, Dict, List, Tuple


def _contains_any_variant(m, hay: str, term: str) -> bool:
    for v in m._coverage_aspect_variants(term):
        if v and v in hay:
            return True
    return False


def _source_body_haystack(m, docs: List[Any], source: str) -> str:
    parts: List[str] = []
    for doc in docs or []:
        if m._normalize_filename_for_match(m._hit_entity_source(doc) or "") != source:
            continue
        if m._is_heading_only_hit(doc):
            continue
        if not m._hit_matches_source_state(doc, m._source_state(source)):
            continue
        parts.append(m._doc_section_name(doc))
        parts.append(m._hit_display_text(doc) or "")
    return "\n".join([p for p in parts if p])


def _lexical_hits_exist(m, anchor: str, source: str, limit: int = 40) -> Tuple[bool, bool]:
    try:
        hits = m._lexical_recall_indexed(anchor, limit=limit, source_filter=source)
    except Exception:
        hits = m._lexical_recall_fallback(anchor, limit=limit, source_filter=source)
    if not hits:
        return False, False
    body_hits = [h for h in hits if not m._is_heading_only_hit(h)]
    return True, (not body_hits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="/tmp/test_json_query_report.json")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--out", default="/tmp/off_topic_reason_audit.json")
    args = ap.parse_args()

    import main as m

    report = json.load(open(args.report, "r", encoding="utf-8"))
    rows = report.get("results") or []
    off = [
        r for r in rows
        if (r.get("observations") or {}).get("evidence_coverage_reason") == "off_topic_in_document"
        or r.get("refusal_reason") == "off_topic_in_document"
    ]

    handler = m.QueryHandler()

    counts: Dict[str, int] = {
        "compare_route": 0,
        "anchor_not_in_doc": 0,
        "anchor_only_in_heading": 0,
        "anchor_in_doc_but_recall_missed": 0,
        "no_single_target": 0,
        "no_anchors": 0,
    }
    cases: Dict[str, List[Dict[str, Any]]] = {k: [] for k in counts}

    for item in off:
        cid = int(item.get("id") or 0)
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        recall = asyncio.run(
            handler._run_lightweight_recall(
                query,
                top_k=int(args.top_k),
                enable_rerank=True,
                filename_hints=m._extract_filename_candidates(m._normalize_query(query)),
                user_id="audit",
            )
        )
        route = str(recall.get("query_route") or "")
        if route in {"multi_doc_compare", "single_doc_compare", "open_topic_compare"}:
            counts["compare_route"] += 1
            cases["compare_route"].append({
                "id": cid,
                "query": query,
                "route": route,
                "target_sources": recall.get("target_sources") or [],
            })
            continue

        targets = [
            m._normalize_filename_for_match(x)
            for x in (recall.get("target_sources") or [])
            if m._normalize_filename_for_match(x)
        ]
        if len(targets) != 1:
            counts["no_single_target"] += 1
            cases["no_single_target"].append({
                "id": cid,
                "query": query,
                "route": route,
                "target_sources": targets,
            })
            continue
        source = targets[0]
        qfilters = recall.get("qfilters") or {}
        evidence_query = str(recall.get("evidence_query") or recall.get("retrieval_query") or query)

        title_terms = m._source_title_aspect_terms([source])
        anchors = m._query_content_anchor_terms(evidence_query, qfilters, title_terms)
        if not anchors:
            counts["no_anchors"] += 1
            cases["no_anchors"].append({
                "id": cid,
                "query": query,
                "source": source,
                "evidence_query": evidence_query,
                "route": route,
            })
            continue

        retrieved_docs = (
            recall.get("retrieve_docs")
            or recall.get("post_filter_docs")
            or recall.get("selected_docs")
            or []
        )
        hay = _source_body_haystack(m, retrieved_docs, source)
        if any(_contains_any_variant(m, hay, a) for a in anchors):
            continue

        exist_any = False
        heading_only_any = False
        for a in anchors[:6]:
            exists, heading_only = _lexical_hits_exist(m, a, source)
            exist_any = exist_any or exists
            heading_only_any = heading_only_any or heading_only

        if not exist_any:
            counts["anchor_not_in_doc"] += 1
            cases["anchor_not_in_doc"].append({
                "id": cid,
                "query": query,
                "source": source,
                "evidence_query": evidence_query,
                "anchors": anchors,
                "source_lock_reason": recall.get("source_lock_reason") or "",
            })
            continue
        if heading_only_any:
            counts["anchor_only_in_heading"] += 1
            cases["anchor_only_in_heading"].append({
                "id": cid,
                "query": query,
                "source": source,
                "evidence_query": evidence_query,
                "anchors": anchors,
            })
            continue

        counts["anchor_in_doc_but_recall_missed"] += 1
        cases["anchor_in_doc_but_recall_missed"].append({
            "id": cid,
            "query": query,
            "source": source,
            "evidence_query": evidence_query,
            "anchors": anchors,
            "source_lock_reason": recall.get("source_lock_reason") or "",
        })

    payload = {
        "off_topic_total": len(off),
        "counts": counts,
        "examples": {
            k: (v[:20] if isinstance(v, list) else v)
            for k, v in cases.items()
        },
    }
    open(args.out, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))
    print("written", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

