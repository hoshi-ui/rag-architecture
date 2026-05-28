import asyncio
import importlib.util
import json
import os
from collections import defaultdict
from typing import Any, Dict, List


TARGET_SOURCE = "七台河市文明祭祀条例_2022-11-03_2022-12-01.docx"
TARGET_QUERIES = [
    "给我文明祭祀条例的 3 条要点",
    "条例是否包含处罚标准",
    "焚烧冥币是否被禁止？",
    "条例核心条款",
]


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
MAIN_PATH = os.path.abspath(os.path.join(THIS_DIR, "..", "main.py"))
REPORT_PATH = os.path.abspath(os.path.join(THIS_DIR, "..", "uploads", "wrong_source_bucket_report.json"))


os.environ.setdefault("APP_ENV", "test_local")
os.environ.setdefault("EMBEDDING_SERVICE_URL", "http://127.0.0.1:8001")
os.environ.setdefault("RERANK_SERVICE_URL", "http://127.0.0.1:8002")
os.environ.setdefault("MILVUS_HOST", "127.0.0.1")
os.environ.setdefault("MILVUS_PORT", "19530")


spec = importlib.util.spec_from_file_location("rag_wrong_source_main", MAIN_PATH)
m = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(m)


def _source_name(hit: Any) -> str:
    return m._normalize_filename_for_match(m._hit_entity_source(hit) or "")


def _preview(text: str, limit: int = 80) -> str:
    s = " ".join((text or "").split())
    return s[:limit]


def _doc_state(source: str) -> Dict[str, Any]:
    info = m._doc_get(source)
    conn = m._lex_db_connect()
    doc_fts_row = conn.execute(
        "SELECT COUNT(*) FROM documents_fts WHERE filename = ?",
        (source,),
    ).fetchone()
    chunk_row = conn.execute(
        "SELECT COUNT(*) FROM chunks_meta WHERE source = ?",
        (source,),
    ).fetchone()
    return {
        "source": source,
        "status": info.get("status"),
        "active_version": info.get("active_version"),
        "pending_version": info.get("pending_version"),
        "canonical_title": info.get("canonical_title"),
        "aliases": info.get("aliases"),
        "filename_stem": info.get("filename_stem"),
        "doc_type": info.get("doc_type"),
        "topic": info.get("topic"),
        "documents_fts_present": bool(doc_fts_row and doc_fts_row[0]),
        "chunk_count": int(chunk_row[0] if chunk_row else 0),
    }


def _doc_prior_builder(query: str, fnames: List[str], allowed_docs: List[str], docs_all: List[Any]):
    source_count: Dict[str, int] = {}
    for item in docs_all:
        src = _source_name(item)
        if src:
            source_count[src] = source_count.get(src, 0) + 1
    doc_keywords = ["条例", "办法", "规定", "要点", "核心条款", "背景"]
    has_doc_words = any(k in query for k in doc_keywords)
    fname_set = {_source_name({"entity": {"source": x}}) for x in fnames} if fnames else set()
    allowed_set = {_source_name({"entity": {"source": x}}) for x in allowed_docs} if allowed_docs else set()
    qnorm = m._normalize_query(query)

    def _doc_prior_for(src: str) -> Dict[str, Any]:
        info = m._doc_get(src)
        stem = info.get("filename_stem") or m._filename_stem(src)
        title = info.get("canonical_title") or stem
        matched_title = bool((stem and stem in qnorm) or (title and title in qnorm))
        parts = {
            "fname_hint": 0.5 if src in fname_set else 0.0,
            "doc_recall_allow": 0.3 if allowed_set and (src in allowed_set) else 0.0,
            "doc_word_bonus": 0.1 if has_doc_words else 0.0,
            "source_count_bonus": min(source_count.get(src, 0) / 5.0, 0.2),
            "title_hit_bonus": 0.2 if matched_title else 0.0,
        }
        return {
            "source": src,
            "prior": round(sum(parts.values()), 6),
            "parts": parts,
            "source_count": source_count.get(src, 0),
            "canonical_title": info.get("canonical_title"),
            "aliases": info.get("aliases"),
            "filename_stem": stem,
        }

    return _doc_prior_for


async def analyze_query(handler: Any, query: str) -> Dict[str, Any]:
    query = m._normalize_query(query)
    fnames = m._extract_filename_candidates(query)
    allowed_docs: List[str] = []
    if not fnames:
        kw = ["条例", "办法", "规定", "要点", "核心条款", "背景"]
        if any(k in query for k in kw):
            allowed_docs = m._doc_recall_indexed(query, limit=10)

    embeddings = await handler.embedding_service.embed([query])
    query_embedding = embeddings[0]
    requested_k = 10
    recall_k = min(max(requested_k, 10), min(m.config.RECALL_TOP_K, m.config.TOP_K))
    pool_n = min(max(m.config.RERANK_KEEP_N, 5), recall_k)

    docs_dense = handler.vector_db.search(query_embedding, top_k=recall_k, filters=None)
    lex_items = m._lexical_recall_indexed(query, getattr(m.config, "LEXICAL_RECALL_LIMIT", 1000))
    docs_all = docs_dense + lex_items

    dense_rank_map: Dict[str, int] = {}
    for index, item in enumerate(docs_dense):
        src = _source_name(item)
        if src and src not in dense_rank_map:
            dense_rank_map[src] = index

    lex_rank_map: Dict[str, int] = {}
    for index, item in enumerate(lex_items):
        src = _source_name(item)
        if src and src not in lex_rank_map:
            lex_rank_map[src] = index

    prior_for = _doc_prior_builder(query, fnames, allowed_docs, docs_all)
    candidates = []
    seen_sources = []
    seen_set = set()
    for item in docs_all:
        src = _source_name(item)
        if src and src not in seen_set:
            seen_sources.append(src)
            seen_set.add(src)
    for src in allowed_docs:
        nsrc = m._normalize_filename_for_match(src)
        if nsrc and nsrc not in seen_set:
            seen_sources.append(nsrc)
            seen_set.add(nsrc)
    for src in seen_sources:
        prior_info = prior_for(src)
        candidates.append(
            {
                **prior_info,
                "dense_rank": dense_rank_map.get(src),
                "lex_rank": lex_rank_map.get(src),
                "target": src == TARGET_SOURCE,
            }
        )

    k_rrf = int(getattr(m.config, "RRF_K", 60))
    w_dense = float(getattr(m.config, "FUSION_W_DENSE", 0.5))
    w_lex = float(getattr(m.config, "FUSION_W_LEX", 0.5))
    w_prior = float(getattr(m.config, "FUSION_W_PRIOR", 0.2))
    w_term = float(getattr(m.config, "FUSION_W_TERM", 0.1))
    w_title = float(getattr(m.config, "FUSION_W_TITLE", 0.2))
    fname_set = {m._normalize_filename_for_match(x) for x in fnames}

    combined_scores = []
    for item in docs_all:
        src = _source_name(item)
        prior_value = prior_for(src)["prior"] if src else 0.0
        rrf_dense = m._rrf(dense_rank_map.get(src), k_rrf)
        rrf_lex = m._rrf(lex_rank_map.get(src), k_rrf)
        term_hit = 1.0 if src in lex_rank_map else 0.0
        title_hit = 1.0 if src in fname_set else 0.0
        score = (w_dense * rrf_dense) + (w_lex * rrf_lex) + (w_prior * prior_value) + (w_term * term_hit) + (w_title * title_hit)
        combined_scores.append((score, item))
    combined_scores.sort(key=lambda pair: pair[0], reverse=True)

    docs_fused = []
    seen_chunks = set()
    allowed_set = {m._normalize_filename_for_match(x) for x in allowed_docs}
    for score, item in combined_scores:
        key = (m._hit_entity_source(item) or "unknown", (m._hit_entity_text(item) or "")[:64])
        if key in seen_chunks:
            continue
        seen_chunks.add(key)
        src = _source_name(item)
        if allowed_docs and src not in allowed_set:
            continue
        docs_fused.append(item)
        if len(docs_fused) >= recall_k:
            break

    docs_pool = docs_fused[:pool_n]
    thr = m._dynamic_thresholds(m._classify_question_type(query), bool(fnames))
    accept_doc_cluster = m._doc_cluster_accept(query, docs_pool, fnames or [])
    relevance_pass = m._passes_relevance_cluster(docs_pool, "distance", thr, top_n=min(3, len(docs_pool)))

    merged_docs = m._merge_and_dedupe_hits(docs_pool, score_mode="distance")
    aggregated_docs = m._aggregate_doc_sections(merged_docs, score_mode="distance")
    filtered_docs = m._filter_low_relevance_sources(aggregated_docs, score_mode="distance")
    filtered_docs = [
        item
        for item in filtered_docs
        if (not allowed_docs) or (_source_name(item) in allowed_set)
    ]

    src_scores: Dict[str, float] = {}
    pre_rerank = []
    for item in filtered_docs:
        src = _source_name(item)
        if src in src_scores:
            continue
        prior_value = prior_for(src)["prior"]
        src_scores[src] = (
            (w_dense * m._rrf(dense_rank_map.get(src), k_rrf))
            + (w_lex * m._rrf(lex_rank_map.get(src), k_rrf))
            + (w_prior * prior_value)
            + (w_term * (1.0 if src in lex_rank_map else 0.0))
            + (w_title * (1.0 if src in fname_set else 0.0))
        )
    pre_rerank = [
        {"source": src, "score": round(score, 6), "target": src == TARGET_SOURCE}
        for src, score in sorted(src_scores.items(), key=lambda pair: pair[1], reverse=True)
    ]

    rerank_results = []
    post_scores = dict(src_scores)
    doc_sources = list(src_scores.keys())
    if doc_sources:
        by_src: Dict[str, List[str]] = defaultdict(list)
        for item in filtered_docs:
            by_src[_source_name(item)].append(m._hit_entity_text(item) or "")
        doc_texts = [(" ".join((by_src.get(src) or [])[:3])).strip() or src for src in doc_sources]
        rerank_n = min(pool_n, len(doc_texts))
        try:
            rerank_results = await handler.rerank_service.rerank(query=query, documents=doc_texts, top_k=rerank_n)
            w_rerank = float(os.getenv("FUSION_W_RERANK_DOC", "0.3"))
            for row in rerank_results or []:
                idx = row.get("index")
                score = float(row.get("score", 0.0))
                if idx is None:
                    continue
                src = doc_sources[int(idx)]
                post_scores[src] = post_scores.get(src, 0.0) + (w_rerank * score)
        except Exception as exc:
            rerank_results = [{"error": str(exc)}]

    post_rerank = [
        {"source": src, "score": round(score, 6), "target": src == TARGET_SOURCE}
        for src, score in sorted(post_scores.items(), key=lambda pair: pair[1], reverse=True)
    ]

    final_result = await handler.retrieve(query=query, user_id="wrong_source_bucket", top_k=10, enable_rerank=True)
    final_sources = [doc.get("source") for doc in final_result.get("documents") or []]

    lexical_hits = []
    for item in lex_items[:10]:
        md = m._hit_metadata(item)
        lexical_hits.append(
            {
                "source": _source_name(item),
                "chunk_id": md.get("chunk_id"),
                "section": md.get("section"),
                "preview": _preview(m._hit_entity_text(item), 120),
                "target": _source_name(item) == TARGET_SOURCE,
            }
        )

    return {
        "query": query,
        "target_source": TARGET_SOURCE,
        "target_doc_state": _doc_state(TARGET_SOURCE),
        "doc_level_recall_top10": allowed_docs,
        "candidate_priors": candidates,
        "lexical_hits_top10": lexical_hits,
        "dense_sources_top10": [
            {
                "rank": index + 1,
                "source": _source_name(item),
                "distance": round(float(m._hit_score(item)), 6),
                "preview": _preview(m._hit_entity_text(item), 100),
                "target": _source_name(item) == TARGET_SOURCE,
            }
            for index, item in enumerate(docs_dense[:10])
        ],
        "rerank_pre_doc_rank": pre_rerank,
        "rerank_post_doc_rank": post_rerank,
        "rerank_raw": rerank_results,
        "final_outcome": {
            "refusal": (final_result.get("metadata") or {}).get("refused") or (final_result.get("metadata") or {}).get("blocked"),
            "selected_sources": final_sources,
            "target_selected": TARGET_SOURCE in final_sources,
            "metadata": final_result.get("metadata") or {},
        },
        "debug_flags": {
            "doc_cluster_accept": accept_doc_cluster,
            "relevance_cluster_pass": relevance_pass,
            "docs_pool_size": len(docs_pool),
            "filtered_docs_size": len(filtered_docs),
        },
    }


async def main() -> None:
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    handler = m.QueryHandler()
    report = {
        "target_source": TARGET_SOURCE,
        "queries": [],
    }
    for query in TARGET_QUERIES:
        try:
            report["queries"].append(await analyze_query(handler, query))
        except Exception as exc:
            report["queries"].append({"query": query, "error": str(exc), "target_doc_state": _doc_state(TARGET_SOURCE)})
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())