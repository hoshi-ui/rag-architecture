import argparse
import asyncio
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
MAIN_PATH = os.path.join(BASE_DIR, "main.py")
DEFAULT_SPECIAL_CASES = os.path.join(REPO_ROOT, "legal_rag_special_training_cases_20260513.json")
DEFAULT_CASE_PATHS = [
    DEFAULT_SPECIAL_CASES,
    os.path.join(THIS_DIR, "real_regulation_cases.json"),
    os.path.join(THIS_DIR, "chinese_retrieval_cases.json"),
    os.path.join(THIS_DIR, "hard_competitive_cases.json"),
]
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "uploads", "training", "reranker_round1")
JSON_BEGIN = "===LEGAL_RAG_TRAINING_JSON_BEGIN==="
JSON_END = "===LEGAL_RAG_TRAINING_JSON_END==="


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reranker/embedding training data from real legal RAG cases.")
    parser.add_argument("--inside-container", action="store_true")
    parser.add_argument("--case", action="append", dest="cases", default=[])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-positives", type=int, default=2)
    parser.add_argument("--max-negatives-per-type", type=int, default=2)
    return parser.parse_args()


ARGS = _parse_args()


def _inside_container() -> bool:
    return os.path.exists("/.dockerenv")


if not _inside_container():
    os.environ.setdefault("APP_ENV", "test_local")
    os.environ.setdefault("EMBEDDING_SERVICE_URL", "http://127.0.0.1:8001")
    os.environ.setdefault("RERANK_SERVICE_URL", "http://127.0.0.1:8002")
    os.environ.setdefault("MILVUS_HOST", "127.0.0.1")
    os.environ.setdefault("MILVUS_PORT", "19530")


def _load_main_module():
    spec = importlib.util.spec_from_file_location("rag_training_main", MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _ensure_case_paths() -> List[str]:
    paths = [os.path.abspath(path) for path in (ARGS.cases or []) if path]
    if paths:
        return paths
    return [os.path.abspath(path) for path in DEFAULT_CASE_PATHS if os.path.exists(path)]


def _container_case_path(host_path: str, staged_dir: str) -> str:
    target = os.path.abspath(host_path)
    if target.startswith(os.path.abspath(BASE_DIR)):
        rel = os.path.relpath(target, BASE_DIR)
        return os.path.join("/app", rel)
    return os.path.join(staged_dir, os.path.basename(target))


def _extract_embedded_json(output: str) -> Dict[str, Any]:
    start = output.find(JSON_BEGIN)
    end = output.find(JSON_END)
    if start < 0 or end <= start:
        raise RuntimeError("builder did not emit embedded JSON")
    return json.loads(output[start + len(JSON_BEGIN):end].strip())


def _run_via_container(case_paths: List[str], output_dir: str) -> None:
    staged_dir = "/tmp/legal_rag_training_cases"
    subprocess.run(["docker", "exec", "rag-app", "mkdir", "-p", staged_dir], check=True)
    subprocess.run(["docker", "cp", __file__, "rag-app:/app/tests/build_legal_rag_training_data.py"], check=True)
    container_cases: List[str] = []
    for path in case_paths:
        container_path = _container_case_path(path, staged_dir)
        if container_path.startswith(staged_dir):
            subprocess.run(["docker", "cp", path, f"rag-app:{container_path}"], check=True)
        container_cases.append(container_path)

    container_output_dir = "/app/uploads/training/reranker_round1"
    cmd = [
        "docker", "exec", "rag-app", "python", "/app/tests/build_legal_rag_training_data.py",
        "--inside-container", "--output-dir", container_output_dir,
        "--top-k", str(ARGS.top_k),
        "--max-positives", str(ARGS.max_positives),
        "--max-negatives-per-type", str(ARGS.max_negatives_per_type),
    ]
    for path in container_cases:
        cmd.extend(["--case", path])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr or result.stdout)
        raise SystemExit(result.returncode)
    summary = _extract_embedded_json(result.stdout)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    subprocess.run(["docker", "cp", f"rag-app:{container_output_dir}", output_dir], check=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _load_case_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return list(payload.get("cases") or [])
    if isinstance(payload, list):
        return payload
    return []


def _safe_list(items: Any) -> List[str]:
    if not items:
        return []
    return [str(item).strip() for item in items if str(item).strip()]


def _normalize_case(case: Dict[str, Any], origin_path: str) -> Dict[str, Any]:
    expected = dict(case.get("expected") or {})
    retrieval = dict(case.get("retrieval_expectation") or {})
    selector = dict(retrieval.get("positive_selector") or {})
    training = dict(case.get("training_generation") or {})
    expected_sources = _safe_list(case.get("expected_sources") or expected.get("expected_sources"))
    forbidden_sources = _safe_list(expected.get("forbidden_sources"))
    category = str(case.get("category") or "generic")
    query = str(case.get("query") or "").strip()
    case_id = str(case.get("case_id") or case.get("id") or f"{os.path.basename(origin_path)}::{query[:24]}")
    target_slots = dict(case.get("target_slots") or {})
    if not target_slots:
        section_terms = []
        lowered = query
        for match in re.findall(r"第[一二三四五六七八九十百千0-9]+[章节条款]", lowered):
            if match not in section_terms:
                section_terms.append(match)
        for token in ["法律责任", "奖励与处罚", "立法程序", "监督检查", "登记", "程序", "流程", "处罚", "责任"]:
            if token in lowered and token not in section_terms:
                section_terms.append(token)
        target_slots = {
            "core_object": [],
            "business_action": [],
            "question_aspect": [],
            "section_terms": section_terms,
            "version_terms": re.findall(r"\d{4}", query),
        }
    allow_roles = _safe_list(selector.get("chunk_type_allow"))
    deny_roles = _safe_list(selector.get("chunk_type_deny_as_positive"))
    if not deny_roles:
        deny_roles = ["title", "toc", "chapter_heading", "section_heading", "document_summary", "appendix_heading"]
    return {
        "case_id": case_id,
        "origin": os.path.basename(origin_path),
        "category": category,
        "query": query,
        "expected_sources": expected_sources,
        "forbidden_sources": forbidden_sources,
        "answer_scope_expected": str(expected.get("answer_scope") or ("full_candidate" if expected_sources else "refusal")),
        "source_policy": str(expected.get("source_policy") or ("must_lock" if expected_sources else "open_negative")),
        "target_slots": target_slots,
        "positive_selector": {
            "allow_roles": allow_roles,
            "deny_roles": deny_roles,
            "must_cover_slots": _safe_list(selector.get("must_cover_slots") or []),
        },
        "hard_negative_hints": _safe_list(retrieval.get("hard_negative_hints") or [
            "same_source_wrong_section",
            "same_source_title_or_toc",
            "similar_title_other_source",
            "old_or_other_version_if_visible",
        ]),
        "use_for_training": bool(training.get("use_for_training", bool(expected_sources))),
    }


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip().lower()


def _text_contains_any(text: str, values: Iterable[str]) -> bool:
    hay = _normalize_text(text)
    for value in values or []:
        token = _normalize_text(value)
        if token and token in hay:
            return True
    return False


def _metadata_section_blob(metadata: Dict[str, Any]) -> str:
    values = [
        metadata.get("section_title") or metadata.get("section") or "",
        " ".join([str(x or "") for x in (metadata.get("section_path") or []) if str(x or "").strip()]),
        metadata.get("parent_section_title") or "",
    ]
    return " ".join(v for v in values if v)


def _chunk_role(metadata: Dict[str, Any]) -> str:
    return str(metadata.get("chunk_role") or "body").strip()


def _is_heading_role(role: str) -> bool:
    return role in {"title", "toc", "chapter_heading", "section_heading", "appendix_heading", "document_summary"}


def _slot_terms(case: Dict[str, Any], slot_name: str) -> List[str]:
    raw = (case.get("target_slots") or {}).get(slot_name) or []
    return [str(item).strip() for item in raw if str(item).strip()]


def _query_terms(query: str) -> List[str]:
    text = re.sub(r"[，。；;：:?？、()（）\[\]{}]", " ", query)
    out: List[str] = []
    for piece in text.split():
        piece = piece.strip()
        if len(piece) >= 2 and piece not in out:
            out.append(piece)
    return out


def _must_cover_chunk(case: Dict[str, Any], chunk: Dict[str, Any]) -> bool:
    selector = case.get("positive_selector") or {}
    required = selector.get("must_cover_slots") or []
    text = " ".join([
        str(chunk.get("text") or ""),
        str(chunk.get("raw_text") or ""),
        _metadata_section_blob(chunk.get("metadata") or {}),
    ])
    for slot in required:
        terms = _slot_terms(case, slot)
        if terms and not _text_contains_any(text, terms):
            return False
    return True


def _chunk_signal_score(case: Dict[str, Any], chunk: Dict[str, Any]) -> float:
    metadata = chunk.get("metadata") or {}
    text = " ".join([
        str(chunk.get("text") or ""),
        str(chunk.get("raw_text") or ""),
        _metadata_section_blob(metadata),
    ])
    score = 0.0
    role = _chunk_role(metadata)
    if role == "article":
        score += 0.6
    elif role in {"body", "paragraph", "table", "appendix"}:
        score += 0.35
    for slot in ["core_object", "business_action", "question_aspect", "section_terms", "version_terms"]:
        terms = _slot_terms(case, slot)
        for term in terms:
            if _text_contains_any(text, [term]):
                score += 1.0 if slot != "section_terms" else 1.2
    for term in _query_terms(case.get("query") or "")[:8]:
        if _text_contains_any(text, [term]):
            score += 0.15
    return score


def _positive_chunks_for_source(m: Any, case: Dict[str, Any], source: str, limit: int) -> List[Dict[str, Any]]:
    chunks = m._get_chunks_for_source(source)
    selector = case.get("positive_selector") or {}
    allow_roles = set(selector.get("allow_roles") or [])
    deny_roles = set(selector.get("deny_roles") or [])
    ranked: List[Tuple[float, Dict[str, Any]]] = []
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        role = _chunk_role(metadata)
        if _is_heading_role(role) or role in deny_roles:
            continue
        if allow_roles and role not in allow_roles:
            continue
        if not _must_cover_chunk(case, chunk):
            continue
        score = _chunk_signal_score(case, chunk)
        if score <= 0:
            continue
        ranked.append((score, chunk))
    ranked.sort(key=lambda item: (item[0], -(item[1].get("chunk_id") or 0)), reverse=True)
    return [item[1] for item in ranked[:limit]]


def _visible_documents(m: Any) -> List[Dict[str, Any]]:
    conn = m._lex_db_connect()
    rows = conn.execute(
        "SELECT source, canonical_title, same_title_group, title_tokens, aliases, active_version, searchable, status FROM documents"
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for source, canonical_title, same_title_group, title_tokens, aliases, active_version, searchable, status in rows:
        source = m._normalize_filename_for_match(source or "")
        if not source:
            continue
        if not m._doc_searchable_flag(source):
            continue
        out.append({
            "source": source,
            "canonical_title": str(canonical_title or ""),
            "same_title_group": str(same_title_group or ""),
            "title_tokens": str(title_tokens or ""),
            "aliases": str(aliases or ""),
            "active_version": active_version,
            "searchable": int(searchable or 0),
            "status": str(status or ""),
        })
    return out


def _similar_other_sources(m: Any, visible_docs: List[Dict[str, Any]], case: Dict[str, Any], expected_sources: List[str], limit: int) -> List[str]:
    query = case.get("query") or ""
    query_terms = _query_terms(query)
    expected_titles = {
        str(m._doc_get(source).get("canonical_title") or "").strip()
        for source in expected_sources
    }
    scored: List[Tuple[float, str]] = []
    expected_set = set(expected_sources)
    for doc in visible_docs:
        source = doc.get("source") or ""
        if source in expected_set:
            continue
        title_blob = " ".join([doc.get("canonical_title") or "", doc.get("title_tokens") or "", doc.get("aliases") or ""])
        score = 0.0
        if expected_titles and _text_contains_any(title_blob, list(expected_titles)):
            score += 2.0
        for term in query_terms[:8]:
            if _text_contains_any(title_blob, [term]):
                score += 0.4
        if score > 0:
            scored.append((score, source))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    out: List[str] = []
    for _, source in scored:
        if source not in out:
            out.append(source)
        if len(out) >= limit:
            break
    return out


def _negative_chunks_from_source(m: Any, case: Dict[str, Any], source: str, kind: str, positive_chunk_ids: set, positive_sections: set, limit: int) -> List[Dict[str, Any]]:
    chunks = m._get_chunks_for_source(source)
    ranked: List[Tuple[float, Dict[str, Any]]] = []
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        role = _chunk_role(metadata)
        chunk_id = int(metadata.get("chunk_id") or chunk.get("chunk_id") or 0)
        if chunk_id in positive_chunk_ids:
            continue
        section_title = str(metadata.get("section_title") or metadata.get("section") or "").strip()
        if kind == "same_source_title_or_toc":
            if not _is_heading_role(role):
                continue
            score = _chunk_signal_score(case, chunk) + 1.5
        elif kind == "same_source_wrong_section":
            if _is_heading_role(role):
                continue
            if positive_sections and section_title in positive_sections:
                continue
            score = _chunk_signal_score(case, chunk)
        elif kind == "same_source_generic_sections":
            if _is_heading_role(role):
                continue
            if not m._is_generic_section_title(section_title):
                continue
            score = _chunk_signal_score(case, chunk) + 0.6
        else:
            if _is_heading_role(role):
                continue
            score = _chunk_signal_score(case, chunk)
        if score <= 0:
            continue
        ranked.append((score, chunk))
    ranked.sort(key=lambda item: (item[0], -(item[1].get("chunk_id") or 0)), reverse=True)
    return [item[1] for item in ranked[:limit]]


def _hard_negative_pool(m: Any, visible_docs: List[Dict[str, Any]], case: Dict[str, Any], expected_sources: List[str], positives: List[Dict[str, Any]], max_per_type: int) -> List[Dict[str, Any]]:
    positive_chunk_ids = {int(item.get("chunk_id") or ((item.get("metadata") or {}).get("chunk_id") or 0)) for item in positives}
    positive_sections = {str((item.get("metadata") or {}).get("section_title") or (item.get("metadata") or {}).get("section") or "").strip() for item in positives if str((item.get("metadata") or {}).get("section_title") or (item.get("metadata") or {}).get("section") or "").strip()}
    hints = case.get("hard_negative_hints") or []
    out: List[Dict[str, Any]] = []

    for source in expected_sources:
        if "same_source_wrong_section" in hints:
            for chunk in _negative_chunks_from_source(m, case, source, "same_source_wrong_section", positive_chunk_ids, positive_sections, max_per_type):
                out.append({"negative_type": "same_source_wrong_section", **chunk})
        if "same_source_title_or_toc" in hints or "title_or_toc" in hints:
            for chunk in _negative_chunks_from_source(m, case, source, "same_source_title_or_toc", positive_chunk_ids, positive_sections, max_per_type):
                out.append({"negative_type": "same_source_title_or_toc", **chunk})
        if "same_source_generic_sections" in hints:
            for chunk in _negative_chunks_from_source(m, case, source, "same_source_generic_sections", positive_chunk_ids, positive_sections, max_per_type):
                out.append({"negative_type": "same_source_generic_sections", **chunk})

        if "old_or_other_version_if_visible" in hints or "same_canonical_title_other_version" in hints:
            canonical = str(m._doc_get(source).get("canonical_title") or "").strip()
            for other in m._find_same_title_candidates(canonical, exclude_source=source)[:max_per_type]:
                if other in expected_sources or not m._doc_searchable_flag(other):
                    continue
                for chunk in _negative_chunks_from_source(m, case, other, "old_or_other_version_if_visible", positive_chunk_ids, positive_sections, 1):
                    out.append({"negative_type": "old_or_other_version_if_visible", **chunk})

    if "similar_title_other_source" in hints:
        for other in _similar_other_sources(m, visible_docs, case, expected_sources, max_per_type):
            for chunk in _negative_chunks_from_source(m, case, other, "similar_title_other_source", positive_chunk_ids, positive_sections, 1):
                out.append({"negative_type": "similar_title_other_source", **chunk})

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in out:
        metadata = item.get("metadata") or {}
        key = (
            item.get("negative_type") or "",
            item.get("source") or "",
            int(metadata.get("chunk_id") or item.get("chunk_id") or 0),
            _normalize_text(item.get("text") or item.get("raw_text") or "")[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _returned_doc_ids(documents: List[Dict[str, Any]]) -> set:
    out = set()
    for doc in documents or []:
        metadata = doc.get("metadata") or {}
        source = str(doc.get("source") or "").strip()
        chunk_id = int(metadata.get("chunk_id") or 0)
        if source and chunk_id:
            out.add((source, chunk_id))
    return out


async def _run_case(m: Any, handler: Any, visible_docs: List[Dict[str, Any]], case: Dict[str, Any]) -> Dict[str, Any]:
    result = await handler.retrieve(query=case["query"], user_id="training_builder", top_k=ARGS.top_k, enable_rerank=True)
    metadata = result.get("metadata") or {}
    documents = result.get("documents") or []
    expected_sources = [src for src in case.get("expected_sources") or [] if m._doc_searchable_flag(src)]
    positives: List[Dict[str, Any]] = []
    for source in expected_sources:
        positives.extend(_positive_chunks_for_source(m, case, source, ARGS.max_positives))
    positive_records: List[Dict[str, Any]] = []
    for chunk in positives:
        md = chunk.get("metadata") or {}
        positive_records.append({
            "source": chunk.get("source") or chunk.get("metadata", {}).get("source") or expected_sources[0] if expected_sources else "",
            "chunk_id": int(md.get("chunk_id") or chunk.get("chunk_id") or 0),
            "chunk_role": _chunk_role(md),
            "section_title": str(md.get("section_title") or md.get("section") or ""),
            "text": chunk.get("text") or chunk.get("raw_text") or "",
        })
    negative_pool = _hard_negative_pool(m, visible_docs, case, expected_sources, positives, ARGS.max_negatives_per_type)
    negative_records: List[Dict[str, Any]] = []
    for chunk in negative_pool:
        md = chunk.get("metadata") or {}
        negative_records.append({
            "negative_type": chunk.get("negative_type") or "unknown",
            "source": chunk.get("source") or "",
            "chunk_id": int(md.get("chunk_id") or chunk.get("chunk_id") or 0),
            "chunk_role": _chunk_role(md),
            "section_title": str(md.get("section_title") or md.get("section") or ""),
            "text": chunk.get("text") or chunk.get("raw_text") or "",
        })

    returned_ids = _returned_doc_ids(documents)
    positive_ids = {(item["source"], item["chunk_id"]) for item in positive_records if item["source"] and item["chunk_id"]}
    source_hit = bool(expected_sources) and any(doc.get("source") in expected_sources for doc in documents)
    section_hit = bool(expected_sources) and any(
        doc.get("source") in expected_sources and (
            str((doc.get("metadata") or {}).get("section_title") or (doc.get("metadata") or {}).get("section") or "") in {item["section_title"] for item in positive_records if item["section_title"]}
            or _text_contains_any(str((doc.get("metadata") or {}).get("section_title") or (doc.get("metadata") or {}).get("section") or ""), _slot_terms(case, "section_terms"))
        )
        for doc in documents
    )
    body_hit = bool(positive_ids) and any(item in returned_ids for item in positive_ids)
    retrieval_gap = bool(expected_sources) and bool(positive_records) and not body_hit

    reranker_triples: List[Dict[str, Any]] = []
    reranker_pairs: List[Dict[str, Any]] = []
    embedding_triples: List[Dict[str, Any]] = []
    if case.get("use_for_training") and positive_records and negative_records:
        for pos in positive_records:
            reranker_pairs.append({
                "case_id": case["case_id"],
                "origin": case["origin"],
                "query": case["query"],
                "passage": pos["text"],
                "label": 1,
                "source": pos["source"],
                "chunk_id": pos["chunk_id"],
                "role": pos["chunk_role"],
                "section_title": pos["section_title"],
            })
            for neg in negative_records:
                reranker_pairs.append({
                    "case_id": case["case_id"],
                    "origin": case["origin"],
                    "query": case["query"],
                    "passage": neg["text"],
                    "label": 0,
                    "source": neg["source"],
                    "chunk_id": neg["chunk_id"],
                    "role": neg["chunk_role"],
                    "section_title": neg["section_title"],
                    "negative_type": neg["negative_type"],
                })
                reranker_triples.append({
                    "case_id": case["case_id"],
                    "origin": case["origin"],
                    "query": case["query"],
                    "positive": pos["text"],
                    "positive_source": pos["source"],
                    "positive_chunk_id": pos["chunk_id"],
                    "hard_negative": neg["text"],
                    "hard_negative_source": neg["source"],
                    "hard_negative_chunk_id": neg["chunk_id"],
                    "hard_negative_type": neg["negative_type"],
                    "retrieval_gap": retrieval_gap,
                })
                if retrieval_gap:
                    embedding_triples.append({
                        "case_id": case["case_id"],
                        "origin": case["origin"],
                        "query": case["query"],
                        "positive": pos["text"],
                        "positive_source": pos["source"],
                        "positive_chunk_id": pos["chunk_id"],
                        "hard_negative": neg["text"],
                        "hard_negative_source": neg["source"],
                        "hard_negative_chunk_id": neg["chunk_id"],
                        "hard_negative_type": neg["negative_type"],
                    })

    return {
        "case_id": case["case_id"],
        "origin": case["origin"],
        "category": case["category"],
        "query": case["query"],
        "expected_sources": expected_sources,
        "forbidden_sources": case.get("forbidden_sources") or [],
        "answer_scope_expected": case["answer_scope_expected"],
        "source_policy": case["source_policy"],
        "retrieve_metadata": metadata,
        "returned_sources": [str(doc.get("source") or "") for doc in documents],
        "source_hit": source_hit,
        "section_hit": section_hit,
        "body_hit": body_hit,
        "retrieval_gap_for_embedding": retrieval_gap,
        "positive_chunks": positive_records,
        "hard_negatives": negative_records,
        "reranker_pairs": reranker_pairs,
        "reranker_triples": reranker_triples,
        "embedding_triples": embedding_triples,
    }


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def _build_inside_container(case_paths: List[str], output_dir: str) -> Dict[str, Any]:
    m = _load_main_module()
    visible_docs = _visible_documents(m)
    handler = m.QueryHandler()
    normalized_cases: List[Dict[str, Any]] = []
    for path in case_paths:
        for case in _load_case_file(path):
            normalized_cases.append(_normalize_case(case, path))

    results: List[Dict[str, Any]] = []
    for case in normalized_cases:
        results.append(await _run_case(m, handler, visible_docs, case))

    reranker_pairs: List[Dict[str, Any]] = []
    reranker_triples: List[Dict[str, Any]] = []
    embedding_triples: List[Dict[str, Any]] = []
    negative_type_counter: Counter[str] = Counter()
    for item in results:
        reranker_pairs.extend(item.get("reranker_pairs") or [])
        reranker_triples.extend(item.get("reranker_triples") or [])
        embedding_triples.extend(item.get("embedding_triples") or [])
        for neg in item.get("hard_negatives") or []:
            negative_type_counter[str(neg.get("negative_type") or "unknown")] += 1

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "training_manifest.json")
    pairs_path = os.path.join(output_dir, "reranker_pairs.jsonl")
    triples_path = os.path.join(output_dir, "reranker_triples.jsonl")
    embedding_path = os.path.join(output_dir, "embedding_triples.jsonl")
    summary_path = os.path.join(output_dir, "summary.json")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "cases": results}, f, ensure_ascii=False, indent=2)
    _write_jsonl(pairs_path, reranker_pairs)
    _write_jsonl(triples_path, reranker_triples)
    _write_jsonl(embedding_path, embedding_triples)

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "output_dir": output_dir,
        "case_count": len(results),
        "training_case_count": sum(1 for item in results if item.get("positive_chunks") and item.get("hard_negatives")),
        "retrieval_gap_cases": [item["case_id"] for item in results if item.get("retrieval_gap_for_embedding")],
        "source_hit_rate": (sum(1 for item in results if item.get("expected_sources") and item.get("source_hit")) / max(1, sum(1 for item in results if item.get("expected_sources")))) if results else 0.0,
        "section_hit_rate": (sum(1 for item in results if item.get("expected_sources") and item.get("section_hit")) / max(1, sum(1 for item in results if item.get("expected_sources")))) if results else 0.0,
        "body_hit_rate": (sum(1 for item in results if item.get("expected_sources") and item.get("body_hit")) / max(1, sum(1 for item in results if item.get("expected_sources")))) if results else 0.0,
        "reranker_pairs": len(reranker_pairs),
        "reranker_triples": len(reranker_triples),
        "embedding_triples": len(embedding_triples),
        "hard_negative_types": dict(negative_type_counter),
        "paths": {
            "manifest": manifest_path,
            "reranker_pairs": pairs_path,
            "reranker_triples": triples_path,
            "embedding_triples": embedding_path,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    case_paths = _ensure_case_paths()
    output_dir = os.path.abspath(ARGS.output_dir)
    if not ARGS.inside_container and not _inside_container():
        _run_via_container(case_paths, output_dir)
        return

    summary = asyncio.run(_build_inside_container(case_paths, output_dir))
    print(JSON_BEGIN)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(JSON_END)


if __name__ == "__main__":
    main()