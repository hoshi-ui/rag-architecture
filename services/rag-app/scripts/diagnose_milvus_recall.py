from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, Optional

import requests


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.config import Config
from app.storage.milvus import VectorDBService


def _field_schema(desc: Dict[str, Any], field_name: str) -> Optional[Dict[str, Any]]:
    schema = desc.get("schema") if isinstance(desc.get("schema"), dict) else {}
    fields: Iterable[Any] = desc.get("fields") or schema.get("fields") or []
    for field in fields:
        if isinstance(field, dict) and field.get("name") == field_name:
            return field
        if getattr(field, "name", None) == field_name:
            return {
                "name": getattr(field, "name", None),
                "type": getattr(field, "type", None) or getattr(field, "dtype", None),
                "params": getattr(field, "params", None),
            }
    return None


def _field_dim(field: Optional[Dict[str, Any]]) -> Optional[int]:
    if not field:
        return None
    candidates = [
        field.get("dim"),
        (field.get("params") or {}).get("dim") if isinstance(field.get("params"), dict) else None,
        (field.get("type_params") or {}).get("dim") if isinstance(field.get("type_params"), dict) else None,
        (field.get("element_type_params") or {}).get("dim") if isinstance(field.get("element_type_params"), dict) else None,
    ]
    for value in candidates:
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def _embed_query(query: str, embedding_url: str, timeout: float) -> Dict[str, Any]:
    response = requests.post(
        f"{embedding_url.rstrip('/')}/embed",
        json={"texts": [query], "normalize": True, "batch_size": 1},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    embeddings = payload.get("embeddings") or []
    embedding = list(embeddings[0] or []) if embeddings else []
    return {
        "embedding": embedding,
        "model": payload.get("model"),
        "usage": payload.get("usage") or {},
    }


def _hit_entity(hit: Any) -> Dict[str, Any]:
    if isinstance(hit, dict):
        entity = hit.get("entity") or {}
        if not isinstance(entity, dict):
            entity = {}
        out = dict(entity)
        out["_score"] = hit.get("score") if "score" in hit else hit.get("distance")
        out["_id"] = hit.get("id")
        return out
    entity = getattr(hit, "entity", None)
    out = dict(entity) if isinstance(entity, dict) else {}
    out["_score"] = getattr(hit, "score", None) if getattr(hit, "score", None) is not None else getattr(hit, "distance", None)
    out["_id"] = getattr(hit, "id", None)
    return out


def _unicode_escape(value: Any, limit: int = 500) -> str:
    return str(value or "")[:limit].encode("unicode_escape").decode("ascii")


def _utf8_hex(value: Any, limit: int = 80) -> str:
    return str(value or "")[:limit].encode("utf-8").hex()


def main() -> int:
    parser = argparse.ArgumentParser(description="Directly diagnose first-stage Milvus recall without filters or rerank.")
    parser.add_argument("--query", default="虐待犬只", help="Query text to embed and search.")
    parser.add_argument("--top-k", type=int, default=10, help="Milvus search limit.")
    parser.add_argument("--embedding-url", default=Config.EMBEDDING_URL, help="Embedding service URL.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Embedding request timeout.")
    args = parser.parse_args()

    db = VectorDBService()
    db.connect()
    desc = db.client.describe_collection(collection_name=db.collection_name) or {}
    embedding_field = _field_schema(desc, "embedding")
    schema_dim = _field_dim(embedding_field)

    embed = _embed_query(args.query, args.embedding_url, args.timeout)
    query_embedding = embed["embedding"]
    query_dim = len(query_embedding)

    print(json.dumps(
        {
            "query": args.query,
            "query_unicode_escape": _unicode_escape(args.query),
            "query_utf8_hex": _utf8_hex(args.query),
            "embedding_url": args.embedding_url,
            "embedding_model": embed.get("model"),
            "embedding_usage": embed.get("usage"),
            "query_embedding_dim": query_dim,
            "milvus_collection": db.collection_name,
            "milvus_embedding_dim": schema_dim,
            "dimension_match": schema_dim == query_dim if schema_dim is not None else None,
            "config": {
                "TOP_K": Config.TOP_K,
                "RETRIEVAL_CANDIDATE_K": getattr(Config, "RETRIEVAL_CANDIDATE_K", None),
                "RECALL_TOP_K": getattr(Config, "RECALL_TOP_K", None),
                "RERANK_TOP_K": Config.RERANK_TOP_K,
                "ENABLE_RERANK": Config.ENABLE_RERANK,
                "ENABLE_CHUNK_RERANK": getattr(Config, "ENABLE_CHUNK_RERANK", None),
            },
        },
        ensure_ascii=False,
        indent=2,
    ))

    if schema_dim is not None and schema_dim != query_dim:
        print(f"DIMENSION_MISMATCH: query_dim={query_dim} milvus_dim={schema_dim}", file=sys.stderr)
        return 2

    search_kwargs = {
        "collection_name": db.collection_name,
        "data": [query_embedding],
        "limit": max(1, int(args.top_k)),
        "output_fields": ["text", "source", "metadata", "article_id", "applicable_subjects"],
        "search_params": {"metric_type": "COSINE", "params": {"ef": 100}},
        "filter": None,
    }
    try:
        results = db.client.search(anns_field="embedding", **search_kwargs)
    except TypeError:
        results = db.client.search(
            **{
                **search_kwargs,
                "search_params": {
                    **search_kwargs["search_params"],
                    "anns_field": "embedding",
                },
            }
        )

    hits = results[0] if results else []
    print(f"raw_milvus_hit_count = {len(hits)}")
    for rank, hit in enumerate(hits, start=1):
        entity = _hit_entity(hit)
        metadata = entity.get("metadata") or {}
        text = str(entity.get("text") or "")
        print("=" * 100)
        print(f"rank = {rank}")
        print(f"score = {entity.get('_score')}")
        print(f"id = {entity.get('_id')}")
        print(f"source = {entity.get('source')}")
        print(f"chunk_id = {metadata.get('chunk_id')}")
        print(f"doc_version = {metadata.get('doc_version')}")
        print(f"article_id = {entity.get('article_id') or metadata.get('article_id')}")
        print(f"applicable_subjects = {entity.get('applicable_subjects') or metadata.get('applicable_subjects')}")
        print(f"text_unicode_escape = {_unicode_escape(text, 160)}")
        print(f"text_utf8_hex = {_utf8_hex(text, 80)}")
        print("text =")
        print(text[:1200])

    if hits:
        first = _hit_entity(hits[0])
        print("=" * 100)
        print("FIRST_TEXT_ONLY")
        print(str(first.get("text") or "")[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
