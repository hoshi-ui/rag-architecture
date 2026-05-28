import argparse
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime
from typing import Dict, Iterable, List, Set, Tuple

from pymilvus import Collection, connections


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DB = os.path.join(ROOT_DIR, "services", "rag-app", "uploads", "lexical_index.db")
DEFAULT_REPORT = os.path.join(ROOT_DIR, "services", "rag-app", "uploads", "milvus_sqlite_reconcile_report.json")


def sqlite_source_set(db_path: str, table: str) -> Set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            f"SELECT DISTINCT source FROM {table} WHERE source IS NOT NULL AND TRIM(source) != ''"
        ).fetchall()
        return {(row[0] or "").strip() for row in rows if (row[0] or "").strip()}
    finally:
        conn.close()


def connect_collection(host: str, port: int, user: str, password: str, secure: bool, collection_name: str) -> Collection:
    connections.connect(
        alias="default",
        uri=f"http://{host}:{port}",
        user=user,
        password=password,
        secure=secure,
    )
    collection = Collection(collection_name)
    collection.load()
    return collection


def milvus_source_counter(collection: Collection, batch_size: int = 5000) -> Tuple[int, Counter]:
    iterator = collection.query_iterator(
        batch_size=batch_size,
        limit=-1,
        expr="id >= 0",
        output_fields=["source"],
    )
    counts: Counter = Counter()
    rows_read = 0
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            for row in batch:
                source = (row.get("source") or "").strip()
                if source:
                    counts[source] += 1
            rows_read += len(batch)
    finally:
        iterator.close()
    return rows_read, counts


def build_report(db_path: str, collection: Collection) -> Dict[str, object]:
    sql_documents = sqlite_source_set(db_path, "documents")
    sql_status = sqlite_source_set(db_path, "doc_status")
    sql_chunks = sqlite_source_set(db_path, "chunks_meta")
    sql_union = sql_documents | sql_status | sql_chunks

    milvus_rows, milvus_counts = milvus_source_counter(collection)
    milvus_sources = set(milvus_counts)
    overlap = sorted(milvus_sources & sql_union)
    orphans = sorted(milvus_sources - sql_union)
    sqlite_only = sorted(sql_union - milvus_sources)

    return {
        "generated_at": datetime.now().isoformat(),
        "sqlite": {
            "db_path": db_path,
            "documents_sources": sorted(sql_documents),
            "doc_status_sources": sorted(sql_status),
            "chunks_meta_sources": sorted(sql_chunks),
            "union_sources": sorted(sql_union),
        },
        "milvus": {
            "collection": collection.name,
            "rows": milvus_rows,
            "unique_sources": len(milvus_sources),
            "top_sources": [{"source": src, "rows": cnt} for src, cnt in milvus_counts.most_common(50)],
            "all_sources": sorted(milvus_sources),
        },
        "reconciliation": {
            "overlap_count": len(overlap),
            "orphan_count": len(orphans),
            "sqlite_only_count": len(sqlite_only),
            "overlap_sources": overlap,
            "orphan_sources": orphans,
            "sqlite_only_sources": sqlite_only,
            "healthy": (len(orphans) == 0),
        },
    }


def delete_sources(collection: Collection, sources: Iterable[str]) -> List[Dict[str, object]]:
    deleted = []
    for source in sources:
        expr = f"source == {json.dumps(source, ensure_ascii=False)}"
        collection.delete(expr)
        deleted.append({"source": source, "expr": expr})
    return deleted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and reconcile Milvus sources against local SQLite source sets.")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--collection", default=os.getenv("MILVUS_COLLECTION", "rag_documents"))
    parser.add_argument("--host", default=os.getenv("MILVUS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MILVUS_PORT", "19530")))
    parser.add_argument("--user", default=os.getenv("MILVUS_USER", "minioadmin"))
    parser.add_argument("--password", default=os.getenv("MILVUS_PASSWORD", "minioadmin"))
    parser.add_argument("--secure", action="store_true", default=os.getenv("MILVUS_SECURE", "false").lower() == "true")
    parser.add_argument("--cleanup-orphans", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collection = connect_collection(args.host, args.port, args.user, args.password, args.secure, args.collection)

    before = build_report(args.db, collection)
    payload: Dict[str, object] = {"before": before}

    cleanup_applied = False
    if args.cleanup_orphans:
        if not args.confirm:
            raise SystemExit("Refusing to delete orphan Milvus sources without --confirm")
        orphan_sources = before["reconciliation"]["orphan_sources"]
        deleted = delete_sources(collection, orphan_sources)
        cleanup_applied = True
        collection.flush()
        payload["cleanup"] = {
            "applied": True,
            "deleted_count": len(deleted),
            "deleted_sources": deleted,
        }

    after = build_report(args.db, collection)
    payload["after"] = after
    payload["cleanup_applied"] = cleanup_applied

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()