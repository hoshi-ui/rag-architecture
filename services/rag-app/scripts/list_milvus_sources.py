from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from typing import Any, Dict


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.storage.milvus import VectorDBService


MOJIBAKE_MARKER_CODES = (
    0x951B,
    0x7ED7,
    0x9418,
    0x9471,
    0x6D93,
    0x6D60,
    0x9428,
    0x9366,
    0x94CF,
    0x95C2,
    0x95C1,
    0x95BB,
    0x9429,
    0x93AC,
    0x5A09,
    0x7F03,
    0x741B,
)
MOJIBAKE_MARKER_RE = re.compile(
    r"[\ue000-\uf8ff\ufffd]|" + "|".join(re.escape(chr(code)) for code in MOJIBAKE_MARKER_CODES)
)


def is_suspect(value: Any) -> bool:
    text = str(value or "")
    return bool(MOJIBAKE_MARKER_RE.search(text))


def unicode_escape(value: Any, limit: int = 500) -> str:
    return str(value or "")[:limit].encode("unicode_escape").decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description="List sources and sample text stored in Milvus.")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--show-escaped", action="store_true", help="Also print unicode-escaped source/text for encoding diagnostics.")
    args = parser.parse_args()

    db = VectorDBService()
    db.connect()
    rows = db.client.query(
        collection_name=db.collection_name,
        filter="",
        output_fields=["source", "text", "metadata"],
        limit=max(1, int(args.limit)),
    )

    counts: Counter[str] = Counter()
    samples: Dict[str, Dict[str, Any]] = {}
    for row in rows or []:
        source = str(row.get("source") or "")
        counts[source] += 1
        samples.setdefault(source, row)

    print("row_count =", len(rows or []))
    print("source_count =", len(counts))
    print()
    for source, count in counts.most_common():
        row = samples.get(source) or {}
        metadata = row.get("metadata") or {}
        text = str(row.get("text") or "")
        flags = []
        if is_suspect(source):
            flags.append("SOURCE_MOJIBAKE")
        if is_suspect(text):
            flags.append("TEXT_MOJIBAKE")
        print("=" * 100)
        print("source =", source)
        print("count =", count)
        print("flags =", ",".join(flags) if flags else "OK")
        print("sample_chunk_id =", metadata.get("chunk_id"))
        print("sample_text =", text[: int(args.samples) * 80])
        if args.show_escaped:
            print("source_unicode_escape =", unicode_escape(source))
            print("sample_text_unicode_escape =", unicode_escape(text, int(args.samples) * 80))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
