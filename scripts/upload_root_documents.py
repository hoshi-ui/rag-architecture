#!/usr/bin/env python3
"""Upload all root-level PDF/DOCX files to the RAG app."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib import error, request


SUPPORTED_SUFFIXES = {".docx", ".pdf"}


def discover_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for item in root.iterdir():
        if not item.is_file():
            continue
        if item.name.startswith("~$"):
            continue
        if item.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(item)
    return sorted(files, key=lambda p: p.name.lower())


def encode_multipart_file(field_name: str, path: Path) -> tuple[bytes, str]:
    boundary = f"----rag-upload-{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{path.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return header + path.read_bytes() + footer, boundary


def upload_one(api_base: str, path: Path, timeout: float, retries: int) -> dict[str, Any]:
    url = api_base.rstrip("/") + "/documents/upload"
    last_error = ""
    for attempt in range(retries + 1):
        try:
            body, boundary = encode_multipart_file("file", path)
            req = request.Request(
                url,
                data=body,
                method="POST",
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Content-Length": str(len(body)),
                },
            )
            with request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
            try:
                payload: Any = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"raw": raw}
            return {"file": path.name, "ok": True, "response": payload}
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            last_error = f"HTTP {exc.code}: {raw}"
        except Exception as exc:
            last_error = str(exc)
        if attempt < retries:
            time.sleep(min(2.0 * (attempt + 1), 5.0))
    return {"file": path.name, "ok": False, "error": last_error}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload root-level .docx and .pdf files to the RAG app.")
    parser.add_argument("--root", default=".", help="Directory to scan. Defaults to the current directory.")
    parser.add_argument("--api-base", default="http://localhost:8080", help="RAG app API base URL.")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of parallel uploads. Defaults to 1.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-upload timeout in seconds.")
    parser.add_argument("--retries", type=int, default=1, help="Retries per file after a failed upload.")
    parser.add_argument("--dry-run", action="store_true", help="Only list files that would be uploaded.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"root_not_found: {root}", file=sys.stderr)
        return 2

    files = discover_files(root)
    print(f"root = {root}")
    print(f"found = {len(files)}")
    for path in files:
        print(f"- {path.name}")

    if args.dry_run or not files:
        return 0

    workers = max(1, int(args.concurrency or 1))
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(upload_one, args.api_base, path, args.timeout, args.retries) for path in files]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "OK" if result.get("ok") else "FAILED"
            print(f"{status} {result.get('file')}")
            print(json.dumps(result, ensure_ascii=False))

    failed = [item for item in results if not item.get("ok")]
    print(json.dumps({"total": len(results), "failed": len(failed)}, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
