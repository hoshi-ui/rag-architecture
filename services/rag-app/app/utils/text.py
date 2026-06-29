"""Shared text helpers."""

import re
from typing import Iterable, List


DIRTY_BLOCK_LABEL_RE = re.compile(
    r"^\s*(?:[?？\ufffd]{2,}|(?:\?[\s?？\ufffd]*){2,})\s*\d+\s*(?:\u5757|block|chunk)\s*$",
    re.IGNORECASE,
)
DIRTY_BLOCK_PREFIX_RE = re.compile(
    r"^\s*(?:[?？\ufffd]{2,}|(?:\?[\s?？\ufffd]*){2,})\s*\d+\s*(?:\u5757|block|chunk)\s+",
    re.IGNORECASE,
)


def normalize_query(query: str) -> str:
    value = (query or "").strip()
    return " ".join(value.split())


def dedupe_keep_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        item = (value or "").strip()
        if item and item not in out:
            out.append(item)
    return out


def sanitize_index_text(value: str) -> str:
    """Remove parser/template noise that must never enter raw/vector/FTS text."""
    text = str(value or "")
    if not text:
        return ""
    lines = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = line.strip()
        if stripped and DIRTY_BLOCK_LABEL_RE.fullmatch(stripped):
            continue
        lines.append(DIRTY_BLOCK_PREFIX_RE.sub("", line).rstrip())
    return "\n".join(lines).strip()


__all__ = ["dedupe_keep_order", "normalize_query", "sanitize_index_text"]
