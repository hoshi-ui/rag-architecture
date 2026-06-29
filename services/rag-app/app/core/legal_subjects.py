import re
from typing import Any, List


SUBJECT_SUFFIXES = ("行为", "职责", "责任", "限制", "要求", "义务", "处罚", "罚则")


def clean_subject_terms(values: Any, limit: int = 8) -> List[str]:
    if isinstance(values, str):
        raw_values = re.split(r"[,，;；、\n]+", values)
    elif isinstance(values, (list, tuple, set)):
        raw_values = list(values)
    else:
        return []
    out: List[str] = []
    for item in raw_values:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if not text or text in out:
            continue
        out.append(text)
        if len(out) >= max(1, int(limit)):
            break
    return out


def subject_term_aliases(term: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(term or "")).strip()
    if not text:
        return []
    aliases = [text]
    for suffix in SUBJECT_SUFFIXES:
        if text.endswith(suffix) and len(text) > len(suffix) + 1:
            root = text[: -len(suffix)].strip()
            if root:
                aliases.append(root)
                if suffix == "行为":
                    aliases.extend([f"{root}人", f"{root}单位"])
            break
    return list(dict.fromkeys(alias for alias in aliases if alias))


def normalize_subject_terms(values: Any, limit: int = 8) -> List[str]:
    terms = clean_subject_terms(values, limit=limit)
    aliases = [alias for term in terms for alias in subject_term_aliases(term)]
    return list(dict.fromkeys(aliases or terms))[: max(1, int(limit))]
