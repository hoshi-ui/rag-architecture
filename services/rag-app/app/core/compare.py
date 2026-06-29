import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.documents import entity_registry


@dataclass
class CompareSubjectSpan:
    raw_text: str
    clean_text: str
    span_start: int = -1
    span_end: int = -1
    connector_before: str = ""
    doc_like: bool = False
    source: str = ""
    match_kind: str = ""
    prior: float = 0.0


@dataclass
class ComparePlan:
    raw_query: str
    has_intent: bool = False
    route: str = ""
    reason: str = "not_compare"
    required: bool = False
    resolved: bool = False
    subject_zone: str = ""
    tail_span: str = ""
    subjects: List[CompareSubjectSpan] = field(default_factory=list)
    matched_sources: List[str] = field(default_factory=list)
    whole_query_sources: List[str] = field(default_factory=list)
    missing_targets: List[str] = field(default_factory=list)
    doc_like_subjects: List[str] = field(default_factory=list)
    common_aspects: List[str] = field(default_factory=list)
    topic_pair: List[str] = field(default_factory=list)
    canonical_aspects: List[str] = field(default_factory=list)
    expanded_aspects: List[str] = field(default_factory=list)
    source_subqueries: Dict[str, Dict[str, str]] = field(default_factory=dict)
    compare_status: str = "not_compare"


_COMPARE_INTENT_MARKERS = (
    "对比",
    "比较",
    "区别",
    "差异",
    "相比",
    "分别",
    "各自",
    "不同",
)
_COMPARE_PAIR_CONNECTORS_RE = re.compile(
    r"(?:和|与|同|跟|及|以及|vs|VS|Vs|versus)"
)
_COMPARE_RESULT_MARKERS = ("分别", "各自", "区别", "差异", "不同")
_COMPARE_LEAD_RE = re.compile(
    r"^(?:请|请帮我|帮我|麻烦|我想|想)?\s*"
    r"(?:对比一下|比较一下|对比|比较|"
    r"区别|分析|分别比较|分别说明|"
    r"分别介绍)?\s*"
)
_COMPARE_CONNECTOR_RE = re.compile(r"(?:和|与|同|跟|及|以及|vs|VS|Vs|versus)")
_COMPARE_SPLIT_RE = re.compile(
    r"(\s*(?:vs|VS|Vs|versus)\s*|"
    r"(?:(?<=条例)|(?<=办法)|(?<=规定)|(?<=规则)|"
    r"(?<=细则)|(?<=决定)|(?<=通知)|(?<=》))"
    r"\s*(?:和|与|同|跟|及|以及)\s*)"
)
_COMPARE_DOC_SUFFIXES = ("条例", "办法", "规定", "规则", "细则", "决定", "通知")
_COMPARE_GENERIC_DOC_TERMS = {
    "处罚",
    "要求",
    "责任",
    "程序",
    "措施",
    "条款",
    "职责",
    "标准",
    "条件",
    "管理",
    "规定",
    "规则",
    "办法",
    "条例",
    "通知",
    "决定",
}
_COMPARE_TAIL_MARKERS = (
    "区别",
    "差异",
    "不同",
    "分别",
    "各自",
    "哪个",
    "哪一个",
    "有什么",
    "有何",
    "哪些",
    "怎么",
    "如何",
    "是否",
)
_COMPARE_ASPECT_CANONICAL_MAP = {
    "处罚措施": "处罚",
    "罚则": "处罚",
    "法律责任": "责任",
    "管理职责": "职责",
    "登记程序": "程序",
    "申请程序": "程序",
    "办理流程": "程序",
    "要求和条件": "要求",
}
_COMPARE_ASPECT_EXPANSIONS = {
    "处罚": ["罚款", "罚则", "法律责任"],
    "责任": ["职责", "义务", "法律责任", "管理职责"],
    "程序": ["流程", "步骤", "申请", "登记"],
    "要求": ["条件", "标准", "规范", "限制"],
    "监督": ["监管", "检查", "监督检查"],
}
_REGION_HINT_RE = re.compile(
    r"([一-鿿]{2,12}(?:省|市|自治区|自治州|地区|盟|县|区))"
)


def query_has_compare_intent(
    query: str,
    normalize_query: Callable[[Any], str],
    classify_question_type: Callable[[str], str],
) -> bool:
    q = normalize_query(query)
    if not q:
        return False
    if classify_question_type(q) == "compare":
        return True
    lower_q = q.lower()
    if any(marker in q for marker in _COMPARE_INTENT_MARKERS):
        return True
    if any(marker in lower_q for marker in ("vs", "versus", "compare")):
        return True
    return bool(_COMPARE_PAIR_CONNECTORS_RE.search(q)) and any(marker in q for marker in _COMPARE_RESULT_MARKERS)


def clean_compare_subject_text(
    text: str,
    normalize_query: Callable[[Any], str],
    strip_section_question_tail: Callable[[str], str],
) -> str:
    subject = normalize_query(text)
    if not subject:
        return ""
    subject = re.sub(
        r"^(?:请|请帮我|帮我|麻烦|"
        r"我想了解|想了解|我想|想)\s*",
        "",
        subject,
    )
    subject = re.sub(
        r"^(?:对比一下|比较一下|对比|比较|"
        r"区别一下|区别|分析一下|分析|"
        r"分别比较一下|分别比较|"
        r"分别说明|分别说说|分别介绍)\s*",
        "",
        subject,
    )
    subject = re.sub(
        r"(?:之间|各自|分别|的区别|"
        r"的差异|的不同)+$",
        "",
        subject,
    )
    subject = re.sub(r"[，。；;：:？?！!\.\s]+$", "", subject)
    subject = re.sub(r"(?:里的?|中的?|内的?)$", "", subject)
    subject = strip_section_question_tail(subject) or normalize_query(subject)
    return re.sub(r"[，。；;：:？?！!\.\s]+$", "", subject)


def extract_compare_subject_spans(
    query: str,
    normalize_query: Callable[[Any], str],
    query_has_compare_intent_fn: Callable[[str], bool],
    clean_subject_text_fn: Callable[[str], str],
    subject_factory: Callable[..., Any],
) -> Dict[str, Any]:
    q = normalize_query(query)
    if not q or not query_has_compare_intent_fn(q):
        return {"subjects": [], "subject_zone": "", "tail_span": ""}

    lead_match = _COMPARE_LEAD_RE.match(q)
    lead_end = lead_match.end() if lead_match else 0
    working = q[lead_end:]
    working = re.sub(r"[。！？?]+$", "", working)

    connector_candidates = list(_COMPARE_CONNECTOR_RE.finditer(working))
    connector_match = connector_candidates[0] if connector_candidates else None
    for candidate in connector_candidates:
        prefix = working[: candidate.start()].rstrip()
        if prefix.endswith("》") or any(prefix.endswith(suffix) for suffix in _COMPARE_DOC_SUFFIXES):
            connector_match = candidate
            break
    if not connector_match:
        return {"subjects": [], "subject_zone": working.strip(), "tail_span": ""}

    tail_start: Optional[int] = None
    for marker in _COMPARE_TAIL_MARKERS:
        idx = working.find(marker, connector_match.end())
        if idx != -1 and (tail_start is None or idx < tail_start):
            tail_start = idx
    for punct in ("，", ",", "；", ";", "：", ":"):
        idx = working.find(punct, connector_match.end())
        if idx != -1 and (tail_start is None or idx < tail_start):
            tail_start = idx

    subject_zone = working[:tail_start] if tail_start is not None else working
    tail_span = working[tail_start:] if tail_start is not None else ""
    parts = _COMPARE_SPLIT_RE.split(subject_zone)

    subjects: List[Any] = []
    cursor = lead_end
    connector_before = ""
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            cursor += len(part)
            connector_before = part.strip()
            continue
        raw = part or ""
        raw_value = raw.strip()
        if not raw_value:
            cursor += len(part)
            continue
        leading_ws = len(raw) - len(raw.lstrip())
        trailing_ws = len(raw.rstrip())
        clean_text = clean_subject_text_fn(raw_value)
        subjects.append(
            subject_factory(
                raw_text=raw_value,
                clean_text=clean_text,
                span_start=cursor + leading_ws,
                span_end=cursor + trailing_ws,
                connector_before=connector_before,
            )
        )
        cursor += len(part)
        connector_before = ""
        if len(subjects) >= 4:
            break

    return {
        "subjects": subjects,
        "subject_zone": subject_zone.strip(),
        "tail_span": tail_span.strip(),
    }


def looks_like_compare_document_target(
    text: str,
    normalize_query: Callable[[Any], str],
    query_anchor_terms: Callable[[str], List[str]],
) -> bool:
    q = normalize_query(text)
    if not q:
        return False
    matched_suffix = next((suffix for suffix in _COMPARE_DOC_SUFFIXES if q.endswith(suffix)), "")
    if not matched_suffix:
        return False
    core = q[: -len(matched_suffix)].strip()
    if not core:
        return False
    core_terms = [term for term in query_anchor_terms(core) if term not in _COMPARE_GENERIC_DOC_TERMS]
    if core_terms:
        return True
    return core not in _COMPARE_GENERIC_DOC_TERMS and len(core) >= 2


def compare_clean_aspect_span(text: str, normalize_query: Callable[[Any], str]) -> str:
    value = normalize_query(text)
    if not value:
        return ""
    value = re.sub(r"^[，。；;：:\s]+", "", value)
    value = re.sub(
        r"^(?:里的?|中的?|内的?|关于|就|在)\s*",
        "",
        value,
    )
    value = re.sub(
        r"^(?:对比一下|比较一下|对比|比较|"
        r"区别一下|区别|分析一下|分析|"
        r"分别|各自)\s*",
        "",
        value,
    )
    value = re.sub(
        r"(?:有什么|有何|有哪些|怎么|如何|"
        r"是否|吗|呢|的区别|的差异|的不同|"
        r"。|？|\?|!)+$",
        "",
        value,
    )
    return normalize_query(value)


def canonicalize_compare_aspect(
    term: str,
    normalize_query: Callable[[Any], str],
    normalize_coverage_aspect: Callable[[Any], str],
) -> str:
    value = compare_clean_aspect_span(term, normalize_query)
    value = normalize_coverage_aspect(value) or value
    value = _COMPARE_ASPECT_CANONICAL_MAP.get(value, value)
    return normalize_query(value)


def expand_compare_aspects(
    aspects: List[str],
    normalize_query: Callable[[Any], str],
    normalize_coverage_aspect: Callable[[Any], str],
    coverage_aspect_variants: Callable[[str], List[str]],
    limit: int = 8,
) -> List[str]:
    out: List[str] = []
    for aspect in aspects or []:
        canonical = canonicalize_compare_aspect(aspect, normalize_query, normalize_coverage_aspect)
        candidates = [canonical]
        candidates.extend(coverage_aspect_variants(canonical))
        candidates.extend(_COMPARE_ASPECT_EXPANSIONS.get(canonical, []))
        if "责任" in canonical and canonical not in {
            "责任",
            "安全责任",
            "管理职责",
            "政府职责",
        }:
            candidates.extend([canonical.replace("责任", "职责"), canonical.replace("责任", "要求")])
        if "处罚" in canonical and canonical != "处罚":
            candidates.extend(["处罚", "法律责任", "罚则"])
        if "程序" in canonical:
            candidates.extend(["流程", "步骤", "审批"])
        for candidate in candidates:
            value = normalize_query(candidate)
            if len(value) < 2 or value in out:
                continue
            out.append(value)
            if len(out) >= limit:
                return out
    return out


def compare_aspects_from_span(
    text: str,
    normalize_query: Callable[[Any], str],
    normalize_coverage_aspect: Callable[[Any], str],
    query_anchor_terms: Callable[[str], List[str]],
    extract_section_query_targets: Callable[[str], List[str]],
    query_semantic_aspects: Callable[[str], Dict[str, Any]],
    query_match_terms: Callable[[str], List[str]],
    limit: int = 4,
) -> List[str]:
    raw = compare_clean_aspect_span(text, normalize_query)
    if not raw:
        return []
    out: List[str] = []

    def add(term: str):
        value = canonicalize_compare_aspect(term, normalize_query, normalize_coverage_aspect)
        if len(value) < 2 or value in out:
            return
        if looks_like_compare_document_target(value, normalize_query, query_anchor_terms):
            return
        out.append(value)

    for target in extract_section_query_targets(raw):
        add(target)
    semantic = query_semantic_aspects(raw)
    for term in semantic.get("terms") or []:
        add(term)
    for term in query_match_terms(raw):
        add(term)
    if not out:
        for piece in re.split(r"\s*(?:和|与|同|跟|及|以及|、|vs|VS|Vs|versus)\s*", raw):
            add(piece)
    if not out:
        add(raw)
    return out[:limit]


def compare_unique_texts(
    items: List[str],
    normalize_query: Callable[[Any], str],
    limit: Optional[int] = None,
) -> List[str]:
    out: List[str] = []
    for item in items or []:
        value = normalize_query(item)
        if not value or value in out:
            continue
        out.append(value)
        if limit and len(out) >= limit:
            break
    return out


def extract_single_doc_compare_topic_pair(
    query: str,
    source: str,
    normalize_query: Callable[[Any], str],
    locate_source_mention_span: Callable[[str, str], Any],
    clean_aspect_span: Callable[[str], str],
    canonicalize_aspect: Callable[[str], str],
    looks_like_document_target: Callable[[str], bool],
    aspects_from_span: Callable[..., List[str]],
) -> List[str]:
    q = normalize_query(query)
    _, end = locate_source_mention_span(q, source)
    tail = q[end:] if end != -1 else q
    tail = clean_aspect_span(tail)
    pair: List[str] = []

    def add(term: str):
        value = canonicalize_aspect(term)
        if len(value) < 2 or value in pair or looks_like_document_target(value):
            return
        pair.append(value)

    for piece in re.split(r"\s*(?:和|与|同|跟|及|以及|、|vs|VS|Vs|versus)\s*", tail):
        add(piece)
    if len(pair) < 2:
        for term in aspects_from_span(tail, limit=4):
            add(term)
    return pair[:2]


def build_compare_source_subqueries(
    plan: Any,
    normalize_filename: Callable[[str], str],
    source_display_title: Callable[[str], str],
    normalize_query: Callable[[Any], str],
    looks_like_section_target: Callable[[str], bool],
) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    aspect_terms = list(plan.canonical_aspects or [])
    expanded_terms = list(plan.expanded_aspects or [])
    section_keywords = (
        "责任",
        "处罚",
        "程序",
        "监督",
        "登记",
        "职责",
        "措施",
        "主体",
        "禁止",
        "限制",
    )
    section_terms = [
        term
        for term in expanded_terms
        if looks_like_section_target(term) or any(token in term for token in section_keywords)
    ]

    for subject in plan.subjects:
        source = normalize_filename(subject.source or "")
        if not source:
            continue
        title = source_display_title(source) if source else "文档"
        subject_hint = subject.clean_text or subject.raw_text or title
        raw_text_query = " ".join(compare_unique_texts([title, subject_hint] + aspect_terms + expanded_terms, normalize_query, limit=8))
        section_query = " ".join(compare_unique_texts([title] + section_terms + aspect_terms, normalize_query, limit=6))
        doc_prior_query = " ".join(compare_unique_texts([subject_hint, title] + aspect_terms, normalize_query, limit=6))
        out[source] = {
            "raw_text_query": raw_text_query or title,
            "section_query": section_query or raw_text_query or title,
            "doc_prior_query": doc_prior_query or raw_text_query or title,
        }

    if plan.route == "single_doc_compare" and len(plan.matched_sources) == 1:
        source = plan.matched_sources[0]
        title = source_display_title(source) if source else "文档"
        topic_terms = list(plan.topic_pair or aspect_terms or expanded_terms)
        out[source] = {
            "raw_text_query": " ".join(compare_unique_texts([title] + topic_terms + expanded_terms, normalize_query, limit=8)) or title,
            "section_query": " ".join(compare_unique_texts([title] + topic_terms, normalize_query, limit=6)) or title,
            "doc_prior_query": " ".join(compare_unique_texts([title] + topic_terms[:2], normalize_query, limit=6)) or title,
        }
    return out


def extract_region_hint(text: str, normalize_query: Callable[[Any], str]) -> str:
    value = normalize_query(text)
    if not value:
        return ""
    match = _REGION_HINT_RE.search(value)
    return match.group(1) if match else ""


def resolve_compare_subject_source(
    subject: str,
    normalize_query: Callable[[Any], str],
    normalize_filename: Callable[[str], str],
    source_display_title: Callable[[str], str],
    looks_like_document_target: Callable[[str], bool],
    extract_strong_title_source_matches: Callable[..., List[Dict[str, Any]]],
    rank_title_source_matches: Callable[..., List[Dict[str, Any]]],
    build_doc_recall_plan: Callable[..., List[Dict[str, Any]]],
    source_lock_validator: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    target = normalize_query(subject)
    if not target:
        return {"subject": "", "source": "", "match_kind": "", "doc_like": False, "prior": 0.0}

    doc_like = looks_like_document_target(target)
    region_hint = extract_region_hint(target, normalize_query)
    loose_source_probe = bool(doc_like or region_hint)
    if not loose_source_probe:
        return {
            "subject": target,
            "source": "",
            "match_kind": "",
            "doc_like": False,
            "prior": 0.0,
        }

    def source_matches_region(source: str) -> bool:
        if not region_hint:
            return True
        title = normalize_query(source_display_title(source))
        if not title:
            return False
        title_norm = entity_registry.normalize_entity_text(title)
        hint_norm = entity_registry.normalize_entity_text(region_hint)
        compact_hint = entity_registry.strip_admin_suffixes(region_hint)
        compact_norm = entity_registry.normalize_entity_text(compact_hint)
        prefix_norm = hint_norm[:2] if len(hint_norm) >= 2 else ""
        return any(
            candidate and candidate in title_norm
            for candidate in (hint_norm, compact_norm, prefix_norm)
        )

    def validate_source(source: str, *, prior: float = 0.0, match_kind: str = "") -> Dict[str, Any]:
        if not callable(source_lock_validator):
            return {"accepted": True}
        try:
            return source_lock_validator(
                target,
                target,
                source,
                prior=prior,
                match_kind=match_kind,
            )
        except Exception:
            return {"accepted": True, "error": "validator_exception"}

    strong_matches = extract_strong_title_source_matches(target, limit=3)
    for top in strong_matches or []:
        src = normalize_filename(top.get("source") or "")
        if not src or not source_matches_region(src):
            continue
        validation = validate_source(src, prior=1.0, match_kind=str(top.get("match_kind") or ""))
        if not validation.get("accepted"):
            continue
        return {
            "subject": target,
            "source": src,
            "match_kind": top.get("match_kind") or "",
            "doc_like": True,
            "prior": 1.0,
            "source_lock_validation": validation,
        }

    ranked_matches = rank_title_source_matches(target, limit=3, include_topic_like=True)
    for top in ranked_matches or []:
        src = normalize_filename(top.get("source") or "")
        if not src or not source_matches_region(src):
            continue
        if (top.get("match_kind") or "") == "topic_like_title" and float(top.get("score") or 0.0) >= 2.5:
            prior = min(float(top.get("score") or 0.0) / 4.0, 1.0)
            validation = validate_source(src, prior=prior, match_kind=str(top.get("match_kind") or ""))
            if not validation.get("accepted"):
                continue
            return {
                "subject": target,
                "source": src,
                "match_kind": top.get("match_kind") or "",
                "doc_like": True,
                "prior": prior,
                "source_lock_validation": validation,
            }

    recall_plan = build_doc_recall_plan(target, limit=3)
    for entry in recall_plan or []:
        reasons = set((entry or {}).get("reasons") or [])
        prior = float((entry or {}).get("prior") or 0.0)
        title_score = float((entry or {}).get("title_score") or 0.0)
        src = normalize_filename((entry or {}).get("source") or "")
        if not src or not source_matches_region(src):
            continue
        if title_score > 0 or (
            reasons.intersection({"documents_fts", "doc_term_overlap", "title_alias_substring"}) and prior >= 0.45
        ):
            validation = validate_source(src, prior=prior, match_kind="doc_recall")
            if not validation.get("accepted"):
                continue
            return {
                "subject": target,
                "source": src,
                "match_kind": "doc_recall",
                "doc_like": True,
                "prior": prior,
                "source_lock_validation": validation,
            }

    return {
        "subject": target,
        "source": "",
        "match_kind": "",
        "doc_like": bool(loose_source_probe),
        "prior": 0.0,
    }


def build_compare_plan(
    query: str,
    plan_factory: Callable[..., Any],
    normalize_query: Callable[[Any], str],
    normalize_filename: Callable[[str], str],
    query_has_compare_intent_fn: Callable[[str], bool],
    extract_subject_spans: Callable[[str], Dict[str, Any]],
    resolve_subject_source: Callable[[str], Dict[str, Any]],
    compare_unique: Callable[..., List[str]],
    extract_strong_title_source_matches: Callable[..., List[Dict[str, Any]]],
    extract_common_aspects: Callable[[Any], List[str]],
    extract_single_doc_topic_pair: Callable[[str, str], List[str]],
    canonicalize_aspect: Callable[[str], str],
    expand_aspects: Callable[..., List[str]],
    build_source_subqueries: Callable[[Any], Dict[str, Dict[str, str]]],
) -> Any:
    q = normalize_query(query)
    plan = plan_factory(raw_query=q, has_intent=bool(q and query_has_compare_intent_fn(q)))
    if not plan.has_intent:
        return plan

    subject_info = extract_subject_spans(q)
    plan.subjects = list(subject_info.get("subjects") or [])
    plan.subject_zone = subject_info.get("subject_zone") or ""
    plan.tail_span = subject_info.get("tail_span") or ""

    for subject in plan.subjects:
        match = resolve_subject_source(subject.clean_text or subject.raw_text)
        subject.doc_like = bool(match.get("doc_like"))
        subject.source = normalize_filename(match.get("source") or "")
        subject.match_kind = match.get("match_kind") or ""
        subject.prior = float(match.get("prior") or 0.0)

    plan.doc_like_subjects = compare_unique(
        [subject.clean_text or subject.raw_text for subject in plan.subjects if subject.doc_like]
    )
    plan.missing_targets = compare_unique(
        [subject.clean_text or subject.raw_text for subject in plan.subjects if subject.doc_like and not subject.source]
    )

    whole_query_title_matches = extract_strong_title_source_matches(q, limit=2)
    plan.whole_query_sources = [
        normalized
        for normalized in (normalize_filename(item.get("source") or "") for item in whole_query_title_matches)
        if normalized
    ]
    plan.matched_sources = compare_unique([subject.source for subject in plan.subjects if subject.source])
    plan.route = "open_topic_compare"
    plan.required = False
    plan.resolved = False
    plan.reason = "not_needed"

    if len(plan.doc_like_subjects) >= 2:
        if len(plan.matched_sources) >= 2:
            plan.route = "multi_doc_compare"
            plan.resolved = not bool(plan.missing_targets)
            if plan.missing_targets:
                plan.reason = "compare_target_incomplete"
        elif len(plan.matched_sources) == 1:
            plan.route = "single_doc_compare"
            plan.resolved = True
            plan.reason = "compare_target_not_found_degraded"
        else:
            plan.route = "compare_targets_not_found"
            plan.required = True
            plan.reason = "compare_targets_not_found"
    elif len(plan.doc_like_subjects) == 1:
        if len(plan.matched_sources) == 1:
            plan.route = "single_doc_compare"
            plan.resolved = True
        else:
            plan.route = "compare_target_not_found"
            plan.required = True
            plan.reason = "compare_target_not_found"
    elif len(plan.whole_query_sources) == 1:
        plan.route = "single_doc_compare"
        plan.matched_sources = compare_unique(plan.whole_query_sources)
        plan.resolved = True

    if plan.route == "multi_doc_compare":
        plan.common_aspects = extract_common_aspects(plan)
    elif plan.route == "single_doc_compare" and len(plan.matched_sources) == 1:
        plan.topic_pair = extract_single_doc_topic_pair(q, plan.matched_sources[0])
    elif plan.route == "open_topic_compare":
        plan.topic_pair = compare_unique([subject.clean_text or subject.raw_text for subject in plan.subjects], limit=2)

    raw_aspects = list(plan.common_aspects or plan.topic_pair)
    plan.canonical_aspects = compare_unique([canonicalize_aspect(item) for item in raw_aspects], limit=4)
    plan.expanded_aspects = expand_aspects(plan.canonical_aspects, limit=8)
    plan.source_subqueries = build_source_subqueries(plan)

    if plan.route == "multi_doc_compare":
        plan.compare_status = "plan_ready"
    elif plan.route == "single_doc_compare":
        plan.compare_status = "single_doc_ready"
    elif plan.route in {"compare_target_not_found", "compare_targets_not_found"}:
        plan.compare_status = "target_missing"
    elif plan.route == "open_topic_compare":
        plan.compare_status = "open_topic"
    return plan


def compare_plan_to_dict(plan: Any) -> Dict[str, Any]:
    return {
        "is_compare": bool(plan.has_intent),
        "route": plan.route,
        "reason": plan.reason,
        "required": bool(plan.required),
        "resolved": bool(plan.resolved),
        "subjects": [subject.clean_text or subject.raw_text for subject in plan.subjects],
        "subject_spans": [
            {
                "raw_text": subject.raw_text,
                "clean_text": subject.clean_text,
                "span_start": subject.span_start,
                "span_end": subject.span_end,
                "doc_like": subject.doc_like,
                "source": subject.source,
                "match_kind": subject.match_kind,
                "prior": subject.prior,
            }
            for subject in plan.subjects
        ],
        "subject_matches": [
            {
                "subject": subject.clean_text or subject.raw_text,
                "source": subject.source,
                "match_kind": subject.match_kind,
                "doc_like": subject.doc_like,
                "prior": subject.prior,
            }
            for subject in plan.subjects
        ],
        "sources": list(plan.matched_sources),
        "doc_like_subjects": list(plan.doc_like_subjects),
        "missing_doc_targets": list(plan.missing_targets),
        "common_aspects": list(plan.common_aspects),
        "topic_pair": list(plan.topic_pair),
        "canonical_aspects": list(plan.canonical_aspects),
        "expanded_aspects": list(plan.expanded_aspects),
        "source_subqueries": dict(plan.source_subqueries),
        "target_text": "、".join(
            plan.missing_targets[:3] or [subject.clean_text or subject.raw_text for subject in plan.subjects[:2]]
        ),
        "clarification": "",
        "strip_title_mentions": bool(plan.matched_sources),
        "compare_status": plan.compare_status,
        "compare_plan": {
            "raw_query": plan.raw_query,
            "subject_zone": plan.subject_zone,
            "tail_span": plan.tail_span,
            "route": plan.route,
            "reason": plan.reason,
            "required": plan.required,
            "resolved": plan.resolved,
            "subjects": [
                {
                    "raw_text": subject.raw_text,
                    "clean_text": subject.clean_text,
                    "span_start": subject.span_start,
                    "span_end": subject.span_end,
                    "doc_like": subject.doc_like,
                    "source": subject.source,
                    "match_kind": subject.match_kind,
                    "prior": subject.prior,
                }
                for subject in plan.subjects
            ],
            "matched_sources": list(plan.matched_sources),
            "whole_query_sources": list(plan.whole_query_sources),
            "missing_targets": list(plan.missing_targets),
            "doc_like_subjects": list(plan.doc_like_subjects),
            "common_aspects": list(plan.common_aspects),
            "topic_pair": list(plan.topic_pair),
            "canonical_aspects": list(plan.canonical_aspects),
            "expanded_aspects": list(plan.expanded_aspects),
            "source_subqueries": dict(plan.source_subqueries),
            "compare_status": plan.compare_status,
        },
    }


def compare_focus_text(
    compare_plan: Optional[Dict[str, Any]],
    normalize_coverage_aspect: Callable[[Any], str],
    normalize_query: Callable[[Any], str],
) -> str:
    plan = dict(compare_plan or {})
    focus_terms: List[str] = []
    for candidate in (
        plan.get("common_aspects") or [],
        plan.get("topic_pair") or [],
        plan.get("canonical_aspects") or [],
        plan.get("expanded_aspects") or [],
    ):
        for term in candidate:
            normalized = normalize_coverage_aspect(term) or normalize_query(term)
            if normalized and normalized not in focus_terms:
                focus_terms.append(normalized)
    return "、".join(focus_terms[:2]) or "核心差异"


def build_multi_doc_compare_grounded_answer(
    source_refs: List[Dict[str, Any]],
    compare_plan: Optional[Dict[str, Any]],
    focus_text_fn: Callable[[Optional[Dict[str, Any]]], str],
) -> str:
    if len(source_refs) < 2:
        return "当前证据不足，无法完成两个文档之间的可靠对比。"
    focus_text = focus_text_fn(compare_plan)
    lines = [f"围绕{focus_text}，两份资料的直接依据分别见 [{source_refs[0]['index']}][{source_refs[1]['index']}]。"]
    for ref in source_refs[:2]:
        section_text = f"{ref['section']}中" if ref.get("section") else "相关片段中"
        snippet = ref.get("snippet") or "未提供可展示片段"
        lines.append(f"- {ref['title']}：{section_text}{snippet} [{ref['index']}]")
    return "\n".join(lines)


def build_single_doc_compare_grounded_answer(
    topic_refs: List[Dict[str, Any]],
    compare_plan: Optional[Dict[str, Any]],
    focus_text_fn: Callable[[Optional[Dict[str, Any]]], str],
) -> str:
    if len(topic_refs) < 2:
        return "当前证据不足，无法完成同一文档内的可靠对比。"
    plan = dict(compare_plan or {})
    topic_pair = [str(item).strip() for item in (plan.get("topic_pair") or []) if str(item).strip()]
    focus_text = "、".join(topic_pair[:2]) or focus_text_fn(plan)
    lines = [f"同一文档中，围绕{focus_text}的直接依据见 [{topic_refs[0]['index']}][{topic_refs[1]['index']}]。"]
    missing_targets = [str(item).strip() for item in (plan.get("missing_targets") or []) if str(item).strip()]
    if missing_targets:
        lines.append(f"未能在当前证据中确认这些目标：{'、'.join(missing_targets[:4])}。")
    for ref in topic_refs[:2]:
        label = ref.get("label") or ref.get("section") or "对比项"
        section_text = f"{ref['section']}中" if ref.get("section") else "相关片段中"
        snippet = ref.get("snippet") or "未提供可展示片段"
        lines.append(f"- {label}：{section_text}{snippet} [{ref['index']}]")
    return "\n".join(lines)


def build_compare_evidence_failure_prompt(source_statuses: List[Dict[str, Any]]) -> str:
    if not source_statuses:
        return "当前没有召回到足够的可对比证据。"

    def status_detail(item: Dict[str, Any]) -> str:
        title = (item.get("title") or item.get("source") or "文档").strip()
        status = str(item.get("status") or "")
        observations = dict(item.get("observations") or {})
        uncovered = [str(term).strip() for term in (observations.get("uncovered_aspects") or []) if str(term).strip()]
        focus_text = "、".join(uncovered[:2]) if uncovered else "关键方面"
        if status == "not_found":
            return f"{title} 未找到可用证据"
        if status == "evidence_insufficient":
            if uncovered:
                return f"{title} 缺少 {focus_text} 的直接依据"
            return f"{title} 证据不足"
        if status == "comparable_partial":
            return f"{title} 只有部分可比证据"
        return f"{title} 暂无可靠对比证据"

    return "无法完成可靠对比：" + "；".join(status_detail(item) for item in source_statuses)


def is_version_switch_query(query: str, normalize_query: Callable[[str], str]) -> bool:
    normalized = normalize_query(query)
    return ("现行有效" in normalized) or ("最新" in normalized)


def build_compare_clarification_prompt(
    missing_targets: List[str],
    matched_sources: Optional[List[str]] = None,
) -> str:
    targets = [str(item or "").strip() for item in (missing_targets or []) if str(item or "").strip()]
    matched = [str(item or "").strip() for item in (matched_sources or []) if str(item or "").strip()]
    if targets and matched:
        return "请补充需要对比的文件或法规。已识别：" + "、".join(matched[:3]) + "；待确认：" + "、".join(targets[:3])
    if targets:
        return "请确认要对比的文件或法规：" + "、".join(targets[:3])
    return "请先说明你想对比哪几部法规或文件。"


def build_compare_target_not_found_prompt(
    missing_targets: List[str],
    matched_sources: Optional[List[str]] = None,
    *,
    doc_get,
    filename_stem,
) -> str:
    missing_items = [item.strip() for item in (missing_targets or []) if (item or "").strip()]
    found_titles: List[str] = []
    for source in matched_sources or []:
        info = doc_get(source)
        title = (info.get("canonical_title") or "").strip() or filename_stem(source)
        if title and title not in found_titles:
            found_titles.append(title)
    missing_text = "、".join(missing_items) if missing_items else "目标文件"
    if found_titles:
        return f"未找到需要对比的目标：{missing_text}。当前已识别：{'、'.join(found_titles)}。请补充准确文件名或重新选择对比对象。"
    return f"未找到需要对比的目标：{missing_text}。请补充准确文件名或先上传相关文件。"


def compare_matrix_presence_state(value: str) -> str:
    state = str(value or "").strip().upper()
    if state in {"PRESENT", "ABSENT_CONFIRMED", "UNKNOWN"}:
        return state
    if state in {"ANSWERABLE", "GUARDED_FULL", "COMPARABLE_PARTIAL"}:
        return "PRESENT"
    if state in {"NOT_FOUND", "MISSING", "ABSENT"}:
        return "ABSENT_CONFIRMED"
    return "UNKNOWN"


def compare_source_set_completeness(
    compare_plan: Dict[str, Any],
    active_fnames: List[str],
    normalize_filename,
) -> Dict[str, Any]:
    sources = [normalize_filename(item or "") for item in (active_fnames or [])]
    sources = [item for item in sources if item]
    required = list((compare_plan or {}).get("doc_like_subjects") or (compare_plan or {}).get("subjects") or [])
    explicit_missing = [
        str(item or "").strip()
        for item in (compare_plan or {}).get("missing_doc_targets", []) or (compare_plan or {}).get("missing_targets", [])
        if str(item or "").strip()
    ]
    expected_count = max(len(required), len(sources) + len(explicit_missing), 2 if sources else 0)
    complete = bool(sources) and len(sources) >= expected_count and not explicit_missing
    return {
        "complete": complete,
        "expected_target_count": expected_count,
        "resolved_source_count": len(sources),
        "sources": sources,
        "missing_targets": [] if complete else (explicit_missing or required[len(sources):expected_count]),
    }


def fallback_compare_refs_from_docs(
    docs: List[Any],
    *,
    normalize_filename,
    hit_entity_source,
    source_display_title,
    build_excerpt,
    hit_display_text,
    limit: int = 4,
) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    seen = set()
    for doc in docs or []:
        source = normalize_filename(hit_entity_source(doc) or "")
        if not source or source in seen:
            continue
        seen.add(source)
        refs.append(
            {
                "source": source,
                "title": source_display_title(source) or source,
                "excerpt": build_excerpt(hit_display_text(doc), "", 240),
            }
        )
        if len(refs) >= limit:
            break
    return refs
