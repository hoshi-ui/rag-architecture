import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.core.legal_subjects import normalize_subject_terms


CITATION_RANGE_SEPARATORS = r"\-–—~～至到"
CITATION_LIST_SEPARATORS = r",，、;；\s"
CITATION_BRACKET_RE = re.compile(
    rf"\[\s*\[?\s*([0-9][0-9{CITATION_LIST_SEPARATORS}{CITATION_RANGE_SEPARATORS}]*[0-9]?)\s*\]?\s*\]"
)
CITATION_RUN_RE = re.compile(r"(?:\[\d+\]\s*)+")
LEGAL_ARTICLE_NO_RE = re.compile(r"第[一二三四五六七八九十百千万零〇0-9]+条")
LEGAL_EXACT_CITATION_RE = re.compile(r"《[^》\n\r]+》\s*第[一二三四五六七八九十百千万零〇0-9]+条")
LEGAL_TITLE_MARKERS = ("条例", "规定", "办法", "规则", "规程", "细则", "法律", "法规", "决定")


class AnswerAdapter:
    def __init__(
        self,
        *,
        normalize_coverage_aspect: Callable[[str], str],
        normalize_query: Callable[[str], str],
        chunk_plain_display_text: Callable[[Any], str],
        hit_display_text: Callable[[Any], str],
        coverage_aspect_variants: Callable[[str], List[str]],
        build_sources: Callable[[List[Any], str, str], List[Dict[str, Any]]],
        answer_limits: Callable[[str], Dict[str, Any]],
        llm_temperature: float,
        llm_top_p: float,
        llm_max_tokens: int,
        llm_presence_penalty: float,
        llm_timeout: float,
        final_fact_verify_max_tokens: int,
        hit_metadata: Optional[Callable[[Any], Dict[str, Any]]] = None,
        hit_entity_source: Optional[Callable[[Any], str]] = None,
        get_chunks_for_source: Optional[Callable[[str, Optional[int]], List[Dict[str, Any]]]] = None,
        normalize_filename_for_match: Optional[Callable[[str], str]] = None,
    ):
        self.normalize_coverage_aspect = normalize_coverage_aspect
        self.normalize_query = normalize_query
        self.chunk_plain_display_text = chunk_plain_display_text
        self.hit_display_text = hit_display_text
        self.hit_metadata = hit_metadata or (lambda hit: {})
        self.hit_entity_source = hit_entity_source or (lambda hit: "")
        self.get_chunks_for_source = get_chunks_for_source or (lambda source, doc_version=None: [])
        self.normalize_filename_for_match = normalize_filename_for_match or (lambda source: str(source or ""))
        self.coverage_aspect_variants = coverage_aspect_variants
        self.build_sources = build_sources
        self.answer_limits = answer_limits
        self.llm_temperature = llm_temperature
        self.llm_top_p = llm_top_p
        self.llm_max_tokens = llm_max_tokens
        self.llm_presence_penalty = llm_presence_penalty
        self.llm_timeout = llm_timeout
        self.final_fact_verify_max_tokens = final_fact_verify_max_tokens


def _answer_context(runtime: Any) -> AnswerAdapter:
    if isinstance(runtime, AnswerAdapter):
        return runtime
    required = [
        "normalize_coverage_aspect",
        "normalize_query",
        "chunk_plain_display_text",
        "hit_display_text",
        "coverage_aspect_variants",
        "build_sources",
        "answer_limits",
    ]
    missing = [name for name in required if not hasattr(runtime, name)]
    if missing:
        raise AttributeError("Answer adapter missing: " + ", ".join(missing))
    return runtime


def extract_json_object_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", raw, flags=re.I)
    if fenced:
        return fenced.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start : end + 1].strip()
    return raw


def parse_structured_answer_payload(text: str) -> Optional[Dict[str, Any]]:
    raw = extract_json_object_text(text).lstrip("\ufeff").strip()
    if not raw:
        return None
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def normalize_structured_answer_refs(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value] if value > 0 else []
    if isinstance(value, str):
        refs = flatten_citation_refs(value)
    elif isinstance(value, (list, tuple, set)):
        refs = []
        for item in value:
            refs.extend(normalize_structured_answer_refs(item))
    else:
        refs = []
    deduped: List[int] = []
    for ref in refs:
        if ref > 0 and ref not in deduped:
            deduped.append(ref)
    return deduped


def flatten_citation_refs(value: Any) -> List[int]:
    text = str(value or "")
    refs: List[int] = []
    token_re = re.compile(rf"(\d+)\s*[{CITATION_RANGE_SEPARATORS}]\s*(\d+)|(\d+)")
    for match in token_re.finditer(text):
        if match.group(1) and match.group(2):
            start = int(match.group(1))
            end = int(match.group(2))
            if start <= end and end - start <= 50:
                candidates = range(start, end + 1)
            else:
                candidates = (start, end)
        else:
            candidates = (int(match.group(3)),)
        for ref in candidates:
            if ref > 0 and ref not in refs:
                refs.append(ref)
    return refs


def flatten_answer_citation_brackets(text: str) -> str:
    def replace_group(match: re.Match[str]) -> str:
        refs = flatten_citation_refs(match.group(1))
        return "".join(f"[{ref}]" for ref in refs)

    return CITATION_BRACKET_RE.sub(replace_group, text)


def dedupe_adjacent_citation_runs(text: str) -> str:
    def replace_run(match: re.Match[str]) -> str:
        refs = flatten_citation_refs(match.group(0))
        trailing = re.search(r"\s+$", match.group(0))
        return "".join(f"[{ref}]" for ref in refs) + (trailing.group(0) if trailing else "")

    return CITATION_RUN_RE.sub(replace_run, text)


def normalize_answer_citation_style(answer: str) -> str:
    text = str(answer or "")
    if not text:
        return ""
    text = re.sub(r"\[\s*(?:证据|依據|依据)\s*([0-9]+)\s*\]", r"[\1]", text)
    text = re.sub(r"(?<!\[)(?:证据|依據|依据)\s*([0-9]+)", r"[\1]", text)
    text = flatten_answer_citation_brackets(text)
    raw_lines = text.splitlines()
    cleaned_lines: List[str] = []
    for idx, line in enumerate(raw_lines):
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        next_nonempty = ""
        for next_line in raw_lines[idx + 1 :]:
            if next_line.strip():
                next_nonempty = next_line.strip()
                break
        if _is_answer_debug_leak_line(stripped):
            continue
        if stripped.endswith((":", "：")) and _is_answer_debug_leak_line(next_nonempty):
            continue
        if "参考证据" in stripped or stripped.startswith("证据说明") or stripped.startswith("(证据"):
            continue
        cleaned_lines.append(line.rstrip())
    normalized = "\n".join(cleaned_lines).strip()
    normalized = re.sub(r"(?ms)^#{1,6}\s*[^\n]+\n(?=(?:\s*\n)*(?:#{1,6}\s|$))", "", normalized).strip()
    return normalized


def _is_answer_debug_leak_line(line: str) -> bool:
    stripped = str(line or "").strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if lower in {"terms:", "terms：", "categories:", "categories：", "aspect_plan:", "aspect_plan："}:
        return True
    if lower.startswith(("terms:", "terms：", "categories:", "categories：", "structured aspect", "evidence mapping")):
        return True
    leak_markers = ("文档标题：", "文档标题:", "分块位置：", "分块位置:", "上文：", "上文:", "正文：", "正文:")
    if stripped.startswith(leak_markers):
        return True
    marker_hits = sum(1 for marker in leak_markers if marker in stripped)
    return marker_hits >= 2


def extract_answer_citation_refs(answer: str) -> List[int]:
    normalized = normalize_answer_citation_style(answer)
    return flatten_citation_refs(" ".join(re.findall(r"\[(\d+)\]", normalized)))


def should_use_structured_answer_schema(
    qtype: str,
    answer_mode: str,
    aspect_plan: str,
    enable_structured_answer_json: bool,
) -> bool:
    if not enable_structured_answer_json:
        return False
    if qtype in {"compare", "compare_degraded", "fallback_brief"}:
        return False
    if answer_mode in {"rag_related_doc", "compare_asymmetric", "compare_degraded"}:
        return False
    return bool(str(aspect_plan or "").strip())


LEGAL_CLAUSE_RE = re.compile(r"第[一二三四五六七八九十百千万零〇两\d]+条")
LEGAL_ITEM_RE = re.compile(r"[（(][一二三四五六七八九十百千万零〇两\d]+[）)]")
LEGAL_DOC_MARKERS = ("条例", "办法", "规定", "规则", "规程", "细则", "法律", "法规", "法条", "条款")
LEGAL_ENUM_INTENT_MARKERS = (
    "哪些",
    "有哪些",
    "有什么",
    "列出",
    "枚举",
    "限制",
    "要求",
    "禁止",
    "不得",
    "应当",
    "规范",
    "义务",
    "责任",
)
def is_legal_clause_enumeration_query(query: str, evidence: str = "", answer_mode: str = "") -> bool:
    q = str(query or "")
    e = str(evidence or "")
    if not q:
        return False
    has_legal_target = any(token in q for token in LEGAL_DOC_MARKERS)
    has_enum_intent = any(token in q for token in LEGAL_ENUM_INTENT_MARKERS)
    has_clause_evidence = bool(LEGAL_CLAUSE_RE.search(e) or LEGAL_ITEM_RE.search(e))
    return has_legal_target and has_enum_intent and has_clause_evidence


def is_legal_clause_enumeration_intent(query: str, answer_mode: str = "") -> bool:
    q = str(query or "")
    if not q:
        return False
    has_legal_target = any(token in q for token in LEGAL_DOC_MARKERS)
    has_enum_intent = any(token in q for token in LEGAL_ENUM_INTENT_MARKERS)
    return has_legal_target and has_enum_intent


def legal_clause_answer_rules(query: str, evidence: str, answer_mode: str = "") -> str:
    if not is_legal_clause_enumeration_query(query, evidence, answer_mode):
        return ""
    return (
        "法条枚举模式：用户询问的是法规条文中的具体限制、要求或禁止事项。"
        "回答必须按 context 中的条文/条项列举，保持与原文相同的法律义务粒度。"
        "禁止把条文改写成政策解读、背景分析、治理目的或抽象主题总结；"
        "禁止添加 context 未出现的适用条件、例外、处罚后果或价值判断。"
        "不要使用 Markdown 小标题、加粗标题或“主要包括以下方面”这类概括性结构。"
        "优先输出形如“- 第二十二条：第（一）项……；第（二）项……[1]”的列表；"
        "同一条文下多个（一）（二）（三）项应合并到同一条展示，但不得改写为宽泛要点；"
        "必须先区分条款约束对象，只抽取与用户问题主体一致的条款；例如问养犬行为时，"
        "应排除犬只经营、交易诊疗、政府部门职责、处罚责任等不同主体条款。"
        "保留条文原句的动宾结构，不要为了统一句式给每个子项强行补同一个谓语。"
        "如果必须概括，概括范围不得超过该条/该项原文本身。\n"
    )


def answer_limits(config: Any, qtype: str) -> Dict[str, Any]:
    if qtype == "definition":
        points = int(getattr(config, "ANSWER_POINTS_DEFINITION", 3) or 3)
        max_tokens = int(getattr(config, "LLM_MAX_TOKENS_DEF", getattr(config, "LLM_MAX_TOKENS", 1200)) or 1200)
    elif qtype == "summary":
        points = int(getattr(config, "ANSWER_POINTS_SUMMARY", 5) or 5)
        max_tokens = int(getattr(config, "LLM_MAX_TOKENS_SUMMARY", getattr(config, "LLM_MAX_TOKENS", 1200)) or 1200)
    elif qtype == "howto":
        points = int(getattr(config, "ANSWER_POINTS_HOWTO", 6) or 6)
        max_tokens = int(getattr(config, "LLM_MAX_TOKENS_HOWTO", getattr(config, "LLM_MAX_TOKENS", 1200)) or 1200)
    elif qtype in {"compare", "compare_degraded"}:
        points = int(getattr(config, "ANSWER_POINTS_COMPARE", 6) or 6)
        max_tokens = int(getattr(config, "LLM_MAX_TOKENS_COMPARE", getattr(config, "LLM_MAX_TOKENS", 1200)) or 1200)
    elif qtype == "architecture":
        points = int(getattr(config, "ANSWER_POINTS_ARCH", 5) or 5)
        max_tokens = int(getattr(config, "LLM_MAX_TOKENS_ARCH", getattr(config, "LLM_MAX_TOKENS", 1200)) or 1200)
    else:
        points = int(getattr(config, "ANSWER_POINTS_DEFAULT", 5) or 5)
        max_tokens = int(getattr(config, "LLM_MAX_TOKENS_OTHER", getattr(config, "LLM_MAX_TOKENS", 1200)) or 1200)
    return {"points": points, "max_tokens": max_tokens}


def answer_mode_for_sources(
    target_sources: List[str],
    selected_docs: List[Any],
    *,
    normalize_filename: Callable[[str], str],
    hit_entity_source: Callable[[Any], str],
    sources_equivalent: Callable[[str, str], bool],
) -> str:
    targets = [normalize_filename(item) for item in (target_sources or []) if normalize_filename(item)]
    if not targets:
        return "target_hit" if selected_docs else "llm_fallback"
    doc_sources = [hit_entity_source(doc) for doc in (selected_docs or [])]
    for target in targets:
        if any(sources_equivalent(source, target) for source in doc_sources):
            return "target_hit"
    return "rag_related_doc"


def build_related_doc_grounded_answer(
    selected_docs: List[Any],
    *,
    hit_display_text: Callable[[Any], str],
    hit_metadata: Callable[[Any], Dict[str, Any]],
) -> str:
    if not selected_docs:
        return "当前证据不足，暂不能基于已检索资料回答。"
    lines = ["当前未能完全确认目标文档，以下基于最相关证据回答："]
    for index, doc in enumerate(selected_docs[:3], start=1):
        metadata = hit_metadata(doc) or {}
        section = str(metadata.get("section") or metadata.get("section_title") or "").strip()
        text = re.sub(r"\s+", " ", str(hit_display_text(doc) or "")).strip()
        if not text:
            continue
        prefix = f"{section}：" if section else ""
        lines.append(f"- {prefix}{text[:220]}[{index}]")
    return "\n".join(lines) if len(lines) > 1 else "当前证据不足，暂不能基于已检索资料回答。"


def build_answer_aspect_plan(
    query: str,
    docs: List[Any],
    *,
    qfilters: Optional[Dict[str, Any]] = None,
    covered_aspects: Optional[List[str]] = None,
    uncovered_aspects: Optional[List[str]] = None,
    normalize_coverage_aspect: Callable[[str], str],
    normalize_query: Callable[[str], str],
    query_semantic_aspects: Callable[[str], List[str]],
    doc_matches_semantic_aspect: Callable[[Any, str], bool],
    aspect_doc_priority_score: Callable[[Any, str], float],
    doc_section_name: Callable[[Any], str],
) -> str:
    aspects: List[str] = []
    for item in covered_aspects or []:
        value = normalize_coverage_aspect(item)
        if _is_public_answer_aspect(value) and value not in aspects:
            aspects.append(value)
    semantic = query_semantic_aspects(query) or {}
    if isinstance(semantic, dict):
        semantic_terms = list(semantic.get("terms") or [])
    else:
        semantic_terms = list(semantic or [])
    for item in semantic_terms:
        value = normalize_coverage_aspect(item)
        if _is_public_answer_aspect(value) and value not in aspects:
            aspects.append(value)
    for item in uncovered_aspects or []:
        value = normalize_coverage_aspect(item)
        if _is_public_answer_aspect(value) and value not in aspects:
            aspects.append(value)
    if not aspects:
        terms = (qfilters or {}).get("terms") or []
        aspects = [str(item).strip() for item in terms[:5] if _is_public_answer_aspect(str(item).strip())]
    lines: List[str] = []
    for aspect in aspects[:8]:
        refs = [
            str(index)
            for index, doc in enumerate(docs or [], start=1)
            if doc_matches_semantic_aspect(doc, aspect)
        ][:4]
        lines.append(f"- aspect: {aspect}; refs: {', '.join(refs) if refs else ''}")
    return "\n".join(lines)


def _is_public_answer_aspect(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if len(text) > 40:
        return False
    lowered = text.lower()
    if lowered in {"terms", "categories", "debug", "structured aspect", "evidence mapping"}:
        return False
    if _is_answer_debug_leak_line(text):
        return False
    if any(marker in text for marker in ("文档标题", "分块位置", "上文", "正文", "metadata", "chunk_range")):
        return False
    return True


def parse_answer_aspect_plan(aspect_plan: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for line in str(aspect_plan or "").splitlines():
        line = line.strip().lstrip("-").strip()
        if not line:
            continue
        name = ""
        refs: List[int] = []
        aspect_match = re.search(r"aspect\s*[:：]\s*([^;；]+)", line, flags=re.I)
        if aspect_match:
            name = aspect_match.group(1).strip()
        else:
            name = line.split(";", 1)[0].strip()
        refs_match = re.search(r"refs?\s*[:：]\s*([0-9,\s，]+)", line, flags=re.I)
        if refs_match:
            refs = normalize_structured_answer_refs(refs_match.group(1))
        if name:
            entries.append({"aspect": name, "refs": refs})
    return entries


def build_structured_answer_prompt(
    query: str,
    evidence: str,
    qtype: str,
    answer_mode: str,
    uncovered_aspects: Optional[List[str]],
    aspect_plan: str,
    evidence_gate_warning: str = "",
) -> str:
    legal_rules = legal_clause_answer_rules(query, evidence, answer_mode)
    citation_rules = (
        "引用纪律：context 编号已按相关性降序排列，[1] 是最高分证据。"
        "citations 只能填写 context 中存在的编号；每个 item 使用最直接、最小充分的证据编号。"
        "不同事实来自不同证据时必须分别填写 citations，禁止用宽泛编号覆盖未被该证据支持的结论。\n"
    )
    return citation_rules + legal_rules + (
        "请严格基于<context>中的证据生成 JSON。\n"
        "如果处于法条枚举模式，JSON 中 aspects 只使用“条文列举”这一个名称；"
        "item.keyword 填写具体条文或条项，例如“第二十二条第（一）项”；"
        "item.content 保持接近原文，不要写成主题解读。\n"
        "重要引用规则：禁止输出“[证据 n]”或“证据 n”，只能在 item.citations 中写数字数组，例如 [1, 2]；"
        "不得引用 context 中不存在的编号。\n"
        "JSON 结构：{\"aspects\":[{\"name\":\"方面\",\"items\":[{\"keyword\":\"关键词\",\"content\":\"结论\",\"citations\":[1]}]}],"
        "\"uncovered_aspects\":[]}。\n"
        f"问题类型：{qtype}\n回答模式：{answer_mode}\n问题：{query}\n"
        f"方面计划：\n{aspect_plan or '(无)'}\n"
        f"未覆盖方面：{', '.join(uncovered_aspects or []) or '(无)'}\n"
        f"证据门控提示：{evidence_gate_warning or '(无)'}\n"
        f"<context>\n{evidence}\n</context>"
    )


def build_answer_prompt(
    query: str,
    evidence: str,
    qtype: str,
    answer_limits: Dict[str, Any],
    answer_mode: str,
    compare_missing_targets: Optional[List[str]],
    compare_source_status_hints: str,
    uncovered_aspects: Optional[List[str]],
    aspect_plan: str,
    evidence_gate_warning: str = "",
) -> str:
    points = int((answer_limits or {}).get("points") or 5)
    legal_rules = legal_clause_answer_rules(query, evidence, answer_mode)
    compare_rules = ""
    if qtype in {"compare", "compare_degraded"}:
        compare_rules = (
            "对比题回答要求：不要只罗列各文档条款。必须先用 1-2 句概括核心不同点，"
            "再按处罚主体、处罚对象/行为、处罚种类、罚款幅度、特别后果等维度归纳差异。"
            "每个差异点都要同时说明两边如何不同；如果某一边证据没有覆盖，要明确写“证据未覆盖”，不要编造。\n"
        )
    citation_rules = (
        "引用纪律：context 编号已按相关性降序排列，[1] 是最高分证据。"
        "每个事实结论必须引用最直接、最小充分的证据编号；不同事实来自不同证据时必须分别引用。"
        "禁止编造 context 中不存在的编号，禁止用宽泛编号覆盖未被该证据支持的结论。"
        "引用格式只能是连续的正文编号，例如 [1] 或 [1][2]，不要写 [1,2]、[1-3]、[[1]]。\n"
    )
    return (
        "请只依据<context>中的内容回答。每个事实性结论后必须使用 [1]、[2] 这种正文编号引用；"
        "禁止输出 [证据 n]、证据 n、来源 n 等格式。没有证据时请拒答。\n"
        f"{legal_rules}"
        f"{compare_rules}"
        f"建议要点数：{points}\n问题类型：{qtype}\n回答模式：{answer_mode}\n"
        f"缺失对比目标：{', '.join(compare_missing_targets or []) or '(无)'}\n"
        f"对比状态提示：{compare_source_status_hints or '(无)'}\n"
        f"方面计划：\n{aspect_plan or '(无)'}\n"
        f"未覆盖方面：{', '.join(uncovered_aspects or []) or '(无)'}\n"
        f"证据门控提示：{evidence_gate_warning or '(无)'}\n"
        f"问题：{query}\n<context>\n{evidence}\n</context>"
    )


def build_answer_verification_prompt(query: str, evidence: str, draft: str, aspect_plan: str) -> str:
    legal_rules = legal_clause_answer_rules(query, evidence)
    return (
        "请核查并改写 draft，使所有事实结论都严格来自 context。"
        "引用格式只能是 [1]、[2]；禁止输出 [证据 n]。"
        "如果 context 不支持某结论，请删除或改为证据不足。\n"
        f"{legal_rules}"
        f"问题：{query}\n方面计划：\n{aspect_plan or '(无)'}\n"
        f"<context>\n{evidence}\n</context>\n<draft>\n{draft}\n</draft>"
    )


def clean_answer_heading_text(text: Any) -> str:
    return re.sub(r"^\*+|\*+$", "", str(text or "").strip()).strip(" :：")


def _safe_re_match(pattern: str, text: str) -> Optional[re.Match[str]]:
    try:
        return re.match(pattern, text or "")
    except re.error:
        return None


def parse_answer_aspect_heading(line: str) -> Tuple[str, str]:
    stripped = str(line or "").strip()
    match = _safe_re_match(r"^#{1,6}\s*(.+?)\s*$", stripped) or _safe_re_match(r"^\*\*(.+?)\*\*\s*$", stripped)
    if not match:
        return "", ""
    aspect = clean_answer_heading_text(match.group(1))
    return aspect, "markdown"


def is_answer_uncovered_heading(line: str) -> bool:
    value = clean_answer_heading_text(line)
    return any(token in value for token in ("未覆盖", "证据不足", "无法确认"))


def detect_answer_aspect_heading_style(answer: str) -> str:
    for line in str(answer or "").splitlines():
        if line.lstrip().startswith("#"):
            return "markdown"
    return "label"


def format_answer_aspect_heading(aspect: str, style: str) -> str:
    return f"### {clean_answer_heading_text(aspect)}" if style == "markdown" else f"{clean_answer_heading_text(aspect)}："


def answer_aspect_match_key(runtime: Any, text: str) -> str:
    context = _answer_context(runtime)
    return context.normalize_query(clean_answer_heading_text(text))


def strip_fallback_legal_prefix(text: str) -> str:
    return re.sub(r"^(根据|依据)?相关(法规|证据|材料)[，,:：\s]*", "", str(text or "").strip())


def aspect_group_has_allowed_citations(lines: List[str], refs: List[int]) -> bool:
    if not refs:
        return False
    allowed = set(refs)
    for ref in extract_answer_citation_refs("\n".join(lines)):
        if ref in allowed:
            return True
    return False


def aspect_fallback_snippet(runtime: Any, doc: Any, limit: int) -> str:
    text = _answer_context(runtime).hit_display_text(doc)
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[: max(40, int(limit or 220))]


def build_missing_aspect_group(runtime: Any, entry: Dict[str, Any], docs: List[Any], heading_style: str) -> List[str]:
    refs = entry.get("refs") or list(range(1, min(len(docs or []), 3) + 1))
    if not refs:
        return []
    ref = refs[0]
    if ref < 1 or ref > len(docs or []):
        return []
    aspect = str(entry.get("aspect") or "相关依据")
    snippet = aspect_fallback_snippet(runtime, docs[ref - 1], 220)
    if not snippet:
        return []
    return [format_answer_aspect_heading(aspect, heading_style), f"- {snippet}[{ref}]"]


def clean_structured_answer_keyword(text: Any) -> str:
    return clean_answer_heading_text(text)[:24]


def clean_structured_answer_text(text: Any) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    value = re.sub(r"\[(?:证据\s*)?(\d+)\]", "", value).strip()
    return value


def is_valid_structured_uncovered_aspect(runtime: Any, text: str) -> bool:
    value = clean_answer_heading_text(text)
    if not value or value.lower() in {"none", "null", "n/a"}:
        return False
    return not bool(re.fullmatch(r"\d+", value))


def structured_aspect_match_score(runtime: Any, left: str, right: str) -> int:
    lkey = answer_aspect_match_key(runtime, left)
    rkey = answer_aspect_match_key(runtime, right)
    if not lkey or not rkey:
        return 0
    if lkey == rkey:
        return 100
    if lkey in rkey or rkey in lkey:
        return 80
    return 0


def match_structured_aspect_entry(runtime: Any, name: str, plan_entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_score = 0
    for entry in plan_entries or []:
        score = structured_aspect_match_score(runtime, name, str(entry.get("aspect") or ""))
        if score > best_score:
            best = entry
            best_score = score
    return best if best_score >= 35 else None


def dedupe_structured_answer_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output: List[Dict[str, Any]] = []
    for item in items or []:
        key = (str(item.get("keyword") or ""), str(item.get("content") or ""))
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def build_missing_structured_aspect(runtime: Any, entry: Dict[str, Any], docs: List[Any]) -> Optional[Dict[str, Any]]:
    refs = entry.get("refs") or list(range(1, min(len(docs or []), 3) + 1))
    if not refs:
        return None
    ref = refs[0]
    if ref < 1 or ref > len(docs or []):
        return None
    content = aspect_fallback_snippet(runtime, docs[ref - 1], 220)
    if not content:
        return None
    return {"name": str(entry.get("aspect") or "相关依据"), "items": [{"keyword": "", "content": content, "citations": [ref]}]}


def normalize_structured_answer_payload(
    runtime: Any,
    payload: Dict[str, Any],
    aspect_plan: str,
    docs: List[Any],
) -> Dict[str, Any]:
    context = _answer_context(runtime)
    plan_entries = parse_answer_aspect_plan(aspect_plan)
    valid_refs = set(range(1, len(docs or []) + 1))
    grouped: Dict[str, Dict[str, Any]] = {}
    for raw_aspect in payload.get("aspects") or []:
        if not isinstance(raw_aspect, dict):
            continue
        raw_name = clean_answer_heading_text(raw_aspect.get("name") or raw_aspect.get("aspect") or "")
        if not raw_name:
            continue
        plan_entry = match_structured_aspect_entry(context, raw_name, plan_entries) if plan_entries else None
        canonical_name = str((plan_entry or {}).get("aspect") or raw_name).strip()
        aspect_key = answer_aspect_match_key(context, canonical_name)
        if not aspect_key:
            continue
        items: List[Dict[str, Any]] = []
        for raw_item in raw_aspect.get("items") or []:
            if not isinstance(raw_item, dict):
                continue
            refs = normalize_structured_answer_refs(
                raw_item.get("citations", raw_item.get("citation", raw_item.get("refs", raw_item.get("ref"))))
            )
            refs = [ref for ref in refs if ref in valid_refs]
            keyword = clean_structured_answer_keyword(raw_item.get("keyword") or raw_item.get("label") or "")
            content = clean_structured_answer_text(raw_item.get("content") or raw_item.get("text") or "")
            if not content or not refs:
                continue
            items.append({"keyword": keyword, "content": content, "citations": refs})
        if not items:
            continue
        bucket = grouped.setdefault(aspect_key, {"name": canonical_name, "items": []})
        bucket["items"].extend(items)

    normalized_aspects: List[Dict[str, Any]] = []
    if plan_entries:
        for entry in plan_entries:
            aspect_name = str(entry.get("aspect") or "").strip()
            aspect_key = answer_aspect_match_key(context, aspect_name)
            bucket = grouped.get(aspect_key)
            if bucket and bucket.get("items"):
                normalized_aspects.append({"name": aspect_name, "items": dedupe_structured_answer_items(bucket["items"])})
            else:
                fallback = build_missing_structured_aspect(context, entry, docs)
                if fallback:
                    normalized_aspects.append(fallback)
    else:
        for bucket in grouped.values():
            normalized_aspects.append({"name": bucket.get("name") or "", "items": dedupe_structured_answer_items(bucket.get("items") or [])})

    uncovered: List[str] = []
    for item in payload.get("uncovered_aspects") or []:
        value = clean_answer_heading_text(item)
        if is_valid_structured_uncovered_aspect(context, value) and value not in uncovered:
            uncovered.append(value)
    return {
        "aspects": [aspect for aspect in normalized_aspects if aspect.get("name") and aspect.get("items")],
        "uncovered_aspects": uncovered,
    }


def render_structured_answer_markdown(runtime: Any, payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    for aspect in payload.get("aspects") or []:
        name = clean_answer_heading_text((aspect or {}).get("name") or "相关依据")
        if name:
            lines.append(f"### {name}")
        for item in (aspect or {}).get("items") or []:
            content = clean_structured_answer_text((item or {}).get("content") or "")
            refs = normalize_structured_answer_refs((item or {}).get("citations") or (item or {}).get("refs"))
            if not content or not refs:
                continue
            ref_text = "".join(f"[{ref}]" for ref in refs)
            keyword = clean_structured_answer_keyword((item or {}).get("keyword") or "")
            prefix = f"{keyword}：" if keyword else ""
            lines.append(f"- {prefix}{content}{ref_text}")
        lines.append("")
    uncovered = [clean_answer_heading_text(item) for item in payload.get("uncovered_aspects") or [] if clean_answer_heading_text(item)]
    if uncovered:
        lines.append("### 证据不足")
        for item in uncovered:
            lines.append(f"- 当前证据不足，无法确认“{item}”。")
    return normalize_answer_citation_style("\n".join(lines))


def structured_answer_from_markdown(
    runtime: Any,
    answer: str,
    aspect_plan: str,
    docs: List[Any],
    uncovered_aspects: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    aspects: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line in normalize_answer_citation_style(answer).splitlines():
        stripped = line.strip()
        heading, _ = parse_answer_aspect_heading(stripped)
        if heading:
            current = {"name": heading, "items": []}
            aspects.append(current)
            continue
        if not stripped or not current:
            continue
        refs = extract_answer_citation_refs(stripped)
        content = re.sub(r"\[\d+\]", "", stripped.lstrip("-* ").strip()).strip()
        if content and refs:
            current["items"].append({"keyword": "", "content": content, "citations": refs})
    payload = {"aspects": aspects, "uncovered_aspects": uncovered_aspects or []}
    normalized = normalize_structured_answer_payload(runtime, payload, aspect_plan, docs)
    return normalized if normalized.get("aspects") else None


def answer_claim_lines(answer: str) -> List[str]:
    lines: List[str] = []
    for line in normalize_answer_citation_style(answer).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or is_answer_uncovered_heading(stripped):
            continue
        stripped = re.sub(r"^[*\-]\s*", "", stripped)
        if stripped:
            lines.append(stripped)
    return lines


def answer_uncited_claim_lines(answer: str) -> List[str]:
    return [line for line in answer_claim_lines(answer) if not re.search(r"\[\d+\]", line)]


def answer_looks_truncated(answer: str) -> bool:
    text = str(answer or "").rstrip()
    if not text:
        return False
    if re.search(r"(\[|［|（|【|、|，|,|:|：|；|;)$", text):
        return True
    if re.search(r"\[\d*$", text):
        return True
    if re.search(r"(可处|处以|包括|如下|以下|分别为|例如|其中)$", text):
        return True
    return False


def build_cited_sources(
    runtime: Any,
    answer: str,
    docs: List[Any],
    query: str,
    score_mode: str,
    fallback_docs: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    context = _answer_context(runtime)
    all_sources = context.build_sources(docs, query, score_mode)
    cited_refs = extract_answer_citation_refs(answer)
    if not cited_refs:
        fallback_items = list(fallback_docs or [])[:3]
        return context.build_sources(fallback_items, query, score_mode) if fallback_items else []
    source_by_ref: Dict[int, Dict[str, Any]] = {}
    for item in all_sources:
        try:
            ref = int(item.get("ref") or 0)
        except Exception:
            ref = 0
        if ref > 0:
            source_by_ref[ref] = item
    selected: List[Dict[str, Any]] = []
    for ref in cited_refs:
        item = source_by_ref.get(ref)
        if item:
            selected.append(item)
    return selected


def citation_source_identity(source: Dict[str, Any]) -> Tuple[str, str, str]:
    source_name = str(source.get("source") or "").strip()
    section = str(source.get("section") or "").strip()
    chunk_range = str(source.get("chunk_range") or "").strip()
    if not chunk_range:
        chunk_range = str(source.get("text") or "")[:160]
    return source_name, section, chunk_range


def _clean_legal_citation_title(raw: Any) -> str:
    title = str(raw or "").strip().strip("《》")
    if not title:
        return ""
    title = title.replace("\\", "/").split("/")[-1].strip()
    title = re.sub(r"\.(?:docx?|pdf|txt|md|html?)$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"[_\-\s]*\d{4}(?:[-_年\s]+)\d{1,2}(?:[-_月\s]+)\d{1,2}日?", "", title)
    title = re.sub(r"[_\-\s]*(?:现行有效|最新版本|有效|版本|v\d+)$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"[_\-\s]+$", "", title).strip()
    if not title or "\n" in title or len(title) > 80:
        return ""
    return title


def _looks_like_legal_citation_title(title: str) -> bool:
    compact = re.sub(r"\s+", "", str(title or ""))
    if not compact:
        return False
    if re.match(r"^第[一二三四五六七八九十百千万零〇0-9]+章", compact):
        return False
    if compact in {"总则", "附则", "第一章总则", "法律责任", "监督检查", "保护与管理", "职责", "范围"}:
        return False
    return any(marker in compact for marker in LEGAL_TITLE_MARKERS)


def _source_citation_title(source: Dict[str, Any]) -> str:
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    clause_meta = metadata.get("clause_metadata") if isinstance(metadata.get("clause_metadata"), dict) else {}
    source_title = _clean_legal_citation_title(metadata.get("source_file") or source.get("source"))
    source_title = source_title if _looks_like_legal_citation_title(source_title) else ""
    candidates = [
        source.get("doc_title"),
        metadata.get("doc_title"),
        clause_meta.get("doc_title"),
        metadata.get("canonical_title"),
    ]
    for candidate in candidates:
        title = _clean_legal_citation_title(candidate)
        if not _looks_like_legal_citation_title(title):
            continue
        if source_title and title in source_title and len(source_title) > len(title):
            return source_title
        if title:
            return title
    return source_title


def _normalize_legal_article_no(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    match = LEGAL_ARTICLE_NO_RE.search(text)
    if match:
        return match.group(0)
    digit_match = re.fullmatch(r"\d{1,4}", text)
    if digit_match:
        return f"第{text}条"
    return ""


def _source_citation_article_no(source: Dict[str, Any]) -> str:
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    clause_meta = metadata.get("clause_metadata") if isinstance(metadata.get("clause_metadata"), dict) else {}
    candidates = [
        source.get("article_no"),
        source.get("article_id"),
        source.get("clause_id"),
        source.get("clause"),
        metadata.get("article_no"),
        metadata.get("article_id"),
        metadata.get("clause_id"),
        metadata.get("clause"),
        clause_meta.get("article_no"),
        clause_meta.get("article_id"),
        clause_meta.get("clause_id"),
    ]
    for candidate in candidates:
        article_no = _normalize_legal_article_no(candidate)
        if article_no:
            return article_no
    return ""


def _source_exact_citation_label(source: Dict[str, Any]) -> str:
    title = _source_citation_title(source)
    article_no = _source_citation_article_no(source)
    if not title or not article_no:
        return ""
    return f"《{title}》{article_no}"


def _inject_exact_legal_citations(answer: str, sources_by_ref: Dict[int, Dict[str, Any]]) -> str:
    if not answer or not sources_by_ref:
        return answer

    labels_by_ref = {
        ref: label
        for ref, source in sources_by_ref.items()
        for label in [_source_exact_citation_label(source)]
        if label
    }
    if not labels_by_ref:
        return answer

    citation_ref_re = re.compile(
        r"(?:《[^》\n\r]+》\s*第[一二三四五六七八九十百千万零〇0-9]+条)?\s*\[(\d+)\]"
    )

    def replace_ref(match: re.Match[str]) -> str:
        ref = int(match.group(1))
        label = labels_by_ref.get(ref)
        if not label:
            return match.group(0)
        if match.group(0).strip().startswith(label):
            return match.group(0)
        prefix = match.string[max(0, match.start() - len(label) - 8): match.start()]
        if re.search(rf"{re.escape(label)}\s*$", prefix):
            return match.group(0)
        if LEGAL_EXACT_CITATION_RE.search(prefix[-80:]) and prefix.rstrip().endswith(label):
            return match.group(0)
        return f"{label}[{ref}]"

    return citation_ref_re.sub(replace_ref, answer)


def rewrite_answer_citation_protocol(
    answer: str,
    sources: List[Dict[str, Any]],
    structured_answer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_answer = normalize_answer_citation_style(answer)
    source_by_old_ref: Dict[int, Dict[str, Any]] = {}
    for index, source in enumerate(sources or [], start=1):
        try:
            old_ref = int(source.get("ref") or index)
        except Exception:
            old_ref = index
        if old_ref > 0 and old_ref not in source_by_old_ref:
            source_by_old_ref[old_ref] = source

    old_refs = extract_answer_citation_refs(normalized_answer)
    if not old_refs:
        old_refs = list(source_by_old_ref.keys())[:3]
    phantom_refs = [ref for ref in old_refs if ref not in source_by_old_ref]
    valid_old_refs = [ref for ref in old_refs if ref in source_by_old_ref]

    ref_map: Dict[int, int] = {}
    identity_to_new_ref: Dict[Tuple[str, str, str], int] = {}
    aligned_sources: List[Dict[str, Any]] = []
    aligned_by_new_ref: Dict[int, Dict[str, Any]] = {}
    for old_ref in valid_old_refs:
        source = source_by_old_ref[old_ref]
        identity = citation_source_identity(source)
        new_ref = identity_to_new_ref.get(identity)
        if not new_ref:
            new_ref = len(identity_to_new_ref) + 1
            identity_to_new_ref[identity] = new_ref
            new_source = dict(source)
            new_source["ref"] = new_ref
            new_source["source_ref"] = new_ref
            new_source["original_ref"] = old_ref
            new_source["original_refs"] = [old_ref]
            aligned_sources.append(new_source)
            aligned_by_new_ref[new_ref] = new_source
        else:
            existing = aligned_by_new_ref.get(new_ref)
            if existing is not None:
                original_refs = list(existing.get("original_refs") or [existing.get("original_ref")])
                if old_ref not in original_refs:
                    original_refs.append(old_ref)
                existing["original_refs"] = [ref for ref in original_refs if ref]
        ref_map[old_ref] = new_ref

    def replace_ref(match: re.Match[str]) -> str:
        old_ref = int(match.group(1))
        new_ref = ref_map.get(old_ref)
        return f"[{new_ref}]" if new_ref else ""

    rewritten_answer = re.sub(r"\[(\d+)\]", replace_ref, normalized_answer)
    rewritten_answer = _inject_exact_legal_citations(rewritten_answer, aligned_by_new_ref)
    rewritten_answer = dedupe_adjacent_citation_runs(rewritten_answer)
    rewritten_answer = re.sub(r"[ \t]{2,}", " ", rewritten_answer)
    rewritten_answer = re.sub(r" +\n", "\n", rewritten_answer).strip()

    structured_refs: List[Dict[str, Any]] = []
    if structured_answer:
        for aspect in structured_answer.get("aspects") or []:
            aspect_name = clean_answer_heading_text((aspect or {}).get("name") or "")
            for item in (aspect or {}).get("items") or []:
                claim = clean_structured_answer_text((item or {}).get("content") or "")
                old_item_refs = normalize_structured_answer_refs((item or {}).get("citations") or (item or {}).get("refs"))
                new_item_refs = list(dict.fromkeys(ref_map[ref] for ref in old_item_refs if ref in ref_map))
                if claim and new_item_refs:
                    structured_refs.append({"aspect": aspect_name, "claim": claim, "refs": new_item_refs})

    answer_refs = extract_answer_citation_refs(rewritten_answer)
    return {
        "answer": rewritten_answer,
        "sources": aligned_sources,
        "answer_refs": answer_refs,
        "structured_refs": structured_refs,
        "citation_ref_map": ref_map,
        "phantom_citation_refs": phantom_refs,
    }


def ensure_answer_aspect_coverage(runtime: Any, answer: str, aspect_plan: str, docs: List[Any]) -> str:
    normalized = normalize_answer_citation_style(answer)
    if not aspect_plan or not docs:
        return normalized
    existing = normalize_query_safe(_answer_context(runtime), normalized)
    heading_style = detect_answer_aspect_heading_style(normalized)
    additions: List[str] = []
    for entry in parse_answer_aspect_plan(aspect_plan):
        aspect = str(entry.get("aspect") or "")
        if aspect and _answer_context(runtime).normalize_query(aspect) not in existing:
            additions.extend(build_missing_aspect_group(runtime, entry, docs, heading_style))
    if additions:
        normalized = normalized.rstrip() + "\n\n" + "\n".join(additions)
    return normalize_answer_citation_style(normalized)


def normalize_query_safe(context: AnswerAdapter, text: str) -> str:
    try:
        return context.normalize_query(text)
    except Exception:
        return str(text or "")


def _log_llm_timing(
    runtime: AnswerAdapter,
    llm_client: Any,
    logger_obj: Any,
    stage: str,
    payload: Dict[str, Any],
    elapsed_sec: float,
    *,
    evidence: str = "",
    output: str = "",
    error: str = "",
) -> None:
    if not bool(getattr(llm_client.config, "ENABLE_LLM_TIMING_LOG", True)):
        return
    messages = payload.get("messages") or []
    prompt_tokens = 0
    try:
        prompt_tokens = int(llm_client.estimate_message_tokens(messages))
    except Exception:
        prompt_tokens = sum(len(str((message or {}).get("content") or "")) for message in messages if isinstance(message, dict))
    logger_obj.info(
        "llm_call stage=%s elapsed_ms=%.1f prompt_tokens_est=%s max_tokens=%s evidence_chars=%s output_chars=%s error=%s",
        stage,
        elapsed_sec * 1000,
        prompt_tokens,
        payload.get("max_tokens"),
        len(str(evidence or "")),
        len(str(output or "")),
        error,
    )


async def generate_answer(
    runtime: Any,
    llm_client: Any,
    logger_obj: Any,
    query: str,
    context: str,
    qtype: str = "other",
    max_tokens: Optional[int] = None,
    answer_mode: str = "target_hit",
    compare_missing_targets: Optional[List[str]] = None,
    compare_source_status_hints: str = "",
    uncovered_aspects: Optional[List[str]] = None,
    aspect_plan: str = "",
    evidence_gate_warning: str = "",
) -> str:
    context_adapter = _answer_context(runtime)
    prompt = build_answer_prompt(
        query=query,
        evidence=context,
        qtype=qtype,
        answer_limits=context_adapter.answer_limits(qtype),
        answer_mode=answer_mode,
        compare_missing_targets=compare_missing_targets,
        compare_source_status_hints=compare_source_status_hints,
        uncovered_aspects=uncovered_aspects,
        aspect_plan=aspect_plan,
        evidence_gate_warning=evidence_gate_warning,
    )
    system_prompt = (
        "你是法规问答助手。只能根据 context 回答，所有事实结论必须带 [1]、[2] 这类编号。"
        "严禁输出 [证据 n] 或“证据 n”。"
    )
    system_prompt += (
        "引用纪律：context 证据编号已按相关性降序排列，[1] 是最高分证据。"
        "每个事实结论必须使用最直接、最小充分的正文编号引用，例如 [1] 或 [1][2]。"
        "不同事实来自不同证据时必须分别引用，禁止用宽泛编号覆盖未被该证据支持的结论。"
        "严禁编造 context 中不存在的编号；严禁输出 [证据 n]、证据n、来源n、[1,2]、[1-3]、[[1]]。"
        "如果 context 不支持结论，必须说明证据不足。"
    )
    system_prompt += legal_clause_answer_rules(query, context, answer_mode)
    payload = llm_client.build_payload(
        system_prompt,
        prompt,
        temperature=context_adapter.llm_temperature,
        top_p=context_adapter.llm_top_p,
        max_tokens=int(max_tokens or context_adapter.llm_max_tokens),
        presence_penalty=context_adapter.llm_presence_penalty,
    )
    started = time.perf_counter()
    try:
        content = await llm_client.chat_text(payload, timeout=context_adapter.llm_timeout)
        _log_llm_timing(context_adapter, llm_client, logger_obj, "draft_answer", payload, time.perf_counter() - started, evidence=context, output=content)
        return normalize_answer_citation_style(content)
    except Exception as exc:
        _log_llm_timing(context_adapter, llm_client, logger_obj, "draft_answer", payload, time.perf_counter() - started, evidence=context, error=type(exc).__name__)
        logger_obj.error(f"LLM generation error: {str(exc)}")
        return "当前证据不足，暂不能基于已检索资料回答。"


async def generate_structured_answer(
    runtime: Any,
    llm_client: Any,
    logger_obj: Any,
    query: str,
    context: str,
    qtype: str = "other",
    max_tokens: Optional[int] = None,
    answer_mode: str = "target_hit",
    uncovered_aspects: Optional[List[str]] = None,
    aspect_plan: str = "",
    docs: Optional[List[Any]] = None,
    evidence_gate_warning: str = "",
) -> Tuple[str, Optional[Dict[str, Any]]]:
    context_adapter = _answer_context(runtime)
    prompt = build_structured_answer_prompt(
        query,
        context,
        qtype,
        answer_mode,
        uncovered_aspects,
        aspect_plan,
        evidence_gate_warning=evidence_gate_warning,
    )
    system_prompt = (
        "你是法规问答助手。请输出严格 JSON，不要输出 Markdown。引用只使用数字数组，不要输出 [证据 n]。"
        + legal_clause_answer_rules(query, context, answer_mode)
    )
    payload = llm_client.build_payload(
        system_prompt,
        prompt,
        temperature=context_adapter.llm_temperature,
        top_p=context_adapter.llm_top_p,
        max_tokens=int(max_tokens or context_adapter.llm_max_tokens),
        presence_penalty=context_adapter.llm_presence_penalty,
    )
    started = time.perf_counter()
    try:
        content = await llm_client.chat_text(payload, timeout=context_adapter.llm_timeout)
        _log_llm_timing(context_adapter, llm_client, logger_obj, "structured_answer", payload, time.perf_counter() - started, evidence=context, output=content)
    except Exception as exc:
        _log_llm_timing(context_adapter, llm_client, logger_obj, "structured_answer", payload, time.perf_counter() - started, evidence=context, error=type(exc).__name__)
        logger_obj.error(f"Structured answer generation error: {str(exc)}")
        return "当前证据不足，暂不能基于已检索资料回答。", None
    payload_obj = parse_structured_answer_payload(content)
    if not payload_obj:
        answer = normalize_answer_citation_style(content)
        return answer, structured_answer_from_markdown(context_adapter, answer, aspect_plan, docs or [], uncovered_aspects)
    structured = normalize_structured_answer_payload(context_adapter, payload_obj, aspect_plan, docs or [])
    if not structured.get("aspects"):
        return normalize_answer_citation_style(content), None
    return render_structured_answer_markdown(context_adapter, structured), structured


async def verify_answer(
    runtime: Any,
    llm_client: Any,
    logger_obj: Any,
    query: str,
    context: str,
    draft: str,
    aspect_plan: str = "",
    max_tokens: Optional[int] = None,
) -> str:
    context_adapter = _answer_context(runtime)
    if not int(context_adapter.final_fact_verify_max_tokens or 0):
        return normalize_answer_citation_style(draft)
    prompt = build_answer_verification_prompt(query, context, draft, aspect_plan)
    verify_max_tokens = int(context_adapter.final_fact_verify_max_tokens or max_tokens or context_adapter.llm_max_tokens)
    payload = llm_client.build_payload(
        "你是法规答案核查器。只保留 context 支持的结论，引用格式只能是 [1]、[2]。",
        prompt,
        temperature=0.0,
        top_p=1.0,
        max_tokens=verify_max_tokens,
        presence_penalty=0.0,
    )
    started = time.perf_counter()
    try:
        content = await llm_client.chat_text(payload, timeout=context_adapter.llm_timeout)
        _log_llm_timing(context_adapter, llm_client, logger_obj, "verify_answer", payload, time.perf_counter() - started, evidence=context, output=content)
        return normalize_answer_citation_style(content)
    except Exception as exc:
        _log_llm_timing(context_adapter, llm_client, logger_obj, "verify_answer", payload, time.perf_counter() - started, evidence=context, error=type(exc).__name__)
        logger_obj.error(f"Answer verification error: {str(exc)}")
        return normalize_answer_citation_style(draft)
