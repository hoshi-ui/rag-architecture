import math
import re
from typing import Any, Callable, Dict, List, Optional


LEGAL_CJK_QUERY_PHRASES = (
    "正常生活",
    "重点管理区",
    "一般管理区",
    "人口密集",
    "管理区域",
    "养犬管理",
    "养犬人",
    "犬只",
    "犬证",
    "犬牌",
    "烈性犬",
    "大型犬",
    "导盲犬",
    "扶助犬",
    "携犬",
    "遛犬",
    "圈养",
    "拴养",
    "干扰",
    "放任",
    "驱使",
    "恐吓",
    "伤害",
    "虐待",
    "遗弃",
    "赌博",
    "尸体",
    "禁止性",
    "禁止",
    "义务",
    "处罚",
    "罚款",
    "没收",
    "责令",
    "改正",
    "限期",
    "补办",
    "登记",
    "签注",
    "变更",
    "注销",
    "标识牌",
    "伪造",
    "变造",
    "买卖",
    "证件",
    "公安机关",
    "城市管理",
    "农业农村",
    "社区",
    "物业",
    "业主委员会",
    "村庄",
    "街道",
    "乡镇",
    "县级人民政府",
    "调整",
    "划定",
    "公布",
)

ASPECT_STOPWORDS = {
    "请",
    "哪些",
    "什么",
    "是否",
    "如何",
    "怎么",
    "于",
    "对",
    "和",
    "与",
    "的",
    "为",
    "在",
    "由",
    "向",
    "适用",
    "适用于",
    "用于",
    "中",
    "内",
    "本市",
    "本条例",
    "条例",
    "规定",
    "办法",
    "规则",
    "管理条例",
    "相关",
    "有关",
    "及其",
    "分别",
    "列出",
    "说明",
    "解释",
}

ASPECT_CORE_MARKERS = (
    "出租房",
    "安全管理",
    "租赁",
    "治安",
    "消防",
    "监督活动",
    "生产经营",
    "居住",
    "民宿",
    "旅馆业客房",
    "客房",
    "处罚",
    "罚款",
    "责任",
    "义务",
    "禁止",
    "程序",
    "流程",
    "期限",
    "标准",
    "条件",
    "范围",
    "定义",
    "职责",
    "备案",
    "审批",
    "登记",
)

LOW_VALUE_CJK_GRAMS = {
    "法规",
    "条例",
    "规定",
    "什么",
    "哪些",
    "如何",
    "以及",
    "或者",
    "本人",
    "可能",
    "请不",
    "不要",
    "回答",
    "机关",
    "职责",
    "领域",
    "资料",
}


def token_overlap_score(query: str, text: str, match_terms: Callable[[str], List[str]]) -> float:
    terms = match_terms(query)
    if not terms:
        return 0.0
    haystack = text or ""
    score = 0.0
    for term in terms:
        if term in haystack:
            score += max(1.0, min(len(term), 8) / 2.0)
    return score


def tok_terms(text: str) -> List[str]:
    s = (text or "").lower()
    terms: List[str] = []
    buf: List[str] = []
    for ch in s:
        if "\u4e00" <= ch <= "\u9fff":
            if buf:
                terms.append("".join(buf))
                buf = []
            terms.append(ch)
        elif ch.isalnum() or ch in ("_",):
            buf.append(ch)
        else:
            if buf:
                term = "".join(buf)
                if term:
                    terms.append(term)
                buf = []
    if buf:
        term = "".join(buf)
        if term:
            terms.append(term)
    return terms


def _append_unique(values: List[str], term: str, limit: int) -> bool:
    term = (term or "").strip().lower()
    if len(term) < 2:
        return len(values) >= limit
    if term not in values:
        values.append(term)
    return len(values) >= limit


def _is_low_value_cjk_gram(term: str) -> bool:
    if term in LOW_VALUE_CJK_GRAMS:
        return True
    if len(term) <= 2 and all(ch in "的是了和与及或在中为对有无按其该本各等" for ch in term):
        return True
    return False


def query_match_terms(text: str, limit: int = 12) -> List[str]:
    """Extract recall-friendly query terms, preserving useful Chinese phrases."""
    normalized = (text or "").lower()
    values: List[str] = []

    for match in re.findall(r"第[一二三四五六七八九十百千万0-9]+[条款项章]", normalized):
        if _append_unique(values, match, limit):
            return values[:limit]

    for phrase in LEGAL_CJK_QUERY_PHRASES:
        if phrase in normalized and _append_unique(values, phrase, limit):
            return values[:limit]

    for match in re.findall(r"[a-z0-9_]{2,}", normalized):
        if _append_unique(values, match, limit):
            return values[:limit]

    for seq in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        for size in (4, 3, 2):
            if len(seq) < size:
                continue
            for idx in range(0, len(seq) - size + 1):
                gram = seq[idx : idx + size]
                if _is_low_value_cjk_gram(gram):
                    continue
                if _append_unique(values, gram, limit):
                    return values[:limit]

    for token in tok_terms(normalized):
        if _append_unique(values, token, limit):
            return values[:limit]
    return values[:limit]


def _strip_doc_titles_for_aspects(text: str) -> str:
    value = re.sub(r"《[^》]{1,120}》", " ", text or "")
    value = re.sub(r"[\"'“”‘’（）()【】\[\]{}]", " ", value)
    return value


def _clean_aspect_candidate(term: str) -> str:
    value = (term or "").strip().lower()
    if not value:
        return ""
    value = re.sub(r"[，。；;:：、？?！!\s]+", "", value)
    for stop in sorted(ASPECT_STOPWORDS, key=len, reverse=True):
        value = value.replace(stop, "")
    value = re.sub(r"第[一二三四五六七八九十百千万0-9]+[条款项章]", "", value)
    value = value.strip()
    if len(value) < 2:
        return ""
    if value in ASPECT_STOPWORDS or _is_low_value_cjk_gram(value):
        return ""
    if len(value) <= 3 and value.endswith(("市", "省", "区", "县")):
        return ""
    if any(suffix in value for suffix in ("条例", "办法", "规定", "规则")):
        return ""
    return value


def normalize_core_aspect_term(term: str) -> str:
    """Normalize one semantic aspect and drop low-value fragments."""
    return _clean_aspect_candidate(term)


def query_core_aspect_terms(text: str, base_terms: Optional[List[str]] = None, limit: int = 4) -> List[str]:
    """Extract coverage-friendly core noun aspects instead of recall n-grams."""
    normalized = (text or "").lower()
    body = _strip_doc_titles_for_aspects(normalized)
    out: List[str] = []

    def add(term: str) -> None:
        value = _clean_aspect_candidate(term)
        if value and value not in out:
            out.append(value)

    for marker in ASPECT_CORE_MARKERS:
        if marker in body:
            add(marker)
        if len(out) >= limit:
            return out[:limit]

    for seq in re.findall(r"[\u4e00-\u9fff]{2,}", body):
        cleaned = _clean_aspect_candidate(seq)
        if not cleaned:
            continue
        if len(cleaned) > 8:
            for marker in ASPECT_CORE_MARKERS:
                if marker in cleaned:
                    add(marker)
                    if len(out) >= limit:
                        return out[:limit]
            continue
        add(cleaned)
        if len(out) >= limit:
            return out[:limit]

    for term in base_terms or []:
        add(term)
        if len(out) >= limit:
            break
    return out[:limit]


def bm25_scores(query: str, documents: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    if not documents:
        return []
    q_terms = tok_terms(query)
    if not q_terms:
        return [0.0] * len(documents)
    docs_terms = [tok_terms(text or "") for text in documents]
    n_docs = len(documents)
    doc_lens = [len(terms) for terms in docs_terms]
    avgdl = (sum(doc_lens) / n_docs) if n_docs else 0.0
    df: Dict[str, int] = {}
    for terms in docs_terms:
        seen = set()
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            df[term] = df.get(term, 0) + 1
    idf: Dict[str, float] = {}
    for term in set(q_terms):
        dft = df.get(term, 0)
        idf[term] = math.log(1 + (n_docs - dft + 0.5) / (dft + 0.5))
    scores: List[float] = []
    for terms, doc_len in zip(docs_terms, doc_lens):
        tf: Dict[str, int] = {}
        for term in terms:
            tf[term] = tf.get(term, 0) + 1
        score = 0.0
        for term in q_terms:
            if term not in idf:
                continue
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            denom = freq + k1 * (1 - b + b * (doc_len / (avgdl or 1.0)))
            score += idf[term] * (freq * (k1 + 1)) / (denom or 1.0)
        scores.append(score)
    return scores


def minmax_norm(arr: List[float]) -> List[float]:
    if not arr:
        return []
    mn = min(arr)
    mx = max(arr)
    if mx - mn <= 1e-9:
        return [0.0 for _ in arr]
    return [(value - mn) / (mx - mn) for value in arr]


def passes_relevance_cluster(
    docs: List[Any],
    score_mode: str,
    thresholds: Dict[str, float],
    hit_score: Callable[[Any], float],
    top_n: int = 3,
) -> bool:
    if not docs:
        return False
    take = docs[: max(1, min(len(docs), top_n))]
    if score_mode == "distance":
        good = [doc for doc in take if hit_score(doc) <= thresholds.get("max_distance", 0.8)]
    else:
        good = [doc for doc in take if hit_score(doc) >= thresholds.get("min_score", 0.25)]
    return len(good) > 0
