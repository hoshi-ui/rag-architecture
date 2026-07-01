import copy
import json
import logging
import os
from typing import Any, Dict

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(path: str = "", *_, **__) -> bool:
        if not path or not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as file_obj:
            for raw_line in file_obj:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        return True


logger = logging.getLogger("rag-app")


DEFAULT_RETRIEVAL_POLICY: Dict[str, Any] = {
    "query_route": {
        "order": [
            "existence", "visibility_probe", "version_switch", "explicit_doc_reference",
            "explicit_regulation_reference", "weak_title_reference", "business_topic_qa",
            "open_regulation_qa", "content_qa",
        ],
        "weak_title_reference": {"require_no_filenames": True},
    },
    "source_resolution": {
        "title_candidate_limit": 5,
        "fallback_candidate_limit": 3,
        "clarification_examples_limit": 3,
    },
    "question_type_patterns": [
        {"type": "screening", "keywords": ["是否符合", "是否满足", "能否", "可否", "有没有条件", "是否需要", "判断"]},
        {"type": "single_doc_extract", "keywords": ["第几条", "哪一条", "怎么规定", "具体规定", "条款内容", "规定了什么", "原文"]},
        {"type": "summary", "keywords": ["总结", "概括", "主要内容", "说明", "梳理"]},
        {"type": "arch", "keywords": ["架构", "体系结构", "流程", "模块", "组件", "链路", "机制", "如何设计", "怎么实现"]},
        {"type": "compare", "keywords": ["比较", "对比", "区别", "差异", "异同", "分别", "相比", "vs", "versus", "difference between", "compare"]},
        {"type": "howto", "keywords": ["怎么办", "如何", "怎么", "步骤", "流程", "办理", "申请", "how to", "steps"]},
        {"type": "definition", "keywords": ["是什么", "什么意思", "定义", "meaning of", "what is"]},
    ],
    "rerank": {
        "section_lookup": {
            "intent_keywords": ["职责", "权限", "程序", "条件", "范围", "要求", "标准", "责任", "处罚", "监督管理", "登记备案", "法律责任"],
            "restriction_keywords": ["不得", "禁止", "限制"],
            "restriction_subject_keywords": ["主体", "对象", "人员"],
            "trigger_qtypes": ["regulation_execution"],
        },
        "clause_lookup": {
            "keywords": ["第", "条", "款", "项", "规定", "原文", "内容", "职责", "处罚", "罚款", "备案", "许可", "审批", "条件", "流程", "范围", "标准"],
            "trigger_qtypes": ["single_doc_extract", "regulation_execution", "howto"],
        },
        "broad": {
            "keywords": ["主要内容", "总体要求", "整体说明", "总结", "概括", "梳理", "介绍", "有哪些", "是什么"],
            "trigger_qtypes": ["summary"],
        },
    },
    "query_filters": {
        "doc_type_rules": [
            {"value": "regulation", "match_any": ["管理条例", "管理办法", "规定"]},
            {"value": "research_report", "match_any": ["研究报告", "调研报告", "白皮书"]},
        ],
        "topic_rules": [
            {"value": "生态环境", "match_any": ["生态环境保护"], "match_all": ["生态", "环境"]},
        ],
    },
}

def load_app_env(base_dir: str) -> None:
    candidates = [
        os.getenv("APP_ENV_FILE"),
        os.path.join(base_dir, "..", "..", "config", "app.env"),
        "/app/config/app.env",
    ]
    for dotenv_path in candidates:
        if dotenv_path and os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
            break
    else:
        load_dotenv()


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_retrieval_policy(default_policy: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    candidates = [
        os.getenv("RETRIEVAL_POLICY_FILE"),
        os.path.join(base_dir, "..", "..", "config", "retrieval_policy.json"),
        "/app/config/retrieval_policy.json",
    ]
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            return deep_merge_dict(default_policy, payload)
        except Exception as exc:
            logger.warning(f"Failed to load retrieval policy from {path}: {exc}")
    return copy.deepcopy(default_policy)


def policy_get(policy: Dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = policy
    for part in (path or "").split("."):
        if not part:
            continue
        if not isinstance(node, dict) or part not in node:
            return default
        node = node.get(part)
    return node if node is not None else default


def policy_keywords(policy: Dict[str, Any], path: str) -> list[str]:
    values = policy_get(policy, path, default=[])
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value).strip()]


def policy_match_rule(query: str, rule: Dict[str, Any]) -> bool:
    q = (query or "").lower()
    match_any = [str(value).lower() for value in (rule or {}).get("match_any") or [] if str(value).strip()]
    match_all = [str(value).lower() for value in (rule or {}).get("match_all") or [] if str(value).strip()]
    branches: list[bool] = []
    if match_any:
        branches.append(any(token in q for token in match_any))
    if match_all:
        branches.append(all(token in q for token in match_all))
    if not branches:
        return False
    return any(branches)


def resolve_runtime_upload_dir(base_dir: str) -> str:
    legacy_upload_dir = os.path.join(base_dir, "uploads")
    local_runtime_root = os.path.join(base_dir, "data")
    default_root = "/storage" if os.path.abspath(base_dir) == "/app" else local_runtime_root
    runtime_root = os.getenv("RAG_RUNTIME_ROOT", default_root)
    candidate = (
        runtime_root
        if os.path.basename(runtime_root.rstrip("/")) == "uploads"
        else os.path.join(runtime_root, "uploads")
    )
    try:
        os.makedirs(candidate, exist_ok=True)
        return candidate
    except Exception:
        os.makedirs(legacy_upload_dir, exist_ok=True)
        return legacy_upload_dir


def resolve_runtime_database_dir(base_dir: str) -> str:
    project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    default_database_dir = os.path.join(project_root, "database")
    candidate = os.getenv("RAG_DATABASE_DIR", default_database_dir)
    try:
        os.makedirs(candidate, exist_ok=True)
        return candidate
    except Exception:
        os.makedirs(default_database_dir, exist_ok=True)
        return default_database_dir


def resolve_project_root(base_dir: str) -> str:
    return os.path.abspath(os.path.join(base_dir, "..", ".."))


def resolve_web_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "web")


def resolve_legacy_upload_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "uploads")


def resolve_default_database_dir(base_dir: str) -> str:
    return os.path.join(resolve_project_root(base_dir), "database")


load_app_env(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class Config:
    APP_ENV = os.getenv("APP_ENV", "").lower()
    APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("APP_PORT", "8080"))
    MILVUS_HOST = os.getenv("MILVUS_HOST", ("127.0.0.1" if APP_ENV == "test_local" else "milvus"))
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_USER = os.getenv("MILVUS_USER", "minioadmin")
    MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "minioadmin")
    MILVUS_SECURE = os.getenv("MILVUS_SECURE", "false").lower() == "true"
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_documents")
    MILVUS_EMBEDDING_DIM = int(os.getenv("MILVUS_EMBEDDING_DIM", os.getenv("EMBEDDING_DIM", "1024")))
    
    EMBEDDING_URL = os.getenv("EMBEDDING_SERVICE_URL", ("http://127.0.0.1:8001" if APP_ENV == "test_local" else "http://embedding-service:8000"))
    RERANK_URL = os.getenv("RERANK_SERVICE_URL", ("http://127.0.0.1:8002" if APP_ENV == "test_local" else "http://rerank-service:8000"))
    OCR_SERVICE_URL = os.getenv("OCR_SERVICE_URL", "")
    OCR_MODE = os.getenv("OCR_MODE", "general")
    OCR_LANG = os.getenv("OCR_LANG", "auto")
    OCR_TIMEOUT = int(os.getenv("OCR_TIMEOUT", "60"))
    OCR_SHARED_CONTAINER_DIR = os.getenv("OCR_SHARED_CONTAINER_DIR", "")
    OCR_SHARED_HOST_DIR = os.getenv("OCR_SHARED_HOST_DIR", "")
    PDF_OCR_MAX_TEXT_CHARS_PER_PAGE = float(os.getenv("PDF_OCR_MAX_TEXT_CHARS_PER_PAGE", "300"))
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    MAX_BATCH_TOTAL_SIZE_MB = int(os.getenv("MAX_BATCH_TOTAL_SIZE_MB", "200"))
    MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "300"))
    MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))
    MAX_XLSX_ROWS = int(os.getenv("MAX_XLSX_ROWS", "20000"))
    MAX_XLSX_COLS = int(os.getenv("MAX_XLSX_COLS", "200"))
    MAX_XLSX_SHEETS = int(os.getenv("MAX_XLSX_SHEETS", "50"))
    MIN_PARSE_TEXT_CHARS = int(os.getenv("MIN_PARSE_TEXT_CHARS", "40"))
    MIN_PARSE_QUALITY_SCORE = float(os.getenv("MIN_PARSE_QUALITY_SCORE", "0.35"))
    SQLITE_BUSY_TIMEOUT_SEC = float(os.getenv("SQLITE_BUSY_TIMEOUT_SEC", "30"))
    
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-7b-instruct")
    LLM_API_BASE = os.getenv("LLM_API_BASE", "http://ollama:11434/v1")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_CHAT_COMPLETIONS_URL = os.getenv("LLM_CHAT_COMPLETIONS_URL", "")
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.6"))
    LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1200"))
    LLM_MAX_OUTPUT_TOKENS_CAP = int(os.getenv("LLM_MAX_OUTPUT_TOKENS_CAP", "2048"))
    LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "262144"))
    LLM_OUTPUT_TOKEN_SAFETY_MARGIN = int(os.getenv("LLM_OUTPUT_TOKEN_SAFETY_MARGIN", "2048"))
    ENABLE_FINAL_FACT_VERIFY = False
    FINAL_FACT_VERIFY_MAX_TOKENS = 0
    ENABLE_LLM_TIMING_LOG = os.getenv("ENABLE_LLM_TIMING_LOG", "true").lower() == "true"
    LLM_PRESENCE_PENALTY = float(os.getenv("LLM_PRESENCE_PENALTY", "1.5"))
    LLM_EXTRA_BODY = os.getenv("LLM_EXTRA_BODY", "")
    ENABLE_STRUCTURED_ANSWER_JSON = os.getenv("ENABLE_STRUCTURED_ANSWER_JSON", "false").lower() == "true"

    ENABLE_LLM_QUERY_PARSE = os.getenv("ENABLE_LLM_QUERY_PARSE", "true").lower() == "true"
    QUERY_PARSE_MAX_TOKENS = int(os.getenv("QUERY_PARSE_MAX_TOKENS", "260"))
    QUERY_PARSE_CACHE_SIZE = int(os.getenv("QUERY_PARSE_CACHE_SIZE", "512"))
    ENABLE_LLM_QUERY_DECOMPOSITION = os.getenv("ENABLE_LLM_QUERY_DECOMPOSITION", "true").lower() == "true"
    LLM_QUERY_DECOMPOSITION_MAX_TOKENS = int(os.getenv("LLM_QUERY_DECOMPOSITION_MAX_TOKENS", "420"))
    LLM_QUERY_DECOMPOSITION_TIMEOUT = int(os.getenv("LLM_QUERY_DECOMPOSITION_TIMEOUT", "8"))
    ENABLE_LLM_ABSTRACTION_UNPACKING = os.getenv("ENABLE_LLM_ABSTRACTION_UNPACKING", "true").lower() == "true"
    LLM_ABSTRACTION_UNPACKING_MAX_TOKENS = int(os.getenv("LLM_ABSTRACTION_UNPACKING_MAX_TOKENS", "80"))
    LLM_ABSTRACTION_UNPACKING_TIMEOUT = int(os.getenv("LLM_ABSTRACTION_UNPACKING_TIMEOUT", "5"))
    ENABLE_RETRIEVAL_HYDE = os.getenv("ENABLE_RETRIEVAL_HYDE", "true").lower() == "true"
    RETRIEVAL_HYDE_TIMEOUT = int(os.getenv("RETRIEVAL_HYDE_TIMEOUT", "6"))
    RETRIEVAL_HYDE_MAX_TOKENS = int(os.getenv("RETRIEVAL_HYDE_MAX_TOKENS", "220"))
    RETRIEVAL_HYDE_MAX_CHARS = int(os.getenv("RETRIEVAL_HYDE_MAX_CHARS", "360"))
    RETRIEVAL_HYDE_ABSTRACT_ONLY = os.getenv("RETRIEVAL_HYDE_ABSTRACT_ONLY", "true").lower() == "true"
    QUERY_PARSE_MAX_DOCS = int(os.getenv("QUERY_PARSE_MAX_DOCS", "2"))
    QUERY_PARSE_MAX_ANCHORS = int(os.getenv("QUERY_PARSE_MAX_ANCHORS", "1"))
    QUERY_PARSE_MAX_ASPECTS = int(os.getenv("QUERY_PARSE_MAX_ASPECTS", "4"))
    QUERY_PARSE_MAX_SECTION_TARGETS = int(os.getenv("QUERY_PARSE_MAX_SECTION_TARGETS", "4"))
    ENABLE_LLM_INTENT_CLASSIFIER = os.getenv("ENABLE_LLM_INTENT_CLASSIFIER", "false").lower() == "true"
    INTENT_CLASSIFIER_MAX_TOKENS = int(os.getenv("INTENT_CLASSIFIER_MAX_TOKENS", "320"))
    INTENT_CLASSIFIER_CACHE_SIZE = int(os.getenv("INTENT_CLASSIFIER_CACHE_SIZE", "512"))
    INTENT_CLASSIFIER_MIN_CONFIDENCE = float(os.getenv("INTENT_CLASSIFIER_MIN_CONFIDENCE", "0.62"))
    ENABLE_LLM_TOOL_ROUTER = os.getenv("ENABLE_LLM_TOOL_ROUTER", "true").lower() == "true"
    LLM_TOOL_ROUTER_MAX_TOKENS = int(os.getenv("LLM_TOOL_ROUTER_MAX_TOKENS", "120"))
    LLM_TOOL_ROUTER_TIMEOUT = int(os.getenv("LLM_TOOL_ROUTER_TIMEOUT", "8"))
    ENABLE_AGENTIC_ROUTER = os.getenv("ENABLE_AGENTIC_ROUTER", "true").lower() == "true"
    AGENTIC_ROUTER_MAX_TOKENS = int(os.getenv("AGENTIC_ROUTER_MAX_TOKENS", "520"))
    AGENTIC_ROUTER_TIMEOUT = int(os.getenv("AGENTIC_ROUTER_TIMEOUT", "8"))
    AGENTIC_ROUTER_MIN_CONFIDENCE = float(os.getenv("AGENTIC_ROUTER_MIN_CONFIDENCE", "0.55"))

    ENABLE_COMPARE_INTENT_TAG = os.getenv("ENABLE_COMPARE_INTENT_TAG", "true").lower() == "true"

    ENABLE_LLM_EVIDENCE_CHECK = os.getenv("ENABLE_LLM_EVIDENCE_CHECK", "false").lower() == "true"
    LLM_EVIDENCE_CHECK_MAX_CHARS = int(os.getenv("LLM_EVIDENCE_CHECK_MAX_CHARS", "1800"))
    LLM_EVIDENCE_CHECK_MAX_TOKENS = int(os.getenv("LLM_EVIDENCE_CHECK_MAX_TOKENS", "48"))
    LLM_EVIDENCE_CHECK_TIMEOUT = int(os.getenv("LLM_EVIDENCE_CHECK_TIMEOUT", "4"))
    LLM_EVIDENCE_CHECK_MIN_DENSE_REL = float(os.getenv("LLM_EVIDENCE_CHECK_MIN_DENSE_REL", "0.0"))
    EVIDENCE_GATE_RERANK_MIN_SCORE = float(os.getenv("EVIDENCE_GATE_RERANK_MIN_SCORE", os.getenv("MIN_EVIDENCE_SCORE", "0.6")))
    ENABLE_LLM_CHUNK_METADATA_ENRICHMENT = os.getenv("ENABLE_LLM_CHUNK_METADATA_ENRICHMENT", "true").lower() == "true"
    LLM_CHUNK_METADATA_TIMEOUT = int(os.getenv("LLM_CHUNK_METADATA_TIMEOUT", "20"))
    LLM_CHUNK_METADATA_MAX_CHARS = int(os.getenv("LLM_CHUNK_METADATA_MAX_CHARS", "1200"))
    DENSE_BACKSTOP_MIN_REL = float(os.getenv("DENSE_BACKSTOP_MIN_REL", "0.58"))
    LOCKED_DOC_RECALL_K = int(os.getenv("LOCKED_DOC_RECALL_K", "60"))
    
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    OVERLAP = int(os.getenv("OVERLAP", "100"))
    TOP_K = int(os.getenv("TOP_K", "80"))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "10"))
    ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"
    MIN_QUERY_CHARS = int(os.getenv("MIN_QUERY_CHARS", "2"))
    MAX_QUERY_CHARS = int(os.getenv("MAX_QUERY_CHARS", "800"))
    CONTEXT_TOP_N = int(os.getenv("CONTEXT_TOP_N", "6"))
    CONTEXT_DOC_MAX_CHARS = int(os.getenv("CONTEXT_DOC_MAX_CHARS", "1200"))
    CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "8000"))
    EVIDENCE_MAX_TOKENS = int(os.getenv("EVIDENCE_MAX_TOKENS", "6500"))
    MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.25"))
    ABSOLUTE_MIN_RELEVANCE_SCORE = float(os.getenv("ABSOLUTE_MIN_RELEVANCE_SCORE", "0.15"))
    MAX_RELEVANCE_DISTANCE = float(os.getenv("MAX_RELEVANCE_DISTANCE", "0.8"))
    MIN_EVIDENCE_SCORE = float(os.getenv("MIN_EVIDENCE_SCORE", "0.6"))
    MIN_SUBSTANTIVE_CHUNKS = int(os.getenv("MIN_SUBSTANTIVE_CHUNKS", "1"))
    MIN_RESCUE_SCORE = float(os.getenv("MIN_RESCUE_SCORE", "0.48"))
    ENABLE_SEMANTIC_SOFTENING = os.getenv("ENABLE_SEMANTIC_SOFTENING", "true").lower() == "true"
    SEMANTIC_SOFTENING_MIN_SIM = float(os.getenv("SEMANTIC_SOFTENING_MIN_SIM", "0.85"))
    SEMANTIC_SOFTENING_MAX_TEXT_CHARS = int(os.getenv("SEMANTIC_SOFTENING_MAX_TEXT_CHARS", "1200"))
    ENABLE_DENSE_TITLE_FALLBACK = os.getenv("ENABLE_DENSE_TITLE_FALLBACK", "true").lower() == "true"
    DENSE_TITLE_MATCH_MIN_SIM = float(os.getenv("DENSE_TITLE_MATCH_MIN_SIM", "0.84"))
    DENSE_TITLE_MATCH_MARGIN = float(os.getenv("DENSE_TITLE_MATCH_MARGIN", "0.03"))
    DENSE_TITLE_PROBE_MAX_CHARS = int(os.getenv("DENSE_TITLE_PROBE_MAX_CHARS", "160"))
    ENABLE_OPEN_TOPIC_MULTI_DOC = os.getenv("ENABLE_OPEN_TOPIC_MULTI_DOC", "true").lower() == "true"
    OPEN_TOPIC_MULTI_DOC_MIN_PRIOR = float(os.getenv("OPEN_TOPIC_MULTI_DOC_MIN_PRIOR", "0.32"))
    OPEN_TOPIC_MULTI_DOC_MAX_SOURCES = int(os.getenv("OPEN_TOPIC_MULTI_DOC_MAX_SOURCES", "3"))
    PARTIAL_TERM_RESCUE_RELAX_RATIO = float(os.getenv("PARTIAL_TERM_RESCUE_RELAX_RATIO", "0.9"))
    PARTIAL_TERM_RESCUE_MIN_SUBSTANTIVE_CHUNKS = int(os.getenv("PARTIAL_TERM_RESCUE_MIN_SUBSTANTIVE_CHUNKS", "3"))
    PARTIAL_TERM_RESCUE_MIN_FOCUS_SCORE = float(os.getenv("PARTIAL_TERM_RESCUE_MIN_FOCUS_SCORE", "0.72"))
    PRIMARY_EVIDENCE_TOPK = int(os.getenv("PRIMARY_EVIDENCE_TOPK", "5"))
    RESCUE_EVIDENCE_TOPK = int(os.getenv("RESCUE_EVIDENCE_TOPK", "15"))
    ALLOW_GUARDED_FULL = os.getenv("ALLOW_GUARDED_FULL", "true").lower() == "true"
    REQUIRE_EVIDENCE = os.getenv("REQUIRE_EVIDENCE", "true").lower() == "true"
    RECALL_TOP_K = int(os.getenv("RECALL_TOP_K", "20"))
    RECALL_RELATIVE_SCORE_RATIO = float(os.getenv("RECALL_RELATIVE_SCORE_RATIO", "0.72"))
    RECALL_MIN_KEEP_N = int(os.getenv("RECALL_MIN_KEEP_N", "3"))
    RERANK_KEEP_N = int(os.getenv("RERANK_KEEP_N", "8"))
    FINAL_CONTEXT_N = int(os.getenv("FINAL_CONTEXT_N", "10"))
    FINAL_CONTEXT_N_MAX = int(os.getenv("FINAL_CONTEXT_N_MAX", "10"))
    ENABLE_DYNAMIC_ELBOW_TRUNCATION = os.getenv("ENABLE_DYNAMIC_ELBOW_TRUNCATION", "true").lower() == "true"
    DYNAMIC_ELBOW_SCORE_DELTA = float(os.getenv("DYNAMIC_ELBOW_SCORE_DELTA", "0.025"))
    DYNAMIC_ELBOW_MAX_EXTRA = int(os.getenv("DYNAMIC_ELBOW_MAX_EXTRA", "5"))
    MAX_MERGED_CHUNKS_PER_EVIDENCE = int(os.getenv("MAX_MERGED_CHUNKS_PER_EVIDENCE", "2"))
    MAX_MERGED_EVIDENCE_CHARS = int(os.getenv("MAX_MERGED_EVIDENCE_CHARS", "1800"))
    MAX_SNIPPETS_PER_SOURCE = int(os.getenv("MAX_SNIPPETS_PER_SOURCE", "2"))
    SECTION_MAX_CHARS = int(os.getenv("SECTION_MAX_CHARS", "1600"))
    ANSWER_MAX_POINTS = int(os.getenv("ANSWER_MAX_POINTS", "5"))
    LLM_MAX_TOKENS_DEF = int(os.getenv("LLM_MAX_TOKENS_DEF", "900"))
    LLM_MAX_TOKENS_SUMMARY = int(os.getenv("LLM_MAX_TOKENS_SUMMARY", "1000"))
    LLM_MAX_TOKENS_HOWTO = int(os.getenv("LLM_MAX_TOKENS_HOWTO", "1200"))
    LLM_MAX_TOKENS_COMPARE = int(os.getenv("LLM_MAX_TOKENS_COMPARE", "1400"))
    LLM_MAX_TOKENS_OTHER = int(os.getenv("LLM_MAX_TOKENS_OTHER", "1000"))
    LLM_MAX_TOKENS_ARCH = int(os.getenv("LLM_MAX_TOKENS_ARCH", "1000"))
    SOURCES_MAX_DISTANCE_ADD = float(os.getenv("SOURCES_MAX_DISTANCE_ADD", "0.08"))
    SOURCES_MIN_SCORE_RATIO = float(os.getenv("SOURCES_MIN_SCORE_RATIO", "0.6"))
    # Hybrid dense/lexical fusion controls.
    FUSION_ALPHA = float(os.getenv("FUSION_ALPHA", "0.5"))
    LEXICAL_RECALL_LIMIT = int(os.getenv("LEXICAL_RECALL_LIMIT", "1000"))
    DISPLAY_SCORE_RATIO = float(os.getenv("DISPLAY_SCORE_RATIO", "0.8"))
    DISPLAY_DISTANCE_MARGIN = float(os.getenv("DISPLAY_DISTANCE_MARGIN", "0.02"))
    TEST_LEX_ONLY = os.getenv("TEST_LEX_ONLY", "false").lower() == "true"
    RRF_K = int(os.getenv("RRF_K", "60"))
    FUSION_W_DENSE = float(os.getenv("FUSION_W_DENSE", "0.80"))
    FUSION_W_LEX = float(os.getenv("FUSION_W_LEX", "0.12"))
    FUSION_W_PRIOR = float(os.getenv("FUSION_W_PRIOR", "0.002"))
    FUSION_W_DOC_PRIOR = float(os.getenv("FUSION_W_DOC_PRIOR", "0.003"))
    FUSION_W_RERANK_DOC = float(os.getenv("FUSION_W_RERANK_DOC", "0.3"))
    FUSION_M_TERM = float(os.getenv("FUSION_M_TERM", "1.08"))
    FUSION_M_TITLE = float(os.getenv("FUSION_M_TITLE", "1.35"))
    FUSION_M_DOC_RECALL = float(os.getenv("FUSION_M_DOC_RECALL", "1.2"))
    FUSION_M_AGREEMENT = float(os.getenv("FUSION_M_AGREEMENT", "1.12"))
    FUSION_MUD_SCORE = float(os.getenv("FUSION_MUD_SCORE", "0.018"))
    DENSE_BACKSTOP_MIN_SCORE = float(os.getenv("DENSE_BACKSTOP_MIN_SCORE", "0.55"))
    QUERY_ANCHOR_DENSE_BYPASS_MIN_SCORE = float(os.getenv("QUERY_ANCHOR_DENSE_BYPASS_MIN_SCORE", "0.6"))
    RETRIEVAL_CANDIDATE_K = int(os.getenv("RETRIEVAL_CANDIDATE_K", "60"))
    CHUNK_RERANK_KEEP_N = int(os.getenv("CHUNK_RERANK_KEEP_N", "18"))
    CHUNK_RERANK_POOL_N = int(os.getenv("CHUNK_RERANK_POOL_N", "60"))
    SOURCE_RERANK_KEEP_N = int(os.getenv("SOURCE_RERANK_KEEP_N", "6"))
    HYBRID_SOURCE_PRUNE_ENABLED = os.getenv("HYBRID_SOURCE_PRUNE_ENABLED", "true").lower() == "true"
    HYBRID_SOURCE_PRUNE_KEEP = int(os.getenv("HYBRID_SOURCE_PRUNE_KEEP", "6"))
    HYBRID_SOURCE_PRUNE_MIN_KEEP = int(os.getenv("HYBRID_SOURCE_PRUNE_MIN_KEEP", "2"))
    HYBRID_SOURCE_PRUNE_RATIO = float(os.getenv("HYBRID_SOURCE_PRUNE_RATIO", "0.68"))
    HYBRID_SOURCE_PRUNE_MIN_SCORE = float(os.getenv("HYBRID_SOURCE_PRUNE_MIN_SCORE", "0.0"))
    ENABLE_CHUNK_RERANK = os.getenv("ENABLE_CHUNK_RERANK", "true").lower() == "true"
    RERANK_LOW_CONF_ONLY = os.getenv("RERANK_LOW_CONF_ONLY", "true").lower() == "true"
    RERANK_SOURCE_SCORE_GAP = float(os.getenv("RERANK_SOURCE_SCORE_GAP", "0.04"))
    RERANK_VERSION_FRESHNESS_REWARD = float(os.getenv("RERANK_VERSION_FRESHNESS_REWARD", "0.04"))
    RERANK_ARTICLE_MATCH_REWARD = float(os.getenv("RERANK_ARTICLE_MATCH_REWARD", "0.2"))
    CLAUSE_RERANK_NON_EXACT_ARTICLE_PENALTY = float(os.getenv("CLAUSE_RERANK_NON_EXACT_ARTICLE_PENALTY", "0.25"))
    CLAUSE_RERANK_HEADING_MATCH_BONUS = float(os.getenv("CLAUSE_RERANK_HEADING_MATCH_BONUS", "0.15"))
    CLAUSE_RERANK_TOPIC_MATCH_BONUS = float(os.getenv("CLAUSE_RERANK_TOPIC_MATCH_BONUS", "0.15"))
    CLAUSE_RERANK_INTENT_MATCH_BONUS = float(os.getenv("CLAUSE_RERANK_INTENT_MATCH_BONUS", "0.18"))
    CLAUSE_RERANK_INTENT_HEADING_BONUS = float(os.getenv("CLAUSE_RERANK_INTENT_HEADING_BONUS", "0.12"))
    CLAUSE_RERANK_NEIGHBOR_ARTICLE_PENALTY = float(os.getenv("CLAUSE_RERANK_NEIGHBOR_ARTICLE_PENALTY", "0.10"))
    SOURCE_LOCK_TITLE_MATCH_BONUS = float(os.getenv("SOURCE_LOCK_TITLE_MATCH_BONUS", "0.35"))
    SOURCE_LOCK_REGION_MATCH_BONUS = float(os.getenv("SOURCE_LOCK_REGION_MATCH_BONUS", "0.25"))
    SOURCE_LOCK_ANCHOR_MATCH_BONUS = float(os.getenv("SOURCE_LOCK_ANCHOR_MATCH_BONUS", "0.30"))
    SOURCE_LOCK_PRIOR_BONUS_CAP = float(os.getenv("SOURCE_LOCK_PRIOR_BONUS_CAP", "0.15"))
    SOURCE_LOCK_PRIOR_BONUS_WEIGHT = float(os.getenv("SOURCE_LOCK_PRIOR_BONUS_WEIGHT", "0.15"))
    SOURCE_LOCK_MIN_ACCEPT_SCORE = float(os.getenv("SOURCE_LOCK_MIN_ACCEPT_SCORE", "0.55"))
    ENABLE_SOURCE_LOCK_SOFT_DEGRADATION = os.getenv("ENABLE_SOURCE_LOCK_SOFT_DEGRADATION", "true").lower() == "true"
    SOURCE_LOCK_SOFT_DEGRADE_MIN_CONFIDENCE = float(os.getenv("SOURCE_LOCK_SOFT_DEGRADE_MIN_CONFIDENCE", "0.90"))
    SOURCE_WEAK_MATCH_UPGRADE_MIN_SCORE = float(os.getenv("SOURCE_WEAK_MATCH_UPGRADE_MIN_SCORE", "0.70"))
    SOURCE_DENSE_TITLE_EXTRA_MARGIN = float(os.getenv("SOURCE_DENSE_TITLE_EXTRA_MARGIN", "0.05"))
    AGENTIC_SUPPLEMENT_REGIONLESS_MIN_SCORE = float(os.getenv("AGENTIC_SUPPLEMENT_REGIONLESS_MIN_SCORE", "8.0"))
    AGENTIC_SUPPLEMENT_MIN_SCORE = float(os.getenv("AGENTIC_SUPPLEMENT_MIN_SCORE", "0.75"))
    AMBIGUOUS_SOFT_LOCK_SINGLE_RAW_TITLE_SCORE = float(os.getenv("AMBIGUOUS_SOFT_LOCK_SINGLE_RAW_TITLE_SCORE", "6.2"))
    AMBIGUOUS_SOFT_LOCK_MULTI_RAW_TITLE_SCORE = float(os.getenv("AMBIGUOUS_SOFT_LOCK_MULTI_RAW_TITLE_SCORE", "5.6"))
    OPEN_TOPIC_HINT_RAW_TITLE_SCORE = float(os.getenv("OPEN_TOPIC_HINT_RAW_TITLE_SCORE", "8.4"))
    HYBRID_STRUCT_W_SECTION_TERM = float(os.getenv("HYBRID_STRUCT_W_SECTION_TERM", "0.24"))
    HYBRID_STRUCT_W_TEXT_TERM = float(os.getenv("HYBRID_STRUCT_W_TEXT_TERM", "0.18"))
    HYBRID_STRUCT_W_SECTION_OVERLAP = float(os.getenv("HYBRID_STRUCT_W_SECTION_OVERLAP", "0.22"))
    HYBRID_STRUCT_W_KEYWORD = float(os.getenv("HYBRID_STRUCT_W_KEYWORD", "0.18"))
    HYBRID_STRUCT_W_TITLE = float(os.getenv("HYBRID_STRUCT_W_TITLE", "0.10"))
    HYBRID_STRUCT_W_BASE = float(os.getenv("HYBRID_STRUCT_W_BASE", "0.08"))
    HYBRID_STRUCT_SIGNAL_WEIGHT = float(os.getenv("HYBRID_STRUCT_SIGNAL_WEIGHT", "0.28"))
    HYBRID_STRUCT_FOLLOW_BONUS = float(os.getenv("HYBRID_STRUCT_FOLLOW_BONUS", "0.16"))
    HYBRID_STRUCT_FOLLOW_WINDOW = int(os.getenv("HYBRID_STRUCT_FOLLOW_WINDOW", "3"))
    HYBRID_STRUCT_GENERIC_SECTION_PENALTY = float(os.getenv("HYBRID_STRUCT_GENERIC_SECTION_PENALTY", "0.08"))
    HYBRID_STRUCT_GENERIC_SHORT_PENALTY = float(os.getenv("HYBRID_STRUCT_GENERIC_SHORT_PENALTY", "0.16"))
    HYBRID_STRUCT_SECTION_MATCH_BONUS = float(os.getenv("HYBRID_STRUCT_SECTION_MATCH_BONUS", "0.22"))
    HYBRID_STRUCT_SECTION_MISMATCH_PENALTY = float(os.getenv("HYBRID_STRUCT_SECTION_MISMATCH_PENALTY", "0.12"))
    HYBRID_STRUCT_TOPIC_BONUS = float(os.getenv("HYBRID_STRUCT_TOPIC_BONUS", "0.20"))
    HYBRID_STRUCT_ASPECT_BONUS = float(os.getenv("HYBRID_STRUCT_ASPECT_BONUS", "0.18"))
    HYBRID_STRUCT_ASPECT_BONUS_CAP = float(os.getenv("HYBRID_STRUCT_ASPECT_BONUS_CAP", "0.48"))
    HYBRID_STRUCT_ARTICLE_ANCHOR_BONUS = float(os.getenv("HYBRID_STRUCT_ARTICLE_ANCHOR_BONUS", "0.55"))
    HYBRID_STRUCT_ARTICLE_ANCHOR_BONUS_CAP = float(os.getenv("HYBRID_STRUCT_ARTICLE_ANCHOR_BONUS_CAP", "0.75"))
    HEADING_RESCUE_LEGAL_SECTION_BONUS = float(os.getenv("HEADING_RESCUE_LEGAL_SECTION_BONUS", "3.0"))
    HEADING_RESCUE_LEGAL_BODY_BONUS = float(os.getenv("HEADING_RESCUE_LEGAL_BODY_BONUS", "1.5"))
    ASPECT_EVIDENCE_BODY_HIT_WEIGHT = float(os.getenv("ASPECT_EVIDENCE_BODY_HIT_WEIGHT", "1.45"))
    ASPECT_EVIDENCE_INHERITED_HIT_WEIGHT = float(os.getenv("ASPECT_EVIDENCE_INHERITED_HIT_WEIGHT", "1.2"))
    ASPECT_EVIDENCE_BODY_EXACT_WEIGHT = float(os.getenv("ASPECT_EVIDENCE_BODY_EXACT_WEIGHT", "0.95"))
    ASPECT_EVIDENCE_INHERITED_EXACT_WEIGHT = float(os.getenv("ASPECT_EVIDENCE_INHERITED_EXACT_WEIGHT", "0.75"))
    ASPECT_EVIDENCE_SECTION_HIT_WEIGHT = float(os.getenv("ASPECT_EVIDENCE_SECTION_HIT_WEIGHT", "0.5"))
    ASPECT_EVIDENCE_SECTION_EXACT_WEIGHT = float(os.getenv("ASPECT_EVIDENCE_SECTION_EXACT_WEIGHT", "0.25"))
    ASPECT_EVIDENCE_GENERIC_SECTION_PENALTY = float(os.getenv("ASPECT_EVIDENCE_GENERIC_SECTION_PENALTY", "0.45"))
    ASPECT_EVIDENCE_RANK_BONUS_BASE = float(os.getenv("ASPECT_EVIDENCE_RANK_BONUS_BASE", "0.25"))
    ASPECT_EVIDENCE_RANK_BONUS_DECAY = float(os.getenv("ASPECT_EVIDENCE_RANK_BONUS_DECAY", "0.01"))
    ASPECT_EVIDENCE_CLAUSE_BONUS = float(os.getenv("ASPECT_EVIDENCE_CLAUSE_BONUS", "0.35"))
    ASPECT_EVIDENCE_SUBSTANTIVE_BONUS = float(os.getenv("ASPECT_EVIDENCE_SUBSTANTIVE_BONUS", "0.25"))
    ASPECT_RESCUE_SECTION_ALIGN_WEIGHT = float(os.getenv("ASPECT_RESCUE_SECTION_ALIGN_WEIGHT", "0.6"))
    ASPECT_RESCUE_SECTION_EXACT_WEIGHT = float(os.getenv("ASPECT_RESCUE_SECTION_EXACT_WEIGHT", "0.15"))
    SUBJECT_FOCUS_TARGET_HIT_BONUS = float(os.getenv("SUBJECT_FOCUS_TARGET_HIT_BONUS", "0.06"))
    SUBJECT_FOCUS_TARGET_BONUS_CAP = float(os.getenv("SUBJECT_FOCUS_TARGET_BONUS_CAP", "0.18"))
    SUBJECT_FOCUS_EXCLUDED_HIT_PENALTY = float(os.getenv("SUBJECT_FOCUS_EXCLUDED_HIT_PENALTY", "0.18"))
    SUBJECT_FOCUS_EXCLUDED_PENALTY_CAP = float(os.getenv("SUBJECT_FOCUS_EXCLUDED_PENALTY_CAP", "0.45"))
    SUBJECT_FOCUS_UNMATCHED_EXCLUDED_PENALTY = float(os.getenv("SUBJECT_FOCUS_UNMATCHED_EXCLUDED_PENALTY", "0.12"))
    ANSWER_STRUCTURED_ASPECT_MATCH_MIN_SCORE = float(os.getenv("ANSWER_STRUCTURED_ASPECT_MATCH_MIN_SCORE", "35"))
    WEAK_QUERY_MAX_CHARS = int(os.getenv("WEAK_QUERY_MAX_CHARS", "18"))
    WEAK_QUERY_DOC_LIMIT = int(os.getenv("WEAK_QUERY_DOC_LIMIT", "6"))
    WEAK_QUERY_EXPANSION_LIMIT = int(os.getenv("WEAK_QUERY_EXPANSION_LIMIT", "3"))
    ENABLE_DOC_FALLBACK = os.getenv("ENABLE_DOC_FALLBACK", "true").lower() == "true"
    DOC_FALLBACK_SOURCE_LIMIT = int(os.getenv("DOC_FALLBACK_SOURCE_LIMIT", "6"))
    DOC_FALLBACK_CHUNK_SCAN_LIMIT = int(os.getenv("DOC_FALLBACK_CHUNK_SCAN_LIMIT", "400"))
    DOC_FALLBACK_MIN_PRIOR = float(os.getenv("DOC_FALLBACK_MIN_PRIOR", "0.18"))
    FUSION_W_DOC_PRIOR = float(os.getenv("FUSION_W_DOC_PRIOR", "0.003"))
    TITLE_CONSTRAINT_BOOST = float(os.getenv("TITLE_CONSTRAINT_BOOST", "1.08"))
    TITLE_CONSTRAINT_PENALTY = float(os.getenv("TITLE_CONSTRAINT_PENALTY", "0.82"))
    CONTEXTUAL_PREV_CHARS = int(os.getenv("CONTEXTUAL_PREV_CHARS", "120"))
    CONTEXTUAL_NEXT_CHARS = int(os.getenv("CONTEXTUAL_NEXT_CHARS", "120"))
    MILVUS_TEXT_MAX_CHARS = int(os.getenv("MILVUS_TEXT_MAX_CHARS", "60000"))
    MILVUS_TEXT_MAX_BYTES = int(os.getenv("MILVUS_TEXT_MAX_BYTES", os.getenv("MILVUS_TEXT_MAX_CHARS", "60000")))

