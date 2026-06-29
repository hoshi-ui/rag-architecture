from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any
from urllib import error, request

from score_documents_metrics import _aggregate, _aggregate_by, _score_case


ROOT = Path(__file__).resolve().parents[2]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(ROOT / "config" / "app.env")

DEFAULT_CASES = ROOT / "evals" / "cases" / "documents_leveled_query_dataset.json"
DEFAULT_OUT_DIR = ROOT / "reports" / "evals" / "documents_leveled_pipeline"
DEFAULT_BASE_URL = os.getenv("RAG_EVAL_API", "http://127.0.0.1:8080")


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())[:80] or "run"


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cases(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        cases = []
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"JSONL line {line_no} is not an object")
            cases.append(item)
        return cases

    payload = _read_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return [item for item in payload["cases"] if isinstance(item, dict)]
    raise ValueError(f"Unsupported cases format: {path}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, cases: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(case, ensure_ascii=False, separators=(",", ":")) for case in cases]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compact(value: Any) -> str:
    return " ".join(str(value or "").split())


def _estimate_tokens(text: str) -> int:
    text = str(text or "")
    if not text:
        return 0
    return int(math.ceil(len(text) / 1.6))


def _extract_token_usage(data: dict[str, Any], answer: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    found: dict[str, Any] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_l = str(key).lower()
                if key_l in {"prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"}:
                    if isinstance(item, (int, float)):
                        found[key_l] = int(item)
                elif key_l in {"usage", "token_usage", "llm_usage"}:
                    visit(item)
                elif isinstance(item, dict):
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(metadata)
    context_text = "\n".join(str(item.get("text") or item.get("text_preview") or "") for item in contexts)
    found.setdefault("estimated_context_tokens", _estimate_tokens(context_text))
    found.setdefault("estimated_answer_tokens", _estimate_tokens(answer))
    found.setdefault("estimated_total_tokens", found["estimated_context_tokens"] + found["estimated_answer_tokens"])
    return found


def _context_text(doc: dict[str, Any]) -> str:
    return _compact(
        doc.get("text")
        or doc.get("content")
        or doc.get("page_content")
        or doc.get("snippet")
        or doc.get("preview")
        or doc.get("text_preview")
        or ""
    )


def _normalize_context(doc: dict[str, Any], index: int) -> dict[str, Any]:
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    clause_meta = metadata.get("clause_metadata") if isinstance(metadata.get("clause_metadata"), dict) else {}
    text = _context_text(doc)
    return {
        "rank": index,
        "ref": doc.get("ref") or index,
        "source": doc.get("source") or metadata.get("source") or doc.get("filename") or "",
        "doc_id": doc.get("doc_id") or metadata.get("doc_id") or clause_meta.get("doc_id") or "",
        "title": doc.get("title") or metadata.get("title") or "",
        "clause": doc.get("clause") or metadata.get("clause") or metadata.get("article_no") or "",
        "clause_id": doc.get("clause_id") or metadata.get("clause_id") or metadata.get("article_no") or metadata.get("article_id") or "",
        "article_no": doc.get("article_no") or metadata.get("article_no") or metadata.get("article_id") or "",
        "article_id": doc.get("article_id") or metadata.get("article_id") or metadata.get("article_no") or "",
        "metadata_available": doc.get("metadata_available") if doc.get("metadata_available") is not None else metadata.get("metadata_available"),
        "section": doc.get("section") or metadata.get("section") or metadata.get("section_title") or "",
        "score": doc.get("score") or metadata.get("score"),
        "text": text,
        "text_preview": text[:800],
        "metadata": metadata,
    }


def _contexts_from_response(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents = data.get("documents") if isinstance(data.get("documents"), list) else []
    sources = data.get("sources") if isinstance(data.get("sources"), list) else []
    normalized_docs = [_normalize_context(doc, index) for index, doc in enumerate(documents, start=1) if isinstance(doc, dict)]
    normalized_sources = [_normalize_context(doc, index) for index, doc in enumerate(sources, start=1) if isinstance(doc, dict)]
    return normalized_docs, normalized_sources


def _post_json(url: str, payload: dict[str, Any], timeout_sec: int, api_key: str = "") -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def _json_env_object(*names: str) -> dict[str, Any]:
    for name in names:
        raw = os.getenv(name)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _call_rag(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "query": str(case.get("query") or ""),
        "user_id": str(args.user_id),
        "top_k": int(args.top_k),
        "enable_rerank": bool(args.enable_rerank),
    }
    url = f"{str(args.base_url).rstrip('/')}/query"
    started = time.perf_counter()
    try:
        data = _post_json(url, payload, int(args.timeout_sec))
        status_code = 200
        error_text = ""
    except error.HTTPError as exc:
        status_code = int(exc.code)
        try:
            data = json.loads(exc.read().decode("utf-8", errors="replace"))
        except Exception:
            data = {}
        error_text = f"HTTPError: {exc}"
    except Exception as exc:
        status_code = 0
        data = {}
        error_text = f"{type(exc).__name__}: {exc}"
    latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
    documents, sources = _contexts_from_response(data)
    answer = str(data.get("answer") or "")
    contexts = documents or sources
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    token_usage = _extract_token_usage(data, answer, contexts)
    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "difficulty_category": case.get("difficulty_category"),
        "difficulty_level": case.get("difficulty_level"),
        "subtype": case.get("subtype"),
        "query": case.get("query"),
        "expected": case.get("expected_sources") or [],
        "expected_sources": case.get("expected_sources") or [],
        "expected_answer": case.get("expected_answer") or "",
        "expected_aspects": case.get("expected_aspects") or [],
        "expected_evidence": case.get("expected_evidence") or [],
        "minimum_required_aspect_count": case.get("minimum_required_aspect_count"),
        "status_code": status_code,
        "ok": status_code == 200 and not error_text,
        "error": error_text,
        "actual_answer": answer,
        "answer": answer,
        "retrieved_contexts": contexts,
        "retrieved_documents": {"hybrid_rerank": contexts},
        "answer_documents": sources or contexts,
        "answer_sources": [item.get("source") for item in (sources or contexts) if item.get("source")],
        "metadata": metadata,
        "latency_ms": {
            "client_observed": latency_ms,
            "server_timing_ms": metadata.get("server_timing_ms") or {},
        },
        "token_usage": token_usage,
        "raw_response": data if args.keep_raw_response else {},
    }


def _judge_prompt(case: dict[str, Any], result: dict[str, Any]) -> str:
    contexts = result.get("retrieved_contexts") or []
    compact_contexts = []
    for item in contexts[:8]:
        compact_contexts.append(
            {
                "source": item.get("source"),
                "clause": item.get("clause"),
                "text": str(item.get("text") or item.get("text_preview") or "")[:1200],
            }
        )
    payload = {
        "query": case.get("query"),
        "expected_aspects": case.get("expected_aspects") or [],
        "expected_evidence": case.get("expected_evidence") or [],
        "retrieved_contexts": compact_contexts,
        "answer": result.get("actual_answer") or "",
    }
    return (
        "You are a strict legal RAG evaluator. Return compact JSON only.\n"
        "Score each field from 0 to 1: faithfulness, answer_relevance, legal_correctness.\n"
        "Also provide score_0_5 and a short reason in Chinese.\n"
        "Faithfulness means every legal claim is supported by retrieved_contexts.\n"
        "Answer relevance means the answer directly solves the query and covers expected_aspects.\n"
        "Legal correctness means cited regulation names and article numbers match expected_evidence.\n\n"
        f"Evaluation payload:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _parse_judge_json(content: str) -> dict[str, Any]:
    text = str(content or "").strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        text = match.group(0)
    parsed = json.loads(text)
    return parsed if isinstance(parsed, dict) else {}


def _extract_judge_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") if isinstance(response.get("choices"), list) else []
    if choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        for value in (
            message.get("content"),
            message.get("reasoning_content"),
            choice.get("text"),
            choice.get("content"),
        ):
            text = str(value or "").strip()
            if text:
                return text
    for value in (response.get("output_text"), response.get("text"), response.get("content")):
        text = str(value or "").strip()
        if text:
            return text
    output = response.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        text = str(part.get("text") or part.get("content") or "").strip()
                        if text:
                            parts.append(text)
        if parts:
            return "\n".join(parts)
    return ""


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_judge_backend() -> str:
    configured = (os.getenv("EVAL_JUDGE_BACKEND") or os.getenv("JUDGE_BACKEND") or "").strip().lower()
    if configured in {"none", "openai"}:
        return configured
    if _env_flag("ENABLE_LLM_AS_JUDGE") or _env_flag("ENABLE_EVAL_LLM_AS_JUDGE"):
        return "openai"
    return "none"


def _configured_judge_base_url(args: argparse.Namespace) -> str:
    base_url = str(
        args.judge_base_url
        or os.getenv("EVAL_JUDGE_BASE_URL")
        or os.getenv("JUDGE_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or ""
    ).rstrip("/")
    if base_url:
        return base_url
    chat_url = str(os.getenv("EVAL_JUDGE_CHAT_COMPLETIONS_URL") or os.getenv("JUDGE_CHAT_COMPLETIONS_URL") or os.getenv("LLM_CHAT_COMPLETIONS_URL") or "").rstrip("/")
    for suffix in ("/v1/chat/completions", "/chat/completions"):
        if chat_url.endswith(suffix):
            return chat_url[: -len(suffix)]
    return chat_url


def _judge_chat_completions_url(base_url: str) -> str:
    url = str(base_url or "").rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def _configured_judge_model(args: argparse.Namespace) -> str:
    return str(
        args.judge_model
        or os.getenv("EVAL_JUDGE_MODEL")
        or os.getenv("JUDGE_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("LLM_MODEL")
        or ""
    )


def _call_judge(case: dict[str, Any], result: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.judge_backend == "none":
        return {"enabled": False}
    if args.judge_backend != "openai":
        return {"enabled": False, "error": f"unsupported judge backend: {args.judge_backend}"}

    base_url = _configured_judge_base_url(args)
    model = _configured_judge_model(args)
    api_key = str(args.judge_api_key or os.getenv("EVAL_JUDGE_API_KEY") or os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY") or "")
    if not base_url or not model:
        return {"enabled": True, "error": "missing judge base URL or model"}
    url = _judge_chat_completions_url(base_url)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict legal RAG evaluator. Return JSON only."},
            {"role": "user", "content": _judge_prompt(case, result)},
        ],
        "temperature": 0,
        "max_tokens": 500,
    }
    payload.update(_json_env_object("EVAL_JUDGE_EXTRA_BODY", "JUDGE_EXTRA_BODY", "LLM_EXTRA_BODY"))
    started = time.perf_counter()
    try:
        response = _post_json(url, payload, int(args.judge_timeout_sec), api_key=api_key)
        content = _extract_judge_content(response)
        if not content:
            raise ValueError(f"empty judge content: {json.dumps(response, ensure_ascii=False)[:1200]}")
        try:
            scores = _parse_judge_json(content)
        except Exception as exc:
            raise ValueError(f"invalid judge JSON: {type(exc).__name__}: {content[:1200]}") from exc
        return {
            "enabled": True,
            "backend": "openai",
            "model": model,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "scores": scores,
            "raw_content": content[:2000],
        }
    except Exception as exc:
        return {
            "enabled": True,
            "backend": "openai",
            "model": model,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _number_0_1(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value)))


def _apply_judge_faithfulness(row: dict[str, Any]) -> None:
    judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
    if not judge.get("enabled") or judge.get("error"):
        return
    scores = judge.get("scores") if isinstance(judge.get("scores"), dict) else {}
    faithfulness = _number_0_1(scores.get("faithfulness"))
    if faithfulness is None:
        return
    generation = row.setdefault("generation", {})
    if not isinstance(generation, dict):
        return
    generation.setdefault("faithfulness_static", generation.get("faithfulness"))
    generation["faithfulness"] = faithfulness
    generation["faithfulness_method"] = "llm_judge"
    generation["faithfulness_reason"] = str(scores.get("reason") or "")[:300]


def _score_results(cases: list[dict[str, Any]], results: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    cases_by_id = {str(case.get("id")): case for case in cases}
    scored = []
    for result in results:
        case = cases_by_id.get(str(result.get("id")))
        if not case:
            continue
        row = _score_case(
            case,
            result,
            retrieval_key="hybrid_rerank",
            top_k=max(1, int(args.score_top_k)),
            evidence_similarity_threshold=float(args.evidence_similarity_threshold),
            claim_similarity_threshold=float(args.claim_similarity_threshold),
        )
        row["judge"] = _call_judge(case, result, args)
        _apply_judge_faithfulness(row)
        row["status_code"] = result.get("status_code")
        row["ok"] = result.get("ok")
        row["latency_ms"] = result.get("latency_ms") or {}
        row["token_usage"] = result.get("token_usage") or {}
        scored.append(row)
    return scored


def _mean(values: list[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def _mean_or_none(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def _judge_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["faithfulness", "answer_relevance", "legal_correctness", "score_0_5"]
    summary: dict[str, Any] = {"enabled_cases": 0, "errored_cases": 0}
    for field in fields:
        values = []
        for row in rows:
            judge = row.get("judge") or {}
            if judge.get("enabled"):
                summary["enabled_cases"] += 0
            if judge.get("error"):
                continue
            scores = judge.get("scores") if isinstance(judge.get("scores"), dict) else {}
            value = scores.get(field)
            if isinstance(value, (int, float)):
                values.append(float(value))
        summary[f"{field}_avg"] = _mean_or_none(values)
    summary["enabled_cases"] = sum(1 for row in rows if (row.get("judge") or {}).get("enabled"))
    summary["errored_cases"] = sum(1 for row in rows if (row.get("judge") or {}).get("error"))
    return summary


def _pipeline_summary(scored: list[dict[str, Any]]) -> dict[str, Any]:
    deterministic = {
        "overall": _aggregate(scored),
        "by_difficulty_category": _aggregate_by(scored, "difficulty_category"),
        "by_difficulty_level": _aggregate_by(scored, "difficulty_level"),
        "by_subtype": _aggregate_by(scored, "subtype"),
    }
    latencies = [
        float((row.get("latency_ms") or {}).get("client_observed"))
        for row in scored
        if isinstance((row.get("latency_ms") or {}).get("client_observed"), (int, float))
    ]
    total_tokens = [
        float((row.get("token_usage") or {}).get("total_tokens") or (row.get("token_usage") or {}).get("estimated_total_tokens") or 0)
        for row in scored
    ]
    return {
        "deterministic": deterministic,
        "judge": _judge_summary(scored),
        "runtime": {
            "case_count": len(scored),
            "ok_count": sum(1 for row in scored if row.get("ok")),
            "latency_ms_avg": _mean(latencies),
            "latency_ms_p95": sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else 0.0,
            "token_total_est_or_actual": int(sum(total_tokens)),
            "token_avg_est_or_actual": _mean(total_tokens),
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "difficulty_category",
        "difficulty_level",
        "subtype",
        "context_precision",
        "context_recall",
        "mrr",
        "ndcg",
        "faithfulness",
        "unsupported_claim_rate",
        "answer_relevance",
        "citation_recall",
        "citation_precision",
        "citation_exact_match",
        "judge_score_0_5",
        "latency_ms",
        "tokens",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            judge_scores = ((row.get("judge") or {}).get("scores") or {}) if isinstance((row.get("judge") or {}).get("scores"), dict) else {}
            writer.writerow(
                {
                    "id": row.get("id"),
                    "difficulty_category": row.get("difficulty_category"),
                    "difficulty_level": row.get("difficulty_level"),
                    "subtype": row.get("subtype"),
                    "context_precision": (row.get("retrieval") or {}).get("context_precision"),
                    "context_recall": (row.get("retrieval") or {}).get("context_recall"),
                    "mrr": (row.get("retrieval") or {}).get("mrr"),
                    "ndcg": (row.get("retrieval") or {}).get("ndcg"),
                    "faithfulness": (row.get("generation") or {}).get("faithfulness"),
                    "unsupported_claim_rate": (row.get("generation") or {}).get("unsupported_claim_rate"),
                    "answer_relevance": (row.get("generation") or {}).get("answer_relevance"),
                    "citation_recall": (row.get("rule_based") or {}).get("citation_recall"),
                    "citation_precision": (row.get("rule_based") or {}).get("citation_precision"),
                    "citation_exact_match": (row.get("rule_based") or {}).get("citation_exact_match"),
                    "judge_score_0_5": judge_scores.get("score_0_5"),
                    "latency_ms": (row.get("latency_ms") or {}).get("client_observed"),
                    "tokens": (row.get("token_usage") or {}).get("total_tokens")
                    or (row.get("token_usage") or {}).get("estimated_total_tokens"),
                }
            )


def _write_xlsx_if_possible(path: Path, rows: list[dict[str, Any]]) -> str:
    try:
        from openpyxl import Workbook
    except Exception as exc:
        return f"openpyxl unavailable: {type(exc).__name__}: {exc}"

    wb = Workbook()
    ws = wb.active
    ws.title = "metrics"
    headers = [
        "id",
        "difficulty_category",
        "difficulty_level",
        "subtype",
        "context_precision",
        "context_recall",
        "mrr",
        "ndcg",
        "faithfulness",
        "answer_relevance",
        "citation_recall",
        "citation_precision",
        "citation_exact_match",
        "latency_ms",
        "tokens",
    ]
    ws.append(headers)
    for row in rows:
        ws.append(
            [
                row.get("id"),
                row.get("difficulty_category"),
                row.get("difficulty_level"),
                row.get("subtype"),
                (row.get("retrieval") or {}).get("context_precision"),
                (row.get("retrieval") or {}).get("context_recall"),
                (row.get("retrieval") or {}).get("mrr"),
                (row.get("retrieval") or {}).get("ndcg"),
                (row.get("generation") or {}).get("faithfulness"),
                (row.get("generation") or {}).get("answer_relevance"),
                (row.get("rule_based") or {}).get("citation_recall"),
                (row.get("rule_based") or {}).get("citation_precision"),
                (row.get("rule_based") or {}).get("citation_exact_match"),
                (row.get("latency_ms") or {}).get("client_observed"),
                (row.get("token_usage") or {}).get("total_tokens")
                or (row.get("token_usage") or {}).get("estimated_total_tokens"),
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return ""


def _markdown_table(rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    sep = ["---"] * len(header)
    body = rows[1:]
    lines = [
        "| " + " | ".join(str(item) for item in header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _markdown_metric_value(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_markdown(path: Path, summary: dict[str, Any], scored: list[dict[str, Any]], artifact_paths: dict[str, str]) -> None:
    overall = (((summary.get("deterministic") or {}).get("overall") or {}))
    retrieval = overall.get("retrieval") or {}
    generation = overall.get("generation") or {}
    rule_based = overall.get("rule_based") or {}
    negative_control = overall.get("negative_control") or {}
    runtime = summary.get("runtime") or {}
    rows = [
        ["Metric", "Score"],
        ["Positive Cases", overall.get("positive_cases", overall.get("cases", 0))],
        ["Negative Cases", overall.get("negative_cases", 0)],
        ["Context Precision", retrieval.get("context_precision", 0)],
        ["Context Recall", retrieval.get("context_recall", 0)],
        ["MRR", retrieval.get("mrr", 0)],
        ["NDCG", retrieval.get("ndcg", 0)],
        ["Faithfulness", generation.get("faithfulness", 0)],
        ["Answer Relevance", generation.get("answer_relevance", 0)],
        ["Citation Recall", rule_based.get("citation_recall", 0)],
        ["Citation Precision", rule_based.get("citation_precision", 0)],
        ["Citation Exact Match", rule_based.get("citation_exact_match", 0)],
        ["Negative Pass Rate", negative_control.get("pass_rate", 0)],
        ["Latency Avg ms", runtime.get("latency_ms_avg", 0)],
        ["Token Avg", runtime.get("token_avg_est_or_actual", 0)],
    ]
    by_diff = [["Difficulty", "Cases", "Recall", "MRR", "Faithfulness", "Citation Recall", "Citation Precision", "Citation EM"]]
    for key, bucket in sorted(((summary.get("deterministic") or {}).get("by_difficulty_category") or {}).items()):
        by_diff.append(
            [
                key,
                bucket.get("cases", 0),
                (bucket.get("retrieval") or {}).get("context_recall", 0),
                (bucket.get("retrieval") or {}).get("mrr", 0),
                (bucket.get("generation") or {}).get("faithfulness", 0),
                (bucket.get("rule_based") or {}).get("citation_recall", 0),
                (bucket.get("rule_based") or {}).get("citation_precision", 0),
                (bucket.get("rule_based") or {}).get("citation_exact_match", 0),
            ]
        )
    positive_scored = [
        row
        for row in scored
        if not ((row.get("negative_control") or {}).get("is_negative") or row.get("is_negative"))
    ]
    worst = sorted(
        positive_scored,
        key=lambda row: (
            _markdown_metric_value((row.get("retrieval") or {}).get("context_recall")),
            _markdown_metric_value((row.get("rule_based") or {}).get("citation_recall")),
            _markdown_metric_value((row.get("rule_based") or {}).get("citation_precision")),
            _markdown_metric_value((row.get("generation") or {}).get("answer_relevance")),
        ),
    )[:10]
    worst_rows = [["Case", "Difficulty", "Recall", "Citation Recall", "Citation Precision", "Citation EM", "Answer Rel", "Query"]]
    for row in worst:
        worst_rows.append(
            [
                row.get("id"),
                row.get("difficulty_category"),
                _markdown_metric_value((row.get("retrieval") or {}).get("context_recall")),
                _markdown_metric_value((row.get("rule_based") or {}).get("citation_recall")),
                _markdown_metric_value((row.get("rule_based") or {}).get("citation_precision")),
                _markdown_metric_value((row.get("rule_based") or {}).get("citation_exact_match")),
                _markdown_metric_value((row.get("generation") or {}).get("answer_relevance")),
                _compact(row.get("query"))[:80],
            ]
        )
    md = [
        "# Documents RAG Evaluation Report",
        "",
        f"- Cases: {runtime.get('case_count', 0)}",
        f"- OK: {runtime.get('ok_count', 0)}",
        f"- Avg latency ms: {runtime.get('latency_ms_avg', 0)}",
        f"- Total tokens estimated/actual: {runtime.get('token_total_est_or_actual', 0)}",
        "",
        "## Overall",
        "",
        _markdown_table(rows),
        "",
        "## By Difficulty",
        "",
        _markdown_table(by_diff),
        "",
        "## Lowest Scoring Cases",
        "",
        _markdown_table(worst_rows),
        "",
        "## Artifacts",
        "",
    ]
    for name, value in artifact_paths.items():
        md.append(f"- {name}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full documents RAG evaluation pipeline.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="Input JSON suite or JSONL cases.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--run-name", default="", help="Optional run name. Defaults to timestamp.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="RAG API base URL.")
    parser.add_argument("--user-id", default="documents_eval_pipeline")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--score-top-k", type=int, default=5)
    parser.add_argument("--enable-rerank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases.")
    parser.add_argument("--prepare-only", action="store_true", help="Only export the JSONL snapshot.")
    parser.add_argument("--keep-raw-response", action="store_true", help="Persist full raw /query response.")
    parser.add_argument("--judge-backend", choices=["none", "openai"], default=_default_judge_backend())
    parser.add_argument("--judge-base-url", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--judge-api-key", default="")
    parser.add_argument("--judge-timeout-sec", type=int, default=int(os.getenv("EVAL_JUDGE_TIMEOUT_SEC") or os.getenv("JUDGE_TIMEOUT_SEC") or "60"))
    parser.add_argument("--evidence-similarity-threshold", type=float, default=0.72)
    parser.add_argument("--claim-similarity-threshold", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases_path = Path(args.cases)
    cases = _load_cases(cases_path)
    if int(args.limit or 0) > 0:
        cases = cases[: int(args.limit)]

    run_name = _safe_name(args.run_name) if args.run_name else _now_stamp()
    out_dir = Path(args.out_dir) / run_name
    jsonl_path = out_dir / "cases.jsonl"
    batch_path = out_dir / "batch_results.json"
    metrics_path = out_dir / "metrics_report.json"
    csv_path = out_dir / "metrics_rows.csv"
    xlsx_path = out_dir / "metrics_rows.xlsx"
    markdown_path = out_dir / "dashboard.md"

    _write_jsonl(jsonl_path, cases)
    if args.prepare_only:
        print(json.dumps({"cases": len(cases), "jsonl": str(jsonl_path)}, ensure_ascii=False, indent=2))
        return

    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.get('id')} {case.get('difficulty_category') or ''}")
        results.append(_call_rag(case, args))

    batch_report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cases_path": str(cases_path.resolve()),
        "jsonl_path": str(jsonl_path.resolve()),
        "base_url": str(args.base_url),
        "top_k": int(args.top_k),
        "enable_rerank": bool(args.enable_rerank),
        "results": results,
    }
    _write_json(batch_path, batch_report)

    scored = _score_results(cases, results, args)
    summary = _pipeline_summary(scored)
    metrics_report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cases_path": str(cases_path.resolve()),
        "batch_report_path": str(batch_path.resolve()),
        "summary": summary,
        "results": scored,
    }
    _write_json(metrics_path, metrics_report)
    _write_csv(csv_path, scored)
    xlsx_error = _write_xlsx_if_possible(xlsx_path, scored)
    artifact_paths = {
        "jsonl": str(jsonl_path),
        "batch_results": str(batch_path),
        "metrics_json": str(metrics_path),
        "metrics_csv": str(csv_path),
    }
    if not xlsx_error:
        artifact_paths["metrics_xlsx"] = str(xlsx_path)
    else:
        artifact_paths["xlsx_status"] = xlsx_error
    _write_markdown(markdown_path, summary, scored, artifact_paths)
    artifact_paths["dashboard_md"] = str(markdown_path)

    print(
        json.dumps(
            {
                "cases": len(cases),
                "ok": summary["runtime"]["ok_count"],
                "dashboard": str(markdown_path),
                "metrics": str(metrics_path),
                "summary": summary["deterministic"]["overall"],
                "judge": summary.get("judge") or {},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
