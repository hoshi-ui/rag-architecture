from datetime import datetime
import json
import re
from typing import Any, Dict

from fastapi import HTTPException

from app.config import Config
from app.documents import ir as document_ir_helpers
from app.services.embedding import EmbeddingService
from app.services.llm_client import LlmClient
from app.services import document_metadata
from app.storage import milvus as milvus_storage
from app.utils.text import sanitize_index_text


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text or "") / 2))


def _index_text_value(text: str) -> str:
    value = sanitize_index_text(str(text or ""))
    max_bytes = int(getattr(Config, "MILVUS_TEXT_MAX_BYTES", 60000) or 60000)
    if max_bytes <= 0 or len(value.encode("utf-8")) <= max_bytes:
        return value
    suffix = "\n[TRUNCATED_FOR_INDEX_TEXT_LIMIT]"
    budget = max(0, max_bytes - len(suffix.encode("utf-8")))
    return value.encode("utf-8")[:budget].decode("utf-8", "ignore").rstrip() + suffix


def _plain_chunk_text(item: Dict[str, Any]) -> str:
    raw = sanitize_index_text(item.get("raw_text") or item.get("content") or item.get("text") or "")
    if any(
        marker in raw
        for marker in (
            "Content:",
            "Previous context:",
            "Next context:",
            "Document:",
            "Section:",
            "章节路径",
            "页码",
            "元素类型",
            "正文：",
            "正文:",
        )
    ):
        stripped = document_metadata.chunk_plain_display_text(raw)
        if stripped:
            return sanitize_index_text(stripped)
    return raw


def _format_for_embedding(filename: str, item: Dict[str, Any], *, chunk_count: int) -> str:
    raw_text = _plain_chunk_text(item)
    section_path = item.get("section_path") or []
    if isinstance(section_path, list):
        section_label = " > ".join(str(part).strip() for part in section_path if str(part).strip())
    else:
        section_label = str(section_path or "").strip()
    if not section_label:
        section_label = str(item.get("section") or item.get("section_title") or "").strip()
    article = str(item.get("article_id") or item.get("article_no") or item.get("clause_label") or "").strip()
    chunk_id = item.get("chunk_id")
    parts = [f"Document: {filename}"]
    if section_label:
        parts.append(f"Section path: {section_label}")
    if article:
        parts.append(f"Clause: {article}")
    if chunk_id is not None and chunk_count > 0:
        try:
            parts.append(f"Chunk: {int(chunk_id) + 1}/{chunk_count}")
        except Exception:
            pass
    previous_context = sanitize_index_text(item.get("previous_context") or "")
    next_context = sanitize_index_text(item.get("next_context") or "")
    if previous_context:
        parts.append(f"Previous context: {previous_context}")
    parts.append(f"Content: {raw_text}")
    if next_context:
        parts.append(f"Next context: {next_context}")
    return "\n".join(part for part in parts if str(part or "").strip())


def _extract_json_object_text(text: str) -> str:
    raw = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, flags=re.I)
    if fenced:
        return fenced.group(1).strip()
    match = re.search(r"\{[\s\S]*\}", raw)
    return match.group(0).strip() if match else raw


def _normalize_enrichment_payload(payload: Any) -> Dict[str, Any]:
    obj = payload if isinstance(payload, dict) else {}
    subjects = obj.get("applicable_subjects") or []
    if isinstance(subjects, str):
        subjects = [part.strip() for part in re.split(r"[,，、/；;]\s*", subjects) if part.strip()]
    elif isinstance(subjects, list):
        subjects = [str(part).strip() for part in subjects if str(part).strip()]
    else:
        subjects = []
    return {
        "applicable_subjects": subjects[:12],
        "action_type": str(obj.get("action_type") or "").strip()[:64],
    }


class DocumentIndexingService:
    def __init__(self, facade: Any):
        self.facade = facade

    async def enrich_chunk_metadata_via_llm(self, runtime: Any, filename: str, items: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        if not bool(getattr(Config, "ENABLE_LLM_CHUNK_METADATA_ENRICHMENT", True)):
            return items
        client = LlmClient(Config)
        if not client.available() or not items:
            return items
        max_chars = int(getattr(Config, "LLM_CHUNK_METADATA_MAX_CHARS", 1200) or 1200)
        timeout = int(getattr(Config, "LLM_CHUNK_METADATA_TIMEOUT", 20) or 20)
        system_prompt = (
            "你是法规 RAG 索引阶段的离线标注器。只输出 JSON，不要解释。"
            "从法条 chunk 中提取适用主体和行为类型。"
            "action_type 只能用简短中文，例如：义务、罚则、权利、程序、定义、职责、禁止、许可、监督检查、其他。"
        )
        response_formats = [{"type": "json_object"}]
        enriched: list[Dict[str, Any]] = []
        disabled_after_error = False
        for idx, item in enumerate(items):
            if disabled_after_error:
                enriched.append(item)
                continue
            raw_text = sanitize_index_text(item.get("raw_text") or item.get("text") or "")
            section_path = item.get("section_path") or []
            if isinstance(section_path, list):
                section_label = " > ".join(str(part) for part in section_path if str(part).strip())
            else:
                section_label = str(section_path or "")
            prompt = (
                f"文档：{filename}\n"
                f"章节路径：{section_label}\n"
                f"法条：{item.get('article_id') or item.get('article_no') or item.get('clause_label') or ''}\n"
                f"正文：{raw_text[:max_chars]}\n\n"
                "请输出严格 JSON："
                '{"applicable_subjects":["养犬人","物业"],"action_type":"义务"}'
            )
            payload = client.build_payload(
                system_prompt,
                prompt,
                temperature=0.0,
                top_p=1.0,
                max_tokens=180,
                presence_penalty=0.0,
            )
            try:
                content = await client.chat_text_with_response_formats(payload, response_formats, timeout=timeout)
                tags = _normalize_enrichment_payload(json.loads(_extract_json_object_text(content)))
                enriched.append({**item, **tags})
            except Exception as exc:
                if idx == 0:
                    disabled_after_error = True
                try:
                    runtime.task_log("", "chunk_metadata_enrichment_failed", {"filename": filename, "chunk": idx, "error": str(exc)})
                except Exception:
                    pass
                enriched.append(item)
        return enriched

    async def index_document_incremental(self, *args: Any, **kwargs: Any) -> Any:
        runtime = self.facade
        task_id = str(kwargs.pop("task_id", args[0] if len(args) > 0 else ""))
        filename = str(kwargs.pop("filename", args[1] if len(args) > 1 else ""))
        text = str(kwargs.pop("text", args[2] if len(args) > 2 else ""))
        metadata = kwargs.pop("metadata", args[3] if len(args) > 3 else None)
        document_ir = kwargs.pop("document_ir", args[4] if len(args) > 4 else None)
        if document_ir and (document_ir.get("elements") or []):
            runtime.store_document_ir(filename, document_ir)
        items = runtime.contextualize_chunk_items(filename, runtime.prepare_structured_items(filename, text, document_ir))
        if not items:
            raise HTTPException(status_code=400, detail="文档内容为空，无法索引")

        if task_id:
            runtime.tasks.setdefault(task_id, {})["stage"] = "metadata_enrichment"
            runtime.task_log(task_id, "metadata_enrichment", {"chunks": len(items)})
        items = await self.enrich_chunk_metadata_via_llm(runtime, filename, items)

        embedding_service = EmbeddingService(Config.EMBEDDING_URL)
        vector_db = runtime.vector_db()
        now = datetime.now().isoformat()
        base_metadata = runtime.index_base_metadata(filename, text, metadata, document_ir)
        current_doc = runtime.doc_get(filename)
        doc_version = current_doc.get("pending_version") or current_doc.get("active_version") or 1
        total = len(items)
        done = 0
        batch: list[Dict[str, Any]] = []
        batch_tokens = 0
        max_batch_tokens = 8000
        max_batch_items = 64
        next_chunk_id = 0

        async def flush_batch(savepoint_name: str) -> None:
            nonlocal batch, batch_tokens, done, next_chunk_id
            if not batch:
                return
            texts = [item["embedding_text"] for item in batch]
            docs: list[Dict[str, Any]] = []
            if not bool(getattr(Config, "TEST_LEX_ONLY", False)):
                embeddings, sparse_embeddings = await embedding_service.embed_batched_with_sparse(
                    texts,
                    per_request=32,
                    timeout=60,
                    retries=2,
                    return_sparse=True,
                )
                for idx, (batch_item, embedding) in enumerate(zip(batch, embeddings)):
                    docs.append(
                        runtime.vector_doc_entry(
                            filename=filename,
                            batch_item=batch_item,
                            embedding_value=embedding,
                            base_metadata=base_metadata,
                            chunk_id=next_chunk_id,
                            chunk_count=total,
                            created_at=now,
                            doc_version=doc_version,
                            sparse_embedding_value=sparse_embeddings[idx] if idx < len(sparse_embeddings) else None,
                        )
                    )
                    next_chunk_id += 1
                vector_db.insert(docs)
            else:
                for batch_item in batch:
                    docs.append(
                        runtime.vector_doc_entry(
                            filename=filename,
                            batch_item=batch_item,
                            embedding_value=None,
                            base_metadata=base_metadata,
                            chunk_id=next_chunk_id,
                            chunk_count=total,
                            created_at=now,
                            doc_version=doc_version,
                        )
                    )
                    next_chunk_id += 1
            with runtime.sqlite_write_lock():
                runtime._lex_store.savepoint(savepoint_name)
                done += len(batch)
                if task_id:
                    runtime.tasks.setdefault(task_id, {})["status"] = "indexing"
                    runtime.tasks[task_id]["stage"] = "embedding_partial"
                    runtime.tasks[task_id]["chunks_indexed"] = done
                    runtime.task_log(task_id, "embedding_batch_done", {"done": done, "total": total})
                for item, batch_item in zip(docs, batch):
                    item_metadata = item.get("metadata") or {}
                    runtime.add_chunk_sql(
                        filename,
                        batch_item.get("raw_text") or item_metadata.get("content") or item["text"],
                        item_metadata.get("section") or "",
                        item_metadata,
                        int(item_metadata.get("chunk_id") or 0),
                    )
                runtime._lex_store.release_savepoint(savepoint_name)
            batch = []
            batch_tokens = 0

        for item in items:
            raw_text = _index_text_value(_plain_chunk_text(item))
            embedding_text = _index_text_value(_format_for_embedding(filename, {**item, "raw_text": raw_text}, chunk_count=total))
            token_count = _estimate_tokens(embedding_text)
            if batch and (batch_tokens + token_count > max_batch_tokens or len(batch) >= max_batch_items):
                await flush_batch("batch_write")
            batch.append(
                {
                    **item,
                    "section": (item.get("section") or "").strip(),
                    "text": raw_text,
                    "raw_text": raw_text,
                    "embedding_text": embedding_text,
                    "normalized_text": item.get("normalized_text") or document_ir_helpers.normalize_ir_text(raw_text),
                    "fts_text": item.get("fts_text") or raw_text,
                }
            )
            batch_tokens += token_count
        if batch:
            await flush_batch("batch_write_last")
        return done

    async def index_document(self, *args: Any, **kwargs: Any) -> Any:
        runtime = self.facade
        filename = str(kwargs.pop("filename", args[0] if len(args) > 0 else ""))
        text = str(kwargs.pop("text", args[1] if len(args) > 1 else ""))
        metadata = kwargs.pop("metadata", args[2] if len(args) > 2 else None)
        document_ir = kwargs.pop("document_ir", args[3] if len(args) > 3 else None)
        if document_ir and (document_ir.get("elements") or []):
            runtime.store_document_ir(filename, document_ir)
        chunk_items = runtime.contextualize_chunk_items(filename, runtime.prepare_structured_items(filename, text, document_ir))
        chunk_items = await self.enrich_chunk_metadata_via_llm(runtime, filename, chunk_items)
        if not chunk_items:
            raise HTTPException(status_code=400, detail="文档内容为空，无法索引")

        embedding_service = EmbeddingService(Config.EMBEDDING_URL)
        vector_db = runtime.vector_db()
        total = len(chunk_items)
        prepared_chunk_items: list[Dict[str, Any]] = []
        for item in chunk_items:
            raw_text = _index_text_value(_plain_chunk_text(item))
            prepared_chunk_items.append(
                {
                    **item,
                    "text": raw_text,
                    "raw_text": raw_text,
                    "embedding_text": _index_text_value(
                        _format_for_embedding(filename, {**item, "raw_text": raw_text}, chunk_count=total)
                    ),
                    "normalized_text": item.get("normalized_text") or document_ir_helpers.normalize_ir_text(raw_text),
                    "fts_text": item.get("fts_text") or raw_text,
                }
            )
        chunk_items = prepared_chunk_items
        embeddings, sparse_embeddings = await embedding_service.embed_batched_with_sparse(
            [item["embedding_text"] for item in chunk_items],
            per_request=64,
            timeout=60,
            retries=2,
            return_sparse=True,
        )
        now = datetime.now().isoformat()
        base_metadata = runtime.index_base_metadata(filename, text, metadata, document_ir)
        docs = []
        for idx, (item, embedding) in enumerate(zip(chunk_items, embeddings)):
            docs.append(
                runtime.vector_doc_entry(
                    filename=filename,
                    batch_item=item,
                    embedding_value=embedding,
                    base_metadata=base_metadata,
                    chunk_id=int(item.get("chunk_id") or 0),
                    chunk_count=len(chunk_items),
                    created_at=now,
                    sparse_embedding_value=sparse_embeddings[idx] if idx < len(sparse_embeddings) else None,
                )
            )
        vector_db.insert(docs)
        for item, chunk_item in zip(docs, chunk_items):
            item_metadata = item.get("metadata") or {}
            runtime.add_chunk_sql(
                filename,
                chunk_item.get("raw_text") or item_metadata.get("content") or item["text"],
                item_metadata.get("section") or "",
                item_metadata,
                int(item_metadata.get("chunk_id") or 0),
            )
        return len(docs)
